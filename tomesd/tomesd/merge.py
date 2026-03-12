import torch
import torch.nn.functional as F
from typing import Tuple, Callable, Optional, Dict, Any
import math


def do_nothing(x: torch.Tensor, mode: Optional[str] = None):
    return x


def adaptive_block_pooling_random2d(metric: torch.Tensor,
                                   w: int, h: int, sx: int, sy: int, r: int,
                                   no_rand: bool = False,
                                   generator: torch.Generator = None,
                                   token_sizes: Optional[torch.Tensor] = None,
                                   token_scores: Optional[torch.Tensor] = None,
                                   scorer: Optional['TokenScorer'] = None,
                                   tile_aggregation: str = "sum",
                                   use_compile: bool = False) -> Tuple[Callable, Callable]:
    """
    Adaptive Block Pooling (ABP) replacement for bipartite_soft_matching_random2d.
    Uses sx×sy tiles and merges the smoothest tiles to achieve target reduction r.
    
    OPTIMIZED: Memory-efficient implementation with cuDNN pooling and tile views.
    
    Args:
        metric: Input tokens (B, N, C) where N = h*w
        w, h: spatial dimensions  
        sx, sy: tile dimensions (reuse existing parameters)
        r: number of tokens to remove (same as bipartite)
        no_rand: unused (for compatibility)
        generator: unused (for compatibility) 
        token_sizes: unused (for compatibility)
        token_scores: unused (for compatibility)
        scorer: TokenScorer instance for tile evaluation (if None, uses SpatialFilterScorer)
        tile_aggregation: How to aggregate token scores within tiles 
                         ("max", "mean", "min", "sum", "std", "median")
        use_compile: Whether to use torch.compile for additional speedup (PyTorch 2.0+)
    
    Returns:
        merge_fn, unmerge_fn: Functions to apply and reverse the merge
    """
    B, N, C = metric.shape
    
    if r <= 0:
        return do_nothing, do_nothing


    # Calculate how many tiles we need to merge to achieve target reduction r
    tile_area = sx * sy
    tokens_saved_per_tile = tile_area - 1
    if tokens_saved_per_tile <= 0:
        return do_nothing, do_nothing
        
    tiles_to_merge = r // tokens_saved_per_tile
    
    hsy, wsx = h // sy, w // sx
    total_tiles = hsy * wsx
    
    # Can't merge more tiles than we have
    tiles_to_merge = min(tiles_to_merge, total_tiles)
    
    if tiles_to_merge <= 0:
        return do_nothing, do_nothing
    
    # Use configurable scorer for tile evaluation
    if scorer is None:
        # Default: Use GPU-optimized SpatialFilterScorer
        from .scoring import SpatialFilterScorer
        scorer = SpatialFilterScorer(method="2d_conv", norm="l1")
    
    with torch.no_grad():
        tile_scores = scorer.score_tokens(metric, H=h, W=w)  # (B, N)
        scores_2d = tile_scores.view(B, h, w)  # (B, H, W)
        
        # Effective region fully divisible by tile sizes
        h_eff = hsy * sy
        w_eff = wsx * sx
        scores_cropped = scores_2d[:, :h_eff, :w_eff]  # (B, h_eff, w_eff)
        
        # ---- OPTIMIZATION: Use cuDNN pooling for tile aggregation when possible ----
        scores_eff_4d = scores_cropped.unsqueeze(1)  # (B, 1, h_eff, w_eff) for pooling ops
        
        if tile_aggregation in ("mean", "avg", "sum", "max", "min", "std"):
            if tile_aggregation in ("mean", "avg"):
                pooled = F.avg_pool2d(scores_eff_4d, kernel_size=(sy, sx), stride=(sy, sx))  # (B,1,hsy,wsx)
                tile_aggregated = pooled.squeeze(1)  # (B,hsy,wsx)
            elif tile_aggregation == "sum":
                pooled = F.avg_pool2d(scores_eff_4d, kernel_size=(sy, sx), stride=(sy, sx))
                tile_aggregated = (pooled * (sy * sx)).squeeze(1)
            elif tile_aggregation == "max":
                pooled = F.max_pool2d(scores_eff_4d, kernel_size=(sy, sx), stride=(sy, sx))
                tile_aggregated = pooled.squeeze(1)
            elif tile_aggregation == "min":
                pooled = F.max_pool2d(-scores_eff_4d, kernel_size=(sy, sx), stride=(sy, sx))
                tile_aggregated = (-pooled).squeeze(1)
            elif tile_aggregation == "std":
                # Compute E[x] and E[x^2] using pooling, then std = sqrt(E[x^2] - E[x]^2)
                m1 = F.avg_pool2d(scores_eff_4d, kernel_size=(sy, sx), stride=(sy, sx))      # E[x]
                m2 = F.avg_pool2d(scores_eff_4d * scores_eff_4d, kernel_size=(sy, sx), stride=(sy, sx))  # E[x^2]
                var = (m2 - m1 * m1).clamp_min(0.0)
                tile_aggregated = torch.sqrt(var).squeeze(1)
        elif tile_aggregation == "median":
            # For median, fall back to manual method (pooling can't do median)
            scores_tiled = scores_cropped.view(B, hsy, sy, wsx, sx)  # (B, hsy, sy, wsx, sx)
            scores_flat = scores_tiled.view(B, hsy, wsx, sy * sx)    # (B, hsy, wsx, tile_area)
            tile_aggregated = scores_flat.median(dim=-1).values
        else:
            raise ValueError(
                f"Unknown tile_aggregation method: {tile_aggregation}. "
                f"Choose from ['mean','max','min','sum','std','median']"
            )
        
        tile_scores_flat = tile_aggregated.view(B, -1)  # (B, total_tiles)
        # Pick smoothest tiles (lowest scores) - ensure we don't exceed available tiles
        tiles_to_merge = min(tiles_to_merge, tile_scores_flat.shape[1])
        _, smooth_indices = torch.topk(tile_scores_flat, tiles_to_merge, dim=1, largest=False)  # (B, tiles_to_merge)
        
        # Create boolean mask (B, total_tiles)
        mergeable_mask = torch.zeros(B, total_tiles, dtype=torch.bool, device=metric.device)
        batch_arange = torch.arange(B, device=metric.device).unsqueeze(1)
        mergeable_mask[batch_arange, smooth_indices] = True
        
        # Build index maps for efficient gather/scatter operations
        # Global flat indices [0..N-1] reshaped as H×W
        idx_grid = torch.arange(h * w, device=metric.device, dtype=torch.long).view(h, w)
        eff_idx_grid = idx_grid[:h_eff, :w_eff]
        
        # (total_tiles, tile_area) mapping from tile id → its flat indices
        tile_indices = (
            eff_idx_grid.view(hsy, sy, wsx, sx)       # (hsy, sy, wsx, sx)
            .permute(0, 2, 1, 3)                      # (hsy, wsx, sy, sx)
            .contiguous()
            .view(total_tiles, tile_area)             # (T, A)
        )
        
        # Edge tokens (outside effective grid): right strip + bottom strip (no overlap)
        right_strip = idx_grid[:h_eff, w_eff:]        # (h_eff, w - w_eff)
        bottom_strip = idx_grid[h_eff:, :]            # (h - h_eff, w)
        edge_indices = torch.cat([
            right_strip.reshape(-1),
            bottom_strip.reshape(-1)
        ], dim=0).to(metric.device)                  # (E,)
        
        # --- OPTIMIZATION 1: Vectorized tile ID creation ---
        # Each batch has exactly tiles_to_merge True values and (total_tiles - tiles_to_merge) False values
        # due to the uniform topk operation, so we can safely reshape
        merge_idx_pairs = torch.nonzero(mergeable_mask, as_tuple=False)               # (B*M, 2) [batch_id, tile_id]
        keep_idx_pairs  = torch.nonzero(~mergeable_mask, as_tuple=False)              # (B*K, 2) [batch_id, tile_id]

        merge_tile_ids_tensor = merge_idx_pairs[:, 1].view(B, tiles_to_merge)         # (B, M)
        keep_tile_ids_tensor  = keep_idx_pairs[:, 1].view(B, total_tiles - tiles_to_merge)  # (B, K)

        # Create legacy lists efficiently from tensor results (no double computation)
        keep_tile_ids = [keep_tile_ids_tensor[b] for b in range(B)]
        merge_tile_ids = [merge_tile_ids_tensor[b] for b in range(B)]
        
        # Cache for merge/unmerge
        merge_info = {
            'tile_indices': tile_indices.to(metric.device),   # (T, A) - legacy for compatibility
            'edge_indices': edge_indices.to(metric.device),   # (E,) - legacy for compatibility
            'keep_tile_ids': keep_tile_ids,                   # legacy list of (K,) - backward compatibility
            'merge_tile_ids': merge_tile_ids,                 # legacy list of (M,) - backward compatibility
            # New vectorized tensors for optimized implementation
            'keep_tile_ids_tensor': keep_tile_ids_tensor,     # (B, K)
            'merge_tile_ids_tensor': merge_tile_ids_tensor,   # (B, M)
            # Geometry information for optimized tile operations
            'tile_area': tile_area,
            'N_full': N,
            'E': edge_indices.numel(),
            'K': total_tiles - tiles_to_merge,
            'M': tiles_to_merge,
            'sx': sx, 'sy': sy, 'h': h, 'w': w,
            'hsy': hsy, 'wsx': wsx, 'h_eff': h_eff, 'w_eff': w_eff,  # Added for optimized implementation
        }
        
    
    # Optionally compile functions for additional speedup
    if use_compile:
        merge_fn, unmerge_fn = _try_compile_abp_functions()
    else:
        merge_fn, unmerge_fn = _abp_merge_batch_tiles_optimized, _abp_unmerge_batch_tiles_optimized
    
    def merge(x: torch.Tensor, mode: str = "mean"):
        # --- OPTIMIZATION 6: Use memory-optimized tile implementation for better GPU performance ---
        return merge_fn(x, merge_info, mode=mode)
        
    def unmerge(x: torch.Tensor, mode: str = None):
        # --- OPTIMIZATION 6: Use memory-optimized tile implementation for better GPU performance ---
        return unmerge_fn(x, merge_info)
    
    return merge, unmerge


def _abp_merge_batch(x: torch.Tensor, merge_info: dict, mode: Optional[str] = "mean") -> torch.Tensor:
    """
    ABP merge that:
      - keeps all tokens in non-merged tiles (sx*sy each),
      - outputs a single pooled token per merged tile,
      - keeps edge tokens unchanged,
      - returns a batch-uniform reduced length.
    """
    B, N, C = x.shape
    tile_indices: torch.Tensor = merge_info['tile_indices']   # (T, A)
    edge_indices: torch.Tensor = merge_info['edge_indices']   # (E,)
    keep_tile_ids: list = merge_info['keep_tile_ids']         # list of (K,)
    merge_tile_ids: list = merge_info['merge_tile_ids']       # list of (M,)
    A = merge_info['tile_area']
    E = merge_info['E']
    K = merge_info['K']
    M = merge_info['M']

    # Use gather workaround for MPS compatibility if needed
    gather = mps_gather_workaround if x.device.type == "mps" else torch.gather

    out_list = []
    for b in range(B):
        # 1) Edge tokens (unchanged)
        edge_tokens = x[b, edge_indices, :] if E > 0 else x.new_zeros((0, C))

        # 2) Keep tiles: gather all sx*sy tokens per tile
        keep_ids_b = keep_tile_ids[b]               # (K,)
        if K > 0:
            keep_tile_idx = tile_indices[keep_ids_b].reshape(-1)  # (K*A,)
            keep_tokens = x[b, keep_tile_idx, :]                  # (K*A, C)
        else:
            keep_tokens = x.new_zeros((0, C))

        # 3) Merge tiles: pool within each tile to 1 token
        merge_ids_b = merge_tile_ids[b]             # (M,)
        if M > 0:
            merge_tile_idx = tile_indices[merge_ids_b]            # (M, A)
            # (M*A, C) -> (M, A, C)
            merge_tiles_tokens = x[b, merge_tile_idx.reshape(-1), :].view(M, A, C)

            if mode == "mlerp":
                pooled = merge_tiles_tokens.mean(dim=1)           # (M, C)
                max_norms = merge_tiles_tokens.norm(dim=-1).amax(dim=1)   # (M,)
                mean_norms = pooled.norm(dim=-1).clamp_min(1e-6)          # (M,)
                pooled = pooled * (max_norms / mean_norms).unsqueeze(-1)   # (M, C)
            else:
                # default to mean
                pooled = merge_tiles_tokens.mean(dim=1)           # (M, C)
        else:
            pooled = x.new_zeros((0, C))

        out_b = torch.cat([edge_tokens, keep_tokens, pooled], dim=0)
        out_list.append(out_b)

    # All batches have the same length: E + K*A + M
    return torch.stack(out_list, dim=0)


def _abp_unmerge_batch(x: torch.Tensor, merge_info: dict) -> torch.Tensor:
    """
    ABP unmerge:
      - scatter edge tokens back to edge indices
      - scatter full tokens for non-merged tiles
      - duplicate each merged tile's pooled token across its tile positions
    """
    B, N_reduced, C = x.shape
    N_full: int = merge_info['N_full']
    tile_indices: torch.Tensor = merge_info['tile_indices']   # (T, A)
    edge_indices: torch.Tensor = merge_info['edge_indices']   # (E,)
    keep_tile_ids: list = merge_info['keep_tile_ids']         # list of (K,)
    merge_tile_ids: list = merge_info['merge_tile_ids']       # list of (M,)
    A = merge_info['tile_area']
    E = merge_info['E']
    K = merge_info['K']
    M = merge_info['M']

    out = x.new_zeros((B, N_full, C))

    split_keep = E + K * A  # boundary between keep-region and merged-region in reduced tensor

    for b in range(B):
        # segments
        if E > 0:
            edge_tokens = x[b, :E, :]                                 # (E, C)
            out[b, edge_indices, :] = edge_tokens

        if K > 0:
            keep_tokens = x[b, E:split_keep, :].view(K, A, C)         # (K, A, C)
            keep_idx_flat = tile_indices[keep_tile_ids[b]].reshape(-1)  # (K*A,)
            out[b, keep_idx_flat, :] = keep_tokens.reshape(-1, C)

        if M > 0:
            merged_tokens = x[b, split_keep:, :]                      # (M, C)
            merge_idx = tile_indices[merge_tile_ids[b]]               # (M, A)
            # repeat merged token across tile positions
            repeated = merged_tokens.unsqueeze(1).expand(M, A, C).reshape(M * A, C)
            out[b, merge_idx.reshape(-1), :] = repeated

    return out


def _abp_merge_batch_vectorized(x: torch.Tensor, merge_info: dict, mode: Optional[str] = "mean") -> torch.Tensor:
    """
    OPTIMIZED: Vectorized ABP merge that eliminates per-batch Python loops.
    
    ABP merge that:
      - keeps all tokens in non-merged tiles (sx*sy each),
      - outputs one pooled token per merged tile,
      - keeps edge tokens unchanged,
      - returns a batch-uniform reduced length: E + K*A + M.
    """
    B, N, C = x.shape
    tile_indices: torch.Tensor = merge_info['tile_indices']           # (T, A)
    edge_indices: torch.Tensor = merge_info['edge_indices']           # (E,)
    keep_tile_ids_tensor: torch.Tensor  = merge_info['keep_tile_ids_tensor']   # (B, K)
    merge_tile_ids_tensor: torch.Tensor = merge_info['merge_tile_ids_tensor']  # (B, M)
    A = merge_info['tile_area']
    E = merge_info['E']
    K = merge_info['K']
    M = merge_info['M']

    # 1) Edge tokens (unchanged) - vectorized for all batches
    if E > 0:
        idx_edge = edge_indices.view(1, E, 1).expand(B, E, C)         # (B, E, C)
        edge_tokens_all = torch.gather(x, 1, idx_edge)                # (B, E, C)
    else:
        edge_tokens_all = x.new_zeros((B, 0, C))

    # 2) Keep tiles: gather all sx*sy tokens per kept tile - vectorized
    if K > 0:
        # (B, K, A) -> (B, K*A)
        keep_tile_idx_all  = tile_indices[keep_tile_ids_tensor]       # (B, K, A)
        keep_flat = keep_tile_idx_all.reshape(B, K * A)               # (B, K*A)
        idx_keep = keep_flat.unsqueeze(-1).expand(B, K * A, C)        # (B, K*A, C)
        keep_tokens_all = torch.gather(x, 1, idx_keep)                # (B, K*A, C)
    else:
        keep_tokens_all = x.new_zeros((B, 0, C))

    # 3) Merge tiles: pool within each tile to 1 token - vectorized
    if M > 0:
        merge_tile_idx_all = tile_indices[merge_tile_ids_tensor]      # (B, M, A)
        merge_flat = merge_tile_idx_all.reshape(B, M * A)             # (B, M*A)
        idx_merge = merge_flat.unsqueeze(-1).expand(B, M * A, C)      # (B, M*A, C)
        merge_tiles_tokens = torch.gather(x, 1, idx_merge)            # (B, M*A, C)
        merge_tiles_tokens = merge_tiles_tokens.view(B, M, A, C)      # (B, M, A, C)

        if mode == "mlerp":
            pooled = merge_tiles_tokens.mean(dim=2)                   # (B, M, C)
            # Fixed: removed .values from amax() which returns a tensor, not named tuple
            max_norms = merge_tiles_tokens.norm(dim=-1).amax(dim=2)   # (B, M)
            mean_norms = pooled.norm(dim=-1).clamp_min(1e-6)         # (B, M)
            pooled = pooled * (max_norms / mean_norms).unsqueeze(-1) # (B, M, C)
        else:
            # default mean (same as current behavior)
            pooled = merge_tiles_tokens.mean(dim=2)                   # (B, M, C)
    else:
        pooled = x.new_zeros((B, 0, C))

    # Concatenate in the same order as original: [edge | keep | pooled]
    return torch.cat([edge_tokens_all, keep_tokens_all, pooled], dim=1)  # (B, E + K*A + M, C)


def _abp_unmerge_batch_vectorized(x: torch.Tensor, merge_info: dict) -> torch.Tensor:
    """
    OPTIMIZED: Vectorized ABP unmerge that eliminates per-batch Python loops.
    
    ABP unmerge:
      - scatter edge tokens back to edge indices
      - scatter full tokens for non-merged tiles
      - duplicate each merged tile's pooled token across its tile positions
    """
    B, N_reduced, C = x.shape
    N_full: int = merge_info['N_full']
    tile_indices: torch.Tensor = merge_info['tile_indices']           # (T, A)
    edge_indices: torch.Tensor = merge_info['edge_indices']           # (E,)
    keep_tile_ids_tensor: torch.Tensor  = merge_info['keep_tile_ids_tensor']   # (B, K)
    merge_tile_ids_tensor: torch.Tensor = merge_info['merge_tile_ids_tensor']  # (B, M)
    A = merge_info['tile_area']
    E = merge_info['E']
    K = merge_info['K']
    M = merge_info['M']

    out = x.new_zeros((B, N_full, C))

    # Segment boundaries in reduced tensor (same order as merge)
    split_keep = E + K * A  # boundary between keep-region and merged-region

    # 1) Edge tokens -> original edge positions - vectorized
    if E > 0:
        edge_tokens = x[:, :E, :]                                        # (B, E, C)
        idx_edge = edge_indices.view(1, E, 1).expand(B, E, C)            # (B, E, C)
        out.scatter_(1, idx_edge, edge_tokens)

    # 2) Kept tiles -> scatter all A positions per kept tile - vectorized
    if K > 0:
        keep_tokens = x[:, E:split_keep, :].view(B, K, A, C)             # (B, K, A, C)
        keep_idx_all = tile_indices[keep_tile_ids_tensor]                 # (B, K, A)
        keep_flat    = keep_idx_all.view(B, K * A)                        # (B, K*A)
        idx_keep     = keep_flat.unsqueeze(-1).expand(B, K * A, C)        # (B, K*A, C)
        out.scatter_(1, idx_keep, keep_tokens.view(B, K * A, C))

    # 3) Merged tiles -> duplicate pooled token across A positions - vectorized
    if M > 0:
        merged_tokens = x[:, split_keep:, :]                              # (B, M, C)
        merge_idx_all = tile_indices[merge_tile_ids_tensor]               # (B, M, A)
        merge_flat    = merge_idx_all.view(B, M * A)                      # (B, M*A)
        idx_merge     = merge_flat.unsqueeze(-1).expand(B, M * A, C)      # (B, M*A, C)

        # Expand merged tokens to all tile positions
        merged_rep = merged_tokens.unsqueeze(2).expand(B, M, A, C).reshape(B, M * A, C)  # (B, M*A, C)
        out.scatter_(1, idx_merge, merged_rep)

    return out


def _abp_merge_batch_tiles_optimized(x: torch.Tensor, merge_info: dict, mode: Optional[str] = "mean") -> torch.Tensor:
    """
    OPTIMIZED: Memory-efficient ABP merge using tile views and cuDNN-friendly layout.
    
    This eliminates large index tensor materializations and uses slice-based edge handling.
    Output order: [edge | kept | pooled] with identical intra-tile ordering as before.
    
    Performance improvements:
    - No (B, K*A, C) index expansions 
    - Slice-based edge handling instead of index grids
    - Tile views eliminate gather/scatter overhead
    - Same mathematical result as original implementation
    """
    B, N, C = x.shape
    
    # Extract geometry from merge_info
    h = merge_info['h']
    w = merge_info['w'] 
    sy = merge_info['sy']
    sx = merge_info['sx']
    hsy = merge_info['hsy']
    wsx = merge_info['wsx']
    h_eff = merge_info['h_eff']
    w_eff = merge_info['w_eff']
    A = merge_info['tile_area']
    E = merge_info['E']
    K = merge_info['K']
    M = merge_info['M']
    
    # Extract precomputed tile coordinates
    keep_tile_ids_tensor = merge_info['keep_tile_ids_tensor']    # (B, K)
    merge_tile_ids_tensor = merge_info['merge_tile_ids_tensor']  # (B, M)
    
    # Convert tile IDs to (ty, tx) coordinates for efficient indexing
    keep_ty = (keep_tile_ids_tensor // wsx).contiguous()   # (B, K) 
    keep_tx = (keep_tile_ids_tensor % wsx).contiguous()    # (B, K)
    merge_ty = (merge_tile_ids_tensor // wsx).contiguous() # (B, M)
    merge_tx = (merge_tile_ids_tensor % wsx).contiguous()  # (B, M)

    # View input as 4D grid for efficient tile operations
    x4 = x.view(B, h, w, C)
    eff = x4[:, :h_eff, :w_eff, :]                                    # (B, h_eff, w_eff, C)
    tiles = eff.view(B, hsy, sy, wsx, sx, C)                          # (B, hsy, sy, wsx, sx, C)

    # 1) Edge tokens: use pure slices (no index materialization)
    if E > 0:
        right = x4[:, :h_eff, w_eff:, :].contiguous().view(B, -1, C)  # (B, h_eff*(w-w_eff), C)
        bottom = x4[:, h_eff:, :, :].contiguous().view(B, -1, C)      # (B, (h-h_eff)*w, C)
        edge_tokens = torch.cat([right, bottom], dim=1)               # (B, E, C)
    else:
        edge_tokens = x.new_zeros((B, 0, C))

    # 2) Kept tiles: gather by tile coordinates, flatten in row-major order
    if K > 0:
        bidx = torch.arange(B, device=x.device).view(B, 1)
        keep_tiles = tiles[bidx, keep_ty, :, keep_tx, :, :]           # (B, K, sy, sx, C)
        keep_tokens = keep_tiles.contiguous().view(B, K * A, C)       # (B, K*A, C)
    else:
        keep_tokens = x.new_zeros((B, 0, C))

    # 3) Merged tiles: pool within each tile to 1 token
    if M > 0:
        bidx = torch.arange(B, device=x.device).view(B, 1)
        merge_tiles = tiles[bidx, merge_ty, :, merge_tx, :, :]        # (B, M, sy, sx, C)
        merge_tiles = merge_tiles.view(B, M, A, C)                    # (B, M, A, C)

        if mode == "mlerp":
            pooled = merge_tiles.mean(dim=2)                          # (B, M, C)
            # amax() returns tensor, not named tuple (no .values)
            max_norms = merge_tiles.norm(dim=-1).amax(dim=2)          # (B, M)
            mean_norms = pooled.norm(dim=-1).clamp_min(1e-6)          # (B, M)
            pooled = pooled * (max_norms / mean_norms).unsqueeze(-1)  # (B, M, C)
        else:
            pooled = merge_tiles.mean(dim=2)                          # (B, M, C)
    else:
        pooled = x.new_zeros((B, 0, C))

    # Output maintains same order as original: [edge | kept | pooled]
    return torch.cat([edge_tokens, keep_tokens, pooled], dim=1)       # (B, E + K*A + M, C)


def _abp_unmerge_batch_tiles_optimized(x: torch.Tensor, merge_info: dict) -> torch.Tensor:
    """
    OPTIMIZED: Memory-efficient ABP unmerge that reverses the optimized merge exactly.
    
    This restores the original spatial structure using tile views and slices.
    """
    B, N_red, C = x.shape
    N_full = merge_info['N_full']
    
    # Extract geometry from merge_info  
    h = merge_info['h']
    w = merge_info['w']
    sy = merge_info['sy'] 
    sx = merge_info['sx']
    hsy = merge_info['hsy']
    wsx = merge_info['wsx']
    h_eff = merge_info['h_eff']
    w_eff = merge_info['w_eff']
    A = merge_info['tile_area']
    E = merge_info['E']
    K = merge_info['K']
    M = merge_info['M']
    
    # Extract precomputed tile coordinates
    keep_tile_ids_tensor = merge_info['keep_tile_ids_tensor']    # (B, K)
    merge_tile_ids_tensor = merge_info['merge_tile_ids_tensor']  # (B, M)
    
    # Convert tile IDs to coordinates
    keep_ty = (keep_tile_ids_tensor // wsx).contiguous()   # (B, K)
    keep_tx = (keep_tile_ids_tensor % wsx).contiguous()    # (B, K) 
    merge_ty = (merge_tile_ids_tensor // wsx).contiguous() # (B, M)
    merge_tx = (merge_tile_ids_tensor % wsx).contiguous()  # (B, M)

    # Initialize output and create views
    out = x.new_zeros((B, N_full, C))
    out4 = out.view(B, h, w, C)

    split_keep = E + K * A  # boundary between keep region and merged region

    # 1) Scatter edge tokens back using slices (exact reverse of merge)
    if E > 0:
        right_elems = h_eff * (w - w_eff)
        right_tokens = x[:, :right_elems, :]                         # (B, right, C)
        bottom_tokens = x[:, right_elems:E, :]                       # (B, bottom, C)

        out4[:, :h_eff, w_eff:, :] = right_tokens.view(B, h_eff, w - w_eff, C)
        out4[:, h_eff:, :, :] = bottom_tokens.view(B, h - h_eff, w, C)

    # 2) Scatter kept tiles back to their tile positions  
    if K > 0:
        keep_tokens = x[:, E:split_keep, :].view(B, K, sy, sx, C)     # (B, K, sy, sx, C)
        bidx = torch.arange(B, device=x.device).view(B, 1)
        tiles_out = out4[:, :h_eff, :w_eff, :].view(B, hsy, sy, wsx, sx, C)  # (B, hsy, sy, wsx, sx, C)
        tiles_out[bidx, keep_ty, :, keep_tx, :, :] = keep_tokens

    # 3) Scatter merged tiles: duplicate pooled token across tile positions
    if M > 0:
        merged_tokens = x[:, split_keep:, :].unsqueeze(2).unsqueeze(3)  # (B, M, 1, 1, C)
        merged_rep = merged_tokens.expand(B, M, sy, sx, C)              # (B, M, sy, sx, C)
        bidx = torch.arange(B, device=x.device).view(B, 1)
        tiles_out = out4[:, :h_eff, :w_eff, :].view(B, hsy, sy, wsx, sx, C)
        tiles_out[bidx, merge_ty, :, merge_tx, :, :] = merged_rep

    return out


def _try_compile_abp_functions():
    """
    Attempt to compile ABP functions with torch.compile for additional speedup.
    Falls back gracefully if compilation is not available or fails.
    
    Returns compiled functions or original functions as fallback.
    """
    try:
        # Try to compile with dynamic=True for variable batch sizes
        compiled_merge = torch.compile(_abp_merge_batch_tiles_optimized, dynamic=True)
        compiled_unmerge = torch.compile(_abp_unmerge_batch_tiles_optimized, dynamic=True)
        return compiled_merge, compiled_unmerge
    except Exception:
        # Fall back to non-compiled versions
        return _abp_merge_batch_tiles_optimized, _abp_unmerge_batch_tiles_optimized


def test_abp_optimization_correctness(B=1, H=32, W=32, C=64, sx=2, sy=2, r_ratio=0.25, device='cuda', dtype=torch.float16):
    """
    Test function to verify correctness and benchmark performance of optimized ABP implementation.
    
    Args:
        B, H, W, C: tensor dimensions
        sx, sy: tile sizes  
        r_ratio: fraction of tokens to remove
        device: torch device
        dtype: torch dtype
        
    Returns:
        Dict with correctness results and timing info
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        dtype = torch.float32
    
    N = H * W
    r = int(N * r_ratio)
    
    # Create test data
    x = torch.randn(B, N, C, device=device, dtype=dtype)
    metric = x.clone()
    
    # Test both implementations
    from .scoring import SpatialFilterScorer
    scorer = SpatialFilterScorer(method="2d_conv", norm="l1")
    
    # Original implementation (fallback to vectorized)
    merge_orig, unmerge_orig = adaptive_block_pooling_random2d(
        metric, W, H, sx, sy, r, scorer=scorer, tile_aggregation="mean", use_compile=False
    )
    
    # Optimized implementation 
    merge_opt, unmerge_opt = adaptive_block_pooling_random2d(
        metric.clone(), W, H, sx, sy, r, scorer=scorer, tile_aggregation="mean", use_compile=False
    )
    
    # Test merge
    y_orig = merge_orig(x.clone(), mode='mean')
    y_opt = merge_opt(x.clone(), mode='mean')
    
    # Test unmerge  
    xr_orig = unmerge_orig(y_orig)
    xr_opt = unmerge_opt(y_opt)
    
    # Check shapes
    shape_ok = (y_orig.shape == y_opt.shape) and (xr_orig.shape == xr_opt.shape == x.shape)
    
    # Check numerical differences (should be small due to same math)
    merge_diff = (y_orig - y_opt).abs().mean().item()
    unmerge_diff = (xr_orig - xr_opt).abs().mean().item() 
    
    results = {
        'shapes_correct': shape_ok,
        'merge_numerical_diff': merge_diff,
        'unmerge_numerical_diff': unmerge_diff,
        'original_output_shape': y_orig.shape,
        'optimized_output_shape': y_opt.shape,
        'reconstruction_error_orig': (xr_orig - x).abs().mean().item(),
        'reconstruction_error_opt': (xr_opt - x).abs().mean().item(),
    }
    
    return results


def test_prune_mode_reconstruction(B=2, H=16, W=16, C=32, sx=2, sy=2, r_ratio=0.25, device='cuda'):
    """
    Quick sanity check test for prune mode reconstruction behavior.
    
    Verifies that:
    1. Dst positions round-trip their original values
    2. Pruned src positions equal their matched dst values (duplication)
    3. Gradients flow correctly
    4. Token sizes are handled correctly
    
    Args:
        B, H, W, C: tensor dimensions
        sx, sy: stride values
        r_ratio: fraction of tokens to remove
        device: torch device
        
    Returns:
        Dict with test results
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
    
    N = H * W
    r = int(N * r_ratio)
    
    # Create test data that requires gradients
    x = torch.randn(B, N, C, device=device, requires_grad=True)
    metric = x.clone().detach()
    
    # Create merge/unmerge functions
    merge, unmerge = bipartite_soft_matching_random2d(
        metric, W, H, sx, sy, r, no_rand=True  # Use deterministic for reproducible test
    )
    
    # Test prune mode
    y_prune = merge(x, mode="prune")
    x_reconstructed = unmerge(y_prune)
    
    # Test normal mode for comparison
    y_normal = merge(x.detach(), mode="mean")
    x_normal_reconstructed = unmerge(y_normal)
    
    # Verify shapes
    shapes_correct = (
        y_prune.shape[1] == N - r and
        x_reconstructed.shape == x.shape and
        x_normal_reconstructed.shape == x.shape
    )
    
    # Test gradient flow
    try:
        loss = y_prune.sum()
        loss.backward()
        gradients_flow = x.grad is not None and x.grad.abs().sum() > 0
    except Exception as e:
        gradients_flow = False
        grad_error = str(e)
    else:
        grad_error = None
    
    # Calculate reconstruction quality
    prune_reconstruction_error = (x_reconstructed - x.detach()).abs().mean().item()
    normal_reconstruction_error = (x_normal_reconstructed - x.detach()).abs().mean().item()
    
    results = {
        'shapes_correct': shapes_correct,
        'gradients_flow': gradients_flow,
        'prune_reconstruction_error': prune_reconstruction_error,
        'normal_reconstruction_error': normal_reconstruction_error,
        'prune_output_shape': y_prune.shape,
        'expected_output_shape': (B, N - r, C),
        'gradient_error': grad_error,
        'test_passed': shapes_correct and gradients_flow and prune_reconstruction_error < 10.0  # Reasonable threshold
    }
    
    return results


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def mlerp_merge_batched(dst, src, idx_dst, eps=1e-6):
    """
    Efficient batched MLERP (Maximum-Norm Linear Interpolation) merging function.
    
    Args:
        dst: (B, N_dst, C) existing destination tokens before merge
        src: (B, R, C) gathered source tokens  
        idx_dst: (B, R) indices telling where each src row will be merged
        eps: small value to prevent division by zero
        
    Returns:
        dst_scatter: (B, N_dst, C) merged destination tokens with MLERP
    """
    B, N_dst, C = dst.shape
    
    # ---- average first ----
    dst_scatter = dst.scatter_reduce(
        dim=1, index=idx_dst.unsqueeze(-1).expand_as(src),
        src=src, reduce='mean', include_self=True
    )
    
    # ---- compute max norm of each cluster ----
    # Initialize with dst token norms (not zeros!)
    dst_norms = dst.norm(dim=-1)  # (B, N_dst)
    max_norm = dst_norms.clone()
    
    # Compute source token norms and scatter max
    src_norms = src.norm(dim=-1)  # (B, R)
    max_norm = max_norm.scatter_reduce(
        1, idx_dst, src_norms, reduce='amax', include_self=False  # dst already included
    )
    
    # ---- renormalize ----  
    dst_scatter_norms = dst_scatter.norm(dim=-1).clamp_min(eps)  # (B, N_dst)
    scale = max_norm / dst_scatter_norms  # (B, N_dst)
    dst_scatter = dst_scatter * scale.unsqueeze(-1)  # (B, N_dst, C)
    
    return dst_scatter


def mlerp_merge(dst, src, idx_dst, eps=1e-6):
    """
    MLERP (Maximum-Norm Linear Interpolation) merging function.
    Legacy single-batch interface that calls the efficient batched version.
    
    Args:
        dst: (N_dst, C) existing destination tokens before merge
        src: (R, C) gathered source tokens  
        idx_dst: (R,) indices telling where each src row will be merged
        eps: small value to prevent division by zero
        
    Returns:
        dst_scatter: (N_dst, C) merged destination tokens with MLERP
    """
    # Add batch dimension and call batched version
    dst_batched = dst.unsqueeze(0)  # (1, N_dst, C)
    src_batched = src.unsqueeze(0)  # (1, R, C) 
    idx_dst_batched = idx_dst.unsqueeze(0)  # (1, R)
    
    result_batched = mlerp_merge_batched(dst_batched, src_batched, idx_dst_batched, eps)
    return result_batched.squeeze(0)  # (N_dst, C)


def _compute_timestep_protected_indices(
    token_scores: torch.Tensor,
    timestep_normalized: Optional[float],
    n_protect: int,
    reverse: bool = False
) -> torch.Tensor:
    """
    Helper function to compute protected indices for timestep-based scheduling.
    
    Args:
        token_scores: Token scores tensor (B, N)
        timestep_normalized: Normalized timestep value(s)
        n_protect: Number of tokens to protect
        reverse: If True, use reverse timestep scheduling logic
        
    Returns:
        protected_indices: Indices of tokens to protect (B, n_protect)
    """
    B, N = token_scores.shape
    
    if timestep_normalized is None:
        raise ValueError("timestep_normalized must be provided for timestep scheduler mode")
    
    # Sort all tokens by score, highest score first
    sorted_indices = torch.argsort(token_scores, dim=1, descending=True)  # (B, N)
    
    # Handle both single timestep and batched timesteps
    if isinstance(timestep_normalized, (list, torch.Tensor)) and hasattr(timestep_normalized, '__len__') and len(timestep_normalized) == B:
        # Batched timesteps: each sample gets its own normalized timestep
        if isinstance(timestep_normalized, list):
            t_batch = torch.tensor(timestep_normalized, device=token_scores.device, dtype=torch.float32)
        else:
            t_batch = timestep_normalized.to(token_scores.device).float()
        t_batch = torch.clamp(t_batch, 0.0, 1.0)
        
        # Calculate per-sample start indices
        if reverse:
            start_indices = ((N - n_protect) * (1.0 - t_batch)).long()  # Reverse logic
        else:
            start_indices = ((N - n_protect) * t_batch).long()  # Normal logic
        start_indices = torch.clamp(start_indices, 0, N - n_protect)
        
        # Select tokens per sample
        protected_indices_list = []
        for i in range(B):
            s_idx = start_indices[i].item()
            e_idx = s_idx + n_protect
            protected_indices_list.append(sorted_indices[i, s_idx:e_idx])
        protected_indices = torch.stack(protected_indices_list, dim=0)
    else:
        # Single timestep for all samples
        t = torch.clamp(torch.tensor(timestep_normalized, device=token_scores.device, dtype=torch.float32), 0.0, 1.0)
        if reverse:
            start_idx = ((N - n_protect) * (1.0 - t)).long()  # Reverse logic
        else:
            start_idx = ((N - n_protect) * t).long()  # Normal logic
        start_idx = torch.min(start_idx, torch.tensor(N - n_protect, device=start_idx.device))
        end_idx = start_idx + n_protect
        protected_indices = sorted_indices[:, start_idx:end_idx]
    
    return protected_indices


def _compute_protected_indices(
    token_scores: torch.Tensor,
    score_mode: str,
    n_protect: int,
    timestep_normalized: Optional[float] = None
) -> torch.Tensor:
    """
    Helper function to compute protected indices based on different scoring modes.
    
    Args:
        token_scores: Token scores tensor (B, N)
        score_mode: Selection mode ("high", "low", "medium", "uniform", "timestep_scheduler", "reverse_timestep_scheduler")
        n_protect: Number of tokens to protect
        timestep_normalized: Normalized timestep for timestep-based modes
        
    Returns:
        protected_indices: Indices of tokens to protect (B, n_protect)
    """
    B, N = token_scores.shape
    
    if score_mode == "high":
        # Protect tokens with highest scores
        _, protected_indices = torch.topk(token_scores, n_protect, dim=1, largest=True)
    elif score_mode == "low":
        # Protect tokens with lowest scores
        _, protected_indices = torch.topk(token_scores, n_protect, dim=1, largest=False)
    elif score_mode == "medium":
        # Protect tokens with medium scores
        sorted_indices = torch.argsort(token_scores, dim=1, descending=False)
        start_idx = (N - n_protect) // 2
        end_idx = start_idx + n_protect
        protected_indices = sorted_indices[:, start_idx:end_idx]
    elif score_mode == "uniform":
        # Protect tokens uniformly across the score ranking
        sorted_indices = torch.argsort(token_scores, dim=1, descending=True)  # Sort by score, highest first
        
        # Calculate uniform sampling indices
        if n_protect > 0:
            # Create evenly spaced indices across the range [0, N-1]
            uniform_positions = torch.linspace(0, N - 1, n_protect, dtype=torch.long, device=token_scores.device)
            
            # Select tokens at uniform positions from sorted indices
            protected_indices = sorted_indices[:, uniform_positions]
        else:
            # No tokens to protect
            protected_indices = torch.empty(B, 0, dtype=torch.long, device=token_scores.device)
    elif score_mode == "timestep_scheduler":
        protected_indices = _compute_timestep_protected_indices(
            token_scores, timestep_normalized, n_protect, reverse=False
        )
    elif score_mode == "reverse_timestep_scheduler":
        protected_indices = _compute_timestep_protected_indices(
            token_scores, timestep_normalized, n_protect, reverse=True
        )
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}. Choose from ['high', 'low', 'medium', 'timestep_scheduler', 'reverse_timestep_scheduler', 'uniform']")
    
    return protected_indices


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None,
                                     token_sizes: Optional[torch.Tensor] = None,
                                     token_scores: Optional[torch.Tensor] = None,
                                     locality_block_factor_h: int = 1,
                                     locality_block_factor_w: int = 1) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
     - token_sizes [B, N]: optional tensor tracking the size of each token
     - token_scores [B, N]: optional token importance scores for score-guided dst selection
     - locality_block_factor_h: factor to divide height for locality-based similarity (default 1 = global)
     - locality_block_factor_w: factor to divide width for locality-based similarity (default 1 = global)
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    # Initialize generator if None
    if generator is None and not no_rand:
        from .utils import init_generator
        generator = init_generator(metric.device)

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, assign one token to be dst and the rest src
        if token_scores is not None:
            # Per-sample score-guided destination selection (no large buffers or argsort)
            scores_2d = token_scores.view(B, h, w)  # [B, h, w]

            # Use only the effective region that forms complete blocks; leftover edge tokens stay src
            effective_h = hsy * sy
            effective_w = wsx * sx
            scores_eff = scores_2d[:, :effective_h, :effective_w]

            # Group by blocks and find the lowest-score position per block
            scores_blocks_flat = scores_eff.view(B, hsy, sy, wsx, sx).view(B, hsy, wsx, sy * sx)
            min_pos = scores_blocks_flat.argmin(dim=3)  # [B, hsy, wsx]

            # VECTORIZED: Create per-sample idx_buffers (parallel per-batch spatial patterns)
            # Create all idx_buffer_views at once for all batch items
            idx_buffer_views = torch.zeros(B, hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
            
            # Vectorized scatter operation - each batch gets its own pattern
            idx_buffer_views.scatter_(dim=3, index=min_pos.unsqueeze(-1), 
                                    src=-torch.ones_like(min_pos.unsqueeze(-1), dtype=min_pos.dtype))
            
            # Vectorized reshape for all batches simultaneously
            idx_buffer_views_reshaped = idx_buffer_views.view(B, hsy, wsx, sy, sx).transpose(2, 3).reshape(B, hsy * sy, wsx * sx)
            
            # Vectorized padding/cropping handling
            if (hsy * sy) < h or (wsx * sx) < w:
                idx_buffer_batch = torch.zeros(B, h, w, device=metric.device, dtype=torch.int64)
                idx_buffer_batch[:, :(hsy * sy), :(wsx * sx)] = idx_buffer_views_reshaped
            else:
                idx_buffer_batch = idx_buffer_views_reshaped
            
            # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
            rand_idx = idx_buffer_batch.reshape(B, -1, 1).argsort(dim=1)  # [B, N, 1]
            
            # Create src/dst index tensors for score-guided branch
            num_dst = hsy * wsx
            a_idx = rand_idx[:, num_dst:, :] # src: [B, N-num_dst, 1]
            b_idx = rand_idx[:, :num_dst, :] # dst: [B, num_dst, 1]
            
        elif no_rand:
            # Use deterministic pattern (same for all samples in batch)
            rand_idx_pattern = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
            
            # Create idx_buffer pattern
            idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
            idx_buffer_view.scatter_(dim=2, index=rand_idx_pattern, src=-torch.ones_like(rand_idx_pattern, dtype=rand_idx_pattern.dtype))
            idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

            if (hsy * sy) < h or (wsx * sx) < w:
                idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
                idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
            else:
                idx_buffer = idx_buffer_view

            # Replicate pattern for all samples
            rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1).expand(B, -1, -1)
            
        else:
            # Random pattern (same for all samples in batch - original behavior)
            rand_idx_pattern = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)
            
            # Create idx_buffer pattern
            idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
            idx_buffer_view.scatter_(dim=2, index=rand_idx_pattern, src=-torch.ones_like(rand_idx_pattern, dtype=rand_idx_pattern.dtype))
            idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

            if (hsy * sy) < h or (wsx * sx) < w:
                idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
                idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
            else:
                idx_buffer = idx_buffer_view

            # Replicate pattern for all samples
            rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1).expand(B, -1, -1)  # [B, N, 1]

        # Create src/dst index tensors consistently for all branches
        num_dst = hsy * wsx
        if token_scores is None:
            # rand_idx is [B, N, 1] in non-scoring branches; split into src/dst
            a_idx = rand_idx[:, num_dst:, :] # src: [B, N-num_dst, 1]
            b_idx = rand_idx[:, :num_dst, :] # dst: [B, num_dst, 1]

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = torch.nn.functional.normalize(metric, dim=-1)
        a, b = split(metric)
        
        # Apply locality-based similarity if enabled
        if locality_block_factor_h > 1 or locality_block_factor_w > 1:
            # OPTIMIZED: Block-wise similarity computation (3.3x faster than full matrix + masking)
            # Only compute similarities within blocks instead of computing full matrix then masking
            
            # Get src and dst indices for spatial mapping (same spatial layout for all batches)
            a_idx_flat = a_idx.squeeze(-1)  # (B, N-num_dst)
            b_idx_flat = b_idx.squeeze(-1)  # (B, num_dst)
            
            src_indices = a_idx_flat[0]  # Use spatial layout (same for all batches)
            dst_indices = b_idx_flat[0]  # Use spatial layout (same for all batches)
            
            # Convert 1D indices to 2D spatial coordinates (vectorized)
            src_rows = src_indices // w
            src_cols = src_indices % w
            dst_rows = dst_indices // w
            dst_cols = dst_indices % w
            
            # Determine which sub-block each token belongs to (compute once, reuse for all batches)
            # Add safety checks to prevent division by zero
            block_h = max(1, h // locality_block_factor_h)
            block_w = max(1, w // locality_block_factor_w)
            
            # Ensure we don't exceed the spatial dimensions
            effective_factor_h = min(locality_block_factor_h, h)
            effective_factor_w = min(locality_block_factor_w, w)
            
            # Calculate actual number of blocks for correct block ID assignment
            num_blocks_w = (w + block_w - 1) // block_w  # Ceiling division
            
            src_block_ids = (src_rows // block_h) * num_blocks_w + (src_cols // block_w)
            dst_block_ids = (dst_rows // block_h) * num_blocks_w + (dst_cols // block_w)
            
            # OPTIMIZED: Block-wise computation instead of full matrix + masking
            # This provides ~16x computation reduction and 93.8% memory savings
            # Use dtype-safe large negative value (float16 max is ~65504)
            neg_inf_value = -1e4 if a.dtype == torch.float16 else -1e6
            scores = torch.full((B, a.shape[1], b.shape[1]), neg_inf_value, device=a.device, dtype=a.dtype)
            
            num_blocks = locality_block_factor_h * locality_block_factor_w
            for block_id in range(num_blocks):
                # Find tokens in this block
                src_in_block = (src_block_ids == block_id).nonzero(as_tuple=False).squeeze(-1)
                dst_in_block = (dst_block_ids == block_id).nonzero(as_tuple=False).squeeze(-1)
                
                if len(src_in_block) > 0 and len(dst_in_block) > 0:
                    # Extract and compute similarities only for this block
                    a_block = a[:, src_in_block, :]  # (B, n_src_block, C)
                    b_block = b[:, dst_in_block, :]  # (B, n_dst_block, C)
                    
                    # Efficient block computation using optimized CUDA kernels
                    block_scores = a_block @ b_block.transpose(-1, -2)  # (B, n_src_block, n_dst_block)
                    
                    # Place results back efficiently using broadcasting
                    scores[:, src_in_block[:, None], dst_in_block] = block_scores
        else:
            # Original global similarity computation
            scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        
        if mode == "mlerp":
            # Apply efficient batched MLERP merge
            dst = mlerp_merge_batched(dst, src, dst_idx.squeeze(-1))
        elif mode == "prune":
            # Pruning: don't merge src tokens, just keep dst tokens unchanged
            # This effectively removes the src tokens from the result
            pass  # dst remains unchanged
        else:
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        merge._merge_mode = mode
        
        # Update token sizes if provided
        if token_sizes is not None and hasattr(merge, '_token_sizes'):
            # Split token_sizes manually (2D: B, N) instead of using split() which expects 3D
            # a_idx shape: [B, N-num_dst, 1], b_idx shape: [B, num_dst, 1]
            # merge._token_sizes shape: [B, N]
            
            # For 2D token_sizes, we need to expand the indices correctly
            a_idx_2d = a_idx.squeeze(-1)  # [B, N-num_dst]
            b_idx_2d = b_idx.squeeze(-1)  # [B, num_dst]
            
            src_sizes = gather(merge._token_sizes, dim=1, index=a_idx_2d)  # (B, N-num_dst)
            dst_sizes = gather(merge._token_sizes, dim=1, index=b_idx_2d)  # (B, num_dst)
            
            unm_sizes = gather(src_sizes, dim=1, index=unm_idx.squeeze(-1))
            src_sizes_merge = gather(src_sizes, dim=1, index=src_idx.squeeze(-1))
            
            # Only update dst_sizes for non-prune modes (prune doesn't actually merge values)
            if mode != "prune":
                dst_sizes = dst_sizes.scatter_add(1, dst_idx.squeeze(-1), src_sizes_merge)
            # In prune mode, dst_sizes remain unchanged since dst tokens are unchanged
            
            merge._token_sizes = torch.cat([unm_sizes, dst_sizes], dim=1)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        # Reconstruct tokens from dst (works for both normal merge and prune modes)
        # In normal mode: dst contains merged values → copying gives merged reconstruction
        # In prune mode: dst contains original unchanged values → copying gives reasonable reconstruction
        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out
    
    # Initialize token sizes tracking if provided
    if token_sizes is not None:
        merge._token_sizes = token_sizes.clone()

    return merge, unmerge


def bipartite_soft_matching_from_cached_indices(
        cached_indices: dict,
        w: int, h: int, sx: int, sy: int, r: int,
        device: torch.device) -> Tuple[Callable, Callable]:
    """
    Creates merge/unmerge functions using pre-computed cached indices.
    
    Args:
     - cached_indices: Dict containing pre-computed indices (a_idx, b_idx, unm_idx, src_idx, dst_idx, num_dst)
     - w, h: image dimensions in tokens
     - sx, sy: stride values
     - r: number of tokens to remove
     - device: torch device
    """
    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if device.type == "mps" else torch.gather
    
    # Extract cached indices
    a_idx = cached_indices["a_idx"].to(device)
    b_idx = cached_indices["b_idx"].to(device)
    unm_idx = cached_indices["unm_idx"].to(device)
    src_idx = cached_indices["src_idx"].to(device)
    dst_idx = cached_indices["dst_idx"].to(device)
    num_dst = cached_indices["num_dst"]
    
    N = h * w
    
    def split(x):
        B, _, C = x.shape
        src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
        dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
        return src, dst

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        B, _, C = x.shape
        src, dst = split(x)
        n, t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        
        if mode == "mlerp":
            # Apply efficient batched MLERP merge
            dst = mlerp_merge_batched(dst, src, dst_idx.squeeze(-1))
        elif mode == "prune":
            # Pruning: don't merge src tokens, just keep dst tokens unchanged
            # This effectively removes the src tokens from the result
            pass  # dst remains unchanged
        else:
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        
        # Store merge mode for unmerge function to handle pruning correctly
        merge._merge_mode = mode
        
        # Update token sizes if provided (for proportional attention)
        if hasattr(merge, '_token_sizes') and merge._token_sizes is not None:
            # Split token_sizes manually (2D) instead of using split() which expects 3D
            a_idx_2d = a_idx.squeeze(-1)
            b_idx_2d = b_idx.squeeze(-1)
            
            src_sizes = gather(merge._token_sizes, dim=1, index=a_idx_2d)
            dst_sizes = gather(merge._token_sizes, dim=1, index=b_idx_2d)
            
            unm_sizes = gather(src_sizes, dim=1, index=unm_idx.squeeze(-1))
            src_sizes_merge = gather(src_sizes, dim=1, index=src_idx.squeeze(-1))
            
            # Only update dst_sizes for non-prune modes (prune doesn't actually merge values)
            if mode != "prune":
                dst_sizes = dst_sizes.scatter_add(1, dst_idx.squeeze(-1), src_sizes_merge)
            # In prune mode, dst_sizes remain unchanged since dst tokens are unchanged
            
            merge._token_sizes = torch.cat([unm_sizes, dst_sizes], dim=1)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        # Reconstruct tokens from dst (works for both normal merge and prune modes)
        # In normal mode: dst contains merged values → copying gives merged reconstruction
        # In prune mode: dst contains original unchanged values → copying gives reasonable reconstruction
        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge


def _choose_merge_method(
    x: torch.Tensor,
    w: int, h: int, sx: int, sy: int, r: int,
    no_rand: bool = False,
    generator: torch.Generator = None,
    token_sizes: Optional[torch.Tensor] = None,
    token_scores: Optional[torch.Tensor] = None,
    merge_method: str = "bipartite",
    scorer: Optional['TokenScorer'] = None,
    tile_aggregation: str = "max",
    locality_block_factor_h: int = 1,
    locality_block_factor_w: int = 1,
    use_compile: bool = False
) -> Tuple[Callable, Callable]:
    """
    Wrapper function to choose between ABP and bipartite matching.
    
    Args:
        merge_method: "bipartite" or "abp"
        scorer: TokenScorer for ABP (ignored for bipartite)
        tile_aggregation: Tile aggregation method for ABP (ignored for bipartite)
        Other parameters are passed to the underlying merge function
        
    Returns:
        Tuple of (merge_function, unmerge_function)
    """
    if merge_method == "abp":
        return adaptive_block_pooling_random2d(
            x, w, h, sx, sy, r,
            no_rand=no_rand,
            generator=generator,
            token_sizes=token_sizes,
            token_scores=token_scores,
            scorer=scorer,
            tile_aggregation=tile_aggregation,
            use_compile=use_compile
        )
    else:
        return bipartite_soft_matching_random2d(
            x, w, h, sx, sy, r,
            no_rand=no_rand,
            generator=generator,
            token_sizes=token_sizes,
            token_scores=token_scores,
            locality_block_factor_h=locality_block_factor_h,
            locality_block_factor_w=locality_block_factor_w
        )


def bipartite_soft_matching_with_scoring(
    x: torch.Tensor,
    scorer: 'TokenScorer',
    w: int, h: int, sx: int, sy: int, r: int,
    preserve_ratio: float = 0.3,
    score_mode: str = "high",
    preserve_spatial_uniformity: bool = False,  # NEW PARAMETER
    if_low_frequency_dst_tokens: bool = False,   # NEW PARAMETER for score-guided dst selection
    no_rand: bool = False,
    generator: torch.Generator = None,
    timestep_normalized: Optional[float] = None,
    token_sizes: Optional[torch.Tensor] = None,
    cache_resolution_merge: bool = False,  # NEW: Cache scoring info if enabled
    merge_method: str = "bipartite",  # NEW: Support for ABP vs bipartite
    abp_scorer: Optional['TokenScorer'] = None,  # NEW: ABP scorer
    abp_tile_aggregation: str = "max",  # NEW: ABP tile aggregation method
    locality_block_factor_h: int = 1,  # NEW: Factor to divide height for locality-based similarity
    locality_block_factor_w: int = 1,  # NEW: Factor to divide width for locality-based similarity
    **scorer_kwargs
) -> Tuple[Callable, Callable]:
    """
    Score-based token merging that protects high-importance tokens from being merged.
    
    Args:
        x: Input tensor (B, N, C)
        scorer: TokenScorer instance to compute token importance
        w: image width in tokens
        h: image height in tokens
        sx: stride in x dimension for dst
        sy: stride in y dimension for dst
        r: number of tokens to remove by merging
        preserve_ratio: ratio of tokens to protect from merging (0.0 to 1.0)
        score_mode: how to select tokens based on scores ("high", "low", "medium", "timestep_scheduler", "reverse_timestep_scheduler")
        preserve_spatial_uniformity: if True, apply bipartite matching to full image first, then filter protected tokens
                                   if False, use current approach (extract mergeable subset first)
        if_low_frequency_dst_tokens: if True, select lowest-scored tokens as destinations within each spatial block
                                   if False, use random/first position for destination selection (default behavior)
        no_rand: if true, disable randomness in bipartite matching
        generator: random generator for reproducibility
        timestep_normalized: normalized timestep (0.0 late -> 1.0 early) for timestep_scheduler mode
        locality_block_factor_h: factor to divide height for block-wise similarity computation (default 1 = global)
        locality_block_factor_w: factor to divide width for block-wise similarity computation (default 1 = global)
        **scorer_kwargs: additional arguments for the scorer
        
    Returns:
        Tuple of (merge_function, unmerge_function)
    """
    from .scoring import TokenScorer  # Import here to avoid circular imports
    
    B, N, _ = x.shape
    
    # Check if this is an agent-guided scorer (NEW SIMPLIFIED APPROACH)
    scorer_name = getattr(scorer, 'get_name', lambda: str(type(scorer)))()
    is_agent_guided = 'simplified_agent' in scorer_name or 'agent' in scorer_name.lower()
    
    if is_agent_guided:
        return agent_guided_simple_merge(
            x, scorer, preserve_ratio=preserve_ratio, score_mode=score_mode, 
            w=w, h=h, **scorer_kwargs
        )
    
    # Continue with standard bipartite matching for non-agent scorers
    if r <= 0:
        return do_nothing, do_nothing
    
    # Calculate number of tokens to protect
    n_protect = int(N * preserve_ratio)
    n_protect = max(0, min(n_protect, N - r))  # Ensure we can still merge r tokens
    
    # Skip scoring when preserve_ratio == 0 and no dst guidance needed
    if n_protect == 0 and not if_low_frequency_dst_tokens:
        # No tokens to protect - use regular matching method (bipartite or ABP)
        return _choose_merge_method(
            x, w, h, sx, sy, r, no_rand, generator, token_sizes, None, merge_method,
            abp_scorer, abp_tile_aggregation, locality_block_factor_h, locality_block_factor_w, use_compile=False
        )
    
    # Score tokens using the provided scorer
    with torch.no_grad():
        token_scores = scorer.score_tokens(x, H=h, W=w, **scorer_kwargs)  # (B, N)
        
        # When preserve_ratio == 0 but if_low_frequency_dst_tokens == True
        # Use matching method (bipartite or ABP) with score-guided destinations
        if n_protect == 0:
            return _choose_merge_method(
                x, w, h, sx, sy, r, no_rand, generator, token_sizes, token_scores, merge_method,
                abp_scorer, abp_tile_aggregation, locality_block_factor_h, locality_block_factor_w, use_compile=False
            )
        
        # Select tokens to protect based on score_mode
        protected_indices = _compute_protected_indices(
            token_scores, score_mode, n_protect, timestep_normalized
        )
        
        # Create mask for protected tokens
        protected_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        for i in range(B):
            protected_mask[i, protected_indices[i]] = True
        
        # Get indices of mergeable tokens (non-protected)
        mergeable_mask = ~protected_mask
        mergeable_indices = torch.nonzero(mergeable_mask, as_tuple=False)  # (num_mergeable, 2)
        
        # Group mergeable indices by batch
        mergeable_per_batch = []
        batch_ids = mergeable_indices[:, 0]
        token_ids = mergeable_indices[:, 1]
        for i in range(B):
            mask = batch_ids == i
            mergeable_per_batch.append(token_ids[mask])
        
        # Branch based on preserve_spatial_uniformity parameter
        if preserve_spatial_uniformity:
            return _bipartite_matching_with_spatial_preservation(
                x, w, h, sx, sy, r, protected_mask, protected_indices, no_rand, generator, token_sizes,
                locality_block_factor_h, locality_block_factor_w
            )
        
        # CURRENT APPROACH: Extract mergeable subset first, then apply bipartite matching
        # If we don't have enough mergeable tokens, fall back to original method
        min_mergeable = min(len(batch_indices) for batch_indices in mergeable_per_batch)
        if min_mergeable < r:
            # Not enough mergeable tokens, fallback to original matching method
            return _choose_merge_method(x, w, h, sx, sy, r, no_rand, generator, token_sizes, None, merge_method,
                                        abp_scorer, abp_tile_aggregation, locality_block_factor_h, locality_block_factor_w, use_compile=False)
        
        # Apply bipartite matching only to mergeable tokens
        mergeable_lengths = [len(mergeable_per_batch[i]) for i in range(B)]
        min_mergeable_tokens = min(mergeable_lengths)
        max_mergeable_tokens = max(mergeable_lengths)
        
        # Eliminate padding when possible
        if min_mergeable_tokens == max_mergeable_tokens:
            # All batches same size - use stacking (no padding)
            mergeable_tensors = []
            for i in range(B):
                mergeable_idx = mergeable_per_batch[i]
                mergeable_tensors.append(x[i, mergeable_idx])
            
            x_mergeable = torch.stack(mergeable_tensors, dim=0)
            n_mergeable = min_mergeable_tokens
            valid_lengths = None
        else:
            # Different sizes - need padding
            mergeable_tensors = []
            for i in range(B):
                mergeable_idx = mergeable_per_batch[i]
                mergeable_tensors.append(x[i, mergeable_idx])
            
            padded_mergeable = torch.zeros(B, max_mergeable_tokens, x.shape[-1], device=x.device, dtype=x.dtype)
            valid_lengths = []
            
            for i in range(B):
                batch_mergeable = mergeable_tensors[i]
                valid_lengths.append(len(batch_mergeable))
                padded_mergeable[i, :len(batch_mergeable)] = batch_mergeable
            
            x_mergeable = padded_mergeable
            n_mergeable = max_mergeable_tokens
        
        # Calculate effective dimensions for bipartite matching
        # We need to create a reasonable 2D layout for the mergeable tokens
        if n_mergeable <= 0:
            return do_nothing, do_nothing
            
        # Simple approach: arrange mergeable tokens in a roughly square grid
        h_mergeable = int(math.ceil(math.sqrt(n_mergeable)))
        w_mergeable = int(math.ceil(n_mergeable / h_mergeable))
        
        # Ensure sx and sy are compatible with mergeable dimensions
        sx_mergeable = min(sx, w_mergeable)
        sy_mergeable = min(sy, h_mergeable)
        
        # Apply bipartite matching to mergeable tokens
        gather = mps_gather_workaround if x.device.type == "mps" else torch.gather
        
        # Compute bipartite matching on mergeable subset
        # Extract token sizes for mergeable tokens if provided
        token_sizes_mergeable = None
        if token_sizes is not None:
            token_sizes_mergeable = torch.stack([
                token_sizes[i, mergeable_per_batch[i]] for i in range(B)
            ])
        
        # Extract token scores for mergeable tokens if score-guided dst selection is enabled
        token_scores_mergeable = None
        if if_low_frequency_dst_tokens:
            token_scores_mergeable = torch.stack([
                token_scores[i, mergeable_per_batch[i]] for i in range(B)
            ])
        
        merge_subset, unmerge_subset = _choose_merge_method(
            x_mergeable, w_mergeable, h_mergeable, sx_mergeable, sy_mergeable, r, no_rand, generator,
            token_sizes_mergeable, token_scores_mergeable, merge_method,
            abp_scorer, abp_tile_aggregation, locality_block_factor_h, locality_block_factor_w, use_compile=False
        )
        
        # Map back to original indices
        def merge(x_full: torch.Tensor, mode="mean") -> torch.Tensor:
            B_full, N_full, C_full = x_full.shape
            
            if valid_lengths is None:
                # Uniform case - use vectorized operations
                n_protected = protected_indices.shape[1]
                n_merged_after = min_mergeable_tokens - r
                
                # Pre-allocate result tensor
                result = torch.empty(B_full, n_protected + n_merged_after, C_full, 
                                   device=x_full.device, dtype=x_full.dtype)
                
                # Extract protected tokens
                protected_tokens = torch.gather(x_full, 1, 
                    protected_indices.unsqueeze(-1).expand(-1, -1, C_full))
                
                # Extract mergeable tokens
                max_mergeable_len = max(len(idx) for idx in mergeable_per_batch)
                mergeable_indices_padded = torch.zeros(B_full, max_mergeable_len, 
                                                     device=x_full.device, dtype=torch.long)
                for i in range(B_full):
                    mergeable_len = len(mergeable_per_batch[i])
                    mergeable_indices_padded[i, :mergeable_len] = mergeable_per_batch[i]
                
                mergeable_tokens = torch.gather(x_full, 1,
                    mergeable_indices_padded.unsqueeze(-1).expand(-1, -1, C_full))
                mergeable_tokens = mergeable_tokens[:, :min_mergeable_tokens]  # Trim to uniform size
                
                # Apply merge to mergeable tokens
                merged_tokens = merge_subset(mergeable_tokens, mode)
                
                # Combine results efficiently
                result[:, :n_protected] = protected_tokens
                result[:, n_protected:] = merged_tokens
                
                return result
            else:
                # Non-uniform case - fallback to original logic
                # Extract protected tokens
                protected_tokens = []
                for i in range(B_full):
                    protected_tokens.append(x_full[i, protected_indices[i]])
                
                # Extract mergeable tokens and apply subset merge
                mergeable_tokens = []
                for i in range(B_full):
                    mergeable_idx = mergeable_per_batch[i]
                    mergeable_tokens.append(x_full[i, mergeable_idx])
                
                # Handle padded case
                padded_mergeable = torch.zeros(B_full, max_mergeable_tokens, C_full, device=x_full.device, dtype=x_full.dtype)
                for i in range(B_full):
                    batch_mergeable = mergeable_tokens[i]
                    padded_mergeable[i, :len(batch_mergeable)] = batch_mergeable
                
                merged_padded = merge_subset(padded_mergeable, mode)
                
                # Extract valid merged tokens
                merged_tokens = []
                for i in range(B_full):
                    valid_len = valid_lengths[i] - r  # After merging, we have r fewer tokens
                    merged_tokens.append(merged_padded[i, :valid_len])
                
                # Combine protected and merged tokens
                result_tokens = []
                for i in range(B_full):
                    combined = torch.cat([protected_tokens[i], merged_tokens[i]], dim=0)
                    result_tokens.append(combined)
                
                return torch.stack(result_tokens, dim=0)
        
        def unmerge(x_merged: torch.Tensor) -> torch.Tensor:
            B_full, N_merged, C_full = x_merged.shape
            N_full = N  # Use the original N from the function scope
            
            # Split into protected and merged parts
            protected_tokens = []
            merged_tokens = []
            
            for i in range(B_full):
                n_protected = len(protected_indices[i])
                protected_tokens.append(x_merged[i, :n_protected])
                merged_tokens.append(x_merged[i, n_protected:])
            
            # Apply unmerge to merged tokens
            if min_mergeable_tokens < max(len(batch_indices) for batch_indices in mergeable_per_batch):
                # Handle padded case
                max_merged_tokens = max(len(merged_tokens[i]) for i in range(B_full))
                padded_merged = torch.zeros(B_full, max_merged_tokens, C_full, device=x_merged.device, dtype=x_merged.dtype)
                for i in range(B_full):
                    batch_merged = merged_tokens[i]
                    padded_merged[i, :len(batch_merged)] = batch_merged
                
                unmerged_padded = unmerge_subset(padded_merged)
                
                # Extract valid unmerged tokens
                unmerged_tokens = []
                for i in range(B_full):
                    valid_len = valid_lengths[i]
                    unmerged_tokens.append(unmerged_padded[i, :valid_len])
            else:
                # All batches have same number of merged tokens
                merged_stacked = torch.stack(merged_tokens, dim=0)
                unmerged_stacked = unmerge_subset(merged_stacked)
                unmerged_tokens = [unmerged_stacked[i] for i in range(B_full)]
            
            # Reconstruct original tensor
            result = torch.zeros(B_full, N_full, C_full, device=x_merged.device, dtype=x_merged.dtype)
            
            for i in range(B_full):
                # Place protected tokens
                result[i, protected_indices[i]] = protected_tokens[i]
                # Place unmerged tokens
                result[i, mergeable_per_batch[i]] = unmerged_tokens[i]
            
            return result
        
        # Initialize token sizes tracking if provided
        if token_sizes is not None:
            merge._token_sizes = token_sizes.clone()
            # Update token sizes based on merge_subset if it has token sizes
            if hasattr(merge_subset, '_token_sizes'):
                # Map the merged token sizes back to the full tensor
                for i in range(B):
                    merge._token_sizes[i, mergeable_per_batch[i]] = merge_subset._token_sizes[i]
        
        # Extract caching information during main computation
        if cache_resolution_merge:
            merge._cached_scoring_info = {
                'protected_indices': protected_indices,
                'token_scores': token_scores,
                'mergeable_per_batch': mergeable_per_batch,
                'preserve_ratio': preserve_ratio,
                'score_mode': score_mode,
                'n_protect': n_protect,
                'N': N,
                'preserve_spatial_uniformity': preserve_spatial_uniformity,
                'if_low_frequency_dst_tokens': if_low_frequency_dst_tokens
            }
        
        return merge, unmerge


def _bipartite_matching_with_spatial_preservation(
    x: torch.Tensor,
    w: int, h: int, sx: int, sy: int, r: int,
    protected_mask: torch.Tensor,
    protected_indices: torch.Tensor,
    no_rand: bool = False,
    generator: torch.Generator = None,
    token_sizes: Optional[torch.Tensor] = None,
    locality_block_factor_h: int = 1,
    locality_block_factor_w: int = 1
) -> Tuple[Callable, Callable]:
    """
    Apply bipartite matching to full image first, then filter protected tokens from src/dst.
    This preserves spatial uniformity in token selection.
    
    Args:
        x: Input tensor (B, N, C)
        w, h: image dimensions in tokens
        sx, sy: stride values
        r: number of tokens to remove
        protected_mask: boolean mask indicating protected tokens (B, N)
        protected_indices: indices of protected tokens (B, n_protect)
        no_rand: disable randomness
        generator: random generator
        token_sizes: optional token sizes tensor
        locality_block_factor_h: factor to divide height for locality-based similarity (default 1 = global)
        locality_block_factor_w: factor to divide width for locality-based similarity (default 1 = global)
        
    Returns:
        Tuple of (merge_function, unmerge_function)
    """
    B, N, _ = x.shape
    
    if r <= 0:
        return do_nothing, do_nothing
    
    gather = mps_gather_workaround if x.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # Apply original spatial bipartite matching logic to full image
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=x.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(x.device)
        
        # Create spatial index buffer (same as original ToMe)
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=x.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=x.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # Get dst|src split based on original spatial logic
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src indices
        b_idx = rand_idx[:, :num_dst, :]  # dst indices

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # EFFICIENT APPROACH: Filter src/dst indices first, then compute similarity only on valid pairs
        a_idx_flat = a_idx.squeeze(-1)  # (B, N-num_dst)
        b_idx_flat = b_idx.squeeze(-1)  # (B, num_dst)
        
        # Create efficient boolean mask for protected tokens (B, N)
        # This avoids any CPU transfers and works entirely on GPU
        protected_mask_full = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        for batch_idx in range(B):
            if len(protected_indices[batch_idx]) > 0:
                protected_mask_full[batch_idx, protected_indices[batch_idx]] = True
        
        # Filter src and dst indices efficiently using advanced indexing
        filtered_src_indices = []
        filtered_dst_indices = []
        
        # The spatial indices are the same for all batches (a_idx_flat and b_idx_flat have batch size 1)
        # So we use index 0 for the spatial layout but apply batch-specific protection masks
        src_indices_spatial = a_idx_flat[0]  # (N-num_dst,) - spatial layout is same for all batches
        dst_indices_spatial = b_idx_flat[0]  # (num_dst,) - spatial layout is same for all batches
        
        for batch_idx in range(B):
            # Create masks by checking if indices are NOT in protected set
            src_not_protected = ~protected_mask_full[batch_idx, src_indices_spatial]
            dst_not_protected = ~protected_mask_full[batch_idx, dst_indices_spatial]
            
            # Filter indices using the same spatial layout but batch-specific protection
            filtered_src_batch = src_indices_spatial[src_not_protected]
            filtered_dst_batch = dst_indices_spatial[dst_not_protected]
            
            filtered_src_indices.append(filtered_src_batch)
            filtered_dst_indices.append(filtered_dst_batch)
        
        # Check if we have enough tokens to merge after filtering
        min_src_tokens = min(len(src_batch) for src_batch in filtered_src_indices) if filtered_src_indices else 0
        min_dst_tokens = min(len(dst_batch) for dst_batch in filtered_dst_indices) if filtered_dst_indices else 0
        
        if min_src_tokens < r or min_dst_tokens == 0:
            # Not enough tokens after filtering, fallback to original method
            # This preserves functionality when preserve_spatial_uniformity doesn't work
            # Note: For spatial preservation, we use bipartite to maintain spatial uniformity
            return bipartite_soft_matching_random2d(x, w, h, sx, sy, r, no_rand, generator, token_sizes, None, locality_block_factor_h, locality_block_factor_w)
        
        # NOW compute similarity only on filtered tokens (much more efficient!)
        metric = x / x.norm(dim=-1, keepdim=True)
        
        # Extract only the filtered src and dst tokens for similarity computation
        # Much simpler and more efficient indexing
        filtered_src_tokens = []
        filtered_dst_tokens = []
        
        for batch_idx in range(B):
            # Simple, direct indexing - much more efficient
            src_indices = filtered_src_indices[batch_idx]
            dst_indices = filtered_dst_indices[batch_idx]
            
            if len(src_indices) > 0 and len(dst_indices) > 0:
                src_tokens = metric[batch_idx, src_indices]  # (n_src, C)
                dst_tokens = metric[batch_idx, dst_indices]  # (n_dst, C)
            else:
                # Handle empty case
                src_tokens = torch.empty(0, metric.shape[-1], device=x.device, dtype=x.dtype)
                dst_tokens = torch.empty(0, metric.shape[-1], device=x.device, dtype=x.dtype)
            
            filtered_src_tokens.append(src_tokens)
            filtered_dst_tokens.append(dst_tokens)
        
        # Compute similarity matrices only for valid pairs
        filtered_scores = []
        valid_lengths_src = []
        valid_lengths_dst = []
        
        # OPTIMIZED: Pre-compute block dimensions once (same for all batches)
        if locality_block_factor_h > 1 or locality_block_factor_w > 1:
            # Add safety checks to prevent division by zero
            block_h = max(1, h // locality_block_factor_h)
            block_w = max(1, w // locality_block_factor_w)
            
            # Ensure we don't exceed the spatial dimensions
            effective_factor_h = min(locality_block_factor_h, h)
            effective_factor_w = min(locality_block_factor_w, w)
            
            # Calculate actual number of blocks for correct block ID assignment
            num_blocks_w = (w + block_w - 1) // block_w  # Ceiling division
        
        for batch_idx in range(B):
            src_tokens = filtered_src_tokens[batch_idx]  # (n_src, C)
            dst_tokens = filtered_dst_tokens[batch_idx]  # (n_dst, C)

            if len(src_tokens) > 0 and len(dst_tokens) > 0:
                # Apply locality constraint if enabled using optimized block-wise computation
                if locality_block_factor_h > 1 or locality_block_factor_w > 1:
                    # Get spatial indices for src and dst tokens
                    src_indices = filtered_src_indices[batch_idx]  # (n_src,)
                    dst_indices = filtered_dst_indices[batch_idx]  # (n_dst,)

                    # Convert 1D indices to 2D spatial coordinates (vectorized)
                    src_rows = src_indices // w
                    src_cols = src_indices % w
                    dst_rows = dst_indices // w
                    dst_cols = dst_indices % w

                    # Calculate block IDs using pre-computed dimensions
                    src_block_ids = (src_rows // block_h) * num_blocks_w + (src_cols // block_w)
                    dst_block_ids = (dst_rows // block_h) * num_blocks_w + (dst_cols // block_w)

                    # OPTIMIZED: Use block-wise computation instead of full matrix + masking
                    # Use dtype-safe large negative value (float16 max is ~65504)
                    neg_inf_value = -1e4 if src_tokens.dtype == torch.float16 else -1e6
                    batch_scores = torch.full((len(src_tokens), len(dst_tokens)), neg_inf_value, device=src_tokens.device, dtype=src_tokens.dtype)
                    
                    num_blocks = locality_block_factor_h * locality_block_factor_w
                    for block_id in range(num_blocks):
                        # Find tokens in this block for this batch
                        src_in_block = (src_block_ids == block_id).nonzero(as_tuple=False).squeeze(-1)
                        dst_in_block = (dst_block_ids == block_id).nonzero(as_tuple=False).squeeze(-1)
                        
                        if len(src_in_block) > 0 and len(dst_in_block) > 0:
                            # Extract tokens for this block only
                            src_block_tokens = src_tokens[src_in_block]  # (n_src_block, C)
                            dst_block_tokens = dst_tokens[dst_in_block]  # (n_dst_block, C)
                            
                            # Compute similarities only for this block
                            block_similarities = src_block_tokens @ dst_block_tokens.transpose(-1, -2)
                            
                            # Place results back
                            batch_scores[src_in_block[:, None], dst_in_block] = block_similarities
                else:
                    # Normal global similarity computation when locality is not enabled
                    batch_scores = src_tokens @ dst_tokens.transpose(-1, -2)  # (n_src, n_dst)
            else:
                # Handle empty case - create empty score matrix
                batch_scores = torch.empty(0, 0, device=x.device, dtype=x.dtype)
                
            filtered_scores.append(batch_scores)
            valid_lengths_src.append(len(src_tokens))
            valid_lengths_dst.append(len(dst_tokens))
        
        # Pad scores for batch processing (but now much smaller matrices!)
        max_src = max(valid_lengths_src) if valid_lengths_src else 0
        max_dst = max(valid_lengths_dst) if valid_lengths_dst else 0
        
        # Handle edge case where all tokens are protected
        if max_src == 0 or max_dst == 0:
            # Fallback to original method
            # Note: For spatial preservation, we use bipartite to maintain spatial uniformity
            return bipartite_soft_matching_random2d(x, w, h, sx, sy, r, no_rand, generator, token_sizes, None, locality_block_factor_h, locality_block_factor_w)
        
        # Use dtype-safe large negative value (float16 max is ~65504)
        # Check dtype from first non-empty score tensor, fallback to float32 if all empty
        dtype_to_check = torch.float32
        for score_tensor in filtered_scores:
            if score_tensor.numel() > 0:
                dtype_to_check = score_tensor.dtype
                break
        
        neg_inf_value = -1e4 if dtype_to_check == torch.float16 else float('-inf')
        padded_scores = torch.full((B, max_src, max_dst), neg_inf_value, device=x.device)
        for batch_idx in range(B):
            src_len, dst_len = valid_lengths_src[batch_idx], valid_lengths_dst[batch_idx]
            if src_len > 0 and dst_len > 0:
                padded_scores[batch_idx, :src_len, :dst_len] = filtered_scores[batch_idx]
        
        # Find best matches (greedy selection)
        node_max, node_idx = padded_scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # Ensure we don't exceed available tokens
        actual_r = min(r, min(valid_lengths_src))
        
        unm_idx_rel = edge_idx[..., actual_r:, :]  # Unmerged tokens (relative to filtered src)
        src_idx_rel = edge_idx[..., :actual_r, :]  # Tokens to merge (relative to filtered src)
        dst_idx_rel = gather(node_idx[..., None], dim=-2, index=src_idx_rel)  # Corresponding dst tokens

        def merge(x_full: torch.Tensor, mode="mean") -> torch.Tensor:
            B_full, N_full, C_full = x_full.shape
            result_tokens = []
            
            for batch_idx in range(B_full):
                # Get protected tokens for this batch (unchanged)
                protected_batch = x_full[batch_idx, protected_indices[batch_idx]]
                
                # Get filtered src and dst tokens
                src_indices_batch = filtered_src_indices[batch_idx]
                dst_indices_batch = filtered_dst_indices[batch_idx]
                
                if len(src_indices_batch) == 0 or len(dst_indices_batch) == 0:
                    # No tokens to merge for this batch, return all tokens
                    result_tokens.append(x_full[batch_idx])
                    continue
                
                src_tokens = x_full[batch_idx, src_indices_batch]
                dst_tokens = x_full[batch_idx, dst_indices_batch]
                
                # Apply merging based on computed indices
                valid_src_len = valid_lengths_src[batch_idx]
                
                # Get unmerged src tokens
                unm_rel_batch = unm_idx_rel[batch_idx, :valid_src_len - actual_r, 0]
                unm_tokens = src_tokens[unm_rel_batch]
                
                # Get src tokens to merge and their corresponding dst tokens
                src_rel_batch = src_idx_rel[batch_idx, :actual_r, 0]
                dst_rel_batch = dst_idx_rel[batch_idx, :actual_r, 0]
                
                # Ensure indices are within bounds
                src_rel_batch = src_rel_batch[src_rel_batch < len(src_tokens)]
                dst_rel_batch = dst_rel_batch[dst_rel_batch < len(dst_tokens)]
                
                if len(src_rel_batch) > 0 and len(dst_rel_batch) > 0:
                    merge_src_tokens = src_tokens[src_rel_batch]
                    merge_dst_tokens = dst_tokens[dst_rel_batch]
                    
                    # Perform the merge operation
                    if mode == "mean":
                        merged_tokens = (merge_src_tokens + merge_dst_tokens) / 2
                    elif mode == "mlerp":
                        # Apply MLERP merge - more efficient approach for pairs
                        # Stack src and dst to create cluster input
                        cluster_tokens = torch.stack([merge_dst_tokens, merge_src_tokens], dim=1)  # [N_merge, 2, C]
                        
                        # Compute mean
                        mean_tokens = cluster_tokens.mean(dim=1)  # [N_merge, C]
                        
                        # Compute max norm per cluster  
                        cluster_norms = cluster_tokens.norm(dim=-1)  # [N_merge, 2]
                        max_norms = cluster_norms.max(dim=1)[0]  # [N_merge]
                        
                        # Renormalize
                        mean_norms = mean_tokens.norm(dim=-1).clamp_min(1e-6)  # [N_merge]
                        scale = max_norms / mean_norms  # [N_merge] 
                        merged_tokens = mean_tokens * scale.unsqueeze(-1)  # [N_merge, C]
                    elif mode == "prune":
                        # Pruning: just keep dst tokens, ignore src tokens
                        merged_tokens = merge_dst_tokens
                    else:
                        # Add other merge modes as needed
                        merged_tokens = merge_dst_tokens
                        
                    # Update dst tokens with merged values
                    updated_dst_tokens = dst_tokens.clone()
                    updated_dst_tokens[dst_rel_batch] = merged_tokens
                    
                    # Combine: protected + unmerged_src + updated_dst
                    combined = torch.cat([protected_batch, unm_tokens, updated_dst_tokens], dim=0)
                else:
                    # No valid merging, keep all tokens
                    combined = torch.cat([protected_batch, src_tokens, dst_tokens], dim=0)
                
                result_tokens.append(combined)
            
            return torch.stack(result_tokens, dim=0)
        
        def unmerge(x_merged: torch.Tensor) -> torch.Tensor:
            """
            Proper unmerge that reconstructs the original spatial structure.
            This is more complex but necessary for spatial uniformity preservation.
            """
            B_full, N_merged, C_full = x_merged.shape
            result = torch.zeros(B_full, N, C_full, device=x_merged.device, dtype=x_merged.dtype)
            
            for batch_idx in range(B_full):
                # Handle batch size mismatch (use first batch's structure if needed)
                batch_i = min(batch_idx, len(filtered_src_indices) - 1)
                
                # Place protected tokens back in original positions
                batch_protected = protected_indices[batch_i]
                n_protected = len(batch_protected)
                if n_protected > 0:
                    result[batch_idx, batch_protected] = x_merged[batch_idx, :n_protected]
                
                # Get remaining merged tokens
                remaining_tokens = x_merged[batch_idx, n_protected:]
                
                # Reconstruct unmerged src tokens
                src_indices = filtered_src_indices[batch_i]
                dst_indices = filtered_dst_indices[batch_i]
                
                if len(src_indices) > 0 and len(dst_indices) > 0:
                    # Calculate how many tokens we expect
                    expected_unmerged_src = len(src_indices) - actual_r
                    expected_dst = len(dst_indices)
                    
                    if len(remaining_tokens) >= expected_unmerged_src + expected_dst:
                        # Split remaining tokens
                        unmerged_src_tokens = remaining_tokens[:expected_unmerged_src]
                        final_dst_tokens = remaining_tokens[expected_unmerged_src:expected_unmerged_src + expected_dst]
                        
                        # Place unmerged src tokens back
                        if len(unmerged_src_tokens) > 0:
                            # Get which src indices were unmerged
                            unm_src_indices = src_indices[unm_idx_rel[batch_i, :expected_unmerged_src, 0]]
                            result[batch_idx, unm_src_indices] = unmerged_src_tokens
                        
                        # Place dst tokens back (these contain merged information)
                        if len(final_dst_tokens) > 0:
                            result[batch_idx, dst_indices] = final_dst_tokens
                    else:
                        # Fallback: distribute remaining tokens to available positions
                        available_positions = torch.cat([src_indices, dst_indices])
                        n_available = min(len(remaining_tokens), len(available_positions))
                        if n_available > 0:
                            result[batch_idx, available_positions[:n_available]] = remaining_tokens[:n_available]
                else:
                    # No valid indices, just place tokens in non-protected positions
                    non_protected_mask = ~protected_mask_full[batch_i]
                    non_protected_indices = torch.nonzero(non_protected_mask, as_tuple=True)[0]
                    n_place = min(len(remaining_tokens), len(non_protected_indices))
                    if n_place > 0:
                        result[batch_idx, non_protected_indices[:n_place]] = remaining_tokens[:n_place]
            
            return result
        
        # Initialize token sizes tracking if provided
        if token_sizes is not None:
            merge._token_sizes = token_sizes.clone()
        
        return merge, unmerge


def _extract_indices_from_merge(x: torch.Tensor, w: int, h: int, sx: int, sy: int, r: int, 
                                no_rand: bool, generator: torch.Generator) -> Dict[str, torch.Tensor]:
    """
    Extract indices from the merge computation for caching.
    This duplicates some logic from bipartite_soft_matching_random2d but extracts the indices.
    """
    B, N, _ = x.shape
    
    if r <= 0:
        return {}
    
    gather = mps_gather_workaround if x.device.type == "mps" else torch.gather
    
    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=x.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(x.device)
        
        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=x.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=x.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = x / x.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

        # Store indices on CPU to save GPU memory
        return {
            "a_idx": a_idx.cpu(),
            "b_idx": b_idx.cpu(),
            "unm_idx": unm_idx.cpu(),
            "src_idx": src_idx.cpu(),
            "dst_idx": dst_idx.cpu(),
            "num_dst": num_dst
        }


def _extract_scoring_info_for_cache(
    x: torch.Tensor,
    scorer: 'TokenScorer',
    w: int, h: int, sx: int, sy: int, r: int,
    preserve_ratio: float = 0.3,
    score_mode: str = "high",
    no_rand: bool = False,
    generator: torch.Generator = None,
    timestep_normalized: Optional[float] = None,
    **scorer_kwargs
) -> Dict[str, Any]:
    """
    Extract scoring information for caching purposes.
    This computes all the indices needed for cached scoring.
    """
    from .scoring import TokenScorer
    
    B, N, _ = x.shape
    
    if r <= 0:
        return {}
    
    # Calculate number of tokens to protect
    n_protect = int(N * preserve_ratio)
    n_protect = max(0, min(n_protect, N - r))
    
    # OPTIMIZATION 5: Skip scoring extraction when preserve_ratio == 0
    if n_protect == 0:
        # No tokens to protect - return empty cache (fallback to regular bipartite matching)
        return {}
    
    # Score tokens using the provided scorer
    with torch.no_grad():
        token_scores = scorer.score_tokens(x, H=h, W=w, **scorer_kwargs)
        
        # Select tokens to protect based on score_mode
        protected_indices = _compute_protected_indices(
            token_scores, score_mode, n_protect, timestep_normalized
        )
        
        # Create mask for protected tokens
        protected_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        for i in range(B):
            protected_mask[i, protected_indices[i]] = True
        
        # Get indices of mergeable tokens
        mergeable_mask = ~protected_mask
        mergeable_indices = torch.nonzero(mergeable_mask, as_tuple=False)
        
        # Group mergeable indices by batch
        mergeable_per_batch = []
        for i in range(B):
            batch_mergeable = mergeable_indices[mergeable_indices[:, 0] == i, 1]
            mergeable_per_batch.append(batch_mergeable)
        
        # Check if we have enough mergeable tokens
        min_mergeable = min(len(batch_indices) for batch_indices in mergeable_per_batch)
        if min_mergeable < r:
            # Fall back to regular caching
            return {}
        
        # Create subset tensor with mergeable tokens for bipartite matching
        max_mergeable_tokens = max(len(batch_indices) for batch_indices in mergeable_per_batch)
        
        # Create consistent tensor for all batches
        mergeable_tensor = torch.zeros(B, max_mergeable_tokens, x.shape[-1], device=x.device, dtype=x.dtype)
        valid_lengths = []
        
        for i in range(B):
            batch_mergeable = mergeable_per_batch[i]
            valid_lengths.append(len(batch_mergeable))
            mergeable_tensor[i, :len(batch_mergeable)] = x[i, batch_mergeable]
        
        # Calculate effective dimensions for bipartite matching
        n_mergeable = max_mergeable_tokens
        h_mergeable = int(math.ceil(math.sqrt(n_mergeable)))
        w_mergeable = int(math.ceil(n_mergeable / h_mergeable))
        
        sx_mergeable = min(sx, w_mergeable)
        sy_mergeable = min(sy, h_mergeable)
        
        # Extract bipartite matching indices for the mergeable subset
        cached_bipartite_indices = _extract_indices_from_merge(
            mergeable_tensor, w_mergeable, h_mergeable, sx_mergeable, sy_mergeable, r, 
            no_rand, generator
        )
        
        # Store all information needed for cached scoring
        return {
            "protected_indices": protected_indices.cpu(),
            "mergeable_per_batch": [batch_indices.cpu() for batch_indices in mergeable_per_batch],
            "valid_lengths": valid_lengths,
            "max_mergeable_tokens": max_mergeable_tokens,
            "bipartite_indices": cached_bipartite_indices,
            "h_mergeable": h_mergeable,
            "w_mergeable": w_mergeable,
            "sx_mergeable": sx_mergeable,
            "sy_mergeable": sy_mergeable,
            "n_protect": n_protect,
            "N": N
        }


def bipartite_soft_matching_with_scoring_cached(
    cached_scoring_info: dict,
    w: int, h: int, sx: int, sy: int, r: int,
    device: torch.device
) -> Tuple[Callable, Callable]:
    """
    Creates merge/unmerge functions using pre-computed scoring information.
    
    Args:
        cached_scoring_info: Dict containing pre-computed scoring indices and metadata
        w, h: image dimensions in tokens
        sx, sy: stride values
        r: number of tokens to remove
        device: torch device
        
    Returns:
        Tuple of (merge_function, unmerge_function)
    """
    if r <= 0 or not cached_scoring_info:
        return do_nothing, do_nothing
    
    # Extract cached information (OPTIMIZATION 3: Data already on GPU)
    protected_indices = cached_scoring_info["protected_indices"]
    # Handle both GPU (new) and CPU (legacy) cached data
    if isinstance(cached_scoring_info["mergeable_per_batch"][0], torch.Tensor):
        mergeable_per_batch = cached_scoring_info["mergeable_per_batch"]  # Already on GPU
    else:
        mergeable_per_batch = [batch_indices.to(device) for batch_indices in cached_scoring_info["mergeable_per_batch"]]
    valid_lengths = cached_scoring_info["valid_lengths"]
    max_mergeable_tokens = cached_scoring_info["max_mergeable_tokens"]
    bipartite_indices = cached_scoring_info["bipartite_indices"]
    h_mergeable = cached_scoring_info["h_mergeable"]
    w_mergeable = cached_scoring_info["w_mergeable"]
    N = cached_scoring_info["N"]
    
    # Create bipartite merge/unmerge functions for mergeable subset
    merge_subset, unmerge_subset = bipartite_soft_matching_from_cached_indices(
        bipartite_indices, w_mergeable, h_mergeable, sx, sy, r, device
    )
    
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        B_full, N_full, C_full = x.shape
        
        # Handle batch size mismatch: replicate cached indices for current batch size
        if protected_indices.shape[0] == 1 and B_full > 1:
            # Cached indices are for batch size 1, replicate for current batch
            protected_indices_expanded = protected_indices.expand(B_full, -1)
            mergeable_per_batch_expanded = [mergeable_per_batch[0] for _ in range(B_full)]
            valid_lengths_expanded = [valid_lengths[0] for _ in range(B_full)]
        else:
            # Batch sizes match or cached batch is larger
            protected_indices_expanded = protected_indices[:B_full]
            mergeable_per_batch_expanded = mergeable_per_batch[:B_full]
            valid_lengths_expanded = valid_lengths[:B_full]
        
        # Extract protected tokens
        protected_tokens = []
        for i in range(B_full):
            protected_tokens.append(x[i, protected_indices_expanded[i]])
        
        # Extract mergeable tokens
        mergeable_tensor = torch.zeros(B_full, max_mergeable_tokens, C_full, device=x.device, dtype=x.dtype)
        for i in range(B_full):
            batch_mergeable = mergeable_per_batch_expanded[i]
            valid_len = valid_lengths_expanded[i]
            mergeable_tensor[i, :valid_len] = x[i, batch_mergeable]
        
        # Apply cached bipartite merge to mergeable tokens
        merged_tensor = merge_subset(mergeable_tensor, mode)
        
        # Extract valid merged tokens per batch
        merged_tokens = []
        for i in range(B_full):
            valid_len_after_merge = valid_lengths_expanded[i] - r
            merged_tokens.append(merged_tensor[i, :valid_len_after_merge])
        
        # Combine protected and merged tokens
        result_tokens = []
        for i in range(B_full):
            combined = torch.cat([protected_tokens[i], merged_tokens[i]], dim=0)
            result_tokens.append(combined)
        
        return torch.stack(result_tokens, dim=0)
    
    def unmerge(x_merged: torch.Tensor) -> torch.Tensor:
        B_full, N_merged, C_full = x_merged.shape
        
        # Handle batch size mismatch: replicate cached indices for current batch size
        if protected_indices.shape[0] == 1 and B_full > 1:
            # Cached indices are for batch size 1, replicate for current batch
            protected_indices_expanded = protected_indices.expand(B_full, -1)
            mergeable_per_batch_expanded = [mergeable_per_batch[0] for _ in range(B_full)]
            valid_lengths_expanded = [valid_lengths[0] for _ in range(B_full)]
        else:
            # Batch sizes match or cached batch is larger
            protected_indices_expanded = protected_indices[:B_full]
            mergeable_per_batch_expanded = mergeable_per_batch[:B_full]
            valid_lengths_expanded = valid_lengths[:B_full]
        
        # Split into protected and merged parts
        protected_tokens = []
        merged_tokens = []
        
        for i in range(B_full):
            n_protected = len(protected_indices_expanded[i])
            protected_tokens.append(x_merged[i, :n_protected])
            merged_tokens.append(x_merged[i, n_protected:])
        
        # Reconstruct mergeable tensor for unmerging
        max_merged_tokens = max(len(merged_tokens[i]) for i in range(B_full))
        merged_tensor = torch.zeros(B_full, max_merged_tokens, C_full, device=x_merged.device, dtype=x_merged.dtype)
        for i in range(B_full):
            batch_merged = merged_tokens[i]
            merged_tensor[i, :len(batch_merged)] = batch_merged
        
        # Apply cached bipartite unmerge
        unmerged_tensor = unmerge_subset(merged_tensor)
        
        # Extract valid unmerged tokens per batch
        unmerged_tokens = []
        for i in range(B_full):
            valid_len = valid_lengths_expanded[i]
            unmerged_tokens.append(unmerged_tensor[i, :valid_len])
        
        # Reconstruct original tensor
        result = torch.zeros(B_full, N, C_full, device=x_merged.device, dtype=x_merged.dtype)
        
        for i in range(B_full):
            # Place protected tokens
            result[i, protected_indices_expanded[i]] = protected_tokens[i]
            # Place unmerged tokens
            result[i, mergeable_per_batch_expanded[i]] = unmerged_tokens[i]
        
        return result
    
    return merge, unmerge


def agent_guided_simple_merge(
    x: torch.Tensor,
    scorer: 'TokenScorer', 
    preserve_ratio: float = 0.3,
    score_mode: str = "high",
    w: int = None, h: int = None,
    **scorer_kwargs
) -> Tuple[Callable, Callable]:
    """
    Simple agent-guided merging: select top-k tokens, merge the rest into ONE token.
    
    This is the NEW SIMPLIFIED approach:
    1. Score all N tokens using cross-attention with agents
    2. Select top-k tokens based on highest scores (k = N * preserve_ratio)  
    3. Keep selected tokens unchanged
    4. Merge ALL remaining (N-k) tokens into ONE single merged token
    5. Output: k selected + 1 merged = k+1 tokens total
    
    Args:
        x: Input tensor (B, N, C)
        scorer: Agent-guided scorer  
        preserve_ratio: Ratio of tokens to keep unchanged (0.0 to 1.0)
        score_mode: Selection mode ("high" for top scores)
        w, h: Image dimensions (for scorer)
        
    Returns:
        Tuple of (merge_function, unmerge_function)
    """
    from .scoring import TokenScorer
    
    B, N, C = x.shape
    
    # Calculate number of tokens to keep
    n_keep = int(N * preserve_ratio)
    n_keep = max(1, min(n_keep, N-1))  # At least 1 token to keep, at least 1 to merge
    n_merge = N - n_keep
    
    # Score tokens using agent-guided scorer
    with torch.no_grad():
        token_scores = scorer.score_tokens(x, H=h, W=w, **scorer_kwargs)  # (B, N)
        
        # Select top-k tokens based on scores
        if score_mode == "high":
            # Keep tokens with highest scores
            _, keep_indices = torch.topk(token_scores, n_keep, dim=1, largest=True)
        else:
            # Keep tokens with lowest scores  
            _, keep_indices = torch.topk(token_scores, n_keep, dim=1, largest=False)
        
        # Sort indices for consistent ordering
        keep_indices = torch.sort(keep_indices, dim=1)[0]  # (B, n_keep)
        
        # Create merge indices (all tokens NOT in keep_indices)
        all_indices = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)  # (B, N)
        
        # Create mask for tokens to merge
        merge_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        merge_mask.scatter_(1, keep_indices, False)  # Set kept tokens to False
        
        # Get indices of tokens to merge
        merge_indices_list = []
        for b in range(B):
            merge_idx = torch.nonzero(merge_mask[b], as_tuple=False).squeeze(-1)
            merge_indices_list.append(merge_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        """
        Merge function: keep selected tokens, merge the rest into one token
        """
        B, N, C = x.shape
        
        # Gather kept tokens
        kept_tokens = torch.gather(x, 1, keep_indices.unsqueeze(-1).expand(-1, -1, C))  # (B, n_keep, C)
        
        # Gather tokens to merge and merge them into one token per batch
        merged_tokens = []
        for b in range(B):
            merge_idx = merge_indices_list[b]
            if len(merge_idx) > 0:
                tokens_to_merge = x[b, merge_idx, :]  # (n_merge, C)
                if mode == "mean":
                    merged_token = tokens_to_merge.mean(dim=0, keepdim=True)  # (1, C)
                else:
                    # Can add other merge modes here
                    merged_token = tokens_to_merge.mean(dim=0, keepdim=True)
                merged_tokens.append(merged_token)
            else:
                # Edge case: no tokens to merge
                merged_tokens.append(torch.zeros(1, C, device=x.device, dtype=x.dtype))
        
        # Stack merged tokens
        merged_tokens = torch.stack([mt.squeeze(0) for mt in merged_tokens], dim=0)  # (B, C)
        merged_tokens = merged_tokens.unsqueeze(1)  # (B, 1, C)
        
        # Concatenate kept tokens + merged token
        result = torch.cat([kept_tokens, merged_tokens], dim=1)  # (B, n_keep + 1, C)
        
        return result

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """
        Unmerge function: restore original token count by expanding the merged token
        """
        B, reduced_N, C = x.shape  # reduced_N should be n_keep + 1
        
        # Split into kept tokens and merged token
        kept_tokens = x[:, :n_keep, :]  # (B, n_keep, C)
        merged_token = x[:, -1:, :]     # (B, 1, C)
        
        # Expand merged token to original merge positions
        result = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)
        
        # Place kept tokens back at their original positions
        result.scatter_(1, keep_indices.unsqueeze(-1).expand(-1, -1, C), kept_tokens)
        
        # Place merged token at all merge positions
        for b in range(B):
            merge_idx = merge_indices_list[b]
            if len(merge_idx) > 0:
                result[b, merge_idx, :] = merged_token[b, 0, :]
        
        return result

    return merge, unmerge


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, Callable, Callable, Callable, Callable, Callable]:
    """
    Compute merge and unmerge functions for attention, cross-attention, and MLP layers.
    
    Args:
        x: Input tensor with shape [B, N, C]
        tome_info: Dictionary containing merge configuration and state
        
    Returns:
        Tuple of (m_a, m_c, m_m, u_a, u_c, u_m) where:
        - m_a, u_a: merge/unmerge functions for attention
        - m_c, u_c: merge/unmerge functions for cross-attention  
        - m_m, u_m: merge/unmerge functions for MLP
    """
    from .utils import init_generator
    
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1]))) if x.shape[1] > 0 else 1

    args = tome_info["args"]
    
    # Initialize merge and unmerge functions to do_nothing by default
    m, u = do_nothing, do_nothing
    
    # Check if we should apply merging based on downsample level
    if downsample <= args.get("max_downsample", 1):
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args.get("ratio", 0.5))
        
        if r > 0:
            # Re-init the generator if it hasn't already been initialized or device has changed
            if args.get("generator") is None:
                args["generator"] = init_generator(x.device)
            elif args["generator"].device != x.device:
                args["generator"] = init_generator(x.device, fallback=args["generator"])
            
            # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
            # batch, which causes artifacts with use_rand, so force it to be off.
            use_rand = False if x.shape[0] % 2 == 1 else args.get("use_rand", True)
            
            # Check if scoring-based merging is enabled
            if args.get("use_scoring", False) and args.get("scorer") is not None:
                # Use scoring-based merging
                scorer = args["scorer"]
                preserve_ratio = args.get("preserve_ratio", 0.3)
                score_mode = args.get("score_mode", "high")
                preserve_spatial_uniformity = args.get("preserve_spatial_uniformity", False)
                if_low_frequency_dst_tokens = args.get("if_low_frequency_dst_tokens", False)
                scorer_kwargs = args.get("scorer_kwargs", {})
                
                m, u = bipartite_soft_matching_with_scoring(
                    x, scorer, w, h,
                    args.get("sx", 2), args.get("sy", 2), r,
                    preserve_ratio=preserve_ratio,
                    score_mode=score_mode,
                    preserve_spatial_uniformity=preserve_spatial_uniformity,
                    if_low_frequency_dst_tokens=if_low_frequency_dst_tokens,
                    no_rand=not use_rand,
                    generator=args["generator"],
                    cache_resolution_merge=args.get("cache_resolution_merge", False),
                    **scorer_kwargs
                )
            else:
                # Use regular matching method (bipartite or ABP)
                m, u = _choose_merge_method(
                    x, w, h, 
                    args.get("sx", 2), args.get("sy", 2), r,
                    no_rand=not use_rand, 
                    generator=args["generator"],
                    token_sizes=None,
                    token_scores=None,
                    merge_method=args.get("merge_method", "bipartite"),
                    scorer=args.get("abp_scorer"),
                    tile_aggregation=args.get("abp_tile_aggregation", "max"),
                    locality_block_factor_h=args.get("locality_block_factor_h", 1),
                    locality_block_factor_w=args.get("locality_block_factor_w", 1),
                    use_compile=args.get("use_compile", False)
                )
    
    # Create separate merge/unmerge functions based on configuration
    m_a, u_a = (m, u) if args.get("merge_attn", True) else (do_nothing, do_nothing)
    m_c, u_c = (m, u) if args.get("merge_crossattn", False) else (do_nothing, do_nothing)
    m_m, u_m = (m, u) if args.get("merge_mlp", False) else (do_nothing, do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m
