import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
import math
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
import torch.nn.functional as F

from . import merge
from .utils import init_generator


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    """
    Compute merge and unmerge functions for DiT models.
    This is a copy of the working function from the original patch.py.
    """
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]
    
    # Create a cache key based on the current layer dimensions and downsample level
    cache_key = f"downsample_{downsample}_tokens_{x.shape[1]}"

    if args.get("single_downsample_level_merge", False) == False:
        if downsample <= args.get("max_downsample", 1):
            w = int(math.ceil(original_w / downsample))
            h = int(math.ceil(original_h / downsample))
            r = int(x.shape[1] * args.get("ratio", 0.5))
    
            # Re-init the generator if it hasn't already been initialized or device has changed.
            if args.get("generator") is None:
                args["generator"] = init_generator(x.device)
            elif args["generator"].device != x.device:
                args["generator"] = init_generator(x.device, fallback=args["generator"])
            
            # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
            # batch, which causes artifacts with use_rand, so force it to be off.
            use_rand = False if x.shape[0] % 2 == 1 else args.get("use_rand", True)
            
            # Initialize token sizes for this layer if proportional attention is enabled
            token_sizes = None
            if args.get("if_proportional_attention", False):
                # Proportional attention only makes sense when using cached indices
                # because it relies on consistent merging patterns across layers
                if args.get("cache_indices_per_image", False):
                    if tome_info.get("token_sizes") is not None:
                        # Use existing token sizes from previous layers, but check batch size compatibility
                        cached_token_sizes = tome_info["token_sizes"]
                        B, N, _ = x.shape
                        
                        # Check if cached token_sizes matches current batch size and token count
                        if (cached_token_sizes.shape[0] == B and 
                            cached_token_sizes.shape[1] == N):
                            token_sizes = cached_token_sizes
                        else:
                            # Batch size or token count mismatch, reinitialize
                            token_sizes = torch.ones(B, N, device=x.device, dtype=torch.float32)
                            tome_info["token_sizes"] = token_sizes
                    else:
                        # Initialize token sizes (all tokens start with size 1)
                        B, N, _ = x.shape
                        token_sizes = torch.ones(B, N, device=x.device, dtype=torch.float32)
                        tome_info["token_sizes"] = token_sizes
                else:
                    # When cache_indices_per_image == false, disable proportional attention
                    # because each layer makes independent merging decisions
                    token_sizes = None
            
            # Check if scoring is enabled
            if args.get("use_scoring", False):
                # Check if caching is enabled for scoring and if we have cached scoring info
                if args.get("cache_indices_per_image", False) and cache_key in tome_info.get("cached_indices", {}):
                    # Use cached scoring information
                    m, u = merge.bipartite_soft_matching_with_scoring_cached(
                        tome_info["cached_indices"][cache_key], w, h, args.get("sx", 2), args.get("sy", 2), r, x.device
                    )
                else:
                    # Use score-based merging (supports both bipartite and ABP through merge_method)
                    m, u = merge.bipartite_soft_matching_with_scoring(
                        x, args["scorer"], w, h, args.get("sx", 2), args.get("sy", 2), r,
                        preserve_ratio=args.get("preserve_ratio", 0.3),
                        score_mode=args.get("score_mode", "high"),
                        preserve_spatial_uniformity=args.get("preserve_spatial_uniformity", False),
                        if_low_frequency_dst_tokens=args.get("if_low_frequency_dst_tokens", False),
                        no_rand=not use_rand,
                        generator=args["generator"],
                        token_sizes=token_sizes,
                        merge_method=args.get("merge_method", "bipartite"),  # NEW: ABP support
                        abp_scorer=args.get("abp_scorer"),  # NEW: ABP scorer
                        abp_tile_aggregation=args.get("abp_tile_aggregation", "max"),  # NEW: ABP aggregation
                        locality_block_factor_h=args.get("locality_block_factor_h", 1),
                        locality_block_factor_w=args.get("locality_block_factor_w", 1),
                        **args.get("scorer_kwargs", {})
                    )
                    
                    # If caching is enabled, store the scoring information for future use
                    if args.get("cache_indices_per_image", False):
                        try:
                            cached_scoring_info = merge._extract_scoring_info_for_cache(
                                x, args["scorer"], w, h, args.get("sx", 2), args.get("sy", 2), r,
                                preserve_ratio=args.get("preserve_ratio", 0.3),
                                score_mode=args.get("score_mode", "high"),
                                no_rand=not use_rand,
                                generator=args["generator"],
                                **args.get("scorer_kwargs", {})
                            )
                            if cached_scoring_info:  # Only cache if extraction was successful
                                if "cached_indices" not in tome_info:
                                    tome_info["cached_indices"] = {}
                                tome_info["cached_indices"][cache_key] = cached_scoring_info
                        except AttributeError:
                            # Function doesn't exist, skip caching
                            pass
            else:
                # Check if caching is enabled and if we have cached indices for this layer
                if args.get("cache_indices_per_image", False) and cache_key in tome_info.get("cached_indices", {}):
                    # Use cached indices
                    m, u = merge.bipartite_soft_matching_from_cached_indices(
                        tome_info["cached_indices"][cache_key], w, h, args.get("sx", 2), args.get("sy", 2), r, x.device
                    )
                else:
                    # Compute new indices using unified merge method selection (supports both bipartite and ABP)
                    m, u = merge._choose_merge_method(
                        x, w, h, args.get("sx", 2), args.get("sy", 2), r,
                        no_rand=not use_rand,
                        generator=args["generator"],
                        token_sizes=token_sizes,
                        token_scores=None,
                        merge_method=args.get("merge_method", "bipartite"),
                        scorer=args.get("abp_scorer"),
                        tile_aggregation=args.get("abp_tile_aggregation", "max"),
                        locality_block_factor_h=args.get("locality_block_factor_h", 1),
                        locality_block_factor_w=args.get("locality_block_factor_w", 1)
                    )
                    
                    # If caching is enabled, store the indices for future use
                    # This currently works for bipartite, ABP caching would need updates
                    if args.get("cache_indices_per_image", False) and args.get("merge_method", "bipartite") == "bipartite":
                        try:
                            # We need to extract the indices from the merge function
                            cached_indices = merge._extract_indices_from_merge(x, w, h, args.get("sx", 2), args.get("sy", 2), r, 
                                                                             not use_rand, args["generator"])
                            if "cached_indices" not in tome_info:
                                tome_info["cached_indices"] = {}
                            tome_info["cached_indices"][cache_key] = cached_indices
                        except AttributeError:
                            # Function doesn't exist, skip caching
                            pass
        else:
            m, u = (merge.do_nothing, merge.do_nothing)
    else:
        # Single downsample level merge logic
        if downsample == args.get("max_downsample", 1):
            w = int(math.ceil(original_w / downsample))
            h = int(math.ceil(original_h / downsample))
            r = int(x.shape[1] * args.get("ratio", 0.5))
    
            # Re-init the generator if it hasn't already been initialized or device has changed.
            if args.get("generator") is None:
                args["generator"] = init_generator(x.device)
            elif args["generator"].device != x.device:
                args["generator"] = init_generator(x.device, fallback=args["generator"])
            
            use_rand = False if x.shape[0] % 2 == 1 else args.get("use_rand", True)
            
            # Similar logic as above for single downsample level
            if args.get("use_scoring", False):
                m, u = merge.bipartite_soft_matching_with_scoring(
                    x, args["scorer"], w, h, args.get("sx", 2), args.get("sy", 2), r,
                    preserve_ratio=args.get("preserve_ratio", 0.3),
                    score_mode=args.get("score_mode", "high"),
                    preserve_spatial_uniformity=args.get("preserve_spatial_uniformity", False),
                    if_low_frequency_dst_tokens=args.get("if_low_frequency_dst_tokens", False),
                    no_rand=not use_rand,
                    generator=args["generator"],
                    merge_method=args.get("merge_method", "bipartite"),  # NEW: ABP support
                    abp_scorer=args.get("abp_scorer"),  # NEW: ABP scorer
                    abp_tile_aggregation=args.get("abp_tile_aggregation", "max"),  # NEW: ABP aggregation
                    locality_block_factor_h=args.get("locality_block_factor_h", 1),
                    locality_block_factor_w=args.get("locality_block_factor_w", 1),
                    **args.get("scorer_kwargs", {})
                )
            else:
                # Use unified merge method selection (supports both bipartite and ABP)
                m, u = merge._choose_merge_method(
                    x, w, h, args.get("sx", 2), args.get("sy", 2), r,
                    no_rand=not use_rand,
                    generator=args["generator"],
                    token_sizes=None,
                    token_scores=None,
                    merge_method=args.get("merge_method", "bipartite"),
                    scorer=args.get("abp_scorer"),
                    tile_aggregation=args.get("abp_tile_aggregation", "max"),
                    locality_block_factor_h=args.get("locality_block_factor_h", 1),
                    locality_block_factor_w=args.get("locality_block_factor_w", 1)
                )
        else:
            m, u = (merge.do_nothing, merge.do_nothing)

    # Create wrapper functions that use the specified method
    method = args.get("method", "mean")
    
    def make_merge_with_method(merge_fn):
        if merge_fn == merge.do_nothing:
            return merge_fn
        return lambda x: merge_fn(x, mode=method)
    
    m_a_base, u_a = (m, u) if args.get("merge_attn", True) else (merge.do_nothing, merge.do_nothing)
    m_c_base, u_c = (m, u) if args.get("merge_crossattn", False) else (merge.do_nothing, merge.do_nothing)  
    m_m_base, u_m = (m, u) if args.get("merge_mlp", False) else (merge.do_nothing, merge.do_nothing)
    
    # Wrap merge functions to use the specified method
    m_a = make_merge_with_method(m_a_base)
    m_c = make_merge_with_method(m_c_base) 
    m_m = make_merge_with_method(m_m_base)

    return m_a, m_c, m_m, u_a, u_c, u_m


def reset_cache_for_new_call(tome_info: Dict):
    """Reset cache when starting a new UNet call"""
    cache_info = tome_info.get("block_cache", {})
    if cache_info.get("enabled", False):
        if tome_info.get("debug_cache", False):
            print(f"Resetting block cache for new UNet call")
        cache_info["cached_functions"].clear()
        cache_info["last_recalc_block"] = -1
        cache_info["current_block_idx"] = 0


def should_recalculate_cache(block_idx: int, tome_info: Dict) -> bool:
    """Check if we should recalculate merge functions for this block"""
    cache_info = tome_info.get("block_cache", {})
    if not cache_info.get("enabled", False):
        return True
    
    interval = cache_info.get("recalc_interval", 4)
    last_recalc = cache_info.get("last_recalc_block", -1)
    
    # Always recalculate if this is the first block or interval reached
    return last_recalc == -1 or (block_idx - last_recalc) >= interval


def cache_block_functions(block_idx: int, merge_fn: Callable, unmerge_fn: Callable, tome_info: Dict):
    """Cache merge/unmerge functions for current interval"""
    cache_info = tome_info.get("block_cache", {})
    if cache_info.get("enabled", False):
        # Store with "current" key so all blocks in interval can access it
        cache_info["cached_functions"]["current"] = (merge_fn, unmerge_fn)
        cache_info["last_recalc_block"] = block_idx


def get_cached_functions(block_idx: int, tome_info: Dict) -> Optional[Tuple[Callable, Callable]]:
    """Get cached merge/unmerge functions for current interval"""
    cache_info = tome_info.get("block_cache", {})
    # All blocks in same interval use the same "current" cached functions
    return cache_info.get("cached_functions", {}).get("current")


def compute_merge_with_cache(x: torch.Tensor, tome_info: Dict[str, Any], block_idx: int) -> Tuple[Callable, ...]:
    """Compute merge with optional block-level caching"""
    
    # Fast path: if caching is disabled, skip all cache logic
    cache_info = tome_info.get("block_cache", {})
    if not cache_info.get("enabled", False):
        return compute_merge(x, tome_info)
    
    # Check if we should use cached functions (only if caching is enabled)
    if not should_recalculate_cache(block_idx, tome_info):
        cached = get_cached_functions(block_idx, tome_info)
        if cached is not None:
            merge_fn, unmerge_fn = cached
            if tome_info.get("debug_cache", False):
                print(f"Cache HIT for block {block_idx}")
            # Return cached functions in the expected format
            return merge_fn, merge_fn, merge_fn, unmerge_fn, unmerge_fn, unmerge_fn
    
    # Compute new merge functions (use existing compute_merge logic)
    if tome_info.get("debug_cache", False):
        print(f"Cache MISS - computing for block {block_idx}")
    m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, tome_info)
    
    # Cache the functions if this is a recalc block
    if should_recalculate_cache(block_idx, tome_info):
        cache_block_functions(block_idx, m_a, u_a, tome_info)
        if tome_info.get("debug_cache", False):
            print(f"Cached functions for block {block_idx}")
    
    return m_a, m_c, m_m, u_a, u_c, u_m


def isinstance_str(x, cls_name):
    """
    Check if x is instance of class_name without importing it
    """
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def hook_tome_model(model: torch.nn.Module):
    """Add hook to capture input size information and reset cache at start of each UNet call"""
    def hook(module, args):
        # Reset cache at the start of each UNet forward call
        if hasattr(module, '_tome_info'):
            reset_cache_for_new_call(module._tome_info)
        
        # Store size information
        if len(args) > 0:
            x = args[0]
            if hasattr(x, 'shape'):
                if len(x.shape) == 4:  # B, C, H, W (original image/latent input)
                    module._tome_info["size"] = (x.shape[2], x.shape[3])
                elif len(x.shape) == 3:  # B, N, C (after patchification)
                    # For transformer inputs, try to infer spatial dimensions
                    batch_size, num_tokens, _ = x.shape
                    spatial_size = int(math.sqrt(num_tokens))
                    if spatial_size * spatial_size == num_tokens:
                        # Try to get patch size from model attributes
                        patch_size = 1  # Default
                        if hasattr(module, 'patch_size'):
                            patch_size = module.patch_size
                        elif hasattr(module, 'transformer') and hasattr(module.transformer, 'patch_size'):
                            patch_size = module.transformer.patch_size
                        elif hasattr(module, 'x_embedder') and hasattr(module.x_embedder, 'patch_size'):
                            patch_size = module.x_embedder.patch_size[0]  # DiT patch embedder
                        else:
                            # Fallback: assume standard DiT patch sizes
                            patch_size = 8
                        
                        module._tome_info["size"] = (spatial_size * patch_size, spatial_size * patch_size)
                    else:
                        # Fallback for non-square token arrangements
                        estimated_size = int(math.sqrt(num_tokens)) * 8  # Rough estimate
                        module._tome_info["size"] = (estimated_size, estimated_size)
    
    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


class ToMeBlock(nn.Module):
    """
    A wrapper class that applies ToMe to transformer blocks selectively
    """
    
    def __init__(self, original_block, tome_info, block_enabled=True):
        super().__init__()
        self.original_block = original_block
        self.tome_info = tome_info
        self.block_enabled = block_enabled
        self._tome_info = tome_info
        
        # Copy all attributes from the original block
        for name, attr in vars(original_block).items():
            if not name.startswith('_'):
                setattr(self, name, attr)
    
    def forward(self, *args, **kwargs):
        if not self.block_enabled:
            # If this block is disabled, use original forward pass
            return self.original_block(*args, **kwargs)
        
        # Apply ToMe logic for enabled blocks
        hidden_states = args[0] if args else kwargs.get('hidden_states')
        
        # Compute merge functions
        m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self.tome_info)
        
        # Apply forward pass with ToMe operations
        # This is a simplified version - would need to be adapted based on the specific transformer block
        # For now, delegate to original block but with tome_info available
        return self.original_block(*args, **kwargs)


class DiTBlockLevelToMeProcessor:
    """Custom attention processor for DiT that supports block-level ToMe control"""
    
    def __init__(self, tome_info=None, block_enabled=True, block_idx=0):
        self.tome_info = tome_info
        self.block_enabled = block_enabled
        self.block_idx = block_idx
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        
        if not self.block_enabled or self.tome_info is None:
            # Fallback to standard attention if block is disabled or no tome_info
            return AttnProcessor2_0()(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            
        residual = hidden_states
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Compute merge functions based on current state (with caching support)
        m_a, m_c, m_m, u_a, u_c, u_m = compute_merge_with_cache(hidden_states, self.tome_info, self.block_idx)
        
        # Apply merging strategy based on configuration
        merge_attn = self.tome_info["args"].get("merge_attn", True)
        merge_crossattn = self.tome_info["args"].get("merge_crossattn", False)
        merge_mlp = self.tome_info["args"].get("merge_mlp", False)
        
        if merge_attn:
            hidden_states = m_a(hidden_states)
        
        # Standard attention computation
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif merge_crossattn and attn.norm_cross is not None:
            encoder_hidden_states = m_c(encoder_hidden_states)
            
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Reshape for attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Use standard attention instead of scaled_dot_product_attention to disable attention accelerators
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        
        # Manual attention computation to avoid any acceleration
        scale = 1 / math.sqrt(head_dim)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_probs, value)
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        # Unmerge if needed
        if merge_attn:
            hidden_states = u_a(hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


def apply_patch(
    model: torch.nn.Module,
    ratio: float = 0.5,
    max_downsample: int = 1,
    sx: int = 2, sy: int = 2,
    use_rand: bool = True,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False,
    single_downsample_level_merge: bool = False,
    cache_indices_per_image: bool = False,
    method: str = 'mean',
    if_proportional_attention: bool = False,
    # Scoring parameters
    use_scoring: bool = False,
    scorer: Optional[Any] = None,
    preserve_ratio: float = 0.3,
    score_mode: str = 'high',
    preserve_spatial_uniformity: bool = False,
    if_low_frequency_dst_tokens: bool = False,
    scorer_kwargs: Optional[Dict] = None,
    # Block-level control parameters
    block_tome_flags: Optional[List[int]] = None,
    # Block-level caching parameters  
    cache_merge_functions: bool = False,
    cache_recalc_interval: int = 4,
    debug_cache: bool = False,
    # Locality-based similarity parameters
    locality_block_factor_h: int = 1,
    locality_block_factor_w: int = 1,
    # ABP (Adaptive Block Pooling) parameters
    merge_method: str = "bipartite",  # "bipartite" or "abp"
    abp_scorer: Optional[Any] = None,  # TokenScorer for ABP tile evaluation
    abp_tile_aggregation: str = "max",  # "max", "min", "sum", "std"
    **kwargs
):
    """
    Apply ToMe patch to a model with optional block-level control for DiT models.
    
    Args:
        model: The model to patch (DiT transformer or diffusion pipeline)
        ratio: Token merging ratio (0.0 to 1.0)
        max_downsample: Maximum downsample level to apply ToMe
        sx, sy: Spatial merging strides
        use_rand: Whether to use randomization
        merge_attn: Whether to merge attention tokens
        merge_crossattn: Whether to merge cross-attention tokens  
        merge_mlp: Whether to merge MLP tokens
        single_downsample_level_merge: Whether to use single level merging
        cache_indices_per_image: Whether to cache indices per image
        method: Merging method ('mean', 'mlerp', 'prune')
        if_proportional_attention: Whether to use proportional attention
        use_scoring: Whether to use scoring-based merging
        scorer: Scorer object for scoring-based merging
        preserve_ratio: Ratio of tokens to preserve in scoring
        score_mode: Scoring mode ('high', 'low', etc.)
        preserve_spatial_uniformity: Whether to preserve spatial uniformity
        if_low_frequency_dst_tokens: Whether to use low frequency destination tokens
        scorer_kwargs: Additional scorer arguments
        block_tome_flags: List of 0/1 flags indicating which blocks to apply ToMe to
                         (only for DiT models). Length should match number of transformer blocks.
        cache_merge_functions: Whether to cache merge/unmerge functions for blocks within a UNet call
        cache_recalc_interval: Recalculate cached functions every N blocks (default: 4)
        debug_cache: Print debug information about cache hits/misses
        locality_block_factor_h: Factor to divide height for locality-based similarity computation.
                                Default: 1 (global similarity). Values > 1 enable spatial locality.
        locality_block_factor_w: Factor to divide width for locality-based similarity computation.
                                Default: 1 (global similarity). Values > 1 enable spatial locality.
        merge_method: Merging algorithm ("bipartite" or "abp"). Default: "bipartite"
                     - "bipartite": Standard bipartite matching (original ToMe)
                     - "abp": Adaptive Block Pooling (tile-based merging for better spatial structure)
        abp_scorer: TokenScorer instance for ABP tile evaluation. If None with ABP, uses SpatialFilterScorer.
                   Only used when merge_method="abp". Should implement score_tokens(x, H, W) method.
        abp_tile_aggregation: How to aggregate token scores within tiles for ABP evaluation.
                             Options: "max", "min", "sum", "std". Default: "max"
                             Only used when merge_method="abp".
    """
    
    # Remove any existing patches first
    remove_patch(model)
    
    # Check if this is a DiT model and block-level control is requested
    is_dit_model = isinstance_str(model, "DiTTransformer2DModel")
    use_block_control = block_tome_flags is not None and is_dit_model
    
    if use_block_control:
        # Apply block-level control for DiT models
        _apply_dit_block_level_patch(
            model, ratio, max_downsample, sx, sy, use_rand, merge_attn, merge_crossattn,
            merge_mlp, single_downsample_level_merge, cache_indices_per_image, method,
            if_proportional_attention, use_scoring, scorer, preserve_ratio, score_mode,
            preserve_spatial_uniformity, if_low_frequency_dst_tokens, scorer_kwargs,
            block_tome_flags, cache_merge_functions, cache_recalc_interval, debug_cache,
            locality_block_factor_h, locality_block_factor_w, merge_method, abp_scorer, abp_tile_aggregation
        )
    else:
        # Apply standard ToMe patch for all blocks
        _apply_standard_patch(
            model, ratio, max_downsample, sx, sy, use_rand, merge_attn, merge_crossattn,
            merge_mlp, single_downsample_level_merge, cache_indices_per_image, method,
            if_proportional_attention, use_scoring, scorer, preserve_ratio, score_mode,
            preserve_spatial_uniformity, if_low_frequency_dst_tokens, scorer_kwargs,
            cache_merge_functions, cache_recalc_interval, debug_cache,
            locality_block_factor_h, locality_block_factor_w, merge_method, abp_scorer, abp_tile_aggregation
        )
    
    return model


def _apply_dit_block_level_patch(
    model: torch.nn.Module,
    ratio: float,
    max_downsample: int,
    sx: int, sy: int,
    use_rand: bool,
    merge_attn: bool,
    merge_crossattn: bool,
    merge_mlp: bool,
    single_downsample_level_merge: bool,
    cache_indices_per_image: bool,
    method: str,
    if_proportional_attention: bool,
    use_scoring: bool,
    scorer: Optional[Any],
    preserve_ratio: float,
    score_mode: str,
    preserve_spatial_uniformity: bool,
    if_low_frequency_dst_tokens: bool,
    scorer_kwargs: Optional[Dict],
    block_tome_flags: List[int],
    cache_merge_functions: bool,
    cache_recalc_interval: int,
    debug_cache: bool,
    locality_block_factor_h: int = 1,
    locality_block_factor_w: int = 1,
    merge_method: str = "bipartite",
    abp_scorer: Optional[Any] = None,
    abp_tile_aggregation: str = "max"
):
    """Apply ToMe patch with block-level control for DiT models"""
    
    # Initialize ToMe info
    model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "single_downsample_level_merge": single_downsample_level_merge,
            "cache_indices_per_image": cache_indices_per_image,
            "method": method,
            "if_proportional_attention": if_proportional_attention,
            # Scoring parameters
            "use_scoring": use_scoring,
            "scorer": scorer,
            "preserve_ratio": preserve_ratio,
            "score_mode": score_mode,
            "preserve_spatial_uniformity": preserve_spatial_uniformity,
            "if_low_frequency_dst_tokens": if_low_frequency_dst_tokens,
            "scorer_kwargs": scorer_kwargs or {},
            # Locality-based similarity parameters
            "locality_block_factor_h": locality_block_factor_h,
            "locality_block_factor_w": locality_block_factor_w,
            # ABP parameters
            "merge_method": merge_method,
            "abp_scorer": abp_scorer,
            "abp_tile_aggregation": abp_tile_aggregation
        },
        "cached_indices": {},  # Store cached indices per image
        "current_image_latent": None,  # Store current image latent for comparison
        "image_counter": 0,  # Keep track of image changes
        "token_sizes": None,  # Track token sizes for proportional attention
        # Block-level cache
        "block_cache": {
            "enabled": cache_merge_functions,
            "recalc_interval": cache_recalc_interval,
            "cached_functions": {},
            "last_recalc_block": -1,
            "current_block_idx": 0
        },
        "debug_cache": debug_cache
    }
    
    # Add hook for size capture
    hook_tome_model(model)
    
    # Apply block-level patching
    transformer_blocks = []
    
    # Find all transformer blocks (DiT uses 'attn' instead of 'attn1')
    for name, module in model.named_modules():
        if "transformer_blocks" in name and (hasattr(module, 'attn') or hasattr(module, 'attn1')):
            transformer_blocks.append((name, module))
    
    # Validate block flags
    if len(block_tome_flags) != len(transformer_blocks):
        print(f"Warning: Expected {len(transformer_blocks)} block flags, got {len(block_tome_flags)}. "
              f"Adjusting to match.")
        # Pad or truncate as needed
        if len(block_tome_flags) < len(transformer_blocks):
            block_tome_flags.extend([1] * (len(transformer_blocks) - len(block_tome_flags)))
        else:
            block_tome_flags = block_tome_flags[:len(transformer_blocks)]
    
    # Apply ToMe to specified blocks only
    enabled_blocks = 0
    for block_idx, ((name, block_module), should_enable) in enumerate(zip(transformer_blocks, block_tome_flags)):
        block_enabled = bool(should_enable)
        
        if block_enabled:
            enabled_blocks += 1
            # Apply ToMe to attention modules in this block
            for attn_name, attn_module in block_module.named_modules():
                if isinstance(attn_module, Attention):
                    processor = DiTBlockLevelToMeProcessor(
                        tome_info=model._tome_info, 
                        block_enabled=True,
                        block_idx=block_idx
                    )
                    attn_module.set_processor(processor)
                    attn_module._tome_enabled = True
            
            # Mark the block as having ToMe applied
            block_module._tome_enabled = True
        else:
            # Keep default processors for this block
            block_module._tome_enabled = False
        
        # Store block information for debugging
        block_module._tome_block_index = block_idx
    
    print(f"Applied ToMe to {enabled_blocks}/{len(transformer_blocks)} transformer blocks using block-level control")


def _apply_standard_patch(
    model: torch.nn.Module,
    ratio: float,
    max_downsample: int,
    sx: int, sy: int,
    use_rand: bool,
    merge_attn: bool,
    merge_crossattn: bool,
    merge_mlp: bool,
    single_downsample_level_merge: bool,
    cache_indices_per_image: bool,
    method: str,
    if_proportional_attention: bool,
    use_scoring: bool,
    scorer: Optional[Any],
    preserve_ratio: float,
    score_mode: str,
    preserve_spatial_uniformity: bool,
    if_low_frequency_dst_tokens: bool,
    scorer_kwargs: Optional[Dict],
    cache_merge_functions: bool,
    cache_recalc_interval: int,
    debug_cache: bool,
    locality_block_factor_h: int = 1,
    locality_block_factor_w: int = 1,
    merge_method: str = "bipartite",
    abp_scorer: Optional[Any] = None,
    abp_tile_aggregation: str = "max"
):
    """Apply standard ToMe patch to all blocks (original behavior)"""
    
    # Determine if this is a diffusers model
    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    
    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            raise RuntimeError("Provided model was not a supported diffusion model.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet", "pipe.transformer", and direct model
        if hasattr(model, "unet"):
            diffusion_model = model.unet
        elif hasattr(model, "transformer"):
            diffusion_model = model.transformer
        else:
            diffusion_model = model
    
    # Initialize ToMe info
    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "single_downsample_level_merge": single_downsample_level_merge,
            "cache_indices_per_image": cache_indices_per_image,
            "method": method,
            "if_proportional_attention": if_proportional_attention,
            # Scoring parameters
            "use_scoring": use_scoring,
            "scorer": scorer,
            "preserve_ratio": preserve_ratio,
            "score_mode": score_mode,
            "preserve_spatial_uniformity": preserve_spatial_uniformity,
            "if_low_frequency_dst_tokens": if_low_frequency_dst_tokens,
            "scorer_kwargs": scorer_kwargs or {},
            # Locality-based similarity parameters
            "locality_block_factor_h": locality_block_factor_h,
            "locality_block_factor_w": locality_block_factor_w,
            # ABP parameters
            "merge_method": merge_method,
            "abp_scorer": abp_scorer,
            "abp_tile_aggregation": abp_tile_aggregation
        },
        "cached_indices": {},  # Store cached indices per image
        "current_image_latent": None,  # Store current image latent for comparison
        "image_counter": 0,  # Keep track of image changes
        "token_sizes": None,  # Track token sizes for proportional attention
        # Block-level cache (not used in standard patch, but keep for consistency)
        "block_cache": {
            "enabled": False,  # Always disabled for standard patch
            "recalc_interval": cache_recalc_interval,
            "cached_functions": {},
            "last_recalc_block": -1,
            "current_block_idx": 0
        },
        "debug_cache": debug_cache
    }
    
    # Add hook for size capture
    hook_tome_model(diffusion_model)
    
    # Apply ToMe to all transformer blocks (existing behavior)
    for _, module in diffusion_model.named_modules():
        # Support both BasicTransformerBlock (UNet) and DiT blocks
        if (isinstance_str(module, "BasicTransformerBlock") or 
            isinstance_str(module, "DiTBlock") or
            (hasattr(module, 'attn') and hasattr(module, 'norm1'))):  # Generic DiT-like block detection
            
            module._tome_info = diffusion_model._tome_info
            
            # Set up proportional attention if requested
            if if_proportional_attention:
                from .proportional_attention import ProportionalAttentionProcessor
                # UNet blocks have attn1 and attn2
                if hasattr(module, 'attn1'):
                    module.attn1.set_processor(ProportionalAttentionProcessor())
                if hasattr(module, 'attn2') and merge_crossattn:
                    module.attn2.set_processor(ProportionalAttentionProcessor())
                # DiT blocks have just attn
                elif hasattr(module, 'attn'):
                    module.attn.set_processor(ProportionalAttentionProcessor())


def remove_patch(model: torch.nn.Module):
    """Remove ToMe patches from the model"""
    
    # Handle both direct models and pipelines
    if hasattr(model, "unet"):
        target_model = model.unet
    elif hasattr(model, "transformer"):
        target_model = model.transformer
    elif hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        target_model = model.model.diffusion_model
    else:
        target_model = model
    
    # Remove hooks
    if hasattr(target_model, "_tome_info"):
        for hook in target_model._tome_info.get("hooks", []):
            hook.remove()
        del target_model._tome_info
    
    # Reset attention processors and clean up block-level flags
    for name, module in target_model.named_modules():
        if isinstance(module, Attention):
            if hasattr(module, '_tome_enabled'):
                module.set_processor(AttnProcessor2_0())  # Reset to default processor
                delattr(module, '_tome_enabled')
        
        # Clean up transformer block flags
        if "transformer_blocks" in name:
            if hasattr(module, '_tome_enabled'):
                delattr(module, '_tome_enabled')
            if hasattr(module, '_tome_block_index'):
                delattr(module, '_tome_block_index')


def clear_cached_indices(model: torch.nn.Module):
    """Clear cached merge indices"""
    
    # Handle both direct models and pipelines
    if hasattr(model, "unet"):
        target_model = model.unet
    elif hasattr(model, "transformer"):
        target_model = model.transformer
    elif hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        target_model = model.model.diffusion_model
    else:
        target_model = model
    
    if hasattr(target_model, "_tome_info"):
        # Clear any cached indices or masks
        keys_to_clear = [k for k in target_model._tome_info.keys() 
                        if k.startswith('cached_') or k.startswith('mask_')]
        for key in keys_to_clear:
            del target_model._tome_info[key]


def clear_block_cache(model: torch.nn.Module):
    """Clear cached merge functions for block-level caching"""
    
    # Handle both direct models and pipelines
    if hasattr(model, "unet"):
        target_model = model.unet
    elif hasattr(model, "transformer"):
        target_model = model.transformer
    elif hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        target_model = model.model.diffusion_model
    else:
        target_model = model
    
    if hasattr(target_model, "_tome_info") and "block_cache" in target_model._tome_info:
        cache_info = target_model._tome_info["block_cache"]
        cache_info["cached_functions"].clear()
        cache_info["last_recalc_block"] = -1
        cache_info["current_block_idx"] = 0


def precompute_cache_from_clean_latent(
    model: torch.nn.Module, 
    clean_latent: torch.Tensor,
    **kwargs
):
    """Precompute cache indices from clean latent"""
    
    # Handle both direct models and pipelines
    if hasattr(model, "unet"):
        target_model = model.unet
    elif hasattr(model, "transformer"):
        target_model = model.transformer
    elif hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        target_model = model.model.diffusion_model
    else:
        target_model = model
    
    if hasattr(target_model, "_tome_info"):
        args = target_model._tome_info.get("args", {})
        if args.get("cache_indices_per_image", False):
            # Store the clean latent for cache computation
            target_model._tome_info["clean_latent"] = clean_latent.detach().clone()
            print("Precomputed cache from clean latent") 