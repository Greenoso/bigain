import torch
import math
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Callable
try:
    from .token_scoring import (
        _compute_token_scores,
        _select_indices_by_mode,
        _validate_common_args,
        _compute_non_uniform_grid_indices
    )
except ImportError:
    from token_scoring import (
        _compute_token_scores,
        _select_indices_by_mode,
        _validate_common_args,
        _compute_non_uniform_grid_indices
    )





def frequency_based_selection(
    x: torch.Tensor,
    k: int,
    selection_method: str = "1d_dft", # Options: '1d_dft', '1d_dct', '2d_conv', 'non_uniform_grid', 'snr', 'noise_magnitude', 'agent_guided', 'agent_spatial', 'agent_statistical', 'agent_frequency', 'agent_clustering', 'agent_hybrid'
    mode: str = "high", # Options: 'high', 'low', 'medium','timestep_scheduler', 'frequency_uniform'
    ranking_method: str = "amplitude", # ('amplitude', 'spectral_centroid' for 1D), ('variance', 'l1norm', 'l2norm' for original) (Ignored for non_uniform_grid)
    H: Union[int, None] = None, # Required for 2d_conv method
    W: Union[int, None] = None, # Required for 2d_conv method
    # *** Description updated to reflect the NEW goal ***
    timestep_normalized: Optional[float] = None, # Normalized timestep (0.0 LATE -> 1.0 EARLY) for scheduler mode
    alpha: float = 2.0, # ---> ADD THIS ARGUMENT (Bias strength for non_uniform_grid, alpha > 1)
    # New parameters for SNR and noise magnitude methods
    clean_signal: Optional[torch.Tensor] = None, # Clean signal x0 for SNR calculation (B, N, C)
    noise: Optional[torch.Tensor] = None, # Noise tensor for SNR/noise magnitude calculation (B, N, C)
    # New parameters for agent-guided methods
    agent_method: Optional[str] = None, # Agent creation method ('adaptive_spatial', 'clustering_centroids', 'statistical_moments', 'frequency_based', 'uniform_sampling')
    importance_method: Optional[str] = None, # Agent importance method ('cross_attention', 'cosine_similarity', 'euclidean_distance', 'information_theoretic')
    base_scoring_method: Optional[str] = None, # Base scoring method for hybrid approaches ('original', '1d_dft', '2d_conv', '2d_conv_l2')
    num_agents: Optional[int] = None, # Number of agents to create (default: 16)
    agent_weight: Optional[float] = None # Weight for agent vs base scoring (1.0 = pure agent, 0.5 = 50/50 hybrid)
    ):
    """
    OPTIMIZED: Selects top-k or bottom-k tokens based on frequency characteristics (1D)
    or spatial filtering response (2D). Can also gradually shift selection
    from low (early step) to high (late step) frequency based on a normalized timestep.

    Args:
        x (torch.Tensor): Input tensor used to compute indices (B, N, C).
                          B=Batch size, N=Num tokens, C=Features.
        k (int): Number of tokens to select.
        selection_method (str): Method to use for selection:
                                '1d_dft', '1d_dct': Use 1D FFT/DCT on feature dim C.
                                '2d_conv': Use 2D spatial convolution (Laplacian) filter.
                                Defaults to '1d_dft'.
        mode (str): Selection mode:
                    'high': Select tokens with highest scores (high freq/centroid/response).
                    'low': Select tokens with lowest scores (low freq/centroid/response).
                    'medium': Select tokens with scores in the middle range.
                    'timestep_scheduler': Gradually shift from 'low' (early step, large raw t)
                                          to 'high' (late step, small raw t).
                                          Requires `timestep_normalized` argument.
                    'frequency_uniform': Select tokens uniformly distributed across the entire
                                        frequency spectrum, including both high and low frequencies.
                    Defaults to 'high'.
        ranking_method (str): Method for calculating token scores from 1D frequency amplitudes:
                              'amplitude': Sum of frequency amplitudes.
                              'spectral_centroid': Weighted average frequency based on amplitude.
                              Applies only when selection_method is '1d_dft' or '1d_dct'.
                              Defaults to 'amplitude'.
        H (int | None): Height dimension for '2d_conv' method.
        W (int | None): Width dimension for '2d_conv' method.
        timestep_normalized (float | None): Normalized timestep value (ideally 0.0 late step to 1.0 early step)
                                            required when mode is 'timestep_scheduler'.
                                            Calculated as raw_t / max_t.
        alpha (float): Bias strength parameter for 'non_uniform_grid' (alpha > 1). Default 2.0.

    Returns:
        function: A function that takes a tensor (B, N, C') and returns the
                  selected tokens (B, k, C'). If k is invalid, returns an identity function.

    Raises:
        ValueError: If method/arguments are invalid or incompatible.
    """
    B, N, C = x.shape
    
    # === OPTIMIZED: Use centralized validation ===
    if not _validate_common_args(x, k, selection_method, mode, ranking_method, H, W, clean_signal, noise, timestep_normalized, alpha, agent_method, importance_method, base_scoring_method, num_agents, agent_weight):
        print(f"Warning: k={k} is out of valid range (0, {N}). Performing no selection.")
        return lambda tensor: tensor

    # === OPTIMIZED: Simplified index computation ===
    with torch.no_grad():
        # Handle non_uniform_grid method (special case)
        if selection_method == "non_uniform_grid":
            selected_indices = _compute_non_uniform_grid_indices(k, H, W, alpha, x.device, B)
        else:
            # Use centralized scoring and selection
            token_scores = _compute_token_scores(x, selection_method, ranking_method, H, W, clean_signal, noise, agent_method, importance_method, base_scoring_method, num_agents, agent_weight)
            selected_indices = _select_indices_by_mode(token_scores, k, mode, timestep_normalized)
        
        # Sort for locality
        selected_indices = selected_indices.sort(dim=1)[0]

    # === OPTIMIZED: Simplified selection function ===
    def select(tensor: torch.Tensor) -> torch.Tensor:
        current_B, current_N, current_C_prime = tensor.shape
        if current_N != N:
            print(f"Warning: Input tensor N ({current_N}) doesn't match N ({N}) used for index calculation.")
        
        # Handle batch size adaptation
        target_B = selected_indices.shape[0]
        if current_B != target_B:
            if current_B % target_B == 0:
                repeat_factor = current_B // target_B
                indices_adapted = selected_indices.repeat_interleave(repeat_factor, dim=0)
            else:
                raise ValueError(f"Input tensor batch size {current_B} incompatible with precomputed indices for B={target_B}")
        else:
            indices_adapted = selected_indices
        
        indices_expanded = indices_adapted.unsqueeze(-1).expand(current_B, k, current_C_prime)
        try:
            selected_tensor = torch.gather(tensor, dim=1, index=indices_expanded.to(tensor.device))
        except RuntimeError as e:
             print(f"Error during torch.gather. Input shape: {tensor.shape}, Index shape: {indices_expanded.shape}, Min/Max index: {indices_expanded.min()}/{indices_expanded.max()}")
             raise e
        return selected_tensor
    
    return select


def frequency_based_selection_blockwise( # Renamed for clarity
    x: torch.Tensor,
    downsampling_factor: int, # Instead of k
    H: int, # Must be provided
    W: int, # Must be provided
    selection_method: str = "1d_dft",
    mode: str = "high",
    ranking_method: str = "amplitude",
    timestep_normalized: Optional[Union[float, torch.Tensor]] = None, # Normalized timestep for scheduler mode (supports batched)
    # New parameters for SNR and noise magnitude methods
    clean_signal: Optional[torch.Tensor] = None, # Clean signal x0 for SNR calculation (B, N, C)
    noise: Optional[torch.Tensor] = None, # Noise tensor for SNR/noise magnitude calculation (B, N, C)
    alpha_t: Optional[torch.Tensor] = None # Alpha values for SNR calculation (B,) or scalar
):
    """
    OPTIMIZED: Selects one token per downsampling_factor x downsampling_factor block based
    on frequency characteristics within that block.

    Args:
        x (torch.Tensor): Input tensor used to compute indices (B, N, C). N = H * W.
        downsampling_factor (int): The factor by which to downsample spatially (e.g., 2 for 2x2 blocks).
        H (int): Height dimension of the token grid.
        W (int): Width dimension of the token grid.
        selection_method (str): 'original','1d_dft', '1d_dct', '2d_conv'.
        mode (str): Selection mode within each block:
                    'high': Select token with the highest score in the block.
                    'low': Select token with the lowest score in the block.
                    'medium': Select token with the score closest to the median score within the block.
                    'frequency_uniform': Select tokens uniformly distributed across the frequency
                                       spectrum within each block, ensuring diverse frequency representation.
                    'timestep_scheduler': Select tokens based on timestep within frequency-sorted spectrum.
                    'reverse_timestep_scheduler': Like timestep_scheduler but with reversed ordering.
                    Defaults to 'high'.
        ranking_method (str): 'amplitude', 'spectral_centroid' (for 1D methods), variance, l1norm, l2norm for original statistics as frequency proxy
        timestep_normalized (float | torch.Tensor | None): Normalized timestep (0.0 LATE -> 1.0 EARLY) for scheduler modes. 
                                                          Can be single value or tensor for per-sample timesteps.

    Returns:
        function: A function that takes a tensor (B, N, C') and returns the
                  selected tokens (B, k, C'), where k = N / (factor*factor).
                  Returns identity if factor <= 1 or dimensions invalid.

    Raises:
        ValueError: If arguments are invalid or dimensions incompatible.
    """
    B, N, C = x.shape
    factor = downsampling_factor

    # === OPTIMIZED: Early validation and setup ===
    if factor <= 1:
        return lambda tensor: tensor # Return identity function
    if H * W != N:
        raise ValueError(f"H ({H}) * W ({W}) = {H*W} does not match N ({N}).")
    
    # Allow non‑divisible H,W by simply cropping to the largest f×f grid
    h_blks = H // factor
    w_blks = W // factor
    crop_H = h_blks * factor
    crop_W = w_blks * factor

    if h_blks == 0 or w_blks == 0:
        return lambda t: t
    k = h_blks * w_blks # Target number of tokens = number of blocks
    
    # === OPTIMIZED: Use centralized validation (simplified for blockwise) ===
    valid_methods = ["original","1d_dft", "1d_dct", "2d_conv", "2d_conv_l2", "snr", "noise_magnitude"]
    if selection_method not in valid_methods: 
        raise ValueError(f"Unknown selection_method: {selection_method}.")
    if selection_method.startswith("1d") and ranking_method not in ["amplitude", "spectral_centroid"]: 
        raise ValueError(f"Unknown ranking_method: {ranking_method}.")
    if selection_method == "original" and ranking_method not in ["variance", "l1norm", "l2norm", "mean_deviation"]: 
        raise ValueError(f"For 'original' method in blockwise, ranking_method must be one of ['variance', 'l1norm', 'l2norm', 'mean_deviation'].")
    if selection_method in ["snr", "noise_magnitude"]:
        if clean_signal is None or noise is None:
            raise ValueError(f"clean_signal and noise must be provided for '{selection_method}' method.")
        if clean_signal.shape != x.shape or noise.shape != x.shape:
            raise ValueError(f"clean_signal and noise must have same shape as x ({x.shape}).")
    if mode not in ["high", "low", "medium", "frequency_uniform", "timestep_scheduler", "reverse_timestep_scheduler"]:
        mode = "high" # Default fallback

    # === OPTIMIZED: Use centralized scoring and blockwise processing ===
    with torch.no_grad():
        # Use centralized scoring
        token_scores = _compute_token_scores(x, selection_method, ranking_method, H, W, clean_signal, noise)
        
        # === Blockwise processing ===
        scores_spatial = token_scores.view(B, H, W)
        scores_crop = scores_spatial[:, :crop_H, :crop_W]
        scores_blocks_unfold = (
            scores_crop
            .unfold(1, factor, factor)  # → (B, h_blks, W, f)
            .unfold(2, factor, factor)  # → (B, h_blks, w_blks, f, f)
        )
        
        scores_blocks = scores_blocks_unfold.reshape(B, k, factor * factor)
        
        # === OPTIMIZED: Simplified mode selection ===
        if mode == "high":
            _, best_local_indices = torch.max(scores_blocks, dim=-1) # (B, k)
        elif mode == "low":
            _, best_local_indices = torch.min(scores_blocks, dim=-1) # (B, k)
        elif mode == "medium":
            median_scores, _ = torch.median(scores_blocks, dim=-1, keepdim=True) # (B, k, 1)
            abs_diff = torch.abs(scores_blocks - median_scores) # (B, k, f*f)
            _, best_local_indices = torch.min(abs_diff, dim=-1) # (B, k)
        elif mode in ["timestep_scheduler", "reverse_timestep_scheduler"]:
            block_sorted_indices = torch.argsort(scores_blocks, dim=-1, descending=True) # (B, k, f*f)
            block_size = factor * factor
            
            if isinstance(timestep_normalized, torch.Tensor) and timestep_normalized.numel() == B:
                t_batch = timestep_normalized.to(x.device).float()
                t_batch = torch.clamp(t_batch, 0.0, 1.0)
                timestep_positions = (t_batch * (block_size - 1)).long()
                timestep_positions = torch.clamp(timestep_positions, 0, block_size - 1)
                best_local_indices = torch.gather(
                    block_sorted_indices, 
                    dim=-1, 
                    index=timestep_positions.unsqueeze(1).unsqueeze(2).expand(-1, k, 1)
                ).squeeze(-1)
            else:
                if timestep_normalized is None:
                    timestep_normalized = 0.5
                t = torch.clamp(torch.tensor(timestep_normalized, device=x.device, dtype=torch.float32), 0.0, 1.0)
                timestep_position = (t * (block_size - 1)).long()
                timestep_position = torch.clamp(timestep_position, 0, block_size - 1)
                best_local_indices = block_sorted_indices[:, :, timestep_position]
        else:
            raise RuntimeError(f"Unsupported mode '{mode}' for blockwise selection")
            
        # === Calculate global indices ===
        block_row_indices = torch.arange(h_blks, device=x.device)
        block_col_indices = torch.arange(w_blks, device=x.device)
        block_start_row_idx = (block_row_indices.unsqueeze(1) * factor).expand(-1, w_blks).reshape(k)
        block_start_col_idx = (block_col_indices.unsqueeze(0) * factor).expand(h_blks, -1).reshape(k)

        local_row_offset = best_local_indices // factor
        local_col_offset = best_local_indices % factor
        selected_global_row = block_start_row_idx.unsqueeze(0) + local_row_offset
        selected_global_col = block_start_col_idx.unsqueeze(0) + local_col_offset
        selected_indices = selected_global_row * W + selected_global_col

    # === OPTIMIZED: Simplified selection function ===
    def select(tensor: torch.Tensor) -> torch.Tensor:
        current_B, current_N, current_C_prime = tensor.shape
        if current_N != N:
            print(f"Warning: Input tensor N ({current_N}) doesn't match N ({N}) used for index calculation.")
        
        target_B = selected_indices.shape[0]
        if current_B != target_B:
            if current_B % target_B == 0:
                repeat_factor = current_B // target_B
                indices_adapted = selected_indices.repeat_interleave(repeat_factor, dim=0)
            else:
                raise ValueError(f"Input tensor batch size {current_B} incompatible with precomputed indices for B={target_B}")
        else:
            indices_adapted = selected_indices

        indices_expanded = indices_adapted.unsqueeze(-1).expand(current_B, k, current_C_prime)
        try:
            selected_tensor = torch.gather(tensor, dim=1, index=indices_expanded.to(tensor.device))
        except RuntimeError as e:
            print(f"Error during torch.gather. Input shape: {tensor.shape}, Index shape: {indices_expanded.shape}, Min/Max index: {indices_expanded.min()}/{indices_expanded.max()}")
            raise e
        return selected_tensor
    return select




def frequency_based_token_mask_blockwise(
    x: torch.Tensor,
    downsampling_factor: int,
    H: int,
    W: int,
    selection_method: str = "1d_dft",
    mode: str = "high",
    ranking_method: str = "amplitude",
    timestep_normalized: Optional[float] = None,
    # New parameters for SNR and noise magnitude methods
    clean_signal: Optional[torch.Tensor] = None, # Clean signal x0 for SNR calculation (B, N, C)
    noise: Optional[torch.Tensor] = None, # Noise tensor for SNR/noise magnitude calculation (B, N, C)
    alpha_t: Optional[torch.Tensor] = None # Alpha values for SNR calculation (B,) or scalar
):
    """
    Generates a boolean mask for tokens based on blockwise frequency characteristics.
    Similar to frequency_based_selection_blockwise but returns a mask instead of a selection function.
    
    Args:
        x (torch.Tensor): Input tensor to compute mask from (B, N, C).
        downsampling_factor (int): The factor by which to downsample spatially (e.g., 2 for 2x2 blocks).
        H (int): Height dimension of the token grid.
        W (int): Width dimension of the token grid.
        selection_method (str): 'original','1d_dft', '1d_dct', '2d_conv'.
        mode (str): Selection mode within each block ('high', 'low', 'medium').
        ranking_method (str): 'amplitude', 'spectral_centroid' (for 1D methods), variance, l1norm, l2norm for original.
        timestep_normalized (float | None): Currently unused but kept for API consistency.
        
    Returns:
        torch.Tensor: Boolean mask of shape (B, N) where True indicates tokens to keep.
    """
    B, N, C = x.shape
    factor = downsampling_factor

    # --- Input Validation ---
    if factor <= 1:
        # No downsampling, keep all tokens
        return torch.ones(B, N, dtype=torch.bool, device=x.device)
    if H * W != N:
        raise ValueError(f"H ({H}) * W ({W}) = {H*W} does not match N ({N}).")
    
    # Allow non-divisible H,W by simply cropping to the largest f×f grid
    h_blks = H // factor
    w_blks = W // factor
    crop_H = h_blks * factor
    crop_W = w_blks * factor

    # If nothing fits, just return all True
    if h_blks == 0 or w_blks == 0:
        return torch.ones(B, N, dtype=torch.bool, device=x.device)

    # Simplified validation for blockwise context
    valid_methods = ["original","1d_dft", "1d_dct", "2d_conv", "2d_conv_l2", "snr", "noise_magnitude"]
    if selection_method not in valid_methods: 
        raise ValueError(f"Unknown selection_method: {selection_method}.")
    is_1d = selection_method.startswith("1d")
    is_2d_conv_l2 = selection_method == "2d_conv_l2"
    if is_1d and ranking_method not in ["amplitude", "spectral_centroid"]: 
        raise ValueError(f"Unknown ranking_method: {ranking_method}.")
    if selection_method == "original" and ranking_method not in ["variance", "l1norm", "l2norm", "mean_deviation"]: 
        raise ValueError(f"For 'original' method in blockwise, ranking_method must be one of ['variance', 'l1norm', 'l2norm', 'mean_deviation'].")
    if selection_method in ["snr", "noise_magnitude"]:
        if clean_signal is None or noise is None:
            raise ValueError(f"clean_signal and noise must be provided for '{selection_method}' method.")
        if clean_signal.shape != x.shape or noise.shape != x.shape:
            raise ValueError(f"clean_signal and noise must have same shape as x ({x.shape}).")

    if mode not in ["high", "low", "medium"]:
        # Default or handle scheduler differently if needed
        mode = "high"

    # --- Calculate Scores and Create Mask ---
    with torch.no_grad():
        x_float = x.float()
        token_scores = None

        # --- Calculate Token Scores (same logic as frequency_based_selection_blockwise) ---
        if is_1d:
            freq_coeffs = None
            if selection_method == "1d_dft": 
                freq_coeffs = torch.fft.fft(x_float, dim=-1)
            elif selection_method == "1d_dct":
                freq_coeffs = torch.fft.fft(x_float, dim=-1).real
            amplitudes = torch.abs(freq_coeffs)
            if ranking_method == "amplitude":
                token_scores = amplitudes[..., 1:].sum(dim=-1) if amplitudes.shape[-1] > 1 else amplitudes.sum(dim=-1)
            elif ranking_method == "spectral_centroid":
                effective_C = amplitudes.shape[-1]
                frequencies = torch.arange(effective_C, device=x.device, dtype=amplitudes.dtype).view(1, 1, -1)
                numerator = torch.sum(frequencies * amplitudes, dim=-1)
                denominator = torch.sum(amplitudes, dim=-1)
                epsilon = 1e-7
                token_scores = numerator / (denominator + epsilon)
        elif selection_method == "2d_conv":
            x_reshaped = x_float.view(B, H, W, C).permute(0, 3, 1, 2)
            kernel_size = 3
            padding = kernel_size // 2
            kernel_vals = [[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]
            kernel = torch.tensor(kernel_vals, dtype=x_float.dtype, device=x.device).repeat(C, 1, 1, 1)
            response = F.conv2d(x_reshaped, kernel, padding=padding, groups=C)
            token_scores = response.abs().sum(dim=1).view(B, N)
        elif is_2d_conv_l2:
            x_reshaped = x_float.view(B, H, W, C).permute(0, 3, 1, 2)
            kernel_size = 3
            padding = kernel_size // 2
            kernel_vals = [[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]
            kernel = torch.tensor(kernel_vals, dtype=x_float.dtype, device=x.device).repeat(C, 1, 1, 1)
            response = F.conv2d(x_reshaped, kernel, padding=padding, groups=C)
            token_scores = torch.linalg.norm(response, ord=2, dim=1).view(B, N)
        elif selection_method == "original":
            if ranking_method == "variance":
                token_scores = torch.var(x_float, dim=-1)
            elif ranking_method == "l1norm":
                token_scores = torch.linalg.norm(x_float, ord=1, dim=-1)
            elif ranking_method == "l2norm":
                token_scores = torch.linalg.norm(x_float, ord=2, dim=-1)
            elif ranking_method == "mean_deviation":
                mean_token = torch.mean(x_float, dim=1, keepdim=True)
                deviation = x_float - mean_token
                token_scores = torch.linalg.norm(deviation, ord=2, dim=-1)
        elif selection_method == "snr":
            # Signal-to-Noise Ratio per token (simplified for ranking)
            # SNR = signal_power / noise_power (no scaling needed for ranking)
            clean_signal_float = clean_signal.float()
            noise_float = noise.float()
            
            # Calculate power (L2 norm squared) per token - no scaling needed for ranking
            signal_power = torch.sum(clean_signal_float ** 2, dim=-1)  # (B, N)
            noise_power = torch.sum(noise_float ** 2, dim=-1)          # (B, N)
            
            # SNR = signal_power / (noise_power + epsilon)
            epsilon = 1e-8
            token_scores = signal_power / (noise_power + epsilon)
            
        elif selection_method == "noise_magnitude":
            # Per-token noise magnitude (simplified for ranking)
            # Just use raw noise magnitude - no scaling needed for ranking
            noise_float = noise.float()
            
            # Calculate L2 norm per token
            token_scores = torch.linalg.norm(noise_float, ord=2, dim=-1)  # (B, N)

        # --- Blockwise Selection on the top-left crop ---
        if token_scores is None:
            raise RuntimeError("Failed to compute token scores for blockwise selection")
        
        scores_spatial = token_scores.view(B, H, W)
        scores_crop = scores_spatial[:, :crop_H, :crop_W]
        
        # Unfold exactly into h_blks×w_blks blocks of size factor×factor
        scores_blocks_unfold = (
            scores_crop
            .unfold(1, factor, factor)  # → (B, h_blks, W, factor)
            .unfold(2, factor, factor)  # → (B, h_blks, w_blks, factor, factor)
        )
        
        # Flatten block dimensions and local dimensions
        scores_blocks = scores_blocks_unfold.reshape(B, h_blks * w_blks, factor * factor)
        
        # Find the index of the best token within each block
        best_local_indices = None
        if mode == "high":
            _, best_local_indices = torch.max(scores_blocks, dim=-1)  # (B, h_blks * w_blks)
        elif mode == "low":
            _, best_local_indices = torch.min(scores_blocks, dim=-1)  # (B, h_blks * w_blks)
        elif mode == "medium":
            median_scores, _ = torch.median(scores_blocks, dim=-1, keepdim=True)  # (B, h_blks * w_blks, 1)
            abs_diff = torch.abs(scores_blocks - median_scores)  # (B, h_blks * w_blks, factor*factor)
            _, best_local_indices = torch.min(abs_diff, dim=-1)  # (B, h_blks * w_blks)
        
        if best_local_indices is None:
            raise RuntimeError(f"Unsupported mode '{mode}' for blockwise selection")
        
        # --- Calculate Global Indices ---
        k = h_blks * w_blks  # Total number of blocks
        
        # Indices for the top-left corner of each block (Row-major)
        block_row_indices = torch.arange(h_blks, device=x.device)  # 0, 1, ... h_blks-1
        block_col_indices = torch.arange(w_blks, device=x.device)  # 0, 1, ... w_blks-1

        # Calculate block_start_row_idx and block_start_col_idx for each of the k blocks
        block_start_row_idx = (block_row_indices.unsqueeze(1) * factor).expand(-1, w_blks).reshape(k)  # Shape (k,)
        block_start_col_idx = (block_col_indices.unsqueeze(0) * factor).expand(h_blks, -1).reshape(k)  # Shape (k,)

        # Calculate the LOCAL row/col offset within the block from the flat local index
        local_row_offset = best_local_indices // factor  # (B, k) row offset (0 to factor-1)
        local_col_offset = best_local_indices % factor   # (B, k) col offset (0 to factor-1)

        # Calculate the final GLOBAL flat index
        selected_global_row = block_start_row_idx.unsqueeze(0) + local_row_offset  # (B, k)
        selected_global_col = block_start_col_idx.unsqueeze(0) + local_col_offset  # (B, k)
        selected_indices = selected_global_row * W + selected_global_col           # (B, k)

        # Create boolean mask
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)
        mask[batch_indices, selected_indices] = True
        
        return mask


def frequency_based_token_mask(
    x: torch.Tensor,
    reduction_ratio: float,
    selection_method: str = "1d_dft",
    mode: str = "high", 
    ranking_method: str = "amplitude",
    H: Union[int, None] = None,
    W: Union[int, None] = None,
    timestep_normalized: Optional[float] = None,
    alpha: float = 2.0,
    # New parameters for SNR and noise magnitude methods
    clean_signal: Optional[torch.Tensor] = None, # Clean signal x0 for SNR calculation (B, N, C)
    noise: Optional[torch.Tensor] = None, # Noise tensor for SNR/noise magnitude calculation (B, N, C)
    alpha_t: Optional[torch.Tensor] = None # Alpha values for SNR calculation (B,) or scalar
    ):
    """
    OPTIMIZED: Generates a boolean mask for tokens based on frequency characteristics.
    
    Args:
        x (torch.Tensor): Input tensor to compute mask from (B, N, C).
        reduction_ratio (float): Ratio of tokens to remove (0.0 to 1.0).
                                If 0.6, keep 40% of tokens with highest scores.
        selection_method (str): Method for frequency analysis ('1d_dft', '1d_dct', '2d_conv', etc.).
        mode (str): Selection mode ('high', 'low', 'medium', etc.).
        ranking_method (str): Method for ranking tokens ('amplitude', 'spectral_centroid', etc.).
        H (int | None): Height dimension for 2D methods.
        W (int | None): Width dimension for 2D methods.
        timestep_normalized (float | None): Normalized timestep for scheduler modes.
        alpha (float): Bias strength for non_uniform_grid method.
        
    Returns:
        torch.Tensor: Boolean mask of shape (B, N) where True indicates tokens to keep.
    """
    B, N, C = x.shape
    
    # Calculate number of tokens to keep
    k = int(N * (1.0 - reduction_ratio))
    k = max(1, min(k, N - 1))  # Ensure valid range
    
    if k == N:
        # Keep all tokens
        return torch.ones(B, N, dtype=torch.bool, device=x.device)
    
    # === OPTIMIZED: Use centralized utilities ===
    with torch.no_grad():
        # Handle non_uniform_grid method (special case that doesn't use scoring)
        if selection_method == "non_uniform_grid":
            selected_indices = _compute_non_uniform_grid_indices(k, H, W, alpha, x.device, B)
        else:
            # Use centralized scoring and selection
            token_scores = _compute_token_scores(x, selection_method, ranking_method, H, W, clean_signal, noise)
            selected_indices = _select_indices_by_mode(token_scores, k, mode, timestep_normalized)
            
        # Create boolean mask from selected indices
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)
        mask[batch_indices, selected_indices] = True
        
        return mask


def frequency_based_selection_blockwise_with_blend(
    x: torch.Tensor,
    downsampling_factor: int,
    H: int,
    W: int,
    blend_factor: float = 0.5,  # 0.0 = pure avg_pool, 1.0 = pure frequency selection
    selection_method: str = "1d_dft",
    mode: str = "high",
    ranking_method: str = "amplitude",
    timestep_normalized: Optional[Union[float, torch.Tensor]] = None,
    # New parameters for SNR and noise magnitude methods
    clean_signal: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    alpha_t: Optional[torch.Tensor] = None
):
    """
    Enhanced version of frequency_based_selection_blockwise that supports linear interpolation
    between frequency-selected tokens and average-pooled tokens within each block.
    
    This function implements a blending mechanism similar to the linear_blend downsample method:
    - At blend_factor = 0.0: Returns pure average pooling (smooth, preserves all information)
    - At blend_factor = 1.0: Returns pure frequency selection (sharp, keeps highest-scored tokens)
    - At blend_factor = 0.5: Returns 50/50 blend between the two approaches
    
    Args:
        x (torch.Tensor): Input tensor used to compute indices (B, N, C). N = H * W.
        downsampling_factor (int): The factor by which to downsample spatially.
        H (int): Height dimension of the token grid.
        W (int): Width dimension of the token grid.
        blend_factor (float): Blend factor for interpolation. 0.0 = pure avg_pool, 1.0 = pure frequency selection.
        selection_method (str): Method for frequency analysis ('1d_dft', '1d_dct', '2d_conv', etc.).
        mode (str): Selection mode ('high', 'low', 'medium', etc.).
        ranking_method (str): Ranking method ('amplitude', 'spectral_centroid', etc.).
        timestep_normalized (float | torch.Tensor | None): Normalized timestep for scheduler modes.
        clean_signal (torch.Tensor | None): Clean signal for SNR calculation.
        noise (torch.Tensor | None): Noise tensor for SNR/noise magnitude calculation.
        alpha_t (torch.Tensor | None): Alpha values for SNR calculation.
        
    Returns:
        function: A function that takes a tensor (B, N, C') and returns the
                  blended tokens (B, k, C'), where k = N / (factor*factor).
    """
    B, N, C = x.shape
    factor = downsampling_factor
    
    # Early validation
    if factor <= 1:
        return lambda tensor: tensor
    if H * W != N:
        raise ValueError(f"H ({H}) * W ({W}) = {H*W} does not match N ({N}).")
    
    # Calculate block dimensions
    h_blks = H // factor
    w_blks = W // factor
    crop_H = h_blks * factor
    crop_W = w_blks * factor
    
    if h_blks == 0 or w_blks == 0:
        return lambda t: t
    k = h_blks * w_blks  # Number of output tokens = number of blocks
    
    # CFG-style extended range: >1.0 emphasizes frequency selection, <0.0 emphasizes average pooling
    
    # Get the frequency selection function for reuse
    freq_select_fn = frequency_based_selection_blockwise(
        x, downsampling_factor, H, W, selection_method, mode, ranking_method,
        timestep_normalized, clean_signal, noise, alpha_t
    )
    
    def select_with_blend(tensor: torch.Tensor) -> torch.Tensor:
        current_B, current_N, current_C_prime = tensor.shape
        
        # === Handle pure cases first for efficiency ===
        if blend_factor == 1.0:
            # Pure frequency selection - use the original function directly
            return freq_select_fn(tensor)
        elif blend_factor == 0.0:
            # Pure average pooling
            tensor_spatial = tensor.view(current_B, H, W, current_C_prime)
            tensor_crop = tensor_spatial[:, :crop_H, :crop_W, :]  # (B, crop_H, crop_W, C')
            
            # Use unfold to create blocks and average pool within each block
            tensor_blocks_unfold = (
                tensor_crop
                .permute(0, 3, 1, 2)  # (B, C', crop_H, crop_W)
                .unfold(2, factor, factor)  # (B, C', h_blks, crop_W, factor)
                .unfold(3, factor, factor)  # (B, C', h_blks, w_blks, factor, factor)
            )
            
            # Reshape and average pool
            tensor_blocks = tensor_blocks_unfold.permute(0, 2, 3, 1, 4, 5)  # (B, h_blks, w_blks, C', factor, factor)
            tensor_blocks = tensor_blocks.reshape(current_B, k, current_C_prime, factor * factor)
            return tensor_blocks.mean(dim=-1)  # (B, k, C')
        
        # === Blended case ===
        # Get frequency selected tokens
        freq_selected_tokens = freq_select_fn(tensor)
        
        # Get average pooled tokens
        tensor_spatial = tensor.view(current_B, H, W, current_C_prime)
        tensor_crop = tensor_spatial[:, :crop_H, :crop_W, :]  # (B, crop_H, crop_W, C')
        
        # Use unfold to create blocks and average pool within each block
        tensor_blocks_unfold = (
            tensor_crop
            .permute(0, 3, 1, 2)  # (B, C', crop_H, crop_W)
            .unfold(2, factor, factor)  # (B, C', h_blks, crop_W, factor)
            .unfold(3, factor, factor)  # (B, C', h_blks, w_blks, factor, factor)
        )
        
        # Reshape and average pool
        tensor_blocks = tensor_blocks_unfold.permute(0, 2, 3, 1, 4, 5)  # (B, h_blks, w_blks, C', factor, factor)
        tensor_blocks = tensor_blocks.reshape(current_B, k, current_C_prime, factor * factor)
        avg_pooled_tokens = tensor_blocks.mean(dim=-1)  # (B, k, C')
        
        # Linear interpolation: blend_factor * freq + (1 - blend_factor) * avg_pool
        return blend_factor * freq_selected_tokens + (1.0 - blend_factor) * avg_pooled_tokens
    
    return select_with_blend







