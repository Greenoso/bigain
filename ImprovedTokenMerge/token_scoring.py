import torch
import math
import torch.nn.functional as F
from typing import Optional, Union





def _compute_token_scores(
    x: torch.Tensor,
    selection_method: str,
    ranking_method: str,
    H: Optional[int] = None,
    W: Optional[int] = None,
    clean_signal: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    agent_method: Optional[str] = None,
    importance_method: Optional[str] = None,
    base_scoring_method: Optional[str] = None,
    num_agents: Optional[int] = None,
    agent_weight: Optional[float] = None
) -> torch.Tensor:
    """
    Centralized token scoring computation to eliminate duplication across functions.
    
    Args:
        x: Input tensor (B, N, C)
        selection_method: Scoring method
        ranking_method: Ranking method for 1D transforms
        H, W: Spatial dimensions (required for 2D methods)
        clean_signal, noise: For SNR/noise methods
    
    Returns:
        torch.Tensor: Token scores (B, N)
    """
    B, N, C = x.shape
    x_float = x.float()
    
    # 1D frequency methods
    if selection_method.startswith("1d"):
        freq_coeffs = None
        if selection_method == "1d_dft":
            freq_coeffs = torch.fft.fft(x_float, dim=-1)
        else:  # 1d_dct
            freq_coeffs = torch.fft.fft(x_float, dim=-1).real  # Approximation for DCT
        
        amplitudes = torch.abs(freq_coeffs)
        if ranking_method == "amplitude":
            return amplitudes[..., 1:].sum(dim=-1) if amplitudes.shape[-1] > 1 else amplitudes.sum(dim=-1)
        elif ranking_method == "spectral_centroid":
            effective_C = amplitudes.shape[-1]
            frequencies = torch.arange(effective_C, device=x.device, dtype=amplitudes.dtype).view(1, 1, -1)
            numerator = torch.sum(frequencies * amplitudes, dim=-1)
            denominator = torch.sum(amplitudes, dim=-1)
            epsilon = 1e-7
            return numerator / (denominator + epsilon)
    
    # 2D convolution methods
    elif selection_method in ["2d_conv", "2d_conv_l2"]:
        x_reshaped = x_float.view(B, H, W, C).permute(0, 3, 1, 2)
        kernel_size = 3
        padding = kernel_size // 2
        kernel_vals = [[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]
        kernel = torch.tensor(kernel_vals, dtype=x_float.dtype, device=x.device).repeat(C, 1, 1, 1)
        response = F.conv2d(x_reshaped, kernel, padding=padding, groups=C)
        
        if selection_method == "2d_conv":
            return response.abs().sum(dim=1).view(B, N)
        else:  # 2d_conv_l2
            return torch.linalg.norm(response, ord=2, dim=1).view(B, N)
    
    # Original statistical methods
    elif selection_method == "original":
        if ranking_method == "variance":
            return torch.var(x_float, dim=-1)
        elif ranking_method == "l1norm":
            return torch.linalg.norm(x_float, ord=1, dim=-1)
        elif ranking_method == "l2norm":
            return torch.linalg.norm(x_float, ord=2, dim=-1)
        elif ranking_method == "mean_deviation":
            mean_token = torch.mean(x_float, dim=1, keepdim=True)
            deviation = x_float - mean_token
            return torch.linalg.norm(deviation, ord=2, dim=-1)
    
    # SNR method
    elif selection_method == "snr":
        clean_signal_float = clean_signal.float()
        noise_float = noise.float()
        signal_power = torch.sum(clean_signal_float ** 2, dim=-1)
        noise_power = torch.sum(noise_float ** 2, dim=-1)
        epsilon = 1e-8
        return signal_power / (noise_power + epsilon)
    
    # Noise magnitude method
    elif selection_method == "noise_magnitude":
        noise_float = noise.float()
        return torch.linalg.norm(noise_float, ord=2, dim=-1)
    
    # Agent-guided methods
    elif selection_method.startswith("agent_"):
        # Import here to avoid circular imports
        try:
            try:
                from .agent_guided_scoring import HybridAgentScorer, TrainingFreeAgentCreator, AgentImportanceScorer
            except ImportError:
                from agent_guided_scoring import HybridAgentScorer, TrainingFreeAgentCreator, AgentImportanceScorer
        except ImportError:
            raise ImportError("Agent-guided scoring requires agent_guided_scoring module. Please ensure it's available.")
        
        # Set defaults for agent parameters
        agent_method = agent_method or 'adaptive_spatial'
        importance_method = importance_method or 'cross_attention'
        num_agents = num_agents or 16
        agent_weight = agent_weight or 1.0
        
        if selection_method == "agent_guided":
            # Pure agent guidance
            agent_creator = TrainingFreeAgentCreator(num_agents)
            importance_scorer = AgentImportanceScorer(importance_method)
            
            agents = agent_creator.create_agents(x, method=agent_method, H=H, W=W)
            return importance_scorer.compute_importance(agents, x)
            
        elif selection_method == "agent_spatial":
            # Force spatial agent method
            agent_creator = TrainingFreeAgentCreator(num_agents)
            importance_scorer = AgentImportanceScorer(importance_method)
            
            agents = agent_creator.create_agents(x, method='adaptive_spatial', H=H, W=W)
            return importance_scorer.compute_importance(agents, x)
            
        elif selection_method == "agent_statistical":
            # Force statistical agent method
            agent_creator = TrainingFreeAgentCreator(num_agents)
            importance_scorer = AgentImportanceScorer(importance_method)
            
            agents = agent_creator.create_agents(x, method='statistical_moments', H=H, W=W)
            return importance_scorer.compute_importance(agents, x)
            
        elif selection_method == "agent_frequency":
            # Force frequency agent method
            agent_creator = TrainingFreeAgentCreator(num_agents)
            importance_scorer = AgentImportanceScorer(importance_method)
            
            agents = agent_creator.create_agents(x, method='frequency_based', H=H, W=W)
            return importance_scorer.compute_importance(agents, x)
            
        elif selection_method == "agent_clustering":
            # Force clustering agent method
            agent_creator = TrainingFreeAgentCreator(num_agents)
            importance_scorer = AgentImportanceScorer(importance_method)
            
            agents = agent_creator.create_agents(x, method='clustering_centroids', H=H, W=W)
            return importance_scorer.compute_importance(agents, x)
            
        elif selection_method == "agent_hybrid":
            # Hybrid agent + base method
            if base_scoring_method is None:
                base_scoring_method = 'original'
            if base_scoring_method not in ['original', '1d_dft', '1d_dct', '2d_conv', '2d_conv_l2']:
                base_scoring_method = 'original'
                
            base_ranking = ranking_method if ranking_method in ['l1norm', 'l2norm', 'variance', 'amplitude', 'spectral_centroid'] else 'l2norm'
            
            hybrid_scorer = HybridAgentScorer(
                agent_method=agent_method,
                importance_method=importance_method, 
                base_scoring_method=base_scoring_method,
                base_ranking_method=base_ranking,
                num_agents=num_agents,
                agent_weight=agent_weight
            )
            return hybrid_scorer.compute_hybrid_scores(x, H=H, W=W, clean_signal=clean_signal, noise=noise)
    
    raise ValueError(f"Unknown selection_method: {selection_method}")


def _select_indices_by_mode(
    token_scores: torch.Tensor,
    k: int,
    mode: str,
    timestep_normalized: Optional[Union[float, torch.Tensor]] = None
) -> torch.Tensor:
    """
    Centralized index selection logic to eliminate duplication across functions.

    Args:
        token_scores: Precomputed token scores (B, N)
        k: Number of tokens to select
        mode: Selection mode
        timestep_normalized: For scheduler modes

    Returns:
        torch.Tensor: Selected indices (B, k)
    """
    B, N = token_scores.shape
    
    if mode == "high":
        return torch.topk(token_scores, k, dim=1, largest=True)[1]
    elif mode == "low":
        return torch.topk(token_scores, k, dim=1, largest=False)[1]
    elif mode == "medium":
        sorted_indices = torch.argsort(token_scores, dim=1, descending=False)
        num_exclude_each_side = (N - k) // 2
        start_idx = num_exclude_each_side
        return sorted_indices[:, start_idx:start_idx + k]
    elif mode == "frequency_uniform":
        sorted_indices = torch.argsort(token_scores, dim=1, descending=False)
        if k >= N:
            return sorted_indices
        uniform_positions = torch.linspace(0, N-1, k, dtype=torch.long, device=token_scores.device)
        return torch.gather(sorted_indices, dim=1, index=uniform_positions.unsqueeze(0).expand(B, -1))
    elif mode in ["timestep_scheduler", "reverse_timestep_scheduler"]:
        return _timestep_scheduler_selection(token_scores, k, timestep_normalized)
    
    raise ValueError(f"Unknown mode: {mode}")


def _timestep_scheduler_selection(
    token_scores: torch.Tensor,
    k: int,
    timestep_normalized: Optional[Union[float, torch.Tensor]]
) -> torch.Tensor:
    """Helper for timestep scheduler mode selection."""
    B, N = token_scores.shape
    sorted_indices = torch.argsort(token_scores, dim=1, descending=True)
    
    # Handle both single timestep and batched timesteps
    if isinstance(timestep_normalized, (list, torch.Tensor)) and len(timestep_normalized) == B:
        # Batched timesteps: each sample gets its own normalized timestep
        if isinstance(timestep_normalized, list):
            t_batch = torch.tensor(timestep_normalized, device=token_scores.device, dtype=torch.float32)
        else:
            t_batch = timestep_normalized.to(token_scores.device).float()
        t_batch = torch.clamp(t_batch, 0.0, 1.0)
        
        # Calculate per-sample start indices
        start_indices = ((N - k) * t_batch).long()
        start_indices = torch.clamp(start_indices, 0, N - k)
        
        # Select tokens per sample
        selected_indices_list = []
        for i in range(B):
            s_idx = start_indices[i].item()
            e_idx = s_idx + k
            selected_indices_list.append(sorted_indices[i, s_idx:e_idx])
        return torch.stack(selected_indices_list, dim=0)
    else:
        # Single timestep for all samples (original behavior)
        if timestep_normalized is None:
            timestep_normalized = 0.5
        t = torch.clamp(torch.tensor(timestep_normalized, device=token_scores.device, dtype=torch.float32), 0.0, 1.0)
        start_idx = ((N - k) * t).long()
        start_idx = torch.min(start_idx, torch.tensor(N - k, device=start_idx.device))
        end_idx = start_idx + k
        return sorted_indices[:, start_idx:end_idx]


def _validate_common_args(
    x: torch.Tensor,
    k: int,
    selection_method: str,
    mode: str,
    ranking_method: str,
    H: Optional[int] = None,
    W: Optional[int] = None,
    clean_signal: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    timestep_normalized: Optional[Union[float, torch.Tensor]] = None,
    alpha: float = 2.0,
    agent_method: Optional[str] = None,
    importance_method: Optional[str] = None,
    base_scoring_method: Optional[str] = None,
    num_agents: Optional[int] = None,
    agent_weight: Optional[float] = None
) -> bool:
    """Centralized validation logic."""
    B, N, C = x.shape
    
    # Basic k validation
    if not 0 < k < N:
        return False
    
    # Method validation
    valid_methods = ["original", "1d_dft", "1d_dct", "2d_conv", "2d_conv_l2", "non_uniform_grid", "snr", "noise_magnitude", 
                     "agent_guided", "agent_spatial", "agent_statistical", "agent_frequency", "agent_clustering", "agent_hybrid"]
    if selection_method not in valid_methods:
        raise ValueError(f"Unknown selection_method: {selection_method}. Choose from {valid_methods}.")

    # H/W validation for spatial methods
    needs_H_W = selection_method in ["2d_conv", "2d_conv_l2", "non_uniform_grid"]
    if needs_H_W:
        if H is None or W is None:
            raise ValueError(f"H and W arguments must be provided for '{selection_method}' method.")
        if H * W != N:
            raise ValueError(f"H ({H}) * W ({W}) = {H*W} does not match N ({N}).")
    
    # Special validation for non_uniform_grid
    if selection_method == "non_uniform_grid":
        if not alpha > 1.0:
             raise ValueError(f"alpha must be > 1 for 'non_uniform_grid', got {alpha}.")
    
    # SNR/noise validation
    if selection_method in ["snr", "noise_magnitude"]:
        if clean_signal is None or noise is None:
            raise ValueError(f"clean_signal and noise must be provided for '{selection_method}' method.")
        if clean_signal.shape != x.shape or noise.shape != x.shape:
            raise ValueError(f"clean_signal and noise must have same shape as x ({x.shape}).")
        valid_snr_modes = ["high", "low", "medium", "timestep_scheduler", "reverse_timestep_scheduler", "frequency_uniform"]
        if mode not in valid_snr_modes:
            raise ValueError(f"Unknown mode: {mode} for {selection_method} method.")
    elif selection_method.startswith("1d"):
        if ranking_method not in ["amplitude", "spectral_centroid"]:
            raise ValueError(f"Unknown ranking_method: {ranking_method} for 1D methods.")
        valid_1d_modes = ["high", "low", "timestep_scheduler", "medium", "reverse_timestep_scheduler", "frequency_uniform"]
        if mode not in valid_1d_modes:
             raise ValueError(f"Unknown mode: {mode} for 1D methods.")
    elif selection_method == "original":
        valid_original_rankings = ["variance", "l1norm", "l2norm", "mean_deviation"]
        if ranking_method not in valid_original_rankings:
             raise ValueError(f"For 'original' method, ranking_method must be one of {valid_original_rankings}.")
        valid_original_modes = ["high", "low", "medium", "timestep_scheduler", "reverse_timestep_scheduler", "frequency_uniform"]
        if mode not in valid_original_modes:
             raise ValueError(f"Unknown mode: {mode} for original method.")
    elif selection_method in ["2d_conv", "2d_conv_l2"]:
        valid_2d_modes = ["high", "low", "medium", "timestep_scheduler", "reverse_timestep_scheduler", "frequency_uniform"]
        if mode not in valid_2d_modes:
              raise ValueError(f"Unknown mode: {mode} for 2d_conv method.")
    elif selection_method.startswith("agent_"):
        # Validation for agent methods
        valid_agent_modes = ["high", "low", "medium", "timestep_scheduler", "reverse_timestep_scheduler", "frequency_uniform"]
        if mode not in valid_agent_modes:
            raise ValueError(f"Unknown mode: {mode} for agent methods.")
            
        # Validate agent parameters
        if agent_method is not None:
            valid_agent_methods = ['adaptive_spatial', 'clustering_centroids', 'statistical_moments', 'frequency_based', 'uniform_sampling']
            if agent_method not in valid_agent_methods:
                raise ValueError(f"Unknown agent_method: {agent_method}. Choose from {valid_agent_methods}.")
                
        if importance_method is not None:
            valid_importance_methods = ['cross_attention', 'cosine_similarity', 'euclidean_distance', 'information_theoretic']
            if importance_method not in valid_importance_methods:
                raise ValueError(f"Unknown importance_method: {importance_method}. Choose from {valid_importance_methods}.")
                
        if num_agents is not None and num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}.")
            
        if agent_weight is not None and not (0.0 <= agent_weight <= 1.0):
            raise ValueError(f"agent_weight must be between 0.0 and 1.0, got {agent_weight}.")
            
        # Special validation for spatial methods
        if (agent_method == 'adaptive_spatial' or selection_method == "agent_spatial") and (H is None or W is None):
            # Don't raise error, just warn - spatial methods can fallback to uniform sampling
            pass

    # Check timestep dependency only if mode requires it AND method isn't grid
    if mode in ["timestep_scheduler", "reverse_timestep_scheduler"] and selection_method != "non_uniform_grid":
        if timestep_normalized is None:
            raise ValueError(f"`timestep_normalized` must be provided when mode is '{mode}'.")
  
    return True


def _compute_non_uniform_grid_indices(k: int, H: int, W: int, alpha: float, device, B: int) -> torch.Tensor:
    """Helper function for non_uniform_grid index computation."""
    N_keep_target = k
    # Estimate target grid dimensions W_prime, H_prime based on k and H/W ratio
    aspect_ratio = W / H if H > 0 else 1
    H_prime_approx = math.sqrt(N_keep_target / aspect_ratio) if aspect_ratio > 0 else N_keep_target
    W_prime_approx = H_prime_approx * aspect_ratio
    H_prime = max(1, int(round(H_prime_approx)))
    W_prime = max(1, int(round(N_keep_target / H_prime)))

    # Generate uniform points
    eps = 1e-6
    u_x = torch.linspace(-1.0 + eps, 1.0 - eps, W_prime, device=device)
    u_y = torch.linspace(-1.0 + eps, 1.0 - eps, H_prime, device=device)

    # Map to non-uniform coordinates
    center_x, center_y = (W - 1) / 2.0, (H - 1) / 2.0
    scale_x, scale_y = (W - 1) / 2.0, (H - 1) / 2.0
    mapped_x = center_x + scale_x * torch.sign(u_x) * torch.pow(torch.abs(u_x), alpha)
    mapped_y = center_y + scale_y * torch.sign(u_y) * torch.pow(torch.abs(u_y), alpha)

    # Round and clip
    indices_x = torch.clip(torch.round(mapped_x), 0, W - 1).long()
    indices_y = torch.clip(torch.round(mapped_y), 0, H - 1).long()

    # Create meshgrid and get unique indices (row, col)
    final_cols, final_rows = torch.meshgrid(indices_x, indices_y, indexing='xy')
    initial_indices_rc = torch.stack((final_rows.flatten(), final_cols.flatten()), dim=-1)
    unique_indices_rc = torch.unique(initial_indices_rc, dim=0)

    N_actual = unique_indices_rc.shape[0]

    # --- Adjustment Step (implement if needed for exact k) ---
    if N_actual == N_keep_target:
        final_indices_rc = unique_indices_rc
    elif N_actual > N_keep_target:
        # Remove furthest from center
        center_y_t, center_x_t = (H - 1) / 2.0, (W - 1) / 2.0
        dist_sq = (unique_indices_rc[:, 0] - center_y_t)**2 + (unique_indices_rc[:, 1] - center_x_t)**2
        indices_to_keep_mask = torch.argsort(dist_sq)[:N_keep_target]
        final_indices_rc = unique_indices_rc[indices_to_keep_mask]
    else:  # N_actual < N_keep_target
        # Add closest missing indices
        num_to_add = N_keep_target - N_actual
        all_rows_idx, all_cols_idx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        all_indices_rc = torch.stack((all_rows_idx.flatten(), all_cols_idx.flatten()), dim=-1)

        # Find missing (can be slow, consider alternatives for large N)
        combined = torch.cat((unique_indices_rc, all_indices_rc))
        uniques, counts = combined.unique(dim=0, return_counts=True)
        missing_indices_rc = uniques[counts == 1]

        if len(missing_indices_rc) < num_to_add:
            final_indices_rc = torch.vstack((unique_indices_rc, missing_indices_rc))
        else:
            center_y_t, center_x_t = (H - 1) / 2.0, (W - 1) / 2.0
            missing_dist_sq = (missing_indices_rc[:, 0] - center_y_t)**2 + (missing_indices_rc[:, 1] - center_x_t)**2
            indices_to_add_mask = torch.argsort(missing_dist_sq)[:num_to_add]
            additional_indices = missing_indices_rc[indices_to_add_mask]
            final_indices_rc = torch.vstack((unique_indices_rc, additional_indices))

    # Convert (row, col) to flat indices
    selected_indices_flat = final_indices_rc[:, 0] * W + final_indices_rc[:, 1]

    # Add batch dimension
    return selected_indices_flat.unsqueeze(0).repeat(B, 1)


# === END UTILITY FUNCTIONS ===
