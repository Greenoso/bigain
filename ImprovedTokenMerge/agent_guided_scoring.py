import torch
import torch.nn.functional as F
import math
from typing import Optional, Union, Dict, Tuple, List
try:
    from .token_scoring import _compute_token_scores
except ImportError:
    from token_scoring import _compute_token_scores


def extract_attention_projections(unet, layer_name="attn1", extract_layers=("to_q", "to_k")):
    """
    Extract Q/K projection layers from UNet attention modules for agent-guided scoring.
    
    Args:
        unet: The UNet model
        layer_name: Which attention layer to extract from ("attn1" for self-attention, "attn2" for cross-attention)  
        extract_layers: Which projections to extract ("to_q", "to_k", "to_v")
        
    Returns:
        Dict with extracted projection layers that can be used for agent scoring
    """
    projections = {}
    
    # Find Attention modules in the UNet
    attn_modules = [(name, module) for name, module in unet.named_modules() 
                    if module.__class__.__name__ == 'Attention' and layer_name in name]
    
    if not attn_modules:
        # Fallback: try to find any Attention module with the required projections
        all_attn_modules = [(name, module) for name, module in unet.named_modules() 
                           if module.__class__.__name__ == 'Attention']
        
        if all_attn_modules:
            # Use first available attention module
            attn_modules = [all_attn_modules[0]]
        else:
            print("Warning: No Attention modules found in UNet")
            return None
    
    # Use the first attention module's projections as representative
    # (All attention modules in the same layer typically have same dimensions)
    attn_name, attn_module = attn_modules[0]
    
    # Extract the requested projection layers directly from the attention module
    for proj_name in extract_layers:
        if hasattr(attn_module, proj_name):
            projection = getattr(attn_module, proj_name)
            projections[proj_name[3:]] = projection  # Remove "to_" prefix (to_q -> q)
        else:
            print(f"Warning: {proj_name} not found in {attn_name}")
    
    if projections:
        print(f"Agent cross-attention: Extracted Q/K projections from UNet ({attn_name})")
    else:
        print("Agent cross-attention: Failed to extract Q/K projections, will use direct similarity")
    
    return projections if projections else None


class TrainingFreeAgentCreator:
    """Training-free agent creation strategies"""
    
    def __init__(self, num_agents: int = 16):
        self.num_agents = num_agents
    
    def create_agents(
        self, 
        tokens: torch.Tensor, 
        method: str = 'adaptive_spatial',
        H: Optional[int] = None,
        W: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create agent tokens without any learnable parameters
        
        Args:
            tokens: [B, N, D] input tokens
            method: Agent creation method
            H, W: Spatial dimensions (required for spatial methods)
            
        Returns:
            agents: [B, num_agents, D] agent tokens
        """
        if method == 'adaptive_spatial':
            return self._create_spatial_agents(tokens, H, W)
        elif method == 'clustering_centroids':
            return self._create_kmeans_agents(tokens)
        elif method == 'statistical_moments':
            return self._create_statistical_agents(tokens)
        elif method == 'frequency_based':
            return self._create_frequency_agents(tokens)
        elif method == 'uniform_sampling':
            return self._create_uniform_sampling_agents(tokens)
        else:
            raise ValueError(f"Unknown agent creation method: {method}")
    
    def _create_spatial_agents(self, tokens: torch.Tensor, H: Optional[int], W: Optional[int]) -> torch.Tensor:
        """VECTORIZED agents creation via spatial pooling"""
        B, N, D = tokens.shape
        
        if H is not None and W is not None and H * W == N:
            # Perfect spatial arrangement
            tokens_2d = tokens.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W] for conv operations
            
            # VECTORIZED: Create spatial agents via efficient pooling
            agent_h = int(math.sqrt(self.num_agents))
            agent_w = int(math.ceil(self.num_agents / agent_h))
            
            # Calculate effective number of agents we can create
            effective_agents = min(self.num_agents, agent_h * agent_w)
            
            pool_h = max(1, H // agent_h)
            pool_w = max(1, W // agent_w)
            
            # Use adaptive average pooling for efficient spatial downsampling
            # This replaces the nested loops with a single vectorized operation
            if pool_h == H // agent_h and pool_w == W // agent_w and agent_h * agent_w <= self.num_agents:
                # Perfect pooling case - use F.avg_pool2d for maximum efficiency
                pooled = F.avg_pool2d(tokens_2d, kernel_size=(pool_h, pool_w), stride=(pool_h, pool_w))
                # pooled shape: [B, D, agent_h, agent_w]
                
                # Reshape to agent format
                pooled_flat = pooled.permute(0, 2, 3, 1).contiguous()  # [B, agent_h, agent_w, D]
                agents = pooled_flat.view(B, -1, D)  # [B, agent_h*agent_w, D]
                
                # Trim to exact number of agents needed
                if agents.shape[1] > self.num_agents:
                    agents = agents[:, :self.num_agents, :]
                
                # Pad if we have fewer agents than needed
                if agents.shape[1] < self.num_agents:
                    pad_needed = self.num_agents - agents.shape[1]
                    # Repeat the last agent for padding
                    last_agent = agents[:, -1:, :].repeat(1, pad_needed, 1)
                    agents = torch.cat([agents, last_agent], dim=1)
            else:
                # Fallback to adaptive pooling for irregular cases
                target_size = (agent_h, agent_w)
                pooled = F.adaptive_avg_pool2d(tokens_2d, target_size)  # [B, D, agent_h, agent_w]
                
                # Reshape and select agents
                pooled_flat = pooled.permute(0, 2, 3, 1).contiguous()  # [B, agent_h, agent_w, D]
                agents = pooled_flat.view(B, -1, D)  # [B, agent_h*agent_w, D]
                
                # Trim or pad to exact number of agents
                if agents.shape[1] > self.num_agents:
                    agents = agents[:, :self.num_agents, :]
                elif agents.shape[1] < self.num_agents:
                    pad_needed = self.num_agents - agents.shape[1]
                    last_agent = agents[:, -1:, :].repeat(1, pad_needed, 1)
                    agents = torch.cat([agents, last_agent], dim=1)
        else:
            # Fallback: uniform sampling for non-spatial arrangements
            agents = self._create_uniform_sampling_agents(tokens)
        
        return agents
    
    def _create_kmeans_agents(self, tokens: torch.Tensor, max_iters: int = 10) -> torch.Tensor:
        """Training-free K-means clustering"""
        B, N, D = tokens.shape
        
        # Initialize centroids by uniform sampling
        indices = torch.linspace(0, N-1, self.num_agents, dtype=torch.long, device=tokens.device)
        centroids = torch.gather(tokens, 1, indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, D))
        
        # Simple K-means iterations (no gradients)
        with torch.no_grad():
            for _ in range(max_iters):
                # Assign tokens to nearest centroid
                distances = torch.cdist(tokens, centroids)  # [B, N, num_agents]
                assignments = distances.argmin(dim=-1)  # [B, N]
                
                # VECTORIZED: Update all centroids in parallel
                # Create one-hot encoding for all assignments at once
                assignments_onehot = F.one_hot(assignments, num_classes=self.num_agents).float()  # [B, N, num_agents]
                
                # Compute sum of tokens for each centroid across all batches and tokens
                # Using batch matrix multiplication for efficiency
                token_sums = torch.bmm(assignments_onehot.transpose(1, 2), tokens)  # [B, num_agents, D]
                
                # Count tokens assigned to each centroid
                assignment_counts = assignments_onehot.sum(dim=1, keepdim=True)  # [B, 1, num_agents]
                assignment_counts = assignment_counts.transpose(1, 2)  # [B, num_agents, 1]
                
                # Avoid division by zero and compute new centroids
                assignment_counts_safe = torch.clamp(assignment_counts, min=1.0)
                new_centroids = token_sums / assignment_counts_safe
                
                # Keep old centroids where no tokens were assigned (assignment_counts == 0)
                keep_old_mask = (assignment_counts == 0).expand_as(new_centroids)
                centroids = torch.where(keep_old_mask, centroids, new_centroids)
        
        return centroids
    
    def _create_statistical_agents(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create agents based on statistical moments"""
        B, N, D = tokens.shape
        agents = []
        
        # Different statistical representations
        agents.append(tokens.mean(dim=1))  # Global average
        agents.append(tokens.max(dim=1)[0])  # Global max
        agents.append(tokens.min(dim=1)[0])  # Global min
        agents.append(tokens.std(dim=1))   # Global std
        
        # Percentile-based agents
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            if len(agents) >= self.num_agents:
                break
            percentile_agent = torch.quantile(tokens, p/100.0, dim=1)
            agents.append(percentile_agent)
        
        # Random sampling agents for diversity
        remaining = self.num_agents - len(agents)
        if remaining > 0:
            rand_indices = torch.randperm(N, device=tokens.device)[:remaining]
            for idx in rand_indices:
                agents.append(tokens[:, idx, :])
        
        return torch.stack(agents[:self.num_agents], dim=1)
    
    def _create_frequency_agents(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create agents using frequency domain analysis"""
        B, N, D = tokens.shape
        
        # Apply FFT to tokens
        freq_tokens = torch.fft.fft(tokens, dim=-1)
        freq_magnitudes = torch.abs(freq_tokens)
        
        # VECTORIZED: Create agents based on different frequency bands
        num_bands = min(self.num_agents, D // 2)
        if num_bands > 0:
            band_size = D // num_bands
            
            # Create all frequency bands simultaneously using tensor reshaping
            # Reshape frequency magnitudes to group by bands
            freq_reshaped = freq_magnitudes[:, :, :num_bands * band_size].view(B, N, num_bands, band_size)
            
            # Compute band magnitudes for all bands at once
            band_magnitudes = freq_reshaped.mean(dim=-1)  # [B, N, num_bands]
            
            # Compute softmax weights for all bands simultaneously
            weights = F.softmax(band_magnitudes, dim=1)  # [B, N, num_bands]
            
            # Vectorized weighted sum: compute all agents at once
            # Expand tokens to match band dimensions and apply weights
            tokens_expanded = tokens.unsqueeze(2).expand(-1, -1, num_bands, -1)  # [B, N, num_bands, D]
            weights_expanded = weights.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, N, num_bands, D]
            
            # Sum across tokens dimension to get all agents at once
            agents = torch.sum(tokens_expanded * weights_expanded, dim=1)  # [B, num_bands, D]
            agents_list = [agents[:, i, :] for i in range(num_bands)]
        else:
            agents_list = []
        
        # Pad with statistical agents if needed
        while len(agents_list) < self.num_agents:
            if len(agents_list) == 0:
                agents_list.append(tokens.mean(dim=1))
            else:
                # Add some noise to create diversity
                noise = torch.randn_like(agents_list[0]) * 0.01
                agents_list.append(agents_list[-1] + noise)
        
        return torch.stack(agents_list[:self.num_agents], dim=1)
    
    def _create_uniform_sampling_agents(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create agents by uniform sampling"""
        B, N, D = tokens.shape
        indices = torch.linspace(0, N-1, self.num_agents, dtype=torch.long, device=tokens.device)
        agents = torch.gather(tokens, 1, indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, D))
        return agents


class AgentImportanceScorer:
    """Compute token importance scores using agents"""
    
    def __init__(self, importance_method: str = 'cross_attention'):
        self.importance_method = importance_method
    
    def compute_importance(
        self, 
        agents: torch.Tensor, 
        tokens: torch.Tensor,
        existing_qkv_proj: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute importance scores without training
        
        Args:
            agents: [B, num_agents, D] agent tokens
            tokens: [B, N, D] input tokens
            existing_qkv_proj: Optional dict with 'q', 'k', 'v' projection layers
            
        Returns:
            importance: [B, N] importance scores
        """
        if self.importance_method == 'cross_attention':
            return self._cross_attention_importance(agents, tokens, existing_qkv_proj)
        elif self.importance_method == 'cosine_similarity':
            return self._cosine_similarity_importance(agents, tokens)
        elif self.importance_method == 'euclidean_distance':
            return self._euclidean_distance_importance(agents, tokens)
        elif self.importance_method == 'information_theoretic':
            return self._entropy_based_importance(agents, tokens)
        else:
            raise ValueError(f"Unknown importance method: {self.importance_method}")
    
    def _cross_attention_importance(
        self, 
        agents: torch.Tensor, 
        tokens: torch.Tensor,
        existing_qkv_proj: Optional[Dict] = None
    ) -> torch.Tensor:
        """Cross-attention based importance"""
        if existing_qkv_proj is not None:
            # Use existing Q, K, V projections from the model
            Q_agents = existing_qkv_proj['q'](agents)
            K_tokens = existing_qkv_proj['k'](tokens)
            
            scores = torch.matmul(Q_agents, K_tokens.transpose(-2, -1))
            scores = F.softmax(scores / math.sqrt(tokens.size(-1)), dim=-1)
            importance = scores.max(dim=1)[0]  # Max attention from any agent
        else:
            # Direct similarity (no projections needed)
            scores = torch.matmul(agents, tokens.transpose(-2, -1))
            scores = F.softmax(scores / math.sqrt(tokens.size(-1)), dim=-1)
            importance = scores.max(dim=1)[0]
        
        return importance
    
    def _cosine_similarity_importance(self, agents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Normalized cosine similarity"""
        agents_norm = F.normalize(agents, dim=-1)
        tokens_norm = F.normalize(tokens, dim=-1)
        scores = torch.matmul(agents_norm, tokens_norm.transpose(-2, -1))
        importance = scores.max(dim=1)[0]
        return importance
    
    def _euclidean_distance_importance(self, agents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Inverse distance as importance"""
        distances = torch.cdist(agents, tokens)  # [B, num_agents, N]
        importance = 1.0 / (distances.min(dim=1)[0] + 1e-8)  # Inverse distance
        return importance
    
    def _entropy_based_importance(self, agents: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """VECTORIZED training-free entropy-based importance scoring"""
        B, N, D = tokens.shape
        
        # VECTORIZED: Compute entropy for ALL tokens at once (10-50x faster)
        # Find similarities between all tokens and all agents simultaneously
        similarities = torch.matmul(tokens, agents.transpose(-2, -1))  # [B, N, num_agents]
        
        # Compute softmax probabilities for all tokens
        probs = F.softmax(similarities, dim=-1)  # [B, N, num_agents]
        
        # Compute entropy for all tokens: high entropy = high importance (more diverse connections)
        # Use broadcasting to compute entropy efficiently
        log_probs = torch.log(probs + 1e-8)  # [B, N, num_agents]
        importance = -(probs * log_probs).sum(dim=-1)  # [B, N]
        
        return importance


class HybridAgentScorer:
    """Combine agent guidance with existing scoring methods"""
    
    def __init__(
        self, 
        agent_method: str = 'adaptive_spatial',
        importance_method: str = 'cross_attention',
        base_scoring_method: str = 'original',
        base_ranking_method: str = 'l2norm',
        num_agents: int = 16,
        agent_weight: float = 0.5
    ):
        """
        Args:
            agent_method: Agent creation method
            importance_method: Agent importance scoring method
            base_scoring_method: Existing scoring method ('original', '1d_dft', '2d_conv', '2d_conv_l2')
            base_ranking_method: Existing ranking method
            num_agents: Number of agents to create
            agent_weight: Weight for agent scores (1-agent_weight for base scores)
        """
        self.agent_creator = TrainingFreeAgentCreator(num_agents)
        self.importance_scorer = AgentImportanceScorer(importance_method)
        self.agent_method = agent_method
        self.base_scoring_method = base_scoring_method
        self.base_ranking_method = base_ranking_method
        self.agent_weight = agent_weight
    
    def compute_hybrid_scores(
        self,
        tokens: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
        clean_signal: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute hybrid scores combining agent guidance with existing methods
        
        Args:
            tokens: [B, N, D] input tokens
            H, W: Spatial dimensions for 2D methods
            clean_signal, noise: For SNR methods
            
        Returns:
            scores: [B, N] hybrid importance scores
        """
        with torch.no_grad():
            # 1. Create agents
            agents = self.agent_creator.create_agents(
                tokens, 
                method=self.agent_method,
                H=H, W=W
            )
            
            # 2. Compute agent-based importance
            agent_scores = self.importance_scorer.compute_importance(agents, tokens)
            
            # 3. Compute base scoring method scores (only supported methods)
            if self.base_scoring_method in ['original', '1d_dft', '1d_dct', '2d_conv', '2d_conv_l2']:
                base_scores = _compute_token_scores(
                    tokens,
                    selection_method=self.base_scoring_method,
                    ranking_method=self.base_ranking_method,
                    H=H, W=W,
                    clean_signal=clean_signal,
                    noise=noise
                )
            else:
                # Fallback to L2 norm if method not supported
                base_scores = torch.linalg.norm(tokens, ord=2, dim=-1)
            
            # 4. Normalize both scores to [0, 1]
            agent_scores_norm = self._normalize_scores(agent_scores)
            base_scores_norm = self._normalize_scores(base_scores)
            
            # 5. Combine scores
            hybrid_scores = (
                self.agent_weight * agent_scores_norm + 
                (1 - self.agent_weight) * base_scores_norm
            )
            
        return hybrid_scores
    
    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1] range per batch"""
        B, N = scores.shape
        scores_flat = scores.view(B, N)
        
        min_scores = scores_flat.min(dim=1, keepdim=True)[0]
        max_scores = scores_flat.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        score_ranges = max_scores - min_scores
        score_ranges = torch.clamp(score_ranges, min=1e-8)
        
        normalized = (scores_flat - min_scores) / score_ranges
        return normalized.view_as(scores) 