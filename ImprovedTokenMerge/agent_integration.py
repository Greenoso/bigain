"""
Agent-guided token downsampling integration module.

IMPORTANT UPDATE: This module now uses SIMPLIFIED agent-guided token merging 
instead of expensive bipartite matching:

1. **Efficient**: Merge unimportant tokens into ONE single token (not bipartite matching)
2. **Fast**: O(N) complexity instead of O(N²) bipartite matching 
3. **Attention-optimized**: Preserves all Q tokens, only reduces K and V tokens
4. **Backward compatible**: Same interface as existing frequency_based_selection

OLD APPROACH (EXPENSIVE):
- Complex bipartite matching between source and destination tokens
- Multiple merge operations and unmerge reconstructions
- O(N²) complexity for finding optimal matches

NEW APPROACH (EFFICIENT):
- Agent-guided importance scoring: O(N×agents) 
- Keep top-k important tokens unchanged
- Merge ALL remaining tokens into ONE single token by averaging
- Total complexity: O(N×agents + N log N) - much faster!

For token downsampling in attention layers:
- Q tokens: Always preserved (needed for query computation)
- K,V tokens: Reduced using agent guidance (k important + 1 merged token)
- Result: Maintains attention quality while dramatically reducing computation

This module provides convenient wrappers and utilities to integrate the new
simplified agent-guided token downsampling methods with existing infrastructure 
while maintaining full backward compatibility.
"""

import torch
from typing import Optional, Callable, Dict, Union
try:
    from .frequency_selection import frequency_based_selection
    from .agent_downsampling import agent_guided_selection, SimpleAgentGuidedMerging
except ImportError:
    from frequency_selection import frequency_based_selection
    from agent_downsampling import agent_guided_selection, SimpleAgentGuidedMerging


def create_simple_agent_selector(
    selection_method: str,
    agent_method: str = "adaptive_spatial",
    importance_method: str = "cross_attention",
    base_scoring_method: Optional[str] = None,
    num_agents: int = 16,
    agent_weight: float = 1.0
) -> Callable:
    """
    Factory function to create simplified agent-guided selectors.
    
    NEW: Uses efficient single-token merging instead of bipartite matching.
    
    Args:
        selection_method: Type of agent selection ('simple_agent', 'hybrid_l2', 'hybrid_l1', 
                         'hybrid_1d_dft', 'hybrid_2d_conv')
        agent_method: Agent creation method
        importance_method: Agent importance scoring method
        base_scoring_method: Base scoring method for hybrid approaches
        num_agents: Number of agents to create (default: 16)
        agent_weight: Weight for agent vs base scoring (default: 1.0)
        
    Returns:
        Configured simplified agent selector function
        
    Performance Notes:
        - simple_agent: O(N×agents + N log N) - fastest
        - hybrid_*: O(N×agents + N×base_method) - slightly slower but more robust
    """
    if selection_method == "simple_agent":
        # Pure simplified agent-guided selection
        def selector(x, k, mode="high", H=None, W=None, timestep_normalized=None):
            return agent_guided_selection(
                x, k,
                agent_method=agent_method,
                importance_method=importance_method,
                mode=mode,
                num_agents=num_agents,
                H=H, W=W,
                timestep_normalized=timestep_normalized
            )
        
    elif selection_method == "hybrid_l2":
        # Agent + L2 norm hybrid with simplified merging
        def selector(x, k, mode="high", H=None, W=None, timestep_normalized=None):
            return agent_guided_selection(
                x, k,
                agent_method=agent_method,
                importance_method=importance_method,
                mode=mode,
                base_scoring_method="original",
                base_ranking_method="l2norm",
                num_agents=num_agents,
                agent_weight=agent_weight,
                H=H, W=W,
                timestep_normalized=timestep_normalized
            )
    
    elif selection_method == "hybrid_l1":
        # Agent + L1 norm hybrid with simplified merging
        def selector(x, k, mode="high", H=None, W=None, timestep_normalized=None):
            return agent_guided_selection(
                x, k,
                agent_method=agent_method,
                importance_method=importance_method,
                mode=mode,
                base_scoring_method="original",
                base_ranking_method="l1norm",
                num_agents=num_agents,
                agent_weight=agent_weight,
                H=H, W=W,
                timestep_normalized=timestep_normalized
            )
            
    elif selection_method == "hybrid_1d_dft":
        # Agent + 1D DFT hybrid with simplified merging
        def selector(x, k, mode="high", H=None, W=None, timestep_normalized=None):
            return agent_guided_selection(
                x, k,
                agent_method=agent_method,
                importance_method=importance_method,
                mode=mode,
                base_scoring_method="1d_dft",
                base_ranking_method="amplitude",
                num_agents=num_agents,
                agent_weight=agent_weight,
                H=H, W=W,
                timestep_normalized=timestep_normalized
            )
            
    elif selection_method == "hybrid_2d_conv":
        # Agent + 2D convolution hybrid with simplified merging
        def selector(x, k, mode="high", H=None, W=None, timestep_normalized=None):
            return agent_guided_selection(
                x, k,
                agent_method=agent_method,
                importance_method=importance_method,
                mode=mode,
                base_scoring_method="2d_conv",
                base_ranking_method="l2norm",
                num_agents=num_agents,
                agent_weight=agent_weight,
                H=H, W=W,
                timestep_normalized=timestep_normalized
            )
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    return selector


def quick_simple_agent_selection(
    x: torch.Tensor,
    k: int,
    mode: str = "high",
    H: Optional[int] = None,
    W: Optional[int] = None,
    agent_method: str = "adaptive_spatial",
    importance_method: str = "cross_attention",
    num_agents: int = 16
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Quick simplified agent selection with sensible defaults.
    
    NEW: Uses efficient O(N×agents) approach instead of O(N²) bipartite matching.
    
    Args:
        x: Input tensor (B, N, C)
        k: Number of important tokens to keep
        mode: Selection mode (default: "high")
        H, W: Spatial dimensions (required for spatial methods)
        agent_method: Agent creation method (default: "adaptive_spatial")
        importance_method: Importance scoring method (default: "cross_attention")
        num_agents: Number of agents (default: 16)
        
    Returns:
        Selection function that returns (B, k+1, C) - k important + 1 merged token
    
    Example:
        >>> selector = quick_simple_agent_selection(x, k=64, H=32, W=32)
        >>> reduced_tokens = selector(x)  # (B, 65, C) = 64 important + 1 merged
    """
    return agent_guided_selection(
        x, k,
        agent_method=agent_method,
        importance_method=importance_method,
        mode=mode,
        num_agents=num_agents,
        H=H, W=W
    )


def quick_hybrid_selection(
    x: torch.Tensor,
    k: int,
    base_method: str = "1d_dft",
    weight: float = 0.6,
    mode: str = "high",
    H: Optional[int] = None,
    W: Optional[int] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Quick hybrid selection combining agent guidance with traditional methods.
    
    NEW: Uses simplified merging - much faster than bipartite matching.
    
    Args:
        x: Input tensor (B, N, C)
        k: Number of important tokens to keep
        base_method: Base scoring method ("1d_dft", "2d_conv", "original")
        weight: Agent weight (0.6 = 60% agent, 40% base method)
        mode: Selection mode (default: "high")
        H, W: Spatial dimensions
        
    Returns:
        Selection function that returns (B, k+1, C)
    
    Performance comparison vs old approach:
        - Old: O(N²) bipartite + O(N×base) = very slow
        - New: O(N×agents + N×base) = much faster
    """
    base_ranking_map = {
        "1d_dft": "amplitude",
        "2d_conv": "l2norm", 
        "original": "l2norm"
    }
    
    return agent_guided_selection(
        x, k,
        agent_method="adaptive_spatial",
        importance_method="cross_attention",
        mode=mode,
        base_scoring_method=base_method,
        base_ranking_method=base_ranking_map.get(base_method, "l2norm"),
        agent_weight=weight,
        H=H, W=W
    )


# Backward compatibility - redirect to new efficient implementations
def create_agent_selector(*args, **kwargs):
    """DEPRECATED: Use create_simple_agent_selector for better performance."""
    print("WARNING: create_agent_selector is deprecated.")
    print("Use create_simple_agent_selector for the new efficient approach.")
    return create_simple_agent_selector(*args, **kwargs)


class AgentTokenScorer:
    """
    TokenScorer-compatible class for agent-guided scoring.
    Integrates with existing TokenScorer interface from tomesd/scoring.py
    """
    
    def __init__(
        self,
        agent_method: str = "adaptive_spatial",
        importance_method: str = "cross_attention",
        base_scoring_method: Optional[str] = None,
        base_ranking_method: str = "l2norm",
        num_agents: int = 16,
        agent_weight: float = 1.0
    ):
        """
        Args:
            agent_method: Agent creation method
            importance_method: Agent importance scoring method  
            base_scoring_method: Optional base scoring method for hybrid approach
            base_ranking_method: Base ranking method for hybrid approach
            num_agents: Number of agents to create
            agent_weight: Weight for agent vs base scoring
        """
        self.agent_method = agent_method
        self.importance_method = importance_method
        self.base_scoring_method = base_scoring_method
        self.base_ranking_method = base_ranking_method
        self.num_agents = num_agents
        self.agent_weight = agent_weight
        
        # Initialize the downsampler
        self.downsampler = SimpleAgentGuidedMerging(
            num_agents=num_agents,
            agent_method=agent_method,
            importance_method=importance_method
        )
        
        # For hybrid approach
        if base_scoring_method is not None:
            try:
                from .agent_guided_scoring import HybridAgentScorer
            except ImportError:
                from agent_guided_scoring import HybridAgentScorer
            self.hybrid_scorer = HybridAgentScorer(
                agent_method=agent_method,
                importance_method=importance_method,
                base_scoring_method=base_scoring_method,
                base_ranking_method=base_ranking_method,
                num_agents=num_agents,
                agent_weight=agent_weight
            )
            self.use_hybrid = True
        else:
            self.use_hybrid = False
    
    def score_tokens(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculate scores for tokens using agent guidance.
        Compatible with TokenScorer interface.
        
        Args:
            x: Input tensor (B, N, C)
            **kwargs: Additional parameters (H, W, timestep, etc.)
            
        Returns:
            Token scores (B, N)
        """
        H = kwargs.get('H', None)
        W = kwargs.get('W', None)
        clean_signal = kwargs.get('clean_signal', None)
        noise = kwargs.get('noise', None)
        
        with torch.no_grad():
            if self.use_hybrid:
                return self.hybrid_scorer.compute_hybrid_scores(
                    x, H=H, W=W, clean_signal=clean_signal, noise=noise
                )
            else:
                _, importance_scores = self.downsampler(x, H=H, W=W)
                return importance_scores
    
    def get_name(self) -> str:
        """Return scorer name for caching/logging"""
        if self.use_hybrid:
            return f"agent_hybrid_{self.agent_method}_{self.importance_method}_{self.base_scoring_method}_{self.agent_weight}"
        else:
            return f"agent_{self.agent_method}_{self.importance_method}"


if __name__ == "__main__":
    # Run compatibility demonstration
    demonstrate_compatibility() 