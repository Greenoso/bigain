import torch
import torch.nn as nn
from typing import Optional, Callable, Any


class ProportionalAttentionWrapper(nn.Module):
    """
    Wrapper for attention modules that implements proportional attention.
    
    This modifies the attention computation to account for token sizes:
    A = softmax(QK^T/√d + log(s))
    
    where s is a row vector containing the size of each token (number of patches it represents).
    """
    
    def __init__(self, attention_module: nn.Module, get_token_sizes: Callable[[], Optional[torch.Tensor]]):
        """
        Args:
            attention_module: The original attention module to wrap
            get_token_sizes: A callable that returns the current token sizes tensor
        """
        super().__init__()
        self.attention = attention_module
        self.get_token_sizes = get_token_sizes
        
    def forward(self, *args, **kwargs):
        """
        Forward pass with proportional attention modification.
        
        We intercept the attention computation and add log(token_sizes) to the scores.
        """
        # Get current token sizes
        token_sizes = self.get_token_sizes()
        
        if token_sizes is None:
            # No token merging has occurred yet, use original attention
            return self.attention(*args, **kwargs)
            
        # Store original processor if using diffusers-style attention
        original_processor = None
        if hasattr(self.attention, 'processor'):
            original_processor = self.attention.processor
            # Create a custom processor that adds log(s) to attention scores
            self.attention.set_processor(
                ProportionalAttentionProcessor(original_processor, token_sizes)
            )
        else:
            # For non-diffusers attention, we need to patch the forward method temporarily
            # This is a bit more complex and would require understanding the specific attention implementation
            # For now, we'll use a simpler approach that may need adjustment based on the specific model
            return self._forward_with_proportional_attention(token_sizes, *args, **kwargs)
            
        try:
            output = self.attention(*args, **kwargs)
        finally:
            # Restore original processor
            if original_processor is not None:
                self.attention.set_processor(original_processor)
                
        return output
        
    def _forward_with_proportional_attention(self, token_sizes: torch.Tensor, *args, **kwargs):
        """
        Fallback implementation for non-diffusers attention modules.
        This may need adjustment based on the specific attention implementation.
        """
        # For now, just call the original attention
        # In a full implementation, we would need to intercept the softmax computation
        return self.attention(*args, **kwargs)


class ProportionalAttentionProcessor:
    """
    Custom attention processor for diffusers that implements proportional attention.
    """
    
    def __init__(self, original_processor: Any, token_sizes: torch.Tensor):
        self.original_processor = original_processor
        self.token_sizes = token_sizes
        
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs
    ):
        """
        Perform attention with proportional scaling based on token sizes.
        """
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Prepare query, key, value
        query = attn.to_q(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross_attention else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Reshape for attention computation
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (attn.scale ** 0.5)
        
        # Add log(token_sizes) for proportional attention
        if not is_cross_attention and self.token_sizes is not None:
            # token_sizes shape: [batch_size, num_tokens]
            # We need to broadcast it properly for the attention computation
            log_sizes = torch.log(self.token_sizes.float())
            # Expand for all heads
            log_sizes = log_sizes.unsqueeze(1).expand(-1, attn.heads, -1)
            log_sizes = log_sizes.reshape(-1, log_sizes.shape[-1])
            # Add to attention scores (only to the key dimension)
            attention_scores = attention_scores + log_sizes.unsqueeze(-2)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Compute attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states