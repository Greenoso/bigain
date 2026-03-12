import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Dict, Any
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention
import math
import torch.nn.functional as F

# Import SiTo functions
from .sito import (
    prune_and_recover_tokens,
    do_nothing
)
from .utils import isinstance_str, init_generator


class DiTSiToProcessor:
    """Custom attention processor for DiT that implements SiTo token pruning."""
    
    def __init__(self):
        self.sito_info = None
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if self.sito_info is None:
            # Fallback to default attention processor if no sito_info
            from diffusers.models.attention_processor import AttnProcessor2_0
            default_processor = AttnProcessor2_0()
            return default_processor(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            
        residual = hidden_states
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, dim = hidden_states.shape
        
        # Compute SiTo prune/recover functions based on current state
        prune_fn, recover_fn = compute_sito_dit(hidden_states, self.sito_info)
        
        # Apply pruning strategy based on sito_info settings
        args = self.sito_info['args']
        
        # Standard attention computation with SiTo integration
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # Apply pruning before query computation if enabled for self-attention
        if args.get('prune_selfattn_flag', True):
            hidden_states = prune_fn(hidden_states)
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # Apply pruning for cross-attention keys/values if enabled
            if args.get('prune_crossattn_flag', False):
                encoder_hidden_states = prune_fn(encoder_hidden_states)
        
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # Reshape for attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Scaled dot product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        # Recover tokens after attention computation
        if args.get('prune_selfattn_flag', True) or args.get('prune_crossattn_flag', False):
            hidden_states = recover_fn(hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


def compute_sito_dit(x: torch.Tensor, sito_info: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Compute SiTo prune/recover functions specifically for DiT architecture.
    
    DiT doesn't have hierarchical levels like U-Net, so we adapt the logic accordingly.
    """
    batch_size, num_tokens, dim = x.shape
    
    # Extract spatial dimensions from sito_info
    if sito_info["size"] is None:
        return do_nothing, do_nothing
        
    original_h, original_w = sito_info["size"]
    timestep = sito_info.get("timestep", 0)
    
    # For DiT, we work with the actual token grid size
    # DiT typically uses patch embeddings, so tokens = (H/patch_size) * (W/patch_size)
    patch_size = sito_info.get("patch_size", 8)  # Default for DiT with 512x512 images and 64x64 tokens
    h = original_h // patch_size
    w = original_w // patch_size
    
    # Ensure our calculation matches the actual number of tokens
    if h * w != num_tokens:
        # Try to infer the correct dimensions
        h = w = int(math.sqrt(num_tokens))
        if h * w != num_tokens:
            return do_nothing, do_nothing
    
    # Get SiTo parameters
    args = sito_info["args"]
    
    # Calculate number of tokens to prune
    num_prune = int(num_tokens * args.get("prune_ratio", 0.5))
    
    if num_prune <= 0:
        return do_nothing, do_nothing
    
    # Use SiTo's prune_and_recover_tokens function
    try:
        prune_fn, recover_fn = prune_and_recover_tokens(
            x,
            num_prune=num_prune,
            w=w,
            h=h,
            sx=args.get('sx', 2),
            sy=args.get('sy', 2),
            noise_alpha=args.get('noise_alpha', 0.1),
            sim_beta=args.get('sim_beta', 1.0),
            current_timestep=timestep
        )
        return prune_fn, recover_fn
    except Exception as e:
        return do_nothing, do_nothing


class DiTSiToBlock(nn.Module):
    """
    Wrapper for DiT transformer blocks that applies SiTo token pruning.
    """
    
    def __init__(self, original_block):
        super().__init__()
        self.original_block = original_block
        self.sito_info = None
        
    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        if self.sito_info is None:
            return self.original_block(hidden_states, encoder_hidden_states, **kwargs)
        
        # Compute SiTo functions
        prune_fn, recover_fn = compute_sito_dit(hidden_states, self.sito_info)
        args = self.sito_info['args']
        
        # Apply SiTo to different components based on flags
        original_hidden_states = hidden_states
        
        # Apply to self-attention if enabled
        if args.get('prune_selfattn_flag', True):
            hidden_states = prune_fn(hidden_states)
        
        # Forward through the original block
        output = self.original_block(hidden_states, encoder_hidden_states, **kwargs)
        
        # Recover tokens if pruning was applied
        if args.get('prune_selfattn_flag', True):
            output = recover_fn(output)
        
        return output


def patch_dit_sito(model: DiTTransformer2DModel, sito_args: Dict[str, Any] = None):
    """
    Patch a DiT model to use SiTo token pruning.
    
    Args:
        model: DiTTransformer2DModel to patch
        sito_args: Dictionary containing SiTo configuration parameters
                  - block_sito_flags: List of 28 boolean/int values (0/1) indicating 
                    which transformer blocks should apply SiTo. 
                    e.g., [1]*28 = all blocks, [0]*28 = no blocks
    """
    if sito_args is None:
        sito_args = {
            "prune_ratio": 0.5,
            "max_downsample_ratio": 1,
            "prune_selfattn_flag": True,
            "prune_crossattn_flag": False,
            "prune_mlp_flag": False,
            "sx": 2,
            "sy": 2,
            "noise_alpha": 0.1,
            "sim_beta": 1.0,
            "block_sito_flags": [1] * 28  # Default: apply to all 28 blocks
        }
    
    # Ensure block_sito_flags is properly configured
    if "block_sito_flags" not in sito_args:
        sito_args["block_sito_flags"] = [1] * 28  # Default to all blocks
    
    # Validate block_sito_flags
    block_flags = sito_args["block_sito_flags"]
    if not isinstance(block_flags, (list, tuple)):
        raise ValueError("block_sito_flags must be a list or tuple")
    
    # Auto-detect number of blocks if not 28
    expected_blocks = len(block_flags)
    if expected_blocks != 28:
        print(f"Warning: Expected 28 blocks for DiT-XL/2, got {expected_blocks} flags. "
              f"Adjusting configuration accordingly.")
    
    # Remove any existing patches
    remove_dit_sito_patch(model)
    
    # Set up SiTo info on the model
    model._sito_info = {
        "size": None,
        "timestep": None,
        "patch_size": 8,  # Default for DiT-XL/2 with 512x512 -> 64x64 tokens
        "hooks": [],
        "args": sito_args
    }
    
    # Add forward hook to capture size and timestep
    def hook(module, args):
        # For DiT, args[0] is hidden_states, args[1] is timestep
        if len(args) >= 1:
            # Extract spatial dimensions from the input
            # DiT input is typically [B, C, H, W] or [B, N, C]
            if len(args[0].shape) == 4:
                _, _, h, w = args[0].shape
                module._sito_info["size"] = (h, w)
            elif len(args[0].shape) == 3:
                # Infer spatial dimensions from token count
                B, N, C = args[0].shape
                # For DiT: if we have N tokens in a square grid, spatial size = sqrt(N) * patch_size
                tokens_per_side = int(math.sqrt(N))
                patch_size = module._sito_info.get("patch_size", 8)
                size = tokens_per_side * patch_size
                module._sito_info["size"] = (size, size)
        
        # Extract timestep
        if len(args) >= 2:
            timestep = args[1]
            if hasattr(timestep, 'item'):
                # Handle tensor timesteps
                if timestep.numel() == 1:
                    module._sito_info["timestep"] = timestep.item()
                else:
                    # For batched timesteps, take the first one (assuming all are the same)
                    module._sito_info["timestep"] = timestep[0].item()
            elif isinstance(timestep, (int, float)):
                module._sito_info["timestep"] = timestep
        
        return None
    
    model._sito_info["hooks"].append(model.register_forward_pre_hook(hook))
    
    # Patch transformer blocks based on block_sito_flags
    block_counter = 0
    block_flags = sito_args["block_sito_flags"]
    
    for name, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            # Check if this block should apply SiTo
            should_apply_sito = False
            if block_counter < len(block_flags):
                should_apply_sito = bool(block_flags[block_counter])
            
            if should_apply_sito:
                # Replace the attention processor with SiTo version
                if hasattr(module, 'attn1') and module.attn1 is not None:
                    processor = DiTSiToProcessor()
                    processor.sito_info = model._sito_info
                    module.attn1.processor = processor
                
                if hasattr(module, 'attn2') and module.attn2 is not None:
                    processor = DiTSiToProcessor()
                    processor.sito_info = model._sito_info
                    module.attn2.processor = processor
                
                # Store reference to sito_info in the module
                module._sito_info = model._sito_info
                module._sito_enabled = True
            else:
                # Keep default processor for this block
                module._sito_enabled = False
            
            # Store block information for debugging
            module._sito_block_index = block_counter
            
            block_counter += 1
    
    print(f"Applied SiTo to {sum(block_flags)} out of {len(block_flags)} transformer blocks")
    
    return model


def remove_dit_sito_patch(model: DiTTransformer2DModel):
    """
    Remove SiTo patches from a DiT model.
    """
    # Remove hooks
    if hasattr(model, '_sito_info') and 'hooks' in model._sito_info:
        for hook in model._sito_info['hooks']:
            hook.remove()
        model._sito_info['hooks'].clear()
    
    # Reset attention processors to default
    for name, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            if hasattr(module, 'attn1') and module.attn1 is not None and hasattr(module.attn1, 'processor'):
                if isinstance(module.attn1.processor, DiTSiToProcessor):
                    module.attn1.processor = module.attn1.default_processor
            
            if hasattr(module, 'attn2') and module.attn2 is not None and hasattr(module.attn2, 'processor'):
                if isinstance(module.attn2.processor, DiTSiToProcessor):
                    module.attn2.processor = module.attn2.default_processor
            
            # Remove sito_info reference and block-level flags
            if hasattr(module, '_sito_info'):
                delattr(module, '_sito_info')
            if hasattr(module, '_sito_enabled'):
                delattr(module, '_sito_enabled')
            if hasattr(module, '_sito_block_index'):
                delattr(module, '_sito_block_index')
    
    # Clean up model-level sito_info
    if hasattr(model, '_sito_info'):
        delattr(model, '_sito_info')
    
    return model


def apply_sito_to_dit_pipeline(pipe, sito_args=None):
    """
    Apply SiTo to a DiT pipeline (e.g., DiTPipeline).
    
    Args:
        pipe: The diffusion pipeline containing the DiT model
        sito_args: SiTo configuration arguments
    """
    if hasattr(pipe, 'transformer'):
        patch_dit_sito(pipe.transformer, sito_args)
    elif hasattr(pipe, 'unet') and isinstance_str(pipe.unet, "DiTTransformer2DModel"):
        patch_dit_sito(pipe.unet, sito_args)
    else:
        print("Warning: Could not find DiT model in pipeline")
    
    return pipe


def remove_sito_from_dit_pipeline(pipe):
    """
    Remove SiTo from a DiT pipeline.
    """
    if hasattr(pipe, 'transformer'):
        remove_dit_sito_patch(pipe.transformer)
    elif hasattr(pipe, 'unet') and isinstance_str(pipe.unet, "DiTTransformer2DModel"):
        remove_dit_sito_patch(pipe.unet)
    
    return pipe 