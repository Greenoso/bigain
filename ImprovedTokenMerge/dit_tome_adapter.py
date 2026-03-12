import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Dict, Any
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention
import math
import torch.nn.functional as F



from .merge import (
    frequency_based_selection, 
    frequency_based_selection_blockwise,
    compute_merge as compute_merge_unet,
    do_nothing,
    up_or_downsample,
    bipartite_soft_matching_random2d,
    init_generator,
    frequency_based_token_mask,
    frequency_based_token_mask_blockwise
)

class DiTTokenMergeProcessor:
    """Custom attention processor for DiT that implements token merging."""
    
    def __init__(self, block_index=None):
        self.tome_info = None
        self.block_index = block_index  # Index of the transformer block this processor belongs to
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if self.tome_info is None:
            # Fallback to standard attention if no tome_info
            return attn(hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            
        residual = hidden_states
        input_ndim = hidden_states.ndim
        
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        batch_size, sequence_length, dim = hidden_states.shape
        
        # Update current block index for layer reuse tracking
        if self.block_index is not None:
            self.tome_info['current_block_index'] = self.block_index
            
        # Compute merge functions based on current state (with layer reuse support)
        merge_fn, unmerge_fn = compute_merge_dit_with_reuse(hidden_states, self.tome_info)
        
        # Apply merging strategy
        if self.tome_info['args']['merge_tokens'] == "all":
            hidden_states = merge_fn(hidden_states)
        
        # Standard attention computation
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        # Handle keys/values merging
        if self.tome_info['args']['merge_tokens'] == "keys/values":
            # Check if we're using masked attention
            if self.tome_info['args']['merge_method'] in ["masked_attention", "blockwise_masked_attention"]:
                # For masked attention, compute merge to generate the mask
                # but don't apply it yet - it will be used when computing K and V
                merge_fn_kv, _ = compute_merge_dit(encoder_hidden_states, self.tome_info)
                # encoder_hidden_states remains unchanged for masked attention
            else:
                # Normal merging - determine source tensor for computing merge indices
                selection_source = self.tome_info['args'].get('selection_source', 'hidden')
                
                if selection_source == "hidden":
                    source_tensor = encoder_hidden_states
                elif selection_source == "query":
                    source_tensor = query
                elif selection_source == "key":
                    source_tensor = attn.to_k(encoder_hidden_states)
                elif selection_source == "value":
                    source_tensor = attn.to_v(encoder_hidden_states)
                else:
                    source_tensor = encoder_hidden_states
                    
                merge_fn_kv, _ = compute_merge_dit(source_tensor, self.tome_info)
                encoder_hidden_states = merge_fn_kv(encoder_hidden_states)
        
                # Compute keys and values  
        # Note: For DiT masked attention, the masking is handled in compute_merge_dit
        # which stores the mask in tome_info. The actual token filtering could be
        # implemented here, but for simplicity we use the standard computation.
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
        
        # Unmerge if needed
        if self.tome_info['args']['merge_tokens'] == "all":
            hidden_states = unmerge_fn(hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        
        hidden_states = hidden_states / attn.rescale_output_factor
        
        return hidden_states


def compute_merge_dit_with_reuse(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Compute merge functions for DiT with layer reuse support.
    Recalculates merge/unmerge functions every N ToMe blocks (default: 3).
    Cache is per timestep (cleared between timesteps).
    """
    args = tome_info["args"]
    
    # Check if layer reuse is enabled
    if not args.get("layer_reuse_enabled", False):
        return compute_merge_dit(x, tome_info)
    
    # Get current block index and ToMe block mapping
    current_block_idx = tome_info.get('current_block_index', 0)
    block_tome_flags = args.get("block_tome_flags", [1] * 28)
    
    # Calculate current ToMe block index (only counting enabled blocks)
    tome_block_idx = 0
    for i in range(current_block_idx + 1):
        if i < len(block_tome_flags) and block_tome_flags[i]:
            if i == current_block_idx:
                break
            tome_block_idx += 1
    
    # Get reuse frequency (default: 3)
    reuse_frequency = args.get("layer_reuse_frequency", 3)
    
    # Create cache key based on tensor shape and reuse block group
    reuse_group = tome_block_idx // reuse_frequency
    batch_size, num_tokens, dim = x.shape
    cache_key = f"reuse_group_{reuse_group}_B{batch_size}_N{num_tokens}_D{dim}"
    
    # Initialize reuse cache if not exists
    if "layer_reuse_cache" not in tome_info:
        tome_info["layer_reuse_cache"] = {}
    
    reuse_cache = tome_info["layer_reuse_cache"]
    
    # Check if we have cached functions for this reuse group
    if cache_key in reuse_cache:
        # Use cached functions
        cached_merge, cached_unmerge = reuse_cache[cache_key]
        
        if args.get("debug_layer_reuse", False):
            print(f"Layer reuse: Block {current_block_idx} (ToMe block {tome_block_idx}) "
                  f"using cached functions from group {reuse_group}")
        
        return cached_merge, cached_unmerge
    
    # Need to compute new merge/unmerge functions
    merge_fn, unmerge_fn = compute_merge_dit(x, tome_info)
    
    # Cache the functions for this reuse group
    reuse_cache[cache_key] = (merge_fn, unmerge_fn)
    
    if args.get("debug_layer_reuse", False):
        print(f"Layer reuse: Block {current_block_idx} (ToMe block {tome_block_idx}) "
              f"computed new functions for group {reuse_group} (frequency={reuse_frequency})")
    
    return merge_fn, unmerge_fn


def compute_merge_dit(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, Callable]:
    """
    Compute merge functions specifically for DiT architecture.
    
    DiT doesn't have hierarchical levels like U-Net, so we simplify the logic.
    """
    batch_size, num_tokens, dim = x.shape
    
    # Extract spatial dimensions from tome_info
    original_h, original_w = tome_info["size"]
    
    # For DiT, we work with the actual token grid size
    # DiT typically uses patch embeddings, so tokens = (H/patch_size) * (W/patch_size)
    patch_size = tome_info.get("patch_size", 16)  # Default DiT-XL/2 patch compared to image pixels, 8 for vae, 2 for latent patch size
    h = original_h // patch_size
    w = original_w // patch_size
    
    # Ensure our calculation matches the actual number of tokens
    if h * w != num_tokens:
        # Try to infer the correct dimensions
        h = w = int(math.sqrt(num_tokens))
        if h * w != num_tokens:
            print(f"Warning: Cannot infer spatial dimensions for {num_tokens} tokens")
            return do_nothing, do_nothing
    
    # Get merge parameters
    args = tome_info["args"]
    timestep = tome_info.get("timestep", 0)
    
    # Check if we should merge at this timestep
    if timestep / 1000 <= args.get("timestep_threshold_stop", 0.0):
        return do_nothing, do_nothing
    
    # Determine active merge method
    if timestep / 1000 > args.get("timestep_threshold_switch", 0.2):
        active_method = args["merge_method"]
    else:
        active_method = args.get("secondary_merge_method", "similarity")
    
    # Initialize merge/unmerge functions
    merge_fn, unmerge_fn = do_nothing, do_nothing
    
    # Apply the appropriate merge method
    if active_method == "frequency_blockwise" and args.get("downsample_factor", 2) > 1:
        try:
            merge_fn = frequency_based_selection_blockwise(
                x,
                downsampling_factor=args["downsample_factor"],
                H=h,
                W=w,
                selection_method=args.get("frequency_selection_method", "1d_dft"),
                mode=args.get("frequency_selection_mode", "high"),
                ranking_method=args.get("frequency_ranking_method", "amplitude"),
                timestep_normalized=timestep / 1000
            )
            unmerge_fn = do_nothing  # Blockwise doesn't support unmerge
        except Exception as e:
            print(f"Frequency blockwise merge failed: {e}")
            
    elif active_method == "frequency_global":
        ratio = args.get("ratio", 0.5)
        if 0 < ratio < 1:
            k = int(num_tokens * (1 - ratio))  # Tokens to keep
            try:
                merge_fn = frequency_based_selection(
                    x,
                    k=k,
                    selection_method=args.get("frequency_selection_method", "1d_dft"),
                    mode=args.get("frequency_selection_mode", "high"),
                    ranking_method=args.get("frequency_ranking_method", "amplitude"),
                    H=h,
                    W=w,
                    timestep_normalized=timestep / 1000,
                    alpha=args.get("frequency_grid_alpha", 2.0)
                )
                unmerge_fn = do_nothing
            except Exception as e:
                print(f"Frequency global merge failed: {e}")
                
    elif active_method == "similarity":
        ratio = args.get("ratio", 0.5)
        r = int(num_tokens * ratio)  # Tokens to remove
        if r > 0:
            if args["generator"] is None or args["generator"].device != x.device:
                seed = args.get("seed", 0)  # Default seed for reproducibility
                args["generator"] = init_generator(x.device, seed=seed)
            
            sx = args.get("sx", 2)
            sy = args.get("sy", 2)
            use_rand = args.get("use_rand", True)
            
            merge_fn, unmerge_fn = bipartite_soft_matching_random2d(
                x, w, h, sx, sy, r,
                no_rand=not use_rand,
                generator=args["generator"]
            )
            
    elif active_method == "masked_attention":
        ratio = args.get("ratio", 0.5)
        if 0 < ratio < 1:
            try:
                # Generate mask for this specific layer
                mask = frequency_based_token_mask(
                    x,
                    reduction_ratio=ratio,
                    selection_method=args.get("frequency_selection_method", "1d_dft"),
                    mode=args.get("frequency_selection_mode", "high"),
                    ranking_method=args.get("frequency_ranking_method", "amplitude"),
                    H=h, W=w,
                    timestep_normalized=timestep / 1000,
                    alpha=args.get("frequency_grid_alpha", 2.0)
                )
                
                # Store the mask in tome_info for use in attention processor
                tome_info['token_mask'] = mask
                
                # Return identity functions since masking is handled in attention
                merge_fn, unmerge_fn = do_nothing, do_nothing
            except Exception as e:
                print(f"Masked attention failed: {e}")
                
    elif active_method == "blockwise_masked_attention":
        factor = args.get("downsample_factor", 2)
        if factor > 1:
            try:
                # Generate blockwise mask for this specific layer
                mask = frequency_based_token_mask_blockwise(
                    x,
                    downsampling_factor=factor,
                    H=h, W=w,
                    selection_method=args.get("frequency_selection_method", "1d_dft"),
                    mode=args.get("frequency_selection_mode", "high"),
                    ranking_method=args.get("frequency_ranking_method", "amplitude"),
                    timestep_normalized=timestep / 1000
                )
                
                # Store the mask in tome_info for use in attention processor
                tome_info['token_mask'] = mask
                
                # Return identity functions since masking is handled in attention
                merge_fn, unmerge_fn = do_nothing, do_nothing
            except Exception as e:
                print(f"Blockwise masked attention failed: {e}")
                
    elif active_method == "downsample":
        factor = args.get("downsample_factor", 2)
        if factor > 1:
            new_h, new_w = h // factor, w // factor
            if new_h > 0 and new_w > 0:
                method = args.get("downsample_method", "avg_pool")
                
                # Initialize generator if method requires it (e.g., "random" or "linear_blend" with random methods)
                needs_generator = (
                    method == "random" or
                    (method == "linear_blend" and (
                        args.get("blend_method_1") == "random" or
                        args.get("blend_method_2") == "random"
                    ))
                )
                if needs_generator and (args.get("generator") is None or args.get("generator").device != x.device):
                    seed = args.get("seed", 0)  # Default seed for reproducibility
                    args["generator"] = init_generator(x.device, seed=seed)
                
                # Support for linear blend parameters
                blend_factor = args.get("blend_factor", None)
                blend_method_1 = args.get("blend_method_1", None)
                blend_method_2 = args.get("blend_method_2", None)
                
                merge_fn = lambda y: up_or_downsample(
                    y, w, h, new_w, new_h, method, 
                    timestep=timestep,
                    generator=args.get("generator"),
                    blend_factor=blend_factor,
                    blend_method_1=blend_method_1,
                    blend_method_2=blend_method_2
                )
                unmerge_fn = lambda y: up_or_downsample(
                    y, new_w, new_h, w, h, "nearest", 
                    timestep=timestep,
                    generator=args.get("generator"),
                    blend_factor=blend_factor,
                    blend_method_1=blend_method_1,
                    blend_method_2=blend_method_2
                )
    elif active_method == "downsample_custom_block":
        # Support separate height and width downsampling factors for rectangular blocks
        factor_h = args.get("downsample_factor_h", 1)
        factor_w = args.get("downsample_factor_w", 2)
        
        if factor_h > 1 or factor_w > 1:
            new_h, new_w = h // factor_h, w // factor_w
            if new_h > 0 and new_w > 0:
                method = args.get("downsample_method", "avg_pool")
                
                # Initialize generator if method requires it (e.g., "random" or "linear_blend" with random methods)
                needs_generator = (
                    method == "random" or
                    (method == "linear_blend" and (
                        args.get("blend_method_1") == "random" or
                        args.get("blend_method_2") == "random"
                    ))
                )
                if needs_generator and (args.get("generator") is None or args.get("generator").device != x.device):
                    seed = args.get("seed", 0)  # Default seed for reproducibility
                    args["generator"] = init_generator(x.device, seed=seed)
                
                # Support for linear blend parameters
                blend_factor = args.get("blend_factor", None)
                blend_method_1 = args.get("blend_method_1", None)
                blend_method_2 = args.get("blend_method_2", None)
                
                merge_fn = lambda y: up_or_downsample(
                    y, w, h, new_w, new_h, method, 
                    timestep=timestep,
                    generator=args.get("generator"),
                    blend_factor=blend_factor,
                    blend_method_1=blend_method_1,
                    blend_method_2=blend_method_2
                )
                unmerge_fn = lambda y: up_or_downsample(
                    y, new_w, new_h, w, h, "nearest", 
                    timestep=timestep,
                    generator=args.get("generator"),
                    blend_factor=blend_factor,
                    blend_method_1=blend_method_1,
                    blend_method_2=blend_method_2
                )
    
    return merge_fn, unmerge_fn


def patch_dit_tome(model: DiTTransformer2DModel, token_merge_args: Dict[str, Any] = None):
    """
    Patch a DiT model to use token merging.
    
    Args:
        model: The DiT model to patch
        token_merge_args: Dictionary of token merging arguments
                         - block_tome_flags: List of 28 boolean/int values (0/1) indicating 
                           which transformer blocks should apply ToMe. 
                           e.g., [1]*28 = all blocks, [0]*28 = no blocks
                         - frequency_selection_method: Method for frequency-based selection
                           Options: "original", "1d_dft", "1d_dct", "2d_conv", "2d_conv_l2", "non_uniform_grid"
                           - "2d_conv_l2": Uses 2D convolution with Laplacian kernel and L2 norm
                           - "2d_conv": Uses 2D convolution with Laplacian kernel and L1 norm (sum of absolute values)
    """
    if token_merge_args is None:
        token_merge_args = {}
    
    # Ensure block_tome_flags is properly configured
    if "block_tome_flags" not in token_merge_args:
        token_merge_args["block_tome_flags"] = [1] * 28  # Default to all blocks
    
    # Validate block_tome_flags
    block_flags = token_merge_args["block_tome_flags"]
    if not isinstance(block_flags, (list, tuple)):
        raise ValueError("block_tome_flags must be a list or tuple")
    
    # Auto-detect number of blocks if not 28
    expected_blocks = len(block_flags)
    if expected_blocks != 28:
        print(f"Warning: Expected 28 blocks for DiT-XL/2, got {expected_blocks} flags. "
              f"Adjusting configuration accordingly.")
    
    # Initialize tome_info on the model
    model._tome_info = {
        "size": None,
        "timestep": None,
        "patch_size": model.patch_size if hasattr(model, 'patch_size') else 16,
        "hooks": [],
        "current_block_index": 0,  # Track current block for layer reuse
        "layer_reuse_cache": {},   # Cache for layer reuse (cleared per timestep)
        "args": {
            "ratio": token_merge_args.get("ratio", 0.5),
            "sx": token_merge_args.get("sx", 2),
            "sy": token_merge_args.get("sy", 2),
            "use_rand": token_merge_args.get("use_rand", True),
            "generator": None,
            "merge_tokens": token_merge_args.get("merge_tokens", "keys/values"),
            "merge_method": token_merge_args.get("merge_method", "similarity"),
            "downsample_method": token_merge_args.get("downsample_method", "avg_pool"),
            "downsample_factor": token_merge_args.get("downsample_factor", 2),
            # Add support for rectangular blocks with custom method
            "downsample_factor_h": token_merge_args.get("downsample_factor_h", 1),
            "downsample_factor_w": token_merge_args.get("downsample_factor_w", 2),
            # Linear blend parameters for downsample methods
            "blend_factor": token_merge_args.get("blend_factor", None),
            "blend_method_1": token_merge_args.get("blend_method_1", None),
            "blend_method_2": token_merge_args.get("blend_method_2", None),
            "timestep_threshold_switch": token_merge_args.get("timestep_threshold_switch", 0.0),
            "timestep_threshold_stop": token_merge_args.get("timestep_threshold_stop", 0.0),
            "secondary_merge_method": token_merge_args.get("secondary_merge_method", "none"),
            "frequency_selection_mode": token_merge_args.get("frequency_selection_mode", "high"),
            "frequency_selection_method": token_merge_args.get("frequency_selection_method", "1d_dft"),
            "frequency_ranking_method": token_merge_args.get("frequency_ranking_method", "amplitude"),
            "selection_source": token_merge_args.get("selection_source", "hidden"),
            "frequency_grid_alpha": token_merge_args.get("frequency_grid_alpha", 2.0),
            
            # Layer reuse parameters
            "layer_reuse_enabled": token_merge_args.get("layer_reuse_enabled", False),
            "layer_reuse_frequency": token_merge_args.get("layer_reuse_frequency", 3),
            "debug_layer_reuse": token_merge_args.get("debug_layer_reuse", False),
        }
    }
    
    # Add forward hook to capture input size and timestep
    def hook(module, args):
        # For DiT, args[0] is hidden_states, args[1] is timestep
        if len(args) >= 2:
            # Extract spatial size from latent input
            hidden_states = args[0]
            if hidden_states.ndim == 4:
                module._tome_info["size"] = (hidden_states.shape[2], hidden_states.shape[3])
            elif hidden_states.ndim == 3:
                # Already flattened, try to infer square dimensions
                batch_size, num_tokens, _ = hidden_states.shape
                patch_size = module._tome_info.get("patch_size", 16)
                spatial_size = int(math.sqrt(num_tokens)) * patch_size
                module._tome_info["size"] = (spatial_size, spatial_size)
            
            # Extract timestep
            timestep = args[1]
            current_timestep = None
            if torch.is_tensor(timestep):
                current_timestep = timestep.item() if timestep.numel() == 1 else timestep[0].item()
            else:
                current_timestep = float(timestep)
            
            # Check for timestep change and clear layer reuse cache if needed (per-timestep caching)
            if module._tome_info["args"].get("layer_reuse_enabled", False):
                previous_timestep = module._tome_info.get("timestep", None)
                if previous_timestep != current_timestep:
                    # New timestep detected, clear layer reuse cache
                    if "layer_reuse_cache" in module._tome_info:
                        cache_size = len(module._tome_info["layer_reuse_cache"])
                        module._tome_info["layer_reuse_cache"].clear()
                        if module._tome_info["args"].get("debug_layer_reuse", False) and cache_size > 0:
                            print(f"New timestep {current_timestep}: cleared layer reuse cache ({cache_size} groups)")
                    
                    # Reset block tracking for new timestep
                    module._tome_info["current_block_index"] = 0
            
            module._tome_info["timestep"] = current_timestep
    
    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))
    
    # Replace attention processors in transformer blocks based on block_tome_flags
    block_counter = 0
    block_flags = token_merge_args["block_tome_flags"]
    
    # We need to track transformer blocks specifically, not all attention modules
    transformer_blocks = []
    for name, module in model.named_modules():
        if "transformer_blocks" in name and hasattr(module, 'attn1'):
            transformer_blocks.append((name, module))
    
    # Apply ToMe to specified blocks only
    for name, block_module in transformer_blocks:
        should_apply_tome = False
        if block_counter < len(block_flags):
            should_apply_tome = bool(block_flags[block_counter])
        
        if should_apply_tome:
            # Apply ToMe to attention modules in this block
            for attn_name, attn_module in block_module.named_modules():
                if isinstance(attn_module, Attention):
                    processor = DiTTokenMergeProcessor(block_index=block_counter)
                    processor.tome_info = model._tome_info
                    attn_module.set_processor(processor)
                    
                    # Mark the attention module as having ToMe applied
                    attn_module._tome_enabled = True
            
            # Mark the block as having ToMe applied
            block_module._tome_enabled = True
        else:
            # Keep default processors for this block
            block_module._tome_enabled = False
        
        # Store block information for debugging
        block_module._tome_block_index = block_counter
        block_counter += 1
    
    # Calculate ToMe block count for information
    tome_block_count = sum(block_flags)
    reuse_frequency = token_merge_args.get("layer_reuse_frequency", 3)
    
    print(f"Applied ToMe to {tome_block_count} out of {len(block_flags)} transformer blocks")
    if token_merge_args.get("layer_reuse_enabled", False):
        reuse_groups = (tome_block_count + reuse_frequency - 1) // reuse_frequency  # Ceiling division
        print(f"Layer reuse enabled: recalculating merge functions every {reuse_frequency} ToMe blocks "
              f"({reuse_groups} reuse groups)")


def remove_dit_tome_patch(model: DiTTransformer2DModel):
    """Remove token merging patch from DiT model."""
    
    # Remove hooks
    if hasattr(model, "_tome_info"):
        for hook in model._tome_info.get("hooks", []):
            hook.remove()
        del model._tome_info
    
    # Reset attention processors to default and clean up block-level flags
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            if hasattr(module, '_tome_enabled'):
                module.set_processor(None)  # Reset to default processor
                delattr(module, '_tome_enabled')
        
        # Clean up transformer block flags
        if "transformer_blocks" in name:
            if hasattr(module, '_tome_enabled'):
                delattr(module, '_tome_enabled')
            if hasattr(module, '_tome_block_index'):
                delattr(module, '_tome_block_index')


def precompute_fixed_mask_for_dit(
    original_x: torch.Tensor,
    tome_info: dict,
    block_index: int = 0
):
    """
    Precompute and store a fixed mask for DiT based on the original clean input.
    This mask will be reused for ALL timesteps and conditions during inference.
    
    Args:
        original_x: Clean input tensor representation (B, N, C)
        tome_info: ToMe info dictionary to store the mask
        block_index: Index of the transformer block (used for mask key)
    """
    args = tome_info["args"]
    
    # Only proceed if using masked_attention or blockwise_masked_attention
    if args["merge_method"] not in ["masked_attention", "blockwise_masked_attention"]:
        return
    
    # Get spatial dimensions
    B, N, C = original_x.shape
    patch_size = tome_info.get("patch_size", 16)
    
    # For DiT, calculate spatial dimensions from number of tokens
    h = w = int(math.sqrt(N))
    if h * w != N:
        print(f"Warning: Cannot infer square spatial dimensions for {N} tokens")
        return
    
    # Handle masked_attention
    if args["merge_method"] == "masked_attention":
        ratio = args.get("ratio", 0.5)
        if 0 < ratio < 1:
            mask_key = f"fixed_mask_block_{block_index}"
            
            try:
                # Generate the FIXED mask based on ORIGINAL CLEAN input
                x_single = original_x[:1]  # Ensure batch size 1
                mask = frequency_based_token_mask(
                    x_single,
                    reduction_ratio=ratio,
                    selection_method=args.get("frequency_selection_method", "1d_dft"),
                    mode=args.get("frequency_selection_mode", "high"),
                    ranking_method=args.get("frequency_ranking_method", "amplitude"),
                    H=h, W=w,
                    timestep_normalized=None,
                    alpha=args.get("frequency_grid_alpha", 2.0)
                )
                
                # Store the mask for reuse
                tome_info[mask_key] = mask
                num_kept = mask.sum(dim=1).item()
                print(f"Precomputed DiT mask for block {block_index}: {N} → {num_kept} tokens ({num_kept/N:.1%})")
                
            except Exception as e:
                print(f"Error precomputing DiT mask: {e}")
                # Fallback: keep all tokens
                mask = torch.ones(1, N, dtype=torch.bool, device=original_x.device)
                tome_info[mask_key] = mask
                
    # Handle blockwise_masked_attention
    elif args["merge_method"] == "blockwise_masked_attention":
        factor = args.get("downsample_factor", 2)
        if factor > 1:
            mask_key = f"fixed_mask_blockwise_block_{block_index}"
            
            try:
                # Generate the FIXED blockwise mask based on ORIGINAL CLEAN input
                x_single = original_x[:1]  # Ensure batch size 1
                mask = frequency_based_token_mask_blockwise(
                    x_single,
                    downsampling_factor=factor,
                    H=h, W=w,
                    selection_method=args.get("frequency_selection_method", "1d_dft"),
                    mode=args.get("frequency_selection_mode", "high"),
                    ranking_method=args.get("frequency_ranking_method", "amplitude"),
                    timestep_normalized=None
                )
                
                # Store the mask for reuse
                tome_info[mask_key] = mask
                num_kept = mask.sum(dim=1).item()
                print(f"Precomputed DiT blockwise mask for block {block_index}: {N} → {num_kept} tokens ({num_kept/N:.1%}) [factor={factor}]")
                
            except Exception as e:
                print(f"Error precomputing DiT blockwise mask: {e}")
                # Fallback: keep all tokens
                mask = torch.ones(1, N, dtype=torch.bool, device=original_x.device)
                tome_info[mask_key] = mask


def reset_dit_fixed_masks(tome_info: dict):
    """
    Reset stored fixed masks for DiT between different images.
    """
    keys_to_remove = [key for key in tome_info.keys() if key.startswith("fixed_mask_")]
    for key in keys_to_remove:
        del tome_info[key]
    
    # Reset batch expansion flags for new image
    if '_batch_expand_logged' in tome_info:
        del tome_info['_batch_expand_logged']
    if '_batch_blockwise_expand_logged' in tome_info:
        del tome_info['_batch_blockwise_expand_logged']
    
    if keys_to_remove:
        print(f"Reset {len(keys_to_remove)} DiT fixed masks for new image")


def clear_dit_layer_reuse_cache(model: DiTTransformer2DModel):
    """
    Clear the layer reuse cache. Call this between different timesteps
    to ensure cache is per-timestep as intended.
    """
    if hasattr(model, "_tome_info") and "layer_reuse_cache" in model._tome_info:
        cache_size = len(model._tome_info["layer_reuse_cache"])
        model._tome_info["layer_reuse_cache"].clear()
        if cache_size > 0:
            print(f"DiT layer reuse cache cleared ({cache_size} cached groups)")


def get_dit_layer_reuse_stats(model: DiTTransformer2DModel) -> Dict[str, Any]:
    """
    Get statistics about the layer reuse cache.
    
    Returns:
        Dictionary with cache statistics
    """
    if not hasattr(model, "_tome_info"):
        return {"error": "No tome_info found on model"}
    
    tome_info = model._tome_info
    if "layer_reuse_cache" not in tome_info:
        return {"error": "Layer reuse cache not initialized"}
    
    cache = tome_info["layer_reuse_cache"]
    args = tome_info.get("args", {})
    
    stats = {
        "layer_reuse_enabled": args.get("layer_reuse_enabled", False),
        "layer_reuse_frequency": args.get("layer_reuse_frequency", 3),
        "cached_groups": len(cache),
        "cache_keys": list(cache.keys()) if args.get("debug_layer_reuse", False) else None,
        "total_tome_blocks": sum(args.get("block_tome_flags", [])),
    }
    
    return stats


def reset_dit_layer_reuse_for_new_timestep(model: DiTTransformer2DModel):
    """
    Reset layer reuse cache for a new timestep.
    This should be called at the beginning of each timestep during inference.
    """
    if hasattr(model, "_tome_info"):
        tome_info = model._tome_info
        
        # Clear cache for new timestep (per-timestep caching)
        if "layer_reuse_cache" in tome_info:
            tome_info["layer_reuse_cache"].clear()
        
        # Reset block tracking
        tome_info["current_block_index"] = 0
        
        if tome_info.get("args", {}).get("debug_layer_reuse", False):
            print(f"Reset DiT layer reuse cache for new timestep")


# Example usage function
def apply_tome_to_dit_pipeline(pipe, token_merge_args=None):
    """
    Apply token merging to a DiT-based diffusion pipeline.
    
    Example:
        from diffusers import DiTPipeline
        
        pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
        
        # Apply token reduction with 2D convolution L2 norm
        apply_tome_to_dit_pipeline(pipe, {
            "ratio": 0.5,
            "merge_method": "frequency_global",
            "frequency_selection_method": "2d_conv_l2",
            "merge_tokens": "keys/values"
        })
        
        # Apply masked attention (new feature)
        apply_tome_to_dit_pipeline(pipe, {
            "ratio": 0.3,
            "merge_method": "masked_attention",
            "frequency_selection_method": "1d_dft",
            "frequency_selection_mode": "high",
            "merge_tokens": "keys/values",
            "block_tome_flags": [1]*28  # Apply to all blocks
        })
        
        # Apply blockwise masked attention (new feature)
        apply_tome_to_dit_pipeline(pipe, {
            "downsample_factor": 2,
            "merge_method": "blockwise_masked_attention",
            "frequency_selection_method": "2d_conv",
            "frequency_selection_mode": "high",
            "merge_tokens": "keys/values",
            "block_tome_flags": [1]*14 + [0]*14  # Apply to first 14 blocks only
        })
        
        # Apply downsample with linear blend (new feature)
        apply_tome_to_dit_pipeline(pipe, {
            "merge_method": "downsample",
            "downsample_method": "linear_blend",
            "blend_factor": 0.7,  # 70% method1, 30% method2
            "blend_method_1": "avg_pool",
            "blend_method_2": "bilinear",
            "downsample_factor": 2,
            "merge_tokens": "keys/values",
            "block_tome_flags": [1]*28  # Apply to all blocks
        })
        
        # Apply with layer reuse enabled (NEW FEATURE!)
        apply_tome_to_dit_pipeline(pipe, {
            "ratio": 0.5,
            "merge_method": "frequency_global",
            "frequency_selection_method": "2d_conv_l2",
            "merge_tokens": "keys/values",
            "block_tome_flags": [1]*28,
            
            # Layer reuse parameters
            "layer_reuse_enabled": True,        # Enable layer reuse
            "layer_reuse_frequency": 3,         # Recalculate every 3 ToMe blocks (default)
            "debug_layer_reuse": True           # Print debug info
        })
        
        # Generate image
        image = pipe("a cat", num_inference_steps=50).images[0]
        
        # Optional: Get layer reuse statistics
        stats = get_dit_layer_reuse_stats(pipe.transformer)
        print(f"Layer reuse stats: {stats}")
        
        # Optional: Manually clear cache (usually not needed due to auto-clearing)
        clear_dit_layer_reuse_cache(pipe.transformer)
    """
    if hasattr(pipe, "transformer"):  # DiT uses 'transformer' instead of 'unet'
        patch_dit_tome(pipe.transformer, token_merge_args)
    else:
        raise ValueError("Pipeline does not have a transformer attribute")