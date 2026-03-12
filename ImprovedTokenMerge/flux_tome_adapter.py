import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Dict, Any, List
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxAttention
from diffusers.models.attention_processor import AttentionProcessor
import math
import torch.nn.functional as F

# Import existing merge functions
try:
    # Try relative import first (when used as package)
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
except ImportError:
    # Fall back to absolute import (when run directly)
    from merge import (
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


class FluxTokenMergeProcessor:
    """Custom attention processor for FLUX that implements token merging on IMAGE tokens only."""
    
    def __init__(self, block_index=None, block_type="double"):
        self.tome_info = None
        self.block_index = block_index  # Index of the transformer block this processor belongs to
        self.block_type = block_type    # "double" or "single"
        
    def __call__(
        self,
        attn: FluxAttention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if self.tome_info is None:
            # Fallback to standard attention if no tome_info
            return attn.processor(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)
            
        # Update current block index for layer reuse tracking
        if self.block_index is not None:
            self.tome_info['current_block_index'] = self.block_index
            
        # Track initial token counts for internal logic
        initial_image_tokens = hidden_states.shape[1]
        initial_text_tokens = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            
        # Get projections (similar to original FluxAttnProcessor)
        from diffusers.models.transformers.transformer_flux import _get_qkv_projections
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Handle encoder (text) tokens if present
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # Concatenate text and image tokens (text first, as in original FLUX)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Apply rotary embeddings BEFORE token merging
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # ========================================
        # Apply token merging AFTER rotary embeddings
        # ========================================
        if self.tome_info['args']['merge_tokens'] in ["all", "keys/values"]:
            # Get text sequence length to know where to split
            text_seq_len = 0
            if encoder_hidden_states is not None:
                text_seq_len = encoder_hidden_states.shape[1]
            elif self.block_type == "single" and self.tome_info.get('text_seq_len'):
                text_seq_len = self.tome_info['text_seq_len']
            
            # Determine which tokens to use for merge computation (use hidden_states for spatial info)
            merge_input = hidden_states
            if self.block_type == "single" and text_seq_len > 0:
                # For single blocks, use only the image part of hidden_states
                merge_input = hidden_states[:, text_seq_len:]
            
            # Compute merge function
            image_merge_fn, image_unmerge_fn = compute_merge_flux_image_only(
                merge_input, self.tome_info, 0  # Always 0 since we extract image tokens
            )
            
            # Apply merging if available
            if image_merge_fn != do_nothing:
                merge_mode = self.tome_info['args']['merge_tokens']
                
                # Apply merging to the appropriate tokens
                if self.tome_info['args']['merge_tokens'] == "all":
                    # Merge queries only 
                    query_before = query.shape[1]
                    
                    if text_seq_len > 0:
                        # Split query into text and image parts
                        text_query, image_query = query[:, :text_seq_len], query[:, text_seq_len:]
                        # Apply merge to image part only
                        query_reshaped = image_query.flatten(2, 3)  # [batch, seq_len, heads*head_dim]
                        query_merged = image_merge_fn(query_reshaped)
                        image_query = query_merged.unflatten(-1, (attn.heads, -1))
                        # Recombine
                        query = torch.cat([text_query, image_query], dim=1)
                    else:
                        # All tokens are image tokens
                        query_reshaped = query.flatten(2, 3)
                        query_merged = image_merge_fn(query_reshaped)
                        query = query_merged.unflatten(-1, (attn.heads, -1))
                    
                    query_after = query.shape[1]
                
                elif self.tome_info['args']['merge_tokens'] == "keys/values":
                    # Merge keys and values
                    key_before = key.shape[1]
                    value_before = value.shape[1]
                    
                    if text_seq_len > 0:
                        # Split key/value into text and image parts
                        text_key, image_key = key[:, :text_seq_len], key[:, text_seq_len:]
                        text_value, image_value = value[:, :text_seq_len], value[:, text_seq_len:]
                        
                        # Apply merge to image parts only
                        key_reshaped = image_key.flatten(2, 3)
                        value_reshaped = image_value.flatten(2, 3)
                        key_merged = image_merge_fn(key_reshaped)
                        value_merged = image_merge_fn(value_reshaped)
                        image_key = key_merged.unflatten(-1, (attn.heads, -1))
                        image_value = value_merged.unflatten(-1, (attn.heads, -1))
                        
                        # Recombine
                        key = torch.cat([text_key, image_key], dim=1)
                        value = torch.cat([text_value, image_value], dim=1)
                    else:
                        # All tokens are image tokens
                        key_reshaped = key.flatten(2, 3)
                        value_reshaped = value.flatten(2, 3)
                        key_merged = image_merge_fn(key_reshaped)
                        value_merged = image_merge_fn(value_reshaped)
                        key = key_merged.unflatten(-1, (attn.heads, -1))
                        value = value_merged.unflatten(-1, (attn.heads, -1))
                    
                    key_after = key.shape[1]
                    value_after = value.shape[1]

        # Scaled dot product attention
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, backend=getattr(attn.processor, '_attention_backend', None)
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Calculate final token counts for output processing
        final_total_tokens = hidden_states.shape[1]
        
        # Handle output splitting and projection
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            # Split back into text and image components
            # Note: text length remains the same, but image length may have changed due to merging
            text_len = encoder_hidden_states.shape[1]
            image_len = hidden_states.shape[1] - text_len
            
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [text_len, image_len], dim=1
            )
            
            # Process final token counts for double blocks
            final_image_tokens = hidden_states.shape[1]
            final_text_tokens = encoder_hidden_states.shape[1]
            
            # Apply output projections
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            # Single block case - no additional processing needed
            return hidden_states


def compute_merge_flux_image_only(x: torch.Tensor, tome_info: Dict[str, Any], text_seq_len: int) -> Tuple[Callable, Callable]:
    """
    Compute merge functions specifically for FLUX image tokens only.
    
    Args:
        x: Input tensor containing image tokens (for double blocks) or combined tokens (for single blocks)
        tome_info: Token merge info dictionary  
        text_seq_len: Length of text sequence (only used for single blocks)
    """
    batch_size, total_tokens, dim = x.shape
    
    # For FLUX: 
    # - Double blocks: x contains ONLY image tokens, text_seq_len = 0 (since encoder_hidden_states exists separately)
    # - Single blocks: x contains text + image tokens concatenated, text_seq_len > 0
    if text_seq_len > 0:
        # Single block case: subtract text tokens
        image_tokens = total_tokens - text_seq_len
        if image_tokens <= 0:
            print(f"Warning: No image tokens found (total: {total_tokens}, text: {text_seq_len})")
            return do_nothing, do_nothing
    else:
        # Double block case: x already contains only image tokens
        image_tokens = total_tokens
    
    # For FLUX, calculate spatial dimensions from image tokens
    # FLUX uses patch embeddings: tokens = (H/patch_size) * (W/patch_size)
    
    # First try perfect square
    h = w = int(math.sqrt(image_tokens))
    if h * w != image_tokens:
        print(f"Warning: Cannot infer square spatial dimensions for {image_tokens} image tokens")
        return do_nothing, do_nothing
    
    # Extract spatial dimensions from tome_info (for reference, but we use calculated h, w)
    if tome_info["size"] is None:
        # Store the inferred size back to tome_info for future use  
        # FLUX total downsampling: 16× (VAE: 16×, patch_size: 1×)
        tome_info["size"] = (h * 16, h * 16)  # Store as pixel space size for consistency
        original_h, original_w = h * 16, h * 16
    else:
        original_h, original_w = tome_info["size"]
    
    # Get merge parameters
    args = tome_info["args"]
    timestep = tome_info.get("timestep", 0)
    
    # Handle None timestep and normalize
    if timestep is None:
        timestep_normalized = 0.0
    else:
        # FLUX pipeline already normalizes timestep by /1000, so it's in [0, 1] range
        # where 1.0 = start/noisy, 0.0 = end/clean
        if torch.is_tensor(timestep):
            timestep_normalized = float(timestep.item() if timestep.numel() == 1 else timestep.mean().item())
        else:
            timestep_normalized = float(timestep)
    threshold_stop = args.get("timestep_threshold_stop", 0.1)
    threshold_switch = args.get("timestep_threshold_switch", 0.5)
    
    if timestep_normalized <= threshold_stop:
        return do_nothing, do_nothing
    
    # Determine active merge method based on timestep
    # FLUX timesteps: 1.0 (start/noisy) → 0.0 (end/clean)
    # Strategy: Use aggressive merging at start, reduce/stop near end
    if timestep_normalized > threshold_switch:
        # High timestep (early in diffusion): use primary method (aggressive merging)
        active_method = args["merge_method"]
    else:
        # Low timestep (late in diffusion): use secondary method (reduced/no merging)
        secondary_method = args.get("secondary_merge_method", "none")
        if secondary_method == "none":
            # If secondary is explicitly "none", skip merging at low timesteps
            return do_nothing, do_nothing
        else:
            active_method = secondary_method
    
    # Extract image tokens for merge computation
    if text_seq_len > 0:
        image_x = x[:, text_seq_len:]  # Extract only image tokens
    else:
        image_x = x  # Already image-only
    
    # Initialize merge/unmerge functions
    merge_fn, unmerge_fn = do_nothing, do_nothing
    
    # Apply the appropriate merge method to IMAGE tokens only
    if active_method == "frequency_blockwise" and args.get("downsample_factor", 2) > 1:
        try:
            image_merge_base_fn = frequency_based_selection_blockwise(
                image_x,
                downsampling_factor=args["downsample_factor"],
                H=h,
                W=w,
                selection_method=args.get("frequency_selection_method", "1d_dft"),
                mode=args.get("frequency_selection_mode", "high"),
                ranking_method=args.get("frequency_ranking_method", "amplitude"),
                timestep_normalized=timestep / 1000
            )
            
            # Wrap to handle text/image separation
            merge_fn = create_image_only_merge_wrapper(image_merge_base_fn, text_seq_len)
            unmerge_fn = do_nothing  # Blockwise doesn't support unmerge
        except Exception as e:
            pass  # Silently handle failure
            
    elif active_method == "frequency_global":
        ratio = args.get("ratio", 0.5)
        if 0 < ratio < 1:
            k = int(image_tokens * (1 - ratio))  # Image tokens to keep
            
            try:
                image_merge_base_fn = frequency_based_selection(
                    image_x,
                    k=k,
                    selection_method=args.get("frequency_selection_method", "1d_dft"),
                    mode=args.get("frequency_selection_mode", "high"),
                    ranking_method=args.get("frequency_ranking_method", "amplitude"),
                    H=h,
                    W=w,
                    timestep_normalized=timestep / 1000,
                    alpha=args.get("frequency_grid_alpha", 2.0)
                )
                
                merge_fn = create_image_only_merge_wrapper(image_merge_base_fn, text_seq_len)
                unmerge_fn = do_nothing
            except Exception as e:
                pass  # Silently handle failure
                
    elif active_method == "similarity":
        ratio = args.get("ratio", 0.5)
        r = int(image_tokens * ratio)  # Image tokens to remove
        
        if r > 0:
            if args["generator"] is None or args["generator"].device != image_x.device:
                seed = args.get("seed", 0)
                args["generator"] = init_generator(image_x.device, seed=seed)
            
            sx = args.get("sx", 2)
            sy = args.get("sy", 2)
            use_rand = args.get("use_rand", True)
            
            image_merge_base_fn, image_unmerge_base_fn = bipartite_soft_matching_random2d(
                image_x, w, h, sx, sy, r,
                no_rand=not use_rand,
                generator=args["generator"]
            )
            
            merge_fn = create_image_only_merge_wrapper(image_merge_base_fn, text_seq_len)
            unmerge_fn = create_image_only_merge_wrapper(image_unmerge_base_fn, text_seq_len)
            
    elif active_method == "downsample":
        factor = args.get("downsample_factor", 2)
        if factor > 1:
            new_h, new_w = h // factor, w // factor
            if new_h > 0 and new_w > 0:
                method = args.get("downsample_method", "avg_pool")
                
                needs_generator = (
                    method == "random" or
                    (method == "linear_blend" and (
                        args.get("blend_method_1") == "random" or
                        args.get("blend_method_2") == "random"
                    ))
                )
                if needs_generator and (args.get("generator") is None or args.get("generator").device != image_x.device):
                    seed = args.get("seed", 0)
                    args["generator"] = init_generator(image_x.device, seed=seed)
                
                blend_factor = args.get("blend_factor", None)
                blend_method_1 = args.get("blend_method_1", None)
                blend_method_2 = args.get("blend_method_2", None)
                
                image_merge_base_fn = lambda y: up_or_downsample(
                    y, w, h, new_w, new_h, method,
                    timestep=timestep,
                    generator=args.get("generator"),
                    blend_factor=blend_factor,
                    blend_method_1=blend_method_1,
                    blend_method_2=blend_method_2
                )
                image_unmerge_base_fn = lambda y: up_or_downsample(
                    y, new_w, new_h, w, h, "nearest",
                    timestep=timestep,
                    generator=args.get("generator"),
                    blend_factor=blend_factor,
                    blend_method_1=blend_method_1,
                    blend_method_2=blend_method_2
                )
                
                merge_fn = create_image_only_merge_wrapper(image_merge_base_fn, text_seq_len)
                unmerge_fn = create_image_only_merge_wrapper(image_unmerge_base_fn, text_seq_len)
    # For unsupported methods, merge_fn and unmerge_fn remain as do_nothing
    
    return merge_fn, unmerge_fn


def create_image_only_merge_wrapper(base_merge_fn: Callable, text_seq_len: int) -> Callable:
    """
    Create a wrapper function that applies merging only to image tokens,
    leaving text tokens unchanged.
    
    Note: This wrapper expects 3D tensors [batch, seq_len, dim] as input,
    which is what the attention processor provides after flattening 4D tensors.
    
    Args:
        base_merge_fn: The base merge function to apply to image tokens
        text_seq_len: Number of text tokens to preserve
    """
    def wrapped_merge_fn(x: torch.Tensor) -> torch.Tensor:
        if text_seq_len == 0:
            # No text tokens, apply merge to entire tensor
            return base_merge_fn(x)
        
        if x.shape[1] <= text_seq_len:
            # Only text tokens, no merging
            return x
        
        # Split into text and image tokens
        text_tokens = x[:, :text_seq_len]
        image_tokens = x[:, text_seq_len:]
        
        # Apply merge function only to image tokens
        merged_image_tokens = base_merge_fn(image_tokens)
        
        # Recombine text and merged image tokens
        return torch.cat([text_tokens, merged_image_tokens], dim=1)
    
    return wrapped_merge_fn


def patch_flux_tome(model: FluxTransformer2DModel, token_merge_args: Dict[str, Any] = None):
    """
    Patch a FLUX model to use token merging on IMAGE tokens only.
    
    Args:
        model: The FLUX model to patch
        token_merge_args: Dictionary of token merging arguments
                         - block_tome_flags_double: List indicating which double blocks should apply ToMe
                         - block_tome_flags_single: List indicating which single blocks should apply ToMe
                         - Other parameters similar to DiT adapter
    """
    if token_merge_args is None:
        token_merge_args = {}
    
    # Get the number of blocks from the model
    num_double_blocks = len(model.transformer_blocks)
    num_single_blocks = len(model.single_transformer_blocks)
    
    # Initialize block flags
    if "block_tome_flags_double" not in token_merge_args:
        token_merge_args["block_tome_flags_double"] = [1] * num_double_blocks  # Default: all double blocks
    if "block_tome_flags_single" not in token_merge_args:
        token_merge_args["block_tome_flags_single"] = [1] * num_single_blocks  # Default: all single blocks
    
    double_flags = token_merge_args["block_tome_flags_double"]
    single_flags = token_merge_args["block_tome_flags_single"]
    
    # Validate flags
    if len(double_flags) != num_double_blocks:
        print(f"Warning: Expected {num_double_blocks} double block flags, got {len(double_flags)}")
    if len(single_flags) != num_single_blocks:
        print(f"Warning: Expected {num_single_blocks} single block flags, got {len(single_flags)}")
    
    # Initialize tome_info on the model
    model._tome_info = {
        "size": None,
        "timestep": None,
        "hooks": [],
        "text_seq_len": None,  # Will be set during forward pass
        "current_block_index": 0,
        "token_reduction_stats": {  # Track token reduction statistics
            "total_blocks": 0,
            "blocks_with_reduction": 0,
            "max_reduction_pct": 0.0,
            "avg_reduction_pct": 0.0,
        },
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
            "blend_factor": token_merge_args.get("blend_factor", None),
            "blend_method_1": token_merge_args.get("blend_method_1", None),
            "blend_method_2": token_merge_args.get("blend_method_2", None),
            "timestep_threshold_switch": token_merge_args.get("timestep_threshold_switch", 0.5),  # Switch from primary to secondary method at mid-point
            "timestep_threshold_stop": token_merge_args.get("timestep_threshold_stop", 0.1),   # Stop merging near end to preserve details
            "secondary_merge_method": token_merge_args.get("secondary_merge_method", "none"),  # Default to none for secondary
            "frequency_selection_mode": token_merge_args.get("frequency_selection_mode", "high"),
            "frequency_selection_method": token_merge_args.get("frequency_selection_method", "1d_dft"),
            "frequency_ranking_method": token_merge_args.get("frequency_ranking_method", "amplitude"),
            "selection_source": token_merge_args.get("selection_source", "hidden"),
            "frequency_grid_alpha": token_merge_args.get("frequency_grid_alpha", 2.0),
            "seed": token_merge_args.get("seed", 0),
        }
    }
    
    # Add forward hook to capture input size, timestep, and text sequence length
    def hook(module, args, kwargs):
        # Handle both positional and keyword arguments for FLUX transformer
        hidden_states = None
        encoder_hidden_states = None
        timestep = None
        
        # Try to extract from keyword arguments first (common in FLUX pipeline)
        if kwargs is not None:
            hidden_states = kwargs.get('hidden_states')
            encoder_hidden_states = kwargs.get('encoder_hidden_states')
            timestep = kwargs.get('timestep')
            
            if timestep is not None:
                pass  # Timestep found in kwargs
        
        # Fallback to positional arguments if needed
        # FLUX forward signature: forward(hidden_states, encoder_hidden_states, pooled_projections, timestep, ...)
        if len(args) >= 1 and hidden_states is None:
            hidden_states = args[0]
        if len(args) >= 2 and encoder_hidden_states is None:
            encoder_hidden_states = args[1]
        if len(args) >= 4 and timestep is None:
            timestep = args[3]  # timestep is typically the 4th argument
            if timestep is not None:
                pass  # Timestep found in args
        
        # If timestep still not found, try to find it in the remaining args
        if timestep is None and len(args) > 2:
            for i, arg in enumerate(args[2:], 2):  # Start from position 2
                if torch.is_tensor(arg) and arg.numel() == 1:
                    # Check if this could be a timestep (FLUX uses range [0, 1] after /1000)
                    val = arg.item()
                    if 0 <= val <= 1.1:  # After /1000, should be in [0, 1] range
                        timestep = arg
                        break
                elif isinstance(arg, (int, float)) and 0 <= arg <= 1.1:
                    timestep = arg
                    break
        
        # Extract spatial size from image latent input
        if hidden_states is not None and hidden_states.ndim == 3:
            batch_size, num_image_tokens, _ = hidden_states.shape
            # Infer spatial size from number of image tokens
            h = w = int(math.sqrt(num_image_tokens))
            # For FLUX, we need to convert back to pixel space
            # FLUX VAE downsamples by 8x, and patch size is typically 1 for latents
            spatial_size = h * 8  # Convert latent size to pixel size
            module._tome_info["size"] = (spatial_size, spatial_size)
        
        # Store text sequence length 
        if encoder_hidden_states is not None:
            module._tome_info["text_seq_len"] = encoder_hidden_states.shape[1]
        # For single blocks, try to infer text seq len from hidden_states if needed
        elif hidden_states is not None and module._tome_info["text_seq_len"] is None:
            # Single blocks receive concatenated tokens, try to infer standard FLUX text length
            # FLUX typically uses 512 tokens for text in most cases
            total_tokens = hidden_states.shape[1] 
            # Common FLUX configurations: 4096 image + 512 text = 4608 total
            if total_tokens == 4608:  # 1024x1024 image (64x64 patches) + 512 text
                inferred_text_len = 512
                module._tome_info["text_seq_len"] = inferred_text_len
            elif total_tokens == 1792:  # 512x512 image (32x32 patches) + 512 text  
                inferred_text_len = 512
                module._tome_info["text_seq_len"] = inferred_text_len
        
        # Extract timestep
        if timestep is not None:
            current_timestep = None
            if torch.is_tensor(timestep):
                current_timestep = timestep.item() if timestep.numel() == 1 else timestep[0].item()
            else:
                current_timestep = float(timestep)
            
            module._tome_info["timestep"] = current_timestep
    
    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook, with_kwargs=True))
    
    # Apply ToMe to double transformer blocks
    tome_double_count = 0
    for i, block in enumerate(model.transformer_blocks):
        should_apply = i < len(double_flags) and bool(double_flags[i])
        
        if should_apply:
            if hasattr(block, 'attn') and isinstance(block.attn, FluxAttention):
                processor = FluxTokenMergeProcessor(block_index=i, block_type="double")
                processor.tome_info = model._tome_info
                block.attn.set_processor(processor)
                block._tome_enabled = True
                tome_double_count += 1
        else:
            block._tome_enabled = False
        
        block._tome_block_index = i
    
    # Apply ToMe to single transformer blocks
    tome_single_count = 0
    for i, block in enumerate(model.single_transformer_blocks):
        should_apply = i < len(single_flags) and bool(single_flags[i])
        
        if should_apply:
            if hasattr(block, 'attn') and isinstance(block.attn, FluxAttention):
                processor = FluxTokenMergeProcessor(block_index=i, block_type="single")
                processor.tome_info = model._tome_info
                block.attn.set_processor(processor)
                block._tome_enabled = True
                tome_single_count += 1
        else:
            block._tome_enabled = False
        
        block._tome_block_index = i
    
    # FLUX ToMe configuration applied silently


def remove_flux_tome_patch(model: FluxTransformer2DModel):
    """Remove token merging patch from FLUX model."""
    
    # Remove hooks
    if hasattr(model, "_tome_info"):
        for hook in model._tome_info.get("hooks", []):
            hook.remove()
        del model._tome_info
    
    # Reset attention processors to default
    for block in model.transformer_blocks + model.single_transformer_blocks:
        if hasattr(block, 'attn') and isinstance(block.attn, FluxAttention):
            if hasattr(block, '_tome_enabled'):
                block.attn.set_processor(block.attn._default_processor_cls())
                delattr(block, '_tome_enabled')
        
        if hasattr(block, '_tome_block_index'):
            delattr(block, '_tome_block_index')


# Example usage function
def apply_tome_to_flux_pipeline(pipe, token_merge_args=None):
    """
    Apply token merging to a FLUX-based diffusion pipeline.
    
    Example:
        from diffusers import FluxPipeline
        
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        
        # Apply token reduction with frequency-based method
        apply_tome_to_flux_pipeline(pipe, {
            "ratio": 0.5,
            "merge_method": "frequency_global",
            "frequency_selection_method": "1d_dft",
            "merge_tokens": "keys/values",
            "block_tome_flags_double": [1]*19,  # Apply to all 19 double blocks
            "block_tome_flags_single": [1]*38,  # Apply to all 38 single blocks
        })
        
        # Apply only to later blocks (preserve early features)
        apply_tome_to_flux_pipeline(pipe, {
            "ratio": 0.3,
            "merge_method": "similarity", 
            "merge_tokens": "keys/values",
            "block_tome_flags_double": [0]*10 + [1]*9,   # Apply to last 9 double blocks
            "block_tome_flags_single": [0]*20 + [1]*18,  # Apply to last 18 single blocks
        })
        
        # Generate image with reduced image tokens
        image = pipe("a cat", num_inference_steps=50).images[0]
    """
    if hasattr(pipe, "transformer"):  # FLUX uses 'transformer'
        patch_flux_tome(pipe.transformer, token_merge_args)
    else:
        raise ValueError("Pipeline does not have a transformer attribute")



