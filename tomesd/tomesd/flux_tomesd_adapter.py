import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, Dict, Any, List
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxAttention
from diffusers.models.attention_processor import AttentionProcessor
import math
import torch.nn.functional as F

# Import tomesd merge functions and scorers
try:
    # Try relative import first (when used as package)
    from .tomesd.merge import (
        bipartite_soft_matching_random2d,
        bipartite_soft_matching_with_scoring,
        adaptive_block_pooling_random2d,
        bipartite_soft_matching_with_scoring_cached,
        do_nothing
    )
    from .tomesd.scoring import (
        TokenScorer,
        FrequencyScorer,
        SpatialFilterScorer,
        StatisticalScorer,
        SignalProcessingScorer,
        SpatialDistributionScorer,
        create_scorer
    )
    from .tomesd.utils import init_generator
except ImportError:
    # Fall back to absolute import (when run directly)
    try:
        from tomesd.merge import (
            bipartite_soft_matching_random2d,
            bipartite_soft_matching_with_scoring,
            adaptive_block_pooling_random2d,
            bipartite_soft_matching_with_scoring_cached,
            do_nothing
        )
        from tomesd.scoring import (
            TokenScorer,
            FrequencyScorer,
            SpatialFilterScorer,
            StatisticalScorer,
            SignalProcessingScorer,
            SpatialDistributionScorer,
            create_scorer
        )
        from tomesd.utils import init_generator
    except ImportError:
        # Last fallback: direct path import
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tomesd', 'tomesd'))
        from merge import (
            bipartite_soft_matching_random2d,
            bipartite_soft_matching_with_scoring,
            adaptive_block_pooling_random2d,
            bipartite_soft_matching_with_scoring_cached,
            do_nothing
        )
        from scoring import (
            TokenScorer,
            FrequencyScorer,
            SpatialFilterScorer,
            StatisticalScorer,
            SignalProcessingScorer,
            SpatialDistributionScorer,
            create_scorer
        )
        from utils import init_generator


class FluxToMeSDProcessor:
    """Custom attention processor for FLUX that implements tomesd token merging on IMAGE tokens only."""
    
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
        
        # Check if token merging should be enabled for this specific block
        # Use the block_index to check if this block should apply merging
        block_enabled = True  # Default enabled
        
        if self.block_index is not None and self.block_type is not None:
            # Get the appropriate block flags from tome_info args
            if self.block_type == "double":
                double_flags = self.tome_info['args'].get('block_tome_flags_double', [])
                if self.block_index < len(double_flags):
                    block_enabled = bool(double_flags[self.block_index])
            elif self.block_type == "single":
                single_flags = self.tome_info['args'].get('block_tome_flags_single', [])
                if self.block_index < len(single_flags):
                    block_enabled = bool(single_flags[self.block_index])
        
        # If token merging is not enabled for this block, use standard attention
        if not block_enabled:
            # Use standard FluxAttnProcessor instead of custom processor
            from diffusers.models.attention_processor import FluxAttnProcessor
            standard_processor = FluxAttnProcessor()
            return standard_processor(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)
            
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
        # Apply tomesd token merging AFTER rotary embeddings
        # ========================================
        # Initialize merge/unmerge functions
        image_merge_fn, image_unmerge_fn = do_nothing, do_nothing
        
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
            
            # Compute merge function using tomesd methods with caching support
            # For single blocks, need to pass text_seq_len for proper token handling
            actual_text_seq_len = text_seq_len if self.block_type == "single" else 0
            image_merge_fn, image_unmerge_fn = compute_merge_flux_with_cache(
                hidden_states, self.tome_info, actual_text_seq_len, self.block_index, self.block_type
            )
            
            # Apply merging if available
            if image_merge_fn != do_nothing:
                merge_mode = self.tome_info['args']['merge_tokens']
                
                # Apply merging to the appropriate tokens
                if self.tome_info['args']['merge_tokens'] == "all":
                    # Merge queries, keys, and values (all QKV tokens)
                    query_before = query.shape[1]
                    key_before = key.shape[1]
                    value_before = value.shape[1]
                    
                    if text_seq_len > 0:
                        # Split QKV into text and image parts
                        text_query, image_query = query[:, :text_seq_len], query[:, text_seq_len:]
                        text_key, image_key = key[:, :text_seq_len], key[:, text_seq_len:]
                        text_value, image_value = value[:, :text_seq_len], value[:, text_seq_len:]
                        
                        # Apply merge to image parts only
                        query_reshaped = image_query.flatten(2, 3)  # [batch, seq_len, heads*head_dim]
                        key_reshaped = image_key.flatten(2, 3)
                        value_reshaped = image_value.flatten(2, 3)
                        
                        query_merged = image_merge_fn(query_reshaped)
                        key_merged = image_merge_fn(key_reshaped)
                        value_merged = image_merge_fn(value_reshaped)
                        
                        image_query = query_merged.unflatten(-1, (attn.heads, -1))
                        image_key = key_merged.unflatten(-1, (attn.heads, -1))
                        image_value = value_merged.unflatten(-1, (attn.heads, -1))
                        
                        # Recombine
                        query = torch.cat([text_query, image_query], dim=1)
                        key = torch.cat([text_key, image_key], dim=1)
                        value = torch.cat([text_value, image_value], dim=1)
                    else:
                        # All tokens are image tokens
                        query_reshaped = query.flatten(2, 3)
                        key_reshaped = key.flatten(2, 3)
                        value_reshaped = value.flatten(2, 3)
                        
                        query_merged = image_merge_fn(query_reshaped)
                        key_merged = image_merge_fn(key_reshaped)
                        value_merged = image_merge_fn(value_reshaped)
                        
                        query = query_merged.unflatten(-1, (attn.heads, -1))
                        key = key_merged.unflatten(-1, (attn.heads, -1))
                        value = value_merged.unflatten(-1, (attn.heads, -1))
                    
                    query_after = query.shape[1]
                    key_after = key.shape[1]
                    value_after = value.shape[1]
                
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

        # ========================================
        # Apply unmerge to restore original token count for residual connection
        # ========================================
        # Apply unmerging if merge function was applied earlier
        if image_unmerge_fn != do_nothing:
            # Get text sequence length to know where to split for unmerging
            text_seq_len = 0
            if encoder_hidden_states is not None:
                text_seq_len = encoder_hidden_states.shape[1]
            elif self.block_type == "single" and self.tome_info.get('text_seq_len'):
                text_seq_len = self.tome_info['text_seq_len']
            
            if text_seq_len > 0:
                # Split attention output into text and image parts
                text_attn, image_attn = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
                
                # Apply unmerge to image part only
                image_attn_unmerged = image_unmerge_fn(image_attn)
                
                # Recombine text and unmerged image parts
                hidden_states = torch.cat([text_attn, image_attn_unmerged], dim=1)
            else:
                # All tokens are image tokens - unmerge the entire output
                hidden_states = image_unmerge_fn(hidden_states)

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


def compute_merge_flux_tomesd(x: torch.Tensor, tome_info: Dict[str, Any], text_seq_len: int) -> Tuple[Callable, Callable]:
    """
    Compute merge functions using tomesd methods for FLUX image tokens only.
    
    Args:
        x: Input tensor containing image tokens (for double blocks) or combined tokens (for single blocks)
        tome_info: Token merge info dictionary with tomesd configuration
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
        # FLUX total downsampling: 8× (VAE: 8×, patch_size: 1×)
        tome_info["size"] = (h * 8, h * 8)  # Store as pixel space size for consistency
        original_h, original_w = h * 8, h * 8
    else:
        original_h, original_w = tome_info["size"]
    
    # Get merge parameters
    args = tome_info["args"]
    timestep = tome_info.get("timestep", 0)
    merge_method_type = args.get("method", "mean")  # Get merge method type (mean, mlerp, prune)
    
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
    
    # Apply the appropriate tomesd method to IMAGE tokens only
    if active_method == "bipartite_random":
        ratio = args.get("ratio", 0.5)
        r = int(image_tokens * ratio)  # Image tokens to remove
        
        if r > 0:
            # Initialize generator if needed
            if args["generator"] is None or args["generator"].device != image_x.device:
                seed = args.get("seed", 0)
                args["generator"] = init_generator(image_x.device)
                if seed is not None:
                    args["generator"].manual_seed(seed)
            
            sx = args.get("sx", 2)
            sy = args.get("sy", 2)
            use_rand = args.get("use_rand", True)
            
            # Check if we should use prune method
            if merge_method_type == "prune":
                # Use dedicated prune functions
                image_merge_base_fn, image_unmerge_base_fn = create_prune_merge_functions(
                    image_x, w, h, sx, sy, r, no_rand=not use_rand, generator=args["generator"]
                )
            else:
                # Use standard bipartite random matching
                image_merge_base_fn, image_unmerge_base_fn = bipartite_soft_matching_random2d(
                    image_x, w, h, sx, sy, r,
                    no_rand=not use_rand,
                    generator=args["generator"],
                    locality_block_factor_h=args.get("locality_block_factor_h", 1),
                    locality_block_factor_w=args.get("locality_block_factor_w", 1)
                )
                
                # Apply merge method type wrapper (mean, mlerp)
                image_merge_base_fn, image_unmerge_base_fn = create_prune_merge_wrapper(
                    image_merge_base_fn, image_unmerge_base_fn, merge_method_type
                )
            
            merge_fn = create_image_only_merge_wrapper_tomesd(image_merge_base_fn, text_seq_len)
            unmerge_fn = create_image_only_merge_wrapper_tomesd(image_unmerge_base_fn, text_seq_len)
            
    elif active_method == "bipartite_scoring":
        ratio = args.get("ratio", 0.5)
        r = int(image_tokens * ratio)  # Image tokens to remove
        
        if r > 0:
            # Create scorer instance
            scorer_method = args.get("scorer_method", "spatial_filter_2d_conv")
            scorer = create_scorer_instance(scorer_method, args)
            
            # Initialize generator if needed
            if args["generator"] is None or args["generator"].device != image_x.device:
                seed = args.get("seed", 0)
                args["generator"] = init_generator(image_x.device)
                if seed is not None:
                    args["generator"].manual_seed(seed)
            
            sx = args.get("sx", 2)
            sy = args.get("sy", 2)
            preserve_ratio = args.get("preserve_ratio", 0.3)
            score_mode = args.get("score_mode", "high")
            use_rand = args.get("use_rand", True)
            preserve_spatial_uniformity = args.get("preserve_spatial_uniformity", False)
            if_low_frequency_dst_tokens = args.get("if_low_frequency_dst_tokens", False)
            use_caching = args.get("use_caching", False)
            
            # Check if we should use prune method
            if merge_method_type == "prune":
                # Use dedicated prune functions for scoring-based method
                image_merge_base_fn, image_unmerge_base_fn = create_prune_merge_functions(
                    image_x, w, h, sx, sy, r, no_rand=not use_rand, generator=args["generator"],
                    scorer=scorer, preserve_ratio=preserve_ratio, score_mode=score_mode,
                    preserve_spatial_uniformity=preserve_spatial_uniformity,
                    if_low_frequency_dst_tokens=if_low_frequency_dst_tokens,
                    timestep_normalized=timestep_normalized
                )
            else:
                # Use standard bipartite scoring with optional caching
                if use_caching:
                    image_merge_base_fn, image_unmerge_base_fn = bipartite_soft_matching_with_scoring_cached(
                        image_x, scorer, w, h, sx, sy, r,
                        preserve_ratio=preserve_ratio,
                        score_mode=score_mode,
                        preserve_spatial_uniformity=preserve_spatial_uniformity,
                        if_low_frequency_dst_tokens=if_low_frequency_dst_tokens,
                        no_rand=not use_rand,
                        generator=args["generator"],
                        timestep_normalized=timestep_normalized,
                        cache_resolution_merge=True
                    )
                else:
                    image_merge_base_fn, image_unmerge_base_fn = bipartite_soft_matching_with_scoring(
                        image_x, scorer, w, h, sx, sy, r,
                        preserve_ratio=preserve_ratio,
                        score_mode=score_mode,
                        preserve_spatial_uniformity=preserve_spatial_uniformity,
                        if_low_frequency_dst_tokens=if_low_frequency_dst_tokens,
                        no_rand=not use_rand,
                        generator=args["generator"],
                        timestep_normalized=timestep_normalized,
                        locality_block_factor_h=args.get("locality_block_factor_h", 1),
                        locality_block_factor_w=args.get("locality_block_factor_w", 1)
                    )
                
                # Apply merge method type wrapper (mean, mlerp)
                image_merge_base_fn, image_unmerge_base_fn = create_prune_merge_wrapper(
                    image_merge_base_fn, image_unmerge_base_fn, merge_method_type
                )
            
            merge_fn = create_image_only_merge_wrapper_tomesd(image_merge_base_fn, text_seq_len)
            unmerge_fn = create_image_only_merge_wrapper_tomesd(image_unmerge_base_fn, text_seq_len)
            
    elif active_method == "adaptive_pooling":
        ratio = args.get("ratio", 0.5)
        r = int(image_tokens * ratio)  # Image tokens to remove
        
        if r > 0:
            # Initialize generator if needed
            if args["generator"] is None or args["generator"].device != image_x.device:
                seed = args.get("seed", 0)
                args["generator"] = init_generator(image_x.device)
                if seed is not None:
                    args["generator"].manual_seed(seed)
            
            sx = args.get("sx", 2)
            sy = args.get("sy", 2)
            use_rand = args.get("use_rand", True)
            
            # Create scorer for ABP
            scorer_method = args.get("abp_scorer_method", "spatial_filter_2d_conv")
            abp_scorer = create_scorer_instance(scorer_method, args)
            tile_aggregation = args.get("abp_tile_aggregation", "max")
            
            # Check if we should use prune method
            if merge_method_type == "prune":
                # Use dedicated prune functions for ABP method
                image_merge_base_fn, image_unmerge_base_fn = create_prune_merge_functions(
                    image_x, w, h, sx, sy, r, no_rand=not use_rand, generator=args["generator"],
                    scorer=abp_scorer, tile_aggregation=tile_aggregation, use_abp=True
                )
            else:
                # Use standard adaptive block pooling
                image_merge_base_fn, image_unmerge_base_fn = adaptive_block_pooling_random2d(
                    image_x, w, h, sx, sy, r,
                    no_rand=not use_rand,
                    generator=args["generator"],
                    scorer=abp_scorer,
                    tile_aggregation=tile_aggregation
                )
                
                # Apply merge method type wrapper (mean, mlerp)
                image_merge_base_fn, image_unmerge_base_fn = create_prune_merge_wrapper(
                    image_merge_base_fn, image_unmerge_base_fn, merge_method_type
                )
            
            merge_fn = create_image_only_merge_wrapper_tomesd(image_merge_base_fn, text_seq_len)
            unmerge_fn = create_image_only_merge_wrapper_tomesd(image_unmerge_base_fn, text_seq_len)
    
    # For unsupported methods, merge_fn and unmerge_fn remain as do_nothing
    
    return merge_fn, unmerge_fn


def create_scorer_instance(scorer_method: str, args: Dict[str, Any]) -> TokenScorer:
    """
    Create a TokenScorer instance based on the method name and arguments.
    
    Args:
        scorer_method: Name of the scorer method
        args: Dictionary containing scorer-specific arguments
    """
    try:
        if scorer_method == "frequency" or scorer_method.startswith("frequency_"):
            # Extract method and ranking from scorer_method
            if scorer_method == "frequency":
                method = "1d_dft"
                ranking = "amplitude"
            else:
                parts = scorer_method.split("_")
                if len(parts) >= 3:
                    method = "_".join(parts[1:3])  # e.g., "1d_dft"
                    ranking = "_".join(parts[3:]) if len(parts) > 3 else "amplitude"
                else:
                    method = "1d_dft"
                    ranking = "amplitude"
            return FrequencyScorer(method=method, ranking=ranking)
            
        elif scorer_method == "spatial_filter" or scorer_method.startswith("spatial_filter_"):
            # Extract spatial filter parameters
            spatial_method = args.get("spatial_method", "2d_conv")
            spatial_norm = args.get("spatial_norm", "l1")
            return SpatialFilterScorer(method=spatial_method, norm=spatial_norm)
            
        elif scorer_method == "statistical" or scorer_method.startswith("statistical_"):
            # Extract statistical method
            if scorer_method == "statistical":
                method = "std"
            else:
                parts = scorer_method.split("_")
                method = "_".join(parts[1:]) if len(parts) > 1 else "std"
            return StatisticalScorer(method=method)
            
        elif scorer_method == "signal_processing" or scorer_method.startswith("signal_processing_"):
            # Extract signal processing parameters
            method = args.get("signal_method", "gradient")
            direction = args.get("signal_direction", "both")
            return SignalProcessingScorer(method=method, direction=direction)
            
        elif scorer_method == "spatial_distribution" or scorer_method.startswith("spatial_distribution_"):
            # Extract spatial distribution parameters  
            alpha = args.get("spatial_alpha", 2.0)
            return SpatialDistributionScorer(alpha=alpha)
            
        elif scorer_method == "similarity" or scorer_method.startswith("similarity_"):
            # Extract similarity parameters
            method = args.get("similarity_method", "local_neighbors_inverted")
            return SimilarityScorer(method=method)
            
        else:
            # Default fallback
            print(f"Warning: Unknown scorer method '{scorer_method}', using default SpatialFilterScorer")
            return SpatialFilterScorer(method="2d_conv", norm="l1")
            
    except Exception as e:
        print(f"Warning: Error creating scorer '{scorer_method}': {e}, using default SpatialFilterScorer")
        return SpatialFilterScorer(method="2d_conv", norm="l1")


def create_prune_merge_functions(image_x: torch.Tensor, w: int, h: int, sx: int, sy: int, r: int, 
                                no_rand: bool = False, generator: Optional[torch.Generator] = None,
                                scorer: Optional[Any] = None, preserve_ratio: float = 0.0,
                                score_mode: str = "high", preserve_spatial_uniformity: bool = False,
                                if_low_frequency_dst_tokens: bool = False, timestep_normalized: float = 0.0,
                                tile_aggregation: str = "max", use_abp: bool = False) -> Tuple[Callable, Callable]:
    """
    Create high-performance prune-specific merge functions that select tokens in a grid pattern.
    
    PERFORMANCE OPTIMIZATIONS:
    - Replaces Python for-loops with vectorized PyTorch operations
    - Pre-computes nearest neighbor mapping using mathematical formulas for regular grids
    - Caches mapping computation outside the merge/unmerge functions
    - Uses advanced indexing for token selection/restoration
    
    This eliminates the O(n*m) complexity that was causing slowdowns with ABP+prune.
    
    Args:
        image_x: Input image tensor
        w, h: Spatial dimensions
        sx, sy: Spatial downsampling factors
        r: Number of tokens to remove
        no_rand: Whether to disable randomization
        generator: Random generator
        scorer: Scorer for ABP method
        preserve_ratio: Ratio of tokens to preserve
        score_mode: Scoring mode
        preserve_spatial_uniformity: Whether to preserve spatial uniformity
        if_low_frequency_dst_tokens: Whether to use low frequency destination tokens
        timestep_normalized: Normalized timestep
        tile_aggregation: Aggregation method for ABP
        use_abp: Whether to use ABP algorithm
    
    Returns:
        Tuple of (prune_merge_fn, prune_unmerge_fn)
    """
    # Simple grid-based pruning: select tokens in a regular pattern
    batch_size, num_tokens, dim = image_x.shape
    
    # Calculate how many tokens to keep
    tokens_to_keep = num_tokens - r
    if tokens_to_keep <= 0:
        return do_nothing, do_nothing
    
    # For prune method, create a simple downsampling pattern using vectorized operations
    # Create grid indices efficiently without loops
    y_coords = torch.arange(h, device=image_x.device)
    x_coords = torch.arange(w, device=image_x.device)
    
    # Find coordinates that match the grid pattern (every sx, sy tokens)
    y_grid = y_coords[y_coords % sy == 0]
    x_grid = x_coords[x_coords % sx == 0]
    
    # Create meshgrid and convert to linear indices
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
    grid_indices = yy * w + xx
    indices_to_keep = grid_indices.flatten()
    
    # If we need more tokens, use uniform sampling
    if len(indices_to_keep) < tokens_to_keep:
        step = max(1, num_tokens // tokens_to_keep)
        indices_to_keep = torch.arange(0, num_tokens, step, device=image_x.device)[:tokens_to_keep]
    else:
        # Limit to the number we need
        indices_to_keep = indices_to_keep[:tokens_to_keep]
    
    def prune_merge_fn(x: torch.Tensor) -> torch.Tensor:
        # Select only the tokens we want to keep (prune the rest)
        return x[:, indices_to_keep, :]
    
    # Pre-compute nearest neighbor mapping once (outside the function for caching)
    # Check if we're using a regular grid pattern for even faster computation
    is_regular_grid = (len(y_grid) * len(x_grid) == len(indices_to_keep) and 
                      len(y_grid) > 0 and len(x_grid) > 0)
    
    if is_regular_grid:
        # Ultra-fast mapping for regular grids using mathematical formulas
        # Convert linear indices to 2D coordinates
        all_y = torch.arange(h, device=image_x.device).repeat_interleave(w)
        all_x = torch.arange(w, device=image_x.device).repeat(h)
        
        # Find nearest grid points efficiently
        nearest_y = (all_y // sy) * sy
        nearest_x = (all_x // sx) * sx
        
        # Clamp to valid grid boundaries
        max_y = y_grid[-1].item() if len(y_grid) > 0 else 0
        max_x = x_grid[-1].item() if len(x_grid) > 0 else 0
        nearest_y = torch.clamp(nearest_y, 0, max_y)
        nearest_x = torch.clamp(nearest_x, 0, max_x)
        
        # Convert back to linear indices in the kept tokens array
        grid_y_idx = nearest_y // sy
        grid_x_idx = nearest_x // sx
        
        # Map to indices in the kept tokens array
        nearest_indices = grid_y_idx * len(x_grid) + grid_x_idx
        
        # Ensure indices are within bounds
        nearest_indices = torch.clamp(nearest_indices, 0, len(indices_to_keep) - 1)
    else:
        # Fallback to distance-based computation for irregular patterns
        all_indices = torch.arange(num_tokens, device=image_x.device)
        
        # Compute distances from all positions to kept positions (vectorized)
        # Shape: [original_tokens, kept_tokens]
        distances = torch.abs(all_indices.unsqueeze(1) - indices_to_keep.unsqueeze(0))
        
        # Find nearest kept token for each original position
        nearest_indices = torch.argmin(distances, dim=1)  # Shape: [original_tokens]
    
    def prune_unmerge_fn(x: torch.Tensor) -> torch.Tensor:
        # Restore original size using pre-computed mapping (very fast)
        # Use advanced indexing to gather the appropriate tokens
        # Shape: [batch_size, original_tokens, dim]
        return x[:, nearest_indices, :]
    
    return prune_merge_fn, prune_unmerge_fn


def create_prune_merge_wrapper(base_merge_fn: Callable, base_unmerge_fn: Callable, merge_method_type: str = "mean") -> Tuple[Callable, Callable]:
    """
    Create wrapper functions that implement different merge methods (mean, mlerp, prune).
    
    Args:
        base_merge_fn: The base merge function that performs token grouping
        base_unmerge_fn: The base unmerge function that restores original size
        merge_method_type: Type of merging - "mean", "mlerp", or "prune"
    
    Returns:
        Tuple of (wrapped_merge_fn, wrapped_unmerge_fn)
    """
    if merge_method_type == "prune":
        # For prune method, we should use the dedicated prune functions
        # This is a fallback wrapper that still uses the base functions
        # The actual prune implementation is handled in compute_merge_flux_tomesd
        return base_merge_fn, base_unmerge_fn
    
    elif merge_method_type == "mlerp":
        # For MLERP method: Maximum-Norm Linear Interpolation
        def mlerp_merge_fn(x: torch.Tensor) -> torch.Tensor:
            # Store original for MLERP computation
            original_x = x.clone()
            
            # Use base merge function to get the merged result
            merged_x = base_merge_fn(x)
            
            # MLERP preserves the maximum norm during interpolation
            # This is a simplified implementation - ideally should be integrated
            # into the base merge function for proper norm preservation
            return merged_x
        
        def mlerp_unmerge_fn(x: torch.Tensor) -> torch.Tensor:
            return base_unmerge_fn(x)
        
        return mlerp_merge_fn, mlerp_unmerge_fn
    
    else:  # Default "mean" method
        return base_merge_fn, base_unmerge_fn


def create_image_only_merge_wrapper_tomesd(base_merge_fn: Callable, text_seq_len: int) -> Callable:
    """
    Create a wrapper function that applies tomesd merging only to image tokens,
    leaving text tokens unchanged.
    
    Note: This wrapper expects 3D tensors [batch, seq_len, dim] as input,
    which is what the attention processor provides after flattening 4D tensors.
    
    Args:
        base_merge_fn: The base tomesd merge function to apply to image tokens
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


def generate_flux_call_id(hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None) -> str:
    """Generate unique ID for current FLUX generation call based on IMAGE token characteristics"""
    # Focus on image tokens only - for FLUX this is the primary input
    image_shape = hidden_states.shape
    image_device = hidden_states.device
    image_dtype = hidden_states.dtype
    
    # For double blocks, encoder_hidden_states contains text tokens separately
    # For single blocks, encoder_hidden_states is None and hidden_states contains both
    call_components = [
        str(id(hidden_states)),
        str(image_shape),
        str(image_device),
        str(image_dtype)
    ]
    
    return "_".join(call_components)


def is_new_flux_call(hidden_states: torch.Tensor, tome_info: Dict, encoder_hidden_states: Optional[torch.Tensor] = None) -> bool:
    """Check if this is a new FLUX generation call that should reset IMAGE token cache"""
    cache_info = tome_info.get("block_cache", {})
    if not cache_info.get("enabled", False):
        return False
    
    current_call_id = generate_flux_call_id(hidden_states, encoder_hidden_states)
    stored_call_id = cache_info.get("generation_call_id")
    
    return stored_call_id != current_call_id


def reset_flux_cache_for_new_call(hidden_states: torch.Tensor, tome_info: Dict, encoder_hidden_states: Optional[torch.Tensor] = None):
    """Reset FLUX IMAGE token cache when starting a new generation call"""
    cache_info = tome_info.get("block_cache", {})
    if cache_info.get("enabled", False):
        # Clear both double and single stream caches
        cache_info["double_stream"]["cached_functions"].clear()
        cache_info["double_stream"]["last_recalc_block"] = -1
        cache_info["double_stream"]["current_block_idx"] = 0
        
        cache_info["single_stream"]["cached_functions"].clear()
        cache_info["single_stream"]["last_recalc_block"] = -1
        cache_info["single_stream"]["current_block_idx"] = 0
        
        # Update call ID based on image tokens
        cache_info["generation_call_id"] = generate_flux_call_id(hidden_states, encoder_hidden_states)
        
        # Store image shape hash for change detection
        image_shape = hidden_states.shape
        cache_info["image_shape_hash"] = str(image_shape)


def should_recalculate_flux_cache(block_idx: int, block_type: str, tome_info: Dict) -> bool:
    """Check if FLUX IMAGE token cache should be recalculated for this block"""
    cache_info = tome_info.get("block_cache", {})
    if not cache_info.get("enabled", False):
        return True  # No caching, always recalculate
    
    stream_cache = cache_info.get(f"{block_type}_stream", {})
    recalc_interval = cache_info.get("recalc_interval", 4)
    last_recalc = stream_cache.get("last_recalc_block", -1)
    
    # Recalculate if:
    # 1. Never calculated before (last_recalc == -1)
    # 2. Interval has passed since last recalculation
    return (last_recalc == -1) or (block_idx - last_recalc >= recalc_interval)


def cache_flux_block_functions(block_idx: int, block_type: str, merge_fn: Callable, unmerge_fn: Callable, tome_info: Dict):
    """Cache FLUX IMAGE token merge/unmerge functions for a specific block"""
    cache_info = tome_info.get("block_cache", {})
    if not cache_info.get("enabled", False):
        return
    
    stream_cache = cache_info.get(f"{block_type}_stream", {})
    if stream_cache:
        stream_cache["cached_functions"][block_idx] = (merge_fn, unmerge_fn)
        stream_cache["last_recalc_block"] = block_idx
        stream_cache["current_block_idx"] = block_idx


def get_flux_cached_functions(block_idx: int, block_type: str, tome_info: Dict) -> Optional[Tuple[Callable, Callable]]:
    """Get cached FLUX IMAGE token merge/unmerge functions for a block"""
    cache_info = tome_info.get("block_cache", {})
    if not cache_info.get("enabled", False):
        return None
    
    stream_cache = cache_info.get(f"{block_type}_stream", {})
    return stream_cache.get("cached_functions", {}).get(block_idx)


def clear_flux_block_cache(tome_info: Dict):
    """Clear all FLUX IMAGE token cache"""
    cache_info = tome_info.get("block_cache", {})
    if cache_info.get("enabled", False):
        cache_info["double_stream"]["cached_functions"].clear()
        cache_info["single_stream"]["cached_functions"].clear()
        cache_info["generation_call_id"] = None
        cache_info["image_shape_hash"] = None


def compute_merge_flux_with_cache(x: torch.Tensor, tome_info: Dict[str, Any], text_seq_len: int, 
                                  block_idx: int, block_type: str) -> Tuple[Callable, Callable]:
    """
    Compute merge functions with optional block-level caching for FLUX IMAGE tokens.
    
    This function focuses exclusively on IMAGE token processing:
    - For double blocks: x contains only image tokens (text tokens are in encoder_hidden_states)
    - For single blocks: x contains text+image tokens, but only image portion is processed
    
    Args:
        x: Input tensor (image tokens for double blocks, text+image for single blocks)
        tome_info: Token merge info dictionary with FLUX configuration
        text_seq_len: Length of text sequence (0 for double blocks, >0 for single blocks)
        block_idx: Index of the current transformer block
        block_type: "double" or "single" stream type
    
    Returns:
        Tuple of (merge_fn, unmerge_fn) for IMAGE tokens only
    """
    
    # Check if this is a new generation call and reset cache if needed
    if is_new_flux_call(x, tome_info):
        reset_flux_cache_for_new_call(x, tome_info)
    
    # Update current block index for the appropriate stream
    cache_info = tome_info.get("block_cache", {})
    if cache_info.get("enabled", False):
        stream_cache = cache_info.get(f"{block_type}_stream", {})
        if stream_cache:
            stream_cache["current_block_idx"] = block_idx
    
    # Check if we should recalculate or use cached functions
    should_recalc = should_recalculate_flux_cache(block_idx, block_type, tome_info)
    
    if not should_recalc:
        # Try to get cached functions for IMAGE tokens
        cached_functions = get_flux_cached_functions(block_idx, block_type, tome_info)
        if cached_functions is not None:
            cached_merge_fn, cached_unmerge_fn = cached_functions
            return cached_merge_fn, cached_unmerge_fn
    
    # No cached functions available or recalculation needed
    # Use the original compute_merge_flux_tomesd function for IMAGE tokens
    merge_fn, unmerge_fn = compute_merge_flux_tomesd(x, tome_info, text_seq_len)
    
    # Cache the newly computed functions if caching is enabled
    if cache_info.get("enabled", False) and should_recalc:
        cache_flux_block_functions(block_idx, block_type, merge_fn, unmerge_fn, tome_info)
    
    return merge_fn, unmerge_fn


def patch_flux_tomesd(model: FluxTransformer2DModel, token_merge_args: Dict[str, Any] = None):
    """
    Patch a FLUX model to use tomesd token merging on IMAGE tokens only.
    
    Args:
        model: The FLUX model to patch
        token_merge_args: Dictionary of tomesd token merging arguments
                         - block_tome_flags_double: List indicating which double blocks should apply ToMeSD
                         - block_tome_flags_single: List indicating which single blocks should apply ToMeSD
                         - merge_method: "bipartite_random", "bipartite_scoring", "adaptive_pooling"
                         - scorer_method: TokenScorer method name
                         - cache_merge_functions: Whether to cache merge/unmerge functions for blocks within a generation call
                         - cache_recalc_interval: Recalculate cached functions every N blocks
                         - Other tomesd-specific parameters
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
            # Basic parameters
            "ratio": token_merge_args.get("ratio", 0.5),
            "sx": token_merge_args.get("sx", 2),
            "sy": token_merge_args.get("sy", 2),
            "use_rand": token_merge_args.get("use_rand", True),
            "generator": None,
            "merge_tokens": token_merge_args.get("merge_tokens", "keys/values"),
            "seed": token_merge_args.get("seed", 0),
            
            # Timestep control
            "timestep_threshold_switch": token_merge_args.get("timestep_threshold_switch", 0.5),
            "timestep_threshold_stop": token_merge_args.get("timestep_threshold_stop", 0.1),
            "secondary_merge_method": token_merge_args.get("secondary_merge_method", "none"),
            
            # ToMeSD-specific parameters
            "merge_method": token_merge_args.get("merge_method", "bipartite_scoring"),
            "scorer_method": token_merge_args.get("scorer_method", "spatial_filter_2d_conv"),
            "method": token_merge_args.get("method", "mean"),  # Merge method type: mean, mlerp, prune
            
            # Bipartite scoring parameters
            "preserve_ratio": token_merge_args.get("preserve_ratio", 0.3),
            "score_mode": token_merge_args.get("score_mode", "high"),
            "preserve_spatial_uniformity": token_merge_args.get("preserve_spatial_uniformity", False),
            "if_low_frequency_dst_tokens": token_merge_args.get("if_low_frequency_dst_tokens", False),
            "use_caching": token_merge_args.get("use_caching", False),
            
            # Scorer-specific parameters
            "spatial_method": token_merge_args.get("spatial_method", "2d_conv"),
            "spatial_norm": token_merge_args.get("spatial_norm", "l1"),
            "signal_method": token_merge_args.get("signal_method", "gradient"),
            "signal_direction": token_merge_args.get("signal_direction", "both"),
            "spatial_alpha": token_merge_args.get("spatial_alpha", 2.0),
            
            # Adaptive pooling parameters
            "abp_scorer_method": token_merge_args.get("abp_scorer_method", "spatial_filter_2d_conv"),
            "abp_tile_aggregation": token_merge_args.get("abp_tile_aggregation", "max"),
            
            # Locality-based similarity parameters
            "locality_block_factor_h": token_merge_args.get("locality_block_factor_h", 1),
            "locality_block_factor_w": token_merge_args.get("locality_block_factor_w", 1),
        },
        # Block-level cache for IMAGE tokens only (FLUX dual-stream architecture)
        "block_cache": {
            "enabled": token_merge_args.get("cache_merge_functions", False),
            "recalc_interval": token_merge_args.get("cache_recalc_interval", 4),
            "double_stream": {
                "cached_functions": {},           # block_idx -> (merge_fn, unmerge_fn) for image tokens
                "last_recalc_block": -1,
                "current_block_idx": 0,
            },
            "single_stream": {
                "cached_functions": {},           # block_idx -> (merge_fn, unmerge_fn) for image tokens
                "last_recalc_block": -1,
                "current_block_idx": 0,
            },
            "generation_call_id": None,          # Track current generation call (image-focused)
            "image_shape_hash": None             # Track image token shape changes
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
    
    # Apply ToMeSD to double transformer blocks
    tome_double_count = 0
    for i, block in enumerate(model.transformer_blocks):
        should_apply = i < len(double_flags) and bool(double_flags[i])
        
        if should_apply:
            if hasattr(block, 'attn') and isinstance(block.attn, FluxAttention):
                processor = FluxToMeSDProcessor(block_index=i, block_type="double")
                processor.tome_info = model._tome_info
                block.attn.set_processor(processor)
                block._tome_enabled = True
                tome_double_count += 1
            else:
                block._tome_enabled = False
        else:
            # For disabled blocks, keep standard processor and mark as disabled
            block._tome_enabled = False
        
        block._tome_block_index = i
    
    # Apply ToMeSD to single transformer blocks
    tome_single_count = 0
    for i, block in enumerate(model.single_transformer_blocks):
        should_apply = i < len(single_flags) and bool(single_flags[i])
        
        if should_apply:
            if hasattr(block, 'attn') and isinstance(block.attn, FluxAttention):
                processor = FluxToMeSDProcessor(block_index=i, block_type="single")
                processor.tome_info = model._tome_info
                block.attn.set_processor(processor)
                block._tome_enabled = True
                tome_single_count += 1
            else:
                block._tome_enabled = False
        else:
            # For disabled blocks, keep standard processor and mark as disabled
            block._tome_enabled = False
        
        block._tome_block_index = i
    
    # FLUX ToMeSD configuration applied silently


def remove_flux_tomesd_patch(model: FluxTransformer2DModel):
    """Remove tomesd token merging patch from FLUX model."""
    
    # Remove hooks and clear cache
    if hasattr(model, "_tome_info"):
        # Clear any cached IMAGE token functions
        clear_flux_block_cache(model._tome_info)
        
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
def apply_tomesd_to_flux_pipeline(pipe, token_merge_args=None):
    """
    Apply tomesd token merging to a FLUX-based diffusion pipeline.
    
    Example:
        from diffusers import FluxPipeline
        
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        
        # Apply bipartite scoring with spatial filter scorer
        apply_tomesd_to_flux_pipeline(pipe, {
            "ratio": 0.5,
            "merge_method": "bipartite_scoring",
            "scorer_method": "spatial_filter_2d_conv",
            "merge_tokens": "keys/values",
            "preserve_ratio": 0.3,
            "score_mode": "high",
            "block_tome_flags_double": [1]*19,  # Apply to all 19 double blocks
            "block_tome_flags_single": [1]*38,  # Apply to all 38 single blocks
        })
        
        # Apply adaptive pooling with frequency scorer to merge all QKV tokens
        apply_tomesd_to_flux_pipeline(pipe, {
            "ratio": 0.4,
            "merge_method": "adaptive_pooling",
            "abp_scorer_method": "frequency_1d_dft_amplitude",
            "abp_tile_aggregation": "max",
            "merge_tokens": "all",  # Merge queries, keys, and values
            "block_tome_flags_double": [0]*10 + [1]*9,   # Apply to last 9 double blocks
            "block_tome_flags_single": [0]*20 + [1]*18,  # Apply to last 18 single blocks
        })
        
        # Apply bipartite random (basic method)
        apply_tomesd_to_flux_pipeline(pipe, {
            "ratio": 0.3,
            "merge_method": "bipartite_random", 
            "sx": 2, "sy": 2,
            "use_rand": True,
            "merge_tokens": "keys/values",
            "timestep_threshold_switch": 0.7,  # Switch methods at high timestep
            "secondary_merge_method": "bipartite_scoring",  # Use scoring at low timesteps
        })
        
        # Apply with block-level caching for better performance (IMAGE tokens only)
        apply_tomesd_to_flux_pipeline(pipe, {
            "ratio": 0.5,
            "merge_method": "bipartite_scoring",
            "scorer_method": "spatial_filter_2d_conv",
            "cache_merge_functions": True,      # Enable block-level IMAGE token caching
            "cache_recalc_interval": 4,        # Recalculate every 4 blocks for dual streams
            "preserve_ratio": 0.2,
            "score_mode": "high",
            "preserve_spatial_uniformity": True,
            "if_low_frequency_dst_tokens": True,
        })
        
        # Apply with timestep-aware scoring
        apply_tomesd_to_flux_pipeline(pipe, {
            "ratio": 0.4,
            "merge_method": "bipartite_scoring",
            "scorer_method": "signal_processing_gradient",
            "score_mode": "timestep_scheduler",  # Dynamic scoring based on timestep
            "signal_method": "gradient",
            "signal_direction": "both",
            "timestep_threshold_switch": 0.6,
            "timestep_threshold_stop": 0.15,
        })
        
        # Use ABP to merge all QKV tokens with spatial filter scorer
        apply_tomesd_to_flux_pipeline(pipe, {
            "ratio": 0.5,
            "merge_method": "adaptive_pooling",
            "abp_scorer_method": "spatial_filter_2d_conv",
            "abp_tile_aggregation": "mean",
            "merge_tokens": "all",  # Merge queries, keys, AND values
            "spatial_method": "2d_conv",
            "spatial_norm": "l2",
            "block_tome_flags_double": [1]*19,  # Apply to all double blocks
            "block_tome_flags_single": [1]*38,  # Apply to all single blocks
        })
        
        # Use ABP to merge all QKV tokens with statistical scorer
        apply_tomesd_to_flux_pipeline(pipe, {
            "ratio": 0.3,
            "merge_method": "adaptive_pooling",
            "abp_scorer_method": "statistical_std",
            "abp_tile_aggregation": "max",
            "merge_tokens": "all",  # Maximum token reduction: Q, K, V all merged
            "sx": 2, "sy": 2,
            "use_rand": False,  # Deterministic merging
        })
        
        # Use locality-based similarity for spatial-aware token merging
        apply_tomesd_to_flux_pipeline(pipe, {
            "ratio": 0.4,
            "merge_method": "bipartite_scoring",
            "scorer_method": "spatial_filter_2d_conv",
            "preserve_ratio": 0.2,
            "score_mode": "high",
            "locality_block_factor_h": 2,  # Divide height into 2 blocks
            "locality_block_factor_w": 2,  # Divide width into 2 blocks
            "merge_tokens": "keys/values",
        })
        
        # Generate image with reduced image tokens
        image = pipe("a cat", num_inference_steps=50).images[0]
    """
    if hasattr(pipe, "transformer"):  # FLUX uses 'transformer'
        patch_flux_tomesd(pipe.transformer, token_merge_args)
    else:
        raise ValueError("Pipeline does not have a transformer attribute")
