import torch
from typing import Tuple, Callable
from diffusers.models.attention_processor import XFormersAttnProcessor, Attention
import xformers, xformers.ops
from typing import Optional
import math
import torch.nn.functional as F
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.import_utils import is_xformers_available
from typing import Union
import torch.fft

# Add these new imports after the existing ones
try:
    # Try relative imports first (when used as package)
    from .token_scoring import (
        _compute_token_scores,
        _select_indices_by_mode,
        _validate_common_args,
        _compute_non_uniform_grid_indices
    )
    from .frequency_selection import (
        frequency_based_selection,
        frequency_based_selection_blockwise,
        frequency_based_selection_blockwise_with_blend,
        frequency_based_token_mask,
        frequency_based_token_mask_blockwise
    )
    from .masked_attention_handler import handle_masked_attention
except ImportError:
    # Fall back to absolute imports (when run directly)
    from token_scoring import (
        _compute_token_scores,
        _select_indices_by_mode,
        _validate_common_args,
        _compute_non_uniform_grid_indices
    )
    from frequency_selection import (
        frequency_based_selection,
        frequency_based_selection_blockwise,
        frequency_based_selection_blockwise_with_blend,
        frequency_based_token_mask,
        frequency_based_token_mask_blockwise
    )
    from masked_attention_handler import handle_masked_attention



if is_xformers_available():
    import xformers
    import xformers.ops
    xformers_is_available = True
else:
    xformers_is_available = False


if hasattr(F, "scaled_dot_product_attention"):
    torch2_is_available = True
else:
    torch2_is_available = False



def block_average_pool_unpool(w_original_level: int, h_original_level: int, downsample_factor: int) -> Tuple[Callable, Callable]: #Dummy
    if downsample_factor <= 1: return do_nothing, do_nothing
    w_pooled, h_pooled = w_original_level // downsample_factor, h_original_level // downsample_factor
    if w_pooled == 0 or h_pooled == 0: return do_nothing, do_nothing
    m = lambda item, mode="mean": up_or_downsample(item, w_original_level, h_original_level, w_pooled, h_pooled, "avg_pool", blend_factor=None)
    u = lambda item: up_or_downsample(item, w_pooled, h_pooled, w_original_level, h_original_level, "nearest", blend_factor=None)
    return m,u


def init_generator(device: torch.device, fallback: torch.Generator = None, seed: int = None):
    """
    Forks the current default random generator given device.
    
    Args:
        device: Target device for the generator
        fallback: Fallback generator if device not supported
        seed: Optional seed for reproducibility. If None, uses current RNG state.
    """
    if seed is not None:
        # Create generator with specific seed for reproducibility
        generator = torch.Generator(device=device if device.type in ["cpu", "cuda"] else "cpu")
        generator.manual_seed(seed)
        return generator
    
    # Original behavior: fork current RNG state
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"), seed=seed)
        else:
            return fallback


def do_nothing(x: torch.Tensor, mode: Optional[str] = None):
    return x


def compute_dynamic_blend_factor(timestep_value, enable_interpolation, start_ratio, end_ratio, static_factor=0.5, log_key=None):
    """
    Compute dynamic blend factor based on timestep for timestep-aware linear blend.
    
    Args:
        timestep_value: Current timestep (0-999 range, where 999=high noise, 0=low noise)
        enable_interpolation: Whether to enable timestep-based interpolation
        start_ratio: Blend factor at timestep 999 (high noise)
        end_ratio: Blend factor at timestep 0 (low noise)  
        static_factor: Fallback static factor when interpolation is disabled
        log_key: Unique key for logging to avoid spam (optional)
        
    Returns:
        float: Dynamic blend factor (CFG-style extended range allowed)
    """
    if not enable_interpolation:
        return static_factor
    
    if timestep_value is None:
        return static_factor
        
    # Normalize timestep to [0, 1] range where 1 = high noise (timestep 999), 0 = low noise (timestep 0)
    timestep_normalized = float(timestep_value) / 999.0
    
    # Linear interpolation: at t=999 (normalized=1.0) use start_ratio, at t=0 (normalized=0.0) use end_ratio
    blend_factor = start_ratio * timestep_normalized + end_ratio * (1.0 - timestep_normalized)
    
    # No clamping - allow CFG-style extended range for more aggressive blending
    
    # Log the first few times for debugging (avoid spam)
    if log_key and not hasattr(compute_dynamic_blend_factor, f'_logged_{log_key}'):
        print(f"Dynamic blend: t={timestep_value} → factor={blend_factor:.3f} (start={start_ratio}, end={end_ratio})")
        setattr(compute_dynamic_blend_factor, f'_logged_{log_key}', True)
    
    return blend_factor

def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def up_or_downsample(item, cur_w, cur_h, new_w, new_h, method, timestep: Optional[int] = None, generator: Optional[torch.Generator] = None, blend_factor: Optional[float] = None, blend_method_1: Optional[str] = None, blend_method_2: Optional[str] = None):
    """
    Upsamples or downsamples the input tensor using various methods.
    
    For linear_blend method, you can specify custom blend components via blend_method_1 and blend_method_2.
    Available methods for blending: max_pool, avg_pool, area, bilinear, bicubic, nearest-exact, 
    top_right, bottom_left, bottom_right, random, uniform_random, uniform_timestep.
    
    Args:
        blend_method_1: First method for linear_blend (default: "nearest-exact")
        blend_method_2: Second method for linear_blend (default: "avg_pool")
        blend_factor: Weight for method1 in CFG-style blend (0.0=pure method2, 1.0=pure method1, >1.0=extrapolate beyond method1, <0.0=extrapolate beyond method2)
    """
    batch_size = item.shape[0]

    item = item.reshape(batch_size, cur_h, cur_w, -1)
    item = item.permute(0, 3, 1, 2)
    df = cur_h // new_h
    
    # Helper function to apply a specific method
    def apply_method(tensor, method_name):
        if method_name == "max_pool":
            return F.max_pool2d(tensor, kernel_size=df, stride=df, padding=0)
        elif method_name == "avg_pool":
            return F.avg_pool2d(tensor, kernel_size=df, stride=df, padding=0)
        elif method_name == "area":
            return F.interpolate(tensor, size=(new_h, new_w), mode="area")
        elif method_name in ["bilinear", "bicubic"]:
            return F.interpolate(tensor, size=(new_h, new_w), mode=method_name, antialias=True)
        elif method_name == "nearest-exact":
            return F.interpolate(tensor, size=(new_h, new_w), mode="nearest-exact")
        elif method_name == "top_right":
            blocks = tensor.unfold(2, df, df).unfold(3, df, df)
            return blocks[:, :, :, :, 0, df - 1]
        elif method_name == "bottom_left":
            blocks = tensor.unfold(2, df, df).unfold(3, df, df)
            return blocks[:, :, :, :, df - 1, 0]
        elif method_name == "bottom_right":
            blocks = tensor.unfold(2, df, df).unfold(3, df, df)
            return blocks[:, :, :, :, df - 1, df - 1]
        elif method_name == "random":
            if generator is None:
                raise ValueError("A generator must be provided for the 'random' downsample method.")
            B, C, _, _ = tensor.shape
            blocks = tensor.unfold(2, df, df).unfold(3, df, df)
            blocks_flat = blocks.reshape(B, C, new_h, new_w, df * df)
            rand_indices = torch.randint(0, df * df, size=(B, C, new_h, new_w, 1), device=tensor.device, generator=generator)
            return torch.gather(blocks_flat, dim=-1, index=rand_indices).squeeze(-1)
        elif method_name == "uniform_random":
            blocks = tensor.unfold(2, df, df).unfold(3, df, df)
            rand_i = torch.randint(0, df, (1,)).item()
            rand_j = torch.randint(0, df, (1,)).item()
            return blocks[:, :, :, :, rand_i, rand_j]
        elif method_name == "uniform_timestep":
            if timestep is None:
                raise ValueError("timestep must be provided for 'uniform_timestep' method")
            _gen_cache: dict[tuple[int, str], torch.Generator] = getattr(up_or_downsample, "_gen_cache", {})
            key = (int(timestep), tensor.device.type)
            if key not in _gen_cache:
                g = torch.Generator(device=tensor.device)
                g.manual_seed(int(timestep))
                _gen_cache[key] = g
            rng = _gen_cache[key]
            up_or_downsample._gen_cache = _gen_cache
            blocks = tensor.unfold(2, df, df).unfold(3, df, df)
            rand_i = torch.randint(0, df, (1,), generator=rng, device=tensor.device).item()
            rand_j = torch.randint(0, df, (1,), generator=rng, device=tensor.device).item()
            return blocks[:, :, :, :, rand_i, rand_j]
        else:
            return F.interpolate(tensor, size=(new_h, new_w), mode=method_name)
    
    if method == "max_pool":
        item = F.max_pool2d(item, kernel_size=df, stride=df, padding=0)
    elif method == "avg_pool":
        item = F.avg_pool2d(item, kernel_size=df, stride=df, padding=0)
    elif method == "area":
        # pure block‑average downsampling; no antialias flag allowed
        item = F.interpolate(item, size=(new_h, new_w), mode="area")
    elif method == ("bilinear", "bicubic"):
        # interpolation + true low‑pass filter
        item = F.interpolate(
            item, size=(new_h, new_w),
            mode=method, antialias=True
        )
    elif method == "top_right":
        # Assumes df is valid integer > 0 and divides H, W
        blocks = item.unfold(2, df, df).unfold(3, df, df)
        # blocks shape: (B, C, new_h, new_w, df, df)
        item = blocks[:, :, :, :, 0, df - 1] # Select top-right from each block
        # item shape: (B, C, new_h, new_w)

    elif method == "bottom_left":
        # Assumes df is valid integer > 0 and divides H, W
        blocks = item.unfold(2, df, df).unfold(3, df, df)
        item = blocks[:, :, :, :, df - 1, 0] # Select bottom-left

    elif method == "bottom_right":
        # Assumes df is valid integer > 0 and divides H, W
        blocks = item.unfold(2, df, df).unfold(3, df, df)
        item = blocks[:, :, :, :, df - 1, df - 1] # Select bottom-right

    elif method == "random":
        if generator is None:
            # It's good practice to ensure a generator is provided for a method named "random"
            raise ValueError("A generator must be provided for the 'random' downsample method.")

        # Assumes df is valid integer > 0 and divides H, W
        B, C, _, _ = item.shape # Get B, C needed for reshape/randint
        # unfold produces shape (B, C, new_h, new_w, df, df)
        blocks = item.unfold(2, df, df).unfold(3, df, df)
        # Reshape to flatten the block spatial dimensions
        blocks_flat = blocks.reshape(B, C, new_h, new_w, df * df)

        # Generate one random index per block location
        rand_indices = torch.randint(
            0, df * df,
            size=(B, C, new_h, new_w, 1), # Shape for gather
            device=item.device,
            generator=generator
        )
        # Gather the random element
        item = torch.gather(blocks_flat, dim=-1, index=rand_indices).squeeze(-1) # Squeeze result
  
    elif method == "uniform_random":
        blocks = item.unfold(2, df, df).unfold(3, df, df)  # (B, C, new_h, new_w, df, df)
        # Randomly choose ONE position (i,j) inside block
        rand_i = torch.randint(0, df, (1,)).item()  # Random row offset inside block
        rand_j = torch.randint(0, df, (1,)).item()  # Random column offset inside block
        item = blocks[:, :, :, :, rand_i, rand_j]  # Pick same (i,j) for all blocks
    elif method == "uniform_timestep":
        # Ensure timestep is provided for this method
        if timestep is None:
            raise ValueError("timestep must be provided for 'uniform_timestep' method")

        # --- deterministic randomness without touching the global RNG ---
        # Cache one generator per (device, timestep) pair ─ small & fast
        _gen_cache: dict[tuple[int, str], torch.Generator] = getattr(
            up_or_downsample, "_gen_cache", {}
        )
        key = (int(timestep), item.device.type)
        if key not in _gen_cache:
            g = torch.Generator(device=item.device)
            g.manual_seed(int(timestep))        # seed *local* generator once
            _gen_cache[key] = g
        rng = _gen_cache[key]
        up_or_downsample._gen_cache = _gen_cache  # store back for next call

        # unfold → (B, C, new_h, new_w, df, df)
        blocks = item.unfold(2, df, df).unfold(3, df, df)
        rand_i = torch.randint(0, df, (1,), generator=rng, device=item.device).item()
        rand_j = torch.randint(0, df, (1,), generator=rng, device=item.device).item()
        item = blocks[:, :, :, :, rand_i, rand_j]

    elif method == "linear_blend":
        # Enhanced linear blend: blend between any two specified methods
        # Default to original behavior (nearest-exact + avg_pool) for backward compatibility
        method1 = blend_method_1 if blend_method_1 is not None else "nearest-exact"
        method2 = blend_method_2 if blend_method_2 is not None else "avg_pool"
        
        if blend_factor is None:
            blend_factor = 0.5  # Default 50/50 blend
            
        # CFG-style extended range: >1.0 extrapolates beyond method1, <0.0 extrapolates beyond method2
        
        # Compute results for both methods
        try:
            result1 = apply_method(item, method1)
            result2 = apply_method(item, method2)
            
            # Linear interpolation: blend_factor * method1 + (1 - blend_factor) * method2
            # Extended range allows aggressive blending like CFG
            item = blend_factor * result1 + (1 - blend_factor) * result2
        except Exception as e:
            print(f"Error in linear_blend with methods {method1} and {method2}: {e}")
            # Fallback to area interpolation
            item = F.interpolate(item, size=(new_h, new_w), mode="area")

    else:
        item = F.interpolate(item, size=(new_h, new_w), mode=method)
    item = item.permute(0, 2, 3, 1)
    item = item.reshape(batch_size, new_h * new_w, -1)

    return item


def compute_merge(x: torch.Tensor, tome_info):

    args = tome_info["args"]

    if args["generator"] is None: # Make sure you have a generator ready
        args["generator"] = init_generator(x.device, fallback=args["generator"])


    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    # Existing logic to determine U-Net downsampling factor 'downsample'
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1]))) if x.shape[1] > 0 else 1
    dim = x.shape[-1]

    # Existing logic to get level-specific parameters
    if dim == 320: # Example dimension, adjust as per your model
        cur_level = "level_1"
        config_downsample_factor = tome_info['args']['downsample_factor']
        config_ratio = tome_info['args']['ratio']
    elif dim == 640: # Example dimension
        cur_level = "level_2"
        config_downsample_factor = tome_info['args']['downsample_factor_level_2']
        config_ratio = tome_info['args']['ratio_level_2']
    else:
        cur_level = "other"
        config_downsample_factor = 1
        config_ratio = 0.0

    cur_h, cur_w = original_h // downsample, original_w // downsample

    # Determine active merge method based on timestep
    if tome_info['timestep'] / 1000 > tome_info['args']['timestep_threshold_switch']:
        active_merge_method = args["merge_method"]
    else:
        active_merge_method = args["secondary_merge_method"]



    m, u = do_nothing, do_nothing # Initialize merge/unmerge functions
    merge_fn, unmerge_fn = m, u

    # Check if merging should happen at this level and timestep
    if cur_level != "other" and tome_info['timestep'] / 1000 > tome_info['args']['timestep_threshold_stop']:

        # --- Handle 'frequency' merge_method ---

        if active_merge_method == "frequency_blockwise" and config_downsample_factor > 1:
             selection_mode = args.get("frequency_selection_mode", "high")
             selection_method = args.get("frequency_selection_method", "1d_dft")
             ranking_method = args.get("frequency_ranking_method", "amplitude")
             blockwise_blend_factor = args.get("blockwise_blend_factor", 0.5)
             
             try:
                 # Check if blending is enabled (not default 0.5 or explicitly different from 1.0)
                 use_blending = blockwise_blend_factor != 1.0
                 
                 if use_blending:
                     # Use the new blended function
                     m = frequency_based_selection_blockwise_with_blend(
                         x,
                         downsampling_factor=config_downsample_factor,
                         H=cur_h,
                         W=cur_w,
                         blend_factor=blockwise_blend_factor,
                         mode=selection_mode,
                         selection_method=selection_method,
                         ranking_method=ranking_method
                         # timestep_normalized is less relevant here, but could be passed if needed
                     )
                 else:
                     # Use the original function for pure frequency selection (blend_factor = 1.0)
                     m = frequency_based_selection_blockwise(
                         x,
                         downsampling_factor=config_downsample_factor,
                         H=cur_h,
                         W=cur_w,
                         mode=selection_mode,
                         selection_method=selection_method,
                         ranking_method=ranking_method
                         # timestep_normalized is less relevant here, but could be passed if needed
                     )
                     
                 # Unmerge for blockwise selection is complex, typically not done.
                 # If needed, it would require storing the original positions and scattering back.
                 # For most applications like ToMe, unmerge isn't needed for this type.
                 u = do_nothing
             except ValueError as e:
                  print(f"Skipping blockwise frequency selection due to error: {e}")
                  m, u = do_nothing, do_nothing # Fallback if dimensions mismatch etc.
        elif active_merge_method == "frequency_global" and config_ratio > 0.0 and config_ratio < 1.0:
            # Assuming your original 'frequency_based_selection' is kept separately
            k = int(x.shape[1] * (1.0 - config_ratio)) # Tokens to KEEP
            selection_mode = args.get("frequency_selection_mode", "high")
            selection_method = args.get("frequency_selection_method", "1d_dft")
            ranking_method = args.get("frequency_ranking_method", "amplitude")
             
            # Handle batched timesteps correctly for frequency_global
            if 'timesteps_batch' in tome_info and tome_info['timesteps_batch'] is not None:
                # Use batch of timesteps - take mean for consensus behavior
                batch_timesteps = tome_info['timesteps_batch']
                timestep_norm = batch_timesteps.float().mean().item() / 1000.0
            else:
                # Fallback to single timestep
                timestep_norm = tome_info['timestep'] / 1000.0
             
            alpha_value = args.get("frequency_grid_alpha", 2.0) # Get alpha, default to 2.0

            try:
                # Use your original global selection function here
                m = frequency_based_selection( # Original global function
                    x, k, mode=selection_mode, selection_method=selection_method,
                    ranking_method=ranking_method, H=cur_h, W=cur_w,
                    timestep_normalized=timestep_norm,
                    alpha=alpha_value # Pass the retrieved alpha

                    )
                u = do_nothing
            except ValueError as e:
                print(f"Skipping global frequency selection due to error: {e}")
                m, u = do_nothing, do_nothing

        # --- Unified Masked Attention Handler ---
        elif "masked_attention" in active_merge_method:
            m, u = handle_masked_attention(
                active_merge_method, x, tome_info, cur_level, 
                config_ratio, config_downsample_factor, cur_h, cur_w
            )

        # --- Existing logic for 'downsample' ---
        elif active_merge_method == "downsample" and config_downsample_factor > 1:
            new_h, new_w = cur_h // config_downsample_factor, cur_w // config_downsample_factor
            # Ensure new dimensions are valid
            if new_h > 0 and new_w > 0:
                downsample_method = args["downsample_method"]
                # Get blend factor for linear_blend method (with dynamic timestep interpolation support)
                if downsample_method == "linear_blend":
                    blend_factor = compute_dynamic_blend_factor(
                        timestep_value=tome_info.get("timestep"),
                        enable_interpolation=args.get("linear_blend_timestep_interpolation", False),
                        start_ratio=args.get("linear_blend_start_ratio", 0.1),
                        end_ratio=args.get("linear_blend_end_ratio", 0.9),
                        static_factor=args.get("linear_blend_factor", 0.5),
                        log_key="main_linear_blend"
                    )
                else:
                    blend_factor = None
                
                # Get blend method parameters for linear_blend
                blend_method_1 = args.get("linear_blend_method_1", None)
                blend_method_2 = args.get("linear_blend_method_2", None)
                
                if downsample_method == "uniform_timestep":
                    current_timestep = tome_info["timestep"]  # <<< Get current timestep
                    m = lambda y: up_or_downsample(y, cur_w, cur_h, new_w, new_h, downsample_method, timestep=current_timestep, blend_factor=blend_factor, blend_method_1=blend_method_1, blend_method_2=blend_method_2)
                    u = lambda y: up_or_downsample(y, new_w, new_h, cur_w, cur_h, downsample_method, timestep=current_timestep, blend_factor=blend_factor, blend_method_1=blend_method_1, blend_method_2=blend_method_2)
                else:
                    m = lambda y: up_or_downsample(y, cur_w, cur_h, new_h, new_w, downsample_method, generator=args["generator"], blend_factor=blend_factor, blend_method_1=blend_method_1, blend_method_2=blend_method_2)
                    u = lambda y: up_or_downsample(y, new_w, new_h, cur_w, cur_h, downsample_method, blend_factor=blend_factor, blend_method_1=blend_method_1, blend_method_2=blend_method_2)
        # --- Existing logic for 'similarity' ---
        elif active_merge_method == "similarity" and config_ratio > 0.0 and config_ratio < 1.0:
            w = int(math.ceil(original_w / downsample)) # U-Net downsampling
            h = int(math.ceil(original_h / downsample)) # U-Net downsampling
            r = int(x.shape[1] * config_ratio) # Number of tokens to REMOVE

            if r > 0: # Ensure r is valid before proceeding
                if args["generator"] is None or args["generator"].device != x.device:
                    args["generator"] = init_generator(x.device, fallback=args["generator"])

                use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
                m, u = bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r,
                                                        no_rand=not use_rand, generator=args["generator"])
            # If r is 0 or less, m, u remain do_nothing

        # If no conditions met, m, u remain do_nothing implicitly from initialization
    # else: # If outside active levels/timesteps, m, u remain do_nothing

    
        elif active_merge_method == "block_avg_pool" and config_downsample_factor > 1:
            m, u = block_average_pool_unpool(cur_w, cur_h, config_downsample_factor)

        # """modification start"""
        elif active_merge_method == "downsample_qkv_upsample_out" and config_downsample_factor > 1:
            w_original = cur_w
            h_original = cur_h
            w_downsampled = w_original // config_downsample_factor
            h_downsampled = h_original // config_downsample_factor

            if w_downsampled > 0 and h_downsampled > 0:
                qkv_down_method = args.get("qkv_downsample_method", "avg_pool")
                out_up_method = args.get("out_upsample_method", "nearest")
                
                # Get blend factors for linear_blend methods (with dynamic timestep interpolation support)
                if qkv_down_method == "linear_blend":
                    qkv_blend_factor = compute_dynamic_blend_factor(
                        timestep_value=tome_info.get("timestep"),
                        enable_interpolation=args.get("qkv_linear_blend_timestep_interpolation", False),
                        start_ratio=args.get("qkv_linear_blend_start_ratio", 0.1),
                        end_ratio=args.get("qkv_linear_blend_end_ratio", 0.9),
                        static_factor=args.get("qkv_linear_blend_factor", 0.5),
                        log_key="qkv_linear_blend"
                    )
                else:
                    qkv_blend_factor = None
                    
                if out_up_method == "linear_blend":
                    out_blend_factor = compute_dynamic_blend_factor(
                        timestep_value=tome_info.get("timestep"),
                        enable_interpolation=args.get("out_linear_blend_timestep_interpolation", False),
                        start_ratio=args.get("out_linear_blend_start_ratio", 0.1),
                        end_ratio=args.get("out_linear_blend_end_ratio", 0.9),
                        static_factor=args.get("out_linear_blend_factor", 0.5),
                        log_key="out_linear_blend"
                    )
                else:
                    out_blend_factor = None
                
                # Get blend method parameters for linear_blend methods
                qkv_blend_method_1 = args.get("qkv_linear_blend_method_1", None)
                qkv_blend_method_2 = args.get("qkv_linear_blend_method_2", None)
                out_blend_method_1 = args.get("out_linear_blend_method_1", None)
                out_blend_method_2 = args.get("out_linear_blend_method_2", None)

                # Merge function (m): Downsamples from original (cur_w, cur_h) to (w_downsampled, h_downsampled)
                m = lambda item_to_downsample: up_or_downsample(
                    item_to_downsample,
                    cur_w=w_original, cur_h=h_original,
                    new_w=w_downsampled, new_h=h_downsampled,
                    method=qkv_down_method,
                    timestep=tome_info["timestep"],
                    blend_factor=qkv_blend_factor,
                    blend_method_1=qkv_blend_method_1,
                    blend_method_2=qkv_blend_method_2
                )

                # Unmerge function (u): Upsamples from (w_downsampled, h_downsampled) back to original (cur_w, cur_h)
                u = lambda item_to_upsample: up_or_downsample(
                    item_to_upsample,
                    cur_w=w_downsampled, cur_h=h_downsampled,
                    new_w=w_original, new_h=h_original,
                    method=out_up_method,
                    timestep=tome_info["timestep"] if out_up_method == "uniform_timestep" else None,
                    blend_factor=out_blend_factor,
                    blend_method_1=out_blend_method_1,
                    blend_method_2=out_blend_method_2
                )
            else:
                # Not enough tokens to downsample effectively for QKV
                m, u = do_nothing, do_nothing
        # """modification end"""
        merge_fn, unmerge_fn = m, u
        # --- End Modification ---

    return merge_fn, unmerge_fn


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, 
                                     h: int, 
                                     sx: int, 
                                     sy: int, 
                                     r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy * sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(
                metric.device)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = torch.gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = torch.gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = torch.gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = torch.gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = torch.gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = torch.gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2,
                     index=torch.gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c),
                     src=unm)
        out.scatter_(dim=-2,
                     index=torch.gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c),
                     src=src)

        return out

    return merge, unmerge


class TokenMergeAttentionProcessor:
    def __init__(self, attn_method="regular"):
        # Default to vanilla attention for fair benchmarking.
        # Pass "auto" to use the old behavior (torch2 > xformers > regular).
        if attn_method == "auto":
            if torch2_is_available:
                self.attn_method = "torch2"
            elif xformers_is_available:
                self.attn_method = "xformers"
            else:
                self.attn_method = "regular"
        else:
            self.attn_method = attn_method
        
        # _tome_info will be set by patch_attention_proc() immediately after instantiation
        # This initialization is just to satisfy type checkers - it should never be None at runtime
        self._tome_info = None  # Will be overridden by patching process

    def reset_logging_flags(self):
        """Reset logging flags for a new image to allow fresh logging."""
        attrs_to_remove = [attr for attr in dir(self) if attr.startswith('_skip_logged_') or 
                          attr.startswith('_success_logged_') or attr.startswith('_exp_success_logged_') or
                          attr.startswith('_batch_exp_logged_')]
        for attr in attrs_to_remove:
            delattr(self, attr)

    def torch2_attention(self, attn, query, key, value, attention_mask, batch_size):
        inner_dim=key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        return hidden_states

    def xformers_attention(self, attn, query, key, value, attention_mask, batch_size):
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size * attn.heads, -1, attention_mask.shape[-1])

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, scale=attn.scale
        )

        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states


    def regular_attention(self, attn, query, key, value, attention_mask, batch_size):
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size * attn.heads, -1, attention_mask.shape[-1])

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        # Ensure the processor was properly patched
        if self._tome_info is None:
            raise RuntimeError("TokenMergeAttentionProcessor was not properly patched. Call patch_attention_proc() first.")
        
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)

        if self._tome_info['args']['merge_tokens'] == "all":
            merge_fn, unmerge_fn = compute_merge(hidden_states, self._tome_info)
            hidden_states = merge_fn(hidden_states)

        #query = attn.to_q(hidden_states, *args)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # --- Determine if this is self-attention or cross-attention ---
        is_self_attention = (encoder_hidden_states is hidden_states) or (
            encoder_hidden_states is not None and 
            torch.equal(encoder_hidden_states, hidden_states)
        )

        """if self._tome_info['args']['merge_tokens'] == "keys/values":
            merge_fn, _ = compute_merge(encoder_hidden_states, self._tome_info)
            encoder_hidden_states = merge_fn(encoder_hidden_states)"""
        
        # --- Block for 'keys/values' merging - MODIFIED ---
        if self._tome_info['args']['merge_tokens'] == "keys/values":
            # Check if we're using masked attention
            if self._tome_info['args']['merge_method'] == "masked_attention":
                # For masked attention, compute merge to generate the mask
                # but don't apply it yet - it will be used when computing K and V
                merge_fn, _ = compute_merge(encoder_hidden_states, self._tome_info)
                # encoder_hidden_states remains unchanged for masked attention
            else:
                # """modification start"""

                # Determine the source tensor for computing merge indices
                selection_source = self._tome_info['args'].get("selection_source", "hidden") # 'hidden', 'query', 'key', 'value'

                if selection_source == "hidden":
                    # Use the prepared encoder_hidden_states
                    source_tensor_for_computation = encoder_hidden_states
                elif selection_source == "query":
                    # Use the query calculated earlier (based on potentially 'all'-merged hidden_states)
                    source_tensor_for_computation = query
                elif selection_source == "key":
                    # Calculate key based on the *current* state of encoder_hidden_states for computation purposes
                    key_for_computation = attn.to_k(encoder_hidden_states)
                    source_tensor_for_computation = key_for_computation
                elif selection_source == "value":
                    # Calculate value based on the *current* state of encoder_hidden_states for computation purposes
                    value_for_computation = attn.to_v(encoder_hidden_states)
                    source_tensor_for_computation = value_for_computation
                else:
                    # Fallback or error handling - default to hidden
                    print(f"Warning: Unknown selection_source '{selection_source}', defaulting to 'hidden'.")
                    source_tensor_for_computation = encoder_hidden_states

                # Compute the merge function based on the selected source tensor
                merge_fn, _ = compute_merge(source_tensor_for_computation, self._tome_info)
                # """modification end"""

                # Apply the computed merge function to encoder_hidden_states
                # This modifies the tensor that K and V will be derived from below
                encoder_hidden_states = merge_fn(encoder_hidden_states)
        # -------------------------------------------------------


        #key = attn.to_k(encoder_hidden_states, *args)
        
        # Check if we're using masked attention for keys/values
        if (self._tome_info['args']['merge_tokens'] == "keys/values" and 
            self._tome_info['args']['merge_method'] == "masked_attention" and
            'token_mask' in self._tome_info and
            is_self_attention):  # Only apply to self-attention layers
            
            # At this point encoder_hidden_states is guaranteed to be non-None due to the logic above
            assert encoder_hidden_states is not None, "encoder_hidden_states should not be None here"
            
            # Get the mask and validate dimensions
            mask = self._tome_info['token_mask']  # Shape: (B, N_mask)
            B_enc, N_enc, C_enc = encoder_hidden_states.shape
            B_mask, N_mask = mask.shape
            
            # Only apply masked attention to 4096-token layers (64x64 resolution)
            if N_enc != 4096:
                # Skip masked attention for non-4096 token layers
                # Only log once per layer size to avoid spam
                skip_key = f"_skip_resolution_logged_{N_enc}"
                if not hasattr(self, skip_key):
                    print(f"Skipping masked attention: layer has {N_enc} tokens (only applies to 4096-token layers)")
                    setattr(self, skip_key, True)
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
            elif N_mask != 4096:
                # Mask was computed for wrong resolution
                skip_key = f"_skip_mask_size_logged_{N_mask}"
                if not hasattr(self, skip_key):
                    print(f"Skipping masked attention: mask has {N_mask} tokens (expected 4096)")
                    setattr(self, skip_key, True)
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
            elif B_mask != B_enc:
                # Batch size mismatch - this shouldn't happen with the new logic
                skip_key = f"_skip_batch_mismatch_logged_{B_mask}_{B_enc}"
                if not hasattr(self, skip_key):
                    print(f"Unexpected batch size mismatch: mask batch {B_mask} vs encoder batch {B_enc}. Skipping masked attention.")
                    setattr(self, skip_key, True)
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
            else:
                # Perfect match - apply masked attention
                try:
                    kept_tokens_list = []
                    kept_counts = []
                    
                    for batch_idx in range(B_enc):
                        kept_indices = torch.where(mask[batch_idx])[0]
                        
                        # Validate indices are within bounds
                        if len(kept_indices) > 0:
                            max_idx = torch.max(kept_indices).item()
                            if max_idx >= N_enc:
                                print(f"Index out of bounds: max_idx={max_idx}, N_enc={N_enc}")
                                raise IndexError("Index out of bounds")
                        
                        kept_tokens = encoder_hidden_states[batch_idx, kept_indices]
                        kept_tokens_list.append(kept_tokens)
                        kept_counts.append(len(kept_indices))
                    
                    # Ensure consistent token counts across batches
                    if len(set(kept_counts)) > 1:
                        min_kept = min(kept_counts)
                        kept_tokens_list = [tokens[:min_kept] for tokens in kept_tokens_list]
                        num_kept = min_kept
                        print(f"Adjusted to minimum kept tokens: {num_kept}")
                    else:
                        num_kept = kept_counts[0]
                    
                    if num_kept > 0:
                        # Stack and compute K,V for kept tokens only
                        kept_encoder_hidden_states = torch.stack(kept_tokens_list, dim=0)
                        key = attn.to_k(kept_encoder_hidden_states)
                        value = attn.to_v(kept_encoder_hidden_states)
                        # Only print success message once per configuration
                        success_key = f"_success_logged_{N_enc}_{num_kept}"
                        if not hasattr(self, success_key):
                            print(f"Masked attention applied: {N_enc} → {num_kept} tokens ({num_kept/N_enc:.1%})")
                            setattr(self, success_key, True)
                    else:
                        print("No tokens kept, using normal computation")
                        key = attn.to_k(encoder_hidden_states)
                        value = attn.to_v(encoder_hidden_states)
                        
                except Exception as e:
                    print(f"Error in masked attention: {e}")
                    key = attn.to_k(encoder_hidden_states)
                    value = attn.to_v(encoder_hidden_states)
        else:
            # Normal computation (no masked attention)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if self.attn_method == "torch2":
            hidden_states = self.torch2_attention(attn, query, key, value, attention_mask, batch_size)
        elif self.attn_method == "xformers":
            hidden_states = self.xformers_attention(attn, query, key, value, attention_mask, batch_size)
        else:
            hidden_states = self.regular_attention(attn, query, key, value, attention_mask, batch_size)

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        #hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if self._tome_info['args']['merge_tokens'] == "all":
            hidden_states = unmerge_fn(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states





def precompute_fixed_mask_from_original(
    original_x0: torch.Tensor,
    tome_info: dict,
    target_level: str = "level_1",
    # New parameters for SNR and noise magnitude mask computation
    original_noise: Optional[torch.Tensor] = None
):
    """
    Precompute and store a fixed mask based on the original clean image.
    This mask will be reused for ALL timesteps and conditions during classification.
    
    Args:
        original_x0: Clean image latent representation (B, N, C)
        tome_info: ToMe info dictionary to store the mask
        target_level: Level to apply the mask (default: "level_1")
    """
    args = tome_info["args"]
    
    # Only proceed if using any of the masked attention methods
    supported_methods = ["masked_attention", "blockwise_masked_attention", 
                         "snr_masked_attention", "snr_blockwise_masked_attention",
                         "noise_magnitude_masked_attention", "noise_magnitude_blockwise_masked_attention"]
    if args["merge_method"] not in supported_methods:
        return
    
    # Only generate mask for 4096-token layers
    if original_x0.shape[1] != 4096:
        print(f"Skipping mask precomputation: input has {original_x0.shape[1]} tokens (only precompute for 4096-token layers)")
        return
    
    # Handle case where size hasn't been set yet (e.g., during FLOP measurement)
    if tome_info["size"] is None:
        # For 4096 tokens, assume standard 512x512 image -> 64x64 latent
        original_h, original_w = 512, 512
        print(f"Size not set, using default 512x512 for mask precomputation")
    else:
        original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // original_x0.shape[1]))) if original_x0.shape[1] > 0 else 1
    dim = original_x0.shape[-1]
    
    # For precomputation, we can compute masks for any channel dimension
    # The mask is based on spatial features, not channel-specific features
    if target_level == "level_1":
        cur_level = "level_1"
        config_ratio = tome_info['args']['ratio']
        config_downsample_factor = tome_info['args']['downsample_factor']
    elif target_level == "level_2":
        cur_level = "level_2"
        config_ratio = tome_info['args']['ratio_level_2']
        config_downsample_factor = tome_info['args']['downsample_factor_level_2']
    else:
        return  # Not the target level
    
    cur_h, cur_w = original_h // downsample, original_w // downsample
    
    # Handle both masked_attention and blockwise_masked_attention
    if args["merge_method"] == "masked_attention" and config_ratio > 0.0 and config_ratio < 1.0:
        mask_key = f"fixed_mask_level_{cur_level}"
        
        # Get selection parameters
        selection_mode = args.get("frequency_selection_mode", "high")
        selection_method = args.get("frequency_selection_method", "1d_dft")
        ranking_method = args.get("frequency_ranking_method", "amplitude")
        alpha_value = args.get("frequency_grid_alpha", 2.0)
        
        try:
            # Generate the FIXED mask based on ORIGINAL CLEAN IMAGE
            # Use only the first sample to ensure batch size 1
            x0_single = original_x0[:1]  # Ensure batch size 1
            mask = frequency_based_token_mask(
                x0_single, 
                reduction_ratio=config_ratio,
                selection_method=selection_method,
                mode=selection_mode,
                ranking_method=ranking_method,
                H=cur_h, W=cur_w,
                timestep_normalized=None,
                alpha=alpha_value
            )
            
            # Store the mask for reuse across ALL timesteps and conditions
            tome_info[mask_key] = mask
            num_kept = mask.sum(dim=1).item()
            print(f"Precomputed fixed mask for {cur_level}: {original_x0.shape[1]} → {num_kept} tokens ({num_kept/original_x0.shape[1]:.1%}) [dim={dim}]")
            
        except Exception as e:
            print(f"Error precomputing fixed mask: {e}")
            # Fallback: keep all tokens
            mask = torch.ones(1, original_x0.shape[1], dtype=torch.bool, device=original_x0.device)
            tome_info[mask_key] = mask
            
    elif args["merge_method"] == "blockwise_masked_attention" and config_downsample_factor > 1:
        mask_key = f"fixed_mask_blockwise_level_{cur_level}"
        
        # Get selection parameters
        selection_mode = args.get("frequency_selection_mode", "high")
        selection_method = args.get("frequency_selection_method", "1d_dft")
        ranking_method = args.get("frequency_ranking_method", "amplitude")
        
        try:
            # Generate the FIXED BLOCKWISE mask based on ORIGINAL CLEAN IMAGE
            # Use only the first sample to ensure batch size 1
            x0_single = original_x0[:1]  # Ensure batch size 1
            mask = frequency_based_token_mask_blockwise(
                x0_single, 
                downsampling_factor=config_downsample_factor,
                H=cur_h, W=cur_w,
                selection_method=selection_method,
                mode=selection_mode,
                ranking_method=ranking_method,
                timestep_normalized=None
            )
            
            # Store the mask for reuse across ALL timesteps and conditions
            tome_info[mask_key] = mask
            num_kept = mask.sum(dim=1).item()
            print(f"Precomputed fixed blockwise mask for {cur_level}: {original_x0.shape[1]} → {num_kept} tokens ({num_kept/original_x0.shape[1]:.1%}) [dim={dim}] [factor={config_downsample_factor}]")
            
        except Exception as e:
            print(f"Error precomputing fixed blockwise mask: {e}")
            # Fallback: keep all tokens
            mask = torch.ones(1, original_x0.shape[1], dtype=torch.bool, device=original_x0.device)
            tome_info[mask_key] = mask
            
    # --- NEW: SNR and Noise Magnitude Mask Precomputation ---
    elif args["merge_method"] == "snr_masked_attention" and config_ratio > 0.0 and config_ratio < 1.0:
        mask_key = f"fixed_snr_mask_level_{cur_level}"
        
        # Validate required parameters
        if original_noise is None:
            print("ERROR: original_noise must be provided for SNR mask precomputation")
            return
            
        # Get selection parameters
        selection_mode = args.get("frequency_selection_mode", "high")
        ranking_method = args.get("frequency_ranking_method", "amplitude")
        alpha_value = args.get("frequency_grid_alpha", 2.0)
        
        try:
            # Generate the FIXED SNR mask based on ORIGINAL CLEAN IMAGE and NOISE
            x0_single = original_x0[:1]  # Ensure batch size 1
            noise_single = original_noise[:1]  # Ensure batch size 1
            
            mask = frequency_based_token_mask(
                x0_single, 
                reduction_ratio=config_ratio,
                selection_method="snr",
                mode=selection_mode,
                ranking_method=ranking_method,
                H=cur_h, W=cur_w,
                timestep_normalized=None,
                alpha=alpha_value,
                clean_signal=x0_single,
                noise=noise_single
            )
            
            # Store the mask for reuse across ALL timesteps and conditions
            tome_info[mask_key] = mask
            num_kept = mask.sum(dim=1).item()
            print(f"Precomputed fixed SNR mask for {cur_level}: {original_x0.shape[1]} → {num_kept} tokens ({num_kept/original_x0.shape[1]:.1%}) [dim={dim}]")
            
        except Exception as e:
            print(f"Error precomputing fixed SNR mask: {e}")
            # Fallback: keep all tokens
            mask = torch.ones(1, original_x0.shape[1], dtype=torch.bool, device=original_x0.device)
            tome_info[mask_key] = mask
            
    elif args["merge_method"] == "snr_blockwise_masked_attention" and config_downsample_factor > 1:
        mask_key = f"fixed_snr_blockwise_mask_level_{cur_level}"
        
        # Validate required parameters
        if original_noise is None:
            print("ERROR: original_noise must be provided for SNR blockwise mask precomputation")
            return
            
        # Get selection parameters
        selection_mode = args.get("frequency_selection_mode", "high")
        ranking_method = args.get("frequency_ranking_method", "amplitude")
        
        try:
            # Generate the FIXED SNR BLOCKWISE mask based on ORIGINAL CLEAN IMAGE and NOISE
            x0_single = original_x0[:1]  # Ensure batch size 1
            noise_single = original_noise[:1]  # Ensure batch size 1
            
            mask = frequency_based_token_mask_blockwise(
                x0_single, 
                downsampling_factor=config_downsample_factor,
                H=cur_h, W=cur_w,
                selection_method="snr",
                mode=selection_mode,
                ranking_method=ranking_method,
                timestep_normalized=None,
                clean_signal=x0_single,
                noise=noise_single
            )
            
            # Store the mask for reuse across ALL timesteps and conditions
            tome_info[mask_key] = mask
            num_kept = mask.sum(dim=1).item()
            print(f"Precomputed fixed SNR blockwise mask for {cur_level}: {original_x0.shape[1]} → {num_kept} tokens ({num_kept/original_x0.shape[1]:.1%}) [dim={dim}] [factor={config_downsample_factor}]")
            
        except Exception as e:
            print(f"Error precomputing fixed SNR blockwise mask: {e}")
            # Fallback: keep all tokens
            mask = torch.ones(1, original_x0.shape[1], dtype=torch.bool, device=original_x0.device)
            tome_info[mask_key] = mask
            
    elif args["merge_method"] == "noise_magnitude_masked_attention" and config_ratio > 0.0 and config_ratio < 1.0:
        mask_key = f"fixed_noise_magnitude_mask_level_{cur_level}"
        
        # Validate required parameters
        if original_noise is None:
            print("ERROR: original_noise must be provided for noise magnitude mask precomputation")
            return
            
        # Get selection parameters
        selection_mode = args.get("frequency_selection_mode", "high")
        ranking_method = args.get("frequency_ranking_method", "amplitude")
        alpha_value = args.get("frequency_grid_alpha", 2.0)
        
        try:
            # Generate the FIXED NOISE MAGNITUDE mask based on ORIGINAL NOISE
            x0_single = original_x0[:1]  # Ensure batch size 1
            noise_single = original_noise[:1]  # Ensure batch size 1
            
            mask = frequency_based_token_mask(
                x0_single, 
                reduction_ratio=config_ratio,
                selection_method="noise_magnitude",
                mode=selection_mode,
                ranking_method=ranking_method,
                H=cur_h, W=cur_w,
                timestep_normalized=None,
                alpha=alpha_value,
                clean_signal=x0_single,
                noise=noise_single
            )
            
            # Store the mask for reuse across ALL timesteps and conditions
            tome_info[mask_key] = mask
            num_kept = mask.sum(dim=1).item()
            print(f"Precomputed fixed noise magnitude mask for {cur_level}: {original_x0.shape[1]} → {num_kept} tokens ({num_kept/original_x0.shape[1]:.1%}) [dim={dim}]")
            
        except Exception as e:
            print(f"Error precomputing fixed noise magnitude mask: {e}")
            # Fallback: keep all tokens
            mask = torch.ones(1, original_x0.shape[1], dtype=torch.bool, device=original_x0.device)
            tome_info[mask_key] = mask
            
    elif args["merge_method"] == "noise_magnitude_blockwise_masked_attention" and config_downsample_factor > 1:
        mask_key = f"fixed_noise_magnitude_blockwise_mask_level_{cur_level}"
        
        # Validate required parameters
        if original_noise is None:
            print("ERROR: original_noise must be provided for noise magnitude blockwise mask precomputation")
            return
            
        # Get selection parameters
        selection_mode = args.get("frequency_selection_mode", "high")
        ranking_method = args.get("frequency_ranking_method", "amplitude")
        
        try:
            # Generate the FIXED NOISE MAGNITUDE BLOCKWISE mask based on ORIGINAL NOISE
            x0_single = original_x0[:1]  # Ensure batch size 1
            noise_single = original_noise[:1]  # Ensure batch size 1
            
            mask = frequency_based_token_mask_blockwise(
                x0_single, 
                downsampling_factor=config_downsample_factor,
                H=cur_h, W=cur_w,
                selection_method="noise_magnitude",
                mode=selection_mode,
                ranking_method=ranking_method,
                timestep_normalized=None,
                clean_signal=x0_single,
                noise=noise_single
            )
            
            # Store the mask for reuse across ALL timesteps and conditions
            tome_info[mask_key] = mask
            num_kept = mask.sum(dim=1).item()
            print(f"Precomputed fixed noise magnitude blockwise mask for {cur_level}: {original_x0.shape[1]} → {num_kept} tokens ({num_kept/original_x0.shape[1]:.1%}) [dim={dim}] [factor={config_downsample_factor}]")
            
        except Exception as e:
            print(f"Error precomputing fixed noise magnitude blockwise mask: {e}")
            # Fallback: keep all tokens
            mask = torch.ones(1, original_x0.shape[1], dtype=torch.bool, device=original_x0.device)
            tome_info[mask_key] = mask


def reset_fixed_masks(tome_info: dict):
    """
    Reset stored fixed masks between different images, but keep global logging flags for run-wide logging.
    """
    keys_to_remove = [key for key in tome_info.keys() if key.startswith("fixed_mask_") or 
                      key.startswith("fixed_snr_") or key.startswith("fixed_noise_magnitude_")]
    for key in keys_to_remove:
        del tome_info[key]
    
    # Also reset batch expansion flags for new image
    batch_flags_to_reset = ['_batch_expand_logged', '_batch_blockwise_expand_logged', 
                           '_batch_snr_expand_logged', '_batch_snr_blockwise_expand_logged',
                           '_batch_noise_magnitude_expand_logged', '_batch_noise_magnitude_blockwise_expand_logged']
    for flag in batch_flags_to_reset:
        if flag in tome_info:
            del tome_info[flag]
    
    if keys_to_remove:
        print(f"Reset {len(keys_to_remove)} fixed masks for new image")




