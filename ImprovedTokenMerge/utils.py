import torch
try:
    from .merge import TokenMergeAttentionProcessor
except ImportError:
    from merge import TokenMergeAttentionProcessor
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor, AttnProcessor
import torch.nn.functional as F

if is_xformers_available():
    xformers_is_available = True
else:
    xformers_is_available = False

if hasattr(F, "scaled_dot_product_attention"):
    torch2_is_available = True
else:
    torch2_is_available = False


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """

    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])

        timestep_arg = args[1]
        if torch.is_tensor(timestep_arg):
            if timestep_arg.ndim == 0:
                # Handle 0-dim tensor (scalar)
                module._tome_info["timestep"] = timestep_arg.item()
                module._tome_info["timesteps_batch"] = None
            elif timestep_arg.numel() == 1:
                # Handle 1-element tensor (potentially multi-dim but with one value)
                module._tome_info["timestep"] = timestep_arg.reshape(-1)[0].item()
                module._tome_info["timesteps_batch"] = None
            else:
                # Handle tensor with multiple elements - store the full batch
                # Store the first timestep for backward compatibility
                module._tome_info["timestep"] = timestep_arg.reshape(-1)[0].item()
                # Store the full batch of timesteps for proper processing
                module._tome_info["timesteps_batch"] = timestep_arg
        elif isinstance(timestep_arg, (int, float)):
            # Handle if timestep is passed as a standard Python number
            module._tome_info["timestep"] = float(timestep_arg)
            module._tome_info["timesteps_batch"] = None
        else:
            # Log an error or raise exception for unexpected type
            print(f"Error: Unexpected timestep type in hook: {type(timestep_arg)}")
            # Assign a default or raise an error
            module._tome_info["timestep"] = 0 # Example default
            module._tome_info["timesteps_batch"] = None

        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))

def remove_patch(pipe: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """

    if hasattr(pipe.unet, "_tome_info"):
        del pipe.unet._tome_info

    for n,m in pipe.unet.named_modules():
        if hasattr(m, "processor"):
            m.processor = AttnProcessor2_0()

def patch_attention_proc(unet, token_merge_args={}):
    unet._tome_info = {
        "size": None,
        "timestep": None,
        "hooks": [],
        "args": {
            # --- Existing Arguments ---
            "ratio": token_merge_args.get("ratio", 0.5),  # ratio of tokens to merge (for similarity and frequency_global)
            "sx": token_merge_args.get("sx", 2),  # stride x for sim calculation
            "sy": token_merge_args.get("sy", 2),  # stride y for sim calculation
            "use_rand": token_merge_args.get("use_rand", True), # for similarity method randomness
            "generator": None, # RNG state for similarity

            "merge_tokens": token_merge_args.get("merge_tokens", "keys/values"),  # ["all", "keys/values"] Where to apply merging
            "merge_method": token_merge_args.get("merge_method", "downsample"),
            # Options: ["none", "similarity", "downsample", "frequency_blockwise", "frequency_global", "block_avg_pool", "downsample_qkv_upsample_out"]
                      
            "downsample_method": token_merge_args.get("downsample_method", "nearest-exact"), # Interpolation for "downsample"
            "downsample_factor_h": token_merge_args.get("downsample_factor_h", 1),
            "downsample_factor_w": token_merge_args.get("downsample_factor_w", 2),
            "downsample_factor": token_merge_args.get("downsample_factor", 2),  # Downsampling factor for "downsample" and "frequency_blockwise"
            "timestep_threshold_switch": token_merge_args.get("timestep_threshold_switch", 0.2), # Switch point (normalized t)
            "timestep_threshold_stop": token_merge_args.get("timestep_threshold_stop", 0.0), # Stop merging point (normalized t)
            "secondary_merge_method": token_merge_args.get("secondary_merge_method", "similarity"),
            # Options: ["none", "similarity", "downsample", "frequency_blockwise", "frequency_global", "block_avg_pool", "downsample_qkv_upsample_out"]
       
            "downsample_factor_level_2": token_merge_args.get("downsample_factor_level_2", 1), # level 2 downsample factor override
            "ratio_level_2": token_merge_args.get("ratio_level_2", 0.5), # level 2 ratio override

            # --- Added Arguments for Frequency Methods ---
            "frequency_selection_mode": token_merge_args.get("frequency_selection_mode", "high"), # ['high', 'low', 'timestep_scheduler', 'reverse_timestep_scheduler']
            "frequency_selection_method": token_merge_args.get("frequency_selection_method", "1d_dft"), # ['1d_dft', '1d_dct', '2d_conv','original']
            "frequency_ranking_method": token_merge_args.get("frequency_ranking_method", "amplitude"), # ['amplitude', 'spectral_centroid','variance','l1norm','l2norm']
            "selection_source": token_merge_args.get("selection_source", "query"), # ['hidden', 'key', 'query', 'value']
            "frequency_grid_alpha": token_merge_args.get("frequency_grid_alpha", 2),
            # New arguments for the "downsample_qkv_upsample_out" method
            "qkv_downsample_method": token_merge_args.get("qkv_downsample_method", "nearest"), # Method for downsampling QKV source (e.g., "avg_pool", "max_pool")
            "out_upsample_method": token_merge_args.get("out_upsample_method", "nearest"),     # Method for upsampling attention output (e.g., "nearest", "bilinear")
            
            # Blockwise blend factor for frequency_blockwise method
            "blockwise_blend_factor": token_merge_args.get("blockwise_blend_factor", 0.5),    # Blend factor for linear interpolation between frequency-selected and average-pooled tokens
            
            # Linear blend factors for linear_blend downsample method
            "linear_blend_factor": token_merge_args.get("linear_blend_factor", 0.5),          # Blend factor for linear_blend downsample method (0.0=avg_pool, 0.5=50/50, 1.0=nearest-exact)
            "qkv_linear_blend_factor": token_merge_args.get("qkv_linear_blend_factor", 0.5),  # Blend factor for QKV linear_blend in downsample_qkv_upsample_out method
            "out_linear_blend_factor": token_merge_args.get("out_linear_blend_factor", 0.5),  # Blend factor for output linear_blend in downsample_qkv_upsample_out method
            
            # Timestep-based interpolation for linear blend
            "linear_blend_timestep_interpolation": token_merge_args.get("linear_blend_timestep_interpolation", False),  # Enable timestep-based interpolation
            "linear_blend_start_ratio": token_merge_args.get("linear_blend_start_ratio", 0.1),  # Start ratio at timestep 999 (high noise)
            "linear_blend_end_ratio": token_merge_args.get("linear_blend_end_ratio", 0.9),    # End ratio at timestep 0 (low noise)
            "qkv_linear_blend_timestep_interpolation": token_merge_args.get("qkv_linear_blend_timestep_interpolation", False),  # Enable for QKV
            "qkv_linear_blend_start_ratio": token_merge_args.get("qkv_linear_blend_start_ratio", 0.1),
            "qkv_linear_blend_end_ratio": token_merge_args.get("qkv_linear_blend_end_ratio", 0.9),
            "out_linear_blend_timestep_interpolation": token_merge_args.get("out_linear_blend_timestep_interpolation", False),  # Enable for output
            "out_linear_blend_start_ratio": token_merge_args.get("out_linear_blend_start_ratio", 0.1),
            "out_linear_blend_end_ratio": token_merge_args.get("out_linear_blend_end_ratio", 0.9),
            
            # Linear blend method selection parameters
            "linear_blend_method_1": token_merge_args.get("linear_blend_method_1", "nearest-exact"),     # First method for linear_blend interpolation
            "linear_blend_method_2": token_merge_args.get("linear_blend_method_2", "avg_pool"),         # Second method for linear_blend interpolation
            "qkv_linear_blend_method_1": token_merge_args.get("qkv_linear_blend_method_1", "nearest-exact"),  # First method for QKV linear_blend
            "qkv_linear_blend_method_2": token_merge_args.get("qkv_linear_blend_method_2", "avg_pool"),       # Second method for QKV linear_blend
            "out_linear_blend_method_1": token_merge_args.get("out_linear_blend_method_1", "nearest-exact"),  # First method for output linear_blend
            "out_linear_blend_method_2": token_merge_args.get("out_linear_blend_method_2", "avg_pool"),       # Second method for output linear_blend
        
        
        
        }
    }
    hook_tome_model(unet)
    attn_modules = [module for name, module in unet.named_modules() if module.__class__.__name__ == 'BasicTransformerBlock']

    for i, module in enumerate(attn_modules):
        # IMPORTANT: Only apply to self-attention (attn1), NOT cross-attention (attn2)
        # This prevents masked attention from being applied to cross-attention layers
        module.attn1.processor = TokenMergeAttentionProcessor(attn_method="regular")
        module.attn1.processor._tome_info = unet._tome_info
        
        # Keep cross-attention (attn2) with default processor for text conditioning
        # Don't modify attn2 to avoid applying masked attention to text-image cross-attention

def remove_patch(pipe: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """

    # Always reset to vanilla attention for fair benchmarking
    for n,m in pipe.unet.named_modules():
        if hasattr(m, "processor"):
            m.processor = AttnProcessor()
