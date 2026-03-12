import os
import time
import argparse
import numpy as np
import random
import json
from PIL import Image
import gc
import pandas as pd
import csv
from datetime import datetime
import subprocess
import re

import torch
import torch.nn as nn
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

from datasets import load_dataset
from tqdm import tqdm
from torch.utils.flop_counter import FlopCounterMode   


from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
)
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0, XFormersAttnProcessor


from diffusion.datasets import get_target_dataset


# Global caches/counters

UNET_TOTAL_TIME = 0.0   # cumulative time
UNET_CALL_COUNT = 0

def measure_unet_flops_once(unet, batch_size=1, img_size=512, sequence_length=77, args=None):
    """
    Measure FLOPs for one UNet forward pass using the same method as eval_prob_adaptive.py.
    
    Args:
        unet: The UNet model
        batch_size: Batch size for measurement
        img_size: Image size (height/width)
        sequence_length: Text sequence length
        args: Arguments for special case handling
    
    Returns:
        float: Number of FLOPs for this forward pass
    """
    print(f"\nMeasuring FLOPs for one UNet forward pass (logical batch = {batch_size})…")
    latent_size = img_size // 8  # SD uses 8x downsampling
    
    try:
        # ----------------------------- dummy inputs -----------------------------
        b_dummy = batch_size                       
        dt = torch.float16 if args and args.dtype == "float16" else torch.float32
        device = next(unet.parameters()).device
        
        dummy_latents = torch.randn(
            b_dummy, 4, latent_size, latent_size,  # 4 channels for SD latent space
            device=device, dtype=dt
        )
        dummy_timestep = torch.tensor(
            [500],  # Mid-point timestep
            device=device
        )
        # Get proper context dimension from UNet config
        ca_dim = (unet.config.cross_attention_dim
                if getattr(unet.config, "cross_attention_dim", None) is not None
                else 768)  # Default SD text encoder dim
        dummy_context = torch.randn(
            b_dummy, sequence_length, ca_dim,
            device=device, dtype=dt
        )

        # For masked_attention or special methods: create dummy mask for FLOP measurement
        needs_dummy_mask = False
        if args and hasattr(args, 'if_attention_proc') and args.if_attention_proc == 1:
            merge_method = getattr(args, 'merge_method', '')
            if merge_method in ["masked_attention", "blockwise_masked_attention", 
                              "snr_masked_attention", "snr_blockwise_masked_attention", 
                              "noise_magnitude_masked_attention", "noise_magnitude_blockwise_masked_attention"]:
                needs_dummy_mask = True
        elif args and getattr(args, 'if_scoring_merge', 0) == 1:
            scoring_method = getattr(args, 'scoring_method', '')
            scoring_signal_method = getattr(args, 'scoring_signal_method', '')
            if scoring_method == 'signal_processing' and scoring_signal_method in ["snr", "noise_magnitude"]:
                needs_dummy_mask = True
                
        if needs_dummy_mask:
            if hasattr(unet, '_tome_info') and unet._tome_info is not None:
                try:
                    # For methods requiring special setup
                    if args.if_attention_proc == 1 and args.merge_method in ["masked_attention", "blockwise_masked_attention"]:
                        from ImprovedTokenMerge.merge import precompute_fixed_mask_from_original
                        # Create dummy latent in attention format for mask generation
                        dummy_latent_reshaped = torch.randn(1, latent_size * latent_size, 4, device=device, dtype=dt)
                        precompute_fixed_mask_from_original(dummy_latent_reshaped, unet._tome_info, target_level="level_1")
                        print("Created temporary mask for FLOP measurement")
                    
                    # For SNR and noise magnitude methods
                    elif (('snr_' in getattr(args, 'merge_method', '') or 'noise_magnitude_' in getattr(args, 'merge_method', '')) or 
                          (getattr(args, 'if_scoring_merge', 0) == 1 and getattr(args, 'scoring_signal_method', '') in ["snr", "noise_magnitude"])):
                        # Store clean image for per-timestep mask computation
                        dummy_latent_reshaped = torch.randn(1, latent_size * latent_size, 4, device=device, dtype=dt)
                        unet._tome_info['current_clean_image'] = dummy_latent_reshaped
                        print("Created temporary clean image for SNR/noise FLOP measurement")
                        
                except ImportError:
                    print("Warning: Could not import mask functions for FLOP measurement")
                except Exception as e:
                    print(f"Warning: Could not create dummy mask for FLOP measurement: {e}")

        # ------------------------------- counter -------------------------------
        with FlopCounterMode(unet, display=True) as fc:   # counter OUTER-most
            with torch.no_grad():                        #    (no grads, low‑RAM)
                _ = unet(dummy_latents,
                        dummy_timestep,
                        encoder_hidden_states=dummy_context).sample

        # ----------------------------- results -----------------------------
        flops_batch1 = fc.get_total_flops()              # for batch = 1
        unet_flops_per_pass = flops_batch1 
        print(f"Measured FLOPs/pass (batch={batch_size}): "
            f"{format_flops(unet_flops_per_pass)}")
        return unet_flops_per_pass

    except Exception as e:
        print("ERROR while counting FLOPs:", e)
        import traceback
        traceback.print_exc()
        return -1

    finally:
        # Reset masks after FLOP measurement for clean evaluation
        if needs_dummy_mask and hasattr(unet, '_tome_info') and unet._tome_info is not None:
            try:
                if args.if_attention_proc == 1 and args.merge_method in ["masked_attention", "blockwise_masked_attention"]:
                    from ImprovedTokenMerge.merge import reset_fixed_masks
                    reset_fixed_masks(unet._tome_info)
                    print("Reset masks after FLOP measurement")
                elif 'current_clean_image' in unet._tome_info:
                    del unet._tome_info['current_clean_image']
                    print("Cleaned up temporary clean image after FLOP measurement")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up after FLOP measurement: {cleanup_error}")
                
        # Always clean up dummy tensors
        try:
            del dummy_latents, dummy_timestep, dummy_context
        except:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def patch_unet_forward(unet):
    """
    Simplified patching that only tracks time and call count (same as DiT script).
    No FLOP counting during generation to avoid hangs.
    """
    # Store original forward in a hidden attribute
    if not hasattr(unet, "__old_forward"):
        unet.__old_forward = unet.forward

    def new_forward(lat, t, *, encoder_hidden_states=None, **kwargs):
        global UNET_TOTAL_TIME, UNET_CALL_COUNT

        # Time the forward pass (same as DiT script)
        torch.cuda.synchronize()
        iter_start_time = time.time()

        # Call the original forward
        out = unet.__old_forward(lat, t, encoder_hidden_states=encoder_hidden_states, **kwargs)

        torch.cuda.synchronize()
        iter_end_time = time.time()

        # Update global counters (same as DiT script)
        UNET_TOTAL_TIME += (iter_end_time - iter_start_time)
        UNET_CALL_COUNT += 1

        return out

    # Patch the forward method
    unet.forward = new_forward


def log_to_csv(args, exp_name, img_dir, total_images, total_time, fid_score, 
               unet_flops_per_pass, UNET_TOTAL_TIME, UNET_CALL_COUNT):
    """
    Updated log_to_csv function to match DiT script format
    """
    summary_csv_path = args.summary_csv
    print(f"Attempting to append results to summary CSV: {summary_csv_path}")
    
    try:
        headers = [
            'Timestamp', 'Experiment Name', 'Dataset',
            'Args JSON',
            'FID Score', 'Total Images', 'Generation Time (s)', 
            'UNet Forward Time (s)', 'UNet Calls', 'UNet FLOPs/pass',
            'Steps', 'Batch Size', 'Image Size', 'Guidance Scale',
            'Acceleration Method', 'Output Directory'
        ]

        # Determine acceleration method for easier filtering
        acceleration_method = "None"
        if args.if_token_merging == 1:
            method_suffix = f"_{getattr(args, 'token_merging_method', 'mean')}" if getattr(args, 'token_merging_method', 'mean') != 'mean' else ""
            prop_attn_suffix = "_ProportionalAttn" if getattr(args, 'if_proportional_attention', 0) else ""
            acceleration_method = f"TokenMerge_{args.token_merging_ratio}{method_suffix}{prop_attn_suffix}"
        elif args.if_agentsd == 1:
            acceleration_method = f"AgentSD_{args.token_merging_ratio}_agent{args.agent_ratio}"
        elif args.if_attention_proc == 1:
            acceleration_method = f"AttnProc_{args.merge_method}"
        elif args.if_scoring_merge == 1:
            method_suffix = f"_{getattr(args, 'token_merging_method', 'mean')}" if getattr(args, 'token_merging_method', 'mean') != 'mean' else ""
            spatial_suffix = "_SpatialUniform" if getattr(args, 'scoring_preserve_spatial_uniformity', 0) else ""
            prop_attn_suffix = "_ProportionalAttn" if getattr(args, 'if_proportional_attention', 0) else ""
            acceleration_method = f"ScoringMerge_{args.scoring_method}_{args.token_merging_ratio}{method_suffix}{spatial_suffix}{prop_attn_suffix}"
        elif args.if_sito == 1:
            acceleration_method = f"SiTo_{args.sito_prune_ratio}"
        if args.if_deepcache == 1:
            acceleration_method += f"+DeepCache_{args.cache_interval}"


        # Convert args to JSON string for detailed logging
        args_dict = vars(args)
        args_json_string = json.dumps(args_dict, sort_keys=True)

        data_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            exp_name,
            args.dataset,
            args_json_string,
            f"{fid_score:.4f}" if fid_score is not None else 'N/A',
            total_images,
            f"{total_time:.2f}",
            f"{UNET_TOTAL_TIME:.4f}",
            UNET_CALL_COUNT,
            format_flops(unet_flops_per_pass) if unet_flops_per_pass >= 0 else 'Failed',
            args.steps,
            args.batch_size,
            args.img_size,
            args.guidance_scale,
            acceleration_method,
            img_dir
        ]
        
        # Only log if we have meaningful results
        if total_images > 0:
            file_exists = os.path.exists(summary_csv_path) 
            with open(summary_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists or os.path.getsize(summary_csv_path) == 0:
                    writer.writerow(headers)
                    print(f"Created header row in {summary_csv_path}")
                writer.writerow(data_row)
            print(f"Successfully appended summary to {summary_csv_path}")
        else:
            print("Skipping CSV logging - no images generated")

    except Exception as e:
        print(f"ERROR: Failed to write to summary CSV file {summary_csv_path}: {e}")
        import traceback
        traceback.print_exc()


def print_memory_usage(stage=""):
    """Print current GPU memory usage (same as DiT script)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")


def cleanup_memory():
    """Comprehensive memory cleanup (same as DiT script)"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def calculate_fid(gt_path, gen_path, batch_size=50):
    """
    Calculates the FID score by executing the pytorch-fid command-line tool.

    This function captures the command's output, parses the FID score,
    and returns it as a float. It handles common errors like missing paths
    or the tool not being installed.
    """
    if not gt_path or not os.path.exists(gt_path):
        print(f"Warning: Ground truth path for FID calculation not found or not provided: {gt_path}")
        return None
    if not os.path.exists(gen_path) or not os.listdir(gen_path):
        print(f"Warning: Generated images path for FID calculation is empty: {gen_path}")
        return None

    print(f"\n--- Calculating FID Score ---")
    print(f"Comparing ground truth ('{gt_path}') with generated images ('{gen_path}')...")
    
    # Construct the command based on the user's provided shell script
    command = [
        "python", "-m", "pytorch_fid",
        gt_path,
        gen_path,
        "--batch-size", str(batch_size)
    ]
    
    try:
        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse the FID score from the output string "FID:  xx.xxxx"
        fid_match = re.search(r"FID:\s*([0-9]+\.?[0-9]*)", output)
        
        if fid_match:
            fid_value = float(fid_match.group(1))
            print(f"Successfully calculated FID Score: {fid_value:.4f}")
            return fid_value
        else:
            print("Warning: Could not parse FID score from the tool's output.")
            print("Output was:", output)
            return None
            
    except FileNotFoundError:
        print("Error: `python` command not found or `pytorch-fid` is not installed.")
        print("Please ensure `pytorch-fid` is installed in your environment (`pip install pytorch-fid`).")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing the pytorch-fid tool: {e}")
        print("Stderr output:", e.stderr)
        return None



# --- Helper function to format FLOPs (from the first code) ---
def format_flops(flops_value):
    """Helper to format FLOPs value into a readable string."""
    if not isinstance(flops_value, (int, float)) or flops_value < 0:
        return "N/A"
    if flops_value // 10**9 > 0:
        return f"{flops_value / 10.**9:.3f} GFLOPs"
    elif flops_value // 10**6 > 0:
        return f"{flops_value / 10.**6:.3f} MFLOPs"
    elif flops_value // 10**3 > 0:
        return f"{flops_value / 10.**3:.3f} KFLOPs"
    else:
        return f"{flops_value} FLOPs"


def debug_attention_processors(unet, prefix=""):
    """
    Debug utility to check what attention processors are currently being used.
    
    Args:
        unet: The UNet model to inspect
        prefix: Optional prefix for log messages
        
    Returns:
        dict: Dictionary mapping module names to processor types
    """
    processors = {}
    processor_counts = {}
    
    for name, module in unet.named_modules():
        if hasattr(module, 'processor'):
            processor_type = type(module.processor).__name__
            processors[name] = processor_type
            processor_counts[processor_type] = processor_counts.get(processor_type, 0) + 1
    
    if prefix:
        print(f"[{prefix}] Attention Processor Summary:")
    else:
        print("Attention Processor Summary:")
    
    for proc_type, count in processor_counts.items():
        print(f"  {proc_type}: {count} modules")
    
    # Check for mixed processor types (which could indicate inconsistency)
    if len(processor_counts) > 1:
        print(f"  WARNING: Multiple processor types detected! This may affect FLOP comparisons.")
        print(f"  Mixed processors: {list(processor_counts.keys())}")
    else:
        print(f"  All modules use consistent processor: {list(processor_counts.keys())[0]}")
    
    return processors




MODEL_IDS = {
    '1-1': "CompVis/stable-diffusion-v1-1",
    '1-2': "CompVis/stable-diffusion-v1-2",
    '1-3': "CompVis/stable-diffusion-v1-3",
    '1-4': "CompVis/stable-diffusion-v1-4",
    '1-5': "runwayml/stable-diffusion-v1-5",
    '2-0': "stabilityai/stable-diffusion-2-base",
    '2-1': "stabilityai/stable-diffusion-2-1-base"
}


INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def seed_everything(seed: int):
    """Seed everything for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For deterministic behavior in cuDNN-based operations
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_scheduler_config(args):
    """Return the config dictionary for EulerDiscreteScheduler based on the SD version."""
    if args.version in {'1-1', '1-2', '1-3', '1-4', '1-5'}:
        config = {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.14.0",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "interpolation_type": "linear",
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None
        }
    elif args.version in {'2-0', '2-1'}:
        config = {
            "_class_name": "EulerDiscreteScheduler",
            "_diffusers_version": "0.10.2",
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "clip_sample": False,
            "num_train_timesteps": 1000,
            "prediction_type": "epsilon",
            "set_alpha_to_one": False,
            "skip_prk_steps": True,
            "steps_offset": 1,
            "trained_betas": None
        }
    else:
        raise NotImplementedError(f"Scheduler config not implemented for {args.version}")
    return config





def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


def get_sd_pipeline(args, pipeline_type='txt2img'):
    """
    Create and configure a Stable Diffusion pipeline.
    
    Args:
        args: The command line arguments
        pipeline_type: The type of pipeline to create ('txt2img' or 'img2img')
    
    Returns:
        Configured StableDiffusionPipeline or StableDiffusionImg2ImgPipeline
    """
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError

    assert args.version in MODEL_IDS.keys()
    model_id = MODEL_IDS[args.version]
    
    # Use proper scheduler config based on version
    scheduler_config = get_scheduler_config(args)
    scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
    
    # Choose the pipeline class based on the pipeline_type
    if pipeline_type == 'img2img':
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, 
            scheduler=scheduler, 
            torch_dtype=dtype
        )
        print(f"Created StableDiffusionImg2ImgPipeline from {model_id}")
    else:  # Default to txt2img
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            scheduler=scheduler, 
            torch_dtype=dtype
        )
        print(f"Created StableDiffusionPipeline from {model_id}")
    
    pipe.disable_xformers_memory_efficient_attention()  # DISABLED for accurate FLOP counting and fair sparse attention comparison



    # Token merging if enabled
    if args.if_token_merging == 1:
        import sys
        import os
        tomesd_path = os.path.join(os.path.dirname(__file__), 'tomesd')
        if tomesd_path not in sys.path:
            sys.path.insert(0, tomesd_path)
        from tomesd.patch import apply_patch

        # Initialize ToMe generator with deterministic seed
        # This ensures ToMe uses the same random pattern across all methods
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        tome_generator = torch.Generator(device=generator_device).manual_seed(args.seed)

        apply_patch(
            pipe.unet,
            ratio=args.token_merging_ratio,
            sx=args.token_merging_sx,
            sy=args.token_merging_sy,
            max_downsample=args.token_merging_max_downsample,
            use_rand=bool(args.token_merging_use_rand),
            single_downsample_level_merge=bool(args.token_merging_single_downsample_level_merge),
            cache_indices_per_image=bool(getattr(args, 'token_merging_cache_indices_per_image', 0)),
            method=getattr(args, 'token_merging_method', 'mean'),
            # Additional parameters
            merge_attn=bool(getattr(args, 'merge_attn', 1)),
            merge_crossattn=bool(getattr(args, 'merge_crossattn', 0)),
            merge_mlp=bool(getattr(args, 'merge_mlp', 0)),
            if_proportional_attention=bool(getattr(args, 'if_proportional_attention', 0)),
            # NEW: Resolution caching parameters
            cache_resolution_merge=bool(getattr(args, 'cache_resolution_merge', 0)),
            cache_resolution_mode=getattr(args, 'cache_resolution_mode', 'global'),
            debug_cache=bool(getattr(args, 'debug_cache', 0)),
            # Locality-based sub-block bipartite matching parameters
            locality_block_factor_h=getattr(args, 'locality_block_factor_h', 1),
            locality_block_factor_w=getattr(args, 'locality_block_factor_w', 1)
        )

        # Set the deterministic generator in tome_info after patching
        # This ensures the same random pattern is used across all methods
        pipe.unet._tome_info["args"]["generator"] = tome_generator

        print(
            f"Token merging applied with the following settings:\n"
            f"  Method:           {getattr(args, 'token_merging_method', 'mean')}\n"
            f"  Ratio:            {args.token_merging_ratio}\n"
            f"  Stride X (sx):    {args.token_merging_sx}\n"
            f"  Stride Y (sy):    {args.token_merging_sy}\n"
            f"  Max Downsample:   {args.token_merging_max_downsample}\n"
            f"  single_downsample_level_merge:   {args.token_merging_single_downsample_level_merge}\n"
            f"  Cache Indices Per Image: {bool(getattr(args, 'token_merging_cache_indices_per_image', 0))}\n"
            f"  Use Random:       {bool(args.token_merging_use_rand)}\n"
            f"  Merge Attn:       {bool(getattr(args, 'merge_attn', 1))}\n"
            f"  Merge Cross-Attn: {bool(getattr(args, 'merge_crossattn', 0))}\n"
            f"  Merge MLP:        {bool(getattr(args, 'merge_mlp', 0))}\n"
            f"  Proportional Attention: {bool(getattr(args, 'if_proportional_attention', 0))}"
        )
    
    elif args.if_agentsd == 1:
        import agentsd
        agentsd.remove_patch(pipe.unet)
        agentsd.apply_patch(pipe.unet,             
                        ratio=args.token_merging_ratio,
                        sx=args.token_merging_sx,
                        sy=args.token_merging_sy,
                        max_downsample=args.token_merging_max_downsample,
                        use_rand=bool(args.token_merging_use_rand),
                        single_downsample_level_merge=bool(args.token_merging_single_downsample_level_merge), 
                        agent_ratio=args.agent_ratio, 
                        attn_precision="fp32")
        print(
            f"Agentsd applied with the following settings:\n"
            f"  Ratio:            {args.token_merging_ratio}\n"
            f"  Agent Ratio:      {args.agent_ratio}\n"
            f"  Stride X (sx):    {args.token_merging_sx}\n"
            f"  Stride Y (sy):    {args.token_merging_sy}\n"
            f"  Max Downsample:   {args.token_merging_max_downsample}\n"
            f"  single_downsample_level_merge:   {args.token_merging_single_downsample_level_merge}\n"
            f"  Use Random:       {bool(args.token_merging_use_rand)}"
        )
    
    elif args.if_attention_proc == 1:
        import ImprovedTokenMerge.utils 
        ImprovedTokenMerge.utils.remove_patch(pipe)
        token_merge_args = {
            # Added from original token merging
            "ratio": args.token_merging_ratio,
            "sx": args.token_merging_sx,  
            "sy": args.token_merging_sy,  
            "use_rand": bool(args.token_merging_use_rand), 
            
            "merge_tokens": args.merge_tokens,
            "merge_method": args.merge_method,
            "downsample_method": args.downsample_method,
            "downsample_factor": args.downsample_factor,
            "timestep_threshold_switch": args.timestep_threshold_switch,
            "timestep_threshold_stop": args.timestep_threshold_stop,
            "secondary_merge_method": args.secondary_merge_method,
            "downsample_factor_level_2": args.downsample_factor_level_2,
            "ratio_level_2": args.ratio_level_2,
            "frequency_selection_mode": args.frequency_selection_mode,
            "frequency_selection_method": args.frequency_selection_method,
            "frequency_ranking_method": args.frequency_ranking_method,
            "selection_source": args.selection_source,
            "frequency_grid_alpha": args.frequency_grid_alpha,
            "qkv_downsample_method": args.qkv_downsample_method,
            "out_upsample_method": args.out_upsample_method,
            "downsample_factor_h": args.downsample_factor_h,
            "downsample_factor_w": args.downsample_factor_w,
            
            # Linear blend factors for linear_blend downsample method
            "linear_blend_factor": getattr(args, 'linear_blend_factor', 0.5),
            "qkv_linear_blend_factor": getattr(args, 'qkv_linear_blend_factor', 0.5),
            "out_linear_blend_factor": getattr(args, 'out_linear_blend_factor', 0.5),
            
            # Timestep-based interpolation for linear blend
            "linear_blend_timestep_interpolation": bool(getattr(args, 'linear_blend_timestep_interpolation', 0)),
            "linear_blend_start_ratio": getattr(args, 'linear_blend_start_ratio', 0.1),
            "linear_blend_end_ratio": getattr(args, 'linear_blend_end_ratio', 0.9),
            "qkv_linear_blend_timestep_interpolation": bool(getattr(args, 'qkv_linear_blend_timestep_interpolation', 0)),
            "qkv_linear_blend_start_ratio": getattr(args, 'qkv_linear_blend_start_ratio', 0.1),
            "qkv_linear_blend_end_ratio": getattr(args, 'qkv_linear_blend_end_ratio', 0.9),
            "out_linear_blend_timestep_interpolation": bool(getattr(args, 'out_linear_blend_timestep_interpolation', 0)),
            "out_linear_blend_start_ratio": getattr(args, 'out_linear_blend_start_ratio', 0.1),
            "out_linear_blend_end_ratio": getattr(args, 'out_linear_blend_end_ratio', 0.9),
            
            # Linear blend method selection parameters
            "linear_blend_method_1": getattr(args, 'linear_blend_method_1', 'nearest-exact'),
            "linear_blend_method_2": getattr(args, 'linear_blend_method_2', 'avg_pool'),
            "qkv_linear_blend_method_1": getattr(args, 'qkv_linear_blend_method_1', 'nearest-exact'),
            "qkv_linear_blend_method_2": getattr(args, 'qkv_linear_blend_method_2', 'avg_pool'),
            "out_linear_blend_method_1": getattr(args, 'out_linear_blend_method_1', 'nearest-exact'),
            "out_linear_blend_method_2": getattr(args, 'out_linear_blend_method_2', 'avg_pool'),
            "blockwise_blend_factor": getattr(args, 'blockwise_blend_factor', 0.5),

        }








        ImprovedTokenMerge.utils.patch_attention_proc(pipe.unet, token_merge_args=token_merge_args)

        # Set deterministic generator for attention processor (same as ToMe and scoring merge)
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        attn_proc_generator = torch.Generator(device=generator_device).manual_seed(args.seed)
        pipe.unet._tome_info["args"]["generator"] = attn_proc_generator

        print(
            f"Attention processor patch applied with settings:\n"
            f"  Ratio:               {args.token_merging_ratio}\n"

            f"  Stride X (sx):    {args.token_merging_sx}\n"
            f"  Stride Y (sy):    {args.token_merging_sy}\n"
            f"  Use Random:       {bool(args.token_merging_use_rand)}\n"

            f"  Merge Tokens:        {args.merge_tokens}\n"
            f"  Merge Method:        {args.merge_method}\n"
            f"  Downsample Method:   {args.downsample_method}\n"
            f"  Downsample Factor:   {args.downsample_factor}\n"
            f"  Linear Blend Factor: {getattr(args, 'linear_blend_factor', 0.5)} (for linear_blend method)\n"
            f"  Linear Blend Methods: {getattr(args, 'linear_blend_method_1', 'nearest-exact')} + {getattr(args, 'linear_blend_method_2', 'avg_pool')}\n"
            f"  QKV Linear Blend:    {getattr(args, 'qkv_linear_blend_factor', 0.5)}\n"
            f"  QKV Blend Methods:   {getattr(args, 'qkv_linear_blend_method_1', 'nearest-exact')} + {getattr(args, 'qkv_linear_blend_method_2', 'avg_pool')}\n"
            f"  Out Linear Blend:    {getattr(args, 'out_linear_blend_factor', 0.5)}\n"
            f"  Out Blend Methods:   {getattr(args, 'out_linear_blend_method_1', 'nearest-exact')} + {getattr(args, 'out_linear_blend_method_2', 'avg_pool')}\n"
            f"  Blockwise Blend:     {getattr(args, 'blockwise_blend_factor', 0.5)} (for frequency_blockwise method)\n"
            f"  Switch Threshold:    {args.timestep_threshold_switch}\n"
            f"  Stop Threshold:      {args.timestep_threshold_stop}\n"
            f"  Secondary Method:    {args.secondary_merge_method}\n"
            f"  L2 Downsample:       {args.downsample_factor_level_2}\n"
            f"  L2 Ratio:            {args.ratio_level_2}\n"
            f"  Frequency Selection mode: {args.frequency_selection_mode}\n"
            f"  Frequency Selection method: {args.frequency_selection_method}\n"
            f"  Frequency Ranking method: {args.frequency_ranking_method}\n"
        )
    elif args.if_sito == 1:
        import SiTo
        SiTo.remove_patch(pipe)
        SiTo.apply_patch(
            pipe,
            prune_ratio=args.sito_prune_ratio,
            max_downsample_ratio=args.sito_max_downsample_ratio,
            prune_selfattn_flag=bool(args.sito_prune_selfattn_flag),
            prune_crossattn_flag=bool(args.sito_prune_crossattn_flag),
            prune_mlp_flag=bool(args.sito_prune_mlp_flag),
            sx=args.sito_sx,
            sy=args.sito_sy,
            noise_alpha=args.sito_noise_alpha,
            sim_beta=args.sito_sim_beta
        )
        print(
            f"SiTo applied with the following settings:\n"
            f"  Prune Ratio:          {args.sito_prune_ratio}\n"
            f"  Max Downsample Ratio: {args.sito_max_downsample_ratio}\n"
            f"  Prune Self-Attn:      {bool(args.sito_prune_selfattn_flag)}\n"
            f"  Prune Cross-Attn:     {bool(args.sito_prune_crossattn_flag)}\n"
            f"  Prune MLP:            {bool(args.sito_prune_mlp_flag)}\n"
            f"  Stride X (sx):        {args.sito_sx}\n"
            f"  Stride Y (sy):        {args.sito_sy}\n"
            f"  Noise Alpha:          {args.sito_noise_alpha}\n"
            f"  Sim Beta:             {args.sito_sim_beta}"
        )
    elif args.if_scoring_merge == 1:
        import sys
        import os
        tomesd_path = os.path.join(os.path.dirname(__file__), 'tomesd')
        if tomesd_path not in sys.path:
            sys.path.insert(0, tomesd_path)
        from tomesd.patch import remove_patch
        remove_patch(pipe.unet)

        # Initialize ToMe generator with deterministic seed
        # This ensures ToMe uses the same random pattern across all methods
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        tome_generator = torch.Generator(device=generator_device).manual_seed(args.seed)

        # Create the appropriate scorer based on method
        # tomesd path already added above
        from tomesd.scoring import (
            FrequencyScorer, SpatialFilterScorer, StatisticalScorer,
            SignalProcessingScorer, SpatialDistributionScorer, SimilarityScorer
        )
        
        if args.scoring_method == 'frequency':
            scorer = FrequencyScorer(
                method=args.scoring_freq_method,
                ranking=args.scoring_freq_ranking
            )
        elif args.scoring_method == 'spatial_filter':
            scorer = SpatialFilterScorer(
                method=args.scoring_spatial_method,
                norm=args.scoring_spatial_norm
            )
        elif args.scoring_method == 'statistical':
            scorer = StatisticalScorer(
                method=args.scoring_stat_method
            )
        elif args.scoring_method == 'signal_processing':
            scorer = SignalProcessingScorer(
                method=args.scoring_signal_method
            )
        elif args.scoring_method == 'spatial_distribution':
            scorer = SpatialDistributionScorer(
                alpha=args.scoring_spatial_alpha
            )
        elif args.scoring_method == 'similarity':
            scorer = SimilarityScorer(
                method=args.scoring_similarity_method
            )
        else:
            raise ValueError(f"Unknown scoring method: {args.scoring_method}")
        
        # Prepare scorer kwargs for timestep scheduling if needed
        scorer_kwargs = {}
        if hasattr(args, 'timestep_normalized'):
            scorer_kwargs['timestep_normalized'] = args.timestep_normalized
        
        # Create ABP scorer if using ABP matching algorithm
        abp_scorer = None
        if getattr(args, 'scoring_matching_algorithm', 'bipartite') == 'abp':
            abp_scorer_method = getattr(args, 'abp_scorer_method', 'spatial_filter')
            if abp_scorer_method == 'frequency':
                abp_scorer = FrequencyScorer(
                    method=getattr(args, 'scoring_freq_method', '1d_dft'),
                    ranking=getattr(args, 'scoring_freq_ranking', 'amplitude')
                )
            elif abp_scorer_method == 'spatial_filter':
                abp_scorer = SpatialFilterScorer(
                    method=getattr(args, 'scoring_spatial_method', '2d_conv'),
                    norm=getattr(args, 'scoring_spatial_norm', 'l1')
                )
            elif abp_scorer_method == 'statistical':
                abp_scorer = StatisticalScorer(
                    method=getattr(args, 'scoring_stat_method', 'l2norm')
                )
            elif abp_scorer_method == 'signal_processing':
                abp_scorer = SignalProcessingScorer(
                    method=getattr(args, 'scoring_signal_method', 'snr')
                )
            elif abp_scorer_method == 'spatial_distribution':
                abp_scorer = SpatialDistributionScorer(
                    alpha=getattr(args, 'scoring_spatial_alpha', 2.0)
                )
            elif abp_scorer_method == 'similarity':
                abp_scorer = SimilarityScorer(
                    method=getattr(args, 'scoring_similarity_method', 'local_neighbors_inverted')
                )
        
        # Apply scoring-based token merging
        # tomesd path already added above
        from tomesd.patch import apply_patch
        apply_patch(
            pipe.unet,
            ratio=args.token_merging_ratio,
            sx=args.token_merging_sx,
            sy=args.token_merging_sy,
            max_downsample=args.token_merging_max_downsample,
            use_rand=bool(args.token_merging_use_rand),
            single_downsample_level_merge=bool(args.token_merging_single_downsample_level_merge),
            cache_indices_per_image=bool(getattr(args, 'token_merging_cache_indices_per_image', 0)),
            merge_attn=bool(getattr(args, 'merge_attn', 1)),
            merge_crossattn=bool(getattr(args, 'merge_crossattn', 0)),
            merge_mlp=bool(getattr(args, 'merge_mlp', 0)),
            method=getattr(args, 'token_merging_method', 'mean'),
            if_proportional_attention=bool(getattr(args, 'if_proportional_attention', 0)),
            # New scoring parameters
            use_scoring=True,
            scorer=scorer,
            preserve_ratio=args.scoring_preserve_ratio,
            score_mode=args.scoring_mode,
            preserve_spatial_uniformity=bool(getattr(args, 'scoring_preserve_spatial_uniformity', 0)),
            if_low_frequency_dst_tokens=bool(getattr(args, 'if_low_frequency_dst_tokens', 0)),
            scorer_kwargs=scorer_kwargs,
            # NEW: Resolution caching parameters
            cache_resolution_merge=bool(getattr(args, 'cache_resolution_merge', 0)),
            cache_resolution_mode=getattr(args, 'cache_resolution_mode', 'global'),
            debug_cache=bool(getattr(args, 'debug_cache', 0)),
            # ABP vs Bipartite matching algorithm for scoring merge
            merge_method=getattr(args, 'scoring_matching_algorithm', 'bipartite'),
            # ABP configuration parameters
            abp_scorer=abp_scorer,
            abp_tile_aggregation=getattr(args, 'abp_tile_aggregation', 'max'),
            # Locality-based sub-block bipartite matching parameters
            locality_block_factor_h=getattr(args, 'locality_block_factor_h', 1),
            locality_block_factor_w=getattr(args, 'locality_block_factor_w', 1)
        )

        # Set the deterministic generator in tome_info after patching
        # This ensures the same random pattern is used as standard ToMe
        pipe.unet._tome_info["args"]["generator"] = tome_generator
        
        print(
            f"Scoring-based token merging applied with the following settings:\n"
            f"  Merging Method:   {getattr(args, 'token_merging_method', 'mean')}\n"
            f"  Scoring Method:   {args.scoring_method}\n"
            f"  Scorer:           {scorer.get_name()}\n"
            f"  Ratio:            {args.token_merging_ratio}\n"
            f"  Preserve Ratio:   {args.scoring_preserve_ratio}\n"
            f"  Score Mode:       {args.scoring_mode}\n"
            f"  Preserve Spatial Uniformity: {bool(getattr(args, 'scoring_preserve_spatial_uniformity', 0))}\n"
            f"  Stride X (sx):    {args.token_merging_sx}\n"
            f"  Stride Y (sy):    {args.token_merging_sy}\n"
            f"  Max Downsample:   {args.token_merging_max_downsample}\n"
            f"  Single Downsample Level: {args.token_merging_single_downsample_level_merge}\n"
            f"  Cache Indices Per Image: {bool(getattr(args, 'token_merging_cache_indices_per_image', 0))}\n"
            f"  Use Random:       {bool(args.token_merging_use_rand)}\n"
            f"  Merge Attn:       {bool(getattr(args, 'merge_attn', 1))}\n"
            f"  Merge Cross-Attn: {bool(getattr(args, 'merge_crossattn', 0))}\n"
            f"  Merge MLP:        {bool(getattr(args, 'merge_mlp', 0))}\n"
            f"  Proportional Attention: {bool(getattr(args, 'if_proportional_attention', 0))}"
            + (f"\n  Matching Algorithm: {getattr(args, 'scoring_matching_algorithm', 'bipartite')}" if getattr(args, 'scoring_matching_algorithm', 'bipartite') != 'bipartite' else "")
            + (f"\n  ABP Scorer:       {abp_scorer.get_name()}" if abp_scorer else "")
            + (f"\n  ABP Tile Agg:     {getattr(args, 'abp_tile_aggregation', 'max')}" if getattr(args, 'scoring_matching_algorithm', 'bipartite') == 'abp' else "")
        )
    else:
        print("No token reduction applied.")
    
    patch_unet_forward(pipe.unet)
    return pipe

def save_images_as_jpg(images, prompts, output_dir, quality=100, start_idx=0):
    """
    Save batch of images directly to JPG files
    
    Args:
        images: numpy array of images (batch, H, W, 3) with values 0-1
        prompts: list of prompts corresponding to images
        output_dir: directory to save images
        quality: JPEG quality (0-100)
        start_idx: starting index for image numbering
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prompts to a JSON file
    prompts_file = os.path.join(output_dir, "prompts.json")
    with open(prompts_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        img_idx = start_idx + i
        
        # Convert from 0-1 float to uint8
        img_int = (img * 255).astype("uint8")
        pil_img = Image.fromarray(img_int)
        
        # Create a filename with sanitized prompt
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in " _-").strip()
        safe_prompt = safe_prompt[:50]  # Limit length
        safe_prompt = safe_prompt.replace(" ", "_")
        filename = f"image_{img_idx:05d}_{safe_prompt}.jpg"
        
        # Save as JPG
        pil_img.save(os.path.join(output_dir, filename), "JPEG", quality=quality)


def main(args):
    # Declare globals to be modified (same as DiT script)
    global UNET_TOTAL_TIME, UNET_CALL_COUNT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_memory_usage("Start")
    seed_everything(args.seed)

    # Load dataset with appropriate transformation
    if hasattr(args, 'interpolation') and args.interpolation in INTERPOLATIONS:
        interpolation = INTERPOLATIONS[args.interpolation]
    else:
        interpolation = InterpolationMode.BICUBIC
        
    transform = get_transform(interpolation, args.img_size)
    dataset = get_target_dataset(args.dataset, train=False, transform=transform)

    # Extract prompts (your existing logic - keep as is)
    if args.dataset == 'coco2017':
        pipeline_type = 'txt2img'
        prompts = []
        for item in dataset:
            if isinstance(item, tuple) and len(item) > 1:
                prompts.append(item[1])
            elif isinstance(item, dict) and 'captions' in item:
                prompts.append(item['captions'][0] if isinstance(item['captions'], list) else item['captions'])
            else:
                prompts.append("")
        print("COCO prompts examples:", prompts[:5])

    elif args.dataset in ["pets", "flowers", "imagenet100", "imagenet"]:
        pipeline_type = 'txt2img'
        print(f"Using text-to-image pipeline for {args.dataset} dataset with generated prompts")
        
        import pandas as pd
        prompts_path = f"prompts/{args.dataset}_prompts.csv"
        prompts_df = pd.read_csv(prompts_path)
        
        class_idx_to_prompt = dict(zip(prompts_df['classidx'], prompts_df['prompt']))
        
        prompts = []
        dataset_size = len(dataset)
        
        if hasattr(dataset, 'samples'):
            labels = [sample[1] for sample in dataset.samples]
        elif hasattr(dataset, 'targets'):
            labels = dataset.targets
        else:
            print("Warning: Using slow method to extract labels...")
            labels = []
            for i in range(dataset_size):
                item = dataset[i]
                if isinstance(item, tuple) and len(item) > 1:
                    labels.append(item[1])
                else:
                    labels.append(0)
        
        # Fix for ImageNet100: map sequential dataset indices to original ImageNet indices
        if args.dataset == "imagenet100":
            from diffusion.dataset.imagenet_classnames import imagenet100_classnames
            # Get original ImageNet class indices from the prompts CSV (sorted to match dataset order)
            original_indices = sorted(prompts_df['classidx'].tolist())
            # Create mapping: sequential dataset index -> original ImageNet index
            sequential_to_original = {seq_idx: orig_idx for seq_idx, orig_idx in enumerate(original_indices)}
            print(f"Created mapping for ImageNet100: {len(sequential_to_original)} classes")
            print(f"Sample mapping: 0->{sequential_to_original[0]}, 1->{sequential_to_original[1]}, 2->{sequential_to_original[2]}")
            
            # Map dataset labels to original ImageNet indices
            mapped_labels = [sequential_to_original[label] for label in labels]
        else:
            mapped_labels = labels
        
        for label in mapped_labels:
            if label in class_idx_to_prompt:
                prompts.append(class_idx_to_prompt[label])
            else:
                print(f"Warning: No prompt found for class index {label}, using empty prompt")
                prompts.append("")
        
        print(f"Using prompts from CSV for {args.dataset} dataset")
        print(f"Mapped {len(set([p for p in prompts if 'class' not in p]))} unique class-specific prompts")
        if len(prompts) > 0:
            print("Examples:", prompts[:3])
    else:
        pipeline_type = 'img2img'
        prompts = ["" for _ in range(len(dataset))]
        print(f"Using empty prompts for {args.dataset} dataset")

    # Set up pipeline
    print("Loading SD pipeline...")
    if args.original:
        pipe = get_sd_pipeline(args, pipeline_type=pipeline_type).to(device)
    else:
        if pipeline_type == 'img2img':
            from diffusers import StableDiffusionImg2ImgPipeline
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
            ).to(device)
        else:
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
            ).to(device)

    print("SD pipeline loaded successfully.")
    print_memory_usage("After model loading")

    # FORCE ALL METHODS TO USE SAME ATTENTION PROCESSOR FOR FAIR FLOP COMPARISON
    # This ensures consistent baseline measurement across all acceleration methods.
    if args.if_attention_proc == 1:
        pass
    elif args.force_attention_processor == 'none':
        print("Skipping processor forcing (--force_attention_processor=none)")
    else:
        try:
            if args.force_attention_processor == 'AttnProcessor':
                chosen_processor = AttnProcessor()
            elif args.force_attention_processor == 'AttnProcessor2_0':
                chosen_processor = AttnProcessor2_0()
            elif args.force_attention_processor == 'XFormersAttnProcessor':
                chosen_processor = XFormersAttnProcessor()
            else:
                raise ValueError(f"Unknown processor: {args.force_attention_processor}")
            pipe.unet.set_attn_processor(chosen_processor)
            print(f"Set attention processor to: {type(chosen_processor).__name__}")
        except Exception as e:
            print(f"Failed to set attention processor: {e}")

    # Reset global counters (same as DiT script)
    UNET_TOTAL_TIME = 0.0
    UNET_CALL_COUNT = 0
    
    # Measure FLOPs for one UNet forward pass (same as DiT script)
    unet_flops_per_pass = 0.0
    print(f"\nMeasuring FLOPs for one UNet forward pass...")
    
    try:
        unet_flops_per_pass = measure_unet_flops_once(
            pipe.unet, 
            batch_size=1, 
            img_size=args.img_size, 
            sequence_length=77,  # Standard SD text sequence length
            args=args  # Pass args for special case handling
        )
        print(f"Measured FLOPs/pass (batch=1): {format_flops(unet_flops_per_pass)}")

    except Exception as e:
        print(f"ERROR while counting FLOPs: {e}")
        import traceback
        traceback.print_exc()
        unet_flops_per_pass = -1
    finally:
        cleanup_memory()

    # Create output directory and experiment name (your existing logic)
    out_dir = f"data/{args.dataset}_generated_images"
    os.makedirs(out_dir, exist_ok=True)
    
    exp_name = f"SD_v{args.version}_steps{args.steps}_guidance{args.guidance_scale}_seed{args.seed}"
    
    # Add acceleration method details to exp_name (your existing logic)
    if args.if_token_merging == 1:
        exp_name += f'_tokenmerge{args.token_merging_ratio}_userand{args.token_merging_use_rand}_maxdownsample{args.token_merging_max_downsample}_singlelevel{args.token_merging_single_downsample_level_merge}_sx{args.token_merging_sx}_sy{args.token_merging_sy}'
        # Add method if not default
        if getattr(args, 'token_merging_method', 'mean') != 'mean':
            exp_name += f'_method{args.token_merging_method}'
        # Add proportional attention if enabled
        if getattr(args, 'if_proportional_attention', 0) == 1:
            exp_name += '_propattn'
        # Add locality-based sub-block parameters if not default
        if getattr(args, 'locality_block_factor_h', 1) > 1 or getattr(args, 'locality_block_factor_w', 1) > 1:
            exp_name += f'_locality_h{getattr(args, "locality_block_factor_h", 1)}_w{getattr(args, "locality_block_factor_w", 1)}'
    elif args.if_agentsd == 1:
        exp_name += f'_agentsd{args.token_merging_ratio}_agentratio{args.agent_ratio}_userand{args.token_merging_use_rand}_maxdownsample{args.token_merging_max_downsample}_singlelevel{args.token_merging_single_downsample_level_merge}_sx{args.token_merging_sx}_sy{args.token_merging_sy}'
        # Add locality-based sub-block parameters if not default
        if getattr(args, 'locality_block_factor_h', 1) > 1 or getattr(args, 'locality_block_factor_w', 1) > 1:
            exp_name += f'_locality_h{getattr(args, "locality_block_factor_h", 1)}_w{getattr(args, "locality_block_factor_w", 1)}'
    elif args.if_attention_proc == 1:
        if args.merge_method == 'frequency_global':
            exp_name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqmethod{args.frequency_selection_method}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_level1ratio{args.token_merging_ratio}'
                    f'_level2method{args.secondary_merge_method}'
                    f'_level2ratio{args.ratio_level_2}')
            if args.frequency_selection_method=='non_uniform_grid':
                exp_name += f'_gridalpha{args.frequency_grid_alpha}'
        elif args.merge_method == 'frequency_blockwise':
            exp_name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqmethod{args.frequency_selection_method}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_level1downsample{args.downsample_factor}'
                    f'_level2method{args.secondary_merge_method}'
                    f'_level2downsample{args.downsample_factor_level_2}')
            # Always add blockwise blend parameter to name
            exp_name += f'_blockwiseblend{getattr(args, "blockwise_blend_factor", 0.5)}'
        elif args.merge_method == 'downsample_qkv_upsample_out':
            exp_name += (f'_mergemethod{args.merge_method}'
                    f'_downsamplemethod{args.downsample_method}'
                    f'_qkv_downsamplemethod{args.qkv_downsample_method}'
                    f'_out_upsamplemethod{args.out_upsample_method}'
                    f'_level1downsample{args.downsample_factor}'
                    f'_level2downsample{args.downsample_factor_level_2}')
            # Always add linear_blend parameters to name
            if getattr(args, "linear_blend_timestep_interpolation", 0):
                exp_name += f'_linearblend_timestep{getattr(args, "linear_blend_start_ratio", 0.1)}-{getattr(args, "linear_blend_end_ratio", 0.9)}'
            else:
                exp_name += f'_linearblend{getattr(args, "linear_blend_factor", 0.5)}'
            exp_name += f'_blendmethods{getattr(args, "linear_blend_method_1", "nearest-exact")}-{getattr(args, "linear_blend_method_2", "avg_pool")}'
            
            if getattr(args, "qkv_linear_blend_timestep_interpolation", 0):
                exp_name += f'_qkvblend_timestep{getattr(args, "qkv_linear_blend_start_ratio", 0.1)}-{getattr(args, "qkv_linear_blend_end_ratio", 0.9)}'
            else:
                exp_name += f'_qkvblend{getattr(args, "qkv_linear_blend_factor", 0.5)}'
            exp_name += f'_qkvblendmethods{getattr(args, "qkv_linear_blend_method_1", "nearest-exact")}-{getattr(args, "qkv_linear_blend_method_2", "avg_pool")}'
            
            if getattr(args, "out_linear_blend_timestep_interpolation", 0):
                exp_name += f'_outblend_timestep{getattr(args, "out_linear_blend_start_ratio", 0.1)}-{getattr(args, "out_linear_blend_end_ratio", 0.9)}'
            else:
                exp_name += f'_outblend{getattr(args, "out_linear_blend_factor", 0.5)}'
            exp_name += f'_outblendmethods{getattr(args, "out_linear_blend_method_1", "nearest-exact")}-{getattr(args, "out_linear_blend_method_2", "avg_pool")}'
        elif args.merge_method == 'downsample_custom_block':
            exp_name += (f'_mergemethod{args.merge_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_h{args.downsample_factor_h}'
                    f'_w{args.downsample_factor_w}'
                    f'_switch{args.timestep_threshold_switch}'
                    f'_stop{args.timestep_threshold_stop}')
        else:
            exp_name += (f'tokendownsampling{args.token_merging_ratio}_userand{args.token_merging_use_rand}__sx{args.token_merging_sx}_sy{args.token_merging_sy}'
                    f'_method{args.merge_method}'
                    f'_down{args.downsample_factor}_downmethod{args.downsample_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_switch{args.timestep_threshold_switch}'
                    f'_stop{args.timestep_threshold_stop}'
                    f'_secondleveldown{args.downsample_factor_level_2}')
            # Add linear_blend_factor to name if using linear_blend method
            if args.downsample_method == 'linear_blend':
                if getattr(args, "linear_blend_timestep_interpolation", 0):
                    exp_name += f'_linearblend_timestep{getattr(args, "linear_blend_start_ratio", 0.1)}-{getattr(args, "linear_blend_end_ratio", 0.9)}'
                else:
                    exp_name += f'_linearblend{getattr(args, "linear_blend_factor", 0.5)}'
                exp_name += f'_blendmethods{getattr(args, "linear_blend_method_1", "nearest-exact")}-{getattr(args, "linear_blend_method_2", "avg_pool")}'
    elif args.if_sito == 1:
        exp_name += (f'_sito_prune{args.sito_prune_ratio}'
                f'_maxdownsample{args.sito_max_downsample_ratio}'
                f'_selfattn{args.sito_prune_selfattn_flag}'
                f'_crossattn{args.sito_prune_crossattn_flag}'
                f'_mlp{args.sito_prune_mlp_flag}'
                f'_sx{args.sito_sx}_sy{args.sito_sy}'
                f'_noisealpha{args.sito_noise_alpha}'
                f'_simbeta{args.sito_sim_beta}')
    elif args.if_scoring_merge == 1:
        exp_name += f'_scoringmerge{args.token_merging_ratio}_method{args.scoring_method}_preserve{args.scoring_preserve_ratio}_mode{args.scoring_mode}_userand{args.token_merging_use_rand}_maxdownsample{args.token_merging_max_downsample}_sx{args.token_merging_sx}_sy{args.token_merging_sy}'
        
        # Add method-specific parameters to name
        if args.scoring_method == "frequency":
            exp_name += f'_freqmethod{args.scoring_freq_method}_freqranking{args.scoring_freq_ranking}'
        elif args.scoring_method == "spatial_filter":
            exp_name += f'_spatialmethod{args.scoring_spatial_method}_spatialnorm{args.scoring_spatial_norm}'
        elif args.scoring_method == "statistical":
            exp_name += f'_statmethod{args.scoring_stat_method}'
        elif args.scoring_method == "signal_processing":
            exp_name += f'_signalmethod{args.scoring_signal_method}'
        elif args.scoring_method == "spatial_distribution":
            exp_name += f'_spatialalpha{args.scoring_spatial_alpha}'
        elif args.scoring_method == "similarity":
            exp_name += f'_similaritymethod{args.scoring_similarity_method}'
        
        # Add merging method if not default
        if getattr(args, 'token_merging_method', 'mean') != 'mean':
            exp_name += f'_mergemethod{args.token_merging_method}'
        
        # Add spatial uniformity parameter if enabled
        if getattr(args, 'scoring_preserve_spatial_uniformity', 0) == 1:
            exp_name += '_spatialuniform'
        
        # Add score-guided destination selection parameter if enabled
        if getattr(args, 'if_low_frequency_dst_tokens', 0) == 1:
            exp_name += '_lowfreqdst'
        
        # Add proportional attention if enabled
        if getattr(args, 'if_proportional_attention', 0) == 1:
            exp_name += '_propattn'
        
        # Add locality-based sub-block parameters if not default
        if getattr(args, 'locality_block_factor_h', 1) > 1 or getattr(args, 'locality_block_factor_w', 1) > 1:
            exp_name += f'_locality_h{getattr(args, "locality_block_factor_h", 1)}_w{getattr(args, "locality_block_factor_w", 1)}'
        
        # Add resolution caching parameters if enabled
        if getattr(args, 'cache_resolution_merge', 0) == 1:
            exp_name += f'_rescache{args.cache_resolution_mode}'
        
        # Add ABP parameters if using ABP algorithm
        if getattr(args, 'scoring_matching_algorithm', 'bipartite') == 'abp':
            exp_name += f'_alg{getattr(args, "scoring_matching_algorithm", "bipartite")}_scorer{getattr(args, "abp_scorer_method", "spatial_filter")}_agg{getattr(args, "abp_tile_aggregation", "max")}'
        elif getattr(args, 'scoring_matching_algorithm', 'bipartite') != 'bipartite':
            exp_name += f'_alg{getattr(args, "scoring_matching_algorithm", "bipartite")}'

    img_dir = os.path.join(out_dir, f"{exp_name}")
    os.makedirs(img_dir, exist_ok=True)
    
    print(f"Generating and saving images to {img_dir}")

    # Generation loop (your existing logic)
    total_images = 0
    images_saved = []
    prompts_saved = []
    
    start_time = time.time()

    num_batch = len(prompts) // args.batch_size
    if len(prompts) % args.batch_size != 0:
        num_batch += 1

    for i in tqdm(range(num_batch)):
        st = args.batch_size * i
        ed = min(args.batch_size * (i + 1), len(prompts))
        batch_prompts = prompts[st:ed]

        # Add timing support similar to eval_prob_adaptive.py
        batch_start_time = time.time()

        # Create a generator with fixed seed for reproducibility
        # This ensures all methods (baseline, ToMe, scoring merge) use the same random latents
        generator = torch.Generator(device=device).manual_seed(args.seed + i)

        # Your existing generation logic here
        if pipeline_type == 'txt2img':
            pipe_output = pipe(
                batch_prompts,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale if hasattr(args, 'guidance_scale') else 7.5,
                generator=generator,
                output_type='np',
                return_dict=True
            )
        else:
            # Image-to-image generation
            batch_images = []
            for idx in range(st, ed):
                # Get the original image
                orig_item = dataset[idx]
                
                # Extract the image based on dataset structure
                if isinstance(orig_item, tuple) and len(orig_item) > 0:
                    orig_image = orig_item[0]  # Assume first element is image
                elif isinstance(orig_item, dict) and 'image' in orig_item:
                    orig_image = orig_item['image']
                else:
                    orig_image = orig_item  # Assume the item itself is the image
                
                batch_images.append(orig_image)
            
            if not batch_images:
                print("Warning: No valid images in batch, skipping")
                continue
                
            # Process batch with img2img pipeline
            strength = args.img2img_strength if hasattr(args, 'img2img_strength') else 0.75
            guidance_scale = args.guidance_scale if hasattr(args, 'guidance_scale') else 7.5

            pipe_output = pipe(
                prompt=batch_prompts,
                image=batch_images,
                strength=strength,
                num_inference_steps=args.steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type='np',
                return_dict=True
            )

        images = pipe_output.images
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        
        # Add debug timing info similar to eval_prob_adaptive.py
        if i < 3:  # Only for first few batches
            print(f"[DEBUG] Batch {i}: {len(images)} images, "
                  f"batch_time={batch_time:.2f}s, "
                  f"avg_per_image={batch_time/len(images):.2f}s")
        
        save_images_as_jpg(
            images=images, 
            prompts=batch_prompts, 
            output_dir=img_dir,
            quality=args.quality,
            start_idx=total_images
        )
        
        total_images += len(images)
        images_saved.append(torch.from_numpy((images * 255).astype("uint8")).permute(0, 3, 1, 2))
        prompts_saved.extend(batch_prompts)

    total_time = round(time.time() - start_time, 2)
    
    # Add debug timing range info similar to eval_prob_adaptive.py
    if UNET_CALL_COUNT > 0:
        time_efficiency = (UNET_TOTAL_TIME / total_time) * 100 if total_time > 0 else 0
        print(f"[DEBUG] UNet timing efficiency: {time_efficiency:.1f}% "
              f"({UNET_TOTAL_TIME:.2f}s UNet / {total_time:.2f}s total)")
    
    # Calculate FID score
    fid_score = calculate_fid(args.gt_path, img_dir, batch_size=args.fid_batch_size)
    
    # Enhanced Logging Section (same as eval_prob_adaptive.py)
    print(f"\n--- Evaluation Summary ---")
    print(f"Total wall-clock time for generation loop: {total_time:.2f} seconds")
    print(f"Total time spent *only* in UNet forward passes (during generation loop): {UNET_TOTAL_TIME:.4f} seconds")
    print(f"Approx. FLOPs per UNet forward pass (batch=1): {format_flops(unet_flops_per_pass)}")
    print(f"Total images generated: {total_images}")
    print(f"UNet calls: {UNET_CALL_COUNT}")
    
    if UNET_CALL_COUNT > 0:
        flops_per_call = unet_flops_per_pass  # FLOPs per single forward pass
        total_flops = flops_per_call * UNET_CALL_COUNT  # Total FLOPs based on actual calls
        avg_time_per_call = UNET_TOTAL_TIME / UNET_CALL_COUNT
        avg_time_per_image = UNET_TOTAL_TIME / total_images if total_images > 0 else 0
    else:
        flops_per_call = 0.0
        total_flops = 0.0
        avg_time_per_call = 0.0
        avg_time_per_image = 0.0
    
    print(f"FLOPs per UNet call: {format_flops(flops_per_call)}")
    print(f"Total UNet FLOPs: {format_flops(total_flops)}")
    print(f"Average UNet time per call: {avg_time_per_call:.4f} s")
    print(f"Average UNet time per image: {avg_time_per_image:.4f} s")
    print(f"FID Score: {fid_score:.4f}" if fid_score is not None else "FID Score: N/A")

    # Create detailed log data (same as eval_prob_adaptive.py)
    log_data = {
        "total_images": total_images,
        "total_pipeline_time_sec": total_time,
        "UNet_calls": UNET_CALL_COUNT,
        "FLOPs_per_UNet_call": format_flops(flops_per_call),
        "total_UNet_FLOPs": format_flops(total_flops),
        "accumulated_UNet_time_sec": UNET_TOTAL_TIME,
        "avg_UNet_time_per_call_sec": avg_time_per_call,
        "avg_UNet_time_per_image_sec": avg_time_per_image,
        "pipeline_type": pipeline_type,
        "fid_score": fid_score if fid_score is not None else "N/A",
        "args": vars(args)
    }

    log_filename = f"log_{exp_name}_{args.steps}_steps.json"
    log_filepath = os.path.join(img_dir, log_filename)
    
    with open(log_filepath, "w") as log_file:
        json.dump(log_data, log_file, indent=4)
    
    print(f"Result log saved at {log_filepath}")
    print(f"All images saved in {img_dir}")

    # Log to CSV for easy comparison across experiments (same as DiT script)
    log_to_csv(args, exp_name, img_dir, total_images, total_time, fid_score, 
               unet_flops_per_pass, UNET_TOTAL_TIME, UNET_CALL_COUNT)

    # Save results
    all_images = torch.cat(images_saved, dim=0)
    out_dir_ckpt = f"data/{args.dataset}_ckpt"
    os.makedirs(out_dir_ckpt, exist_ok=True)
    save_name = f"{exp_name}.pt"
    torch.save({"images": all_images, "prompts": prompts_saved}, os.path.join(out_dir_ckpt, save_name))
    print(f"Saved all generated images and prompts to {os.path.join(out_dir_ckpt, save_name)}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # === Dataset args ===

    
    parser.add_argument('--dataset', type=str, default='pets',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft','coco2017','imagenet100','imagenet100_randomseed0'], help='Dataset to use')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image size for processing')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        choices=list(INTERPOLATIONS.keys()),
                        help='Interpolation method for image resizing')





    # === Model version / Pipeline args ===
    parser.add_argument('--version', type=str, default='2-0',
                        choices=list(MODEL_IDS.keys()),
                        help='Stable Diffusion model version to load')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=('float16', 'float32'),
                        help='Model data type to use')
    

    # === Image-to-Image specific args ===
    parser.add_argument('--img2img_strength', type=float, default=0.75,
                        help='Strength for img2img transformation (0.0 preserves original, 1.0 completely transforms)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Guidance scale for stable diffusion (higher = more adherence to prompt)')


    # === Baseline vs. DeepCache toggles ===
    parser.add_argument("--original", action="store_true",
                        help="If set, load the official diffusers pipeline; else load a custom DeepCache pipeline.")
    parser.add_argument('--if_deepcache', type=int, default=0,
                        help='Enable DeepCache caching functionality (set to 1 to enable)')
    parser.add_argument('--cache_interval', type=int, default=1001,
                        help='Cache interval for DeepCache')
    parser.add_argument('--cache_branch_id', type=int, default=0,
                        help='Cache branch id for DeepCache')

    # === Token merging / Agentsd / AttentionProc toggles (if needed) ===
    parser.add_argument('--if_token_merging', type=int, default=0)
    parser.add_argument('--if_agentsd', type=int, default=0)
    parser.add_argument('--token_merging_ratio', type=float, default=0.5)
    parser.add_argument('--agent_ratio', type=float, default=0.5)
    parser.add_argument('--token_merging_use_rand', type=int, default=1)
    parser.add_argument('--token_merging_max_downsample', type=int, default=1)
    parser.add_argument('--token_merging_sx', type=int, default=2)
    parser.add_argument('--token_merging_sy', type=int, default=2)
    parser.add_argument('--token_merging_single_downsample_level_merge',
                        type=int, default=0)
    parser.add_argument('--merge_attn', type=int, default=1,
                        help='Whether or not to merge tokens for attention (1 for True, 0 for False)')
    parser.add_argument('--merge_crossattn', type=int, default=0,
                        help='Whether or not to merge tokens for cross attention (1 for True, 0 for False)')
    parser.add_argument('--merge_mlp', type=int, default=0,
                        help='Whether or not to merge tokens for MLP layers (1 for True, 0 for False)')
    parser.add_argument('--token_merging_cache_indices_per_image', type=int, default=0,
                        help='Whether to cache merge indices per image (1 for True, 0 for False)')

    ## for Token Merging Method (MLERP support)
    # Usage example: --if_token_merging=1 --token_merging_ratio=0.5 --token_merging_method=mlerp
    parser.add_argument('--token_merging_method', type=str, default='mean',
                        choices=['mean', 'mlerp', 'prune'],
                        help='Token merging method to use. Options: "mean" (standard average merging), '
                             '"mlerp" (MLERP - Maximum-Norm Linear Interpolation for better feature preservation), '
                             '"prune" (remove selected source tokens completely - most aggressive reduction). '
                             'MLERP preserves feature magnitudes and typically provides better accuracy at high reduction ratios.')

    ## for Proportional Attention
    parser.add_argument('--if_proportional_attention', type=int, default=0,
                        help='Whether to use proportional attention that accounts for token sizes (1 for True, 0 for False)')

    ## for Locality-based Sub-block Bipartite Matching
    parser.add_argument('--locality_block_factor_h', type=int, default=1,
                        help='Factor to divide height into sub-blocks for locality-based bipartite matching. '
                             '1 = global matching (default), >1 = local sub-block matching. '
                             'Higher values create more sub-blocks, constraining token matching to smaller spatial regions.')
    parser.add_argument('--locality_block_factor_w', type=int, default=1,
                        help='Factor to divide width into sub-blocks for locality-based bipartite matching. '
                             '1 = global matching (default), >1 = local sub-block matching. '
                             'Higher values create more sub-blocks, constraining token matching to smaller spatial regions.')

    ## for Resolution Caching within UNet (NEW)
    parser.add_argument('--cache_resolution_merge', type=int, default=0,
                        help='Whether to enable within-UNet resolution caching for scoring merge (1 for True, 0 for False). '
                             'Caches merge/unmerge functions for highest resolution layers (4096 tokens) to reuse across '
                             'multiple self-attention blocks within a single UNet forward pass. Only works with scoring merge.')
    parser.add_argument('--cache_resolution_mode', type=str, default='global',
                        choices=['global', 'block_specific'],
                        help='Resolution caching mode: "global" caches once per resolution across all block types (faster), '
                             '"block_specific" caches separately for down/up/mid blocks per resolution (more precise).')
    parser.add_argument('--debug_cache', type=int, default=0,
                        help='Whether to enable debug prints for resolution caching (1 for True, 0 for False). '
                             'Shows cache hits/misses and performance statistics.')

    # === Attention processor (downsampling) ===
    parser.add_argument('--if_attention_proc', type=int, default=0,
                        help='Whether to apply attention processor patching (1 for True, 0 for False)')
    parser.add_argument('--merge_tokens', type=str, default='keys/values',
                        choices=['keys/values', 'all'],
                        help='Which tokens to merge')
    parser.add_argument('--merge_method', type=str, default='downsample',
                        choices=['none', 'similarity', 'downsample','downsample_custom_block', 'frequency_blockwise','frequency_global', 'block_avg_pool', 'downsample_qkv_upsample_out', 'masked_attention', 'blockwise_masked_attention', 'snr_masked_attention', 'snr_blockwise_masked_attention', 'noise_magnitude_masked_attention', 'noise_magnitude_blockwise_masked_attention'],
                        help='Method to use for merging tokens')

    parser.add_argument('--qkv_downsample_method', type=str, default='nearest',
                        help='Method for downsampling QKV source (e.g., "avg_pool", "max_pool")')
    parser.add_argument('--out_upsample_method', type=str, default='nearest',
                        help='Method for upsampling attention output (e.g., "nearest", "bilinear")')

                        
    parser.add_argument('--downsample_method', type=str, default='nearest-exact',
                        help='Interpolation method for downsampling')
    parser.add_argument('--downsample_factor', type=int, default=2,
                        help='Factor to downsample tokens by')
    parser.add_argument('--linear_blend_factor', type=float, default=0.5,
                        help='Blend factor for linear_blend downsample method. '
                             '0.0 = pure avg_pool (smooth), 0.5 = 50/50 blend (default), 1.0 = pure nearest-exact (sharp)')
    parser.add_argument('--qkv_linear_blend_factor', type=float, default=0.5,
                        help='Blend factor for QKV linear_blend in downsample_qkv_upsample_out method')
    parser.add_argument('--out_linear_blend_factor', type=float, default=0.5,
                        help='Blend factor for output linear_blend in downsample_qkv_upsample_out method')
    parser.add_argument('--blockwise_blend_factor', type=float, default=0.5,
                        help='Blend factor for frequency_blockwise method. '
                             '0.0 = pure blockwise avg_pool, 0.5 = 50/50 blend (default), 1.0 = pure frequency_selected')
    
    # Timestep-based interpolation for linear blend
    parser.add_argument('--linear_blend_timestep_interpolation', type=int, default=0,
                        help='Enable timestep-based interpolation for linear_blend_factor (1 for True, 0 for False). '
                             'When enabled, blend factor changes dynamically from start_ratio to end_ratio based on timestep.')
    parser.add_argument('--linear_blend_start_ratio', type=float, default=0.1,
                        help='Starting blend factor at timestep 999 (high noise) for timestep interpolation. '
                             '0.0 = pure avg_pool, 1.0 = pure nearest-exact (default: 0.1)')
    parser.add_argument('--linear_blend_end_ratio', type=float, default=0.9,
                        help='Ending blend factor at timestep 0 (low noise) for timestep interpolation. '
                             '0.0 = pure avg_pool, 1.0 = pure nearest-exact (default: 0.9)')
    
    # QKV timestep-based interpolation
    parser.add_argument('--qkv_linear_blend_timestep_interpolation', type=int, default=0,
                        help='Enable timestep-based interpolation for QKV linear_blend_factor (1 for True, 0 for False)')
    parser.add_argument('--qkv_linear_blend_start_ratio', type=float, default=0.1,
                        help='Starting QKV blend factor at timestep 999 (default: 0.1)')
    parser.add_argument('--qkv_linear_blend_end_ratio', type=float, default=0.9,
                        help='Ending QKV blend factor at timestep 0 (default: 0.9)')
    
    # Output timestep-based interpolation
    parser.add_argument('--out_linear_blend_timestep_interpolation', type=int, default=0,
                        help='Enable timestep-based interpolation for output linear_blend_factor (1 for True, 0 for False)')
    parser.add_argument('--out_linear_blend_start_ratio', type=float, default=0.1,
                        help='Starting output blend factor at timestep 999 (default: 0.1)')
    parser.add_argument('--out_linear_blend_end_ratio', type=float, default=0.9,
                        help='Ending output blend factor at timestep 0 (default: 0.9)')
    
    # Linear blend method selection parameters
    parser.add_argument('--linear_blend_method_1', type=str, default='nearest-exact',
                        choices=['max_pool', 'avg_pool', 'area', 'bilinear', 'bicubic', 'nearest-exact', 
                                 'top_right', 'bottom_left', 'bottom_right', 'random', 'uniform_random', 'uniform_timestep'],
                        help='First method for linear_blend interpolation (default: nearest-exact)')
    parser.add_argument('--linear_blend_method_2', type=str, default='avg_pool',
                        choices=['max_pool', 'avg_pool', 'area', 'bilinear', 'bicubic', 'nearest-exact', 
                                 'top_right', 'bottom_left', 'bottom_right', 'random', 'uniform_random', 'uniform_timestep'],
                        help='Second method for linear_blend interpolation (default: avg_pool)')
    
    # QKV linear blend method selection parameters
    parser.add_argument('--qkv_linear_blend_method_1', type=str, default='nearest-exact',
                        choices=['max_pool', 'avg_pool', 'area', 'bilinear', 'bicubic', 'nearest-exact', 
                                 'top_right', 'bottom_left', 'bottom_right', 'random', 'uniform_random', 'uniform_timestep'],
                        help='First method for QKV linear_blend in downsample_qkv_upsample_out method')
    parser.add_argument('--qkv_linear_blend_method_2', type=str, default='avg_pool',
                        choices=['max_pool', 'avg_pool', 'area', 'bilinear', 'bicubic', 'nearest-exact', 
                                 'top_right', 'bottom_left', 'bottom_right', 'random', 'uniform_random', 'uniform_timestep'],
                        help='Second method for QKV linear_blend in downsample_qkv_upsample_out method')
    
    # Output linear blend method selection parameters
    parser.add_argument('--out_linear_blend_method_1', type=str, default='nearest-exact',
                        choices=['max_pool', 'avg_pool', 'area', 'bilinear', 'bicubic', 'nearest-exact', 
                                 'top_right', 'bottom_left', 'bottom_right', 'random', 'uniform_random', 'uniform_timestep'],
                        help='First method for output linear_blend in downsample_qkv_upsample_out method')
    parser.add_argument('--out_linear_blend_method_2', type=str, default='avg_pool',
                        choices=['max_pool', 'avg_pool', 'area', 'bilinear', 'bicubic', 'nearest-exact', 
                                 'top_right', 'bottom_left', 'bottom_right', 'random', 'uniform_random', 'uniform_timestep'],
                        help='Second method for output linear_blend in downsample_qkv_upsample_out method')
    parser.add_argument('--timestep_threshold_switch', type=float, default=0,
                        help='Percentage of generation left to switch to secondary method')
    parser.add_argument('--timestep_threshold_stop', type=float, default=0.0,
                        help='Percentage left to revert to normal attention')
    parser.add_argument('--secondary_merge_method', type=str, default='similarity',
                        choices=['none', 'similarity', 'downsample', 'frequency_blockwise','frequency_global', 'block_avg_pool', 'downsample_qkv_upsample_out', 'masked_attention', 'blockwise_masked_attention'],
                        help='Method to use after threshold switch')
    parser.add_argument('--downsample_factor_level_2', type=int, default=1,
                        help='Downsample amount for down block 2 depth')
    parser.add_argument('--ratio_level_2', type=float, default=0.0,
                        help='Ratio for similarity based merging for down block 2')
    parser.add_argument('--downsample_factor_h', type=int, default=1,
                        help='Downsample factor for height dimension, only used if merge_method is downsample_custom_block')
    parser.add_argument('--downsample_factor_w', type=int, default=1,
                        help='Downsample factor for width dimension, only used if merge_method is downsample_custom_block')

    parser.add_argument('--extra_guidance_scale', type=float, default=0,
                        help='Additional guidance scale to compensate for token merging')
    # --- Frequency-specific args (relevant if merge_method is frequency_*) ---

    parser.add_argument('--frequency_selection_mode', type=str, default='high',
                        help='Frequency selection mode for downsampling (high/low)')
    parser.add_argument('--frequency_selection_method', type=str, default='1d_dft',
                        choices=['original','1d_dft', '1d_dct', '2d_conv','non_uniform_grid','2d_conv_l2'], # Added choices based on your comment
                        help='Method for selecting frequencies: 1D DFT/DCT on features (1d_dft/1d_dct) or 2D spatial convolution (2d_conv). Only used if merge_method is "frequency".')
    parser.add_argument('--frequency_ranking_method', type=str, default='amplitude',choices=['amplitude', 'spectral_centroid','variance', 'l1norm', 'l2norm', 'mean_deviation'],
                        help='Method for ranking frequencies (e.g., by amplitude). Only used if merge_method is "frequency".')
    parser.add_argument('--selection_source', type=str, default='hidden',
                        choices=['hidden', 'key', 'query', 'value'],
                        help='Source for frequency selection (hidden/key/query/value). Only used if merge_method is "frequency" and downsample_method is "similarity".')
    parser.add_argument('--frequency_grid_alpha', type=float, default=2,
                        help='Bias strength parameter for non_uniform_grid (alpha > 1). Default 2.0')

    # SiTo arguments
    parser.add_argument('--if_sito', type=int, default=0,
                       help='Whether to apply SiTo method (1 for True, 0 for False)')
    parser.add_argument('--sito_prune_ratio', type=float, default=0.5,
                       help='Ratio of tokens to prune for SiTo')
    parser.add_argument('--sito_max_downsample_ratio', type=int, default=1,
                       help='Maximum downsample ratio for SiTo')
    parser.add_argument('--sito_prune_selfattn_flag', type=int, default=1,
                       help='Whether to prune self-attention tokens (1 for True, 0 for False)')
    parser.add_argument('--sito_prune_crossattn_flag', type=int, default=0,
                       help='Whether to prune cross-attention tokens (1 for True, 0 for False)')
    parser.add_argument('--sito_prune_mlp_flag', type=int, default=0,
                       help='Whether to prune MLP tokens (1 for True, 0 for False)')
    parser.add_argument('--sito_sx', type=int, default=2,
                       help='Stride in x dimension for SiTo')
    parser.add_argument('--sito_sy', type=int, default=2,
                       help='Stride in y dimension for SiTo')
    parser.add_argument('--sito_noise_alpha', type=float, default=0.1,
                       help='Noise alpha parameter for SiTo')
    parser.add_argument('--sito_sim_beta', type=float, default=1.0,
                       help='Similarity beta parameter for SiTo')

    # === Scoring-based Token Merging arguments ===
    parser.add_argument('--if_scoring_merge', type=int, default=0,
                       help='Whether to apply scoring-based token merging (1 for True, 0 for False)')
    parser.add_argument('--scoring_method', type=str, default='statistical',
                       choices=['frequency', 'spatial_filter', 'statistical', 'signal_processing', 'spatial_distribution', 'similarity'],
                       help='Scoring method for token merging')
    parser.add_argument('--scoring_preserve_ratio', type=float, default=0.3,
                       help='Ratio of tokens to protect from merging (0.0 to 1.0)')
    parser.add_argument('--scoring_mode', type=str, default='high',
                       choices=['high', 'low', 'medium', 'timestep_scheduler', 'reverse_timestep_scheduler', 'uniform'],
                       help='Mode for selecting tokens based on scores')
    parser.add_argument('--scoring_preserve_spatial_uniformity', type=int, default=0,
                       help='Whether to preserve spatial uniformity in scoring-based merging (1 for True, 0 for False). '
                            'True: applies bipartite matching to full image first, then filters protected tokens. '
                            'False: extracts mergeable subset first (current default behavior).')
    parser.add_argument('--if_low_frequency_dst_tokens', type=int, default=0,
                       help='Whether to select lowest-scored tokens as destinations within each spatial block (1 for True, 0 for False). '
                            'When enabled, uses token scores to guide destination selection instead of random/first position. '
                            'Only applies when scoring-based token merging is enabled.')
    
    # Frequency scorer specific parameters
    parser.add_argument('--scoring_freq_method', type=str, default='1d_dft',
                       choices=['1d_dft', '1d_dct'],
                       help='Method for frequency scoring')
    parser.add_argument('--scoring_freq_ranking', type=str, default='amplitude',
                       choices=['amplitude', 'spectral_centroid'],
                       help='Method for ranking frequencies')
    
    # Spatial filter scorer specific parameters
    parser.add_argument('--scoring_spatial_method', type=str, default='2d_conv',
                       choices=['2d_conv', '2d_conv_l2'],
                       help='Method for spatial filter scoring')
    parser.add_argument('--scoring_spatial_norm', type=str, default='l1',
                       choices=['l1', 'l2'],
                       help='Norm type for spatial filter scoring')
    
    # Statistical scorer specific parameters
    parser.add_argument('--scoring_stat_method', type=str, default='l2norm',
                       choices=['variance', 'l1norm', 'l2norm', 'mean_deviation'],
                       help='Method for statistical scoring')
    
    # Signal processing scorer specific parameters
    parser.add_argument('--scoring_signal_method', type=str, default='snr',
                       choices=['snr', 'noise_magnitude'],
                       help='Method for signal processing scoring')
    
    # Spatial distribution scorer specific parameters
    parser.add_argument('--scoring_spatial_alpha', type=float, default=2.0,
                       help='Alpha parameter for spatial distribution scoring (must be > 1)')
    
    # Similarity scorer specific parameters
    parser.add_argument('--scoring_similarity_method', type=str, default='local_neighbors_inverted',
                       choices=['local_neighbors_inverted', 'cosine_similarity', 'global_mean'],
                       help='Method for similarity scoring')
    
    # ABP (Adaptive Block Pooling) specific parameters
    parser.add_argument('--scoring_matching_algorithm', type=str, default='bipartite',
                       choices=['bipartite', 'abp'],
                       help='Matching algorithm for scoring-based token merging. Options: '
                            '"bipartite" (standard bipartite matching - more accurate), '
                            '"abp" (Adaptive Block Pooling - 2-10x faster with similar quality). '
                            'ABP is recommended for speed-critical applications.')
    parser.add_argument('--abp_scorer_method', type=str, default='spatial_filter',
                       choices=['frequency', 'spatial_filter', 'statistical', 'signal_processing', 'spatial_distribution', 'similarity'],
                       help='Scorer method to use for ABP tile evaluation. Only used when scoring_matching_algorithm=abp.')
    parser.add_argument('--abp_tile_aggregation', type=str, default='max',
                       choices=['max', 'min', 'sum', 'std'],
                       help='Method to aggregate token scores within each tile for ABP. '
                            'max: highest score per tile (default), min: lowest score per tile, '
                            'sum: total score per tile (same ordering as mean but faster), '
                            'std: standard deviation per tile (captures score variance). '
                            'Only used when scoring_matching_algorithm=abp.')

    # === Sampling setup ===
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    
    # === Image quality ===
    parser.add_argument("--quality", type=int, default=100,
                     help="JPEG quality (0-100)")

    # === FID Calculation & CSV Logging ===
    parser.add_argument('--gt_path', type=str, 
                        default="/path/to/datasets/imagenet/val_flat",
                        help="Path to ground truth images for FID calculation.")
    parser.add_argument('--fid_batch_size', type=int, default=50,
                        help="Batch size for FID calculation")
    parser.add_argument('--summary_csv', type=str, default='generation_experiments_summary.csv',
                        help='Path to the CSV file for logging experiment summaries.')

    # === Random seed ===
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for reproducibility")
    
    # === Attention processor consistency ===
    parser.add_argument('--force_attention_processor', type=str, default='AttnProcessor',
                        choices=['AttnProcessor', 'AttnProcessor2_0', 'XFormersAttnProcessor', 'none'])

    args = parser.parse_args()
    main(args)