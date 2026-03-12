
# Standard Library
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

# Third-party
import torch
import torch.nn as nn
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

from tqdm import tqdm
import subprocess
import re

from torch.utils.flop_counter import FlopCounterMode

from diffusers import DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0, XFormersAttnProcessor

from diffusion.datasets import get_target_dataset
from diffusion.models_dit import get_dit_model


# Global caches/counters
# Added Transformer specific counters and a FLOPs formatting helper
TRANSFORMER_TOTAL_TIME = 0.0   # cumulative time
TRANSFORMER_CALL_COUNT = 0



def log_to_csv(args, exp_name, img_dir, total_images, total_time, fid_score, 
               transformer_flops_per_pass, TRANSFORMER_TOTAL_TIME, TRANSFORMER_CALL_COUNT):
    """
    Log experiment results to a CSV file for easy comparison across runs
    """
    summary_csv_path = args.summary_csv
    print(f"Attempting to append results to summary CSV: {summary_csv_path}")
    
    try:
        headers = [
            'Timestamp', 'Experiment Name', 'Dataset',
            'Args JSON',
            'FID Score', 'Total Images', 'Generation Time (s)', 
            'DiT Forward Time (s)', 'DiT Calls', 'DiT FLOPs/pass',
            'Steps', 'Batch Size', 'Image Size', 'Guidance Scale',
            'Acceleration Method', 'Output Directory'
        ]

        # Determine acceleration method for easier filtering
        acceleration_method = "None"
        if args.if_token_merging == 1:
            method_suffix = f"_{getattr(args, 'token_merging_method', 'mean')}" if getattr(args, 'token_merging_method', 'mean') != 'mean' else ""
            prop_attn_suffix = "_ProportionalAttn" if getattr(args, 'if_proportional_attention', 0) else ""
            abp_suffix = ""
            if getattr(args, 'merge_method_alg', 'bipartite') == 'abp':
                abp_suffix = f"_ABP_{getattr(args, 'abp_tile_aggregation', 'max')}"
                if getattr(args, 'abp_scorer', None) is not None:
                    abp_suffix += f"_sc{args.abp_scorer}"
            acceleration_method = f"TokenMerge_{args.token_merging_ratio}{method_suffix}{abp_suffix}{prop_attn_suffix}"
        elif args.if_agentsd == 1:
            acceleration_method = f"AgentSD_{args.token_merging_ratio}_agent{args.agent_ratio}"
        elif args.if_attention_proc == 1:
            acceleration_method = f"AttnProc_{args.merge_method}"
        elif getattr(args, 'if_scoring_merge', 0) == 1:
            method_suffix = f"_{getattr(args, 'token_merging_method', 'mean')}" if getattr(args, 'token_merging_method', 'mean') != 'mean' else ""
            spatial_suffix = "_SpatialUniform" if getattr(args, 'scoring_preserve_spatial_uniformity', 0) else ""
            prop_attn_suffix = "_ProportionalAttn" if getattr(args, 'if_proportional_attention', 0) else ""
            abp_suffix = ""
            if getattr(args, 'merge_method_alg', 'bipartite') == 'abp':
                abp_suffix = f"_ABP_{getattr(args, 'abp_tile_aggregation', 'max')}"
                if getattr(args, 'abp_scorer', None) is not None:
                    abp_suffix += f"_scorer{args.abp_scorer}"
            acceleration_method = f"ScoringMerge_{args.scoring_method}_{args.token_merging_ratio}{method_suffix}{abp_suffix}{spatial_suffix}{prop_attn_suffix}"
        elif args.if_sito == 1:
            acceleration_method = f"SiTo_{args.sito_prune_ratio}_sx{args.sito_sx}_sy{args.sito_sy}_alpha{args.sito_noise_alpha}_beta{args.sito_sim_beta}"
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
            f"{TRANSFORMER_TOTAL_TIME:.4f}",
            TRANSFORMER_CALL_COUNT,
            format_flops(transformer_flops_per_pass) if transformer_flops_per_pass >= 0 else 'Failed',
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
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")


def cleanup_memory():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def debug_tome_layer_status(model, detailed=True):
    """
    Comprehensive debug function to inspect ToMe status across all layers
    
    Args:
        model: The transformer model to inspect
        detailed: Whether to show detailed per-block information
    
    Returns:
        dict: Summary of ToMe status
    """
    print(f"\n{'='*80}")
    print(f"DEBUG: ToMe Layer Status Inspection")
    print(f"{'='*80}")
    
    # Check if model has ToMe info
    has_tome_info = hasattr(model, '_tome_info') and model._tome_info is not None
    print(f"Model has _tome_info: {has_tome_info}")
    
    if not has_tome_info:
        print("No ToMe information found in model")
        return {"has_tome": False, "enabled_blocks": 0, "total_blocks": 0}
    
    # Extract ToMe configuration
    tome_info = model._tome_info
    tome_args = tome_info.get("args", {})
    
    print(f"\nToMe Configuration:")
    print(f"  • Ratio: {tome_args.get('ratio', 'N/A')}")
    print(f"  • Method: {tome_args.get('method', 'N/A')}")
    print(f"  • Merge Attention: {tome_args.get('merge_attn', 'N/A')}")
    print(f"  • Merge Cross-Attention: {tome_args.get('merge_crossattn', 'N/A')}")
    print(f"  • Merge MLP: {tome_args.get('merge_mlp', 'N/A')}")
    print(f"  • Use Scoring: {tome_args.get('use_scoring', 'N/A')}")
    print(f"  • Cache Indices: {tome_args.get('cache_indices_per_image', 'N/A')}")
    
    # Count transformer blocks and check ToMe status
    transformer_blocks = []
    tome_enabled_blocks = 0
    
    print(f"\nTransformer Block Analysis:")
    
    for name, module in model.named_modules():
        if "transformer_blocks" in name and (hasattr(module, 'attn') or hasattr(module, 'attn1')):
            transformer_blocks.append((name, module))
    
    print(f"Found {len(transformer_blocks)} transformer blocks")
    
    # Detailed block inspection
    if detailed and transformer_blocks:
        print(f"\nPer-Block ToMe Status:")
        print(f"{'Block':<20} {'Class':<25} {'ToMe Enabled':<12} {'Attn Processor':<30}")
        print(f"{'-'*90}")
        
        for i, (name, block) in enumerate(transformer_blocks):
            # Check if block has ToMe enabled
            tome_enabled = getattr(block, '_tome_enabled', False)
            block_class = block.__class__.__name__
            
            # Check attention processor type
            attn_processor = "Unknown"
            has_tome_processor = False
            if hasattr(block, 'attn') and hasattr(block.attn, 'processor'):
                attn_processor = block.attn.processor.__class__.__name__
                has_tome_processor = "ToMe" in attn_processor
            elif hasattr(block, 'attn1') and hasattr(block.attn1, 'processor'):
                attn_processor = block.attn1.processor.__class__.__name__
                has_tome_processor = "ToMe" in attn_processor
            
            # A block is ToMe-enabled if it has ToMe processor, _tome_enabled flag, or is ToMeBlock class
            is_tome_enabled = (has_tome_processor or 
                             tome_enabled or 
                             block_class == "ToMeBlock")
            
            if is_tome_enabled:
                tome_enabled_blocks += 1
                status_icon = "Y"
            else:
                status_icon = "N"
            
            block_short_name = name.split('.')[-1] if '.' in name else name
            print(f"{block_short_name:<20} {block_class:<25} {status_icon:<12} {attn_processor:<30}")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  • Total transformer blocks: {len(transformer_blocks)}")
    print(f"  • ToMe-enabled blocks: {tome_enabled_blocks}")
    print(f"  • Coverage: {tome_enabled_blocks/len(transformer_blocks)*100:.1f}%" if transformer_blocks else "  • Coverage: 0%")
    
    # Check for cached indices
    cached_indices = tome_info.get("cached_indices", {})
    if cached_indices:
        print(f"  • Cached merge patterns: {len(cached_indices)} levels")
    
    print(f"{'='*80}\n")
    
    return {
        "has_tome": True,
        "enabled_blocks": tome_enabled_blocks,
        "total_blocks": len(transformer_blocks),
        "coverage_percent": tome_enabled_blocks/len(transformer_blocks)*100 if transformer_blocks else 0,
        "tome_args": tome_args
    }


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



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



# Modified save_images_as_jpg to handle class labels instead of text prompts
def save_images_as_jpg(images, class_labels, output_dir, quality=100, start_idx=0):
    """
    Save batch of images directly to JPG files, labeled by class.

    Args:
        images: numpy array of images (batch, H, W, 3) with values 0-1
        class_labels: list of class labels corresponding to images
        output_dir: directory to save images
        quality: JPEG quality (0-100)
        start_idx: starting index for image numbering
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save class labels to a JSON file
    labels_file = os.path.join(output_dir, "class_labels.json")
    all_labels = []
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            try:
                all_labels = json.load(f)
            except json.JSONDecodeError:
                pass  # File is empty or corrupt, will overwrite
    
    all_labels.extend(class_labels)

    with open(labels_file, 'w') as f:
        json.dump(all_labels, f, indent=2)

    for i, (img, label) in enumerate(zip(images, class_labels)):
        img_idx = start_idx + i
        # Convert from 0-1 float to uint8
        if img.dtype != np.uint8:
            img_int = (np.clip(img, 0, 1) * 255).astype("uint8")
        else:
            img_int = img
        pil_img = Image.fromarray(img_int)

        # Create a filename with class label
        filename = f"image_{img_idx:05d}_class_{label}.jpg"

        # Save as JPG
        pil_img.save(os.path.join(output_dir, filename), "JPEG", quality=quality)


def main(args):
    # Declare globals to be modified
    global TRANSFORMER_TOTAL_TIME, TRANSFORMER_CALL_COUNT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_memory_usage("Start")
    seed_everything(args.seed)


    if args.class_csv_path is not None:
        # Load class IDs from CSV file (e.g., imagenet100_prompts.csv)
        class_df = pd.read_csv(args.class_csv_path)
        if 'classidx' in class_df.columns:
            class_indices = class_df['classidx'].tolist()
        elif 'class_id' in class_df.columns:
            class_indices = class_df['class_id'].tolist()
        elif 'class_idx' in class_df.columns:
            class_indices = class_df['class_idx'].tolist()
        else:
            # Try to find any column that looks like class indices
            possible_cols = [col for col in class_df.columns if 'class' in col.lower() and ('id' in col.lower() or 'idx' in col.lower())]
            if possible_cols:
                class_indices = class_df[possible_cols[0]].tolist()
            else:
                raise ValueError(f"Could not find class index column in {args.class_csv_path}. Available columns: {class_df.columns.tolist()}")
        print(f"Loaded {len(class_indices)} class IDs from {args.class_csv_path}")
    elif args.class_subset is not None:
        class_indices = np.load(args.class_subset).tolist()
        print(f"Loaded {len(class_indices)} class IDs from subset file {args.class_subset}")
    else:
        class_indices = list(range(args.num_classes))
        print(f"Using default class range: 0 to {args.num_classes-1}")
    
    print(f"Evaluating {len(class_indices)} classes: {class_indices[:10]}{'...' if len(class_indices) > 10 else ''}")



    # Load dataset with appropriate transformation
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    dataset = get_target_dataset(args.dataset, train=False, transform=transform)

    # Simplified data handling logic. No more prompts, only class labels.
    # The dataset now returns (image, label) tuples.
    # We will extract the labels during the batching process.
    print(f"Loaded dataset '{args.dataset}' with {len(dataset)} samples.")
    # Removed all prompt extraction logic for coco, pets, flowers, etc.

    # Removed the call to get_sd_pipeline and the concept of pipeline_type.
    # Instead, we load the DiT components directly.
    print("Loading DiT model components...")
    vae, _, _, transformer, _ = get_dit_model(args)
    print("Creating DDIM Solver configuration...")
    ddim_config = {
                "_class_name": "DDIMScheduler",
                "_diffusers_version": "0.12.0.dev0",
                "beta_end": 0.02,
                "beta_schedule": "linear",
                "beta_start": 0.0001,
                "clip_sample": False,
                "num_train_timesteps": 1000,
                "prediction_type": "epsilon",
                "set_alpha_to_one": True,
                "steps_offset": 0,
                "trained_betas": None
                }


    # Create the DDIMScheduler instance from the configuration.
    scheduler = DDIMScheduler.from_config(ddim_config)
    scheduler.set_timesteps(args.steps, device=device) 
    print(f"Successfully created DDIM Solver ({type(scheduler).__name__}).")








    vae = vae.to(device)
    transformer = transformer.to(device)

    # Enable eval mode to save memory
    vae.eval()
    transformer.eval()
    print("DiT model loaded successfully.")
    print_memory_usage("After model loading")



    # Note: Scoring-based token merging is already applied in get_dit_model() in models_dit.py
    # No need to apply it again here to avoid duplication and potential block_tome_flags issues

    # Force consistent attention processor for fair benchmarking
    if args.force_attention_processor == 'none':
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
            transformer.set_attn_processor(chosen_processor)
            print(f"Set DiT attention processor to: {type(chosen_processor).__name__}")
        except Exception as e:
            print(f"Warning: Failed to set attention processor ({e}). Using DiT's current state.")

    # DEBUG: Inspect ToMe layer status after model setup
    print(f"\nDEBUG: Checking ToMe status after model setup...")
    tome_status = debug_tome_layer_status(transformer, detailed=True)
    
    # Reset global counters
    TRANSFORMER_TOTAL_TIME = 0.0
    TRANSFORMER_CALL_COUNT = 0
    
    # Measure FLOPs for one DiT forward pass
    transformer_flops_per_pass = 0.0
    print(f"\nMeasuring FLOPs for one DiT forward pass...")
    latent_size = args.img_size // 8
 
    try:
        b_dummy = 1
        dt = torch.float16 if args.dtype == "float16" else torch.float32
        
        with torch.no_grad():  # Ensure no gradients
            dummy_latents = torch.randn(
                b_dummy, 4, latent_size, latent_size,
                device=device, dtype=dt
            )
            dummy_timestep = torch.tensor([500], device=device)
            dummy_class_labels = torch.tensor([0], device=device, dtype=torch.long)

            with FlopCounterMode(transformer, display=False) as fc:
                _ = transformer(dummy_latents, dummy_timestep, class_labels=dummy_class_labels, return_dict=False)[0]

            transformer_flops_per_pass = fc.get_total_flops()
            print(f"Measured FLOPs/pass (batch=1): {format_flops(transformer_flops_per_pass)}")

    except Exception as e:
        print(f"ERROR while counting FLOPs: {e}")
        transformer_flops_per_pass = -1
    finally:
        del dummy_latents, dummy_timestep, dummy_class_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear any ToMe cached state from FLOPs measurement
        # to prevent interference with actual generation
        if hasattr(transformer, '_tome_info') and transformer._tome_info is not None:
            if 'cached_indices' in transformer._tome_info:
                transformer._tome_info['cached_indices'].clear()
            if 'token_sizes' in transformer._tome_info:
                transformer._tome_info['token_sizes'] = None
            if 'current_image_latent' in transformer._tome_info:
                transformer._tome_info['current_image_latent'] = None
            transformer._tome_info['image_counter'] = 0
            print("Cleared ToMe cached state after FLOPs measurement")




    # Create output directory
    out_dir = f"data/{args.dataset}_generated_images"
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Experiment Naming Logic ---
    # This block generates a detailed folder name consistent with eval_prob_adaptive_dit.py
    name = f"DiT{args.img_size}"


    if args.if_token_merging == 1:
        name += f'_tokenmerge{args.token_merging_ratio}_userand{args.token_merging_use_rand}_maxdownsample{args.token_merging_max_downsample}_singlelevel{args.token_merging_single_downsample_level_merge}_sx{args.token_merging_sx}_sy{args.token_merging_sy}'
        # Add method to name if not default
        if getattr(args, 'token_method', 'mean') != 'mean':
            name += f'_method{args.token_merging_method}'
        # Add ABP information if ABP is used
        if getattr(args, 'merge_alg', 'bipartite') == 'abp':
            name += f'_abp'
            if getattr(args, 'abp_aggre', 'max') != 'max':
                name += f'_agg{args.abp_tile_aggregation}'
            if getattr(args, 'abp_scorer', None) is not None:
                name += f'_scorer{args.abp_scorer}'
        # Add proportional attention if enabled
        if getattr(args, 'if_proportional_attention', 0) == 1:
            name += '_propattn'
        # Add cache parameters to folder name
        if getattr(args, 'cache_merge_functions', 0) == 1:
            name += f'_cache{args.cache_recalc_interval}'
        
        # Add block flags information to folder name if custom flags are used for token merging
        if args.block_tome_flags is not None and getattr(args, 'block_tome_flags_parsed', None) is not None:
            # Compress the block flags representation for folder name
            block_tome_flags = args.block_tome_flags_parsed
            num_enabled = sum(block_tome_flags)
            total_blocks = len(block_tome_flags)
            
            # Always add a hash for any custom block flags pattern
            pattern_str = ''.join(map(str, block_tome_flags))
            import hashlib
            pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()[:6]
            
            if num_enabled == total_blocks:
                name += f'_allblocks_pat{pattern_hash}'
            elif num_enabled == 0:
                name += f'_noblocks_pat{pattern_hash}'
            else:
                name += f'_blocks{num_enabled}of{total_blocks}_pat{pattern_hash}'
    elif args.if_agentsd == 1:
        name += f'_agentsd{args.token_merging_ratio}_agentratio{args.agent_ratio}_userand{args.token_merging_use_rand}_maxdownsample{args.token_merging_max_downsample}_singlelevel{args.token_merging_single_downsample_level_merge}_sx{args.token_merging_sx}_sy{args.token_merging_sy}'
        # Add method to name if not default
        if getattr(args, 'token_merging_method', 'mean') != 'mean':
            name += f'_method{args.token_merging_method}'
        # Add ABP information if ABP is used
        if getattr(args, 'merge_method_alg', 'bipartite') == 'abp':
            name += f'_abp'
            if getattr(args, 'abp_tile_aggregation', 'max') != 'max':
                name += f'_agg{args.abp_tile_aggregation}'
            if getattr(args, 'abp_scorer', None) is not None:
                name += f'_scorer{args.abp_scorer}'
        # Add proportional attention if enabled
        if getattr(args, 'if_proportional_attention', 0) == 1:
            name += '_propattn'
        # Add cache parameters to folder name
        if getattr(args, 'cache_merge_functions', 0) == 1:
            name += f'_cache{args.cache_recalc_interval}'
    elif getattr(args, 'if_scoring_merge', 0) == 1:
        name += f'_scomerge{args.token_merging_ratio}_method{args.scoring_method}_preserve{args.scoring_preserve_ratio}_mode{args.scoring_mode}_userand{args.token_merging_use_rand}_maxdownsample{args.token_merging_max_downsample}_sx{args.token_merging_sx}_sy{args.token_merging_sy}'
        
        # Add method-specific parameters to name
        if args.scoring_method == "frequency":
            name += f'_freqmethod{args.scoring_freq_method}_freqranking{args.scoring_freq_ranking}'
        elif args.scoring_method == "spatial_filter":
            name += f'_spatial{args.scoring_spatial_method}_spanorm{args.scoring_spatial_norm}'
        elif args.scoring_method == "statistical":
            name += f'_statmethod{args.scoring_stat_method}'
        elif args.scoring_method == "signal_processing":
            name += f'_signalmethod{args.scoring_signal_method}'
        elif args.scoring_method == "spatial_distribution":
            name += f'_spatialalpha{args.scoring_spatial_alpha}'
        
        # Add merging method if not default
        if getattr(args, 'token_merging_method', 'mean') != 'mean':
            name += f'_mergemethod{args.token_merging_method}'
        
        # Add ABP information if ABP is used
        if getattr(args, 'merge_method_alg', 'bipartite') == 'abp':
            name += f'_abp'
            if getattr(args, 'abp_tile_aggregation', 'max') != 'max':
                name += f'_agg{args.abp_tile_aggregation}'
            if getattr(args, 'abp_scorer', None) is not None:
                name += f'_scorer{args.abp_scorer}'
        
        # Add spatial uniformity parameter if enabled
        if getattr(args, 'scoring_preserve_spatial_uniformity', 0) == 1:
            name += '_spatialuniform'
        
        # Add score-guided destination selection parameter if enabled
        if getattr(args, 'if_low_frequency_dst_tokens', 0) == 1:
            name += '_lowfreqdst'
        
        # Add proportional attention if enabled
        if getattr(args, 'if_proportional_attention', 0) == 1:
            name += '_propattn'
        
        # Add cache parameters to folder name
        if getattr(args, 'cache_merge_functions', 0) == 1:
            name += f'_cache{args.cache_recalc_interval}'
        
        # Add block flags information to folder name if custom flags are used for scoring merge
        if args.block_tome_flags is not None and getattr(args, 'block_tome_flags_parsed', None) is not None:
            # Compress the block flags representation for folder name
            block_tome_flags = args.block_tome_flags_parsed
            num_enabled = sum(block_tome_flags)
            total_blocks = len(block_tome_flags)
            
            # Always add a hash for any custom block flags pattern
            pattern_str = ''.join(map(str, block_tome_flags))
            import hashlib
            pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()[:6]
            
            if num_enabled == total_blocks:
                name += f'_allblocks_pat{pattern_hash}'
            elif num_enabled == 0:
                name += f'_noblocks_pat{pattern_hash}'
            else:
                name += f'_blocks{num_enabled}of{total_blocks}_pat{pattern_hash}'
    elif args.if_attention_proc == 1:
        if args.merge_method == 'frequency_global':
            name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqmethod{args.frequency_selection_method}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_ratio{args.token_merging_ratio}')
            if args.frequency_selection_method=='non_uniform_grid':
                name += f'_gridalpha{args.frequency_grid_alpha}'
        elif args.merge_method == 'frequency_blockwise':
            name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqmethod{args.frequency_selection_method}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_downsample{args.downsample_factor}')
            # Add blockwise blend parameter to name if not default
            if getattr(args, 'blockwise_blend_factor', 0.5) != 0.5:
                name += f'_blockwiseblend{args.blockwise_blend_factor}'
        elif args.merge_method == 'downsample_qkv_upsample_out':
            name += (f'_mergemethod{args.merge_method}'
                    f'_downsamplemethod{args.downsample_method}'
                    f'_qkv_downsamplemethod{args.qkv_downsample_method}'
                    f'_out_upsamplemethod{args.out_upsample_method}'
                    f'_level1downsample{args.downsample_factor}')
        elif args.merge_method == 'similarity':
            name += (f'_mergemethod{args.merge_method}'
                    f'_ratio{args.token_merging_ratio}'
                    f'_mergetokens{args.merge_tokens}'
                    f'_selectionsource{args.selection_source}'
                    f'_sx{args.token_merging_sx}'
                    f'_sy{args.token_merging_sy}')
        elif args.merge_method == 'downsample_custom_block':
            name += (f'_mergemethod{args.merge_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_h{args.downsample_factor_h}'
                    f'_w{args.downsample_factor_w}'
                    f'_selectionsource{args.selection_source}'
                    f'_switch{args.timestep_threshold_switch}'
                    f'_stop{args.timestep_threshold_stop}')
        else:
            name += (f'_downsampling{args.token_merging_ratio}_userand{args.token_merging_use_rand}_sx{args.token_merging_sx}_sy{args.token_merging_sy}'
                    f'_method{args.merge_method}'
                    f'_down{args.downsample_factor}_downmethod{args.downsample_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_switch{args.timestep_threshold_switch}'
                    f'_stop{args.timestep_threshold_stop}')
            
            # Add linear_blend parameter to name if using linear_blend method
            if args.downsample_method == 'linear_blend':
                name += f'_linearblend{getattr(args, "blend_factor", 0.5)}'
                name += f'_blendmethods{getattr(args, "blend_method_1", "nearest-exact")}-{getattr(args, "blend_method_2", "avg_pool")}'
        
        # Add cache parameters to folder name
        if getattr(args, 'cache_merge_functions', 0) == 1:
            name += f'_cache{args.cache_recalc_interval}'
        
        # Add block flags information to folder name if custom flags are used for attention processor
        if args.block_tome_flags is not None and getattr(args, 'block_tome_flags_parsed', None) is not None:
            # Compress the block flags representation for folder name
            block_tome_flags = args.block_tome_flags_parsed
            num_enabled = sum(block_tome_flags)
            total_blocks = len(block_tome_flags)
            
            # Always add a hash for any custom block flags pattern
            pattern_str = ''.join(map(str, block_tome_flags))
            import hashlib
            pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()[:6]
            
            if num_enabled == total_blocks:
                name += f'_allblocks_pat{pattern_hash}'
            elif num_enabled == 0:
                name += f'_noblocks_pat{pattern_hash}'
            else:
                name += f'_blocks{num_enabled}of{total_blocks}_pat{pattern_hash}'
                        
    elif args.if_sito == 1:
        name += (f'_sito_prune{args.sito_prune_ratio}'
                f'_maxdown{args.sito_max_downsample_ratio}'
                f'_sx{args.sito_sx}_sy{args.sito_sy}'
                f'_noisealpha{args.sito_noise_alpha}'
                f'_simbeta{args.sito_sim_beta}'
                f'_selfattn{args.sito_prune_selfattn}'
                f'_crossattn{args.sito_prune_crossattn}'
                f'_mlp{args.sito_prune_mlp}')
        
        # Add block flags information to folder name if custom flags are used
        if args.block_sito_flags is not None and getattr(args, 'block_sito_flags_parsed', None) is not None:
            # Compress the block flags representation for folder name
            block_sito_flags = args.block_sito_flags_parsed
            num_enabled = sum(block_sito_flags)
            total_blocks = len(block_sito_flags)
            
            # Always add a hash for any custom block flags pattern
            pattern_str = ''.join(map(str, block_sito_flags))
            import hashlib
            pattern_hash = hashlib.md5(pattern_str.encode()).hexdigest()[:6]
            
            if num_enabled == total_blocks:
                name += f'_allblocks_pat{pattern_hash}'
            elif num_enabled == 0:
                name += f'_noblocks_pat{pattern_hash}'
            else:
                name += f'_blocks{num_enabled}of{total_blocks}_pat{pattern_hash}'
    
    if args.if_deepcache == 1:
        name += f'_deepcache_interval{args.cache_interval}_branch{args.cache_branch_id}'








    # Combine the detailed name with steps and seed for the final folder name
    exp_name = f"{name}_{args.steps}steps_seed{args.seed}"
    img_dir = os.path.join(out_dir, exp_name)
    os.makedirs(img_dir, exist_ok=True)
    
    print(f"Generating and saving images to {img_dir}")











    
    total_images = 0
    all_generated_images = []
    all_class_labels = []

    num_samples = len(dataset)
    num_batch = (num_samples + args.batch_size - 1) // args.batch_size

    # DEBUG: Enable comprehensive ToMe monitoring during generation
    tome_debug_enabled = False
    if hasattr(transformer, '_tome_info') and transformer._tome_info is not None:
        transformer._tome_info["debug_generation"] = True
        tome_debug_enabled = True
        print(f"\nDEBUG: Enabling real-time ToMe monitoring during generation")
        print(f"DEBUG: Will monitor {num_batch} batches with {args.steps} steps each")
        
        # Quick pre-generation ToMe verification (using same logic as detailed inspection)
        enabled_count = 0
        transformer_blocks = []
        
        # First collect all transformer blocks (same logic as debug_tome_layer_status)
        for name, module in transformer.named_modules():
            if "transformer_blocks" in name and (hasattr(module, 'attn') or hasattr(module, 'attn1')):
                transformer_blocks.append((name, module))
        
        # Then count ToMe-enabled blocks using the same logic
        for name, block in transformer_blocks:
            # Check attention processor type
            has_tome_processor = False
            if hasattr(block, 'attn') and hasattr(block.attn, 'processor'):
                processor_name = block.attn.processor.__class__.__name__
                has_tome_processor = "ToMe" in processor_name
            elif hasattr(block, 'attn1') and hasattr(block.attn1, 'processor'):
                processor_name = block.attn1.processor.__class__.__name__
                has_tome_processor = "ToMe" in processor_name
            
            # A block is ToMe-enabled if it has ToMe processor, _tome_enabled flag, or is ToMeBlock class
            is_tome_enabled = (has_tome_processor or 
                             getattr(block, '_tome_enabled', False) or 
                             block.__class__.__name__ == "ToMeBlock")
            
            if is_tome_enabled:
                enabled_count += 1
        
        print(f"DEBUG: Found {enabled_count} ToMe-enabled transformer blocks for generation")
        
        # Set up ToMe call tracking
        transformer._tome_info["generation_calls"] = 0
        transformer._tome_info["generation_merges"] = 0


    start_time = time.time()

    for i in tqdm(range(num_batch), desc="Generating Batches"):
        st = args.batch_size * i
        ed = min(args.batch_size * (i + 1), num_samples)
        
        # Use same sequence as dataset but map to correct class indices 
        batch_class_labels = []
        for j in range(st, ed):
            _, original_label = dataset[j]
            
            # Find the position of this label in class_indices
            try:
                batch_class_labels.append(class_indices[original_label])
            except ValueError:
                # If label not in class_indices, skip this sample
                print(f"Warning: Label {original_label} not found in class_indices, skipping sample {j}")
                continue







        current_batch_size = len(batch_class_labels)
        guidance_scale = args.guidance_scale

        with torch.no_grad():
            # 1. Prepare inputs for DiT
            class_labels_tensor = torch.tensor(batch_class_labels, device=device, dtype=torch.long)
            latents = torch.randn(
                (current_batch_size, 4, latent_size, latent_size),
                device=device,
                dtype=torch.float16 if args.dtype == 'float16' else torch.float32
            ) 
            
            # 2. Setup for classifier-free guidance
            if guidance_scale > 1:
                latent_model_input = torch.cat([latents] * 2)
                class_null = torch.tensor([1000] * current_batch_size, device=device)  # 1000 = null class for ImageNet
                class_labels_input = torch.cat([class_labels_tensor, class_null], 0)
            else:
                latent_model_input = latents
                class_labels_input = class_labels_tensor
            
            # 3. Denoising loop
            for t in scheduler.timesteps:
                # Prepare latent input
                if guidance_scale > 1:
                    half = latent_model_input[:len(latent_model_input) // 2]
                    latent_model_input = torch.cat([half, half], dim=0)
                
                # Scale model input
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                
                # Handle timesteps properly (from official pipeline)
                timesteps = t
                if not torch.is_tensor(timesteps):
                    # This requires sync between CPU and GPU. Try to pass timesteps as tensors if you can
                    is_mps = latent_model_input.device.type == "mps"
                    is_npu = latent_model_input.device.type == "npu"
                    if isinstance(timesteps, float):
                        dtype = torch.float32 if (is_mps or is_npu) else torch.float64
                    else:
                        dtype = torch.int32 if (is_mps or is_npu) else torch.int64
                    timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
                elif len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(latent_model_input.device)
                
                # Broadcast to batch dimension
                timesteps = timesteps.expand(latent_model_input.shape[0])
                
                # Predict noise
                torch.cuda.synchronize(); iter_start_time = time.time()
                
                # DEBUG: Monitor ToMe activity during transformer call
                if tome_debug_enabled and i == 0 and list(scheduler.timesteps).index(t) < 3:  # Only for first batch, first few steps
                    step_idx = list(scheduler.timesteps).index(t)
                    print(f"DEBUG: Batch {i+1}, Step {step_idx+1}/{len(scheduler.timesteps)} (t={t:.0f})")
                    print(f"  • Input shape: {latent_model_input.shape}")
                    print(f"  • Class labels: {class_labels_input.shape}")
                    
                    # Check if ToMe cached indices exist for this call
                    if hasattr(transformer, '_tome_info') and transformer._tome_info is not None:
                        cached_indices = transformer._tome_info.get("cached_indices", {})
                        if cached_indices:
                            print(f"  • ToMe cached patterns: {list(cached_indices.keys())}")
                        else:
                            print(f"  • ToMe: No cached patterns (will compute fresh)")
                        
                        # Increment call counter
                        transformer._tome_info["generation_calls"] = transformer._tome_info.get("generation_calls", 0) + 1
                
                noise_pred = transformer(
                    latent_model_input, 
                    timestep=timesteps, 
                    class_labels=class_labels_input
                ).sample
                
                torch.cuda.synchronize(); iter_end_time = time.time()
                TRANSFORMER_TOTAL_TIME += (iter_end_time - iter_start_time)
                TRANSFORMER_CALL_COUNT += 1
                
                # DEBUG: Report post-call ToMe status
                if tome_debug_enabled and i == 0 and list(scheduler.timesteps).index(t) < 3:
                    step_time = iter_end_time - iter_start_time
                    print(f"  • Step completed in {step_time:.4f}s")
                    
                    # Check if ToMe statistics are available
                    if hasattr(transformer, '_tome_info') and transformer._tome_info is not None:
                        tome_info = transformer._tome_info
                        generation_calls = tome_info.get("generation_calls", 0)
                        print(f"  • Total ToMe calls so far: {generation_calls}")
                        
                        # Check for new cached patterns
                        cached_indices = tome_info.get("cached_indices", {})
                        if cached_indices:
                            print(f"  • Updated cached patterns: {len(cached_indices)} levels")
                
                # Apply classifier-free guidance
                if guidance_scale > 1:
                    eps, rest = noise_pred[:, :4], noise_pred[:, 4:]  # 4 = latent channels
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                    
                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                    eps = torch.cat([half_eps, half_eps], dim=0)
                    
                    noise_pred = torch.cat([eps, rest], dim=1)
                
                # Handle learned sigma (from official pipeline)
                if transformer.config.out_channels // 2 == 4:  # 4 = latent channels
                    model_output, _ = torch.split(noise_pred, 4, dim=1)
                else:
                    model_output = noise_pred
                
                # Compute previous image: x_t -> x_{t-1}
                latent_model_input = scheduler.step(model_output, t, latent_model_input).prev_sample
            
            # 4. Extract final latents (handle guidance)
            if guidance_scale > 1:
                latents, _ = latent_model_input.chunk(2, dim=0)
            else:
                latents = latent_model_input
            
            # 5. Decode the latents with VAE
            latents = 1 / vae.config.scaling_factor * latents
            decoded_images = vae.decode(latents).sample
            
            # 6. Post-process images to save
            decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
            decoded_images = (decoded_images * 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()

            images_to_save = decoded_images



        
        # Save directly to JPG
        save_images_as_jpg(
            images=images_to_save, 
            class_labels=batch_class_labels,
            output_dir=img_dir,
            quality=args.quality,
            start_idx=total_images
        )
        
        total_images += len(images_to_save)
        all_generated_images.append(torch.from_numpy(images_to_save).permute(0, 3, 1, 2))
        all_class_labels.extend(batch_class_labels)

    total_time = round(time.time() - start_time, 2)
    
    # DEBUG: Final ToMe generation summary
    if tome_debug_enabled and hasattr(transformer, '_tome_info') and transformer._tome_info is not None:
        print(f"\nDEBUG: ToMe Generation Summary")
        print(f"{'='*60}")
        tome_info = transformer._tome_info
        generation_calls = tome_info.get("generation_calls", 0)
        cached_indices = tome_info.get("cached_indices", {})
        
        print(f"• Total ToMe-monitored calls: {generation_calls}")
        print(f"• Cached merge patterns: {len(cached_indices)} levels")
        print(f"• Total transformer calls: {TRANSFORMER_CALL_COUNT}")
        print(f"• ToMe monitoring coverage: {generation_calls/TRANSFORMER_CALL_COUNT*100:.1f}%" if TRANSFORMER_CALL_COUNT > 0 else "• ToMe monitoring coverage: 0%")
        
        if cached_indices:
            print(f"• Cached pattern keys: {list(cached_indices.keys())}")
        
        print(f"{'='*60}\n")
    
    # --- FID Score Calculation and Logging ---
    # Calculate FID score against the provided ground truth path
    fid_score = calculate_fid(args.gt_path, img_dir, batch_size=args.fid_batch_size)
    
    # Update metadata with the calculated FID score
    metadata = {
        "total_images": total_images,
        "generation_time_sec": total_time,
        "steps": args.steps,
        "model_used": "DiT",
        "experiment_name": exp_name,
        "fid_score": fid_score if fid_score is not None else "N/A"
    }
    
    metadata_file = os.path.join(img_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    



    # --- Enhanced Logging Section ---
    print(f"\n--- Run Statistics ---")
    print(f"Total images generated: {total_images}")
    print(f"Total pipeline time: {total_time:.2f} s")
    print(f"DiT Transformer calls: {TRANSFORMER_CALL_COUNT}")
    
    if TRANSFORMER_CALL_COUNT > 0:
        flops_per_call = (transformer_flops_per_pass * args.steps)
        total_flops = flops_per_call * total_images
    else:
        flops_per_call = 0.0
        total_flops = 0.0
    
    print(f"FLOPs per Image (approx): {format_flops(flops_per_call)}")
    print(f"Approx total DiT FLOPs: {format_flops(total_flops)}")
    print(f"Accumulated DiT Transformer time: {TRANSFORMER_TOTAL_TIME:.3f} s")
    print(f"FID Score: {fid_score:.4f}" if fid_score is not None else "FID Score: N/A")

    # Create detailed log data (keep existing functionality)
    log_data = {
        "total_images": total_images,
        "total_pipeline_time_sec": total_time,
        "DiT_transformer_calls": TRANSFORMER_CALL_COUNT,
        "FLOPs_per_image": format_flops(flops_per_call),
        "approx_total_DiT_FLOPs": format_flops(total_flops),
        "accumulated_DiT_time_sec": TRANSFORMER_TOTAL_TIME,
        "fid_score": fid_score if fid_score is not None else "N/A",
        "args": vars(args)
    }

    log_filename = f"log_{exp_name}_{args.steps}_steps.json"
    log_filepath = os.path.join(img_dir, log_filename)
    
    # Save the log data as a JSON file (keep existing)
    with open(log_filepath, "w") as log_file:
        json.dump(log_data, log_file, indent=4)
    
    print(f"Result log saved at {log_filepath}")
    print(f"All images saved in {img_dir}")

    # NEW: Log to CSV for easy comparison across experiments
    log_to_csv(args, exp_name, img_dir, total_images, total_time, fid_score, 
               transformer_flops_per_pass, TRANSFORMER_TOTAL_TIME, TRANSFORMER_CALL_COUNT)

    # Save results in a single torch file (keep existing)
    out_dir_ckpt = f"data/{args.dataset}_ckpt"
    os.makedirs(out_dir_ckpt, exist_ok=True)
    save_name = f"images-{exp_name}_{args.steps}_steps.pt"
    print(f"Saved all generated images and labels to {os.path.join(out_dir_ckpt, save_name)}")







if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # === Dataset args ===
    # Restricted dataset choices to ImageNet variants
    parser.add_argument('--dataset', type=str, default='imagenet100',
                        choices=['imagenet', 'imagenet100', 'imagenet100_randomseed0'], help='Dataset to use')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for processing (DiT models are often 256 or 512)')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        choices=list(INTERPOLATIONS.keys()),
                        help='Interpolation method for image resizing')
    parser.add_argument('--class_csv_path', type=str, default=None, 
                        help='Path to CSV file with class IDs (e.g., imagenet100_prompts.csv)')
    parser.add_argument('--class_subset', type=str, default=None, 
                        help='Path to subset of class indices to evaluate')

    parser.add_argument('--summary_csv', type=str, default='generation_experiments_summary.csv',
                        help='Path to the CSV file for logging experiment summaries.')

    # === Model version / Pipeline args ===

    parser.add_argument('--version', type=str, default='dit',
                        help='Model version for get_dit_model. Use "dit" for standard DiT.')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes for the model (e.g., 1000 for ImageNet)')

    # === DeepCache toggles ===
    parser.add_argument('--if_deepcache', type=int, default=0,
                        help='Enable DeepCache caching functionality (set to 1 to enable)')
    parser.add_argument('--cache_interval', type=int, default=1001,
                        help='Cache interval for DeepCache')
    parser.add_argument('--cache_branch_id', type=int, default=0,
                        help='Cache branch id for DeepCache')

    # === SiTo acceleration args ===
    parser.add_argument('--if_sito', type=int, default=0,
                        help='Whether to apply SiTo token pruning (1 for True, 0 for False)')
    parser.add_argument('--sito_prune_ratio', type=float, default=0.5,
                        help='Ratio of tokens to prune in SiTo method (0.0 to 1.0)')
    parser.add_argument('--sito_max_downsample_ratio', type=int, default=1,
                        help='Maximum downsample ratio for applying SiTo')
    parser.add_argument('--sito_sx', type=int, default=2,
                        help='Stride in x dimension for SiTo patch processing')
    parser.add_argument('--sito_sy', type=int, default=2,
                        help='Stride in y dimension for SiTo patch processing')
    parser.add_argument('--sito_noise_alpha', type=float, default=0.1,
                        help='Weight for noise component in SiTo scoring')
    parser.add_argument('--sito_sim_beta', type=float, default=1.0,
                        help='Weight for similarity component in SiTo scoring')
    parser.add_argument('--sito_prune_selfattn', type=int, default=1,
                        help='Whether to apply SiTo to self-attention layers (1 for True, 0 for False)')
    parser.add_argument('--sito_prune_crossattn', type=int, default=0,
                        help='Whether to apply SiTo to cross-attention layers (1 for True, 0 for False)')
    parser.add_argument('--sito_prune_mlp', type=int, default=0,
                        help='Whether to apply SiTo to MLP layers (1 for True, 0 for False)')
    parser.add_argument('--block_sito_flags', type=str, default=None,
                        help='Comma-separated list of 0s and 1s indicating which transformer blocks should apply SiTo. '
                             'For DiT-XL/2 models, expects 28 values (one per transformer block). '
                             'E.g., "1,1,1,0,0,0,..." to apply to first 3 blocks only. '
                             'If not specified, applies to all blocks.')

    # === Acceleration Method Toggles (Token Merging, etc.) ===
    parser.add_argument('--if_token_merging', type=int, default=0)
    parser.add_argument('--if_agentsd', type=int, default=0)
    parser.add_argument('--token_merging_ratio', type=float, default=0.5)
    parser.add_argument('--agent_ratio', type=float, default=0.5)
    parser.add_argument('--token_merging_use_rand', type=int, default=1)
    parser.add_argument('--token_merging_max_downsample', type=int, default=1)
    parser.add_argument('--token_merging_sx', type=int, default=2)
    parser.add_argument('--token_merging_sy', type=int, default=2)
    parser.add_argument('--token_merging_single_downsample_level_merge', type=int, default=0)
    parser.add_argument('--merge_attn', type=int, default=1)
    parser.add_argument('--merge_crossattn', type=int, default=0)
    parser.add_argument('--merge_mlp', type=int, default=0)
    parser.add_argument('--token_merging_cache_indices_per_image', type=int, default=0,
                        help='Whether to cache merge indices per image (1 for True, 0 for False)')

    ## for Block-level Caching (DiT scoring merge acceleration)
    parser.add_argument('--cache_merge_functions', type=int, default=0,
                        help='Whether to cache merge/unmerge functions for blocks within a UNet call (1 for True, 0 for False). '
                             'Only works with scoring-based merging and block-level control. Provides speedup by reusing '
                             'expensive scoring computations across nearby blocks.')
    parser.add_argument('--cache_recalc_interval', type=int, default=4,
                        help='Recalculate cached functions every N blocks (higher = more reuse = faster but potentially lower quality). '
                             'Only used when cache_merge_functions=1. Higher values provide more speedup but may reduce quality '
                             'since cached functions become stale over more blocks.')

    ## for Token Merging Method (MLERP support)
    parser.add_argument('--token_merging_method', type=str, default='mean',
                        choices=['mean', 'mlerp', 'prune'],
                        help='Token merging method to use. Options: "mean" (standard average merging), '
                             '"mlerp" (MLERP - Maximum-Norm Linear Interpolation for better feature preservation), '
                             '"prune" (remove selected source tokens completely - most aggressive reduction). '
                             'MLERP preserves feature magnitudes and typically provides better accuracy at high reduction ratios.')

    ## for Proportional Attention
    parser.add_argument('--if_proportional_attention', type=int, default=0,
                        help='Whether to use proportional attention that accounts for token sizes (1 for True, 0 for False)')

    ## for ABP (Adaptive Block Pooling) 
    parser.add_argument('--merge_method_alg', type=str, default='bipartite',
                        choices=['bipartite', 'abp'],
                        help='Merging algorithm to use. Options: "bipartite" (standard bipartite matching), '
                             '"abp" (Adaptive Block Pooling - tile-based merging for better spatial structure). '
                             'ABP uses tile-based approach and automatically selects optimal algorithm based on data size.')
    parser.add_argument('--abp_scorer', type=str, default=None,
                        help='TokenScorer class name for ABP tile evaluation. If None with ABP, uses SpatialFilterScorer. '
                             'Only used when merge_method_alg="abp". Should be a valid scorer class.')
    parser.add_argument('--abp_tile_aggregation', type=str, default='max',
                        choices=['max', 'min', 'sum', 'std'],
                        help='How to aggregate token scores within tiles for ABP evaluation. '
                             'Options: "max" (maximum score), "min" (minimum score), "sum" (sum of scores), "std" (standard deviation). '
                             'Only used when merge_method_alg="abp".')

    # === Scoring-based Token Merging arguments ===
    parser.add_argument('--if_scoring_merge', type=int, default=0,
                       help='Whether to apply scoring-based token merging (1 for True, 0 for False)')
    parser.add_argument('--scoring_method', type=str, default='statistical',
                       choices=['frequency', 'spatial_filter', 'statistical', 'signal_processing', 'spatial_distribution'],
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

    # === Attention processor (downsampling) ===
    parser.add_argument('--if_attention_proc', type=int, default=0)
    parser.add_argument('--merge_tokens', type=str, default='keys/values', choices=['keys/values', 'all'])
    parser.add_argument('--merge_method', type=str, default='downsample', choices=['none', 'similarity', 'downsample','downsample_custom_block', 'frequency_blockwise','frequency_global'])
    parser.add_argument('--downsample_method', type=str, default='nearest-exact',
                        help='Interpolation method for downsampling. Options: nearest-exact, max_pool, avg_pool, area, bilinear, bicubic, top_right, bottom_left, bottom_right, random, uniform_random, uniform_timestep, linear_blend')
    parser.add_argument('--downsample_factor', type=int, default=2)
    parser.add_argument('--downsample_factor_h', type=int, default=1,
                        help='Downsample factor for height dimension, only used if merge_method is downsample_custom_block')
    parser.add_argument('--downsample_factor_w', type=int, default=1,
                        help='Downsample factor for width dimension, only used if merge_method is downsample_custom_block')

    # Linear blend parameters for downsample methods
    parser.add_argument('--blend_factor', type=float, default=None,
                        help='Blend factor for linear_blend downsample method (0.0=method2, 1.0=method1). None=use 0.5 default')
    parser.add_argument('--blend_method_1', type=str, default=None,
                        help='First method for linear_blend (default: nearest-exact)')
    parser.add_argument('--blend_method_2', type=str, default=None,
                        help='Second method for linear_blend (default: avg_pool)')

    parser.add_argument('--timestep_threshold_switch', type=float, default=0)
    parser.add_argument('--timestep_threshold_stop', type=float, default=0.0)
    parser.add_argument('--secondary_merge_method', type=str, default='similarity', choices=['none', 'similarity', 'downsample', 'frequency_blockwise','frequency_global'])
    parser.add_argument('--downsample_factor_level_2', type=int, default=1)
    parser.add_argument('--ratio_level_2', type=float, default=0.0)
    parser.add_argument('--extra_guidance_scale', type=float, default=0)
    
    # --- Frequency-specific args ---
    parser.add_argument('--frequency_selection_mode', type=str, default='high')
    parser.add_argument('--frequency_selection_method', type=str, default='1d_dft', choices=['original','1d_dft', '1d_dct', '2d_conv','non_uniform_grid','2d_conv_l2'])
    parser.add_argument('--frequency_ranking_method', type=str, default='amplitude',choices=['amplitude', 'spectral_centroid','variance', 'l1norm', 'l2norm', 'mean_deviation'],
                        help='Method for ranking frequencies (e.g., by amplitude). Only used if merge_method is "frequency".')
    parser.add_argument('--selection_source', type=str, default='hidden', choices=['hidden', 'key', 'query', 'value'])
    parser.add_argument('--frequency_grid_alpha', type=float, default=2)

    # --- Block-wise ToMe control ---
    parser.add_argument('--block_tome_flags', type=str, default=None,
                        help='Comma-separated list of 0s and 1s indicating which transformer blocks should apply ToMe. '
                             'For DiT-XL/2 models, expects 28 values (one per transformer block). '
                             'E.g., "1,1,1,0,0,0,..." to apply to first 3 blocks only. '
                             'If not specified, applies to all blocks.')

    # === Sampling setup ===
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--fid_batch_size", type=int, default=1)
    parser.add_argument('--gt_path', type=str, 
                        default="/path/to/datasets/imagenet-100/val_flat",
                        help="Path to ground truth images for FID calculation.")
    parser.add_argument('--guidance_scale', type=float, default=4.0,
                    help='Guidance scale for classifier-free guidance')

    
    # === Image quality ===
    parser.add_argument("--quality", type=int, default=95,
                     help="JPEG quality (0-100)")

    # === Random seed ===
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for reproducibility")

    # Attention processor for fair benchmarking
    parser.add_argument('--force_attention_processor', type=str, default='AttnProcessor',
                        choices=['AttnProcessor', 'AttnProcessor2_0', 'XFormersAttnProcessor', 'none'])

    args = parser.parse_args()
    
    # Parse block tome flags if provided (similar to eval_prob_adaptive_dit.py)
    if args.block_tome_flags is not None:
        try:
            block_tome_flags = [int(x.strip()) for x in args.block_tome_flags.split(',')]
            
            # Validate that values are 0 or 1
            if not all(flag in [0, 1] for flag in block_tome_flags):
                raise ValueError("All values must be 0 or 1")
            
            # For DiT models, warn if the number doesn't match expected
            expected_blocks = 28  # DiT-XL/2 has 28 transformer blocks
            if len(block_tome_flags) != expected_blocks:
                print(f"Warning: DiT-XL/2 typically has {expected_blocks} blocks, but got {len(block_tome_flags)} flags. "
                      f"This may still work if the model architecture differs.")
            
            print(f"Using custom block ToMe flags: {block_tome_flags}")
            print(f"Will apply ToMe to {sum(block_tome_flags)}/{len(block_tome_flags)} transformer blocks")
            
        except ValueError as e:
            raise ValueError(f"Invalid block_tome_flags format: {args.block_tome_flags}. "
                           f"Expected comma-separated 0s and 1s. Error: {e}")
    else:
        block_tome_flags = None
        print("Using default block ToMe flags (all blocks enabled)")
    
    # Store the parsed flags back in args for use in get_dit_model
    args.block_tome_flags_parsed = block_tome_flags

    # Parse block sito flags if provided
    if args.block_sito_flags is not None:
        try:
            block_sito_flags = [int(x.strip()) for x in args.block_sito_flags.split(',')]
            
            # Validate that values are 0 or 1
            if not all(flag in [0, 1] for flag in block_sito_flags):
                raise ValueError("All values must be 0 or 1")
            
            # For DiT models, warn if the number doesn't match expected
            expected_blocks = 28  # DiT-XL/2 has 28 transformer blocks
            if len(block_sito_flags) != expected_blocks:
                print(f"Warning: DiT-XL/2 typically has {expected_blocks} blocks, but got {len(block_sito_flags)} flags. "
                      f"This may still work if the model architecture differs.")
            
            print(f"Using custom block SiTo flags: {block_sito_flags}")
            print(f"Will apply SiTo to {sum(block_sito_flags)}/{len(block_sito_flags)} transformer blocks")
            
        except ValueError as e:
            raise ValueError(f"Invalid block_sito_flags format: {args.block_sito_flags}. "
                           f"Expected comma-separated 0s and 1s. Error: {e}")
    else:
        block_sito_flags = None
        print("Using default block SiTo flags (all blocks enabled)")
    
    # Store the parsed flags back in args for use in get_dit_model
    args.block_sito_flags_parsed = block_sito_flags
    
    # Parse ABP scorer if provided
    if getattr(args, 'abp_scorer', None) is not None:
        # Convert string to actual scorer object
        try:
            import sys
            import os
            tomesd_path = os.path.join(os.path.dirname(__file__), 'tomesd')
            if tomesd_path not in sys.path:
                sys.path.insert(0, tomesd_path)
            from tomesd.tomesd.scoring import (
                FrequencyScorer, SpatialFilterScorer, StatisticalScorer,
                SignalProcessingScorer, SpatialDistributionScorer
            )
            
            scorer_name = args.abp_scorer.lower()
            if scorer_name == 'frequencyscorer':
                args.abp_scorer = FrequencyScorer()
            elif scorer_name == 'spatialfilterscorer':
                args.abp_scorer = SpatialFilterScorer()
            elif scorer_name == 'statisticalscorer':
                args.abp_scorer = StatisticalScorer()
            elif scorer_name == 'signalprocessingscorer':
                args.abp_scorer = SignalProcessingScorer()
            elif scorer_name == 'spatialdistributionscorer':
                args.abp_scorer = SpatialDistributionScorer()
            else:
                print(f"Warning: Unknown ABP scorer '{args.abp_scorer}', will use default SpatialFilterScorer")
                args.abp_scorer = None
        except Exception as e:
            print(f"Warning: Failed to instantiate ABP scorer '{args.abp_scorer}': {e}, using default")
            args.abp_scorer = None
    
    main(args)