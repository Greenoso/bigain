# Standard Library
import argparse
import os
import os.path as osp
import time
import random
import json
from datetime import datetime
import csv

# COCO2017 mAP Reload Usage:
# To reload and recalculate COCO2017 precision and mAP stats from existing saved files:
# python eval_prob_adaptive.py --dataset=coco2017 --prompt_path=prompts.csv --load_stats [other args]
# This will skip re-evaluation and instead recalculate mAP metrics from saved error data.

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import tqdm
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.flop_counter import FlopCounterMode
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0, XFormersAttnProcessor
import gc 


# Local
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr



# to record the time and computational resources for the unet
UNET_TOTAL_TIME = 0.0   # cumulative time


device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


# --- Helper function to format FLOPs (from example code) ---
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


def seed_everything(seed: int):
    """Seed everything for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def calculate_average_precision(ranked_labels, true_labels):
    """
    Calculate Average Precision (AP) for a single image in multi-label classification.
    
    Args:
        ranked_labels: List of class indices ranked by confidence (best to worst)
        true_labels: List of ground truth class indices for this image
        
    Returns:
        Average Precision (AP) value between 0 and 1
    """
    num_ground_truth = len(true_labels)
    if num_ground_truth == 0:
        return 0.0
        
    ap = 0.0
    num_relevant_found = 0
    
    for k, predicted_label in enumerate(ranked_labels, start=1):
        if predicted_label in true_labels:
            num_relevant_found += 1
            precision_at_k = num_relevant_found / k
            ap += precision_at_k
            
    return ap / num_ground_truth if num_ground_truth > 0 else 0.0


def rank_classes_by_error(pred_errors, prompts_df):
    """
    Rank all classes by their prediction errors (ascending order - best first).
    Note: This is a simplified ranking that doesn't replicate the adaptive algorithm.
    
    Args:
        pred_errors: Dictionary containing prediction errors for each class
        prompts_df: DataFrame containing prompt information
        
    Returns:
        List of class indices ranked by error (lowest error first)
    """
    class_error_pairs = []
    
    for class_idx in range(len(prompts_df)):
        if class_idx in pred_errors:
            mean_error = pred_errors[class_idx]['pred_errors'].mean().item()
            # Handle potential NaN/Inf values
            if not np.isfinite(mean_error):
                mean_error = float('inf')
        else:
            mean_error = float('inf')
            
        class_label = prompts_df.classidx[class_idx]
        class_error_pairs.append((class_label, mean_error))
    
    # Sort by error (ascending) and return just the class labels
    ranked_classes = [class_label for class_label, _ in sorted(class_error_pairs, key=lambda x: x[1])]
    return ranked_classes


def simulate_adaptive_ranking(pred_errors, prompts_df, args):
    """
    Simulate the adaptive ranking algorithm used in eval_prob_adaptive.
    This provides more accurate mAP calculation that matches the actual algorithm.
    
    Args:
        pred_errors: Dictionary containing prediction errors for each class
        prompts_df: DataFrame containing prompt information  
        args: Arguments containing n_samples and to_keep parameters
        
    Returns:
        List of class indices ranked by the adaptive algorithm (best first)
    """
    if args.single_timestep >= 0:
        # Single timestep mode: just rank by error
        return rank_classes_by_error(pred_errors, prompts_df)
    
    # Multi-stage adaptive mode - replicate the exact logic from eval_prob_adaptive
    try:
        remaining_prompt_idxs = list(range(len(prompts_df)))
        
        # Properly simulate timestep-based evaluation stages
        # The original algorithm evaluates prompts on different numbers of timesteps per stage
        # We need to slice the errors correctly based on cumulative timestep evaluation
        
        cumulative_timesteps_evaluated = 0
        
        for stage_idx, (n_samples, n_to_keep) in enumerate(zip(args.n_samples, args.to_keep)):
            if len(remaining_prompt_idxs) <= 1:
                break
                
            # Calculate how many timesteps have been evaluated up to this stage
            cumulative_timesteps_evaluated += n_samples
            
            # Use only the errors from timesteps evaluated up to this stage
            errors_for_topk = []
            for prompt_idx in remaining_prompt_idxs:
                if prompt_idx in pred_errors and 'pred_errors' in pred_errors[prompt_idx]:
                    all_errors = pred_errors[prompt_idx]['pred_errors']
                    
                    # Use only the errors from timesteps evaluated in this cumulative process
                    # The errors are stored in the order they were evaluated
                    if len(all_errors) >= cumulative_timesteps_evaluated:
                        # Use errors from all timesteps evaluated so far (cumulative)
                        relevant_errors = all_errors[:cumulative_timesteps_evaluated]
                    else:
                        # Use all available errors if we don't have enough
                        relevant_errors = all_errors
                    
                    if len(relevant_errors) > 0:
                        mean_error = relevant_errors.mean().item()
                    else:
                        mean_error = float('inf')
                else:
                    mean_error = float('inf')
                    
                if not np.isfinite(mean_error):
                    mean_error = float('inf')
                    
                # Use negative errors like the original algorithm (line 333)
                errors_for_topk.append(-mean_error)
            
            # Use torch.topk like the original algorithm (lines 334-335)
            if len(errors_for_topk) > 0:
                errors_tensor = torch.tensor(errors_for_topk)
                best_idxs = torch.topk(errors_tensor, k=min(n_to_keep, len(errors_for_topk)), dim=0).indices.tolist()
                remaining_prompt_idxs = [remaining_prompt_idxs[i] for i in best_idxs]
            else:
                break
        
        # Convert final prompt indices to class indices and build full ranking
        final_ranking = []
        
        # Add the winner (best remaining prompt) 
        # Properly select the best prompt from remaining_prompt_idxs
        if remaining_prompt_idxs:
            # If there's only one left (which should be the case), that's the winner
            if len(remaining_prompt_idxs) == 1:
                winner_prompt_idx = remaining_prompt_idxs[0]
            else:
                # If multiple remain, select the one with the lowest error
                best_error = float('inf')
                winner_prompt_idx = remaining_prompt_idxs[0]  # fallback
                for prompt_idx in remaining_prompt_idxs:
                    if prompt_idx in pred_errors and 'pred_errors' in pred_errors[prompt_idx]:
                        mean_error = pred_errors[prompt_idx]['pred_errors'].mean().item()
                        if mean_error < best_error:
                            best_error = mean_error
                            winner_prompt_idx = prompt_idx
            
            winner_class = prompts_df.classidx[winner_prompt_idx]
            final_ranking.append(winner_class)
        
        # Add remaining classes by simple error ranking for approximation
        all_classes = rank_classes_by_error(pred_errors, prompts_df)
        for class_idx in all_classes:
            if class_idx not in final_ranking:
                final_ranking.append(class_idx)
                
        return final_ranking
        
    except Exception as e:
        print(f"[WARNING] Adaptive ranking simulation failed: {e}. Falling back to simple ranking.")
        return rank_classes_by_error(pred_errors, prompts_df)


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


def eval_prob_adaptive(unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    num_prompts = len(text_embeds) # Get total number of prompts
    

    if all_noise is None:
        # Determine required noise size based on mode
        # --- Adjust required noise size check ---
        if args.single_timestep >= 0:
            # Need noise for all prompts * n_trials at the single timestep
            required_noise_samples = args.n_trials
        else:
            # Original adaptive mode: need noise for max_n_samples * n_trials
            max_n_samples = max(args.n_samples)
            required_noise_samples = max_n_samples * args.n_trials

        all_noise = torch.randn((required_noise_samples, 4, latent_size, latent_size), device=latent.device)
 

        
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        #scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()



    # --- Single Timestep Evaluation Path ---
    if args.single_timestep >= 0:
        target_t = args.single_timestep
        if not (0 <= target_t < T):
             raise ValueError(f"--single_timestep must be between 0 and {T-1}, but got {target_t}")

        # Prepare inputs for eval_error for the single timestep across all prompts
        ts = [target_t] * (num_prompts * args.n_trials)

        text_embed_idxs = [p_idx for p_idx in range(num_prompts) for _ in range(args.n_trials)]
        noise_idxs = [trial_idx for _ in range(num_prompts) for trial_idx in range(args.n_trials)]

        # Check if generated/loaded noise is sufficient

        if len(all_noise) < max(noise_idxs) + 1:
            raise ValueError(
            f"Need at least {max(noise_idxs)+1} noise samples, found {len(all_noise)}."
            )
        # Call eval_error once for all prompts at the target timestep
        
        pred_errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss,
                                args.eval_denoise_steps,
                                args.eval_step_stride,
                                args.eval_error_method)

        # Calculate the mean error for each prompt at this timestep
        mean_errors_per_prompt = []
        data = {} # Store results similarly to the original format
        for p_idx in range(num_prompts):
            start_idx = p_idx * args.n_trials
            end_idx = start_idx + args.n_trials
            prompt_errors = pred_errors[start_idx:end_idx]
            mean_errors_per_prompt.append(prompt_errors.mean().item())
            # Store data for this prompt (only one timestep evaluated)
            data[p_idx] = {'t': torch.tensor([target_t] * args.n_trials, device='cpu'),
                           'pred_errors': prompt_errors.cpu()} # Store errors on CPU

        # Find the prompt index with the minimum mean error
        pred_idx = np.argmin(mean_errors_per_prompt)

        return pred_idx, data
    # --- End Single Timestep Path ---

    else:
        data = dict()
        t_evaluated = set()
        remaining_prmpt_idxs = list(range(len(text_embeds)))
        start = T // max(args.n_samples) // 2
        t_to_eval = list(range(start, T, T // max(args.n_samples)))[:max(args.n_samples)]

        for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
            ts = []
            noise_idxs = []
            text_embed_idxs = []
            curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
            curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
            for prompt_i in remaining_prmpt_idxs:
                for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                    ts.extend([t] * args.n_trials)
                    if args.fix_noise_across_timestep:
                        # Use the same noise indices for each prompt across all timesteps
                        noise_idxs.extend(list(range(args.n_trials)))
                    else:
                        # Original behavior: different noise indices for different timesteps
                        noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                    text_embed_idxs.extend([prompt_i] * args.n_trials)
            t_evaluated.update(curr_t_to_eval)

            pred_errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                    text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss,
                                    args.eval_denoise_steps,
                                    args.eval_step_stride,
                                    args.eval_error_method)

            # match up computed errors to the data
            for prompt_i in remaining_prmpt_idxs:
                mask = torch.tensor(text_embed_idxs) == prompt_i
                prompt_ts = torch.tensor(ts)[mask]
                prompt_pred_errors = pred_errors[mask]
                if prompt_i not in data:
                    data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
                else:
                    data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                    data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

            # compute the next remaining idxs
            errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
            best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
            remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

        # organize the output
        assert len(remaining_prmpt_idxs) == 1
        pred_idx = remaining_prmpt_idxs[0]

        return pred_idx, data


def eval_error(
        unet, scheduler, latent, all_noise, ts, noise_idxs,
        text_embeds, text_embed_idxs,
        batch_size: int = 32,
        dtype: str = "float32",
        loss: str = "l2",
        num_denoise_steps: int = 1,          
        step_stride: int = 1,
        eval_error_method: str = "direct"
):
    """
    * num_denoise_steps == 1  ➜ original behaviour (ε̂ vs. ε)
    * num_denoise_steps  > 1  ➜ run k steps, then:
        - eval_error_method == "trajectory": compare x_k vs. ground truth at same timestep
        - eval_error_method == "direct": compare x_k vs. clean x₀
        - eval_error_method == "weighted": like direct but weighted by 1/(1-ᾱ_t) for fair comparison
    """
    import time
    global UNET_TOTAL_TIME

    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    pred_errors = torch.empty(len(ts), device="cpu")
    alphas_cumprod = scheduler.alphas_cumprod.to(device)           
    if dtype == "float16":                                         
        alphas_cumprod = alphas_cumprod.half()                     

    x0_original = latent.to(device, dtype=torch.float16 if dtype == "float16" else torch.float32)

    # Build timetable only if we need it for k-step denoising
    if num_denoise_steps > 1:
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        timesteps_tensor = scheduler.timesteps.to(device)                 # [T]
        timestep_to_idx = {int(t.item()): i for i, t in enumerate(timesteps_tensor)}

    idx = 0
    with torch.inference_mode():
        num_iters = len(ts) // batch_size + int(len(ts) % batch_size != 0)
        for _ in tqdm.trange(num_iters, leave=False):
            this_batch_size = min(batch_size, len(ts) - idx)

            # ───────────────────── gather batch data ─────────────────────
            batch_ts      = torch.tensor(ts[idx: idx+this_batch_size], device=device)
            noise         = all_noise[noise_idxs[idx: idx+this_batch_size]].to(device)
            text_input    = text_embeds[text_embed_idxs[idx: idx+this_batch_size]].to(device)

            if dtype == "float16":
                noise = noise.half()




            sqrt_alpha_prod      = alphas_cumprod[batch_ts].sqrt().view(-1,1,1,1)
            sqrt_one_minus_alpha = (1. - alphas_cumprod[batch_ts]).sqrt().view(-1,1,1,1)
            noised_latent        = x0_original*sqrt_alpha_prod + noise*sqrt_one_minus_alpha
            noised_latent        = noised_latent.half() if dtype == "float16" else noised_latent
            
            # ──────────────────────── k = 1 path ─────────────────────────
            if num_denoise_steps == 1:
                t_input = batch_ts.half() if dtype == "float16" else batch_ts

                # For SNR/noise methods: store current noise and clean image for on-the-fly mask computation
                if hasattr(unet, '_tome_info') and unet._tome_info is not None:
                    if hasattr(unet, '_tome_info') and 'current_clean_image' in unet._tome_info:
                        # Store current noise for this batch - reshape to match attention format
                        # noise shape: (batch_size, 4, 64, 64) -> (batch_size, 4096, 4)
                        B_noise, C_noise, H_noise, W_noise = noise.shape
                        noise_reshaped = noise.permute(0, 2, 3, 1).reshape(B_noise, H_noise * W_noise, C_noise)
                        unet._tome_info['current_noise'] = noise_reshaped
                        unet._tome_info['current_timesteps'] = batch_ts

                torch.cuda.synchronize(); start_time = time.time()
                noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
                torch.cuda.synchronize(); UNET_TOTAL_TIME += time.time() - start_time

                # Clean up stored noise info to avoid memory accumulation
                if hasattr(unet, '_tome_info') and unet._tome_info is not None:
                    if 'current_noise' in unet._tome_info:
                        del unet._tome_info['current_noise']
                    if 'current_timesteps' in unet._tome_info:
                        del unet._tome_info['current_timesteps']

                if noise_pred.dtype != noise.dtype:
                    noise_pred = noise_pred.to(noise.dtype)

                if   loss == "l2":
                    error = F.mse_loss(noise_pred, noise, reduction='none').mean((1,2,3))
                elif loss == "l1":
                    error = F.l1_loss (noise_pred, noise, reduction='none').mean((1,2,3))
                elif loss == "huber":
                    error = F.huber_loss(noise_pred, noise,reduction='none').mean((1,2,3))
                else:
                    raise NotImplementedError

            # ──────────────────────── k > 1 path ─────────────────────────
            else:
                # --- Setup for k-step evaluation ---
                current_latent = noised_latent
                current_ts = batch_ts.clone()

                # Setup scheduler timesteps
                scheduler.set_timesteps(scheduler.config.num_train_timesteps)
                timesteps_tensor = scheduler.timesteps.to(device)
                
                # Get alphas_cumprod for manual step calculation
                alphas_cumprod = scheduler.alphas_cumprod.to(device)
                if dtype == "float16":
                    alphas_cumprod = alphas_cumprod.half()

                # Create mapping from timestep VALUE to its INDEX
                timestep_to_idx = {int(t.item()): i for i, t in enumerate(timesteps_tensor)}
                
                # Get starting indices
                try:
                    current_indices = torch.tensor(
                        [timestep_to_idx[int(t.item())] for t in current_ts], 
                        device=device, dtype=torch.long
                    )
                except KeyError as e:
                    raise KeyError(f"Starting timestep {e} not found in scheduler's timesteps.") from e

                # --- K-step denoising loop ---
                for step in range(num_denoise_steps):
                    # Check boundaries
                    if torch.any(current_indices >= len(timesteps_tensor) - 1):
                        active_mask = current_indices < len(timesteps_tensor) - 1
                        if not torch.any(active_mask):
                            break
                    else:
                        active_mask = torch.ones_like(current_indices, dtype=torch.bool)

                    # Scale model input
                    scaled_latent = torch.empty_like(current_latent)
                    for t_val in torch.unique(current_ts):
                        mask = (current_ts == t_val)
                        scaled_latent[mask] = scheduler.scale_model_input(
                            current_latent[mask], t_val.item()
                        )
                    
                    t_input = current_ts.half() if dtype == "float16" else current_ts

                    # For SNR/noise methods: store current noise and clean image for on-the-fly mask computation
                    if hasattr(unet, '_tome_info') and unet._tome_info is not None:
                        if hasattr(unet, '_tome_info') and 'current_clean_image' in unet._tome_info:
                            # Store current noise for this batch - reshape to match attention format
                            # noise shape: (batch_size, 4, 64, 64) -> (batch_size, 4096, 4)
                            B_noise, C_noise, H_noise, W_noise = noise.shape
                            noise_reshaped = noise.permute(0, 2, 3, 1).reshape(B_noise, H_noise * W_noise, C_noise)
                            unet._tome_info['current_noise'] = noise_reshaped
                            unet._tome_info['current_timesteps'] = current_ts

                    # Predict noise
                    torch.cuda.synchronize(); start_time = time.time()
                    noise_pred = unet(scaled_latent, t_input, 
                                    encoder_hidden_states=text_input).sample
                    torch.cuda.synchronize(); UNET_TOTAL_TIME += time.time() - start_time

                    # Clean up stored noise info to avoid memory accumulation
                    if hasattr(unet, '_tome_info') and unet._tome_info is not None:
                        if 'current_noise' in unet._tome_info:
                            del unet._tome_info['current_noise']
                        if 'current_timesteps' in unet._tome_info:
                            del unet._tome_info['current_timesteps']

                    # Use scheduler.step() properly by processing each unique timestep separately
                    new_latent = torch.empty_like(current_latent)
                    new_indices = current_indices.clone()
                    
                    for t_val in torch.unique(current_ts):
                        mask = (current_ts == t_val)
                        if torch.any(mask):
                            # Get the corresponding noise predictions and latents
                            masked_noise_pred = noise_pred[mask] 
                            masked_latent = current_latent[mask]
                            
                            # Use manual calculation instead of scheduler.step() to avoid step_index issues
                            timestep_val = int(t_val.item())
                            
                            # Manual DDIM/Euler step calculation
                            alpha_prod_t = alphas_cumprod[t_val.long()].view(1, 1, 1, 1)
                            
                            # Predict x0 from noise prediction
                            pred_x0 = (masked_latent - (1 - alpha_prod_t).sqrt() * masked_noise_pred) / alpha_prod_t.sqrt()
                            
                            # Get next timestep
                            current_idx = timestep_to_idx[timestep_val]
                            next_idx = min(current_idx + step_stride, len(timesteps_tensor) - 1)
                            next_t = timesteps_tensor[next_idx]
                            alpha_prod_t_next = alphas_cumprod[next_t.long()].view(1, 1, 1, 1)
                            
                            # Calculate next sample using DDIM formula
                            new_latent[mask] = alpha_prod_t_next.sqrt() * pred_x0 + (1 - alpha_prod_t_next).sqrt() * masked_noise_pred
                            
                            # Update indices for next timestep
                            current_idx = timestep_to_idx[timestep_val]
                            next_idx = min(current_idx + step_stride, len(timesteps_tensor) - 1)
                            new_indices[mask] = next_idx

                    # Update state for next iteration
                    current_latent = new_latent
                    current_indices = new_indices
                    current_ts = timesteps_tensor[current_indices]
                    

                # --- End K-step loop ---
                
                # --- Error Calculation Based on Method ---
                if eval_error_method == "trajectory":
                    # Method 1: Compare final latent to ground truth at the same timestep
                    # This ensures we are comparing latents at the same noise level, providing a more stable error metric.
                    
                    # Get the cumulative alpha for the final timesteps
                    final_ts_alphas = alphas_cumprod[current_ts.long()].view(-1, 1, 1, 1)

                    # Construct the ground truth latent at the final timestep `t'` using the original x0 and the same initial noise `ε`.
                    sqrt_alpha_final = final_ts_alphas.sqrt()
                    sqrt_one_minus_alpha_final = (1. - final_ts_alphas).sqrt()
                    ground_truth_final_latent = (x0_original * sqrt_alpha_final + noise * sqrt_one_minus_alpha_final).to(current_latent.dtype)
                    
                    # The model's prediction is the `current_latent` after the k-step loop.
                    final_latent_output = current_latent
                    
                    # Calculate the trajectory error.
                    if loss == "l2":
                        error = F.mse_loss(final_latent_output, ground_truth_final_latent, reduction='none').mean((1,2,3))
                    elif loss == "l1":
                        error = F.l1_loss(final_latent_output, ground_truth_final_latent, reduction='none').mean((1,2,3))
                    elif loss == "huber":
                        error = F.huber_loss(final_latent_output, ground_truth_final_latent, reduction='none').mean((1,2,3))
                    else:
                        raise NotImplementedError
                            
                elif eval_error_method == "direct":
                    # Method 2: Compare final latent directly to clean x0
                    # Traditional approach comparing denoised result to target
                    
                    # Expand x0_original to match batch size for proper comparison
                    x0_expanded = x0_original.expand_as(current_latent)
                    final_latent = current_latent.to(x0_expanded.dtype)
                    
                    # Calculate error using expanded x0
                    if loss == "l2":
                        error = F.mse_loss(final_latent, x0_expanded, reduction='none').mean((1,2,3))
                    elif loss == "l1":
                        error = F.l1_loss(final_latent, x0_expanded, reduction='none').mean((1,2,3))
                    elif loss == "huber":
                        error = F.huber_loss(final_latent, x0_expanded, reduction='none').mean((1,2,3))
                    else:
                        raise NotImplementedError
                        
                elif eval_error_method == "weighted":
                    # Method 3: Compare final latent to clean x0 with noise-level weighting
                    # Addresses the issue that samples at different final timesteps carry different noise levels
                    
                    # Expand x0_original to match batch size for proper comparison
                    x0_expanded = x0_original.expand_as(current_latent)
                    final_latent = current_latent.to(x0_expanded.dtype)
                    
                    # Calculate noise variance at final timesteps: σ² = 1 - ᾱ_t
                    final_ts_alphas = alphas_cumprod[current_ts.long()]  # ᾱ_t for final timesteps
                    sigma_squared = torch.clamp(1.0 - final_ts_alphas, min=1e-8)  # (B,)
                    
                    # Calculate weights based on loss type
                    if loss == "l2":
                        # For L2 loss: weight = 1/(1-ᾱ_t) to normalize expected MSE
                        weights = 1.0 / sigma_squared
                    elif loss == "l1":
                        # For L1 loss: weight = 1/√(1-ᾱ_t) to normalize expected MAE  
                        weights = 1.0 / torch.sqrt(sigma_squared)
                    elif loss == "huber":
                        # For Huber loss: use L1 weighting as safe default
                        weights = 1.0 / torch.sqrt(sigma_squared)
                    else:
                        raise NotImplementedError
                    
                    # Calculate raw error first
                    error_raw = final_latent - x0_expanded
                    bsz = error_raw.size(0)
                    
                    # Apply loss function with weighting
                    if loss == "l2":
                        # Apply weighting to MSE: weight * ||x||²
                        error = (weights.view(-1, 1) * error_raw.pow(2).view(bsz, -1)).mean(1)
                    elif loss == "l1":
                        # Apply weighting to MAE: weight * ||x||₁
                        error = (weights.view(-1, 1) * error_raw.abs().view(bsz, -1)).mean(1)
                    elif loss == "huber":
                        # Apply weighting to Huber loss
                        huber_raw = F.huber_loss(final_latent, x0_expanded, reduction='none').view(bsz, -1)
                        error = (weights.view(-1, 1) * huber_raw).mean(1)
                    else:
                        raise NotImplementedError
                        
                elif eval_error_method == "clean_signal":
                    # Method 4: Compare final latent to clean signal component (x0_original * sqrt_alpha_final)
                    # This compares the denoised result to just the clean signal component without noise
                    
                    # Get the cumulative alpha for the final timesteps
                    final_ts_alphas = alphas_cumprod[current_ts.long()].view(-1, 1, 1, 1)
                    
                    # Calculate the clean signal component at the final timestep
                    sqrt_alpha_final = final_ts_alphas.sqrt()
                    clean_signal_component = (x0_original * sqrt_alpha_final).to(current_latent.dtype)
                    
                    # The model's prediction is the current_latent after the k-step loop
                    final_latent_output = current_latent
                    
                    # Calculate the clean signal error
                    if loss == "l2":
                        error = F.mse_loss(final_latent_output, clean_signal_component, reduction='none').mean((1,2,3))
                    elif loss == "l1":
                        error = F.l1_loss(final_latent_output, clean_signal_component, reduction='none').mean((1,2,3))
                    elif loss == "huber":
                        error = F.huber_loss(final_latent_output, clean_signal_component, reduction='none').mean((1,2,3))
                    else:
                        raise NotImplementedError
                else:
                    raise ValueError(f"Unknown eval_error_method: {eval_error_method}")

            # ─────────────────────── store & advance ─────────────────────
            pred_errors[idx: idx+this_batch_size] = error.detach().cpu()
            idx += this_batch_size
        print(f"[DEBUG] Error range for this batch: min={pred_errors.min().item():.10f}, max={pred_errors.max().item():.10f}, mean={pred_errors.mean().item():.6f},std={pred_errors.var().item():.6f}")

    return pred_errors



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


def main():
    global UNET_TOTAL_TIME
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='pets',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft','coco2017','imagenet100','imagenet100_randomseed0'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='2-0', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(64,128,256, 512,1024, 2048, 4096), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to csv file with prompts to use')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l2', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)
    parser.add_argument('--if_token_merging', type=int,default=0, help='Whether to apply token merging')
    parser.add_argument('--if_agentsd',type=int,default=0,help='Whether or not to apply agent attention token merging (1 for True, 0 for False)')
    parser.add_argument('--token_merging_ratio', type=float, default=0.5, help='Ratio of tokens to merge')
    parser.add_argument('--agent_ratio', type=float, default=0.5, help='Ratio of agent tokens for agentsd')
    parser.add_argument('--token_merging_use_rand', type=int, default=1, help='if use random for source token merging')

    parser.add_argument('--token_merging_max_downsample',type=int,default=1,choices=[1, 2, 4, 8],
                        help='Apply ToMe to layers with at most this amount of downsampling (e.g., 1 applies only to layers with no downsampling)')

    parser.add_argument('--token_merging_sx',type=int,default=2,help='Stride in the x dimension for computing dst sets (see paper)')

    parser.add_argument('--token_merging_sy',type=int,default=2,help='Stride in the y dimension for computing dst sets (see paper)')

    parser.add_argument('--token_merging_single_downsample_level_merge',type=int,default=0,help='Whether only merge the layer with downsample size equal token_merging_max_downsample')

    parser.add_argument('--token_merging_cache_indices_per_image',type=int,default=0,help='Whether to cache merge indices per image (1 for True, 0 for False)')

    parser.add_argument('--merge_attn',type=int,default=1,help='Whether or not to merge tokens for attention (1 for True, 0 for False)')

    parser.add_argument('--merge_crossattn',type=int,default=0,help='Whether or not to merge tokens for cross attention (1 for True, 0 for False)')

    parser.add_argument('--merge_mlp',type=int,default=0,help='Whether or not to merge tokens for MLP layers (1 for True, 0 for False)')

    ## for Token Merging Method (MLERP support)
    # Usage example: --if_token_merging=1 --token_merging_ratio=0.5 --token_merging_method=mlerp
    # MLERP typically recovers 1-2pp accuracy at large reduction ratios compared to standard mean merging
    parser.add_argument('--token_merging_method', type=str, default='mean',
                        choices=['mean', 'mlerp', 'prune'],
                        help='Token merging method to use. Options: "mean" (standard average merging), '
                             '"mlerp" (MLERP - Maximum-Norm Linear Interpolation for better feature preservation), '
                             '"prune" (remove selected source tokens completely - most aggressive reduction). '
                             'MLERP preserves feature magnitudes and typically provides better accuracy at high reduction ratios. '
                             'Prune provides maximum speedup but may reduce accuracy more than merging methods.')

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

    ## for Scoring-based Token Merging
    parser.add_argument('--if_scoring_merge', type=int, default=0,
                        help='Whether to apply scoring-based token merging (1 for True, 0 for False)')
    parser.add_argument('--scoring_method', type=str, default='statistical',
                        choices=['frequency', 'spatial_filter', 'statistical', 'signal_processing', 'spatial_distribution', 'similarity'],
                        help='Type of scorer to use for scoring-based merging')
    parser.add_argument('--scoring_preserve_ratio', type=float, default=0.3,
                        help='Ratio of tokens to protect from merging based on scores (0.0 to 1.0)')
    parser.add_argument('--scoring_mode', type=str, default='high',
                        choices=['high', 'low', 'medium', 'timestep_scheduler', 'reverse_timestep_scheduler', 'uniform'],
                        help='How to select tokens based on scores')
    parser.add_argument('--scoring_preserve_spatial_uniformity', type=int, default=0,
                        help='Whether to preserve spatial uniformity in scoring-based merging (1 for True, 0 for False). '
                             'True: applies bipartite matching to full image first, then filters protected tokens. '
                             'False: extracts mergeable subset first (current default behavior).')
    parser.add_argument('--if_low_frequency_dst_tokens', type=int, default=0,
                        help='Whether to select lowest-scored tokens as destinations within each spatial block (1 for True, 0 for False). '
                             'When enabled, uses token scores to guide destination selection instead of random/first position. '
                             'Only applies when scoring-based token merging is enabled.')
    
    # Frequency scorer specific arguments
    parser.add_argument('--scoring_freq_method', type=str, default='1d_dft',
                        choices=['1d_dft', '1d_dct'],
                        help='Frequency analysis method for FrequencyScorer')
    parser.add_argument('--scoring_freq_ranking', type=str, default='amplitude',
                        choices=['amplitude', 'spectral_centroid'],
                        help='Ranking method for FrequencyScorer')
    
    # Spatial filter scorer specific arguments
    parser.add_argument('--scoring_spatial_method', type=str, default='2d_conv',
                        choices=['2d_conv', '2d_conv_l2'],
                        help='Spatial filter method for SpatialFilterScorer')
    parser.add_argument('--scoring_spatial_norm', type=str, default='l1',
                        choices=['l1', 'l2'],
                        help='Norm type for SpatialFilterScorer')
    
    # Statistical scorer specific arguments
    parser.add_argument('--scoring_stat_method', type=str, default='l2norm',
                        choices=['variance', 'l1norm', 'l2norm', 'mean_deviation'],
                        help='Statistical method for StatisticalScorer')
    
    # Signal processing scorer specific arguments
    parser.add_argument('--scoring_signal_method', type=str, default='snr',
                        choices=['snr', 'noise_magnitude'],
                        help='Signal processing method for SignalProcessingScorer')
    
    # Spatial distribution scorer specific arguments
    parser.add_argument('--scoring_spatial_alpha', type=float, default=2.0,
                        help='Alpha parameter for SpatialDistributionScorer (must be > 1)')
    
    # Similarity scorer specific arguments
    parser.add_argument('--scoring_similarity_method', type=str, default='local_neighbors_inverted',
                        choices=['local_neighbors', 'global_mean', 'local_neighbors_inverted', 'global_mean_inverted'],
                        help='Similarity method for SimilarityScorer. '
                             'local_neighbors: similarity with spatial neighbors. '
                             'global_mean: similarity with global token mean. '
                             '_inverted methods: return negative similarity (low scores for similar tokens) - RECOMMENDED. '
                             'Use inverted methods with score_mode="high" to protect dissimilar tokens and merge similar ones.')

    # Scoring merge matching algorithm
    parser.add_argument('--scoring_matching_algorithm', type=str, default='bipartite',
                        choices=['bipartite', 'abp'],
                        help='Matching algorithm for scoring-based token merging. Options: '
                             '"bipartite" (standard bipartite matching - more accurate), '
                             '"abp" (Adaptive Block Pooling - 2-10x faster with similar quality). '
                             'ABP is recommended for speed-critical applications.')
    
    # ABP-specific parameters
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

    ## for Token Downsampling
    parser.add_argument('--if_attention_proc', type=int, default=0,
                        help='Whether to apply attention processor patching (1 for True, 0 for False)')
    parser.add_argument('--merge_tokens', type=str, default='keys/values',
                        choices=['keys/values', 'all'],
                        help='Which tokens to merge')
    parser.add_argument('--merge_method', type=str, default='downsample',
                        choices=['none', 'similarity', 'downsample','downsample_custom_block', 'frequency_blockwise','frequency_global', 'block_avg_pool', 'downsample_qkv_upsample_out', 'masked_attention', 'blockwise_masked_attention', 'snr_masked_attention', 'snr_blockwise_masked_attention', 'noise_magnitude_masked_attention', 'noise_magnitude_blockwise_masked_attention'],
                        help='Method to use for merging tokens')
    parser.add_argument('--downsample_method', type=str, default='nearest-exact',
                        help='Interpolation method for downsampling (use "linear_blend" for linear interpolation)')
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
    parser.add_argument('--extra_guidance_scale', type=float, default=0,
                        help='Additional guidance scale to compensate for token merging')
    parser.add_argument('--downsample_factor_h', type=int, default=2,
                        help='Downsample factor for height dimension, only used if merge_method is downsample_custom_block')
    parser.add_argument('--downsample_factor_w', type=int, default=2,
                        help='Downsample factor for width dimension, only used if merge_method is downsample_custom_block')

    # --- Frequency-specific args (relevant if merge_method is frequency_*) ---

    parser.add_argument('--frequency_selection_mode', type=str, default='high',
                        help='Frequency selection mode for downsampling (high/low/...)')
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

    # ImprovedTokenMerge/attention processor arguments
    parser.add_argument('--qkv_downsample_method', type=str, default='nearest',
                       help='Downsample method for QKV')
    parser.add_argument('--out_upsample_method', type=str, default='nearest',
                       help='Upsample method for output')
    
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

    # Agent-guided token downsampling arguments
    parser.add_argument('--if_agent_guided', type=int, default=0,
                        help='Whether to apply NEW SIMPLIFIED agent-guided token merging (1 for True, 0 for False). '
                             'MAJOR UPDATE: 10-100x faster than old bipartite matching, O(N×agents) complexity, '
                             '50-90%% memory reduction, outputs k important + 1 merged token')
    parser.add_argument('--agent_method', type=str, default='adaptive_spatial',
                        choices=['adaptive_spatial', 'clustering_centroids', 'statistical_moments', 'frequency_based', 'uniform_sampling'],
                        help='Agent creation method for simplified agent-guided token selection. '
                             'adaptive_spatial: Best for vision transformers (requires H,W). '
                             'statistical_moments: General use, captures statistical properties. '
                             'clustering_centroids: Discovers natural token clusters. '
                             'frequency_based: For frequency domain structure. '
                             'uniform_sampling: Fast baseline with even distribution.')
    parser.add_argument('--agent_importance_method', type=str, default='cross_attention',
                        choices=['cross_attention', 'cosine_similarity', 'euclidean_distance', 'information_theoretic'],
                        help='Agent importance scoring method for simplified approach. '
                             'cross_attention: Uses agent-token attention (semantically meaningful). '
                             'cosine_similarity: Normalized dot product (scale-invariant). '
                             'euclidean_distance: Proximity-based scoring (intuitive). '
                             'information_theoretic: Entropy-based diversity-aware selection.')
    parser.add_argument('--num_agents', type=int, default=16,
                        help='Number of agent tokens to create (typically 8-32). '
                             'PERFORMANCE: More agents = better quality but slower. '
                             'Recommended: 16 for balance between quality and speed in simplified approach.')
    parser.add_argument('--agent_base_method', type=str, default=None,
                        choices=['None', 'original', '1d_dft', '1d_dct', '2d_conv', '2d_conv_l2'],
                        help='Base scoring method for HYBRID simplified approach (None for pure simplified agent). '
                             'None: Pure simplified agent (fastest, O(N×agents)). '
                             'original: Hybrid with L1/L2 norm. 1d_dft: Hybrid with DFT. '
                             '2d_conv: Hybrid with 2D convolution features.')
    parser.add_argument('--agent_base_ranking', type=str, default='l2norm',
                        choices=['l1norm', 'l2norm', 'variance', 'amplitude', 'spectral_centroid'],
                        help='Base ranking method for hybrid simplified approach (used with --agent_base_method). '
                             'l2norm: L2 norm ranking. amplitude: DFT amplitude ranking. '
                             'spectral_centroid: Frequency-based ranking.')
    parser.add_argument('--agent_weight', type=float, default=1.0,
                        help='Weight for agent vs base scoring in hybrid simplified approach. '
                             '1.0 = pure simplified agent (fastest). '
                             '0.7 = 70%% agent + 30%% base method (good balance). '
                             '0.5 = 50/50 hybrid (more robust but slower).')
    parser.add_argument('--agent_preserve_ratio', type=float, default=0.3,
                        help='Ratio of tokens to keep as important in simplified approach (rest merged into single token). '
                             'SIMPLIFIED MERGING: All unimportant tokens merged into ONE token by averaging. '
                             'Range: 0.0 to 1.0. Higher = more tokens kept, less compression.')
    parser.add_argument('--agent_score_mode', type=str, default='high',
                        choices=['high', 'low', 'medium', 'timestep_scheduler', 'reverse_timestep_scheduler', 'uniform'],
                        help='How to select important tokens based on simplified agent scores. '
                             'high: Keep highest scoring tokens (most important). '
                             'low: Keep lowest scoring tokens. medium: Keep medium scoring tokens. '
                             'timestep_scheduler: Adaptive based on diffusion timestep.')
    parser.add_argument('--agent_preserve_spatial_uniformity', type=int, default=0,
                        help='Whether to preserve spatial uniformity in simplified agent-guided merging '
                             '(1 for True, 0 for False). '
                             'NEW: Works with simplified O(N×agents) approach instead of expensive bipartite matching.')

    # DeepCache arguments
    parser.add_argument('--if_deepcache', type=int, default=0,
                        help='Enable DeepCache caching functionality (set to 1 to enable)')
    parser.add_argument('--cache_interval', type=int, default=1001,
                        help='Cache interval for DeepCache, default always ')
    parser.add_argument('--cache_branch_id', type=int, default=0,
                        help='Cache branch id for DeepCache')
    # for random seed
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    
    # for attention processor consistency
    parser.add_argument('--force_attention_processor', type=str, default='AttnProcessor',
                        choices=['AttnProcessor', 'AttnProcessor2_0', 'XFormersAttnProcessor', 'none'])

    parser.add_argument('--fix_noise_across_timestep', action='store_true',
                    help='If set, use the same noise across different timesteps for each prompt')

    parser.add_argument('--summary_csv', type=str, default='classification_experiments_summary.csv',
                        help='Path to the CSV file for logging experiment summaries.')

    # Force recalculation
    parser.add_argument('--force_recalc', action='store_true',
                        help='Force recalculation of errors even if result files already exist')

    parser.add_argument('--eval_denoise_steps', type=int, default=1,
                        help='Number of denoising steps to perform in eval_error. Default=1 (original method).')
   
    parser.add_argument('--single_timestep', type=int, default=-1,
                        help='If set to a non-negative value, evaluate ONLY this specific timestep '
                             'instead of the adaptive multi-timestep strategy. '
                             'Disables --to_keep and --n_samples.')
  

    parser.add_argument('--eval_step_stride', type=int, default=1,
                    help='Stride to use when updating timesteps within the eval_error loop '
                            'if eval_denoise_steps > 1. Default=1 (standard single step).')

    parser.add_argument('--eval_error_method', type=str, default='direct',
                        choices=['trajectory', 'direct', 'weighted', 'clean_signal'],
                        help='Error calculation method for multi-step denoising (eval_denoise_steps > 1). '
                             '"trajectory": Compare final latent to ground truth at same timestep (more stable). '
                             '"direct": Compare final latent directly to clean x0 (traditional approach). '
                             '"weighted": Like direct but weighted by inverse noise variance for fair comparison. '
                             '"clean_signal": Compare final latent to x0_original * sqrt_alpha_final (clean signal component).')

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
    
    # QKV timestep interpolation parameters
    parser.add_argument('--qkv_linear_blend_timestep_interpolation', type=int, default=0,
                        help='Enable timestep-based interpolation for qkv_linear_blend_factor (1 for True, 0 for False)')
    parser.add_argument('--qkv_linear_blend_start_ratio', type=float, default=0.1,
                        help='Starting QKV blend factor at timestep 999 for timestep interpolation')
    parser.add_argument('--qkv_linear_blend_end_ratio', type=float, default=0.9,
                        help='Ending QKV blend factor at timestep 0 for timestep interpolation')
    
    # Output timestep interpolation parameters
    parser.add_argument('--out_linear_blend_timestep_interpolation', type=int, default=0,
                        help='Enable timestep-based interpolation for out_linear_blend_factor (1 for True, 0 for False)')
    parser.add_argument('--out_linear_blend_start_ratio', type=float, default=0.1,
                        help='Starting output blend factor at timestep 999 for timestep interpolation')
    parser.add_argument('--out_linear_blend_end_ratio', type=float, default=0.9,
                        help='Ending output blend factor at timestep 0 for timestep interpolation')
 
      
    args = parser.parse_args()
    
    # Convert string 'None' to Python None for agent_base_method
    if hasattr(args, 'agent_base_method') and args.agent_base_method == 'None':
        args.agent_base_method = None
    
    assert len(args.to_keep) == len(args.n_samples)

    # --- Validate arguments based on single_timestep mode ---
    if args.single_timestep >= 0:
        print(f"INFO: Running in single timestep mode (t={args.single_timestep}). "
              "Arguments --to_keep and --n_samples will be ignored.")
        # Optionally clear or ignore them if needed later, but eval_prob_adaptive will bypass them
        args.to_keep = []
        args.n_samples = []
    else:
        # Ensure adaptive args are provided if not in single timestep mode
        if not args.to_keep or not args.n_samples:
            parser.error("Arguments --to_keep and --n_samples are required when --single_timestep is not set.")
        assert len(args.to_keep) == len(args.n_samples), "--to_keep and --n_samples must have the same number of elements for adaptive mode."



    # Validate token merging method parameter
    if args.token_merging_method != 'mean' and not any([args.if_token_merging, args.if_agentsd, args.if_scoring_merge, args.if_agent_guided]):
        print(f"WARNING: --token_merging_method={args.token_merging_method} specified but no token merging method is enabled. "
              "Consider setting --if_token_merging=1, --if_scoring_merge=1, --if_agent_guided=1")

    # Apply the seed
    seed_everything(args.seed)



    # make run output folder
    name = f"v{args.version}_{args.n_trials}trials_"
    if args.single_timestep >= 0:
        name += f'single_t{args.single_timestep}' # Identify single timestep runs
    else:
        # Original adaptive naming
        name += '_'.join(map(str, args.to_keep)) + 'keep_'
        name += '_'.join(map(str, args.n_samples)) + 'samples'
    name += f'_seed{args.seed}'
    if args.interpolation != 'bicubic':
        name += f'_{args.interpolation}'
    if args.loss == 'l1':
        name += '_l1'
    elif args.loss == 'huber':
        name += '_huber'
    if args.img_size != 512:
        name += f'_{args.img_size}'
    
    if args.if_token_merging == 1:
        name += f'_tokenmerge{args.token_merging_ratio}_userand{args.token_merging_use_rand}_maxdownsample{args.token_merging_max_downsample}_singlelevel{args.token_merging_single_downsample_level_merge}_sx{args.token_merging_sx}_sy{args.token_merging_sy}_cache{args.token_merging_cache_indices_per_image}'
        
        # Add merge layer types to name
        merge_layers = []
        if args.merge_attn == 1:
            merge_layers.append("attn")
        if args.merge_crossattn == 1:
            merge_layers.append("crossattn")
        if args.merge_mlp == 1:
            merge_layers.append("mlp")
        if merge_layers:
            name += f'_merge{"+".join(merge_layers)}'
        
        # Add method to name if not default
        if args.token_merging_method != 'mean':
            name += f'_method{args.token_merging_method}'
        if args.if_proportional_attention == 1:
            name += '_propattn'
        # Add locality-based sub-block parameters if not default
        if getattr(args, 'locality_block_factor_h', 1) > 1 or getattr(args, 'locality_block_factor_w', 1) > 1:
            name += f'_locality_h{getattr(args, "locality_block_factor_h", 1)}_w{getattr(args, "locality_block_factor_w", 1)}'
    elif args.if_agentsd == 1:
        name += f'_agentsd{args.token_merging_ratio}_agentratio{args.agent_ratio}_userand{args.token_merging_use_rand}_maxdownsample{args.token_merging_max_downsample}_singlelevel{args.token_merging_single_downsample_level_merge}_sx{args.token_merging_sx}_sy{args.token_merging_sy}_cache{args.token_merging_cache_indices_per_image}'
        # Add method to name if not default
        if args.token_merging_method != 'mean':
            name += f'_method{args.token_merging_method}'
        if args.if_proportional_attention == 1:
            name += '_propattn'
        # Add locality-based sub-block parameters if not default
        if getattr(args, 'locality_block_factor_h', 1) > 1 or getattr(args, 'locality_block_factor_w', 1) > 1:
            name += f'_locality_h{getattr(args, "locality_block_factor_h", 1)}_w{getattr(args, "locality_block_factor_w", 1)}'
    elif args.if_scoring_merge == 1:
        name += (f'_scoringmerge{args.token_merging_ratio}'
                f'_method{args.scoring_method}'
                f'_preserve{args.scoring_preserve_ratio}'
                f'_mode{args.scoring_mode}'
                f'_userand{args.token_merging_use_rand}'
                f'_maxdown{args.token_merging_max_downsample}'
                f'_sx{args.token_merging_sx}_sy{args.token_merging_sy}'
                f'_cache{args.token_merging_cache_indices_per_image}')
        
        # Add merge layer types to name
        merge_layers = []
        if args.merge_attn == 1:
            merge_layers.append("attn")
        if args.merge_crossattn == 1:
            merge_layers.append("crossattn")
        if args.merge_mlp == 1:
            merge_layers.append("mlp")
        if merge_layers:
            name += f'_merge{"+".join(merge_layers)}'
        
        # Add matching algorithm to name for clarity
        name += f'_alg{getattr(args, "scoring_matching_algorithm", "bipartite")}'
        
        # Add ABP-specific parameters if using ABP
        if getattr(args, "scoring_matching_algorithm", "bipartite") == "abp":
            name += f'_abpscorer{getattr(args, "abp_scorer_method", "spatial_filter")}'
            name += f'_abpagg{getattr(args, "abp_tile_aggregation", "max")}'
        
        # Add merging method to name if not default
        if args.token_merging_method != 'mean':
            name += f'_mergemethod{args.token_merging_method}'
        
        # Add spatial uniformity parameter to name
        if args.scoring_preserve_spatial_uniformity == 1:
            name += '_spatialuniform'
        
        # Add score-guided destination selection parameter to name
        if args.if_low_frequency_dst_tokens == 1:
            name += '_lowfreqdst'
        
        # Add method-specific parameters to name
        if args.scoring_method == 'frequency':
            name += f'_freqmethod{args.scoring_freq_method}_freqranking{args.scoring_freq_ranking}'
        elif args.scoring_method == 'spatial_filter':
            name += f'_spatialmethod{args.scoring_spatial_method}_spatialnorm{args.scoring_spatial_norm}'
        elif args.scoring_method == 'statistical':
            name += f'_statmethod{args.scoring_stat_method}'
        elif args.scoring_method == 'signal_processing':
            name += f'_signalmethod{args.scoring_signal_method}'
        elif args.scoring_method == 'spatial_distribution':
            name += f'_spatialalpha{args.scoring_spatial_alpha}'
        elif args.scoring_method == 'similarity':
            name += f'_similaritymethod{args.scoring_similarity_method}'
            
        if args.if_proportional_attention == 1:
            name += '_propattn'
        
        # Add locality-based sub-block parameters if not default
        if getattr(args, 'locality_block_factor_h', 1) > 1 or getattr(args, 'locality_block_factor_w', 1) > 1:
            name += f'_locality_h{getattr(args, "locality_block_factor_h", 1)}_w{getattr(args, "locality_block_factor_w", 1)}'
        
        # Add resolution caching parameters if enabled
        if getattr(args, 'cache_resolution_merge', 0) == 1:
            name += f'_rescache{args.cache_resolution_mode}'
    elif args.if_agent_guided == 1:
        # Agent-guided token merging naming (NEW SIMPLIFIED APPROACH)
        name += (f'_agentguided_SIMPLIFIED{args.token_merging_ratio}'
                f'_method{args.agent_method}'
                f'_importance{args.agent_importance_method}'
                f'_agents{args.num_agents}'
                f'_preserve{args.agent_preserve_ratio}'
                f'_mode{args.agent_score_mode}')
        
        # Add merge layer types to name
        merge_layers = []
        if args.merge_attn == 1:
            merge_layers.append("attn")
        if args.merge_crossattn == 1:
            merge_layers.append("crossattn")
        if args.merge_mlp == 1:
            merge_layers.append("mlp")
        if merge_layers:
            name += f'_merge{"+".join(merge_layers)}'
        
        # Add method-specific performance indicator
        if args.agent_base_method is None or args.agent_base_method == 'None':
            # Pure simplified agent (fastest)
            name += f'_PURE_AGENT_SIMPLIFIED'
        else:
            # Hybrid simplified agent
            name += f'_HYBRID_SIMPLIFIED'
        
        # Add base method info for hybrid
        if args.agent_base_method is not None and args.agent_base_method != 'None':
            name += f'_base{args.agent_base_method}_ranking{args.agent_base_ranking}_weight{args.agent_weight}'
        
        # Add merging method to name if not default
        if args.token_merging_method != "mean":
            name += f'_mergemethod{args.token_merging_method}'
        
        # Add spatial uniformity parameter to name
        if args.agent_preserve_spatial_uniformity == 1:
            name += f'_spatialuniform'
        
        # Add locality-based sub-block parameters if not default
        if getattr(args, 'locality_block_factor_h', 1) > 1 or getattr(args, 'locality_block_factor_w', 1) > 1:
            name += f'_locality_h{getattr(args, "locality_block_factor_h", 1)}_w{getattr(args, "locality_block_factor_w", 1)}'
        
        # Add performance indicators to distinguish from old expensive approach
        name += f'_PERFORMANCE_OPTIMIZED'  # Indicates 10-100x faster than bipartite
    elif args.if_attention_proc == 1:
        if args.merge_method == 'frequency_global':
            name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqmethod{args.frequency_selection_method}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_level1ratio{args.token_merging_ratio}'
                    f'_level2method{args.secondary_merge_method}'
                    f'_level2ratio{args.ratio_level_2}')
            if args.frequency_selection_method=='non_uniform_grid':
                name += f'_gridalpha{args.frequency_grid_alpha}'
        elif args.merge_method == 'frequency_blockwise':
            name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqmethod{args.frequency_selection_method}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_level1downsample{args.downsample_factor}'
                    f'_level2method{args.secondary_merge_method}'
                    f'_level2downsample{args.downsample_factor_level_2}')
            # Add blockwise blend parameter to name if not default
            if getattr(args, 'blockwise_blend_factor', 0.5) != 0.5:
                name += f'_blockwiseblend{args.blockwise_blend_factor}'
        elif args.merge_method == 'masked_attention':
            name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqmethod{args.frequency_selection_method}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_level1ratio{args.token_merging_ratio}'
                    f'_level2method{args.secondary_merge_method}'
                    f'_level2ratio{args.ratio_level_2}')
            if args.frequency_selection_method=='non_uniform_grid':
                name += f'_gridalpha{args.frequency_grid_alpha}'
        elif args.merge_method == 'blockwise_masked_attention':
            name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqmethod{args.frequency_selection_method}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_level1downsample{args.downsample_factor}'
                    f'_level2method{args.secondary_merge_method}'
                    f'_level2downsample{args.downsample_factor_level_2}')
        elif args.merge_method in ['snr_masked_attention', 'snr_blockwise_masked_attention', 'noise_magnitude_masked_attention', 'noise_magnitude_blockwise_masked_attention']:
            name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}')
            if 'blockwise' in args.merge_method:
                name += f'_level1downsample{args.downsample_factor}'
                name += f'_level2method{args.secondary_merge_method}'
                name += f'_level2downsample{args.downsample_factor_level_2}'
            else:
                name += f'_level1ratio{args.token_merging_ratio}'
                name += f'_level2method{args.secondary_merge_method}'
                name += f'_level2ratio{args.ratio_level_2}'
        elif args.merge_method == 'downsample_qkv_upsample_out':
            name += (f'_mergemethod{args.merge_method}'
                    f'_downsamplemethod{args.downsample_method}'
                    f'_qkv_downsamplemethod{args.qkv_downsample_method}'
                    f'_out_upsamplemethod{args.out_upsample_method}'
                    f'_level1downsample{args.downsample_factor}'
                    f'_level2downsample{args.downsample_factor_level_2}')
            # Add linear_blend parameters to name if using linear_blend methods
            if args.downsample_method == 'linear_blend':
                if getattr(args, 'linear_blend_timestep_interpolation', 0):
                    name += f'_linearblend_timestep{getattr(args, "linear_blend_start_ratio", 0.1)}-{getattr(args, "linear_blend_end_ratio", 0.9)}'
                else:
                    name += f'_linearblend{args.linear_blend_factor}'
                name += f'_blendmethods{getattr(args, "linear_blend_method_1", "nearest-exact")}-{getattr(args, "linear_blend_method_2", "avg_pool")}'
            if getattr(args, 'qkv_downsample_method', 'nearest') == 'linear_blend':
                if getattr(args, 'qkv_linear_blend_timestep_interpolation', 0):
                    name += f'_qkvblend_timestep{getattr(args, "qkv_linear_blend_start_ratio", 0.1)}-{getattr(args, "qkv_linear_blend_end_ratio", 0.9)}'
                else:
                    name += f'_qkvblend{args.qkv_linear_blend_factor}'
                name += f'_qkvblendmethods{getattr(args, "qkv_linear_blend_method_1", "nearest-exact")}-{getattr(args, "qkv_linear_blend_method_2", "avg_pool")}'
            if getattr(args, 'out_upsample_method', 'nearest') == 'linear_blend':
                if getattr(args, 'out_linear_blend_timestep_interpolation', 0):
                    name += f'_outblend_timestep{getattr(args, "out_linear_blend_start_ratio", 0.1)}-{getattr(args, "out_linear_blend_end_ratio", 0.9)}'
                else:
                    name += f'_outblend{args.out_linear_blend_factor}'
                name += f'_outblendmethods{getattr(args, "out_linear_blend_method_1", "nearest-exact")}-{getattr(args, "out_linear_blend_method_2", "avg_pool")}'
        elif args.merge_method == 'downsample_custom_block':
            name += (f'_mergemethod{args.merge_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_h{args.downsample_factor_h}'
                    f'_w{args.downsample_factor_w}'
                    f'_switch{args.timestep_threshold_switch}'
                    f'_stop{args.timestep_threshold_stop}')
        else:
            name += (f'tokendownsampling{args.token_merging_ratio}_userand{args.token_merging_use_rand}__sx{args.token_merging_sx}_sy{args.token_merging_sy}'
                    f'_method{args.merge_method}'
                    f'_down{args.downsample_factor}_downmethod{args.downsample_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_switch{args.timestep_threshold_switch}'
                    f'_stop{args.timestep_threshold_stop}'
                    f'_secondleveldown{args.downsample_factor_level_2}')
            # Add linear_blend parameter to name if using linear_blend method
            if args.downsample_method == 'linear_blend':
                if getattr(args, 'linear_blend_timestep_interpolation', 0):
                    name += f'_linearblend_timestep{getattr(args, "linear_blend_start_ratio", 0.1)}-{getattr(args, "linear_blend_end_ratio", 0.9)}'
                else:
                    name += f'_linearblend{args.linear_blend_factor}'
                name += f'_blendmethods{getattr(args, "linear_blend_method_1", "nearest-exact")}-{getattr(args, "linear_blend_method_2", "avg_pool")}'
    elif args.if_sito == 1:
        name += (f'_sito_prune{args.sito_prune_ratio}'
                f'_maxdownsample{args.sito_max_downsample_ratio}'
                f'_selfattn{args.sito_prune_selfattn_flag}'
                f'_crossattn{args.sito_prune_crossattn_flag}'
                f'_mlp{args.sito_prune_mlp_flag}'
                f'_sx{args.sito_sx}_sy{args.sito_sy}'
                f'_noisealpha{args.sito_noise_alpha}'
                f'_simbeta{args.sito_sim_beta}')
    
    if args.if_deepcache == 1:
        name += f'_deepcache_interval{args.cache_interval}_branch{args.cache_branch_id}'
    if args.eval_denoise_steps > 1:
        name += f'_denoise_steps{args.eval_denoise_steps}'
        if args.eval_step_stride > 1:
             name += f'_stride{args.eval_step_stride}'
             name += f'_error_method{args.eval_error_method}'

    if args.fix_noise_across_timestep:
        name += '_fixed_noise'

    if args.extra is not None:
        run_folder = osp.join(LOG_DIR, args.dataset + '_' + args.extra, name)
    else:
        run_folder = osp.join(LOG_DIR, args.dataset, name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')
    print(f'Errors will be saved to: {run_folder}')

    
    ### We'll store line-by-line logs in "results_log.txt"
    results_txt_path = osp.join(run_folder, "results_log.txt")
    

    # set up dataset and prompts
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    target_dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    prompts_df = pd.read_csv(args.prompt_path)

    # load pretrained models
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    torch.backends.cudnn.benchmark = True
    
    # Import functions for fixed mask computation if using masked_attention or blockwise_masked_attention
    if args.if_attention_proc == 1 and args.merge_method in ["masked_attention", "blockwise_masked_attention"]:
        from ImprovedTokenMerge.merge import precompute_fixed_mask_from_original, reset_fixed_masks


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
            unet.set_attn_processor(chosen_processor)
            print(f"Set attention processor to: {type(chosen_processor).__name__}")
        except Exception as e:
            print(f"Failed to set attention processor: {e}")
   

    # --- Measure FLOPs for one UNet forward pass ---
    unet_flops_per_pass = 0.0

    # ------------------------------------------------------------------
    #  Measure FLOPs for ONE UNet forward pass (safe & accurate)
    # ------------------------------------------------------------------
    print("\nMeasuring FLOPs for one UNet forward pass (logical batch = 1)…")

    try:
        # ----------------------------- dummy inputs -----------------------------
        b_dummy = 1                       
        dt      = torch.float16 if args.dtype == "float16" else torch.float32
        dummy_latents = torch.randn(
            b_dummy, unet.config.in_channels, latent_size, latent_size,
            device=device, dtype=dt
        )
        dummy_timestep = torch.tensor(
            [scheduler.config.num_train_timesteps // 2],
            device=device
        )
        ca_dim = (unet.config.cross_attention_dim
                if getattr(unet.config, "cross_attention_dim", None) is not None
                else getattr(text_encoder.config, "hidden_size", 768))
        dummy_context = torch.randn(
            b_dummy, tokenizer.model_max_length, ca_dim,
            device=device, dtype=dt
        )

        # For masked_attention or blockwise_masked_attention or new SNR/noise methods: create dummy mask for FLOP measurement
        needs_dummy_mask = False
        if args.if_attention_proc == 1 and args.merge_method in ["masked_attention", "blockwise_masked_attention", "snr_masked_attention", "snr_blockwise_masked_attention", "noise_magnitude_masked_attention", "noise_magnitude_blockwise_masked_attention"]:
            needs_dummy_mask = True
        elif getattr(args, 'if_scoring_merge', 0) == 1 and getattr(args, 'scoring_method', '') == 'signal_processing' and getattr(args, 'scoring_signal_method', '') in ["snr", "noise_magnitude"]:
            needs_dummy_mask = True
            
        if needs_dummy_mask:
            if hasattr(unet, '_tome_info') and unet._tome_info is not None:
                from ImprovedTokenMerge.merge import precompute_fixed_mask_from_original
                
                # Create dummy latent in attention format for mask generation
                dummy_latent_reshaped = torch.randn(1, 4096, 4, device=device, dtype=dt)
                
                # For SNR and noise magnitude methods, we need additional parameters for FLOP measurement
                additional_params = {}
                if ((args.if_attention_proc == 1 and ('snr_' in args.merge_method or 'noise_magnitude_' in args.merge_method)) or 
                    (getattr(args, 'if_scoring_merge', 0) == 1 and getattr(args, 'scoring_method', '') == 'signal_processing' and getattr(args, 'scoring_signal_method', '') in ["snr", "noise_magnitude"])):
                    # Create dummy noise for FLOP measurement
                    dummy_noise = torch.randn_like(dummy_latent_reshaped)
                    additional_params['original_noise'] = dummy_noise
                
                precompute_fixed_mask_from_original(dummy_latent_reshaped, unet._tome_info, target_level="level_1", **additional_params)
                print("Created temporary mask for FLOP measurement")

        # ------------------------------- counter -------------------------------
        with FlopCounterMode(unet, display=True) as fc:   # counter OUTER-most
            with torch.no_grad():                        #    (no grads, low‑RAM)
                _ = unet(dummy_latents,
                        dummy_timestep,
                        encoder_hidden_states=dummy_context).sample

        # ----------------------------- results -----------------------------
        flops_batch1 = fc.get_total_flops()              # for batch = 1
        unet_flops_per_pass = flops_batch1 
        print(f"Measured FLOPs/pass (batch={1}): "
            f"{format_flops(unet_flops_per_pass)}")

    except Exception as e:
        print("ERROR while counting FLOPs:", e)
        import traceback; traceback.print_exc()
        unet_flops_per_pass = -1

    finally:
        # Reset masks after FLOP measurement for clean evaluation
        if args.if_attention_proc == 1 and args.merge_method in ["masked_attention", "blockwise_masked_attention", "snr_masked_attention", "snr_blockwise_masked_attention", "noise_magnitude_masked_attention", "noise_magnitude_blockwise_masked_attention"]:
            if hasattr(unet, '_tome_info') and unet._tome_info is not None:
                from ImprovedTokenMerge.merge import reset_fixed_masks
                reset_fixed_masks(unet._tome_info)
                print("Reset masks after FLOP measurement")
                
        del dummy_latents, dummy_timestep, dummy_context
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




    # load noise
    if args.noise_path is not None:
        assert not args.zero_noise
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None

    # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L276
    text_input = tokenizer(prompts_df.prompt.tolist(), padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            text_embeddings = text_encoder(
                text_input.input_ids[i: i + 100].to(device),
            )[0]
            embeddings.append(text_embeddings)
    text_embeddings = torch.cat(embeddings, dim=0)
    assert len(text_embeddings) == len(prompts_df)

    # subset of dataset to evaluate
    if args.subset_path is not None:
        idxs = np.load(args.subset_path).tolist()
    else:
        idxs = list(range(len(target_dataset)))
    idxs_to_eval = idxs[args.worker_idx::args.n_workers]

    formatstr = get_formatstr(len(target_dataset) - 1)
    correct = 0
    total = 0
    pbar = tqdm.tqdm(idxs_to_eval)


    UNET_TOTAL_TIME = 0.0
    total_start_time = time.time()
    
    # Track the true_minus_min_error statistics
    true_minus_min_errors = []
    
    # Track mAP statistics for COCO dataset
    coco_ap_values = []  # Average Precision values for each COCO image
    coco_precision_at_1_values = []  # Precision@1 values for each COCO image
    coco_recall_at_5_values = []     # Recall@5 values for each COCO image
    
    for i in pbar:
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
        fname = osp.join(run_folder, formatstr.format(i) + '.pt')
        if os.path.exists(fname) and not args.force_recalc:
            print('Skipping', i)
            if args.load_stats:
                data = torch.load(fname)
                loaded_pred = data['pred']
                loaded_label = data['label']
                
                # Handle accuracy calculation (both single-label and multi-label)
                if isinstance(loaded_label, list):
                    # Multi-label dataset (e.g., COCO): label is a set of valid category indices
                    if loaded_pred in loaded_label:
                        correct += 1
                else:
                    # Single-label dataset: label is a single integer
                    if loaded_pred == loaded_label:
                        correct += 1
                        
                total += 1
                
                # Load true_minus_min_error if available
                if 'true_minus_min_error' in data and data['true_minus_min_error'] is not None:
                    true_minus_min_errors.append(data['true_minus_min_error'])
                
                # Recalculate COCO mAP metrics if this is COCO2017 dataset and we have the necessary data
                if args.dataset == 'coco2017' and isinstance(loaded_label, list) and 'errors' in data:
                    try:
                        loaded_errors = data['errors']
                        
                        # Precision@1: Use the loaded prediction
                        precision_at_1 = 1.0 if loaded_pred in loaded_label else 0.0
                        coco_precision_at_1_values.append(precision_at_1)
                        
                        # For mAP and Recall@5, use adaptive ranking simulation
                        ranked_classes = simulate_adaptive_ranking(loaded_errors, prompts_df, args)
                        
                        # Calculate Average Precision (AP) for this image
                        ap = calculate_average_precision(ranked_classes, loaded_label)
                        coco_ap_values.append(ap)
                        
                        # Calculate Recall@5
                        num_true_in_top_5 = sum(1 for label in ranked_classes[:5] if label in loaded_label)
                        recall_at_5 = num_true_in_top_5 / len(loaded_label) if len(loaded_label) > 0 else 0.0
                        coco_recall_at_5_values.append(recall_at_5)
                        
                        # Debug output for first few images
                        if i < 3:
                            print(f"[COCO mAP RELOAD] Image {i}: AP={ap:.4f}, P@1={precision_at_1:.4f}, R@5={recall_at_5:.4f}")
                            print(f"[COCO mAP RELOAD] True labels: {loaded_label}")
                            print(f"[COCO mAP RELOAD] Loaded prediction: {loaded_pred}, Ranking top 5: {ranked_classes[:5]}")
                            
                    except Exception as reload_map_error:
                        print(f"[COCO mAP RELOAD ERROR] Failed to recalculate mAP for image {i}: {reload_map_error}")
                        # Continue without adding to mAP statistics for this image
                        
            continue
        elif os.path.exists(fname) and args.force_recalc:
            print(f'Force recalculation enabled - overwriting existing file for sample {i}')
        if args.dataset == 'coco2017':
            sample = target_dataset[i]
            image = sample['image']
            label = sample['labels'] 
        else:
            image, label = target_dataset[i]
        # For COCO, `label` is a list of indices (multi-label).
        # For other datasets, it's an integer (single-label).
        
        with torch.no_grad():
            img_input = image.to(device).unsqueeze(0)
            if args.dtype == 'float16':
                img_input = img_input.half()
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= vae.config.scaling_factor
            
            # Clear cached indices for new image if cache_indices_per_image is enabled
            if (args.if_token_merging == 1 or args.if_scoring_merge == 1) and getattr(args, 'token_merging_cache_indices_per_image', 0):
                import tomesd
                tomesd.clear_cached_indices(unet)
                
                # Precompute cache indices from clean latent
                # Use the first prompt's text embedding as a representative dummy
                dummy_text_embedding = text_embeddings[0:1]  # Take first embedding, keep batch dim
                tomesd.precompute_cache_from_clean_latent(
                    unet, x0, 
                    dummy_encoder_hidden_states=dummy_text_embedding
                )
            
                    # For masked_attention or blockwise_masked_attention or new SNR/noise methods: precompute fixed mask from original clean image
        if args.if_attention_proc == 1 and args.merge_method in ["masked_attention", "blockwise_masked_attention", "snr_masked_attention", "snr_blockwise_masked_attention", "noise_magnitude_masked_attention", "noise_magnitude_blockwise_masked_attention"]:
            # Reset any existing masks from previous images (but keep global logging flags for run-wide logging)
            if hasattr(unet, '_tome_info') and unet._tome_info is not None:
                reset_fixed_masks(unet._tome_info)
            
            # Only precompute for non-SNR/noise methods (masked_attention, blockwise_masked_attention)
            # SNR and noise methods will be computed on-the-fly per image-noise pair
            if args.merge_method in ["masked_attention", "blockwise_masked_attention"]:
                # Precompute the fixed mask from the ORIGINAL CLEAN IMAGE
                # This ensures the mask is based on clean content, not noisy representations
                if hasattr(unet, '_tome_info') and unet._tome_info is not None:
                    # Reshape x0 to match attention layer format: (B, C, H, W) -> (B, H*W, C)
                    # x0 shape is (1, 4, 64, 64), need to reshape to (1, 4096, 4) for 64x64 resolution
                    B, C, H, W = x0.shape
                    x0_reshaped = x0.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (1, 4096, 4)
                    
                    precompute_fixed_mask_from_original(x0_reshaped, unet._tome_info, target_level="level_1")
                    print(f"Precomputed fixed mask from original clean image for sample {i}")
                else:
                    print("Warning: UNet _tome_info not available for mask precomputation")
            
            # For SNR and noise methods, store the clean image for later use
            snr_noise_merge_methods = ["snr_masked_attention", "snr_blockwise_masked_attention", "noise_magnitude_masked_attention", "noise_magnitude_blockwise_masked_attention"]
            scoring_snr_noise_methods = ["snr", "noise_magnitude"]
            
            use_snr_noise = False
            if args.if_attention_proc == 1 and getattr(args, 'merge_method', '') in snr_noise_merge_methods:
                use_snr_noise = True
            elif getattr(args, 'if_scoring_merge', 0) == 1 and getattr(args, 'scoring_method', '') == 'signal_processing' and getattr(args, 'scoring_signal_method', '') in scoring_snr_noise_methods:
                use_snr_noise = True
                
            if use_snr_noise:
                if hasattr(unet, '_tome_info') and unet._tome_info is not None:
                    # Store clean image for per-timestep mask computation
                    B, C, H, W = x0.shape
                    x0_reshaped = x0.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (1, 4096, 4)
                    unet._tome_info['current_clean_image'] = x0_reshaped
                    print(f"Stored clean image for on-the-fly SNR/noise mask computation for sample {i}")
                
        pred_idx, pred_errors = eval_prob_adaptive(unet, x0, text_embeddings, scheduler, args, latent_size, all_noise)
        pred = prompts_df.classidx[pred_idx]
        
        # Calculate true_error - min_error difference for single ground truth settings
        # This measures how much worse the true label performs compared to the best (minimum error) label
        # Positive values indicate the true label has higher error than the minimum
        # Negative values indicate the true label is actually the best choice
        true_minus_min_error = None
        if not isinstance(label, list):
            # Single-label dataset: calculate mean errors for each class and find difference
            class_mean_errors = []
            true_label_error = None
            
            # Get mean error for each evaluated class
            for class_idx in range(len(text_embeddings)):
                if class_idx in pred_errors:
                    mean_error = pred_errors[class_idx]['pred_errors'].mean().item()
                    class_mean_errors.append(mean_error)
                    
                    # Check if this is the true label
                    class_label = prompts_df.classidx[class_idx]
                    if args.dataset == "imagenet100":
                        # Use the mapping logic for ImageNet100
                        original_indices = sorted(prompts_df['classidx'].tolist())
                        sequential_to_original = {seq_idx: orig_idx for seq_idx, orig_idx in enumerate(original_indices)}
                        mapped_true_label = sequential_to_original[label]
                        if class_label == mapped_true_label:
                            true_label_error = mean_error
                    else:
                        if class_label == label:
                            true_label_error = mean_error
            
            # Debug check to verify we found the true label and evaluated expected classes
            if i < 5:  # Only print for first few samples
                total_classes = len(text_embeddings)
                evaluated_classes = len(class_mean_errors)
                found_true_label = true_label_error is not None
                print(f"[DEBUG] Sample {i}: total_classes={total_classes}, evaluated_classes={evaluated_classes}, "
                      f"found_true_label={found_true_label}, true_label={label}")
            
            # Calculate difference if we found the true label and have multiple classes
            if true_label_error is not None and len(class_mean_errors) > 1:
                min_error = min(class_mean_errors)
                true_minus_min_error = true_label_error - min_error
                
                # Debug print to verify calculation (can be removed later)
                if i < 5:  # Only print for first few samples to avoid spam
                    print(f"[DEBUG] Sample {i}: true_label_error={true_label_error:.8f}, "
                          f"min_error={min_error:.8f}, difference={true_minus_min_error:.8f}, "
                          f"num_classes_evaluated={len(class_mean_errors)}")
                    print(f"[DEBUG] Sample {i}: class_errors range: min={min(class_mean_errors):.8f}, "
                          f"max={max(class_mean_errors):.8f}, mean={sum(class_mean_errors)/len(class_mean_errors):.8f}")
        
        # Fix for ImageNet100: map dataset sequential index to original ImageNet index for comparison
        if args.dataset == "imagenet100" and not isinstance(label, list):
            # Create the same mapping as in sd_generation.py
            original_indices = sorted(prompts_df['classidx'].tolist())
            sequential_to_original = {seq_idx: orig_idx for seq_idx, orig_idx in enumerate(original_indices)}
            # Map the dataset label (sequential) to original ImageNet index
            mapped_label = sequential_to_original[label]
        else:
            mapped_label = label
            
        torch.save(dict(errors=pred_errors, pred=pred, label=mapped_label, true_minus_min_error=true_minus_min_error), fname)
        print(f'Saved error data to: {fname}')
        
        # Collect the true_minus_min_error for summary statistics
        if true_minus_min_error is not None:
            true_minus_min_errors.append(true_minus_min_error)
            
        # Calculate mAP metrics specifically for COCO dataset
        if args.dataset == 'coco2017' and isinstance(mapped_label, list):
            try:
                # For consistency with accuracy calculation, use the same adaptive prediction
                # Precision@1: Use the same prediction that eval_prob_adaptive selected
                precision_at_1 = 1.0 if pred in mapped_label else 0.0
                coco_precision_at_1_values.append(precision_at_1)
                
                # For mAP and Recall@5, use adaptive ranking simulation for accuracy
                ranked_classes = simulate_adaptive_ranking(pred_errors, prompts_df, args)
                

                
                # Calculate Average Precision (AP) for this image
                ap = calculate_average_precision(ranked_classes, mapped_label)
                coco_ap_values.append(ap)
                
                # Calculate Recall@5 (how many true labels are in top 5?)
                num_true_in_top_5 = sum(1 for label in ranked_classes[:5] if label in mapped_label)
                recall_at_5 = num_true_in_top_5 / len(mapped_label) if len(mapped_label) > 0 else 0.0
                coco_recall_at_5_values.append(recall_at_5)
                
                # Debug output for first few images
                if i < 3:
                    print(f"[COCO mAP DEBUG] Image {i}: AP={ap:.4f}, P@1={precision_at_1:.4f}, R@5={recall_at_5:.4f}")
                    print(f"[COCO mAP DEBUG] True labels: {mapped_label}")
                    print(f"[COCO mAP DEBUG] Adaptive winner: {pred}, Adaptive ranking top 5: {ranked_classes[:5]}")
                    adaptive_rank_1 = ranked_classes[0] if ranked_classes else None
                    match_status = "TRUE" if pred == adaptive_rank_1 else "FALSE"
                    print(f"[COCO mAP DEBUG] Consistency check - Adaptive winner: {pred}, Ranking #1: {adaptive_rank_1}, Match: {match_status}")
                
            except Exception as map_error:
                print(f"[COCO mAP ERROR] Failed to calculate mAP for image {i}: {map_error}")
                # Continue without adding to mAP statistics for this image
            
        # Handle accuracy calculation (both single-label and multi-label)
        if isinstance(mapped_label, list):
            # Multi-label dataset (e.g., COCO): label is a set of valid category indices
            print("prediction:",pred,"labels:",mapped_label)
            if pred in mapped_label:
                correct += 1
        else:
            # Single-label dataset (e.g., CIFAR, MNIST): label is a single integer
            if pred == mapped_label:
                correct += 1     
        total += 1
    ###
    total_end_time = time.time()  # Record end
    total_loop_time = total_end_time - total_start_time




    print(f"\n--- Evaluation Summary ---")
    print(f"Total wall-clock time for evaluation loop: {total_loop_time:.2f} seconds")
    print(f"Total time spent *only* in UNet forward passes (during eval loop): {UNET_TOTAL_TIME:.4f} seconds")
    print(f"Approx. FLOPs per UNet forward pass (batch=1): {format_flops(unet_flops_per_pass)}")

    if total > 0:
        final_acc=100.0 * correct / total
        final_acc_str = f"Final accuracy: {final_acc:.2f}%"
        print(final_acc_str)
    else:
        final_acc_str = "No samples processed."

    # Summary statistics for true_minus_min_error
    if true_minus_min_errors:
        true_minus_min_mean = np.mean(true_minus_min_errors)
        true_minus_min_std = np.std(true_minus_min_errors)
        true_minus_min_min = np.min(true_minus_min_errors)
        true_minus_min_max = np.max(true_minus_min_errors)
        
        true_minus_min_summary = (f"True-Min Error Stats - Mean: {true_minus_min_mean:.6f}, "
                                 f"Std: {true_minus_min_std:.6f}, "
                                 f"Min: {true_minus_min_min:.6f}, "
                                 f"Max: {true_minus_min_max:.6f}, "
                                 f"Count: {len(true_minus_min_errors)}")
        print(true_minus_min_summary)
    else:
        true_minus_min_summary = "True-Min Error Stats: No single-label samples processed."
        print(true_minus_min_summary)

    # Summary statistics for COCO mAP metrics
    coco_map_summary = ""
    if args.dataset == 'coco2017' and coco_ap_values:
        coco_mean_ap = np.mean(coco_ap_values)
        coco_mean_p1 = np.mean(coco_precision_at_1_values) 
        coco_mean_r5 = np.mean(coco_recall_at_5_values)
        coco_std_ap = np.std(coco_ap_values)
        
        coco_map_summary = (f"COCO mAP Stats - mAP: {coco_mean_ap:.4f}, "
                           f"Precision@1: {coco_mean_p1:.4f}, "
                           f"Recall@5: {coco_mean_r5:.4f}, "
                           f"mAP Std: {coco_std_ap:.4f}, "
                           f"Count: {len(coco_ap_values)}")
        print(coco_map_summary)
        print(f"COCO mAP Details - mAP: {coco_mean_ap:.6f}, P@1: {coco_mean_p1:.6f}, R@5: {coco_mean_r5:.6f}")
    elif args.dataset == 'coco2017':
        coco_map_summary = "COCO mAP Stats: No valid COCO images processed for mAP calculation."
        print(coco_map_summary)
    else:
        coco_map_summary = "COCO mAP Stats: Not applicable (not COCO dataset)."


    # --- Append final summary to results log file ---
    try:
        with open(results_txt_path, "a") as f:
            f.write("\n=== Final Summary ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(final_acc_str + "\n")
            f.write(f"Total wall-clock time for loop: {total_loop_time:.2f} sec\n")
            f.write(f"Total UNet forward pass time (accumulated in loop): {UNET_TOTAL_TIME:.4f} sec\n")
            f.write(f"FLOPs per UNet pass (batch={1}): {format_flops(unet_flops_per_pass)}\n")
            f.write(true_minus_min_summary + "\n")
            f.write(coco_map_summary + "\n")

            f.write("\n=== Command-Line Arguments ===\n")
            sorted_args = sorted(vars(args).items())
            for arg, value in sorted_args:
                f.write(f"{arg}: {value}\n")
            f.write("========================\n")
        print(f"Appended summary to: {results_txt_path}")
    except Exception as e:
        print(f"Error writing final summary to log file {results_txt_path}: {e}")


    # --- Append summary to the main CSV log file ---
    summary_csv_path = args.summary_csv
    print(f"Attempting to append results to summary CSV: {summary_csv_path}")
    try:
        headers = [
            'Timestamp', 'Run Folder Name',
            'Args JSON', # Single column for all arguments as a JSON string
            'Accuracy (%)', 'Total Time (s)', 'UNet Eval Time (s)', 'UNet FLOPs/pass (Batch)',
            'True-Min Error Mean', 'True-Min Error Std', 'True-Min Error Count',
            'COCO mAP', 'COCO Precision@1', 'COCO Recall@5', 'COCO mAP Count',
            'Log File Path'
        ]
        
        # Convert args namespace to a dictionary for serialization
        args_dict = vars(args)
        
        # Convert the dictionary to a JSON string for storing in a single CSV cell
        # Using sort_keys ensures consistent ordering within the JSON string
        args_json_string = json.dumps(args_dict, sort_keys=True)

        # Calculate COCO mAP values for CSV (if applicable)
        coco_mean_ap_csv = 'N/A'
        coco_mean_p1_csv = 'N/A' 
        coco_mean_r5_csv = 'N/A'
        coco_count_csv = '0'
        
        if args.dataset == 'coco2017' and coco_ap_values:
            coco_mean_ap_csv = f"{np.mean(coco_ap_values):.6f}"
            coco_mean_p1_csv = f"{np.mean(coco_precision_at_1_values):.6f}"
            coco_mean_r5_csv = f"{np.mean(coco_recall_at_5_values):.6f}"
            coco_count_csv = f"{len(coco_ap_values)}"

        data_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            osp.basename(run_folder), # Get just the final directory name
            args_json_string,        # Store all args as a single JSON string
            f"{final_acc:.2f}" if total > 0 else 'N/A', # Format accuracy
            f"{total_loop_time:.2f}",
            f"{UNET_TOTAL_TIME:.4f}",
            format_flops(unet_flops_per_pass) if unet_flops_per_pass >= 0 else 'Failed', # Format FLOPs or indicate failure
            f"{true_minus_min_mean:.6f}" if true_minus_min_errors else 'N/A',  # True-Min Error Mean
            f"{true_minus_min_std:.6f}" if true_minus_min_errors else 'N/A',   # True-Min Error Std  
            f"{len(true_minus_min_errors)}" if true_minus_min_errors else '0',  # True-Min Error Count
            coco_mean_ap_csv,        # COCO mAP
            coco_mean_p1_csv,        # COCO Precision@1
            coco_mean_r5_csv,        # COCO Recall@5
            coco_count_csv,          # COCO mAP Count
            results_txt_path # Path to the detailed log for this run
        ]
        # Check if UNET_TOTAL_TIME is greater than 0 before writing to CSV
        # This ensures we only log successful runs
        if UNET_TOTAL_TIME>100:
            file_exists = osp.exists(summary_csv_path)
            with open(summary_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header only if file is new or empty
                if not file_exists or os.path.getsize(summary_csv_path) == 0:
                    writer.writerow(headers)
                    print(f"Created header row in {summary_csv_path}")
                writer.writerow(data_row)
            print(f"Successfully appended summary to {summary_csv_path}")

    except Exception as e:
        print(f"ERROR: Failed to write to summary CSV file {summary_csv_path}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
