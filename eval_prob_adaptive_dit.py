# Standard Library
import argparse
import os
import os.path as osp
import time
import random
import json
from datetime import datetime
import csv

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
from diffusion.models_dit import get_dit_model, get_scheduler_config  # Modified import
from diffusion.utils import LOG_DIR, get_formatstr

# to record the time and computational resources for the transformer
TRANSFORMER_TOTAL_TIME = 0.0   # cumulative time

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

### Helper function to format FLOPs
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

    # For deterministic behavior in cuDNN-based operations
    torch.backends.cudnn.deterministic = True
    # Disabling benchmark mode helps reduce variance; might impact speed
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

def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)

def init_toca_cache(args, num_denoise_steps):
    """Initialize ToCa cache system for multi-step denoising"""
    try:
        import sys
        import os.path as osp
        toca_path = osp.join(osp.dirname(osp.abspath(__file__)), 'ToCa', 'DiT-ToCa')
        if toca_path not in sys.path:
            sys.path.append(toca_path)
        from cache_functions import cache_init
        
        model_kwargs = {
            'cache_type': args.toca_cache_type,
            'ratio_scheduler': args.toca_ratio_scheduler,
            'fresh_ratio': args.toca_fresh_ratio,
            'fresh_threshold': args.toca_fresh_threshold,
            'force_fresh': 'global',  # Use global force fresh strategy
            'soft_fresh_weight': 0.1,  # Default weight for cache frequency score
            'test_FLOPs': False  # Set to True if you want FLOP counting
        }
        
        cache_dic, current = cache_init(model_kwargs, num_denoise_steps)
        return cache_dic, current
    except ImportError as e:
        raise ImportError(f"Failed to initialize ToCa cache: {e}")

def create_toca_dit_model(args, device):
    """Create ToCa-enabled DiT model"""
    try:
        import sys
        import os.path as osp
        toca_path = osp.join(osp.dirname(osp.abspath(__file__)), 'ToCa', 'DiT-ToCa')
        if toca_path not in sys.path:
            sys.path.append(toca_path)
        from models import DiT_XL_2, DiT_XL_4
        from download import find_model
        
        # Choose model based on image size
        if args.img_size == 256:
            model_constructor = DiT_XL_2
            model_name = "DiT-XL-2-256x256.pt"
        elif args.img_size == 512:
            model_constructor = DiT_XL_4  # Use XL/4 for 512 to match patch size
            model_name = "DiT-XL-2-512x512.pt"  
        else:
            raise ValueError(f"Unsupported image size {args.img_size} for ToCa DiT")
        
        # Load pretrained weights
        state_dict = find_model(model_name)
        
        # Create model instance
        transformer = model_constructor()
        transformer.load_state_dict(state_dict)
        transformer = transformer.to(device)
        
        # Set dtype
        if args.dtype == 'float16':
            transformer = transformer.half()
        
        print(f"Loaded ToCa-enabled DiT model for {args.img_size}x{args.img_size}")
        return transformer
        
    except Exception as e:
        raise RuntimeError(f"Failed to create ToCa DiT model: {e}")

def eval_prob_adaptive(transformer, latent, class_embeddings, scheduler, args, latent_size=64, all_noise=None):
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    num_classes = len(class_embeddings) # Get total number of classes
    
    if all_noise is None:
        # Determine required noise size based on mode
        if args.single_timestep >= 0:
            # Need noise for all classes * n_trials at the single timestep
            required_noise_samples = args.n_trials
        else:
            # Original adaptive mode: need noise for max_n_samples * n_trials
            max_n_samples = max(args.n_samples)
            required_noise_samples = max_n_samples * args.n_trials
       
        all_noise = torch.randn((required_noise_samples, 4, latent_size, latent_size), device=latent.device)
 
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

    # Single Timestep Evaluation Path
    if args.single_timestep >= 0:
        target_t = args.single_timestep
        if not (0 <= target_t < T):
             raise ValueError(f"--single_timestep must be between 0 and {T-1}, but got {target_t}")

        # Prepare inputs for eval_error for the single timestep across all classes
        ts = [target_t] * (num_classes * args.n_trials)
        class_embed_idxs = [c_idx for c_idx in range(num_classes) for _ in range(args.n_trials)]
        noise_idxs = [trial_idx for _ in range(num_classes) for trial_idx in range(args.n_trials)]

        # Check if generated/loaded noise is sufficient
        if len(all_noise) < max(noise_idxs) + 1:
            raise ValueError(
            f"Need at least {max(noise_idxs)+1} noise samples, found {len(all_noise)}."
            )
        
        # Call eval_error once for all classes at the target timestep
        pred_errors = eval_error(transformer, scheduler, latent, all_noise, ts, noise_idxs,
                                class_embeddings, class_embed_idxs, args.batch_size, args.dtype, args.loss,
                                args.eval_denoise_steps,
                                args.eval_step_stride,
                                args.eval_error_method,
                                args)

        # Calculate the mean error for each class at this timestep
        mean_errors_per_class = []
        data = {} # Store results similarly to the original format
        for c_idx in range(num_classes):
            start_idx = c_idx * args.n_trials
            end_idx = start_idx + args.n_trials
            class_errors = pred_errors[start_idx:end_idx]
            mean_errors_per_class.append(class_errors.mean().item())
            # Store data for this class (only one timestep evaluated)
            data[c_idx] = {'t': torch.tensor([target_t] * args.n_trials, device='cpu'),
                           'pred_errors': class_errors.cpu()} # Store errors on CPU

        # Find the class index with the minimum mean error
        pred_idx = np.argmin(mean_errors_per_class)

        return pred_idx, data

    else:
        # Original adaptive evaluation logic
        data = dict()
        t_evaluated = set()
        remaining_class_idxs = list(range(len(class_embeddings)))
        max_n_samples = max(args.n_samples)
        start = T // max_n_samples // 2
        t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

        for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
            ts = []
            noise_idxs = []
            class_embed_idxs = []
            curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
            curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
            for class_i in remaining_class_idxs:
                for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                    ts.extend([t] * args.n_trials)
                    noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                    class_embed_idxs.extend([class_i] * args.n_trials)
            t_evaluated.update(curr_t_to_eval)

            pred_errors = eval_error(transformer, scheduler, latent, all_noise, ts, noise_idxs,
                                    class_embeddings, class_embed_idxs, args.batch_size, args.dtype, args.loss,
                                    args.eval_denoise_steps,
                                    args.eval_step_stride,
                                    args.eval_error_method,
                                    args)

            # match up computed errors to the data
            for class_i in remaining_class_idxs:
                mask = torch.tensor(class_embed_idxs) == class_i
                class_ts = torch.tensor(ts)[mask]
                class_pred_errors = pred_errors[mask]
                if class_i not in data:
                    data[class_i] = dict(t=class_ts, pred_errors=class_pred_errors)
                else:
                    data[class_i]['t'] = torch.cat([data[class_i]['t'], class_ts])
                    data[class_i]['pred_errors'] = torch.cat([data[class_i]['pred_errors'], class_pred_errors])

            # compute the next remaining idxs
            errors = [-data[class_i]['pred_errors'].mean() for class_i in remaining_class_idxs]
            best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
            remaining_class_idxs = [remaining_class_idxs[i] for i in best_idxs]

        # organize the output
        assert len(remaining_class_idxs) == 1
        pred_idx = remaining_class_idxs[0]

        return pred_idx, data

def eval_error(
        transformer, scheduler, latent, all_noise, ts, noise_idxs,
        class_embeddings, class_embed_idxs,
        batch_size: int = 32,
        dtype: str = "float32",
        loss: str = "l2",
        num_denoise_steps: int = 1,
        step_stride: int = 1,
        eval_error_method: str = "trajectory",
        args=None
):
    """
    Modified eval_error function for DiT models using class conditioning
    
    * num_denoise_steps == 1  ➜ original behaviour (ε̂ vs. ε)
    * num_denoise_steps  > 1  ➜ run k steps, then:
        - eval_error_method == "trajectory": compare x_k vs. ground truth at same timestep
        - eval_error_method == "direct": compare x_k vs. clean x₀
        - eval_error_method == "weighted": like direct but weighted by 1/(1-ᾱ_t) for fair comparison
        - eval_error_method == "clean_signal": compare x_k vs. clean signal component
    """
    import time
    global TRANSFORMER_TOTAL_TIME

    assert len(ts) == len(noise_idxs) == len(class_embed_idxs)
    pred_errors = torch.empty(len(ts), device="cpu")
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    if dtype == "float16":
        alphas_cumprod = alphas_cumprod.half()

    x0_original = latent.to(device, dtype=torch.float16 if dtype == "float16" else torch.float32)

    # Build timetable only if we need it for k-step denoising
    if num_denoise_steps > 1:
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        timesteps_tensor = scheduler.timesteps.to(device)
        timestep_to_idx = {int(t.item()): i for i, t in enumerate(timesteps_tensor)}

    idx = 0
    with torch.inference_mode():
        num_iters = len(ts) // batch_size + int(len(ts) % batch_size != 0)
        for _ in tqdm.trange(num_iters, leave=False):
            this_batch_size = min(batch_size, len(ts) - idx)

            # gather batch data
            batch_ts = torch.tensor(ts[idx: idx+this_batch_size], device=device)
            noise = all_noise[noise_idxs[idx: idx+this_batch_size]].to(device)
            
            # Get class inputs instead of text inputs
            class_input = torch.tensor([class_embeddings[i] for i in class_embed_idxs[idx: idx+this_batch_size]], 
                                     device=device, dtype=torch.long)

            if dtype == "float16":
                noise = noise.half()

            sqrt_alpha_prod = alphas_cumprod[batch_ts].sqrt().view(-1,1,1,1)
            sqrt_one_minus_alpha = (1. - alphas_cumprod[batch_ts]).sqrt().view(-1,1,1,1)
            noised_latent = x0_original*sqrt_alpha_prod + noise*sqrt_one_minus_alpha
            noised_latent = noised_latent.half() if dtype == "float16" else noised_latent
            
            # k = 1 path
            if num_denoise_steps == 1:
                t_input = batch_ts.half() if dtype == "float16" else batch_ts

                torch.cuda.synchronize(); start_time = time.time()
                # DiT model call - use class_labels parameter instead of encoder_hidden_states
                model_output = transformer(noised_latent, t_input, class_labels=class_input, return_dict=False)[0]

                # FIX: DiT outputs both noise prediction and covariance prediction
                # We only need the noise prediction (first half of channels)
                channels = model_output.shape[1]
                if channels == 2 * noise.shape[1]:  # DiT outputs 2C channels (noise + covariance)
                    noise_pred = model_output[:, :channels//2]  # Take first half (noise prediction)
                else:
                    noise_pred = model_output  # Use full output if dimensions match



                torch.cuda.synchronize(); TRANSFORMER_TOTAL_TIME += time.time() - start_time

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

            # k > 1 path
            else:
                # Check if ToCa is enabled
                use_toca = (args is not None and hasattr(args, 'if_toca') and args.if_toca == 1 and 
                           hasattr(transformer, 'blocks') and num_denoise_steps > 1)
                
                if use_toca:
                    # Initialize ToCa cache for this batch
                    try:
                        cache_dic, current = init_toca_cache(args, num_denoise_steps)
                        print(f"ToCa cache initialized for batch with {num_denoise_steps} denoising steps")
                    except Exception as e:
                        print(f"Warning: ToCa initialization failed ({e}), proceeding without ToCa")
                        use_toca = False
                
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
                    # Update ToCa step tracking
                    if use_toca:
                        current['step'] = step
                    
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

                    # Predict noise - DiT model call with ToCa support
                    torch.cuda.synchronize(); start_time = time.time()
                    
                    if use_toca:
                        # ToCa forward pass - returns tensor directly
                        model_output = transformer(scaled_latent, t_input, current, cache_dic, class_input)
                    else:
                        # Standard diffusers forward pass - returns tuple
                        model_output = transformer(scaled_latent, t_input, 
                                        class_labels=class_input, return_dict=False)[0]
                    
                    # Handle DiT output format (may have 2C channels for noise + covariance)
                    channels = model_output.shape[1]
                    if channels == 2 * noise.shape[1]:  # DiT outputs 2C channels (noise + covariance)
                        noise_pred = model_output[:, :channels//2]  # Take first half (noise prediction)
                    else:
                        noise_pred = model_output  # Use full output if dimensions match
                    
                    torch.cuda.synchronize(); TRANSFORMER_TOTAL_TIME += time.time() - start_time

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

            # store & advance
            pred_errors[idx: idx+this_batch_size] = error.detach().cpu()
            idx += this_batch_size
        print(f"[DEBUG] Error range for this batch: min={pred_errors.min().item():.10f}, max={pred_errors.max().item():.10f}, mean={pred_errors.mean().item():.6f},std={pred_errors.var().item():.6f}")

    return pred_errors

def main():
    global TRANSFORMER_TOTAL_TIME
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10', 'food', 'caltech101', 'imagenet',
                                 'objectnet', 'aircraft','coco2017','imagenet100','imagenet100_randomseed0'], help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # model args - simplified for DiT
    parser.add_argument('--img_size', type=int, default=256, choices=[256, 512], help='Image size for DiT model')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    
    # Class conditioning - can use either CSV file or manual specification
    parser.add_argument('--class_csv_path', type=str, default=None, help='Path to CSV file with class IDs (e.g., imagenet100_prompts.csv)')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes to evaluate (default: 1000 for ImageNet)')
    parser.add_argument('--class_subset', type=str, default=None, help='Path to subset of class indices to evaluate')
    
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
    
    # Acceleration method args (same as original)
    parser.add_argument('--if_token_merging', type=int,default=0, help='Whether to apply token merging')
    parser.add_argument('--if_agentsd',type=int,default=0,help='Whether or not to apply agent attention token merging (1 for True, 0 for False)')
    parser.add_argument('--token_merging_ratio', type=float, default=0.5, help='Ratio of tokens to merge')
    parser.add_argument('--agent_ratio', type=float, default=0.5, help='Ratio of agent tokens for agentsd')
    parser.add_argument('--token_merging_use_rand', type=int, default=1, help='if use random for source token merging')
    parser.add_argument('--token_merging_max_downsample',type=int,default=1,choices=[1, 2, 4, 8],
                        help='Apply ToMe to layers with at most this amount of downsampling')
    parser.add_argument('--token_merging_sx',type=int,default=2,help='Stride in the x dimension for computing dst sets')
    parser.add_argument('--token_merging_sy',type=int,default=2,help='Stride in the y dimension for computing dst sets')
    parser.add_argument('--token_merging_single_downsample_level_merge',type=int,default=0,help='Whether only merge the layer with downsample size equal token_merging_max_downsample')
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
    
    parser.add_argument('--token_merging_cache_indices_per_image',type=int,default=0,help='Whether to cache merge indices per image (1 for True, 0 for False)')

    ## for Block-level Caching (DiT scoring merge acceleration)
    parser.add_argument('--cache_merge_functions', type=int, default=0,
                        help='Whether to cache merge/unmerge functions for blocks within a UNet call (1 for True, 0 for False). '
                             'Only works with scoring-based merging and block-level control. Provides speedup by reusing '
                             'expensive scoring computations across nearby blocks.')
    parser.add_argument('--cache_recalc_interval', type=int, default=4,
                        help='Recalculate cached merge functions every N blocks (default: 4). '
                             'Higher values = more reuse = faster but potentially lower quality. '
                             'Only used when --cache_merge_functions=1.')

    ## for Proportional Attention
    parser.add_argument('--if_proportional_attention', type=int, default=0,
                        help='Whether to use proportional attention that accounts for token sizes (1 for True, 0 for False)')

    ## for ABP (Adaptive Block Pooling) 
    parser.add_argument('--merge_method_alg', type=str, default='bipartite',
                        choices=['bipartite', 'abp'],
                        help='Merging algorithm to use. Options: \"bipartite\" (standard bipartite matching), '
                             '\"abp\" (Adaptive Block Pooling - tile-based merging with scoring)')
    parser.add_argument('--abp_scorer', type=str, default=None,
                        help='ABP scorer class name for tile evaluation. Options: \"FrequencyScorer\", '
                             '\"SpatialFilterScorer\", \"StatisticalScorer\", \"SignalProcessingScorer\", '
                             '\"SpatialDistributionScorer\". Only used when merge_method_alg=\"abp\"')
    parser.add_argument('--abp_tile_aggregation', type=str, default='max',
                        choices=['max', 'min', 'sum', 'std'],
                        help='Tile aggregation method for ABP. Determines how tile scores are '
                             'aggregated for merging decisions')

    ## for Scoring-based Token Merging
    parser.add_argument('--if_scoring_merge', type=int, default=0,
                        help='Whether to apply scoring-based token merging (1 for True, 0 for False)')
    parser.add_argument('--scoring_method', type=str, default='statistical',
                        choices=['frequency', 'spatial_filter', 'statistical', 'signal_processing', 'spatial_distribution'],
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

    # Token Downsampling args (same as original)
    parser.add_argument('--if_attention_proc', type=int, default=0,
                        help='Whether to apply attention processor patching (1 for True, 0 for False)')
    parser.add_argument('--merge_tokens', type=str, default='keys/values',
                        choices=['keys/values', 'all'],
                        help='Which tokens to merge')
    parser.add_argument('--merge_method', type=str, default='downsample',
                        choices=['none', 'downsample_custom_block','similarity', 'downsample', 'frequency_blockwise','frequency_global', 'block_avg_pool', 'downsample_qkv_upsample_out', 'masked_attention', 'blockwise_masked_attention', 'toca'],
                        help='Method to use for merging tokens')
    parser.add_argument('--block_tome_flags', type=str, default=None,
                        help='Comma-separated list of 0s and 1s indicating which transformer blocks should apply ToMe. '
                             'For DiT-XL/2 models, expects 28 values (one per transformer block). '
                             'E.g., "1,1,1,0,0,0,..." to apply to first 3 blocks only. '
                             'If not specified, applies to all blocks.')

    parser.add_argument('--downsample_factor_h', type=int, default=2,
                        help='Downsample factor for height dimension, only used if merge_method is downsample_custom_block')
    parser.add_argument('--downsample_factor_w', type=int, default=2,
                        help='Downsample factor for width dimension, only used if merge_method is downsample_custom_block')

    # Linear blend parameters for downsample methods
    parser.add_argument('--blend_factor', type=float, default=None,
                        help='Blend factor for linear_blend downsample method (0.0=method2, 1.0=method1). None=use 0.5 default')
    parser.add_argument('--blend_method_1', type=str, default=None,
                        help='First method for linear_blend (default: nearest-exact)')
    parser.add_argument('--blend_method_2', type=str, default=None,
                        help='Second method for linear_blend (default: avg_pool)')

    parser.add_argument('--downsample_method', type=str, default='nearest-exact',
                        help='Interpolation method for downsampling')
    parser.add_argument('--downsample_factor', type=int, default=2,
                        help='Factor to downsample tokens by')
    parser.add_argument('--timestep_threshold_switch', type=float, default=0,
                        help='Percentage of generation left to switch to secondary method')
    parser.add_argument('--timestep_threshold_stop', type=float, default=0.0,
                        help='Percentage left to revert to normal attention')
    parser.add_argument('--secondary_merge_method', type=str, default='similarity',
                        choices=['none', 'similarity', 'downsample', 'frequency_blockwise','frequency_global', 'block_avg_pool', 'downsample_qkv_upsample_out'],
                        help='Method to use after threshold switch')
    parser.add_argument('--downsample_factor_level_2', type=int, default=1,
                        help='Downsample amount for down block 2 depth')
    parser.add_argument('--ratio_level_2', type=float, default=0.0,
                        help='Ratio for similarity based merging for down block 2')
    parser.add_argument('--extra_guidance_scale', type=float, default=0,
                        help='Additional guidance scale to compensate for token merging')
    
    # Frequency-specific args
    parser.add_argument('--frequency_selection_mode', type=str, default='high',
                        help='Frequency selection mode for downsampling (high/low/...)')
    parser.add_argument('--frequency_selection_method', type=str, default='1d_dft',
                        choices=['original','1d_dft', '1d_dct', '2d_conv','non_uniform_grid','2d_conv_l2','mean_deviation'],
                        help='Method for selecting frequencies')
    parser.add_argument('--frequency_ranking_method', type=str, default='amplitude',choices=['amplitude', 'spectral_centroid','variance', 'l1norm', 'l2norm', 'mean_deviation'],
                        help='Method for ranking frequencies (e.g., by amplitude). Only used if merge_method is "frequency".')
    parser.add_argument('--selection_source', type=str, default='hidden',
                        choices=['hidden', 'key', 'query', 'value'],
                        help='Source for frequency selection')
    parser.add_argument('--frequency_grid_alpha', type=float, default=2,
                        help='Bias strength parameter for non_uniform_grid (alpha > 1). Default 2.0')


    # DeepCache args
    parser.add_argument('--if_deepcache', type=int, default=0,
                        help='Enable DeepCache caching functionality (set to 1 to enable)')
    parser.add_argument('--cache_interval', type=int, default=1001,
                        help='Cache interval for DeepCache, default always ')
    parser.add_argument('--cache_branch_id', type=int, default=0,
                        help='Cache branch id for DeepCache')
    
    # SiTo acceleration args
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
    
    # ToCa args  
    parser.add_argument('--if_toca', type=int, default=0,
                       help='Enable ToCa acceleration (1 for True, 0 for False)')
    parser.add_argument('--toca_fresh_ratio', type=float, default=0.5,
                       help='ToCa fresh ratio (portion of tokens to recompute)')
    parser.add_argument('--toca_fresh_threshold', type=int, default=3,
                       help='ToCa fresh threshold (interval for full computation)')
    parser.add_argument('--toca_cache_type', type=str, default='attention',
                       choices=['attention', 'random', 'similarity', 'norm', 'kv-norm'],
                       help='ToCa token selection method')
    parser.add_argument('--toca_ratio_scheduler', type=str, default='ToCa-ddim50',
                       choices=['constant', 'linear', 'ToCa-ddim50', 'ToCa-ddpm250'],
                       help='ToCa fresh ratio scheduling strategy')

    # Random seed
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")

    # Attention processor for fair benchmarking
    parser.add_argument('--force_attention_processor', type=str, default='AttnProcessor',
                        choices=['AttnProcessor', 'AttnProcessor2_0', 'XFormersAttnProcessor', 'none'])

    # CSV logging
    parser.add_argument('--summary_csv', type=str, default='classification_experiments_summary.csv',
                        help='Path to the CSV file for logging experiment summaries.')

    # Force recalculation
    parser.add_argument('--force_recalc', action='store_true',
                        help='Force recalculation of errors even if result files already exist')

    # Evaluation args
    parser.add_argument('--eval_denoise_steps', type=int, default=1,
                        help='Number of denoising steps to perform in eval_error. Default=1 (original method).')
   
    parser.add_argument('--single_timestep', type=int, default=-1,
                        help='If set to a non-negative value, evaluate ONLY this specific timestep '
                             'instead of the adaptive multi-timestep strategy. '
                             'Disables --to_keep and --n_samples.')
  
    parser.add_argument('--eval_step_stride', type=int, default=1,
                    help='Stride to use when updating timesteps within the eval_error loop '
                            'if eval_denoise_steps > 1. Default=1 (standard single step).')
    
    parser.add_argument('--eval_error_method', type=str, default='trajectory',
                    choices=['direct', 'trajectory', 'weighted', 'clean_signal'],
                    help='Method for error calculation when eval_denoise_steps > 1. '
                         'trajectory: compare final latent to ground truth at same timestep, '
                         'direct: compare final latent to clean x0, '
                         'weighted: like direct but weighted by noise level, '
                         'clean_signal: compare to clean signal component only.')
      
    parser.add_argument('--qkv_downsample_method', type=str, default='nearest',
                        help='Method for downsampling QKV source (e.g., "avg_pool", "max_pool")')
    parser.add_argument('--out_upsample_method', type=str, default='nearest',
                        help='Method for upsampling attention output (e.g., "nearest", "bilinear")')
    parser.add_argument('--version', type=str, default='dit',
                    help='Model version: keep Stable-Diffusion values (1-5, 2-0, 2-1) or "dit"')
    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)

    # Validate arguments based on single_timestep mode
    if args.single_timestep >= 0:
        print(f"INFO: Running in single timestep mode (t={args.single_timestep}). "
              "Arguments --to_keep and --n_samples will be ignored.")
        args.to_keep = []
        args.n_samples = []
    else:
        # Ensure adaptive args are provided if not in single timestep mode
        if not args.to_keep or not args.n_samples:
            parser.error("Arguments --to_keep and --n_samples are required when --single_timestep is not set.")
        assert len(args.to_keep) == len(args.n_samples), "--to_keep and --n_samples must have the same number of elements for adaptive mode."

    # Validate ToCa arguments
    if args.if_toca == 1:
        if args.eval_denoise_steps <= 1:
            parser.error("ToCa is only supported for eval_denoise_steps > 1")
        
        # Check if ToCa modules are available
        try:
            import sys
            toca_path = osp.join(osp.dirname(osp.abspath(__file__)), 'ToCa', 'DiT-ToCa')
            if toca_path not in sys.path:
                sys.path.append(toca_path)
            from models import DiT_XL_2, DiT_XL_4
            from cache_functions import cache_init
            print("ToCa modules found and imported successfully")
        except ImportError as e:
            parser.error(f"ToCa modules not found: {e}. Please ensure ToCa is properly installed.")

    # Validate token merging method parameter
    if args.token_merging_method != 'mean' and not any([args.if_token_merging, args.if_agentsd, getattr(args, 'if_scoring_merge', 0)]):
        print(f"WARNING: --token_merging_method={args.token_merging_method} specified but no token merging method is enabled. "
              "Consider setting --if_token_merging=1 or --if_scoring_merge=1")
    
    # Apply the seed
    seed_everything(args.seed)

    # Set up class embeddings instead of text embeddings
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

    # Parse block tome flags if provided
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
            tomesd_path = os.path.join(os.path.dirname(__file__), 'tomesd')
            if tomesd_path not in sys.path:
                sys.path.insert(0, tomesd_path)
            from tomesd.tomesd.scoring import (
                FrequencyScorer, SpatialFilterScorer, StatisticalScorer,
                SignalProcessingScorer, SpatialDistributionScorer
            )
            
            scorer_name = args.abp_scorer
            scorer_map = {
                'FrequencyScorer': FrequencyScorer,
                'SpatialFilterScorer': SpatialFilterScorer,
                'StatisticalScorer': StatisticalScorer,
                'SignalProcessingScorer': SignalProcessingScorer,
                'SpatialDistributionScorer': SpatialDistributionScorer
            }
            
            if scorer_name in scorer_map:
                # Create scorer instance with default parameters
                args.abp_scorer = scorer_map[scorer_name]()
                print(f"Successfully created ABP scorer: {scorer_name}")
            else:
                raise ValueError(f"Unknown ABP scorer: {scorer_name}. Available: {list(scorer_map.keys())}")
                
        except ImportError as e:
            print(f"Warning: Could not import ABP scorer modules: {e}")
            print("ABP scorer will remain as string - ensure tomesd path is correct")
        except Exception as e:
            print(f"Warning: Could not create ABP scorer {args.abp_scorer}: {e}")
            print("ABP scorer will remain as string")

    # make run output folder
    name = f"DiT{args.img_size}_{args.n_trials}trials_"
    
    # Add dataset-specific naming
    if args.class_csv_path is not None:
        csv_name = osp.splitext(osp.basename(args.class_csv_path))[0]
        name += f"{csv_name}_"
    elif len(class_indices) != args.num_classes:
        name += f"{len(class_indices)}c_"
    
    if args.single_timestep >= 0:
        name += f'st{args.single_timestep}' # Identify single timestep runs
    else:
        # Original adaptive naming - shortened
        name += '_'.join(map(str, args.to_keep)) + 'k_'
        name += '_'.join(map(str, args.n_samples)) 
    name += f'_seed{args.seed}'
    if args.interpolation != 'bicubic':
        name += f'_{args.interpolation[0:3]}'  # bicubic->bic, bilinear->bil
    if args.loss == 'l1':
        name += '_l1'
    elif args.loss == 'huber':
        name += '_huber'
    if args.img_size != 256:
        name += f'_{args.img_size}'
    
    # Add acceleration method names to folder
    if args.if_token_merging == 1:
        name += f'_tokenmerge{args.token_merging_ratio}_userand{args.token_merging_use_rand}_maxdownsample{args.token_merging_max_downsample}_singlelevel{args.token_merging_single_downsample_level_merge}_sx{args.token_merging_sx}_sy{args.token_merging_sy}_cache{args.token_merging_cache_indices_per_image}'
        # Add method to name if not default
        if args.token_merging_method != 'mean':
            name += f'_method{args.token_merging_method}'
        # Add ABP information if ABP is used
        if getattr(args, 'merge_method_alg', 'bipartite') == 'abp':
            name += f'_abp'
            if getattr(args, 'abp_tile_aggregation', 'max') != 'max':
                name += f'_agg{args.abp_tile_aggregation}'
            if getattr(args, 'abp_scorer', None) is not None:
                scorer_name = args.abp_scorer.__class__.__name__ if hasattr(args.abp_scorer, '__class__') else str(args.abp_scorer)
                name += f'_scorer{scorer_name}'
        if args.if_proportional_attention == 1:
            name += '_pa'
        # Add cache parameters to folder name
        if getattr(args, 'cache_merge_functions', 0) == 1:
            name += f'_cache{args.cache_recalc_interval}'
    elif args.if_agentsd == 1:
        name += f'_agentsd{args.token_merging_ratio}_agentratio{args.agent_ratio}_userand{args.token_merging_use_rand}_maxdownsample{args.token_merging_max_downsample}_singlelevel{args.token_merging_single_downsample_level_merge}_sx{args.token_merging_sx}_sy{args.token_merging_sy}_cache{args.token_merging_cache_indices_per_image}'
        # Add method to name if not default
        if args.token_merging_method != 'mean':
            name += f'_method{args.token_merging_method}'
        # Add ABP information if ABP is used
        if getattr(args, 'merge_method_alg', 'bipartite') == 'abp':
            name += f'_abp'
            if getattr(args, 'abp_tile_aggregation', 'max') != 'max':
                name += f'_agg{args.abp_tile_aggregation}'
            if getattr(args, 'abp_scorer', None) is not None:
                scorer_name = args.abp_scorer.__class__.__name__ if hasattr(args.abp_scorer, '__class__') else str(args.abp_scorer)
                name += f'_scorer{scorer_name}'
        if args.if_proportional_attention == 1:
            name += '_pa'
    elif args.if_scoring_merge == 1:
        name += (f'_scoringmerge{args.token_merging_ratio}'
                f'_method{args.scoring_method}'
                f'_preserve{args.scoring_preserve_ratio}'
                f'_mode{args.scoring_mode}'
                f'_userand{args.token_merging_use_rand}'
                f'_maxdownsample{args.token_merging_max_downsample}'
                f'_sx{args.token_merging_sx}_sy{args.token_merging_sy}'
                f'_cache{args.token_merging_cache_indices_per_image}')
        
        # Add merging method to name if not default
        if args.token_merging_method != 'mean':
            name += f'_mm{args.token_merging_method[:4]}'  # mergemethod -> mm
        
        # Add ABP information if ABP is used
        if getattr(args, 'merge_method_alg', 'bipartite') == 'abp':
            name += f'_abp'
            if getattr(args, 'abp_tile_aggregation', 'max') != 'max':
                name += f'_agg{args.abp_tile_aggregation}'
            if getattr(args, 'abp_scorer', None) is not None:
                scorer_name = args.abp_scorer.__class__.__name__ if hasattr(args.abp_scorer, '__class__') else str(args.abp_scorer)
                name += f'_scorer{scorer_name}'
        
        # Add spatial uniformity parameter to name
        if args.scoring_preserve_spatial_uniformity == 1:
            name += '_spatialuniform'
        
        # Add score-guided destination selection parameter to name
        if args.if_low_frequency_dst_tokens == 1:
            name += '_lfd'  # lowfreqdst -> lfd
        
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
            
        if args.if_proportional_attention == 1:
            name += '_pa'  # propattn -> pa
        # Add cache parameters to folder name  
        if getattr(args, 'cache_merge_functions', 0) == 1:
            name += f'_cache{args.cache_recalc_interval}'
        
        # Add block flags information to folder name if custom flags are used for scoring merge
        if args.block_tome_flags is not None and getattr(args, 'block_tome_flags_parsed', None) is not None:
            # Compress the block flags representation for folder name
            block_tome_flags = args.block_tome_flags_parsed
            num_enabled = sum(block_tome_flags)
            total_blocks = len(block_tome_flags)
            
            # Use shorter hash for any custom block flags pattern
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
        elif args.merge_method == 'masked_attention':
            name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqmethod{args.frequency_selection_method}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_ratio{args.token_merging_ratio}')
            if args.frequency_selection_method=='non_uniform_grid':
                name += f'_gridalpha{args.frequency_grid_alpha}'
        elif args.merge_method == 'blockwise_masked_attention':
            name += (f'_mergemethod{args.merge_method}'
                    f'_freqmode_{args.frequency_selection_mode}'
                    f'_freqmethod{args.frequency_selection_method}'
                    f'_freqrankingmethod{args.frequency_ranking_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_downsample{args.downsample_factor}')
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
                    f'_switch{args.timestep_threshold_switch}'
                     f'_stop{args.timestep_threshold_stop}')
        else:
            name += (f'downsampling{args.token_merging_ratio}_userand{args.token_merging_use_rand}__sx{args.token_merging_sx}_sy{args.token_merging_sy}'
                    f'_method{args.merge_method}'
                    f'_down{args.downsample_factor}_downmethod{args.downsample_method}'
                    f'_selectionsource{args.selection_source}'
                    f'_switch{args.timestep_threshold_switch}'
                    f'_stop{args.timestep_threshold_stop}')
            # Add linear_blend parameter to name if using linear_blend method
            if args.downsample_method == 'linear_blend':
                name += f'_linearblend{getattr(args, "blend_factor", 0.5)}'
                name += f'_blendmethods{getattr(args, "blend_method_1", "nearest-exact")}-{getattr(args, "blend_method_2", "avg_pool")}'
 


        # Add block flags information to folder name if custom flags are used for attention processor
        if args.block_tome_flags is not None and block_tome_flags is not None:
            # Compress the block flags representation for folder name
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
        if args.block_sito_flags is not None and block_sito_flags is not None:
            # Compress the block flags representation for folder name
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
    if args.if_toca == 1:
        name += f'_toca_ratio{args.toca_fresh_ratio}_thresh{args.toca_fresh_threshold}_cache{args.toca_cache_type}_sched{args.toca_ratio_scheduler}'
    if args.eval_denoise_steps > 1:
        name += f'_denoise_steps{args.eval_denoise_steps}'
        if args.eval_step_stride > 1:
             name += f'_stride{args.eval_step_stride}'
        name += f'_error_method{args.eval_error_method}'

    if args.extra is not None:
        run_folder = osp.join(LOG_DIR, args.dataset + '_' + args.extra, name)
    else:
        run_folder = osp.join(LOG_DIR, args.dataset, name)
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')
    print(f'Errors will be saved to: {run_folder}')

    # Store line-by-line logs in "results_log.txt"
    results_txt_path = osp.join(run_folder, "results_log.txt")

    # set up dataset
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    target_dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    

    # load pretrained models - use DiT instead of SD
    vae, _, _, transformer, scheduler = get_dit_model(args)
    vae = vae.to(device)
    transformer = transformer.to(device)
    torch.backends.cudnn.benchmark = True
    
    # Import functions for fixed mask computation if using masked_attention or blockwise_masked_attention
    if args.if_attention_proc == 1 and args.merge_method in ["masked_attention", "blockwise_masked_attention"]:
        from ImprovedTokenMerge.dit_tome_adapter import precompute_fixed_mask_for_dit, reset_dit_fixed_masks

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

    # Measure FLOPs for one DiT forward pass
    transformer_flops_per_pass = 0.0
    print(f"\nMeasuring FLOPs for one DiT forward pass (logical batch = 1)…")

    try:
        b_dummy = 1
        dt = torch.float16 if args.dtype == "float16" else torch.float32
        dummy_latents = torch.randn(
            b_dummy, 4, latent_size, latent_size,  # DiT expects 4 channel input
            device=device, dtype=dt
        )
        dummy_timestep = torch.tensor(
            [scheduler.config.num_train_timesteps // 2],
            device=device
        )
        dummy_class_labels = torch.tensor([0], device=device, dtype=torch.long)  # Class 0

        with FlopCounterMode(transformer, display=True) as fc:
            with torch.no_grad():
                _ = transformer(dummy_latents, dummy_timestep, class_labels=dummy_class_labels, return_dict=False)[0]

        flops_batch1 = fc.get_total_flops()
        transformer_flops_per_pass = flops_batch1 
        print(f"Measured FLOPs/pass (batch={1}): {format_flops(transformer_flops_per_pass)}")

    except Exception as e:
        print("ERROR while counting FLOPs:", e)
        import traceback; traceback.print_exc()
        transformer_flops_per_pass = -1

    finally:
        del dummy_latents, dummy_timestep, dummy_class_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # load noise
    if args.noise_path is not None:
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None

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

    TRANSFORMER_TOTAL_TIME = 0.0
    total_start_time = time.time()
    
    for i in pbar:
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
        fname = osp.join(run_folder, formatstr.format(i) + '.pt')
        if os.path.exists(fname) and not args.force_recalc:
            print('Skipping', i)
            if args.load_stats:
                data = torch.load(fname)
                correct += int(data['predicted_idx_in_class_indices'] == data['label'])
                total += 1
            continue
        elif os.path.exists(fname) and args.force_recalc:
            print(f'Force recalculation enabled - overwriting existing file for sample {i}')
            
        if args.dataset == 'coco2017':
            sample = target_dataset[i]
            image = sample['image']
            label = sample['labels'] 
        else:
            image, label = target_dataset[i]
        
        with torch.no_grad():
            img_input = image.to(device).unsqueeze(0)
            if args.dtype == 'float16':
                img_input = img_input.half()
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= vae.config.scaling_factor
            
            # For masked_attention or blockwise_masked_attention: precompute fixed mask from original clean image
            if args.if_attention_proc == 1 and args.merge_method in ["masked_attention", "blockwise_masked_attention"]:
                # Reset any existing masks from previous images (but keep global logging flags for run-wide logging)
                if hasattr(transformer, '_tome_info') and transformer._tome_info is not None:
                    reset_dit_fixed_masks(transformer._tome_info)
                
                # Precompute the fixed mask from the ORIGINAL CLEAN IMAGE
                # This ensures the mask is based on clean content, not noisy representations
                if hasattr(transformer, '_tome_info') and transformer._tome_info is not None:
                    # Reshape x0 to match attention layer format: (B, C, H, W) -> (B, H*W, C)
                    # For DiT: x0 shape is (1, 4, H, W), need to reshape to (1, H*W, 4)
                    B, C, H, W = x0.shape
                    x0_reshaped = x0.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (1, H*W, 4)
                    
                    # Use block index 0 as a representative for mask computation 
                    # (DiT applies same mask pattern across selected blocks)
                    precompute_fixed_mask_for_dit(x0_reshaped, transformer._tome_info, block_index=0)
                    print(f"Precomputed fixed DiT mask from original clean image for sample {i}")
                else:
                    print("Warning: DiT _tome_info not available for mask precomputation")
            




        pred_idx, pred_errors = eval_prob_adaptive(transformer, x0, class_indices, scheduler, args, latent_size, all_noise)
        # pred_idx is the position in class_indices (0-99)
        # Don't use pred = class_indices[pred_idx] for comparison
        predicted_actual_class_value = class_indices[pred_idx]

        if isinstance(label, list):
            if pred_idx in label:  # Compare positions, not ImageNet indices
                correct += 1
        else:
            if pred_idx == label:  # Compare positions directly
                correct += 1

        # For logging, you can still show the ImageNet class:
        pred_imagenet_class = class_indices[pred_idx]
        save_data = {
            'errors': pred_errors,  # The detailed error dictionary
            'predicted_idx_in_class_indices': pred_idx, # The index of the prediction within your 'class_indices' list
            'predicted_actual_class_value': predicted_actual_class_value, # The actual class value (e.g., ImageNet ID)
            'label': label # The ground truth label for this example
        }
        torch.save(save_data, fname)
                   
        total += 1

    total_end_time = time.time()
    total_loop_time = total_end_time - total_start_time

    print(f"\n--- Evaluation Summary ---")
    print(f"Total wall-clock time for evaluation loop: {total_loop_time:.2f} seconds")
    print(f"Total time spent *only* in DiT forward passes (during eval loop): {TRANSFORMER_TOTAL_TIME:.4f} seconds")
    print(f"Approx. FLOPs per DiT forward pass (batch=1): {format_flops(transformer_flops_per_pass)}")

    if total > 0:
        final_acc = 100.0 * correct / total
        final_acc_str = f"Final accuracy: {final_acc:.2f}%"
        print(final_acc_str)
    else:
        final_acc_str = "No samples processed."

    # Append final summary to results log file
    try:
        with open(results_txt_path, "a") as f:
            f.write("\n=== Final Summary ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(final_acc_str + "\n")
            f.write(f"Total wall-clock time for loop: {total_loop_time:.2f} sec\n")
            f.write(f"Total DiT forward pass time (accumulated in loop): {TRANSFORMER_TOTAL_TIME:.4f} sec\n")
            f.write(f"FLOPs per DiT pass (batch={1}): {format_flops(transformer_flops_per_pass)}\n")
            
            # Add ToCa info if enabled
            if args.if_toca == 1:
                f.write(f"\n=== ToCa Configuration ===\n")
                f.write(f"ToCa enabled: True\n")
                f.write(f"Fresh ratio: {args.toca_fresh_ratio}\n")
                f.write(f"Fresh threshold: {args.toca_fresh_threshold}\n")
                f.write(f"Cache type: {args.toca_cache_type}\n")
                f.write(f"Ratio scheduler: {args.toca_ratio_scheduler}\n")

            f.write("\n=== Command-Line Arguments ===\n")
            sorted_args = sorted(vars(args).items())
            for arg, value in sorted_args:
                f.write(f"{arg}: {value}\n")
            f.write("========================\n")
        print(f"Appended summary to: {results_txt_path}")
    except Exception as e:
        print(f"Error writing final summary to log file {results_txt_path}: {e}")

    # Append summary to the main CSV log file
    summary_csv_path = args.summary_csv
    print(f"Attempting to append results to summary CSV: {summary_csv_path}")
    try:
        headers = [
            'Timestamp', 'Run Folder Name',
            'Args JSON',
            'Accuracy (%)', 'Total Time (s)', 'DiT Eval Time (s)', 'DiT FLOPs/pass (Batch)',
            'Log File Path'
        ]

        args_dict = vars(args)
        args_json_string = json.dumps(args_dict, sort_keys=True)

        data_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            osp.basename(run_folder),
            args_json_string,
            f"{final_acc:.2f}" if total > 0 else 'N/A',
            f"{total_loop_time:.2f}",
            f"{TRANSFORMER_TOTAL_TIME:.4f}",
            format_flops(transformer_flops_per_pass) if transformer_flops_per_pass >= 0 else 'Failed',
            results_txt_path
        ]
        
        if TRANSFORMER_TOTAL_TIME > 0:  # Only log successful runs
            file_exists = osp.exists(summary_csv_path)
            with open(summary_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
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