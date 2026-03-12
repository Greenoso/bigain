"""
Unified Masked Attention Handler

This module consolidates all masked attention logic into a single, configurable handler
to eliminate the repetitive code patterns in the original merge.py file.
"""

import torch
import math
from typing import Optional, Tuple, Callable, Dict, Any
from enum import Enum

try:
    from .frequency_selection import (
        frequency_based_token_mask,
        frequency_based_token_mask_blockwise
    )
except ImportError:
    from frequency_selection import (
        frequency_based_token_mask,
        frequency_based_token_mask_blockwise
    )


class MaskSelectionMethod(Enum):
    """Enumeration of different mask selection methods."""
    FREQUENCY = "frequency"
    SNR = "snr"
    NOISE_MAGNITUDE = "noise_magnitude"


class MaskMode(Enum):
    """Enumeration of mask computation modes."""
    GLOBAL = "global"       # Uses config_ratio, calls frequency_based_token_mask
    BLOCKWISE = "blockwise" # Uses config_downsample_factor, calls frequency_based_token_mask_blockwise


class MaskComputationType(Enum):
    """Enumeration of mask computation types."""
    PRECOMPUTED = "precomputed"  # Uses stored masks
    ON_THE_FLY = "on_the_fly"    # Computes masks dynamically


class MaskedAttentionConfig:
    """Configuration for masked attention computation."""
    
    def __init__(self, merge_method: str):
        """Initialize config based on merge method name."""
        self.selection_method = self._parse_selection_method(merge_method)
        self.mode = self._parse_mode(merge_method)
        self.computation_type = self._parse_computation_type(merge_method)
    
    def _parse_selection_method(self, merge_method: str) -> MaskSelectionMethod:
        """Parse selection method from merge method name."""
        if "snr" in merge_method:
            return MaskSelectionMethod.SNR
        elif "noise_magnitude" in merge_method:
            return MaskSelectionMethod.NOISE_MAGNITUDE
        else:
            return MaskSelectionMethod.FREQUENCY
    
    def _parse_mode(self, merge_method: str) -> MaskMode:
        """Parse mode from merge method name."""
        if "blockwise" in merge_method:
            return MaskMode.BLOCKWISE
        else:
            return MaskMode.GLOBAL
    
    def _parse_computation_type(self, merge_method: str) -> MaskComputationType:
        """Parse computation type from merge method name."""
        if self.selection_method == MaskSelectionMethod.FREQUENCY:
            return MaskComputationType.PRECOMPUTED
        else:
            return MaskComputationType.ON_THE_FLY


def should_apply_masked_attention(cur_level: str, x_shape: torch.Size, config: MaskedAttentionConfig, 
                                config_ratio: float, config_downsample_factor: int) -> bool:
    """Check if masked attention should be applied based on conditions."""
    # Only apply to level_1 with exactly 4096 tokens
    if cur_level != "level_1" or x_shape[1] != 4096:
        return False
    
    # Check method-specific conditions
    if config.mode == MaskMode.GLOBAL:
        return config_ratio > 0.0 and config_ratio < 1.0
    else:  # BLOCKWISE
        return config_downsample_factor > 1


def get_mask_key(config: MaskedAttentionConfig, cur_level: str) -> str:
    """Generate the appropriate mask key for storage."""
    if config.computation_type == MaskComputationType.PRECOMPUTED:
        if config.mode == MaskMode.BLOCKWISE:
            return f"fixed_mask_blockwise_level_{cur_level}"
        else:
            return f"fixed_mask_level_{cur_level}"
    else:
        # For on-the-fly computation, we don't store masks with keys
        return ""


def validate_on_the_fly_requirements(config: MaskedAttentionConfig, tome_info: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that required data is available for on-the-fly computation."""
    if config.computation_type != MaskComputationType.ON_THE_FLY:
        return True, ""
    
    if config.selection_method == MaskSelectionMethod.SNR:
        if 'current_clean_image' not in tome_info or 'current_noise' not in tome_info:
            return False, "No current noise/clean image available for on-the-fly SNR mask computation"
    
    elif config.selection_method == MaskSelectionMethod.NOISE_MAGNITUDE:
        if 'current_noise' not in tome_info:
            return False, "No current noise available for on-the-fly noise magnitude mask computation"
    
    return True, ""


def compute_mask_on_the_fly(x: torch.Tensor, config: MaskedAttentionConfig, tome_info: Dict[str, Any], 
                           config_ratio: float, config_downsample_factor: int, cur_h: int, cur_w: int) -> torch.Tensor:
    """Compute mask on-the-fly for SNR and noise magnitude methods."""
    args = tome_info['args']
    current_batch_size = x.shape[0]
    
    # Get common parameters
    selection_mode = args.get("frequency_selection_mode", "high")
    ranking_method = args.get("frequency_ranking_method", "amplitude")
    
    # Handle timestep information
    timestep_normalized = None
    if 'timesteps_batch' in tome_info and tome_info['timesteps_batch'] is not None:
        batch_timesteps = tome_info['timesteps_batch']
        timestep_normalized = batch_timesteps.float() / 1000.0
    elif 'timestep' in tome_info:
        timestep_normalized = tome_info['timestep'] / 1000.0
    
    # Prepare clean signal and noise
    current_clean = None
    current_noise = None
    
    if config.selection_method == MaskSelectionMethod.SNR:
        current_clean = tome_info['current_clean_image']
        current_noise = tome_info['current_noise']
        
        if current_clean.shape[0] == 1 and current_batch_size > 1:
            current_clean = current_clean.expand(current_batch_size, -1, -1)
    
    elif config.selection_method == MaskSelectionMethod.NOISE_MAGNITUDE:
        current_noise = tome_info['current_noise']
        current_clean = tome_info.get('current_clean_image')
        
        if current_clean is not None and current_clean.shape[0] == 1 and current_batch_size > 1:
            current_clean = current_clean.expand(current_batch_size, -1, -1)
    
    # Compute mask based on mode
    if config.mode == MaskMode.BLOCKWISE:
        # Compute H, W from token count (assuming square grid)
        sqrt_tokens = int(math.sqrt(x.shape[1]))
        H, W = sqrt_tokens, sqrt_tokens
        
        mask = frequency_based_token_mask_blockwise(
            x,
            downsampling_factor=config_downsample_factor,
            H=H, W=W,
            selection_method=config.selection_method.value,
            mode=selection_mode,
            ranking_method=ranking_method,
            timestep_normalized=timestep_normalized,
            clean_signal=current_clean,
            noise=current_noise
        )
    else:  # GLOBAL
        mask = frequency_based_token_mask(
            x,
            reduction_ratio=config_ratio,
            selection_method=config.selection_method.value,
            mode=selection_mode,
            ranking_method=ranking_method,
            H=None, W=None,
            timestep_normalized=timestep_normalized,
            clean_signal=current_clean,
            noise=current_noise
        )
    
    return mask


def handle_precomputed_mask(config: MaskedAttentionConfig, tome_info: Dict[str, Any], 
                          x: torch.Tensor, cur_level: str) -> Optional[torch.Tensor]:
    """Handle precomputed mask retrieval and validation."""
    mask_key = get_mask_key(config, cur_level)
    
    if mask_key not in tome_info:
        print(f"ERROR: No precomputed {config.mode.value} mask found during evaluation!")
        print("   Masks should be precomputed before evaluation starts.")
        return None
    
    # Use the precomputed mask
    log_key = f'_mask_{config.mode.value}_load_logged_global'
    if not tome_info.get(log_key, False):
        print(f"Using precomputed fixed {config.mode.value} mask from original clean image")
        tome_info[log_key] = True
    
    stored_mask = tome_info[mask_key]
    
    # Validate mask dimensions
    if stored_mask.shape[1] != x.shape[1]:
        print(f"Warning: Stored {config.mode.value} mask size ({stored_mask.shape[1]}) doesn't match current layer size ({x.shape[1]}). Skipping masked attention for this layer.")
        return None
    
    # Expand mask to match current batch size
    current_batch_size = x.shape[0]
    stored_batch_size = stored_mask.shape[0]
    
    if current_batch_size == stored_batch_size:
        mask = stored_mask
    else:
        mask = stored_mask.repeat(current_batch_size, 1)
        expand_log_key = f'_batch_{config.mode.value}_expand_logged'
        if not tome_info.get(expand_log_key, False):
            print(f"Expanded {config.mode.value} mask from batch size {stored_batch_size} to {current_batch_size}")
            tome_info[expand_log_key] = True
    
    return mask


def log_masked_attention_skip(config: MaskedAttentionConfig, cur_level: str, x_shape: torch.Size):
    """Log why masked attention is being skipped."""
    if cur_level != "level_1":
        print(f"Skipping {config.selection_method.value} {config.mode.value} masked attention for {cur_level} (only applies to level_1)")
    elif x_shape[1] != 4096:
        print(f"Skipping {config.selection_method.value} {config.mode.value} masked attention for {x_shape[1]}-token layer (only applies to 4096-token layers)")


def handle_masked_attention(active_merge_method: str, x: torch.Tensor, tome_info: Dict[str, Any],
                          cur_level: str, config_ratio: float, config_downsample_factor: int,
                          cur_h: int, cur_w: int) -> Tuple[Callable, Callable]:
    """
    Unified handler for all masked attention methods.
    
    Returns:
        Tuple of (merge_fn, unmerge_fn) - both will be do_nothing for masked attention
    """
    try:
        from .merge import do_nothing  # Import here to avoid circular imports
    except ImportError:
        from merge import do_nothing
    
    # Parse configuration from method name
    config = MaskedAttentionConfig(active_merge_method)
    
    # Check if masked attention should be applied
    if not should_apply_masked_attention(cur_level, x.shape, config, config_ratio, config_downsample_factor):
        log_masked_attention_skip(config, cur_level, x.shape)
        return do_nothing, do_nothing
    
    try:
        mask = None
        
        if config.computation_type == MaskComputationType.PRECOMPUTED:
            # Handle precomputed masks
            mask = handle_precomputed_mask(config, tome_info, x, cur_level)
        else:
            # Handle on-the-fly computation
            valid, error_msg = validate_on_the_fly_requirements(config, tome_info)
            if not valid:
                log_key = f'_mask_{config.selection_method.value}_{config.mode.value}_missing_logged'
                if not tome_info.get(log_key, False):
                    print(f"ERROR: {error_msg}")
                    tome_info[log_key] = True
                return do_nothing, do_nothing
            
            mask = compute_mask_on_the_fly(x, config, tome_info, config_ratio, config_downsample_factor, cur_h, cur_w)
            
            # Log successful computation
            log_key = f'_{config.selection_method.value}_{config.mode.value}_mask_computed_logged'
            if not tome_info.get(log_key, False):
                print(f"Computing {config.selection_method.value} {config.mode.value} mask on-the-fly per image-noise pair")
                tome_info[log_key] = True
        
        if mask is not None:
            # Store the mask for use in attention processor
            tome_info['token_mask'] = mask
        
        return do_nothing, do_nothing
    
    except Exception as e:
        print(f"Error in {config.selection_method.value} {config.mode.value} masked attention: {e}")
        return do_nothing, do_nothing
