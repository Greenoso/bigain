import torch
from diffusers import AutoencoderKL, DiTPipeline
from diffusers.models import DiTTransformer2DModel

from DeepCache import DeepCacheSDHelper
import tomesd.tomesd as tomesd
import tomesd.tomesd.patch_dit
#import agentsd
import ImprovedTokenMerge.utils
from ImprovedTokenMerge.dit_tome_adapter import apply_tome_to_dit_pipeline, remove_dit_tome_patch
from SiTo.dit_sito_adapter import patch_dit_sito, remove_dit_sito_patch



DIT_MODEL_IDS = {
    '256': "facebook/DiT-XL-2-256",
    '512': "facebook/DiT-XL-2-512"
}



def get_dit_model(args):
    """Load DiT model with acceleration methods applied"""
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError

    # Check if ToCa is enabled and we're using multi-step denoising
    if hasattr(args, 'if_toca') and args.if_toca == 1 and hasattr(args, 'eval_denoise_steps') and args.eval_denoise_steps > 1:
        # Import create_toca_dit_model from the eval script
        import sys
        import os.path as osp
        eval_script_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
        if eval_script_path not in sys.path:
            sys.path.append(eval_script_path)
        
        try:
            # Load VAE and scheduler from diffusers, but use ToCa transformer
            model_size = str(args.img_size)
            assert model_size in DIT_MODEL_IDS.keys(), f"Unsupported image size {args.img_size} for DiT"
            model_id = DIT_MODEL_IDS[model_size]
            
            # Load only VAE and scheduler from diffusers
            pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=dtype, config=get_scheduler_config(args))
            vae = pipe.vae
            scheduler = pipe.scheduler
            
            # Load ToCa transformer
            from eval_prob_adaptive_dit import create_toca_dit_model
            transformer = create_toca_dit_model(args, device="cuda" if torch.cuda.is_available() else "cpu")
            
            print("Using ToCa-enabled DiT transformer for multi-step denoising")
            
        except Exception as e:
            print(f"Warning: Failed to load ToCa model ({e}), falling back to standard DiT")
            # Fallback to standard loading
            model_size = str(args.img_size)
            model_id = DIT_MODEL_IDS[model_size]
            pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=dtype, config=get_scheduler_config(args))
            vae = pipe.vae
            transformer = pipe.transformer
            scheduler = pipe.scheduler
    else:
        # Standard DiT loading
        model_size = str(args.img_size)
        assert model_size in DIT_MODEL_IDS.keys(), f"Unsupported image size {args.img_size} for DiT"
        model_id = DIT_MODEL_IDS[model_size]
        
        # Load DiT pipeline
        pipe = DiTPipeline.from_pretrained(model_id, torch_dtype=dtype, config=get_scheduler_config(args))
        
        # Extract components
        vae = pipe.vae
        transformer = pipe.transformer  # This is the DiT model
        scheduler = pipe.scheduler
    
    # Apply acceleration methods to transformer (equivalent to unet in SD)
    if args.if_token_merging == 1:
        tomesd.tomesd.patch_dit.apply_patch(
            transformer,
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
            # Block-level control
            block_tome_flags=getattr(args, 'block_tome_flags_parsed', None),
            # Block-level caching parameters
            cache_merge_functions=bool(getattr(args, 'cache_merge_functions', 0)),
            cache_recalc_interval=getattr(args, 'cache_recalc_interval', 4),
            # ABP (Adaptive Block Pooling) parameters
            merge_method=getattr(args, 'merge_method_alg', 'bipartite'),
            abp_scorer=getattr(args, 'abp_scorer', None),
            abp_tile_aggregation=getattr(args, 'abp_tile_aggregation', 'max')
        )
        # Check if block-level control is being used
        block_flags = getattr(args, 'block_tome_flags_parsed', None)
        if block_flags is not None:
            enabled_blocks = sum(block_flags)
            total_blocks = len(block_flags)
            block_control_info = f"  Block-level Control: {enabled_blocks}/{total_blocks} blocks enabled\n"
        else:
            block_control_info = "  Block-level Control: All blocks enabled (default)\n"
            
        merge_method_used = getattr(args, 'merge_method_alg', 'bipartite')
        abp_info = ""
        if merge_method_used == 'abp':
            abp_scorer_name = getattr(args, 'abp_scorer', None)
            abp_scorer_display = abp_scorer_name.get_name() if hasattr(abp_scorer_name, 'get_name') else 'SpatialFilterScorer (default)'
            abp_info = (f"  ABP Scorer:       {abp_scorer_display}\n"
                       f"  ABP Tile Aggregation: {getattr(args, 'abp_tile_aggregation', 'max')}\n")
        
        print(
            f"Token merging applied to DiT with the following settings:\n"
            f"  Method:           {getattr(args, 'token_merging_method', 'mean')}\n"
            f"  Merge Algorithm:  {merge_method_used}\n" + abp_info +
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
            f"  Proportional Attention: {bool(getattr(args, 'if_proportional_attention', 0))}\n"
            f"  Cache Merge Functions: {bool(getattr(args, 'cache_merge_functions', 0))}\n"
            f"  Cache Recalc Interval: {getattr(args, 'cache_recalc_interval', 4)}\n"
            + block_control_info
        )

    elif getattr(args, 'if_scoring_merge', 0) == 1:
        # Create the appropriate scorer based on method
        import sys
        import os
        tomesd_path = os.path.join(os.path.dirname(__file__), '..', 'tomesd')
        if tomesd_path not in sys.path:
            sys.path.insert(0, tomesd_path)
        from tomesd.tomesd.scoring import (
            FrequencyScorer, SpatialFilterScorer, StatisticalScorer,
            SignalProcessingScorer, SpatialDistributionScorer
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
        else:
            raise ValueError(f"Unknown scoring method: {args.scoring_method}")
        
        # Prepare scorer kwargs for timestep scheduling if needed
        scorer_kwargs = {}
        if hasattr(args, 'timestep_normalized'):
            scorer_kwargs['timestep_normalized'] = args.timestep_normalized
        
        # Apply scoring-based token merging to DiT transformer
        tomesd.tomesd.patch_dit.apply_patch(
            transformer,
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
            # Block-level control
            block_tome_flags=getattr(args, 'block_tome_flags_parsed', None),
            # Block-level caching parameters
            cache_merge_functions=bool(getattr(args, 'cache_merge_functions', 0)),
            cache_recalc_interval=getattr(args, 'cache_recalc_interval', 4),
            # ABP (Adaptive Block Pooling) parameters
            merge_method=getattr(args, 'merge_method_alg', 'bipartite'),
            abp_scorer=getattr(args, 'abp_scorer', None),
            abp_tile_aggregation=getattr(args, 'abp_tile_aggregation', 'max')
        )

        # Check if block-level control is being used for scoring merge
        block_flags = getattr(args, 'block_tome_flags_parsed', None)
        if block_flags is not None:
            enabled_blocks = sum(block_flags)
            total_blocks = len(block_flags)
            block_control_info = f"  Block-level Control: {enabled_blocks}/{total_blocks} blocks enabled\n"
        else:
            block_control_info = "  Block-level Control: All blocks enabled (default)\n"

        merge_method_used = getattr(args, 'merge_method_alg', 'bipartite')
        abp_info = ""
        if merge_method_used == 'abp':
            abp_scorer_name = getattr(args, 'abp_scorer', None)
            abp_scorer_display = abp_scorer_name.get_name() if hasattr(abp_scorer_name, 'get_name') else 'SpatialFilterScorer (default)'
            abp_info = (f"  ABP Scorer:       {abp_scorer_display}\n"
                       f"  ABP Tile Aggregation: {getattr(args, 'abp_tile_aggregation', 'max')}\n")
        
        print(
            f"Scoring-based token merging applied to DiT with the following settings:\n"
            f"  Merging Method:   {getattr(args, 'token_merging_method', 'mean')}\n"
            f"  Merge Algorithm:  {merge_method_used}\n" + abp_info +
            f"  Scoring Method:   {args.scoring_method}\n"
            f"  Scorer:           {scorer.get_name()}\n"
            f"  Ratio:            {args.token_merging_ratio}\n"
            f"  Preserve Ratio:   {args.scoring_preserve_ratio}\n"
            f"  Score Mode:       {args.scoring_mode}\n"
            f"  Preserve Spatial Uniformity: {bool(getattr(args, 'scoring_preserve_spatial_uniformity', 0))}\n"
            f"  Score-guided Destination Selection: {bool(getattr(args, 'if_low_frequency_dst_tokens', 0))}\n"
            f"  Stride X (sx):    {args.token_merging_sx}\n"
            f"  Stride Y (sy):    {args.token_merging_sy}\n"
            f"  Max Downsample:   {args.token_merging_max_downsample}\n"
            f"  Single Downsample Level: {args.token_merging_single_downsample_level_merge}\n"
            f"  Cache Indices Per Image: {bool(getattr(args, 'token_merging_cache_indices_per_image', 0))}\n"
            f"  Use Random:       {bool(args.token_merging_use_rand)}\n"
            f"  Merge Attn:       {bool(getattr(args, 'merge_attn', 1))}\n"
            f"  Merge Cross-Attn: {bool(getattr(args, 'merge_crossattn', 0))}\n"
            f"  Merge MLP:        {bool(getattr(args, 'merge_mlp', 0))}\n"
            f"  Proportional Attention: {bool(getattr(args, 'if_proportional_attention', 0))}\n"
            f"  Cache Merge Functions: {bool(getattr(args, 'cache_merge_functions', 0))}\n"
            f"  Cache Recalc Interval: {getattr(args, 'cache_recalc_interval', 4)}\n"
            + block_control_info
        )

    elif args.if_attention_proc == 1:
        # Import the DiT adapter functions
        
        # Remove any existing patches
        if hasattr(pipe, 'transformer'):
            # This is DiT - use DiT-specific removal
            remove_dit_tome_patch(pipe.transformer)
        else:
            # This is U-Net - use existing removal
            raise NotImplementedError("This is not a DiT model, cannot remove DiT-specific patch.")
        
        # Prepare token merge arguments
        token_merge_args = {
            "ratio": args.token_merging_ratio,
            "sx": args.token_merging_sx,  
            "sy": args.token_merging_sy,  
            "use_rand": bool(args.token_merging_use_rand), 
            "merge_tokens": args.merge_tokens,
            "merge_method": args.merge_method,
            "downsample_method": args.downsample_method,
            "downsample_factor": args.downsample_factor,
            # Custom block downsample parameters for DiT
            "downsample_factor_h": getattr(args, 'downsample_factor_h', 1),
            "downsample_factor_w": getattr(args, 'downsample_factor_w', 2),
            # Linear blend parameters for downsample methods
            "blend_factor": getattr(args, 'blend_factor', None),
            "blend_method_1": getattr(args, 'blend_method_1', None),
            "blend_method_2": getattr(args, 'blend_method_2', None),
            "timestep_threshold_switch": args.timestep_threshold_switch,
            "timestep_threshold_stop": args.timestep_threshold_stop,
            "secondary_merge_method": args.secondary_merge_method,
            "frequency_selection_mode": args.frequency_selection_mode,
            "frequency_selection_method": args.frequency_selection_method,
            "frequency_ranking_method": args.frequency_ranking_method,
            "selection_source": getattr(args, 'selection_source', 'hidden'),  # Add if not in args
            "frequency_grid_alpha": getattr(args, 'frequency_grid_alpha', 2.0),  # Add if not in args
            "block_tome_flags": getattr(args, 'block_tome_flags_parsed', None),  # Add block flags
        }
        
        # Apply the appropriate patching based on architecture
        if hasattr(pipe, 'transformer'):
            # This is DiT - use the DiT adapter
            apply_tome_to_dit_pipeline(pipe, token_merge_args)
            
            print(
                f"Attention processor patch applied to DiT with settings:\n"
                f"  Ratio:               {args.token_merging_ratio}\n"
                f"  Stride X (sx):       {args.token_merging_sx}\n"
                f"  Stride Y (sy):       {args.token_merging_sy}\n"
                f"  Use Random:          {bool(args.token_merging_use_rand)}\n"
                f"  Merge Tokens:        {args.merge_tokens}\n"
                f"  Merge Method:        {args.merge_method}\n"
                f"  Downsample Method:   {args.downsample_method}\n"
                f"  Downsample Factor:   {args.downsample_factor}\n"
                f"  Switch Threshold:    {args.timestep_threshold_switch}\n"
                f"  Stop Threshold:      {args.timestep_threshold_stop}\n"
                f"  Secondary Method:    {args.secondary_merge_method}\n"
                f"  Frequency Selection mode:   {args.frequency_selection_mode}\n"
                f"  Frequency Selection method: {args.frequency_selection_method}\n"
                f"  Frequency Ranking method:   {args.frequency_ranking_method}\n"
            )
        else:
            # This is U-Net - use the existing patch method
            # Add level-specific parameters for U-Net
            token_merge_args["downsample_factor_level_2"] = args.downsample_factor_level_2
            token_merge_args["ratio_level_2"] = args.ratio_level_2
            
            ImprovedTokenMerge.utils.patch_attention_proc(pipe.unet, token_merge_args=token_merge_args)
            
            print(
                f"Attention processor patch applied to U-Net with settings:\n"
                f"  Ratio:               {args.token_merging_ratio}\n"
                f"  Stride X (sx):       {args.token_merging_sx}\n"
                f"  Stride Y (sy):       {args.token_merging_sy}\n"
                f"  Use Random:          {bool(args.token_merging_use_rand)}\n"
                f"  Merge Tokens:        {args.merge_tokens}\n"
                f"  Merge Method:        {args.merge_method}\n"
                f"  Downsample Method:   {args.downsample_method}\n"
                f"  Downsample Factor:   {args.downsample_factor}\n"
                f"  Switch Threshold:    {args.timestep_threshold_switch}\n"
                f"  Stop Threshold:      {args.timestep_threshold_stop}\n"
                f"  Secondary Method:    {args.secondary_merge_method}\n"
                f"  L2 Downsample:       {args.downsample_factor_level_2}\n"
                f"  L2 Ratio:            {args.ratio_level_2}\n"
                f"  Frequency Selection mode:   {args.frequency_selection_mode}\n"
                f"  Frequency Selection method: {args.frequency_selection_method}\n"
                f"  Frequency Ranking method:   {args.frequency_ranking_method}\n"
            )
    elif args.if_sito == 1:
        # Apply SiTo token pruning to DiT transformer
        
        # Prepare SiTo arguments
        sito_args = {
            "prune_ratio": args.sito_prune_ratio,
            "max_downsample_ratio": args.sito_max_downsample_ratio,
            "prune_selfattn_flag": bool(getattr(args, 'sito_prune_selfattn', 1)),
            "prune_crossattn_flag": bool(getattr(args, 'sito_prune_crossattn', 0)),
            "prune_mlp_flag": bool(getattr(args, 'sito_prune_mlp', 0)),
            "sx": args.sito_sx,
            "sy": args.sito_sy,
            "noise_alpha": args.sito_noise_alpha,
            "sim_beta": args.sito_sim_beta,
            "block_sito_flags": getattr(args, 'block_sito_flags_parsed', None) or [1] * 28
        }
        
        # Apply SiTo patch to the transformer
        patch_dit_sito(transformer, sito_args)
        
        # Check if block-level control is being used
        block_flags = sito_args["block_sito_flags"]
        if block_flags is not None:
            enabled_blocks = sum(block_flags)
            total_blocks = len(block_flags)
            block_control_info = f"  Block-level Control: {enabled_blocks}/{total_blocks} blocks enabled\n"
        else:
            block_control_info = "  Block-level Control: All blocks enabled (default)\n"
        
        print(
            f"SiTo token pruning applied to DiT with the following settings:\n"
            f"  Prune Ratio:          {args.sito_prune_ratio}\n"
            f"  Max Downsample Ratio: {args.sito_max_downsample_ratio}\n"
            f"  Stride X (sx):        {args.sito_sx}\n"
            f"  Stride Y (sy):        {args.sito_sy}\n"
            f"  Noise Alpha:          {args.sito_noise_alpha}\n"
            f"  Similarity Beta:      {args.sito_sim_beta}\n"
            f"  Prune Self-Attn:      {bool(getattr(args, 'sito_prune_selfattn', 1))}\n"
            f"  Prune Cross-Attn:     {bool(getattr(args, 'sito_prune_crossattn', 0))}\n"
            f"  Prune MLP:            {bool(getattr(args, 'sito_prune_mlp', 0))}\n"
            + block_control_info
        )
    else:
        print("No token merging applied.")
    
    # Integrate DeepCache if enabled
    if args.if_deepcache == 1:
        try:
            from DeepCache import DeepCacheSDHelper
            helper = DeepCacheSDHelper(pipe=pipe)
            helper.set_params(
                cache_interval=args.cache_interval,
                cache_branch_id=args.cache_branch_id,
            )
            helper.enable()
            print(f"DeepCache applied to DiT with cache_interval={args.cache_interval} and cache_branch_id={args.cache_branch_id}")
        except Exception as e:
            print(f"Warning: DeepCache could not be applied to DiT: {e}")
    else:
        print("DeepCache not applied to DiT")
    
    return vae, None, None, transformer, scheduler  # Return None for tokenizer and text_encoder

def get_scheduler_config(args):

    config = {
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


    return config