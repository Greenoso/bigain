import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, \
    EulerDiscreteScheduler

from DeepCache import DeepCacheSDHelper
import sys
import os
# Add local tomesd to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tomesd'))
import tomesd
#import agentsd
import ImprovedTokenMerge.utils
import SiTo

MODEL_IDS = {
    '1-1': "CompVis/stable-diffusion-v1-1",
    '1-2': "CompVis/stable-diffusion-v1-2",
    '1-3': "CompVis/stable-diffusion-v1-3",
    '1-4': "CompVis/stable-diffusion-v1-4",
    '1-5': "runwayml/stable-diffusion-v1-5",
    '2-0': "stabilityai/stable-diffusion-2-base",
    '2-1': "stabilityai/stable-diffusion-2-1-base"
}


def get_sd_model(args):
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    else:
        raise NotImplementedError

    assert args.version in MODEL_IDS.keys()
    model_id = MODEL_IDS[args.version]
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=dtype)



    # Integrate DeepCache if enabled
    if args.if_deepcache == 1:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=args.cache_interval,
            cache_branch_id=args.cache_branch_id,
        )
        helper.enable()
        print(f"DeepCache applied with cache_interval={args.cache_interval} and cache_branch_id={args.cache_branch_id}")
    else:
        print("DeepCache not applied")



    
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    if args.if_token_merging == 1:
        tomesd.apply_patch(
            unet,
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
            # Locality-based sub-block bipartite matching parameters
            locality_block_factor_h=getattr(args, 'locality_block_factor_h', 1),
            locality_block_factor_w=getattr(args, 'locality_block_factor_w', 1)
        )

        # Check if locality-based parameters are being used
        locality_h = getattr(args, 'locality_block_factor_h', 1)
        locality_w = getattr(args, 'locality_block_factor_w', 1)
        locality_info = ""
        if locality_h > 1 or locality_w > 1:
            locality_info = f"\n  Locality-based Sub-block Matching: H factor={locality_h}, W factor={locality_w}"
        
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
            f"  Proportional Attention: {bool(getattr(args, 'if_proportional_attention', 0))}{locality_info}"
        )

    elif getattr(args, 'if_scoring_merge', 0) == 1:
        # Create the appropriate scorer based on method
        import sys
        import os
        tomesd_path = os.path.join(os.path.dirname(__file__), '..', 'tomesd')
        if tomesd_path not in sys.path:
            sys.path.insert(0, tomesd_path)
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
            # sys.path already added above
            from tomesd.scoring import (
                FrequencyScorer, SpatialFilterScorer, StatisticalScorer,
                SignalProcessingScorer, SpatialDistributionScorer, SimilarityScorer
            )
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
        tomesd.apply_patch(
            unet,
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
            # ABP vs Bipartite matching algorithm for scoring merge
            merge_method=getattr(args, 'scoring_matching_algorithm', 'bipartite'),
            # ABP configuration parameters
            abp_scorer=abp_scorer,
            abp_tile_aggregation=getattr(args, 'abp_tile_aggregation', 'max'),
            # Locality-based sub-block bipartite matching parameters
            locality_block_factor_h=getattr(args, 'locality_block_factor_h', 1),
            locality_block_factor_w=getattr(args, 'locality_block_factor_w', 1)
        )

        # Check if locality-based parameters are being used
        locality_h = getattr(args, 'locality_block_factor_h', 1)
        locality_w = getattr(args, 'locality_block_factor_w', 1)
        locality_info = ""
        if locality_h > 1 or locality_w > 1:
            locality_info = f"\n  Locality-based Sub-block Matching: H factor={locality_h}, W factor={locality_w}"
        
        print(
            f"Scoring-based token merging applied with the following settings:\n"
            f"  Matching Algorithm: {getattr(args, 'scoring_matching_algorithm', 'bipartite')}\n"
            f"  Merging Method:   {getattr(args, 'token_merging_method', 'mean')}\n"
            f"  Scoring Method:   {args.scoring_method}\n"
            f"  Scorer:           {scorer.get_name()}\n"
            + (f"  ABP Scorer:       {abp_scorer.get_name() if abp_scorer else 'N/A'}\n"
               f"  ABP Tile Agg:     {getattr(args, 'abp_tile_aggregation', 'max')}\n" 
               if getattr(args, 'scoring_matching_algorithm', 'bipartite') == 'abp' else "") +
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
            f"  Proportional Attention: {bool(getattr(args, 'if_proportional_attention', 0))}{locality_info}"
        )

    # elif args.if_agentsd==1:
    #     agentsd.remove_patch(unet)
    #     agentsd.apply_patch(unet,             
    #                         ratio=args.token_merging_ratio,
    #                         sx=args.token_merging_sx,
    #                         sy=args.token_merging_sy,
    #                         max_downsample=args.token_merging_max_downsample,
    #                         use_rand=bool(args.token_merging_use_rand),
    #                         single_downsample_level_merge=bool(args.token_merging_single_downsample_level_merge), 
    #                         agent_ratio=0.5, 
    #                         attn_precision="fp32")
    #     print(
    #         f"Agentsd applied with the following settings:\n"
    #         f"  Ratio:            {args.token_merging_ratio}\n"
    #         f"  Agent Ratio:      {args.agent_ratio}\n"
    #         f"  Stride X (sx):    {args.token_merging_sx}\n"
    #         f"  Stride Y (sy):    {args.token_merging_sy}\n"
    #         f"  Max Downsample:   {args.token_merging_max_downsample}\n"
    #         f"  single_downsample_level_merge:   {args.token_merging_single_downsample_level_merge}\n"
    #         f"  Use Random:       {bool(args.token_merging_use_rand)}"
            
    #     )

    elif getattr(args, 'if_agent_guided', 0) == 1:
        # Apply NEW SIMPLIFIED agent-guided token merging (10-100x faster than old bipartite approach)
        from ImprovedTokenMerge.agent_integration import create_simple_agent_selector
        from ImprovedTokenMerge.agent_guided_scoring import HybridAgentScorer
        # Add tomesd to path if not already there
        import sys
        import os
        tomesd_path = os.path.join(os.path.dirname(__file__), '..', 'tomesd')
        if tomesd_path not in sys.path:
            sys.path.insert(0, tomesd_path)
        from tomesd.scoring import create_scorer
        

        
        # Determine if using hybrid approach
        use_hybrid = (getattr(args, 'agent_base_method', None) not in [None, 'None'] and 
                     getattr(args, 'agent_weight', 1.0) < 1.0)
        
        if use_hybrid:
            # Hybrid approach: combine simplified agent with existing method
            print(f"   - Mode: HYBRID (Agent + {args.agent_base_method})")
            print(f"   - Agent Weight: {getattr(args, 'agent_weight', 0.7)} (rest: base method)")
            
            # Create hybrid scorer using new simplified approach
            scorer = HybridAgentScorer(
                agent_method=getattr(args, 'agent_method', 'adaptive_spatial'),
                importance_method=getattr(args, 'agent_importance_method', 'cross_attention'),
                base_scoring_method=args.agent_base_method,
                base_ranking_method=getattr(args, 'agent_base_ranking', 'l2norm'),
                num_agents=getattr(args, 'num_agents', 16),
                agent_weight=getattr(args, 'agent_weight', 0.7)
            )
            approach_name = f"Hybrid Simplified Agent + {args.agent_base_method}"
        else:
            # Pure simplified agent approach (fastest)
            print(f"   - Mode: PURE SIMPLIFIED AGENT")
            print(f"   - Agent Method: {getattr(args, 'agent_method', 'adaptive_spatial')}")
            print(f"   - Importance Method: {getattr(args, 'agent_importance_method', 'cross_attention')}")
            
            # For pure agent, we'll create a wrapper scorer that uses the new simplified approach
            class SimplifiedAgentScorer:
                def __init__(self, agent_method, importance_method, num_agents):
                    self.agent_method = agent_method
                    self.importance_method = importance_method
                    self.num_agents = num_agents
                    
                def get_name(self):
                    return f"simplified_agent_{self.agent_method}_{self.importance_method}"
                
                def score_tokens(self, x, H=None, W=None, **kwargs):
                    """Use new simplified approach for token scoring"""
                    from ImprovedTokenMerge.agent_downsampling import SimpleAgentGuidedMerging
                    from ImprovedTokenMerge.agent_guided_scoring import extract_attention_projections
                    
                    merger = SimpleAgentGuidedMerging(
                        num_agents=self.num_agents,
                        agent_method=self.agent_method,
                        importance_method=self.importance_method
                    )
                    
                    # Extract Q/K projections from UNet for cross attention scoring (cache for efficiency)
                    if not hasattr(self, '_cached_qkv_proj'):
                        existing_qkv_proj = None
                        if hasattr(self, 'unet') and self.unet is not None:
                            existing_qkv_proj = extract_attention_projections(self.unet, layer_name="attn1")
                        self._cached_qkv_proj = existing_qkv_proj
                    else:
                        existing_qkv_proj = self._cached_qkv_proj
                    
                    # Create agents and compute importance scores with projections
                    agents = merger.agent_creator.create_agents(x, method=self.agent_method, H=H, W=W)
                    scores = merger.importance_scorer.compute_importance(agents, x, existing_qkv_proj=existing_qkv_proj)
                    
                    return scores
            
            scorer = SimplifiedAgentScorer(
                agent_method=getattr(args, 'agent_method', 'adaptive_spatial'),
                importance_method=getattr(args, 'agent_importance_method', 'cross_attention'),
                num_agents=getattr(args, 'num_agents', 16)
            )
            
            # Store UNet reference in scorer for Q/K projection extraction
            scorer.unet = unet
            
            approach_name = f"Pure Simplified Agent"
        
        # Apply simplified agent-guided scoring with tomesd

        tomesd.apply_patch(
            unet,
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
            # Simplified agent-guided scoring parameters
            use_scoring=True,
            scorer=scorer,
            preserve_ratio=getattr(args, 'agent_preserve_ratio', 0.3),
            score_mode=getattr(args, 'agent_score_mode', 'high'),
            preserve_spatial_uniformity=bool(getattr(args, 'agent_preserve_spatial_uniformity', 0)),
            if_low_frequency_dst_tokens=bool(getattr(args, 'if_low_frequency_dst_tokens', 0)),
            # Locality-based sub-block bipartite matching parameters
            locality_block_factor_h=getattr(args, 'locality_block_factor_h', 1),
            locality_block_factor_w=getattr(args, 'locality_block_factor_w', 1)
        )
        
        # Check if locality-based parameters are being used
        locality_h = getattr(args, 'locality_block_factor_h', 1)
        locality_w = getattr(args, 'locality_block_factor_w', 1)
        locality_grid_info = ""
        if locality_h > 1 or locality_w > 1:
            locality_grid_info = f"\n     - Locality Sub-block: H factor={locality_h}, W factor={locality_w}"
        
        print(
            f"Simplified Agent-Guided Token Merging Applied:\n"
            f"  AGENT CONFIGURATION:\n"
            f"     - Agent Method:      {getattr(args, 'agent_method', 'adaptive_spatial')}\n"
            f"     - Importance Method: {getattr(args, 'agent_importance_method', 'cross_attention')}\n"
            f"     - Number of Agents:  {getattr(args, 'num_agents', 16)}\n"
            f"     - Base Method:       {getattr(args, 'agent_base_method', 'None (Pure simplified agent)')}\n"
            f"     - Base Ranking:      {getattr(args, 'agent_base_ranking', 'l2norm')}\n"
            f"     - Agent Weight:      {getattr(args, 'agent_weight', 1.0)}\n"
            f"     - Scorer Name:       {scorer.get_name()}\n"
            f"  MERGING CONFIGURATION:\n"
            f"     - Merging Ratio:     {args.token_merging_ratio}\n"
            f"     - Preserve Ratio:    {getattr(args, 'agent_preserve_ratio', 0.3)}\n"
            f"     - Score Mode:        {getattr(args, 'agent_score_mode', 'high')}\n"
            f"     - Preserve Spatial Uniformity: {bool(getattr(args, 'agent_preserve_spatial_uniformity', 0))}\n"
            f"     - Score-guided Destination: {bool(getattr(args, 'if_low_frequency_dst_tokens', 0))}\n"
            f"  GRID CONFIGURATION:\n"
            f"     - Stride X (sx):     {args.token_merging_sx}\n"
            f"     - Stride Y (sy):     {args.token_merging_sy}\n"
            f"     - Max Downsample:    {args.token_merging_max_downsample}\n"
            f"     - Single Downsample Level: {args.token_merging_single_downsample_level_merge}\n"
            f"     - Cache Per Image:   {bool(getattr(args, 'token_merging_cache_indices_per_image', 0))}\n"
            f"     - Use Random:        {bool(args.token_merging_use_rand)}{locality_grid_info}\n"
            f"  ATTENTION CONFIGURATION:\n"
            f"     - Merge Attn:        {bool(getattr(args, 'merge_attn', 1))}\n"
            f"     - Merge Cross-Attn:  {bool(getattr(args, 'merge_crossattn', 0))}\n"
            f"     - Merge MLP:         {bool(getattr(args, 'merge_mlp', 0))}\n"
            f"     - Proportional Attention: {bool(getattr(args, 'if_proportional_attention', 0))}\n"
            f"  KEY BENEFITS:\n"
            f"     - Q tokens: Always preserved (full query capability)\n"
            f"     - K,V tokens: Reduced using simplified agent guidance\n"
            f"     - Attention computation: 25-75% faster depending on ratio\n"
            f"     - Perfect for: Diffusion acceleration, ViT efficiency, real-time apps"
        )

    elif args.if_attention_proc == 1:

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
            "linear_blend_factor": getattr(args, 'linear_blend_factor', 0.5),
            "qkv_linear_blend_factor": getattr(args, 'qkv_linear_blend_factor', 0.5),
            "out_linear_blend_factor": getattr(args, 'out_linear_blend_factor', 0.5),
            "blockwise_blend_factor": getattr(args, 'blockwise_blend_factor', 0.5),
            "linear_blend_method_1": getattr(args, 'linear_blend_method_1', 'nearest-exact'),
            "linear_blend_method_2": getattr(args, 'linear_blend_method_2', 'avg_pool'),
            "qkv_linear_blend_method_1": getattr(args, 'qkv_linear_blend_method_1', 'nearest-exact'),
            "qkv_linear_blend_method_2": getattr(args, 'qkv_linear_blend_method_2', 'avg_pool'),
            "out_linear_blend_method_1": getattr(args, 'out_linear_blend_method_1', 'nearest-exact'),
            "out_linear_blend_method_2": getattr(args, 'out_linear_blend_method_2', 'avg_pool'),
            
            # Timestep-based interpolation for linear blend
            "linear_blend_timestep_interpolation": getattr(args, 'linear_blend_timestep_interpolation', 0),
            "linear_blend_start_ratio": getattr(args, 'linear_blend_start_ratio', 0.1),
            "linear_blend_end_ratio": getattr(args, 'linear_blend_end_ratio', 0.9),
            "qkv_linear_blend_timestep_interpolation": getattr(args, 'qkv_linear_blend_timestep_interpolation', 0),
            "qkv_linear_blend_start_ratio": getattr(args, 'qkv_linear_blend_start_ratio', 0.1),
            "qkv_linear_blend_end_ratio": getattr(args, 'qkv_linear_blend_end_ratio', 0.9),
            "out_linear_blend_timestep_interpolation": getattr(args, 'out_linear_blend_timestep_interpolation', 0),
            "out_linear_blend_start_ratio": getattr(args, 'out_linear_blend_start_ratio', 0.1),
            "out_linear_blend_end_ratio": getattr(args, 'out_linear_blend_end_ratio', 0.9),
            
            "timestep_threshold_switch": args.timestep_threshold_switch,
            "timestep_threshold_stop": args.timestep_threshold_stop,
            "secondary_merge_method": args.secondary_merge_method,
            "downsample_factor_level_2": args.downsample_factor_level_2,
            "ratio_level_2": args.ratio_level_2,
            "frequency_selection_mode": args.frequency_selection_mode,
            "frequency_selection_method": args.frequency_selection_method,
            "frequency_ranking_method": args.frequency_ranking_method,
            "selection_source": getattr(args, 'selection_source', 'hidden'),  # Add if not in args
            "frequency_grid_alpha": getattr(args, 'frequency_grid_alpha', 2.0),  # Add if not in args
      
        }
        ImprovedTokenMerge.utils.patch_attention_proc(pipe.unet, token_merge_args=token_merge_args)
        print(
            f"Attention processor patch applied with settings:\n"
            f"  Ratio:               {args.token_merging_ratio}\n"

            f"  Stride X (sx):    {args.token_merging_sx}\n"
            f"  Stride Y (sy):    {args.token_merging_sy}\n"
            f"  Use Random:       {bool(args.token_merging_use_rand)}"

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

    else:
        print("No token merging applied.")
    return vae, tokenizer, text_encoder, unet, scheduler


def get_scheduler_config(args):
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
        raise NotImplementedError

    return config
