import torch
import math
import inspect
from typing import Type, Dict, Any, Tuple, Callable, Optional

from . import merge
from .utils import isinstance_str, init_generator



def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]
    
    # Create a cache key based on the current layer dimensions and downsample level (for per-image caching)
    cache_key = f"downsample_{downsample}_tokens_{x.shape[1]}"
    
    # NEW: Check for multi-resolution caching within UNet call
    resolution_cache_enabled = (args.get("cache_resolution_merge", False) and 
                              args.get("use_scoring", False))
    
    if resolution_cache_enabled:
        m, u = _compute_merge_with_resolution_cache(x, tome_info, downsample, cache_key)
    else:
        m, u = _compute_merge_original(x, tome_info, downsample, cache_key)
    
    # Create wrapper functions that use the specified method
    method = args["method"]
    
    def make_merge_with_method(merge_fn):
        if merge_fn == merge.do_nothing:
            return merge_fn
        return lambda x: merge_fn(x, mode=method)
    
    m_a_base, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c_base, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)  
    m_m_base, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)
    
    # Wrap merge functions to use the specified method
    m_a = make_merge_with_method(m_a_base)
    m_c = make_merge_with_method(m_c_base) 
    m_m = make_merge_with_method(m_m_base)
    
    # Update token sizes after merging if proportional attention is enabled
    if args["if_proportional_attention"] and tome_info["token_sizes"] is not None:
        # Only persist token_sizes across layers when using cached indices
        # When not using cached indices, each layer should start fresh
        if args["cache_indices_per_image"]:
            # If any merging happened, update token sizes (use base merge function, not wrapped)
            if hasattr(m, '_token_sizes') and m._token_sizes is not None:
                tome_info["token_sizes"] = m._token_sizes
        # When not using cached indices, we don't update tome_info["token_sizes"]
        # so each layer starts with fresh token_sizes=1

    # Instrumentation removed for speed optimization

    return m_a, m_c, m_m, u_a, u_c, u_m


def _compute_merge_with_resolution_cache(x: torch.Tensor, tome_info: Dict[str, Any], 
                                       downsample: int, cache_key: str) -> Tuple[Callable, Callable]:
    """
    Compute merge/unmerge functions with multi-resolution caching enabled.
    Returns only the base m, u functions (not the per-attention-type variants).
    """
    args = tome_info["args"]
    
    # Determine resolution based on token count
    n_tokens = x.shape[1]
    resolution_key = str(n_tokens)  # Use token count as resolution identifier
    
    # Get cache mode and generate appropriate cache key
    cache_mode = args.get("cache_resolution_mode", "global")
    
    if cache_mode == "global":
        # Cache once per resolution across all block types
        res_cache_key = resolution_key
        cache_dict = tome_info["resolution_cache"]["global"]
    else:  # block_specific
        # Simplified for speed optimization - fallback to global mode
        res_cache_key = resolution_key
        cache_mode = "global"
        cache_dict = tome_info["resolution_cache"]["global"]
    
    # Check if we have cached merge/unmerge functions for this resolution
    if res_cache_key in cache_dict:
        # Statistics removed for speed optimization
        m, u = cache_dict[res_cache_key]
        # Debug print removed for speed optimization
    else:
        # Compute merge/unmerge functions and cache them
        # Statistics removed for speed optimization
        
        m, u = _compute_merge_original(x, tome_info, downsample, cache_key)
        
        # Cache the functions for reuse within this UNet call
        cache_dict[res_cache_key] = (m, u)
        # Debug print removed for speed optimization
    
    return m, u


def _initialize_generator(args: Dict[str, Any], device: torch.device) -> None:
    """Helper function to initialize generator if needed."""
    if args["generator"] is None:
        args["generator"] = init_generator(device)
    elif args["generator"].device != device:
        args["generator"] = init_generator(device, fallback=args["generator"])


def _initialize_token_sizes(args: Dict[str, Any], tome_info: Dict[str, Any], x: torch.Tensor) -> Optional[torch.Tensor]:
    """Helper function to initialize token sizes for proportional attention."""
    token_sizes = None
    if args["if_proportional_attention"]:
        # Proportional attention only makes sense when using cached indices
        # because it relies on consistent merging patterns across layers
        if args["cache_indices_per_image"]:
            if tome_info["token_sizes"] is not None:
                # Use existing token sizes from previous layers, but check batch size compatibility
                cached_token_sizes = tome_info["token_sizes"]
                B, N, _ = x.shape
                
                # Check if cached token_sizes matches current batch size and token count
                if (cached_token_sizes.shape[0] == B and 
                    cached_token_sizes.shape[1] == N):
                    token_sizes = cached_token_sizes
                else:
                    # Batch size or token count mismatch, reinitialize
                    token_sizes = torch.ones(B, N, device=x.device, dtype=torch.float32)
                    tome_info["token_sizes"] = token_sizes
            else:
                # Initialize token sizes (all tokens start with size 1)
                B, N, _ = x.shape
                token_sizes = torch.ones(B, N, device=x.device, dtype=torch.float32)
                tome_info["token_sizes"] = token_sizes
        else:
            # When cache_indices_per_image == false, disable proportional attention
            # because each layer makes independent merging decisions
            token_sizes = None
    return token_sizes


def _extract_scoring_params(args: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to extract scoring parameters once."""
    return {
        "preserve_ratio": args["preserve_ratio"],
        "score_mode": args["score_mode"],
        "preserve_spatial_uniformity": args["preserve_spatial_uniformity"],
        "if_low_frequency_dst_tokens": args.get("if_low_frequency_dst_tokens", False),
        "scorer_kwargs": args["scorer_kwargs"]
    }


def _compute_merge_with_scoring(x: torch.Tensor, args: Dict[str, Any], tome_info: Dict[str, Any], 
                               cache_key: str, w: int, h: int, r: int, use_rand: bool, 
                               token_sizes: Optional[torch.Tensor]) -> Tuple[Callable, Callable]:
    """Helper function to compute merge with scoring enabled."""
    # Check if caching is enabled for scoring and if we have cached scoring info
    if args["cache_indices_per_image"] and cache_key in tome_info["cached_indices"]:
        # Use cached scoring information
        m, u = merge.bipartite_soft_matching_with_scoring_cached(
            tome_info["cached_indices"][cache_key], w, h, args["sx"], args["sy"], r, x.device
        )
    else:
        # Extract scoring parameters once to avoid redundant dict lookups
        scoring_params = _extract_scoring_params(args)
        
        # Use score-based merging with optimized parameter passing
        m, u = merge.bipartite_soft_matching_with_scoring(
            x, args["scorer"], w, h, args["sx"], args["sy"], r,
            preserve_ratio=scoring_params["preserve_ratio"],
            score_mode=scoring_params["score_mode"],
            preserve_spatial_uniformity=scoring_params["preserve_spatial_uniformity"],
            if_low_frequency_dst_tokens=scoring_params["if_low_frequency_dst_tokens"],
            no_rand=not use_rand,
            generator=args["generator"],
            token_sizes=token_sizes,
            cache_resolution_merge=args.get("cache_resolution_merge", False),
            merge_method=args.get("merge_method", "bipartite"),
            abp_scorer=args.get("abp_scorer"),
            abp_tile_aggregation=args.get("abp_tile_aggregation", "max"),
            locality_block_factor_h=args.get("locality_block_factor_h", 1),
            locality_block_factor_w=args.get("locality_block_factor_w", 1),
            **scoring_params["scorer_kwargs"]
        )
        
        # If caching is enabled, store the scoring information for future use
        if args["cache_indices_per_image"]:
            # Check if merge function has cached info
            if hasattr(m, '_cached_scoring_info'):
                tome_info["cached_indices"][cache_key] = m._cached_scoring_info
            elif not args.get("cache_resolution_merge", False):
                # Use fallback if cache_resolution_merge is disabled
                cached_scoring_info = merge._extract_scoring_info_for_cache(
                    x, args["scorer"], w, h, args["sx"], args["sy"], r,
                    preserve_ratio=scoring_params["preserve_ratio"],
                    score_mode=scoring_params["score_mode"],
                    no_rand=not use_rand,
                    generator=args["generator"],
                    **scoring_params["scorer_kwargs"]
                )
                if cached_scoring_info:  # Only cache if extraction was successful
                    tome_info["cached_indices"][cache_key] = cached_scoring_info
            # If cache_resolution_merge=True but _cached_scoring_info is missing, skip fallback
    
    # Debug instrumentation removed for speed optimization
    return m, u


# Function removed for speed optimization


def _compute_merge_without_scoring(x: torch.Tensor, args: Dict[str, Any], tome_info: Dict[str, Any],
                                  cache_key: str, w: int, h: int, r: int, use_rand: bool,
                                  token_sizes: Optional[torch.Tensor]) -> Tuple[Callable, Callable]:
    """Helper function to compute merge without scoring."""
    # Check if caching is enabled and if we have cached indices for this layer
    if args["cache_indices_per_image"] and cache_key in tome_info["cached_indices"]:
        # Use cached indices
        m, u = merge.bipartite_soft_matching_from_cached_indices(
            tome_info["cached_indices"][cache_key], w, h, args["sx"], args["sy"], r, x.device
        )
    else:
        # Choose merge method
        if args.get("merge_method", "bipartite") == "abp":
            # NEW: Use ABP instead of bipartite
            m, u = merge.adaptive_block_pooling_random2d(x, w, h, args["sx"], args["sy"], r,
                                                        no_rand=not use_rand, 
                                                        generator=args["generator"],
                                                        token_sizes=token_sizes)
            
            # Debug instrumentation removed for speed optimization
        else:
            # Original bipartite matching
            m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, 
                                                          no_rand=not use_rand, generator=args["generator"], 
                                                          token_sizes=token_sizes,
                                                          locality_block_factor_h=args.get("locality_block_factor_h", 1),
                                                          locality_block_factor_w=args.get("locality_block_factor_w", 1))
            
            # Debug instrumentation removed for speed optimization
        
        # If caching is enabled, store the indices for future use (skip for ABP for now)
        if args["cache_indices_per_image"] and args.get("merge_method", "bipartite") == "bipartite":
            # We need to extract the indices from the merge function
            # For now, we'll recompute with the same parameters to extract indices
            cached_indices = merge._extract_indices_from_merge(x, w, h, args["sx"], args["sy"], r, 
                                                         not use_rand, args["generator"])
            tome_info["cached_indices"][cache_key] = cached_indices
    return m, u


def _wrap_merge_with_token_sizes(m: Callable, token_sizes: Optional[torch.Tensor]) -> Callable:
    """Helper function to wrap merge function with token sizes if needed."""
    if token_sizes is not None and hasattr(m, '__call__') and m != merge.do_nothing:
        # Create a wrapper that initializes token sizes
        original_merge = m
        def merge_with_token_sizes(x_input, mode="mean"):
            # Initialize token sizes for this merge function call
            original_merge._token_sizes = token_sizes.clone()
            return original_merge(x_input, mode)
        return merge_with_token_sizes
    return m


def _compute_merge_for_level(x: torch.Tensor, tome_info: Dict[str, Any], downsample: int, 
                            cache_key: str, original_h: int, original_w: int) -> Tuple[Callable, Callable]:
    """Helper function to compute merge for a specific downsample level."""
    args = tome_info["args"]
    
    w = int(math.ceil(original_w / downsample))
    h = int(math.ceil(original_h / downsample))
    r = int(x.shape[1] * args["ratio"])

    # Re-init the generator if it hasn't already been initialized or device has changed.
    _initialize_generator(args, x.device)
    
    # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
    # batch, which causes artifacts with use_rand, so force it to be off.
    use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
    
    # Initialize token sizes for this layer if proportional attention is enabled
    token_sizes = _initialize_token_sizes(args, tome_info, x)
    
    # Check if scoring is enabled
    if args["use_scoring"]:
        m, u = _compute_merge_with_scoring(x, args, tome_info, cache_key, w, h, r, use_rand, token_sizes)
    else:
        m, u = _compute_merge_without_scoring(x, args, tome_info, cache_key, w, h, r, use_rand, token_sizes)
    
    # Initialize token sizes for merge function if proportional attention is enabled
    if args["if_proportional_attention"]:
        m = _wrap_merge_with_token_sizes(m, token_sizes)
    
    return m, u


def _compute_merge_original(x: torch.Tensor, tome_info: Dict[str, Any], 
                          downsample: int, cache_key: str) -> Tuple[Callable, Callable]:
    """
    Original compute_merge logic, extracted for use with/without resolution caching.
    """
    original_h, original_w = tome_info["size"]
    args = tome_info["args"]
    
    # Default to do_nothing
    m, u = merge.do_nothing, merge.do_nothing
    
    if args["single_downsample_level_merge"] == False:
        if downsample <= args["max_downsample"]:
            m, u = _compute_merge_for_level(x, tome_info, downsample, cache_key, original_h, original_w)
        else:
            m, u = merge.do_nothing, merge.do_nothing
    else:
        if downsample == args["max_downsample"]:
            m, u = _compute_merge_for_level(x, tome_info, downsample, cache_key, original_h, original_w)
        else:
            m, u = merge.do_nothing, merge.do_nothing

    return m, u


def clear_resolution_cache(model: torch.nn.Module, debug: bool = False):
    """
    Clear the resolution cache for a new UNet call.
    This is called automatically at the start of each UNet forward pass.
    
    Args:
        model: The UNet model
        debug: Whether to print debug information
    """
    model = model.unet if hasattr(model, "unet") else model
    
    if hasattr(model, "_tome_info") and model._tome_info is not None:
        if "resolution_cache" in model._tome_info:
            model._tome_info["resolution_cache"]["global"].clear()
            model._tome_info["resolution_cache"]["block_specific"].clear()
            # Debug print removed for speed optimization


# Function removed for speed optimization










def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock






def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock






# Function removed for speed optimization


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size and timestep. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        
        # NEW: Clear resolution cache at start of each UNet forward pass
        if "resolution_cache" in module._tome_info:
            module._tome_info["resolution_cache"]["global"].clear()
            module._tome_info["resolution_cache"]["block_specific"].clear()
            module._tome_info["unet_call_id"] = module._tome_info.get("unet_call_id", 0) + 1
            # Debug print removed for speed optimization
            
        # Timestep capture removed for speed optimization
        
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))








def apply_patch(
    model: torch.nn.Module,
    ratio: float = 0.5,
    max_downsample: int = 1,
    sx: int = 2, sy: int = 2,
    use_rand: bool = True,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False,
    merge_tokens: str = "keys/values",  # Add merge_tokens parameter with default value
    single_downsample_level_merge: bool=False,
    cache_indices_per_image: bool = False,
    # New scoring parameters
    use_scoring: bool = False,
    scorer: Optional[Any] = None,
    preserve_ratio: float = 0.3,
    score_mode: str = "high",
    preserve_spatial_uniformity: bool = False,
    if_low_frequency_dst_tokens: bool = False,
    scorer_kwargs: Optional[Dict[str, Any]] = None,
    # Proportional attention parameter
    if_proportional_attention: bool = False,
    # MLERP merging parameter
    method: str = "mean",
    # NEW: Multi-resolution caching parameters
    cache_resolution_merge: bool = False,
    cache_resolution_mode: str = "global",
    debug_cache: bool = False,
    # NEW: ABP merging parameter
    merge_method: str = "bipartite",
    # NEW: ABP configuration parameters
    abp_scorer: Optional[Any] = None,
    abp_tile_aggregation: str = "max",
    # NEW: Locality-based similarity parameters
    locality_block_factor_h: int = 1,
    locality_block_factor_w: int = 1):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
     - cache_indices_per_image: If True, compute merge indices once per image (using original image)
                               and reuse them for all timesteps. Default: False.
    
    Scoring Args (NEW):
     - use_scoring: Whether to use score-based token merging instead of random selection. Default: False.
     - scorer: TokenScorer instance to compute token importance scores. If None and use_scoring=True,
               defaults to StatisticalScorer(method="l2norm").
     - preserve_ratio: Ratio of tokens to protect from merging based on scores (0.0 to 1.0). Default: 0.3.
     - score_mode: How to select tokens based on scores ("high", "low", "medium"). Default: "high".
     - preserve_spatial_uniformity: If True, apply bipartite matching to full image first, then filter protected tokens.
                                   If False, extract mergeable subset first (current approach). Default: False.
                                   Setting to True preserves spatial uniformity but is computationally more expensive.
     - if_low_frequency_dst_tokens: If True, select lowest-scored tokens as destinations within each spatial block.
                                   When enabled, uses token scores to guide destination selection instead of random/first position.
                                   Only applies when scoring-based token merging is enabled. Default: False.
     - scorer_kwargs: Additional keyword arguments to pass to the scorer. Default: None.
     
    Proportional Attention Args:
     - if_proportional_attention: Whether to use proportional attention that accounts for token sizes.
                                 This modifies attention scores by adding log(s) where s is the token size.
                                 Helps maintain influence of merged tokens in attention. Default: False.
     
    MLERP Args:
     - method: Token merging method to use. Options are:
               - "mean": Standard average merging (default)
               - "mlerp": MLERP (Maximum-Norm Linear Interpolation) merging that preserves feature magnitudes
               - "prune": Simply remove tokens without merging (if supported by method)
               MLERP typically provides better accuracy at large reduction ratios.
    
    Multi-Resolution Caching Args (NEW):
     - cache_resolution_merge: Whether to cache merge/unmerge functions per resolution within UNet call.
                              This significantly speeds up scoring merge by reusing computations across layers
                              of the same resolution. Default: False.
     - cache_resolution_mode: Caching granularity mode. Options are:
                             - "global": Cache once per resolution across all block types (faster, less memory)
                             - "block_specific": Cache separately for down/up/mid blocks per resolution (more precise)
                             Default: "global".
     - debug_cache: Whether to print debug information about cache hits/misses. Default: False.
     
     ABP (Adaptive Block Pooling) Args (NEW):
     - abp_scorer: TokenScorer instance for tile evaluation when using ABP. If None, uses SpatialFilterScorer.
                   Only used when merge_method="abp".
     - abp_tile_aggregation: How to aggregate token scores within tiles for ABP. Options are:
                           - "max": Highest score per tile (default)
                           - "min": Lowest score per tile 
                           - "sum": Total score per tile (same ordering as mean but faster)
                           - "std": Standard deviation per tile (captures score variance)
                           Only used when merge_method="abp".
    
     Locality-Based Similarity Args (NEW):
     - locality_block_factor_h: Factor to divide height for locality-based similarity computation.
                               Default: 1 (global similarity). Values > 1 enable spatial locality.
                               Example: 2 divides the image into 2x2 blocks for within-block matching.
     - locality_block_factor_w: Factor to divide width for locality-based similarity computation.
                               Default: 1 (global similarity). Values > 1 enable spatial locality.
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model

    # Set default scorer if scoring is enabled but no scorer provided
    if use_scoring and scorer is None:
        from .scoring import StatisticalScorer
        scorer = StatisticalScorer(method="l2norm")
    
    # Set default scorer_kwargs if None
    if scorer_kwargs is None:
        scorer_kwargs = {}

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "single_downsample_level_merge": single_downsample_level_merge,
            "cache_indices_per_image": cache_indices_per_image,
            # New scoring parameters
            "use_scoring": use_scoring,
            "scorer": scorer,
            "preserve_ratio": preserve_ratio,
            "score_mode": score_mode,
            "preserve_spatial_uniformity": preserve_spatial_uniformity,
            "if_low_frequency_dst_tokens": if_low_frequency_dst_tokens,
            "scorer_kwargs": scorer_kwargs,
            # Proportional attention parameter
            "if_proportional_attention": if_proportional_attention,
            # MLERP merging parameter
            "method": method,
            # NEW: Multi-resolution caching parameters
            "cache_resolution_merge": cache_resolution_merge,
            "cache_resolution_mode": cache_resolution_mode,
            "debug_cache": debug_cache,
            # NEW: ABP merging parameter
            "merge_method": merge_method,
            # NEW: ABP configuration parameters
            "abp_scorer": abp_scorer,
            "abp_tile_aggregation": abp_tile_aggregation,
            # NEW: Locality-based similarity parameters
            "locality_block_factor_h": locality_block_factor_h,
            "locality_block_factor_w": locality_block_factor_w
        },
        "cached_indices": {},  # Store cached indices per image
        "current_image_latent": None,  # Store current image latent for comparison
        "token_sizes": None,  # Track token sizes for proportional attention
        # NEW: Multi-resolution caching infrastructure
        "resolution_cache": {
            "global": {},  # resolution -> (merge_fn, unmerge_fn)
            "block_specific": {}  # resolution_blocktype -> (merge_fn, unmerge_fn)
        },
        "unet_call_id": 0  # Track UNet forward pass iterations
        # Statistics removed for speed optimization
    }
    hook_tome_model(diffusion_model)

    for module_name, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info
            
            # Block type info extraction removed for speed optimization

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model





def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model


def clear_cached_indices(model: torch.nn.Module):
    """
    Clear cached indices for a new image when cache_indices_per_image is enabled.
    This should be called when starting to process a new image.
    """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model
    
    if hasattr(model, "_tome_info") and model._tome_info is not None:
        if "cached_indices" in model._tome_info:
            model._tome_info["cached_indices"].clear()
        # Image counter removed for speed optimization


def precompute_cache_from_clean_latent(model: torch.nn.Module, clean_latent: torch.Tensor, 
                                       dummy_timestep: torch.Tensor = None, 
                                       dummy_encoder_hidden_states: torch.Tensor = None):
    """
    Precompute cache indices from clean latent for cache_indices_per_image.
    This does a single forward pass with the clean latent to compute and store cache indices.
    
    Args:
        model: The UNet model
        clean_latent: Clean latent tensor (x0)
        dummy_timestep: Timestep tensor (if None, uses middle timestep)
        dummy_encoder_hidden_states: Text embeddings (if None, uses zeros)
    """
    # For diffusers
    unet = model.unet if hasattr(model, "unet") else model
    
    # Check if cache_indices_per_image is enabled
    if not (hasattr(unet, "_tome_info") and 
            unet._tome_info is not None and 
            unet._tome_info["args"].get("cache_indices_per_image", False)):
        return
    
    device = clean_latent.device
    dtype = clean_latent.dtype
    batch_size = clean_latent.shape[0]
    
    # Create dummy inputs if not provided
    if dummy_timestep is None:
        # Use a middle timestep for cache computation
        dummy_timestep = torch.tensor([500], device=device)
        if batch_size > 1:
            dummy_timestep = dummy_timestep.repeat(batch_size)
    
    if dummy_encoder_hidden_states is None:
        # Create dummy text embeddings
        # Standard CLIP text embedding size is 77 tokens x 768 dims
        dummy_encoder_hidden_states = torch.zeros(batch_size, 77, 768, device=device, dtype=dtype)
    
    # Convert timestep to appropriate dtype
    if dtype == torch.float16:
        dummy_timestep = dummy_timestep.half()
    
    # Debug print removed for speed optimization
    
    # Do a single forward pass with clean latent to compute cache indices
    with torch.no_grad():
        try:
            _ = unet(clean_latent, dummy_timestep, encoder_hidden_states=dummy_encoder_hidden_states)
            # Debug print removed for speed optimization
        except Exception as e:
            # Debug print removed for speed optimization
            # Clear any partial cache that might have been created
            if "cached_indices" in unet._tome_info:
                unet._tome_info["cached_indices"].clear()