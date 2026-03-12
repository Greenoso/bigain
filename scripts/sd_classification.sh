# =================================================================================
# CONFIGURATION PARAMETERS
# =================================================================================


# --- Core Experiment Configuration ---

VERSION="2-0"
SEED=0
IMG_SIZE=512
DATASET="pets" # Change this to 'pets' or 'flowers' as needed
SPLIT="test"
N_TRIALS=1
BATCH_SIZE=16
TO_KEEP="5 1" # Needs to be quoted if passed as a single arg, or handle spaces carefully
N_SAMPLES="5 20" # Needs to be quoted if passed as a single arg, or handle spaces carefully
LOSS="l2"
PROMPT_PATH="prompts/${DATASET}_prompts.csv"
N_WORKERS=1
WORKER_IDX=0

# --- Acceleration Method Flags (Set to 1 to enable a method) ---
IF_TOKEN_MERGING=0 # for token merging
IF_ATTENTION_PROC=0 # for token downsampling
IF_SITO=0 # for SiTo method
IF_SCORING_MERGE=0 # for scoring-based token merging
IF_AGENT_GUIDED=0 # for agent-guided token downsampling
IF_PROPORTIONAL_ATTENTION=0 # for proportional attention (can be used with token merging methods)
# Proportional attention can be combined with IF_TOKEN_MERGING=1 or IF_SCORING_MERGE=1
# It modifies attention computation to account for token sizes: A = softmax(QK^T/√d + log(s))


# --- Token Merging Parameters ---
#TOKEN_MERGING_RATIO=0 # Ratio of tokens to merge
TOKEN_MERGING_METHOD="mean" # Token merging method: 'mean' (default), 'mlerp' (better accuracy), 'prune' (maximum speedup)
                            # 'mean': Standard average merging
                            # 'mlerp': MLERP (Maximum-Norm Linear Interpolation) - preserves feature magnitudes
                            # 'prune': Simply removes selected tokens - fastest but may reduce accuracy more
TOKEN_MERGING_USE_RAND=1 # 1 means use random source token
TOKEN_MERGING_SX=2
TOKEN_MERGING_SY=2
TOKEN_MERGING_MAX_DOWNSAMPLE=1
TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE=0
TOKEN_MERGING_CACHE_INDICES_PER_IMAGE=0 # 1 means cache indices per image

MERGE_ATTN=1 # Whether to merge tokens for self-attention (1 for True, 0 for False) - RECOMMENDED
MERGE_CROSSATTN=0 # Whether to merge tokens for cross-attention (1 for True, 0 for False) - NOT RECOMMENDED  
MERGE_MLP=0 # Whether to merge tokens for MLP layers (1 for True, 0 for False) - VERY NOT RECOMMENDED



# --- Token downsampling Parameters ---
DOWNSAMPLE_FACTOR=2
DOWNSAMPLE_FACTOR_LEVEL_2=1
MERGE_TOKENS="keys/values"
MERGE_METHOD="downsample" # Options: 'frequency_global', 'frequency_blockwise', 'downsample','block_avg_pool',
DOWNSAMPLE_METHOD="nearest-exact" # Options: 'nearest-exact', 'max_pool','avg_pool', 'area', "bilinear", "bicubic",'top_right','bottom_left','bottom_right','random',uniform_random,uniform_timestep,'linear_blend'
LINEAR_BLEND_FACTOR=0 # For linear_blend method: 0.0=pure avg_pool (smooth), 0.5=50/50 blend (default), 1.0=pure nearest-exact (sharp)
QKV_LINEAR_BLEND_FACTOR=0 # For QKV linear_blend in downsample_qkv_upsample_out method
OUT_LINEAR_BLEND_FACTOR=0 # For output linear_blend in downsample_qkv_upsample_out method

# Linear blend method selection parameters (enhanced functionality)
LINEAR_BLEND_METHOD_1="nearest-exact" # First method for linear_blend: max_pool, avg_pool, area, bilinear, bicubic, nearest-exact, top_right, bottom_left, bottom_right, random, uniform_random, uniform_timestep
LINEAR_BLEND_METHOD_2="avg_pool" # Second method for linear_blend: max_pool, avg_pool, area, bilinear, bicubic, nearest-exact, top_right, bottom_left, bottom_right, random, uniform_random, uniform_timestep
QKV_LINEAR_BLEND_METHOD_1="nearest-exact" # First method for QKV linear_blend in downsample_qkv_upsample_out method
QKV_LINEAR_BLEND_METHOD_2="avg_pool" # Second method for QKV linear_blend in downsample_qkv_upsample_out method
OUT_LINEAR_BLEND_METHOD_1="nearest-exact" # First method for output linear_blend in downsample_qkv_upsample_out method
OUT_LINEAR_BLEND_METHOD_2="avg_pool" # Second method for output linear_blend in downsample_qkv_upsample_out method

# Timestep-based interpolation for linear blend (dynamic blend factors)
LINEAR_BLEND_TIMESTEP_INTERPOLATION=0  # 0/1 flag to enable timestep-based interpolation
LINEAR_BLEND_START_RATIO=0.1          # Start ratio at timestep 999 (high noise)
LINEAR_BLEND_END_RATIO=0.9            # End ratio at timestep 0 (low noise)
QKV_LINEAR_BLEND_TIMESTEP_INTERPOLATION=0  # 0/1 flag for QKV timestep interpolation
QKV_LINEAR_BLEND_START_RATIO=0.1      # QKV start ratio at timestep 999
QKV_LINEAR_BLEND_END_RATIO=0.9        # QKV end ratio at timestep 0
OUT_LINEAR_BLEND_TIMESTEP_INTERPOLATION=0  # 0/1 flag for output timestep interpolation
OUT_LINEAR_BLEND_START_RATIO=0.1      # Output start ratio at timestep 999
OUT_LINEAR_BLEND_END_RATIO=0.9        # Output end ratio at timestep 0

BLOCKWISE_BLEND_FACTOR=0 # For frequency_blockwise method: 0.0=pure avg_pool (smooth), 0.5=50/50 blend (default), 1.0=pure frequency_selected (sharp)
FREQUENCY_SELECTION_MODE="high" # Options: 'high', 'low', 'timestep_scheduler'
FREQUENCY_SELECTION_METHOD="original" # choices=['original','1d_dft', '1d_dct', '2d_conv','non_uniform_grid'],
FREQUENCY_RANKING_METHOD="mean_deviation" # choices=['spectral_centroid',  'amplitude',"variance", "l1norm", "l2norm","mean_deviation"]
TIMESTEP_THRESHOLD_STOP=0.0
TIMESTEP_THRESHOLD_SWITCH=0 # Set to 0 if you don't want to use this feature
SECONDARY_MERGE_METHOD="none" # or "none" if you want to test that
QKV_DOWNSAMPLE_METHOD="nearest-exact" # Example default
OUT_UPSAMPLE_METHOD="nearest-exact"  # Example default
RATIO_LEVEL_1=0  # Enable actual token reduction for agent-guided merging (50% reduction)
RATIO_LEVEL_2=0
EXTRA_GUIDANCE_SCALE=0
FREQUENCY_GRID_ALPHA=0
SELECTION_SOURCE='hidden' # Options: 'key', 'value', 'hidden','query''


# --- SiTo Parameters ---
SITO_PRUNE_RATIO=0
SITO_MAX_DOWNSAMPLE_RATIO=1
SITO_PRUNE_SELFATTN_FLAG=1
SITO_PRUNE_CROSSATTN_FLAG=0
SITO_PRUNE_MLP_FLAG=0
SITO_SX=2
SITO_SY=2
SITO_NOISE_ALPHA=0.1
SITO_SIM_BETA=1.0

# --- Scoring-based Token Merging Parameters ---
SCORING_METHOD="statistical" # Options: 'frequency', 'spatial_filter', 'statistical', 'signal_processing', 'spatial_distribution', 'similarity'
SCORING_PRESERVE_RATIO=0 # Ratio of tokens to protect from merging (0.0 to 1.0)
SCORING_MODE="high" # Options: 'high', 'low', 'medium', 'timestep_scheduler', 'reverse_timestep_scheduler'
SCORING_PRESERVE_SPATIAL_UNIFORMITY=0 # Whether to preserve spatial uniformity (1 for True, 0 for False)
IF_LOW_FREQUENCY_DST_TOKENS=0 # 0/1 flag for score-guided destination selection (lowest-scored tokens as destinations)

# Frequency scorer specific parameters
SCORING_FREQ_METHOD="1d_dft" # Options: '1d_dft', '1d_dct'
SCORING_FREQ_RANKING="amplitude" # Options: 'amplitude', 'spectral_centroid'

# Spatial filter scorer specific parameters
SCORING_SPATIAL_METHOD="2d_conv" # Options: '2d_conv', '2d_conv_l2'
SCORING_SPATIAL_NORM="l1" # Options: 'l1', 'l2'

# Statistical scorer specific parameters
SCORING_STAT_METHOD="l2norm" # Options: 'variance', 'l1norm', 'l2norm', 'mean_deviation'

# Signal processing scorer specific parameters
SCORING_SIGNAL_METHOD="snr" # Options: 'snr', 'noise_magnitude'

# Spatial distribution scorer specific parameters
SCORING_SPATIAL_ALPHA=2.0 # Alpha parameter (must be > 1)

# Similarity scorer specific parameters  
SCORING_SIMILARITY_METHOD="local_neighbors_inverted" # Options: 'local_neighbors', 'global_mean', 'local_neighbors_inverted', 'global_mean_inverted'
# Note: _inverted methods return negative similarity (low scores for similar tokens) - RECOMMENDED for intuitive merging
# Use inverted methods with SCORING_MODE="high" to protect dissimilar tokens and merge similar ones

# --- Resolution Caching Parameters (NEW) ---
CACHE_RESOLUTION_MERGE=0 # Whether to enable within-UNet resolution caching for scoring merge (0/1)
CACHE_RESOLUTION_MODE="global" # Caching mode: "global" (faster) or "block_specific" (more precise)
DEBUG_CACHE=0 # Whether to enable debug prints for resolution caching (0/1)

# --- ABP (Adaptive Block Pooling) Parameters (NEW) ---
SCORING_MATCHING_ALGORITHM="bipartite" # Options: 'bipartite' (standard), 'abp' (Adaptive Block Pooling - faster)

ABP_SCORER_METHOD="spatial_filter" # Options:
# 'spatial_filter' - Edge detection via Laplacian convolution (best for spatial structure)
# 'frequency' - FFT/DCT spectral analysis (good for frequency patterns)
# 'statistical' - L1/L2 norms, variance (general-purpose magnitude-based)
# 'signal_processing' - SNR/noise analysis (good for noisy environments)
# 'spatial_distribution' - Center-biased scoring (natural images with center focus)
# 'similarity' - Cosine similarity with neighbors/global mean (token relationship-based)

ABP_TILE_AGGREGATION="sum" # Options: 'max' (peak), 'min' (worst-case), 'sum' (total), 'std' (variance)


# --- Agent-Guided Token Merging Parameters (NEW SIMPLIFIED APPROACH) ---
# MAJOR UPDATE: Now uses simplified O(N×agents) approach instead of expensive O(N²) bipartite matching
# Performance improvements: 10-100x faster, 50-90% less memory usage
AGENT_METHOD="adaptive_spatial" # Options: 'adaptive_spatial', 'clustering_centroids', 'statistical_moments', 'frequency_based', 'uniform_sampling'
AGENT_IMPORTANCE_METHOD="cross_attention" # Options: 'cross_attention', 'cosine_similarity', 'euclidean_distance', 'information_theoretic'
NUM_AGENTS=16 # Number of agent tokens to create (typically 8-32, balance between quality and speed)
AGENT_BASE_METHOD="None" # Base scoring method for hybrid approach (None for pure agent, or 'original', '1d_dft', '1d_dct', '2d_conv', '2d_conv_l2')
AGENT_BASE_RANKING="l2norm" # Options: 'l1norm', 'l2norm', 'variance', 'amplitude', 'spectral_centroid' (used with hybrid approach)
AGENT_WEIGHT=1.0 # Weight for agent vs base scoring (1.0 = pure simplified agent, 0.5 = 50/50 hybrid)
AGENT_PRESERVE_RATIO=0.3 # Ratio of tokens to keep as important (rest merged into single token) (0.0 to 1.0)  
AGENT_SCORE_MODE="high" # Options: 'high', 'low', 'medium', 'timestep_scheduler', 'reverse_timestep_scheduler', 'uniform'
AGENT_PRESERVE_SPATIAL_UNIFORMITY=0 # Whether to preserve spatial uniformity in agent-guided merging (1 for True, 0 for False)

# NEW: Simplified merging performance notes:
# - Pure agent (AGENT_BASE_METHOD="None"): Fastest, O(N×agents + N log N) complexity
# - Hybrid agent (AGENT_BASE_METHOD set): Slightly slower but more robust, O(N×agents + N×base_method)
# - Output format: Returns k important tokens + 1 merged token = k+1 total tokens
# - For attention: Preserves all Q tokens, reduces only K,V tokens for maximum efficiency

# --- Locality-based Sub-block Bipartite Matching Parameters (NEW) ---
LOCALITY_BLOCK_FACTOR_H=1 # Factor to divide height for locality-based similarity (1=global, >1=local sub-blocks)
LOCALITY_BLOCK_FACTOR_W=1 # Factor to divide width for locality-based similarity (1=global, >1=local sub-blocks)


# --- Evaluation Parameters ---
SINGLE_TIMESTEP=-1 # If set to a non-negative value, evaluate ONLY this specific timestep instead of the adaptive multi-timestep strategy. Disables --to_keep and --n_samples.
EVAL_DENOISE_STEPS=1 # Number of denoising steps to perform in eval_error. Default=1 (original method).
EVAL_STEP_STRIDE=1 # The stride for the evaluation steps. Default=1 (original method). This is used to control the number of steps in the evaluation process.
EVAL_ERROR_METHOD="trajectory" # Options: 'trajectory', 'direct', 'weighted', 'clean_signal'
FIX_NOISE_ACROSS_TIMESTEP=0 # If set to 1, use the same noise across different timesteps for each prompt


# --- Output and Logging Setup ---
BASE_LOG_DIR="/path/to/results"
LOG_DIR="${BASE_LOG_DIR}/${DATASET}/${MERGE_METHOD}_classification_log_${LOSS}loss"




# =================================================================================
# SCRIPT 
# =================================================================================


# Function to build the experiment name based on current parameters
build_experiment_name() {
    local LOG_NAME="v${VERSION}_${N_TRIALS}trials_"

    if [[ ${SINGLE_TIMESTEP} -ge 0 ]]; then
        LOG_NAME+="single_t${SINGLE_TIMESTEP}"
    else
        LOG_NAME+="$(echo ${TO_KEEP} | tr ' ' '_')keep_"
        LOG_NAME+="$(echo ${N_SAMPLES} | tr ' ' '_')samples"
    fi

    LOG_NAME+="_randomseed${SEED}"

    if [[ "${LOSS}" == "l1" ]]; then
        LOG_NAME+="_l1"
    elif [[ "${LOSS}" == "huber" ]]; then
        LOG_NAME+="_huber"
    elif [[ "${LOSS}" == "l2" ]]; then
        LOG_NAME+="_l2"
    else
        LOG_NAME+="_${LOSS}"
    fi

    if [[ ${IMG_SIZE} -ne 512 ]]; then
        LOG_NAME+="_${IMG_SIZE}"
    fi

    if [[ ${IF_TOKEN_MERGING} -eq 1 ]]; then
        TOKEN_MERGING_SINGLE_DOWNLEVEL_MERGE=${TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE}
        LOG_NAME+="_tokenmerge${RATIO_LEVEL_1}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}_singlelevel${TOKEN_MERGING_SINGLE_DOWNLEVEL_MERGE}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}_cache${TOKEN_MERGING_CACHE_INDICES_PER_IMAGE}"
        
        # Add merge layer types to name
        merge_layers=""
        if [[ ${MERGE_ATTN} -eq 1 ]]; then
            merge_layers+="attn"
        fi
        if [[ ${MERGE_CROSSATTN} -eq 1 ]]; then
            if [[ -n "${merge_layers}" ]]; then
                merge_layers+="+crossattn"
            else
                merge_layers+="crossattn"
            fi
        fi
        if [[ ${MERGE_MLP} -eq 1 ]]; then
            if [[ -n "${merge_layers}" ]]; then
                merge_layers+="+mlp"
            else
                merge_layers+="mlp"
            fi
        fi
        if [[ -n "${merge_layers}" ]]; then
            LOG_NAME+="_merge${merge_layers}"
        fi
        
        # Add method to name if not default
        if [[ "${TOKEN_MERGING_METHOD}" != "mean" ]]; then
            LOG_NAME+="_method${TOKEN_MERGING_METHOD}"
        fi
        if [[ ${IF_PROPORTIONAL_ATTENTION} -eq 1 ]]; then
            LOG_NAME+="_propattn"
        fi
        # Add locality-based sub-block parameters if not default
        if [[ ${LOCALITY_BLOCK_FACTOR_H} -ne 1 || ${LOCALITY_BLOCK_FACTOR_W} -ne 1 ]]; then
            LOG_NAME+="_locality_h${LOCALITY_BLOCK_FACTOR_H}_w${LOCALITY_BLOCK_FACTOR_W}"
        fi
    elif [[ ${IF_ATTENTION_PROC} -eq 1 ]]; then
        if [[ "${MERGE_METHOD}" == "frequency_global" ]]; then
        LOG_NAME+="_mergemethod${MERGE_METHOD}_freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_level1ratio${RATIO_LEVEL_1}_level2method${SECONDARY_MERGE_METHOD}_level2ratio${RATIO_LEVEL_2}"
        if [[ "${FREQUENCY_SELECTION_METHOD}" == "non_uniform_grid" ]]; then
            LOG_NAME+="_gridalpha${FREQUENCY_GRID_ALPHA}"
        fi
        elif [[ "${MERGE_METHOD}" == "frequency_blockwise" ]]; then
        LOG_NAME+="_mergemethod${MERGE_METHOD}_freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_level1downsample${DOWNSAMPLE_FACTOR}"
        # Add blockwise blend parameter to name if not default
        if [[ "${BLOCKWISE_BLEND_FACTOR}" != "0.5" ]]; then
            LOG_NAME+="_blockwiseblend${BLOCKWISE_BLEND_FACTOR}"
        fi
        
        elif [[ "${MERGE_METHOD}" == "masked_attention" ]]; then
        LOG_NAME+="_mergemethod${MERGE_METHOD}_freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_level1ratio${RATIO_LEVEL_1}_level2method${SECONDARY_MERGE_METHOD}_level2ratio${RATIO_LEVEL_2}"
        if [[ "${FREQUENCY_SELECTION_METHOD}" == "non_uniform_grid" ]]; then
            LOG_NAME+="_gridalpha${FREQUENCY_GRID_ALPHA}"
        fi
        
        elif [[ "${MERGE_METHOD}" == "blockwise_masked_attention" ]]; then
        LOG_NAME+="_mergemethod${MERGE_METHOD}_freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_level1downsample${DOWNSAMPLE_FACTOR}_level2method${SECONDARY_MERGE_METHOD}_level2downsample${DOWNSAMPLE_FACTOR_LEVEL_2}"
        
        elif [[ "${MERGE_METHOD}" == "snr_masked_attention" ]]; then
        LOG_NAME+="_mergemethod${MERGE_METHOD}_freqmode_${FREQUENCY_SELECTION_MODE}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_level1ratio${RATIO_LEVEL_1}_level2method${SECONDARY_MERGE_METHOD}_level2ratio${RATIO_LEVEL_2}"
        
        elif [[ "${MERGE_METHOD}" == "snr_blockwise_masked_attention" ]]; then
        LOG_NAME+="_mergemethod${MERGE_METHOD}_freqmode_${FREQUENCY_SELECTION_MODE}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_level1downsample${DOWNSAMPLE_FACTOR}_level2method${SECONDARY_MERGE_METHOD}_level2downsample${DOWNSAMPLE_FACTOR_LEVEL_2}"
        
        elif [[ "${MERGE_METHOD}" == "noise_magnitude_masked_attention" ]]; then
        LOG_NAME+="_mergemethod${MERGE_METHOD}_freqmode_${FREQUENCY_SELECTION_MODE}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_level1ratio${RATIO_LEVEL_1}_level2method${SECONDARY_MERGE_METHOD}_level2ratio${RATIO_LEVEL_2}"
        
        elif [[ "${MERGE_METHOD}" == "noise_magnitude_blockwise_masked_attention" ]]; then
        LOG_NAME+="_mergemethod${MERGE_METHOD}_freqmode_${FREQUENCY_SELECTION_MODE}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_level1downsample${DOWNSAMPLE_FACTOR}_level2method${SECONDARY_MERGE_METHOD}_level2downsample${DOWNSAMPLE_FACTOR_LEVEL_2}"
        
        elif [[ "${MERGE_METHOD}" == "downsample_qkv_upsample_out" ]]; then
            LOG_NAME+="_mergemethod${MERGE_METHOD}"
            LOG_NAME+="_downsamplemethod${DOWNSAMPLE_METHOD}" # General downsample method for the operation
            LOG_NAME+="_qkv_downsamplemethod${QKV_DOWNSAMPLE_METHOD}"
            LOG_NAME+="_out_upsamplemethod${OUT_UPSAMPLE_METHOD}"
            LOG_NAME+="_level1downsample${DOWNSAMPLE_FACTOR}"
            LOG_NAME+="_level2downsample${DOWNSAMPLE_FACTOR_LEVEL_2}"
            # Add linear_blend parameters to name if using linear_blend methods
            if [[ "${DOWNSAMPLE_METHOD}" == "linear_blend" ]]; then
                if [[ ${LINEAR_BLEND_TIMESTEP_INTERPOLATION} -eq 1 ]]; then
                    LOG_NAME+="_linearblend_timestep${LINEAR_BLEND_START_RATIO}-${LINEAR_BLEND_END_RATIO}"
                else
                    LOG_NAME+="_linearblend${LINEAR_BLEND_FACTOR}"
                fi
                LOG_NAME+="_blendmethods${LINEAR_BLEND_METHOD_1}-${LINEAR_BLEND_METHOD_2}"
            fi
            if [[ "${QKV_DOWNSAMPLE_METHOD}" == "linear_blend" ]]; then
                if [[ ${QKV_LINEAR_BLEND_TIMESTEP_INTERPOLATION} -eq 1 ]]; then
                    LOG_NAME+="_qkvblend_timestep${QKV_LINEAR_BLEND_START_RATIO}-${QKV_LINEAR_BLEND_END_RATIO}"
                else
                    LOG_NAME+="_qkvblend${QKV_LINEAR_BLEND_FACTOR}"
                fi
                LOG_NAME+="_qkvblendmethods${QKV_LINEAR_BLEND_METHOD_1}-${QKV_LINEAR_BLEND_METHOD_2}"
            fi
            if [[ "${OUT_UPSAMPLE_METHOD}" == "linear_blend" ]]; then
                if [[ ${OUT_LINEAR_BLEND_TIMESTEP_INTERPOLATION} -eq 1 ]]; then
                    LOG_NAME+="_outblend_timestep${OUT_LINEAR_BLEND_START_RATIO}-${OUT_LINEAR_BLEND_END_RATIO}"
                else
                    LOG_NAME+="_outblend${OUT_LINEAR_BLEND_FACTOR}"
                fi
                LOG_NAME+="_outblendmethods${OUT_LINEAR_BLEND_METHOD_1}-${OUT_LINEAR_BLEND_METHOD_2}"
            fi

        else
        LOG_NAME+="_tokendownsampling${RATIO_LEVEL_1}_userand${TOKEN_MERGING_USE_RAND}_method${MERGE_METHOD}_down${DOWNSAMPLE_FACTOR}_downmethod${DOWNSAMPLE_METHOD}_selectionsource${SELECTION_SOURCE}_switch${TIMESTEP_THRESHOLD_SWITCH}_stop${TIMESTEP_THRESHOLD_STOP}"
        # Add linear_blend parameter to name if using linear_blend method
        if [[ "${DOWNSAMPLE_METHOD}" == "linear_blend" ]]; then
            if [[ ${LINEAR_BLEND_TIMESTEP_INTERPOLATION} -eq 1 ]]; then
                LOG_NAME+="_linearblend_timestep${LINEAR_BLEND_START_RATIO}-${LINEAR_BLEND_END_RATIO}"
            else
                LOG_NAME+="_linearblend${LINEAR_BLEND_FACTOR}"
            fi
            LOG_NAME+="_blendmethods${LINEAR_BLEND_METHOD_1}-${LINEAR_BLEND_METHOD_2}"
        fi
        fi
    elif [[ ${IF_SITO} -eq 1 ]]; then
        LOG_NAME+="_sito_prune${SITO_PRUNE_RATIO}_maxdownsample${SITO_MAX_DOWNSAMPLE_RATIO}_selfattn${SITO_PRUNE_SELFATTN_FLAG}_crossattn${SITO_PRUNE_CROSSATTN_FLAG}_mlp${SITO_PRUNE_MLP_FLAG}_sx${SITO_SX}_sy${SITO_SY}_noisealpha${SITO_NOISE_ALPHA}_simbeta${SITO_SIM_BETA}"
    elif [[ ${IF_SCORING_MERGE} -eq 1 ]]; then
        LOG_NAME+="_scoringmerge${RATIO_LEVEL_1}_method${SCORING_METHOD}_preserve${SCORING_PRESERVE_RATIO}_mode${SCORING_MODE}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}"

		# Add merge layer types to name (same logic as token merging)
		merge_layers=""
		if [[ ${MERGE_ATTN} -eq 1 ]]; then
			merge_layers+="attn"
		fi
		if [[ ${MERGE_CROSSATTN} -eq 1 ]]; then
			if [[ -n "${merge_layers}" ]]; then
				merge_layers+="+crossattn"
			else
				merge_layers+="crossattn"
			fi
		fi
		if [[ ${MERGE_MLP} -eq 1 ]]; then
			if [[ -n "${merge_layers}" ]]; then
				merge_layers+="+mlp"
			else
				merge_layers+="mlp"
			fi
		fi
		if [[ -n "${merge_layers}" ]]; then
			LOG_NAME+="_merge${merge_layers}"
		fi
        
        # Add merging method to name if not default
        if [[ "${TOKEN_MERGING_METHOD}" != "mean" ]]; then
            LOG_NAME+="_mergemethod${TOKEN_MERGING_METHOD}"
        fi
        
        # Add spatial uniformity parameter to name
        if [[ ${SCORING_PRESERVE_SPATIAL_UNIFORMITY} -eq 1 ]]; then
            LOG_NAME+="_spatialuniform"
        fi
        
        # Add score-guided destination selection parameter to name
        if [[ ${IF_LOW_FREQUENCY_DST_TOKENS} -eq 1 ]]; then
            LOG_NAME+="_lowfreqdst"
        fi
        
        # Add method-specific parameters to name
        if [[ "${SCORING_METHOD}" == "frequency" ]]; then
            LOG_NAME+="_freqmethod${SCORING_FREQ_METHOD}_freqranking${SCORING_FREQ_RANKING}"
        elif [[ "${SCORING_METHOD}" == "spatial_filter" ]]; then
            LOG_NAME+="_spatialmethod${SCORING_SPATIAL_METHOD}_spatialnorm${SCORING_SPATIAL_NORM}"
        elif [[ "${SCORING_METHOD}" == "statistical" ]]; then
            LOG_NAME+="_statmethod${SCORING_STAT_METHOD}"
        elif [[ "${SCORING_METHOD}" == "signal_processing" ]]; then
            LOG_NAME+="_signalmethod${SCORING_SIGNAL_METHOD}"
        elif [[ "${SCORING_METHOD}" == "spatial_distribution" ]]; then
            LOG_NAME+="_spatialalpha${SCORING_SPATIAL_ALPHA}"
        elif [[ "${SCORING_METHOD}" == "similarity" ]]; then
            LOG_NAME+="_similaritymethod${SCORING_SIMILARITY_METHOD}"
        fi
        if [[ ${TOKEN_MERGING_CACHE_INDICES_PER_IMAGE} -eq 1 ]]; then
            LOG_NAME+="_cacheindices"
        fi
        if [[ ${IF_PROPORTIONAL_ATTENTION} -eq 1 ]]; then
            LOG_NAME+="_propattn"
        fi
        # Add locality-based sub-block parameters if not default
        if [[ ${LOCALITY_BLOCK_FACTOR_H} -ne 1 || ${LOCALITY_BLOCK_FACTOR_W} -ne 1 ]]; then
            LOG_NAME+="_locality_h${LOCALITY_BLOCK_FACTOR_H}_w${LOCALITY_BLOCK_FACTOR_W}"
        fi
        
        # Add resolution caching parameters if enabled
        if [[ ${CACHE_RESOLUTION_MERGE} -eq 1 ]]; then
            LOG_NAME+="_rescache${CACHE_RESOLUTION_MODE}"
        fi
        
        # Add ABP parameters if using ABP algorithm
        if [[ "${SCORING_MATCHING_ALGORITHM}" == "abp" ]]; then
            LOG_NAME+="_alg${SCORING_MATCHING_ALGORITHM}_scorer${ABP_SCORER_METHOD}_agg${ABP_TILE_AGGREGATION}"
        elif [[ "${SCORING_MATCHING_ALGORITHM}" != "bipartite" ]]; then
            LOG_NAME+="_alg${SCORING_MATCHING_ALGORITHM}"
        fi
    elif [[ ${IF_AGENT_GUIDED} -eq 1 ]]; then
        # Agent-guided token downsampling naming
        if [[ "${AGENT_BASE_METHOD}" != "None" && -n "${AGENT_BASE_METHOD}" && "${AGENT_WEIGHT}" != "1.0" ]]; then
            # Hybrid approach
            LOG_NAME+="_agenthybrid${RATIO_LEVEL_1}_agentmethod${AGENT_METHOD}_importance${AGENT_IMPORTANCE_METHOD}_agents${NUM_AGENTS}_basemethod${AGENT_BASE_METHOD}_baseranking${AGENT_BASE_RANKING}_weight${AGENT_WEIGHT}_preserve${AGENT_PRESERVE_RATIO}_mode${AGENT_SCORE_MODE}"
        else
            # Pure agent approach
            LOG_NAME+="_agentpure${RATIO_LEVEL_1}_agentmethod${AGENT_METHOD}_importance${AGENT_IMPORTANCE_METHOD}_agents${NUM_AGENTS}_preserve${AGENT_PRESERVE_RATIO}_mode${AGENT_SCORE_MODE}"
        fi
        
        # Add merging method to name if not default
        if [[ "${TOKEN_MERGING_METHOD}" != "mean" ]]; then
            LOG_NAME+="_mergemethod${TOKEN_MERGING_METHOD}"
        fi
        
        # Add spatial uniformity parameter to name
        if [[ ${AGENT_PRESERVE_SPATIAL_UNIFORMITY} -eq 1 ]]; then
            LOG_NAME+="_spatialuniform"
        fi
        
        # Add grid parameters
        LOG_NAME+="_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}"
        
        # Add cache indices if enabled
        if [[ ${TOKEN_MERGING_CACHE_INDICES_PER_IMAGE} -eq 1 ]]; then
            LOG_NAME+="_cacheindices"
        fi
        
        # Add proportional attention if enabled
        if [[ ${IF_PROPORTIONAL_ATTENTION} -eq 1 ]]; then
            LOG_NAME+="_propattn"
        fi
        
        # Add locality-based sub-block parameters if not default
        if [[ ${LOCALITY_BLOCK_FACTOR_H} -ne 1 || ${LOCALITY_BLOCK_FACTOR_W} -ne 1 ]]; then
            LOG_NAME+="_locality_h${LOCALITY_BLOCK_FACTOR_H}_w${LOCALITY_BLOCK_FACTOR_W}"
        fi
    fi

    if [[ ${EVAL_DENOISE_STEPS} -gt 1 ]]; then
        LOG_NAME+="_denoise_steps${EVAL_DENOISE_STEPS}_stride${EVAL_STEP_STRIDE}"
        LOG_NAME+="_error_method${EVAL_ERROR_METHOD}"
    fi

    if [[ ${FIX_NOISE_ACROSS_TIMESTEP} -eq 1 ]]; then
        LOG_NAME+="_fixed_noise"
    fi

    echo "${LOG_NAME}"
}

# Function to construct and run the python command
run_experiment() {
    # Build the unique name for this configuration
    EXPERIMENT_NAME=$(build_experiment_name)
    
    # Make sure the log directory exists
    mkdir -p "${LOG_DIR}"
    
    # Final log file path
    LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.out"

    # Build the command with all arguments
    COMMAND="python eval_prob_adaptive.py \
        --dataset ${DATASET} \
        --split ${SPLIT} \
        --n_trials ${N_TRIALS} \
        --batch_size ${BATCH_SIZE} \
        --to_keep ${TO_KEEP} \
        --n_samples ${N_SAMPLES} \
        --loss ${LOSS} \
        --prompt_path ${PROMPT_PATH} \
        --n_workers ${N_WORKERS} \
        --worker_idx ${WORKER_IDX} \
        --if_token_merging ${IF_TOKEN_MERGING} \
        --if_attention_proc ${IF_ATTENTION_PROC} \
        --if_sito ${IF_SITO} \
        --downsample_factor ${DOWNSAMPLE_FACTOR} \
        --token_merging_ratio ${RATIO_LEVEL_1} \
        --token_merging_method ${TOKEN_MERGING_METHOD} \
        --token_merging_use_rand ${TOKEN_MERGING_USE_RAND} \
        --merge_tokens ${MERGE_TOKENS} \
        --merge_method ${MERGE_METHOD} \
        --downsample_method ${DOWNSAMPLE_METHOD} \
        --linear_blend_factor ${LINEAR_BLEND_FACTOR} \
        --qkv_linear_blend_factor ${QKV_LINEAR_BLEND_FACTOR} \
        --out_linear_blend_factor ${OUT_LINEAR_BLEND_FACTOR} \
        --blockwise_blend_factor ${BLOCKWISE_BLEND_FACTOR} \
        --linear_blend_method_1 ${LINEAR_BLEND_METHOD_1} \
        --linear_blend_method_2 ${LINEAR_BLEND_METHOD_2} \
        --qkv_linear_blend_method_1 ${QKV_LINEAR_BLEND_METHOD_1} \
        --qkv_linear_blend_method_2 ${QKV_LINEAR_BLEND_METHOD_2} \
        --out_linear_blend_method_1 ${OUT_LINEAR_BLEND_METHOD_1} \
        --out_linear_blend_method_2 ${OUT_LINEAR_BLEND_METHOD_2} \
        --linear_blend_timestep_interpolation ${LINEAR_BLEND_TIMESTEP_INTERPOLATION} \
        --linear_blend_start_ratio ${LINEAR_BLEND_START_RATIO} \
        --linear_blend_end_ratio ${LINEAR_BLEND_END_RATIO} \
        --qkv_linear_blend_timestep_interpolation ${QKV_LINEAR_BLEND_TIMESTEP_INTERPOLATION} \
        --qkv_linear_blend_start_ratio ${QKV_LINEAR_BLEND_START_RATIO} \
        --qkv_linear_blend_end_ratio ${QKV_LINEAR_BLEND_END_RATIO} \
        --out_linear_blend_timestep_interpolation ${OUT_LINEAR_BLEND_TIMESTEP_INTERPOLATION} \
        --out_linear_blend_start_ratio ${OUT_LINEAR_BLEND_START_RATIO} \
        --out_linear_blend_end_ratio ${OUT_LINEAR_BLEND_END_RATIO} \
        --frequency_selection_mode ${FREQUENCY_SELECTION_MODE} \
        --frequency_selection_method ${FREQUENCY_SELECTION_METHOD} \
        --frequency_ranking_method ${FREQUENCY_RANKING_METHOD} \
        --timestep_threshold_switch ${TIMESTEP_THRESHOLD_SWITCH} \
        --timestep_threshold_stop ${TIMESTEP_THRESHOLD_STOP} \
        --secondary_merge_method ${SECONDARY_MERGE_METHOD} \
        --ratio_level_2 ${RATIO_LEVEL_2} \
        --extra_guidance_scale ${EXTRA_GUIDANCE_SCALE} \
        --frequency_grid_alpha ${FREQUENCY_GRID_ALPHA} \
        --eval_denoise_steps ${EVAL_DENOISE_STEPS} \
        --eval_error_method ${EVAL_ERROR_METHOD} \
        --single_timestep ${SINGLE_TIMESTEP} \
        --seed ${SEED} \
        --version ${VERSION} \
        --img_size ${IMG_SIZE} \
        --token_merging_sx ${TOKEN_MERGING_SX} \
        --token_merging_sy ${TOKEN_MERGING_SY} \
        --token_merging_max_downsample ${TOKEN_MERGING_MAX_DOWNSAMPLE} \
        --token_merging_single_downsample_level_merge ${TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE}\
        --token_merging_cache_indices_per_image ${TOKEN_MERGING_CACHE_INDICES_PER_IMAGE} \
        --eval_step_stride ${EVAL_STEP_STRIDE} \
        --qkv_downsample_method ${QKV_DOWNSAMPLE_METHOD} \
        --out_upsample_method ${OUT_UPSAMPLE_METHOD} \
        --sito_prune_ratio ${SITO_PRUNE_RATIO} \
        --sito_max_downsample_ratio ${SITO_MAX_DOWNSAMPLE_RATIO} \
        --sito_prune_selfattn_flag ${SITO_PRUNE_SELFATTN_FLAG} \
        --sito_prune_crossattn_flag ${SITO_PRUNE_CROSSATTN_FLAG} \
        --sito_prune_mlp_flag ${SITO_PRUNE_MLP_FLAG} \
        --sito_sx ${SITO_SX} \
        --sito_sy ${SITO_SY} \
        --sito_noise_alpha ${SITO_NOISE_ALPHA} \
        --sito_sim_beta ${SITO_SIM_BETA} \
        --merge_attn ${MERGE_ATTN} \
        --merge_crossattn ${MERGE_CROSSATTN} \
        --merge_mlp ${MERGE_MLP} \
        --if_scoring_merge ${IF_SCORING_MERGE} \
        --scoring_method ${SCORING_METHOD} \
        --scoring_preserve_ratio ${SCORING_PRESERVE_RATIO} \
        --scoring_mode ${SCORING_MODE} \
        --scoring_preserve_spatial_uniformity ${SCORING_PRESERVE_SPATIAL_UNIFORMITY} \
        --if_low_frequency_dst_tokens ${IF_LOW_FREQUENCY_DST_TOKENS} \
        --scoring_freq_method ${SCORING_FREQ_METHOD} \
        --scoring_freq_ranking ${SCORING_FREQ_RANKING} \
        --scoring_spatial_method ${SCORING_SPATIAL_METHOD} \
        --scoring_spatial_norm ${SCORING_SPATIAL_NORM} \
        --scoring_stat_method ${SCORING_STAT_METHOD} \
        --scoring_signal_method ${SCORING_SIGNAL_METHOD} \
        --scoring_spatial_alpha ${SCORING_SPATIAL_ALPHA} \
        --scoring_similarity_method ${SCORING_SIMILARITY_METHOD} \
        --if_agent_guided ${IF_AGENT_GUIDED} \
        --agent_method ${AGENT_METHOD} \
        --agent_importance_method ${AGENT_IMPORTANCE_METHOD} \
        --num_agents ${NUM_AGENTS} \
        --agent_base_method ${AGENT_BASE_METHOD} \
        --agent_base_ranking ${AGENT_BASE_RANKING} \
        --agent_weight ${AGENT_WEIGHT} \
        --agent_preserve_ratio ${AGENT_PRESERVE_RATIO} \
        --agent_score_mode ${AGENT_SCORE_MODE} \
        --agent_preserve_spatial_uniformity ${AGENT_PRESERVE_SPATIAL_UNIFORMITY} \
        --if_proportional_attention ${IF_PROPORTIONAL_ATTENTION} \
        --cache_resolution_merge ${CACHE_RESOLUTION_MERGE} \
        --cache_resolution_mode ${CACHE_RESOLUTION_MODE} \
        --debug_cache ${DEBUG_CACHE} \
        --scoring_matching_algorithm ${SCORING_MATCHING_ALGORITHM} \
        --abp_scorer_method ${ABP_SCORER_METHOD} \
        --abp_tile_aggregation ${ABP_TILE_AGGREGATION} \
        --locality_block_factor_h ${LOCALITY_BLOCK_FACTOR_H} \
        --locality_block_factor_w ${LOCALITY_BLOCK_FACTOR_W} \
        --force_recalc" # --force_recalc load_stats

    # Add fix_noise_across_timestep flag if enabled
    if [[ ${FIX_NOISE_ACROSS_TIMESTEP} -eq 1 ]]; then
        COMMAND="${COMMAND} --fix_noise_across_timestep"
    fi

    # --- Execute The Command ---
    echo "======================================================"
    echo "Starting Experiment: ${EXPERIMENT_NAME}"
    echo "Logging to: ${LOG_FILE}"
    echo "------------------------------------------------------"
    
    # Write command to log file for reproducibility
    echo "COMMAND:" > "${LOG_FILE}"
    echo "${COMMAND}" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
    echo "OUTPUT:" >> "${LOG_FILE}"
    
    # Execute the command and append stdout/stderr to the log file
    eval "${COMMAND}" >> "${LOG_FILE}" 2>&1
    
    echo "Experiment finished. Results saved to: ${LOG_FILE}"
    echo "======================================================"
    echo ""
}

# =================================================================================
# EXAMPLE EXPERIMENTS
# =================================================================================
# Uncomment one of the following examples to run experiments.
# All examples use the configuration parameters defined above.

# --- Baseline (No Acceleration) ---
# IF_TOKEN_MERGING=0; IF_ATTENTION_PROC=0; IF_SITO=0; IF_SCORING_MERGE=0; IF_AGENT_GUIDED=0
# run_experiment

# --- Original ToMe ---
# IF_TOKEN_MERGING=1; IF_ATTENTION_PROC=0; IF_SITO=0; IF_SCORING_MERGE=0; IF_AGENT_GUIDED=0
# TOKEN_MERGING_RATIO=0.7
# TOKEN_MERGING_METHOD="mean"
# run_experiment

# --- LGTM: Scoring-based Merging ---
# IF_TOKEN_MERGING=0; IF_ATTENTION_PROC=0; IF_SITO=0; IF_SCORING_MERGE=1; IF_AGENT_GUIDED=0
# TOKEN_MERGING_RATIO=0.7
# SCORING_METHOD="spatial_filter"
# SCORING_SPATIAL_METHOD="2d_conv"
# SCORING_SPATIAL_NORM="l1"
# SCORING_MODE="low"
# SCORING_PRESERVE_RATIO=0
# run_experiment

# --- Nearest-Exact Downsampling (ToDo) ---
# IF_TOKEN_MERGING=0; IF_ATTENTION_PROC=1; IF_SITO=0; IF_SCORING_MERGE=0; IF_AGENT_GUIDED=0
# MERGE_METHOD="downsample"
# DOWNSAMPLE_METHOD="nearest-exact"
# DOWNSAMPLE_FACTOR=2
# run_experiment

# --- IEKVD: Linear Blend Downsampling ---
# IF_TOKEN_MERGING=0; IF_ATTENTION_PROC=1; IF_SITO=0; IF_SCORING_MERGE=0; IF_AGENT_GUIDED=0
# MERGE_METHOD="downsample"
# DOWNSAMPLE_METHOD="linear_blend"
# DOWNSAMPLE_FACTOR=2
# LINEAR_BLEND_FACTOR=0.5
# LINEAR_BLEND_METHOD_1="nearest-exact"
# LINEAR_BLEND_METHOD_2="avg_pool"
# run_experiment

# --- SiTo Pruning ---
# IF_TOKEN_MERGING=0; IF_ATTENTION_PROC=0; IF_SITO=1; IF_SCORING_MERGE=0; IF_AGENT_GUIDED=0
# SITO_PRUNE_RATIO=0.7
# run_experiment

echo ""
echo "All experiments completed."
echo "Results saved to: ${LOG_DIR}"
