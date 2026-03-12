#!/bin/bash

# Stable Diffusion Generation Script
# This script launches image generation tasks using stable diffusion, creating uniquely named output
# directories for each configuration

# --- Core Model & Generation Configuration ---
IMG_SIZE=512              # 256 or 512
SEED=0
STEPS=50                  # Number of denoising steps
BATCH_SIZE=16            # Adjust based on GPU memory (SD typically uses smaller batches than DiT)
DTYPE="float16"           # float16 or float32
QUALITY=100               # JPEG quality for saved images (95 is recommended for FID)
VERSION="2-0"             # Stable diffusion version: 1-1, 1-2, 1-3, 1-4, 1-5, 2-0, 2-1

# --- Dataset & Paths Configuration ---
DATASET="imagenet"
# Base directory where all generated images and logs will be saved
BASE_OUTPUT_DIR="/path/to/results/sd_generation"

# --- Stable Diffusion Specific Parameters ---
GUIDANCE_SCALE=7.5        # Classifier-free guidance scale
INTERPOLATION="bicubic"   # Image interpolation method

# --- Acceleration Method Flags (Set to 1 to enable a method) ---
IF_TOKEN_MERGING=0
IF_AGENTSD=0
IF_ATTENTION_PROC=0
IF_DEEPCACHE=0
IF_SITO=0
IF_SCORING_MERGE=0

# --- Token Merging & AgentSD Parameters ---
TOKEN_MERGING_RATIO=0
AGENT_RATIO=0
TOKEN_MERGING_USE_RAND=1
TOKEN_MERGING_SX=2
TOKEN_MERGING_SY=2
TOKEN_MERGING_MAX_DOWNSAMPLE=1
TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE=0
MERGE_ATTN=1
MERGE_CROSSATTN=0
MERGE_MLP=0

# --- New Token Merging Features ---
TOKEN_MERGING_METHOD="mean"  # Options: mean, mlerp, prune
IF_PROPORTIONAL_ATTENTION=0  # 0/1 flag for proportional attention

# --- Attention Processor Parameters ---
MERGE_TOKENS="keys/values"  # Options: "keys/values" all
MERGE_METHOD="downsample"   # Options: downsample, similarity, frequency_global, frequency_blockwise
DOWNSAMPLE_METHOD="nearest-exact"  # Options: 'nearest-exact', 'max_pool','avg_pool', 'area', "bilinear", "bicubic",'top_right','bottom_left','bottom_right','random',uniform_random,uniform_timestep,'linear_blend'
DOWNSAMPLE_FACTOR=2
DOWNSAMPLE_FACTOR_LEVEL_2=1
LINEAR_BLEND_FACTOR=0.5           # Blend factor for linear_blend method: 0.0=avg_pool, 0.5=50/50 blend, 1.0=nearest-exact
QKV_LINEAR_BLEND_FACTOR=0.5       # Blend factor for QKV linear_blend in downsample_qkv_upsample_out method
OUT_LINEAR_BLEND_FACTOR=0.5       # Blend factor for output linear_blend in downsample_qkv_upsample_out method

# Timestep-based interpolation for linear blend (dynamic blend factors)
LINEAR_BLEND_TIMESTEP_INTERPOLATION=0  # 0/1 flag to enable timestep-based interpolation
LINEAR_BLEND_START_RATIO=0.1          # Start ratio at timestep 999 (high noise)
LINEAR_BLEND_END_RATIO=0.9            # End ratio at timestep 0 (low noise)
QKV_LINEAR_BLEND_TIMESTEP_INTERPOLATION=0  # Enable for QKV
QKV_LINEAR_BLEND_START_RATIO=0.1
QKV_LINEAR_BLEND_END_RATIO=0.9
OUT_LINEAR_BLEND_TIMESTEP_INTERPOLATION=0  # Enable for output
OUT_LINEAR_BLEND_START_RATIO=0.1
OUT_LINEAR_BLEND_END_RATIO=0.9

# Linear blend method selection parameters (enhanced functionality)
LINEAR_BLEND_METHOD_1="nearest-exact" # First method for linear_blend: max_pool, avg_pool, area, bilinear, bicubic, nearest-exact, top_right, bottom_left, bottom_right, random, uniform_random, uniform_timestep
LINEAR_BLEND_METHOD_2="avg_pool"       # Second method for linear_blend (default: avg_pool)
QKV_LINEAR_BLEND_METHOD_1="nearest-exact" # First method for QKV linear_blend in downsample_qkv_upsample_out method
QKV_LINEAR_BLEND_METHOD_2="avg_pool"   # Second method for QKV linear_blend (default: avg_pool)
OUT_LINEAR_BLEND_METHOD_1="nearest-exact" # First method for output linear_blend in downsample_qkv_upsample_out method
OUT_LINEAR_BLEND_METHOD_2="avg_pool"   # Second method for output linear_blend (default: avg_pool)
BLOCKWISE_BLEND_FACTOR=0.5        # Blend factor for frequency_blockwise method
QKV_DOWNSAMPLE_METHOD="nearest"   # Method for downsampling QKV source (e.g., "avg_pool", "max_pool", "linear_blend")
OUT_UPSAMPLE_METHOD="nearest"     # Method for upsampling attention output (e.g., "nearest", "bilinear", "linear_blend")
TIMESTEP_THRESHOLD_SWITCH=0.0
TIMESTEP_THRESHOLD_STOP=0
SECONDARY_MERGE_METHOD="similarity"
RATIO_LEVEL_2=0.0
SELECTION_SOURCE="hidden"
EXTRA_GUIDANCE_SCALE=0

# --- Frequency Method Parameters ---
FREQUENCY_SELECTION_MODE="high"       # high or low
FREQUENCY_SELECTION_METHOD="original"   #['original','1d_dft', '1d_dct', '2d_conv','non_uniform_grid']
FREQUENCY_RANKING_METHOD="mean_deviation"  # choices=['spectral_centroid',  'amplitude',"variance", "l1norm", "l2norm","mean_deviation"]
FREQUENCY_GRID_ALPHA=2.0

# --- DeepCache Configuration ---
CACHE_INTERVAL=1001
CACHE_BRANCH_ID=0
ORIGINAL=1  # Set to 1 to use original diffusers pipeline

# --- SiTo Parameters ---
SITO_PRUNE_RATIO=0.5
SITO_MAX_DOWNSAMPLE_RATIO=1
SITO_PRUNE_SELFATTN_FLAG=1
SITO_PRUNE_CROSSATTN_FLAG=0
SITO_PRUNE_MLP_FLAG=0
SITO_SX=2
SITO_SY=2
SITO_NOISE_ALPHA=0.1
SITO_SIM_BETA=1.0

# --- Scoring-based Token Merging Parameters ---
SCORING_METHOD="statistical" # Options: 'frequency', 'spatial_filter', 'statistical', 'signal_processing', 'spatial_distribution'
SCORING_PRESERVE_RATIO=0.3 # Ratio of tokens to protect from merging (0.0 to 1.0)
SCORING_MODE="high" # Options: 'high', 'low', 'medium', 'timestep_scheduler', 'reverse_timestep_scheduler'
SCORING_PRESERVE_SPATIAL_UNIFORMITY=0 # 0/1 flag for preserving spatial uniformity in scoring-based merging
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
SCORING_SIMILARITY_METHOD="local_neighbors_inverted" # Options: 'local_neighbors_inverted', 'cosine_similarity', 'global_mean'

# --- ABP (Adaptive Block Pooling) Parameters (NEW) ---
SCORING_MATCHING_ALGORITHM="bipartite" # Options: 'bipartite' (standard), 'abp' (Adaptive Block Pooling - faster)
ABP_SCORER_METHOD="spatial_filter" # Options: 'frequency', 'spatial_filter', 'statistical', 'signal_processing', 'spatial_distribution', 'similarity'
ABP_TILE_AGGREGATION="max" # Options: 'max', 'min', 'sum', 'std' (how to aggregate scores within tiles)

# --- Resolution Caching Parameters (NEW) ---
CACHE_RESOLUTION_MERGE=0 # Whether to enable within-UNet resolution caching for scoring merge (0/1)
CACHE_RESOLUTION_MODE="global" # Caching mode: "global" (faster) or "block_specific" (more precise)
DEBUG_CACHE=0 # Whether to enable debug prints for resolution caching (0/1)

# --- Attention Processor Consistency (NEW) ---
FORCE_ATTENTION_PROCESSOR="none" # Options: AttnProcessor, AttnProcessor2_0, XFormersAttnProcessor, none

# --- Locality-based Sub-block Bipartite Matching Parameters (NEW) ---
LOCALITY_BLOCK_FACTOR_H=1 # Factor to divide height for locality-based similarity (1=global, >1=local sub-blocks)
LOCALITY_BLOCK_FACTOR_W=1 # Factor to divide width for locality-based similarity (1=global, >1=local sub-blocks)

# =================================================================================
# SCRIPT LOGIC
# =================================================================================

# Function to build the experiment name based on current parameters
build_experiment_name() {
    local name="SD${VERSION}_${IMG_SIZE}"

    # Add acceleration method details
    if [[ ${IF_TOKEN_MERGING} -eq 1 ]]; then
        name+="_tokenmerge${TOKEN_MERGING_RATIO}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}_singlelevel${TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}"
        # Add method if not default
        if [[ "${TOKEN_MERGING_METHOD}" != "mean" ]]; then
            name+="_method${TOKEN_MERGING_METHOD}"
        fi
        # Add proportional attention if enabled
        if [[ ${IF_PROPORTIONAL_ATTENTION} -eq 1 ]]; then
            name+="_propattn"
        fi
        # Add locality-based sub-block parameters if not default
        if [[ ${LOCALITY_BLOCK_FACTOR_H} -ne 1 || ${LOCALITY_BLOCK_FACTOR_W} -ne 1 ]]; then
            name+="_locality_h${LOCALITY_BLOCK_FACTOR_H}_w${LOCALITY_BLOCK_FACTOR_W}"
        fi
    elif [[ ${IF_AGENTSD} -eq 1 ]]; then
        name+="_agentsd${TOKEN_MERGING_RATIO}_agentratio${AGENT_RATIO}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}_singlelevel${TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}"
        # Add locality-based sub-block parameters if not default
        if [[ ${LOCALITY_BLOCK_FACTOR_H} -ne 1 || ${LOCALITY_BLOCK_FACTOR_W} -ne 1 ]]; then
            name+="_locality_h${LOCALITY_BLOCK_FACTOR_H}_w${LOCALITY_BLOCK_FACTOR_W}"
        fi
    elif [[ ${IF_ATTENTION_PROC} -eq 1 ]]; then
        if [[ "${MERGE_METHOD}" == "frequency_global" ]]; then
            name+="_freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_ratio${TOKEN_MERGING_RATIO}"
        elif [[ "${MERGE_METHOD}" == "frequency_blockwise" ]]; then
            name+="_freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_downsample${DOWNSAMPLE_FACTOR}"
            # Always add blockwise blend parameter to name
            name+="_blockwiseblend${BLOCKWISE_BLEND_FACTOR}"
        elif [[ "${MERGE_METHOD}" == "similarity" ]]; then
            name+="_similarityratio${TOKEN_MERGING_RATIO}_mergetoken${MERGE_TOKENS}_selectionsource${SELECTION_SOURCE}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}"
        elif [[ "${MERGE_METHOD}" == "downsample_qkv_upsample_out" ]]; then
            name+="_mergemethod${MERGE_METHOD}_downsamplemethod${DOWNSAMPLE_METHOD}_qkv_downsamplemethod${QKV_DOWNSAMPLE_METHOD}_out_upsamplemethod${OUT_UPSAMPLE_METHOD}_level1downsample${DOWNSAMPLE_FACTOR}_level2downsample${DOWNSAMPLE_FACTOR_LEVEL_2}"
            # Always add linear_blend parameters to name
            if [[ ${LINEAR_BLEND_TIMESTEP_INTERPOLATION} -eq 1 ]]; then
                name+="_linearblend_timestep${LINEAR_BLEND_START_RATIO}-${LINEAR_BLEND_END_RATIO}"
            else
                name+="_linearblend${LINEAR_BLEND_FACTOR}"
            fi
            name+="_blendmethods${LINEAR_BLEND_METHOD_1}-${LINEAR_BLEND_METHOD_2}"
            
            if [[ ${QKV_LINEAR_BLEND_TIMESTEP_INTERPOLATION} -eq 1 ]]; then
                name+="_qkvblend_timestep${QKV_LINEAR_BLEND_START_RATIO}-${QKV_LINEAR_BLEND_END_RATIO}"
            else
                name+="_qkvblend${QKV_LINEAR_BLEND_FACTOR}"
            fi
            name+="_qkvblendmethods${QKV_LINEAR_BLEND_METHOD_1}-${QKV_LINEAR_BLEND_METHOD_2}"
            
            if [[ ${OUT_LINEAR_BLEND_TIMESTEP_INTERPOLATION} -eq 1 ]]; then
                name+="_outblend_timestep${OUT_LINEAR_BLEND_START_RATIO}-${OUT_LINEAR_BLEND_END_RATIO}"
            else
                name+="_outblend${OUT_LINEAR_BLEND_FACTOR}"
            fi
            name+="_outblendmethods${OUT_LINEAR_BLEND_METHOD_1}-${OUT_LINEAR_BLEND_METHOD_2}"
        else
            name+="_tokendownsampling${TOKEN_MERGING_RATIO}_userand${TOKEN_MERGING_USE_RAND}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}_down${DOWNSAMPLE_FACTOR}_downmethod${DOWNSAMPLE_METHOD}_selectionsource${SELECTION_SOURCE}_switch${TIMESTEP_THRESHOLD_SWITCH}_stop${TIMESTEP_THRESHOLD_STOP}"
                    # Add linear_blend_factor to name if using linear_blend method
            if [[ "${DOWNSAMPLE_METHOD}" == "linear_blend" ]]; then
                if [[ ${LINEAR_BLEND_TIMESTEP_INTERPOLATION} -eq 1 ]]; then
                    name+="_linearblend_timestep${LINEAR_BLEND_START_RATIO}-${LINEAR_BLEND_END_RATIO}"
                else
                    name+="_linearblend${LINEAR_BLEND_FACTOR}"
                fi
                name+="_blendmethods${LINEAR_BLEND_METHOD_1}-${LINEAR_BLEND_METHOD_2}"
            fi
        fi
    elif [[ ${IF_SITO} -eq 1 ]]; then
        name+="_sito_prune${SITO_PRUNE_RATIO}_maxdownsample${SITO_MAX_DOWNSAMPLE_RATIO}_selfattn${SITO_PRUNE_SELFATTN_FLAG}_crossattn${SITO_PRUNE_CROSSATTN_FLAG}_mlp${SITO_PRUNE_MLP_FLAG}_sx${SITO_SX}_sy${SITO_SY}_noisealpha${SITO_NOISE_ALPHA}_simbeta${SITO_SIM_BETA}"
    elif [[ ${IF_SCORING_MERGE} -eq 1 ]]; then
        name+="_scoringmerge${TOKEN_MERGING_RATIO}_method${SCORING_METHOD}_preserve${SCORING_PRESERVE_RATIO}_mode${SCORING_MODE}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}"
        
        # Add method-specific parameters to name
        if [[ "${SCORING_METHOD}" == "frequency" ]]; then
            name+="_freqmethod${SCORING_FREQ_METHOD}_freqranking${SCORING_FREQ_RANKING}"
        elif [[ "${SCORING_METHOD}" == "spatial_filter" ]]; then
            name+="_spatialmethod${SCORING_SPATIAL_METHOD}_spatialnorm${SCORING_SPATIAL_NORM}"
        elif [[ "${SCORING_METHOD}" == "statistical" ]]; then
            name+="_statmethod${SCORING_STAT_METHOD}"
        elif [[ "${SCORING_METHOD}" == "signal_processing" ]]; then
            name+="_signalmethod${SCORING_SIGNAL_METHOD}"
        elif [[ "${SCORING_METHOD}" == "spatial_distribution" ]]; then
            name+="_spatialalpha${SCORING_SPATIAL_ALPHA}"
        elif [[ "${SCORING_METHOD}" == "similarity" ]]; then
            name+="_similaritymethod${SCORING_SIMILARITY_METHOD}"
        fi
        
        # Add merging method if not default
        if [[ "${TOKEN_MERGING_METHOD}" != "mean" ]]; then
            name+="_mergemethod${TOKEN_MERGING_METHOD}"
        fi
        
        # Add spatial uniformity parameter if enabled
        if [[ ${SCORING_PRESERVE_SPATIAL_UNIFORMITY} -eq 1 ]]; then
            name+="_spatialuniform"
        fi
        
        # Add score-guided destination selection parameter if enabled
        if [[ ${IF_LOW_FREQUENCY_DST_TOKENS} -eq 1 ]]; then
            name+="_lowfreqdst"
        fi
        
        # Add proportional attention if enabled
        if [[ ${IF_PROPORTIONAL_ATTENTION} -eq 1 ]]; then
            name+="_propattn"
        fi
        
        # Add resolution caching parameters if enabled
        if [[ ${CACHE_RESOLUTION_MERGE} -eq 1 ]]; then
            name+="_rescache${CACHE_RESOLUTION_MODE}"
        fi
        
        # Add ABP parameters if using ABP algorithm
        if [[ "${SCORING_MATCHING_ALGORITHM}" == "abp" ]]; then
            name+="_alg${SCORING_MATCHING_ALGORITHM}_scorer${ABP_SCORER_METHOD}_agg${ABP_TILE_AGGREGATION}"
        elif [[ "${SCORING_MATCHING_ALGORITHM}" != "bipartite" ]]; then
            name+="_alg${SCORING_MATCHING_ALGORITHM}"
        fi
        
        # Add locality-based sub-block parameters if not default
        if [[ ${LOCALITY_BLOCK_FACTOR_H} -ne 1 || ${LOCALITY_BLOCK_FACTOR_W} -ne 1 ]]; then
            name+="_locality_h${LOCALITY_BLOCK_FACTOR_H}_w${LOCALITY_BLOCK_FACTOR_W}"
        fi
    fi

    if [[ ${IF_DEEPCACHE} -eq 1 ]]; then
        name+="_deepcache${CACHE_INTERVAL}"
    fi

    # If no acceleration is used, mark as baseline
    if [[ ${IF_TOKEN_MERGING} -eq 0 && ${IF_AGENTSD} -eq 0 && ${IF_ATTENTION_PROC} -eq 0 && ${IF_DEEPCACHE} -eq 0 && ${IF_SITO} -eq 0 ]]; then
        name+="_baseline"
    fi

    name+="_${STEPS}steps_seed${SEED}_guidance${GUIDANCE_SCALE}"
    echo "${name}"
}

# Function to construct and run the python command
run_experiment() {
    # Build the unique name for this configuration
    EXPERIMENT_NAME=$(build_experiment_name)
    
    # Define the full output directory path
    RUN_FOLDER="${BASE_OUTPUT_DIR}/${DATASET}/${EXPERIMENT_NAME}"
    
    # Create the directory and a log file for this run
    mkdir -p "${RUN_FOLDER}"
    LOG_FILE="${RUN_FOLDER}/generation_log.txt"

    # Set ground truth path based on dataset
    # Uses DATASET_ROOT environment variable - set this before running!
    if [[ -z "${DATASET_ROOT}" ]]; then
        echo "ERROR: DATASET_ROOT environment variable is not set."
        echo "Please set it to your datasets directory, e.g.: export DATASET_ROOT=/path/to/datasets"
        exit 1
    fi

    if [[ "${DATASET}" == "imagenet100" ]]; then
        GT_PATH_ARG="${DATASET_ROOT}/imagenet-100/val_flat"
    elif [[ "${DATASET}" == "imagenet" ]]; then
        GT_PATH_ARG="${DATASET_ROOT}/imagenet/val_flat"
    elif [[ "${DATASET}" == "pets" ]]; then
        GT_PATH_ARG="${DATASET_ROOT}/oxford-iiit-pet/images"
    elif [[ "${DATASET}" == "flowers" ]]; then
        GT_PATH_ARG="${DATASET_ROOT}/flowers-102/jpg"
    elif [[ "${DATASET}" == "coco2017" ]]; then
        GT_PATH_ARG="${DATASET_ROOT}/coco/images/val2017"
    else
        GT_PATH_ARG="${DATASET_ROOT}/${DATASET}/val"
    fi

    # Build the command with all arguments
    CMD="python sd_generation.py \
        --dataset ${DATASET} \
        --img_size ${IMG_SIZE} \
        --steps ${STEPS} \
        --batch_size ${BATCH_SIZE} \
        --dtype ${DTYPE} \
        --quality ${QUALITY} \
        --seed ${SEED} \
        --version ${VERSION} \
        --guidance_scale ${GUIDANCE_SCALE} \
        --interpolation ${INTERPOLATION} \
        --if_deepcache ${IF_DEEPCACHE} \
        --cache_interval ${CACHE_INTERVAL} \
        --cache_branch_id ${CACHE_BRANCH_ID} \
        --if_token_merging ${IF_TOKEN_MERGING} \
        --if_agentsd ${IF_AGENTSD} \
        --if_attention_proc ${IF_ATTENTION_PROC} \
        --if_sito ${IF_SITO} \
        --token_merging_ratio ${TOKEN_MERGING_RATIO} \
        --agent_ratio ${AGENT_RATIO} \
        --token_merging_use_rand ${TOKEN_MERGING_USE_RAND} \
        --token_merging_sx ${TOKEN_MERGING_SX} \
        --token_merging_sy ${TOKEN_MERGING_SY} \
        --token_merging_max_downsample ${TOKEN_MERGING_MAX_DOWNSAMPLE} \
        --token_merging_single_downsample_level_merge ${TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE} \
        --merge_attn ${MERGE_ATTN} \
        --merge_crossattn ${MERGE_CROSSATTN} \
        --merge_mlp ${MERGE_MLP} \
        --token_merging_method ${TOKEN_MERGING_METHOD} \
        --if_proportional_attention ${IF_PROPORTIONAL_ATTENTION} \
        --merge_tokens ${MERGE_TOKENS} \
        --merge_method ${MERGE_METHOD} \
        --downsample_method ${DOWNSAMPLE_METHOD} \
        --downsample_factor ${DOWNSAMPLE_FACTOR} \
        --linear_blend_factor ${LINEAR_BLEND_FACTOR} \
        --qkv_linear_blend_factor ${QKV_LINEAR_BLEND_FACTOR} \
        --out_linear_blend_factor ${OUT_LINEAR_BLEND_FACTOR} \
        --linear_blend_method_1 ${LINEAR_BLEND_METHOD_1} \
        --linear_blend_method_2 ${LINEAR_BLEND_METHOD_2} \
        --qkv_linear_blend_method_1 ${QKV_LINEAR_BLEND_METHOD_1} \
        --qkv_linear_blend_method_2 ${QKV_LINEAR_BLEND_METHOD_2} \
        --out_linear_blend_method_1 ${OUT_LINEAR_BLEND_METHOD_1} \
        --out_linear_blend_method_2 ${OUT_LINEAR_BLEND_METHOD_2} \
        --blockwise_blend_factor ${BLOCKWISE_BLEND_FACTOR} \
        --linear_blend_timestep_interpolation ${LINEAR_BLEND_TIMESTEP_INTERPOLATION} \
        --linear_blend_start_ratio ${LINEAR_BLEND_START_RATIO} \
        --linear_blend_end_ratio ${LINEAR_BLEND_END_RATIO} \
        --qkv_linear_blend_timestep_interpolation ${QKV_LINEAR_BLEND_TIMESTEP_INTERPOLATION} \
        --qkv_linear_blend_start_ratio ${QKV_LINEAR_BLEND_START_RATIO} \
        --qkv_linear_blend_end_ratio ${QKV_LINEAR_BLEND_END_RATIO} \
        --out_linear_blend_timestep_interpolation ${OUT_LINEAR_BLEND_TIMESTEP_INTERPOLATION} \
        --out_linear_blend_start_ratio ${OUT_LINEAR_BLEND_START_RATIO} \
        --out_linear_blend_end_ratio ${OUT_LINEAR_BLEND_END_RATIO} \
        --downsample_factor_level_2 ${DOWNSAMPLE_FACTOR_LEVEL_2} \
        --timestep_threshold_switch ${TIMESTEP_THRESHOLD_SWITCH} \
        --timestep_threshold_stop ${TIMESTEP_THRESHOLD_STOP} \
        --secondary_merge_method ${SECONDARY_MERGE_METHOD} \
        --ratio_level_2 ${RATIO_LEVEL_2} \
        --selection_source ${SELECTION_SOURCE} \
        --extra_guidance_scale ${EXTRA_GUIDANCE_SCALE} \
        --frequency_selection_mode ${FREQUENCY_SELECTION_MODE} \
        --frequency_selection_method ${FREQUENCY_SELECTION_METHOD} \
        --frequency_ranking_method ${FREQUENCY_RANKING_METHOD} \
        --frequency_grid_alpha ${FREQUENCY_GRID_ALPHA} \
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
        --cache_resolution_merge ${CACHE_RESOLUTION_MERGE} \
        --cache_resolution_mode ${CACHE_RESOLUTION_MODE} \
        --debug_cache ${DEBUG_CACHE} \
        --scoring_matching_algorithm ${SCORING_MATCHING_ALGORITHM} \
        --abp_scorer_method ${ABP_SCORER_METHOD} \
        --abp_tile_aggregation ${ABP_TILE_AGGREGATION} \
        --locality_block_factor_h ${LOCALITY_BLOCK_FACTOR_H} \
        --locality_block_factor_w ${LOCALITY_BLOCK_FACTOR_W} \
        --force_attention_processor ${FORCE_ATTENTION_PROCESSOR} \
        --gt_path ${GT_PATH_ARG} \
        --fid_batch_size 1 \
        --summary_csv ${RUN_FOLDER}/generation_experiments_summary.csv"

    # Add original flag if needed
    if [[ ${ORIGINAL} -eq 1 ]]; then
        CMD+=" --original"
    fi

    # --- Execute The Command ---
    echo "======================================================"
    echo "Starting Run: ${EXPERIMENT_NAME}"
    echo "Logging to: ${LOG_FILE}"
    echo "------------------------------------------------------"
    
    # Write command to log file for reproducibility
    echo "COMMAND:" > "${LOG_FILE}"
    echo "${CMD}" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
    echo "OUTPUT:" >> "${LOG_FILE}"
    
    # Execute the command and append stdout/stderr to the log file
    eval "${CMD}" >> "${LOG_FILE}" 2>&1
    
    echo "Run finished. Results saved in: ${RUN_FOLDER}"
    echo "======================================================"
    echo ""
}

# =================================================================================
# EXAMPLE EXPERIMENTS
# =================================================================================
# Uncomment one of the following examples to run experiments.
# All examples use the configuration parameters defined above.

# --- Baseline (No Acceleration) ---
# IF_TOKEN_MERGING=0; IF_AGENTSD=0; IF_ATTENTION_PROC=0; IF_DEEPCACHE=0; IF_SITO=0; IF_SCORING_MERGE=0
# run_experiment

# --- Original ToMe ---
# IF_TOKEN_MERGING=1; IF_AGENTSD=0; IF_ATTENTION_PROC=0; IF_DEEPCACHE=0; IF_SITO=0; IF_SCORING_MERGE=0
# TOKEN_MERGING_RATIO=0.7
# TOKEN_MERGING_METHOD="mean"
# run_experiment

# --- LGTM: Scoring-based Merging ---
# IF_TOKEN_MERGING=0; IF_AGENTSD=0; IF_ATTENTION_PROC=0; IF_DEEPCACHE=0; IF_SITO=0; IF_SCORING_MERGE=1
# TOKEN_MERGING_RATIO=0.7
# SCORING_METHOD="spatial_filter"
# SCORING_SPATIAL_METHOD="2d_conv"
# SCORING_SPATIAL_NORM="l1"
# SCORING_MODE="low"
# SCORING_PRESERVE_RATIO=0
# run_experiment

# --- Nearest-Exact Downsampling (ToDo) ---
# IF_TOKEN_MERGING=0; IF_AGENTSD=0; IF_ATTENTION_PROC=1; IF_DEEPCACHE=0; IF_SITO=0; IF_SCORING_MERGE=0
# MERGE_METHOD="downsample"
# DOWNSAMPLE_METHOD="nearest-exact"
# DOWNSAMPLE_FACTOR=2
# run_experiment

# --- IEKVD: Linear Blend Downsampling ---
# IF_TOKEN_MERGING=0; IF_AGENTSD=0; IF_ATTENTION_PROC=1; IF_DEEPCACHE=0; IF_SITO=0; IF_SCORING_MERGE=0
# MERGE_METHOD="downsample"
# DOWNSAMPLE_METHOD="linear_blend"
# DOWNSAMPLE_FACTOR=2
# LINEAR_BLEND_FACTOR=0.5
# LINEAR_BLEND_METHOD_1="nearest-exact"
# LINEAR_BLEND_METHOD_2="avg_pool"
# run_experiment

# --- SiTo Pruning ---
# IF_TOKEN_MERGING=0; IF_AGENTSD=0; IF_ATTENTION_PROC=0; IF_DEEPCACHE=0; IF_SITO=1; IF_SCORING_MERGE=0
# SITO_PRUNE_RATIO=0.7
# run_experiment

echo ""
echo "All experiments completed."
echo "Results saved to: ${BASE_OUTPUT_DIR}"
