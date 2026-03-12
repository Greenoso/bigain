#!/bin/bash

# DiT Generation Script with Detailed Parameter Logging & FID Calculation
# This script launches image generation tasks, creating uniquely named output
# directories for each configuration, and calculates FID scores upon completion.

# --- Core Model & Generation Configuration ---
IMG_SIZE=512              # 256 or 512
SEED=0
STEPS=50                  # Number of denoising steps
BATCH_SIZE=8             # Adjust based on GPU memory
DTYPE="float16"           # float16 or float32
QUALITY=100                # JPEG quality for saved images (95 is recommended for FID)

# --- Dataset & Paths Configuration ---
DATASET="imagenet"

# Validate DATASET_ROOT environment variable
if [[ -z "${DATASET_ROOT}" ]]; then
    echo "ERROR: DATASET_ROOT environment variable is not set."
    echo "Please set it to your datasets directory, e.g.: export DATASET_ROOT=/path/to/datasets"
    exit 1
fi

# Path to the ground truth dataset for FID calculation
# Set ground truth path based on dataset
if [[ "${DATASET}" == "imagenet100" ]]; then
    GT_PATH="${DATASET_ROOT}/imagenet-100/val_flat"
elif [[ "${DATASET}" == "imagenet" ]]; then
    GT_PATH="${DATASET_ROOT}/imagenet/val_flat"
else
    GT_PATH="${DATASET_ROOT}/${DATASET}/val"
fi
# Base directory where all generated images and logs will be saved
BASE_OUTPUT_DIR="./results/dit_generation"

# --- DiT Model Class Configuration ---
# Get the script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASS_CSV_PATH="${SCRIPT_DIR}/../prompts/imagenet_prompts.csv"
NUM_CLASSES=1000

# --- Acceleration Method Flags (Set to 1 to enable a method) ---
IF_TOKEN_MERGING=0
IF_AGENTSD=0
IF_ATTENTION_PROC=1
IF_DEEPCACHE=0
IF_SCORING_MERGE=0
IF_SITO=0

# --- Token Merging & AgentSD Parameters ---
TOKEN_MERGING_RATIO=0
AGENT_RATIO=0
TOKEN_MERGING_USE_RAND=1
TOKEN_MERGING_SX=2
TOKEN_MERGING_SY=2
TOKEN_MERGING_MAX_DOWNSAMPLE=2
TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE=0
MERGE_ATTN=1
MERGE_CROSSATTN=0
MERGE_MLP=0
TOKEN_MERGING_CACHE_INDICES_PER_IMAGE=0
TOKEN_MERGING_METHOD="mean"  # Options: mean, mlerp, prune
IF_PROPORTIONAL_ATTENTION=0  # 0/1 flag for proportional attention

# --- Block-level Caching Parameters (for DiT scoring merge acceleration) ---
CACHE_MERGE_FUNCTIONS=0     # 0/1 flag to enable block-level function caching (only works with scoring merge + block control)
CACHE_RECALC_INTERVAL=5      # Recalculate cached functions every N blocks (higher = more reuse = faster but potentially lower quality)

# --- Attention Processor Parameters ---
MERGE_TOKENS="keys/values"  # Options: "keys/values" all
MERGE_METHOD="downsample"   # Options: downsample, similarity, frequency_global, frequency_blockwise
DOWNSAMPLE_METHOD="nearest-exact"  # Options: 'nearest-exact', 'max_pool','avg_pool', 'area', "bilinear", "bicubic",'top_right','bottom_left','bottom_right','random',uniform_random,uniform_timestep,linear_blend
DOWNSAMPLE_FACTOR=2
DOWNSAMPLE_FACTOR_H=1 # only used by downsample_custom_block
DOWNSAMPLE_FACTOR_W=2 # only used by downsample_custom_block
# Linear blend parameters for downsample methods (only used when DOWNSAMPLE_METHOD="linear_blend")
BLEND_FACTOR=""  # Default: empty string (uses 0.5). Set to value like 0.7 for 70% method1, 30% method2
BLEND_METHOD_1=""  # Default: empty string (uses nearest-exact). Options: same as DOWNSAMPLE_METHOD
BLEND_METHOD_2=""  # Default: empty string (uses avg_pool). Options: same as DOWNSAMPLE_METHOD
TIMESTEP_THRESHOLD_SWITCH=0.0
TIMESTEP_THRESHOLD_STOP=0.0
SECONDARY_MERGE_METHOD="similarity"
RATIO_LEVEL_2=0.0
SELECTION_SOURCE="hidden"
BLOCK_TOME_FLAGS=""  # Comma-separated list of 0s and 1s for block-wise ToMe control (e.g., "1,1,1,0,0,0,..." for 28 blocks)
# Example usage for block flags:
BLOCK_TOME_FLAGS="0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"  
#BLOCK_TOME_FLAGS="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1"  # Last 10 blocks only
#BLOCK_TOME_FLAGS="0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0"  # Every other block
# last 10 blocks only
#BLOCK_TOME_FLAGS="0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0" 
#all blocks
#BLOCK_TOME_FLAGS="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" 

# --- Frequency Method Parameters ---
FREQUENCY_SELECTION_MODE="high"       # high or low
FREQUENCY_SELECTION_METHOD="1d_dft"   #['original','1d_dft', '1d_dct', '2d_conv','non_uniform_grid']
FREQUENCY_RANKING_METHOD="amplitude"  # choices=['spectral_centroid',  'amplitude',"variance", "l1norm", "l2norm"]
FREQUENCY_GRID_ALPHA=2.0

# --- DeepCache Configuration ---
CACHE_INTERVAL=1001
CACHE_BRANCH_ID=0

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

# --- ABP (Adaptive Block Pooling) Parameters ---
MERGE_METHOD_ALG="bipartite" # Options: 'bipartite', 'abp' - merging algorithm to use
ABP_SCORER="" # ABP scorer class name (e.g., 'FrequencyScorer', 'StatisticalScorer') - only used if MERGE_METHOD_ALG="abp"
ABP_TILE_AGGREGATION="max" # Options: 'max', 'min', 'sum', 'std' - tile aggregation method for ABP

# --- SiTo Configuration ---
# SiTo (Similarity-based Token Pruning) accelerates inference by pruning tokens based on similarity
SITO_PRUNE_RATIO=0.5
SITO_MAX_DOWNSAMPLE_RATIO=1
SITO_SX=2
SITO_SY=2
SITO_NOISE_ALPHA=0.1
SITO_SIM_BETA=1.0
SITO_PRUNE_SELFATTN=1
SITO_PRUNE_CROSSATTN=0
SITO_PRUNE_MLP=0
BLOCK_SITO_FLAGS=""  # Example: "1,1,1,0,0,0,..." for first 3 blocks only

# =================================================================================
# SCRIPT LOGIC
# =================================================================================

# Function to build the experiment name based on current parameters
build_experiment_name() {
    local name="DiT${IMG_SIZE}"

    # Add acceleration method details
    if [[ ${IF_TOKEN_MERGING} -eq 1 ]]; then
        name+="tokenmerge${TOKEN_MERGING_RATIO}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}_singlelevel${TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}"
        # Add method to name if not default
        if [[ "${TOKEN_MERGING_METHOD}" != "mean" ]]; then
            name+="_method${TOKEN_MERGING_METHOD}"
        fi
        # Add ABP information if ABP is used
        if [[ "${MERGE_METHOD_ALG}" == "abp" ]]; then
            name+="_abp"
            if [[ "${ABP_TILE_AGGREGATION}" != "max" ]]; then
                name+="_agg${ABP_TILE_AGGREGATION}"
            fi
            if [[ -n "${ABP_SCORER}" ]]; then
                name+="_scorer${ABP_SCORER}"
            fi
        fi
        
        # Add proportional attention if enabled
        if [[ ${IF_PROPORTIONAL_ATTENTION} -eq 1 ]]; then
            name+="_propattn"
        fi
        # Add cache parameters to folder name
        if [[ ${CACHE_MERGE_FUNCTIONS} -eq 1 ]]; then
            name+="_cache${CACHE_RECALC_INTERVAL}"
        fi
        
        # Add block flags information to folder name if custom flags are used for token merging
        if [[ -n "${BLOCK_TOME_FLAGS}" ]]; then
            # Parse block flags to count enabled blocks
            IFS=',' read -ra BLOCK_ARRAY <<< "${BLOCK_TOME_FLAGS}"
            local num_enabled=0
            local total_blocks=${#BLOCK_ARRAY[@]}
            
            for flag in "${BLOCK_ARRAY[@]}"; do
                if [[ "${flag}" == "1" ]]; then
                    ((num_enabled++))
                fi
            done
            
            if [[ ${num_enabled} -eq ${total_blocks} ]]; then
                name+="_allblocks"
            elif [[ ${num_enabled} -eq 0 ]]; then
                name+="_noblocks"
            else
                name+="_blocks${num_enabled}of${total_blocks}"
                # Always add a hash for partial patterns to differentiate between different arrangements
                if [[ ${total_blocks} -le 28 ]]; then
                    local pattern_str="${BLOCK_TOME_FLAGS//,/}"  # Remove commas
                    local pattern_hash=$(echo -n "${pattern_str}" | md5sum | cut -c1-6)
                    name+="_pat${pattern_hash}"
                fi
            fi
        fi
    elif [[ ${IF_AGENTSD} -eq 1 ]]; then
        name+="agentsd${TOKEN_MERGING_RATIO}_agentratio${AGENT_RATIO}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}_singlelevel${TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}"
    elif [[ ${IF_SCORING_MERGE} -eq 1 ]]; then
        name+="scoringmerge${TOKEN_MERGING_RATIO}_method${SCORING_METHOD}_preserve${SCORING_PRESERVE_RATIO}_mode${SCORING_MODE}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}"
        
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
        fi
        
        # Add merging method if not default
        if [[ "${TOKEN_MERGING_METHOD}" != "mean" ]]; then
            name+="_mergemethod${TOKEN_MERGING_METHOD}"
        fi
        
        # Add ABP information if ABP is used
        if [[ "${MERGE_METHOD_ALG}" == "abp" ]]; then
            name+="_abp"
            if [[ "${ABP_TILE_AGGREGATION}" != "max" ]]; then
                name+="_agg${ABP_TILE_AGGREGATION}"
            fi
            if [[ -n "${ABP_SCORER}" ]]; then
                name+="_scorer${ABP_SCORER}"
            fi
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
        
        # Add cache parameters to folder name
        if [[ ${CACHE_MERGE_FUNCTIONS} -eq 1 ]]; then
            name+="_cache${CACHE_RECALC_INTERVAL}"
        fi
        
        # Add block flags information to folder name if custom flags are used for scoring merge
        if [[ -n "${BLOCK_TOME_FLAGS}" ]]; then
            # Parse block flags to count enabled blocks
            IFS=',' read -ra BLOCK_ARRAY <<< "${BLOCK_TOME_FLAGS}"
            local num_enabled=0
            local total_blocks=${#BLOCK_ARRAY[@]}
            
            for flag in "${BLOCK_ARRAY[@]}"; do
                if [[ "${flag}" == "1" ]]; then
                    ((num_enabled++))
                fi
            done
            
            if [[ ${num_enabled} -eq ${total_blocks} ]]; then
                name+="_allblocks"
            elif [[ ${num_enabled} -eq 0 ]]; then
                name+="_noblocks"
            else
                name+="_blocks${num_enabled}of${total_blocks}"
                # Always add a hash for partial patterns to differentiate between different arrangements
                if [[ ${total_blocks} -le 28 ]]; then
                    local pattern_str="${BLOCK_TOME_FLAGS//,/}"  # Remove commas
                    local pattern_hash=$(echo -n "${pattern_str}" | md5sum | cut -c1-6)
                    name+="_pat${pattern_hash}"
                fi
            fi
        fi
    elif [[ ${IF_ATTENTION_PROC} -eq 1 ]]; then
        if [[ "${MERGE_METHOD}" == "frequency_global" ]]; then
            name+="freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_ratio${TOKEN_MERGING_RATIO}"
        elif [[ "${MERGE_METHOD}" == "frequency_blockwise" ]]; then
            name+="freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_downsample${DOWNSAMPLE_FACTOR}"
        elif [[ "${MERGE_METHOD}" == "downsample_qkv_upsample_out" ]]; then
            name+="downsamplemethod${DOWNSAMPLE_METHOD}_qkv_downsamplemethod${QKV_DOWNSAMPLE_METHOD}_out_upsamplemethod${OUT_UPSAMPLE_METHOD}_level1downsample${DOWNSAMPLE_FACTOR}"
        
        elif [[ "${MERGE_METHOD}" == "downsample_custom_block" ]]; then
            name+="customblock_h${DOWNSAMPLE_FACTOR_H}_w${DOWNSAMPLE_FACTOR_W}_downmethod${DOWNSAMPLE_METHOD}_selectionsource${SELECTION_SOURCE}_switch${TIMESTEP_THRESHOLD_SWITCH}_stop${TIMESTEP_THRESHOLD_STOP}"
        elif [[ "${MERGE_METHOD}" == "similarity" ]]; then
            name+="similarityratio${TOKEN_MERGING_RATIO}_mergetoken${MERGE_TOKENS}_selectionsource${SELECTION_SOURCE}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}"

        else
            name+="tokendownsampling${TOKEN_MERGING_RATIO}_userand${TOKEN_MERGING_USE_RAND}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}_down${DOWNSAMPLE_FACTOR}_downmethod${DOWNSAMPLE_METHOD}_selectionsource${SELECTION_SOURCE}_switch${TIMESTEP_THRESHOLD_SWITCH}_stop${TIMESTEP_THRESHOLD_STOP}"
            
            # Add linear_blend parameter to name if using linear_blend method
            if [[ "${DOWNSAMPLE_METHOD}" == "linear_blend" ]]; then
                name+="_linearblend${BLEND_FACTOR}"
                name+="_blendmethods${BLEND_METHOD_1}-${BLEND_METHOD_2}"
            fi
        fi

        # Add block flags information to folder name if custom flags are used for attention processor
        if [[ -n "${BLOCK_TOME_FLAGS}" ]]; then
            # Parse block flags to count enabled blocks
            IFS=',' read -ra BLOCK_ARRAY <<< "${BLOCK_TOME_FLAGS}"
            local num_enabled=0
            local total_blocks=${#BLOCK_ARRAY[@]}
            
            for flag in "${BLOCK_ARRAY[@]}"; do
                if [[ "${flag}" == "1" ]]; then
                    ((num_enabled++))
                fi
            done
            
            if [[ ${num_enabled} -eq ${total_blocks} ]]; then
                name+="_allblocks"
            elif [[ ${num_enabled} -eq 0 ]]; then
                name+="_noblocks"
            else
                name+="_blocks${num_enabled}of${total_blocks}"
                # Always add a hash for partial patterns to differentiate between different arrangements
                if [[ ${total_blocks} -le 28 ]]; then
                    local pattern_str="${BLOCK_TOME_FLAGS//,/}"  # Remove commas
                    local pattern_hash=$(echo -n "${pattern_str}" | md5sum | cut -c1-6)
                    name+="_pat${pattern_hash}"
                fi
            fi
        fi

    elif [[ ${IF_SITO} -eq 1 ]]; then
        name+="sito_prune${SITO_PRUNE_RATIO}_maxdown${SITO_MAX_DOWNSAMPLE_RATIO}_sx${SITO_SX}_sy${SITO_SY}_noisealpha${SITO_NOISE_ALPHA}_simbeta${SITO_SIM_BETA}_selfattn${SITO_PRUNE_SELFATTN}_crossattn${SITO_PRUNE_CROSSATTN}_mlp${SITO_PRUNE_MLP}"
        
        # Add block flags information to folder name if custom flags are used for SiTo
        if [[ -n "${BLOCK_SITO_FLAGS}" ]]; then
            # Parse block flags to count enabled blocks
            IFS=',' read -ra BLOCK_ARRAY <<< "${BLOCK_SITO_FLAGS}"
            local num_enabled=0
            local total_blocks=${#BLOCK_ARRAY[@]}
            
            for flag in "${BLOCK_ARRAY[@]}"; do
                if [[ "${flag}" == "1" ]]; then
                    ((num_enabled++))
                fi
            done
            
            if [[ ${num_enabled} -eq ${total_blocks} ]]; then
                name+="_allblocks"
            elif [[ ${num_enabled} -eq 0 ]]; then
                name+="_noblocks"
            else
                name+="_blocks${num_enabled}of${total_blocks}"
                # Always add a hash for partial patterns to differentiate between different arrangements
                if [[ ${total_blocks} -le 28 ]]; then
                    local pattern_str="${BLOCK_SITO_FLAGS//,/}"  # Remove commas
                    local pattern_hash=$(echo -n "${pattern_str}" | md5sum | cut -c1-6)
                    name+="_pat${pattern_hash}"
                fi
            fi
        fi
    fi

    if [[ ${IF_DEEPCACHE} -eq 1 ]]; then
        name+="_deepcache${CACHE_INTERVAL}"
    fi

    # If no acceleration is used, mark as baseline
    if [[ ${IF_TOKEN_MERGING} -eq 0 && ${IF_AGENTSD} -eq 0 && ${IF_ATTENTION_PROC} -eq 0 && ${IF_DEEPCACHE} -eq 0 && ${IF_SCORING_MERGE} -eq 0 && ${IF_SITO} -eq 0 ]]; then
        name+="_baseline"
    fi

    name+="_${STEPS}steps_seed${SEED}"
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

    # Build the command with all arguments
    CMD="python ${SCRIPT_DIR}/../dit_generation.py \
        --dataset ${DATASET} \
        --img_size ${IMG_SIZE} \
        --num_classes ${NUM_CLASSES} \
        --steps ${STEPS} \
        --batch_size ${BATCH_SIZE} \
        --dtype ${DTYPE} \
        --quality ${QUALITY} \
        --seed ${SEED} \
        --gt_path ${GT_PATH} \
        --if_deepcache ${IF_DEEPCACHE} \
        --cache_interval ${CACHE_INTERVAL} \
        --cache_branch_id ${CACHE_BRANCH_ID} \
        --if_token_merging ${IF_TOKEN_MERGING} \
        --if_agentsd ${IF_AGENTSD} \
        --if_attention_proc ${IF_ATTENTION_PROC} \
        --if_scoring_merge ${IF_SCORING_MERGE} \
        --if_sito ${IF_SITO} \
        --sito_prune_ratio ${SITO_PRUNE_RATIO} \
        --sito_max_downsample_ratio ${SITO_MAX_DOWNSAMPLE_RATIO} \
        --sito_sx ${SITO_SX} \
        --sito_sy ${SITO_SY} \
        --sito_noise_alpha ${SITO_NOISE_ALPHA} \
        --sito_sim_beta ${SITO_SIM_BETA} \
        --sito_prune_selfattn ${SITO_PRUNE_SELFATTN} \
        --sito_prune_crossattn ${SITO_PRUNE_CROSSATTN} \
        --sito_prune_mlp ${SITO_PRUNE_MLP} \
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
        --token_merging_cache_indices_per_image ${TOKEN_MERGING_CACHE_INDICES_PER_IMAGE} \
        --token_merging_method ${TOKEN_MERGING_METHOD} \
        --if_proportional_attention ${IF_PROPORTIONAL_ATTENTION} \
        --cache_merge_functions ${CACHE_MERGE_FUNCTIONS} \
        --cache_recalc_interval ${CACHE_RECALC_INTERVAL} \
        --merge_tokens ${MERGE_TOKENS} \
        --merge_method ${MERGE_METHOD} \
        --downsample_method ${DOWNSAMPLE_METHOD} \
        --downsample_factor ${DOWNSAMPLE_FACTOR} \
        --downsample_factor_h ${DOWNSAMPLE_FACTOR_H} \
        --downsample_factor_w ${DOWNSAMPLE_FACTOR_W} \
        --timestep_threshold_switch ${TIMESTEP_THRESHOLD_SWITCH} \
        --timestep_threshold_stop ${TIMESTEP_THRESHOLD_STOP} \
        --secondary_merge_method ${SECONDARY_MERGE_METHOD} \
        --ratio_level_2 ${RATIO_LEVEL_2} \
        --selection_source ${SELECTION_SOURCE} \
        --frequency_selection_mode ${FREQUENCY_SELECTION_MODE} \
        --frequency_selection_method ${FREQUENCY_SELECTION_METHOD} \
        --frequency_ranking_method ${FREQUENCY_RANKING_METHOD} \
        --frequency_grid_alpha ${FREQUENCY_GRID_ALPHA} \
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
        --merge_method_alg ${MERGE_METHOD_ALG} \
        --abp_tile_aggregation ${ABP_TILE_AGGREGATION}"

    # Add optional blend parameters if specified
    if [[ -n "${BLEND_FACTOR}" ]]; then
        CMD+=" --blend_factor ${BLEND_FACTOR}"
    fi
    
    if [[ -n "${BLEND_METHOD_1}" ]]; then
        CMD+=" --blend_method_1 ${BLEND_METHOD_1}"
    fi
    
    if [[ -n "${BLEND_METHOD_2}" ]]; then
        CMD+=" --blend_method_2 ${BLEND_METHOD_2}"
    fi

    # Add block_tome_flags if specified
    if [[ -n "${BLOCK_TOME_FLAGS}" ]]; then
        CMD+=" --block_tome_flags ${BLOCK_TOME_FLAGS}"
    fi

    # Add block_sito_flags if specified
    if [[ -n "${BLOCK_SITO_FLAGS}" ]]; then
        CMD+=" --block_sito_flags ${BLOCK_SITO_FLAGS}"
    fi

# Add ABP scorer if specified
if [[ -n "${ABP_SCORER_METHOD}" ]]; then
    CMD+=" --abp_scorer ${ABP_SCORER_METHOD}"
fi

    # If class_csv_path is set, add it to the command
    if [[ -n "${CLASS_CSV_PATH}" ]]; then
        CMD+=" --class_csv_path ${CLASS_CSV_PATH}"
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
# EXPERIMENTS - Uncomment to run
# =================================================================================

# --- Baseline (No Acceleration) ---
# IF_TOKEN_MERGING=0; IF_AGENTSD=0; IF_ATTENTION_PROC=0; IF_SCORING_MERGE=0; IF_SITO=0
# run_experiment

# --- Original ToMe ---
# IF_TOKEN_MERGING=1; IF_AGENTSD=0; IF_ATTENTION_PROC=0; IF_SCORING_MERGE=0; IF_SITO=0
# TOKEN_MERGING_RATIO=0.7
# TOKEN_MERGING_METHOD="mean"
# run_experiment

--- LGTM: Scoring-based Merging ---
IF_TOKEN_MERGING=0; IF_AGENTSD=0; IF_ATTENTION_PROC=0; IF_SCORING_MERGE=1; IF_SITO=0
TOKEN_MERGING_RATIO=0.7
SCORING_METHOD="spatial_filter"
SCORING_SPATIAL_METHOD="2d_conv"
SCORING_SPATIAL_NORM="l1"
SCORING_PRESERVE_RATIO=0
SCORING_MODE="high"
IF_LOW_FREQUENCY_DST_TOKENS=1
MERGE_METHOD_ALG="bipartite"
run_experiment

# --- Nearest-Exact Downsampling (ToDo) ---
# IF_TOKEN_MERGING=0; IF_AGENTSD=0; IF_ATTENTION_PROC=1; IF_SCORING_MERGE=0; IF_SITO=0
# MERGE_METHOD="downsample"
# DOWNSAMPLE_METHOD="nearest-exact"
# DOWNSAMPLE_FACTOR=2
# run_experiment

# --- IEKVD: Linear Blend Downsampling ---
# IF_TOKEN_MERGING=0; IF_AGENTSD=0; IF_ATTENTION_PROC=1; IF_SCORING_MERGE=0; IF_SITO=0
# MERGE_METHOD="downsample"
# DOWNSAMPLE_METHOD="linear_blend"
# DOWNSAMPLE_FACTOR=2
# BLEND_FACTOR="0.5"
# BLEND_METHOD_1="nearest-exact"
# BLEND_METHOD_2="avg_pool"
# run_experiment

# --- SiTo Pruning ---
# IF_TOKEN_MERGING=0; IF_AGENTSD=0; IF_ATTENTION_PROC=0; IF_SCORING_MERGE=0; IF_SITO=1
# SITO_PRUNE_RATIO=0.7
# run_experiment

echo "All experiments completed. Uncomment examples above to run."
