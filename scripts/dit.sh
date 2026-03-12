#!/bin/bash

# DiT Evaluation Script

# --- Model Configuration ---
IMG_SIZE=512  # 256 or 512 for DiT
SEED=0

# --- Dataset Configuration ---
DATASET="imagenet100"
SPLIT="test"
CLASS_CSV_PATH="prompts/${DATASET}_prompts.csv"
NUM_CLASSES=1000  # Default number of classes
N_TRIALS=1
BATCH_SIZE=16
DTYPE="float16"
INTERPOLATION="bicubic"  # Options: bicubic, bilinear, lanczos

# --- Adaptive Evaluation Parameters ---
TO_KEEP="5 1" # 5 1
N_SAMPLES="5 20" # 5 20
LOSS="l2"  # Options: l2, l1, huber
N_WORKERS=1
WORKER_IDX=0
LOAD_STATS=0

# --- Acceleration Configuration ---
IF_TOKEN_MERGING=0
IF_AGENTSD=0
IF_ATTENTION_PROC=1
IF_SCORING_MERGE=0
IF_SITO=0

# --- Token Merging Parameters ---
TOKEN_MERGING_RATIO=0
AGENT_RATIO=0
TOKEN_MERGING_USE_RAND=1
TOKEN_MERGING_SX=2
TOKEN_MERGING_SY=2
TOKEN_MERGING_MAX_DOWNSAMPLE=8
TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE=0
MERGE_ATTN=1
MERGE_CROSSATTN=0
MERGE_MLP=0
TOKEN_MERGING_CACHE_INDICES_PER_IMAGE=0
TOKEN_MERGING_METHOD="mean"  # Options: mean, mlerp, prune
IF_PROPORTIONAL_ATTENTION=0  # 0/1 flag for proportional attention

# --- Block-level Caching Parameters (for DiT scoring merge acceleration) ---
CACHE_MERGE_FUNCTIONS=0      # 0/1 flag to enable block-level function caching (only works with scoring merge + block control)
CACHE_RECALC_INTERVAL=5      # Recalculate cached functions every N blocks (higher = more reuse = faster but potentially lower quality)

# --- Attention Processor Parameters ---
MERGE_TOKENS="keys/values"
MERGE_METHOD="downsample" ###### downsample downsample_custom_block similarity frequency_global frequency_blockwise downsample_qkv_upsample_out masked_attention blockwise_masked_attention
DOWNSAMPLE_METHOD="avg_pool" # Options: 'nearest-exact', 'max_pool','avg_pool', 'area', "bilinear", "bicubic",'top_right','bottom_left','bottom_right','random',uniform_random,uniform_timestep,linear_blend
DOWNSAMPLE_FACTOR=2
DOWNSAMPLE_FACTOR_H=2
DOWNSAMPLE_FACTOR_W=2
# Linear blend parameters for downsample methods (only used when DOWNSAMPLE_METHOD="linear_blend")
BLEND_FACTOR=""  # Default: empty string (uses 0.5). Set to value like 0.7 for 70% method1, 30% method2
BLEND_METHOD_1=""  # Default: empty string (uses nearest-exact). Options: same as DOWNSAMPLE_METHOD
BLEND_METHOD_2=""  # Default: empty string (uses avg_pool). Options: same as DOWNSAMPLE_METHOD
DOWNSAMPLE_FACTOR_LEVEL_2=1
TIMESTEP_THRESHOLD_SWITCH=0.0
TIMESTEP_THRESHOLD_STOP=0.0
SECONDARY_MERGE_METHOD="similarity"
RATIO_LEVEL_2=0.0  # For compatibility (not used in DiT)
# BLOCK_TOME_FLAGS=""  # Comma-separated list of 0s and 1s for block-wise ToMe control (e.g., "1,1,1,0,0,0,..." for 28 blocks)
# Example usage for block flags:
BLOCK_TOME_FLAGS="0,1,1,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"  
#BLOCK_TOME_FLAGS="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"  

# --- Frequency Method Parameters ---
FREQUENCY_SELECTION_MODE="high"
FREQUENCY_SELECTION_METHOD="original"   #['original','1d_dft', '1d_dct', '2d_conv','non_uniform_grid']
FREQUENCY_RANKING_METHOD="l2norm" # choices=['spectral_centroid',  'amplitude',"variance", "l1norm", "l2norm"]
SELECTION_SOURCE="hidden"
FREQUENCY_GRID_ALPHA=2.0


# --- QKV Downsample Parameters ---
QKV_DOWNSAMPLE_METHOD="nearest"
OUT_UPSAMPLE_METHOD="nearest"

# --- Scoring-based Token Merging Parameters ---
SCORING_METHOD="statistical" # Options: 'frequency', 'spatial_filter', 'statistical', 'signal_processing', 'spatial_distribution'
SCORING_PRESERVE_RATIO=0.0 # Ratio of tokens to protect from merging (0.0 to 1.0)
SCORING_MODE="high" # Options: 'high', 'low', 'medium', 'timestep_scheduler', 'reverse_timestep_scheduler'
SCORING_PRESERVE_SPATIAL_UNIFORMITY=0 # 0/1 flag for preserving spatial uniformity in scoring-based merging
IF_LOW_FREQUENCY_DST_TOKENS=1 # 0/1 flag for score-guided destination selection (lowest-scored tokens as destinations)

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

# --- Additional Parameters ---
EXTRA_GUIDANCE_SCALE=0.0
EXTRA="None"  # Set to string value or None

# --- Evaluation Configuration ---
SINGLE_TIMESTEP=-1
EVAL_DENOISE_STEPS=1
EVAL_STEP_STRIDE=1
EVAL_ERROR_METHOD="trajectory"  # Options: 'trajectory', 'direct', 'weighted', 'clean_signal'

# --- DeepCache Configuration ---
IF_DEEPCACHE=0
CACHE_INTERVAL=1001
CACHE_BRANCH_ID=0

# --- ToCa Configuration ---
# ToCa (Token Caching) accelerates multi-step denoising by caching transformer computations
# Requires: EVAL_DENOISE_STEPS > 1 and other acceleration methods disabled
# 
# IF_TOCA: 0=disabled, 1=enabled
# TOCA_FRESH_RATIO: Portion of tokens to recompute (0.0-1.0). Lower = more caching = faster
# TOCA_FRESH_THRESHOLD: Interval for full computation. 3 = full compute every 3rd step
# TOCA_CACHE_TYPE: Token selection strategy
#   - "attention": Use attention scores (recommended)
#   - "random": Random selection  
#   - "similarity": Cosine similarity to cached tokens
#   - "norm": L2 norm of tokens
#   - "kv-norm": Key-value norm (experimental)
# TOCA_RATIO_SCHEDULER: Fresh ratio scheduling
#   - "ToCa-ddim50": Optimized for DDIM 50-step (recommended)
#   - "ToCa-ddpm250": Optimized for DDPM 250-step
#   - "constant": Fixed ratio throughout
#   - "linear": Linear decay
IF_TOCA=0
TOCA_FRESH_RATIO=0.6
TOCA_FRESH_THRESHOLD=3
TOCA_CACHE_TYPE="attention"  # Options: attention, random, similarity, norm, kv-norm
TOCA_RATIO_SCHEDULER="constant"  # Options: constant, linear, ToCa-ddim50, ToCa-ddpm250

# --- SiTo Configuration ---
# SiTo (Similarity-based Token Pruning) accelerates inference by pruning tokens based on similarity
# 
# IF_SITO: 0=disabled, 1=enabled
# SITO_PRUNE_RATIO: Ratio of tokens to prune (0.0-1.0). Higher = more acceleration, potentially lower quality
# SITO_MAX_DOWNSAMPLE_RATIO: Maximum downsample ratio for applying SiTo (typically 1)
# SITO_SX, SITO_SY: Stride in x,y dimensions for patch processing (typically 2,2)
# SITO_NOISE_ALPHA: Weight for noise component in scoring (0.0-1.0)
# SITO_SIM_BETA: Weight for similarity component in scoring (typically 1.0)
# SITO_PRUNE_SELFATTN: Apply SiTo to self-attention layers (1=yes, 0=no)
# SITO_PRUNE_CROSSATTN: Apply SiTo to cross-attention layers (1=yes, 0=no)  
# SITO_PRUNE_MLP: Apply SiTo to MLP layers (1=yes, 0=no)
# BLOCK_SITO_FLAGS: Comma-separated list of 0s and 1s for block-wise SiTo control
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

# --- Summary CSV ---
SUMMARY_CSV="classification_experiments_dit_summary.csv"

# --- Output Directory ---
BASE_LOG_DIR="/path/to/results/dit_results"

# =================================================================================
# FUNCTIONS
# =================================================================================

# Function to build base experiment name (core parameters)
build_base_experiment_name() {
    local BASE_NAME="DiT${IMG_SIZE}_${N_TRIALS}trials_"
    
    # Add dataset-specific naming
    if [[ -n "${CLASS_CSV_PATH}" ]]; then
        local CSV_NAME=$(basename "${CLASS_CSV_PATH}" .csv)
        BASE_NAME+="${CSV_NAME}_"
    else
        # If we're using a subset of classes
        if [[ ${NUM_CLASSES} -ne 1000 ]]; then
            BASE_NAME+="${NUM_CLASSES}classes_"
        fi
    fi
    
    # Add timestep info
    if [[ ${SINGLE_TIMESTEP} -ge 0 ]]; then
        BASE_NAME+="single_t${SINGLE_TIMESTEP}"
    else
        # Adaptive naming
        BASE_NAME+="$(echo ${TO_KEEP} | tr ' ' '_')keep_"
        BASE_NAME+="$(echo ${N_SAMPLES} | tr ' ' '_')samples"
    fi
    
    BASE_NAME+="_randomseed${SEED}"
    
    # Add interpolation if not default
    if [[ "${INTERPOLATION}" != "bicubic" ]]; then
        BASE_NAME+="_${INTERPOLATION}"
    fi
    
    # Add loss if not default
    if [[ "${LOSS}" == "l1" ]]; then
        BASE_NAME+="_l1"
    elif [[ "${LOSS}" == "huber" ]]; then
        BASE_NAME+="_huber"
    fi
    
    # Add img_size if not default (256)
    if [[ ${IMG_SIZE} -ne 256 ]]; then
        BASE_NAME+="_${IMG_SIZE}"
    fi
    
    echo "${BASE_NAME}"
}

# Function to build method-specific details
build_method_details() {
    local METHOD_DETAILS=""
    
    # Add acceleration method details
    if [[ ${IF_TOKEN_MERGING} -eq 1 ]]; then
        METHOD_DETAILS="tokenmerge${TOKEN_MERGING_RATIO}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}_singlelevel${TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}"
        # Add method to name if not default
        if [[ "${TOKEN_MERGING_METHOD}" != "mean" ]]; then
            METHOD_DETAILS+="_method${TOKEN_MERGING_METHOD}"
        fi
        # Add ABP information if ABP is used
        if [[ "${MERGE_METHOD_ALG}" == "abp" ]]; then
            METHOD_DETAILS+="_abp"
            if [[ "${ABP_TILE_AGGREGATION}" != "max" ]]; then
                METHOD_DETAILS+="_agg${ABP_TILE_AGGREGATION}"
            fi
            if [[ -n "${ABP_SCORER}" ]]; then
                METHOD_DETAILS+="_scorer${ABP_SCORER}"
            fi
        fi
        
        # Add proportional attention if enabled
        if [[ ${IF_PROPORTIONAL_ATTENTION} -eq 1 ]]; then
            METHOD_DETAILS+="_propattn"
        fi
        # Add cache parameters to folder name
        if [[ ${CACHE_MERGE_FUNCTIONS} -eq 1 ]]; then
            METHOD_DETAILS+="_cache${CACHE_RECALC_INTERVAL}"
        fi
    elif [[ ${IF_AGENTSD} -eq 1 ]]; then
        METHOD_DETAILS="agentsd${TOKEN_MERGING_RATIO}_agentratio${AGENT_RATIO}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}_singlelevel${TOKEN_MERGING_SINGLE_DOWNSAMPLE_LEVEL_MERGE}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}"
    elif [[ ${IF_SCORING_MERGE} -eq 1 ]]; then
        METHOD_DETAILS="scoringmerge${TOKEN_MERGING_RATIO}_method${SCORING_METHOD}_preserve${SCORING_PRESERVE_RATIO}_mode${SCORING_MODE}_userand${TOKEN_MERGING_USE_RAND}_maxdownsample${TOKEN_MERGING_MAX_DOWNSAMPLE}"
        
        # Add method-specific parameters to name
        if [[ "${SCORING_METHOD}" == "frequency" ]]; then
            METHOD_DETAILS+="_freqmethod${SCORING_FREQ_METHOD}_freqranking${SCORING_FREQ_RANKING}"
        elif [[ "${SCORING_METHOD}" == "spatial_filter" ]]; then
            METHOD_DETAILS+="_spatialmethod${SCORING_SPATIAL_METHOD}_spatialnorm${SCORING_SPATIAL_NORM}"
        elif [[ "${SCORING_METHOD}" == "statistical" ]]; then
            METHOD_DETAILS+="_statmethod${SCORING_STAT_METHOD}"
        elif [[ "${SCORING_METHOD}" == "signal_processing" ]]; then
            METHOD_DETAILS+="_signalmethod${SCORING_SIGNAL_METHOD}"
        elif [[ "${SCORING_METHOD}" == "spatial_distribution" ]]; then
            METHOD_DETAILS+="_spatialalpha${SCORING_SPATIAL_ALPHA}"
        fi
        
        # Add merging method if not default
        if [[ "${TOKEN_MERGING_METHOD}" != "mean" ]]; then
            METHOD_DETAILS+="_mergemethod${TOKEN_MERGING_METHOD}"
        fi
        
        # Add ABP information if ABP is used
        if [[ "${MERGE_METHOD_ALG}" == "abp" ]]; then
            METHOD_DETAILS+="_abp"
            if [[ "${ABP_TILE_AGGREGATION}" != "max" ]]; then
                METHOD_DETAILS+="_agg${ABP_TILE_AGGREGATION}"
            fi
            if [[ -n "${ABP_SCORER}" ]]; then
                METHOD_DETAILS+="_scorer${ABP_SCORER}"
            fi
        fi
        
        # Add spatial uniformity parameter if enabled
        if [[ ${SCORING_PRESERVE_SPATIAL_UNIFORMITY} -eq 1 ]]; then
            METHOD_DETAILS+="_spatialuniform"
        fi
        
        # Add score-guided destination selection parameter if enabled
        if [[ ${IF_LOW_FREQUENCY_DST_TOKENS} -eq 1 ]]; then
            METHOD_DETAILS+="_lowfreqdst"
        fi
        
        # Add proportional attention if enabled
        if [[ ${IF_PROPORTIONAL_ATTENTION} -eq 1 ]]; then
            METHOD_DETAILS+="_propattn"
        fi
        
        # Add cache parameters to folder name
        if [[ ${CACHE_MERGE_FUNCTIONS} -eq 1 ]]; then
            METHOD_DETAILS+="_cache${CACHE_RECALC_INTERVAL}"
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
            
            # Always add a hash for any custom block flags pattern
            local pattern_str="${BLOCK_TOME_FLAGS//,/}"  # Remove commas
            local pattern_hash=$(echo -n "${pattern_str}" | md5sum | cut -c1-6)
            
            if [[ ${num_enabled} -eq ${total_blocks} ]]; then
                METHOD_DETAILS+="_allblocks_pat${pattern_hash}"
            elif [[ ${num_enabled} -eq 0 ]]; then
                METHOD_DETAILS+="_noblocks_pat${pattern_hash}"
            else
                METHOD_DETAILS+="_blocks${num_enabled}of${total_blocks}_pat${pattern_hash}"
            fi
        fi

    elif [[ ${IF_ATTENTION_PROC} -eq 1 ]]; then
        if [[ "${MERGE_METHOD}" == "frequency_global" ]]; then
            METHOD_DETAILS="freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_ratio${TOKEN_MERGING_RATIO}"
            if [[ "${FREQUENCY_SELECTION_METHOD}" == "non_uniform_grid" ]]; then
                METHOD_DETAILS+="_gridalpha${FREQUENCY_GRID_ALPHA}"
            fi
        elif [[ "${MERGE_METHOD}" == "frequency_blockwise" ]]; then
            METHOD_DETAILS="freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_downsample${DOWNSAMPLE_FACTOR}"
        elif [[ "${MERGE_METHOD}" == "downsample_qkv_upsample_out" ]]; then
            METHOD_DETAILS="downsamplemethod${DOWNSAMPLE_METHOD}_qkv_downsamplemethod${QKV_DOWNSAMPLE_METHOD}_out_upsamplemethod${OUT_UPSAMPLE_METHOD}_level1downsample${DOWNSAMPLE_FACTOR}"
        elif [[ "${MERGE_METHOD}" == "downsample_custom_block" ]]; then
            METHOD_DETAILS="customblock_h${DOWNSAMPLE_FACTOR_H}_w${DOWNSAMPLE_FACTOR_W}_downmethod${DOWNSAMPLE_METHOD}_selectionsource${SELECTION_SOURCE}_switch${TIMESTEP_THRESHOLD_SWITCH}_stop${TIMESTEP_THRESHOLD_STOP}"
        elif [[ "${MERGE_METHOD}" == "masked_attention" ]]; then
            METHOD_DETAILS="freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_ratio${TOKEN_MERGING_RATIO}"
            if [[ "${FREQUENCY_SELECTION_METHOD}" == "non_uniform_grid" ]]; then
                METHOD_DETAILS+="_gridalpha${FREQUENCY_GRID_ALPHA}"
            fi
        elif [[ "${MERGE_METHOD}" == "blockwise_masked_attention" ]]; then
            METHOD_DETAILS="freqmode_${FREQUENCY_SELECTION_MODE}_freqmethod${FREQUENCY_SELECTION_METHOD}_freqrankingmethod${FREQUENCY_RANKING_METHOD}_selectionsource${SELECTION_SOURCE}_downsample${DOWNSAMPLE_FACTOR}"
        else
            METHOD_DETAILS="tokendownsampling${TOKEN_MERGING_RATIO}_userand${TOKEN_MERGING_USE_RAND}_sx${TOKEN_MERGING_SX}_sy${TOKEN_MERGING_SY}_down${DOWNSAMPLE_FACTOR}_downmethod${DOWNSAMPLE_METHOD}_selectionsource${SELECTION_SOURCE}_switch${TIMESTEP_THRESHOLD_SWITCH}_stop${TIMESTEP_THRESHOLD_STOP}"
            if [[ "${DOWNSAMPLE_METHOD}" == "linear_blend" ]]; then
                METHOD_DETAILS+="_linearblend${BLEND_FACTOR}"
                METHOD_DETAILS+="_blendmethods${BLEND_METHOD_1}-${BLEND_METHOD_2}"
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
            
            # Always add a hash for any custom block flags pattern
            local pattern_str="${BLOCK_TOME_FLAGS//,/}"  # Remove commas
            local pattern_hash=$(echo -n "${pattern_str}" | md5sum | cut -c1-6)
            
            if [[ ${num_enabled} -eq ${total_blocks} ]]; then
                METHOD_DETAILS+="_allblocks_pat${pattern_hash}"
            elif [[ ${num_enabled} -eq 0 ]]; then
                METHOD_DETAILS+="_noblocks_pat${pattern_hash}"
            else
                METHOD_DETAILS+="_blocks${num_enabled}of${total_blocks}_pat${pattern_hash}"
            fi
        fi
        

    elif [[ ${IF_SITO} -eq 1 ]]; then
        METHOD_DETAILS="sito_prune${SITO_PRUNE_RATIO}_maxdown${SITO_MAX_DOWNSAMPLE_RATIO}_sx${SITO_SX}_sy${SITO_SY}_noisealpha${SITO_NOISE_ALPHA}_simbeta${SITO_SIM_BETA}_selfattn${SITO_PRUNE_SELFATTN}_crossattn${SITO_PRUNE_CROSSATTN}_mlp${SITO_PRUNE_MLP}"
        
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
            
            # Always add a hash for any custom block flags pattern
            local pattern_str="${BLOCK_SITO_FLAGS//,/}"  # Remove commas
            local pattern_hash=$(echo -n "${pattern_str}" | md5sum | cut -c1-6)
            
            if [[ ${num_enabled} -eq ${total_blocks} ]]; then
                METHOD_DETAILS+="_allblocks_pat${pattern_hash}"
            elif [[ ${num_enabled} -eq 0 ]]; then
                METHOD_DETAILS+="_noblocks_pat${pattern_hash}"
            else
                METHOD_DETAILS+="_blocks${num_enabled}of${total_blocks}_pat${pattern_hash}"
            fi
        fi

    fi
    
    # Add DeepCache if enabled
    if [[ ${IF_DEEPCACHE} -eq 1 ]]; then
        if [[ -n "${METHOD_DETAILS}" ]]; then
            METHOD_DETAILS+="_"
        fi
        METHOD_DETAILS+="deepcache_interval${CACHE_INTERVAL}_branch${CACHE_BRANCH_ID}"
    fi
    
    # Add ToCa if enabled
    if [[ ${IF_TOCA} -eq 1 ]]; then
        if [[ -n "${METHOD_DETAILS}" ]]; then
            METHOD_DETAILS+="_"
        fi
        METHOD_DETAILS+="toca_ratio${TOCA_FRESH_RATIO}_thresh${TOCA_FRESH_THRESHOLD}_cache${TOCA_CACHE_TYPE}_sched${TOCA_RATIO_SCHEDULER}"
    fi
    
    # Add evaluation settings
    if [[ ${EVAL_DENOISE_STEPS} -gt 1 ]]; then
        if [[ -n "${METHOD_DETAILS}" ]]; then
            METHOD_DETAILS+="_"
        fi
        METHOD_DETAILS+="denoise_steps${EVAL_DENOISE_STEPS}"
        if [[ ${EVAL_STEP_STRIDE} -gt 1 ]]; then
            METHOD_DETAILS+="_stride${EVAL_STEP_STRIDE}"
        fi
        METHOD_DETAILS+="_error_method${EVAL_ERROR_METHOD}"
    fi
    
    # If no specific method details, use a default name
    if [[ -z "${METHOD_DETAILS}" ]]; then
        METHOD_DETAILS="baseline"
    fi
    
    echo "${METHOD_DETAILS}"
}

# Function to construct and run the python command
run_experiment() {
    # Build base experiment name and method details
    BASE_EXPERIMENT_NAME=$(build_base_experiment_name)
    METHOD_DETAILS=$(build_method_details)
    
    # Determine output folder based on EXTRA parameter
    if [[ "${EXTRA}" != "None" ]] && [[ -n "${EXTRA}" ]]; then
        DATASET_DIR="${BASE_LOG_DIR}/${DATASET}_${EXTRA}"
    else
        DATASET_DIR="${BASE_LOG_DIR}/${DATASET}"
    fi
    
    # Create the new directory structure:
    # BASE_LOG_DIR/DATASET/BASE_EXPERIMENT_NAME/method_[METHOD_NAME]/METHOD_DETAILS
    EXPERIMENT_DIR="${DATASET_DIR}/${BASE_EXPERIMENT_NAME}"
    METHOD_DIR="${EXPERIMENT_DIR}/method_${MERGE_METHOD}"
    RUN_FOLDER="${METHOD_DIR}/${METHOD_DETAILS}"
    
    mkdir -p "${RUN_FOLDER}"
    
    # Output files
    LOG_FILE="${RUN_FOLDER}/results_log.txt"
    
    # Build command
    CMD="python eval_prob_adaptive_dit.py \
        --dataset ${DATASET} \
        --img_size ${IMG_SIZE} \
        --split ${SPLIT} \
        --class_csv_path ${CLASS_CSV_PATH} \
        --num_classes ${NUM_CLASSES} \
        --to_keep ${TO_KEEP} \
        --n_samples ${N_SAMPLES} \
        --batch_size ${BATCH_SIZE} \
        --n_trials ${N_TRIALS} \
        --dtype ${DTYPE} \
        --interpolation ${INTERPOLATION} \
        --n_workers ${N_WORKERS} \
        --worker_idx ${WORKER_IDX} \
        --loss ${LOSS} \
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
        --downsample_factor_level_2 ${DOWNSAMPLE_FACTOR_LEVEL_2} \
        --timestep_threshold_switch ${TIMESTEP_THRESHOLD_SWITCH} \
        --timestep_threshold_stop ${TIMESTEP_THRESHOLD_STOP} \
        --secondary_merge_method ${SECONDARY_MERGE_METHOD} \
        --ratio_level_2 ${RATIO_LEVEL_2} \
        --extra_guidance_scale ${EXTRA_GUIDANCE_SCALE} \
        --frequency_selection_mode ${FREQUENCY_SELECTION_MODE} \
        --frequency_selection_method ${FREQUENCY_SELECTION_METHOD} \
        --frequency_ranking_method ${FREQUENCY_RANKING_METHOD} \
        --selection_source ${SELECTION_SOURCE} \
        --frequency_grid_alpha ${FREQUENCY_GRID_ALPHA} \
        --qkv_downsample_method ${QKV_DOWNSAMPLE_METHOD} \
        --out_upsample_method ${OUT_UPSAMPLE_METHOD} \
        --if_deepcache ${IF_DEEPCACHE} \
        --cache_interval ${CACHE_INTERVAL} \
        --cache_branch_id ${CACHE_BRANCH_ID} \
        --if_toca ${IF_TOCA} \
        --toca_fresh_ratio ${TOCA_FRESH_RATIO} \
        --toca_fresh_threshold ${TOCA_FRESH_THRESHOLD} \
        --toca_cache_type ${TOCA_CACHE_TYPE} \
        --toca_ratio_scheduler ${TOCA_RATIO_SCHEDULER} \
        --seed ${SEED} \
        --summary_csv ${SUMMARY_CSV} \
        --eval_denoise_steps ${EVAL_DENOISE_STEPS} \
        --single_timestep ${SINGLE_TIMESTEP} \
        --eval_step_stride ${EVAL_STEP_STRIDE} \
        --eval_error_method ${EVAL_ERROR_METHOD} \
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
        --abp_tile_aggregation ${ABP_TILE_AGGREGATION} \
        --load_stats" # --load_stats force_recalc
    
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
    if [[ -n "${ABP_SCORER}" ]]; then
        CMD+=" --abp_scorer ${ABP_SCORER}"
    fi
    
    # Add optional flags
    if [[ ${LOAD_STATS} -eq 1 ]]; then
        CMD+=" --load_stats"
    fi
    
    if [[ "${EXTRA}" != "None" ]] && [[ -n "${EXTRA}" ]]; then
        CMD+=" --extra ${EXTRA}"
    fi
    
    # --- Execute The Command ---
    echo "======================================================"
    echo "Starting Run: ${BASE_EXPERIMENT_NAME} | Method: ${MERGE_METHOD}"
    echo "Method Details: ${METHOD_DETAILS}"
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
