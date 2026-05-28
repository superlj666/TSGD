#!/bin/bash

### Evaluation type
TYPE="eval_llm" 
# TYPE="eval_llm_exp" 
# TYPE="eval_llm_k" 
# TYPE="eval_llm_exp_k"

### OpenRouter
SOLVER_MODEL="x-ai/grok-4.1-fast"
SOLVER_TEMPERATURE=0.0

### Dataset
# DATASET_NAME="yentinglin/aime_2025"
# DATASET_NAME="HuggingFaceH4/aime_2024"
# DATASET_NAME="KbsdJames/Omni-MATH"
# DATASET_NAME="data/omni_math_rule.jsonl"
# DATASET_NAME="HuggingFaceH4/MATH-500"
# DATASET_NAME="data/math_IA_5_test.jsonl"
# DATASET_NAME="data/math_test.jsonl"
# DATASET_NAME="data/math_precalculus_5_test.jsonl"
DATASET_NAME="data/math_geometry_5_test.jsonl"

# Baseline LLM Evaluation
K=48
MAX_WORKERS=10
TIMEOUT=300
MAX_TOKENS=4096
MAX_RETRIES=3
MAX_SAMPLES=9999 ### small when debug
SEED=42

### Experience Library
EXPERIENCE_DIR="data/math_precalculus_5_train_init_x-ai_grok-4.1-fast"
# EXPERIENCE_DIR="data/math_precalculus_5_train_reg_x-ai_grok-4.1-fast"
SIMILARITY_THRESHOLD=0.2
RETRIEVAL_TOP_K=10
RETRIEVAL_MODE="problem" # Options: subject, vector, hybrid
PROBLEM_EMBEDDING_PATH="data/math_precalculus_5_train_text-embedding-3-large_idx.npz"
RERANK=false  # Set to true to enable Agentic Rerank

### Experience Pool Limits
TOTAL_LIMIT=300
SUBJECT_LIMIT=100
EXP_MAX_TOKENS=1024

### ROLLOUT LIST
K_VALUES=(1 2 4 8 16 32)

echo ">>> Running $TYPE on $DATASET_NAME with $SOLVER_MODEL ..."
echo "---------------- Hyperparameters ----------------"
echo "TYPE:          $TYPE"
echo "SOLVER_MODEL:    $SOLVER_MODEL"
echo "DATASET_NAME:  $DATASET_NAME"
echo "SOLVER_TEMPERATURE:   $SOLVER_TEMPERATURE"
echo "K (pass@k):    $K"
echo "MAX_WORKERS:   $MAX_WORKERS"
echo "MAX_TOKENS:    $MAX_TOKENS"
echo "TIMEOUT:       $TIMEOUT"
echo "MAX_RETRIES:   $MAX_RETRIES"
echo "MAX_SAMPLES:   $MAX_SAMPLES"
echo "EXP_DIR:       $EXPERIENCE_DIR"
echo "SIM_THRESHOLD: $SIMILARITY_THRESHOLD"
echo "RETR_TOP_K:    $RETRIEVAL_TOP_K"
echo "RETR_MODE:     $RETRIEVAL_MODE"
echo "RERANK:        $RERANK"
echo "-------------------------------------------------"

# Define a function to build the command base to avoid duplication
run_inference() {
    local k=$1
    local exp_dir=$2
    
    # Sanitize names for filename in shell
    local SAFE_MODEL=$(echo "${SOLVER_MODEL}" | sed 's/\//_/g' | sed 's/:/_/g')
    local SAFE_DATASET=$(echo "${DATASET_NAME}" | sed 's/\//_/g')
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local LOG_FILE="logs/run_${TYPE}_${SAFE_MODEL}_${SAFE_DATASET}_pass${k}_${TIMESTAMP}.log"

    # Prepare common arguments
    local args=(
        --solver_model "$SOLVER_MODEL"
        --dataset_name "$DATASET_NAME"
        --project_name "Eval_${SOLVER_MODEL}_${DATASET_NAME}"
        --max_workers "$MAX_WORKERS"
        --k "$k"
        --solver_temperature "$SOLVER_TEMPERATURE"
        --max_tokens "$MAX_TOKENS"
        --timeout "$TIMEOUT"
        --max_retries "$MAX_RETRIES"
        --max_samples "$MAX_SAMPLES"
        --seed "$SEED"
        --log_file "$LOG_FILE"
        --debug
    )

    # Add experience-related arguments if exp_dir is provided
    if [ -n "$exp_dir" ]; then
        args+=(
            --experience_dir "$exp_dir"
            --problem_embedding_path "$PROBLEM_EMBEDDING_PATH"
            --similarity_threshold "$SIMILARITY_THRESHOLD"
            --retrieval_top_k "$RETRIEVAL_TOP_K"
            --retrieval_mode "$RETRIEVAL_MODE"
            --total_limit "$TOTAL_LIMIT"
            --subject_limit "$SUBJECT_LIMIT"
            --exp_max_tokens "$EXP_MAX_TOKENS"
        )
        
        if [ "$RERANK" = true ]; then
            args+=(--rerank)
        fi
    fi

    python src/core/inference.py "${args[@]}"
}

### Evaluate LLM with rollout@k
if [ "$TYPE" == "eval_llm" ]; then
    run_inference "$K" ""
fi

### Evaluate LLM with rollout@(1 2 ...)
if [ "$TYPE" == "eval_llm_k" ]; then
    for K_VAL in "${K_VALUES[@]}"
    do
        run_inference "$K_VAL" ""
        echo ">>> Completed K=$K_VAL"
        echo ""
    done
fi

### Evaluate LLM + Experience Library with rollout@k
if [ "$TYPE" == "eval_llm_exp" ]; then
    run_inference "$K" "$EXPERIENCE_DIR"
fi

### Evaluate LLM + Experience Library with rollout@(1 2 ...)
if [ "$TYPE" == "eval_llm_exp_k" ]; then
    for K_VAL in "${K_VALUES[@]}"
    do
        run_inference "$K_VAL" "$EXPERIENCE_DIR"
        echo ">>> Completed K=$K_VAL"
        echo ""
    done
fi