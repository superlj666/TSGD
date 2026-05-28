#!/bin/bash

# ==============================================================================
# TSGD: Combined Initialization and Regularization (Pruning) Workflow
# ==============================================================================

### 1. Global Configuration
DATA_NAME="math_precalculus_5_train"
MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0.0
EMBEDDING_MODEL="text-embedding-3-large"

### 2. Initialization Parameters
MAX_SAMPLES=9999
MAX_WORKERS=10

### 3. Regularization Parameters
TOTAL_LIMIT=300
SUBJECT_LIMIT=100
EXP_MAX_TOKENS=1000

### 4. Paths and Logistics
INPUT_PATH="data"
SAFE_MODEL=$(echo "${MODEL_NAME}" | sed 's/\//_/g' | sed 's/:/_/g')
SAFE_DATASET=$(echo "${DATA_NAME}" | sed 's/\//_/g')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output directories
INIT_DIR="${INPUT_PATH}/${SAFE_DATASET}_init_${SAFE_MODEL}"
REG_DIR="${INPUT_PATH}/${SAFE_DATASET}_reg_${SAFE_MODEL}"

# Logging
LOG_FILE="logs/workflow_${SAFE_MODEL}_${SAFE_DATASET}_${TIMESTAMP}.log"

# Ensure directories exist
mkdir -p "$INIT_DIR" "$REG_DIR" "logs"

# ------------------------------------------------------------------------------
# STEP 1: Initialization (Mining)
# ------------------------------------------------------------------------------
echo "===================================================================="
echo ">>> STEP 1: Initialization | Model: $MODEL_NAME"
echo ">>> Log: $LOG_FILE"
echo "===================================================================="

python3 src/agents/initializer.py \
    --initializer_model "$MODEL_NAME" \
    --initializer_temperature "$TEMPERATURE" \
    --input_path "${INPUT_PATH}/${DATA_NAME}.jsonl" \
    --max_samples "$MAX_SAMPLES" \
    --max_workers "$MAX_WORKERS" \
    --experience_path "$INIT_DIR" \
    --log_file "$LOG_FILE" \
    --embedding_model "$EMBEDDING_MODEL" \
    --debug

if [ $? -ne 0 ]; then
    echo -e "\n[ERROR] Initialization failed! Please check: $LOG_FILE"
    exit 1
fi

# ------------------------------------------------------------------------------
# STEP 2: Regularization (Pruning)
# ------------------------------------------------------------------------------
PROBLEM_EMB="${INPUT_PATH}/${DATA_NAME}_${EMBEDDING_MODEL}_idx.npz"

echo -e "\n===================================================================="
echo ">>> STEP 2: Regularization | Model: $MODEL_NAME"
echo ">>> Log: $LOG_FILE"
echo "===================================================================="

python3 src/agents/regularizer.py \
    --regularizer_model "$MODEL_NAME" \
    --regularizer_temperature "$TEMPERATURE" \
    --problem_embedding_path "$PROBLEM_EMB" \
    --experience_path "${INIT_DIR}/experience_pool.jsonl" \
    --output_path "${REG_DIR}/experience_pool.jsonl" \
    --meta_path "${INIT_DIR}/question_meta.json" \
    --meta_output_path "${REG_DIR}/question_meta.json" \
    --total_limit "$TOTAL_LIMIT" \
    --subject_limit "$SUBJECT_LIMIT" \
    --exp_max_tokens "$EXP_MAX_TOKENS" \
    --project_name "explearn_reg" \
    --log_file "$LOG_FILE" \
    --debug

if [ $? -ne 0 ]; then
    echo "!!! Regularization failed. Check $LOG_FILE"
    exit 1
fi

echo -e "\n===================================================================="
echo ">>> Workflow Complete!"
echo "Result: ${REG_DIR}/experience_pool.jsonl"
echo "Log: $LOG_FILE"
echo "===================================================================="
