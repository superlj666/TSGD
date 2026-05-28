#!/bin/bash
set -e

# Test script for TSGD Trainer Flow
# This script focuses on experimental hyperparameters. 
# Infrastructure and stable defaults are managed in src/.env

# 1. Setup
echo ">>> Setting up test environment..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LOG_LEVEL=DEBUG

# --- Configuration (Experimental) ---
# Global Defaults
MODEL_NAME="x-ai/grok-4.1-fast"
TEMPERATURE=0.3

# Agent Specific Settings
# 1. Solver (Main reasoning agent)
SOLVER_MODEL="x-ai/grok-4.1-fast"
SOLVER_TEMPERATURE=0.0

# 2. Optimizer (Experience extraction)
OPTIMIZER_MODEL="x-ai/grok-4.1-fast"
OPTIMIZER_TEMPERATURE=0.0

# 3. Initializer (Seed generation)
INITIALIZER_MODEL="x-ai/grok-4.1-fast"
INITIALIZER_TEMPERATURE=0.0

# 4. Regularizer (Utility scoring)
REGULARIZER_MODEL="x-ai/grok-4.1-fast"
REGULARIZER_TEMPERATURE=0.0

# Data & Path Settings
TRAIN_DATA="./data/math_precalculus_5_train.jsonl"
VAL_DATA="./data/math_precalculus_5_val.jsonl"
TEST_DATA="./data/math_precalculus_5_test.jsonl"
EXPERIENCE_DIR="experiments/exp6_init+train/math_precalculus_5_train_init_x-ai_grok-4.1-fast"
EMBEDDING_PATH="./data/math_precalculus_5_train_text-embedding-3-large_idx.npz"

# Pass@K Evaluation
K=1

# Training Sample Sizes
EPOCHS=3
SEED_SAMPLES=2
TRAIN_SAMPLES=200

# Resource & Constraints
MAX_WORKERS=10
TOTAL_LIMIT=300
SUBJECT_LIMIT=9999
EXP_MAX_TOKENS=1024

# Log Configuration
SAFE_MODEL=$(echo "${MODEL_NAME}" | sed 's/\//_/g' | sed 's/:/_/g')
SAFE_DATASET="precalculus"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# LOG_FILE="${EXPERIENCE_DIR}/run_trainer_${SAFE_MODEL}_${SAFE_DATASET}_pass${K}_${TIMESTAMP}.log"
LOG_FILE="logs/run_trainer_${SAFE_MODEL}_${SAFE_DATASET}_pass${K}_${TIMESTAMP}.log"

# --- Execution ---
echo ">>> Running Trainer (Seeding + TSGD)..."
echo "---------------- Configuration ----------------"
echo "Train Data:   $TRAIN_DATA"
echo "Val Data:     $VAL_DATA"
echo "Exp Dir:      $EXPERIENCE_DIR"
echo "Global Model: $MODEL_NAME (Temp: $TEMPERATURE)"
echo "Solver:       ${SOLVER_MODEL:-$MODEL_NAME} (Temp: $SOLVER_TEMPERATURE)"
echo "Optimizer:    ${OPTIMIZER_MODEL:-$MODEL_NAME} (Temp: $OPTIMIZER_TEMPERATURE)"
echo "Initializer:  ${INITIALIZER_MODEL:-$MODEL_NAME} (Temp: $INITIALIZER_TEMPERATURE)"
echo "Regularizer:  ${REGULARIZER_MODEL:-$MODEL_NAME} (Temp: $REGULARIZER_TEMPERATURE)"
echo "Epochs:       $EPOCHS"
echo "-----------------------------------------------"

python src/core/train.py \
    --dataset_name "precalculus" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --experience_dir "$EXPERIENCE_DIR" \
    --embedding_path "$EMBEDDING_PATH" \
    --model_name "$MODEL_NAME" \
    --solver_model "$SOLVER_MODEL" \
    --optimizer_model "$OPTIMIZER_MODEL" \
    --initializer_model "$INITIALIZER_MODEL" \
    --regularizer_model "$REGULARIZER_MODEL" \
    --epochs "$EPOCHS" \
    --seed_samples "$SEED_SAMPLES" \
    --train_samples "$TRAIN_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --solver_temperature "$SOLVER_TEMPERATURE" \
    --optimizer_temperature "$OPTIMIZER_TEMPERATURE" \
    --initializer_temperature "$INITIALIZER_TEMPERATURE" \
    --regularizer_temperature "$REGULARIZER_TEMPERATURE" \
    --max_workers "$MAX_WORKERS" \
    --total_limit "$TOTAL_LIMIT" \
    --subject_limit "$SUBJECT_LIMIT" \
    --exp_max_tokens "$EXP_MAX_TOKENS" \
    --log_file "$LOG_FILE" \
    --enable_dual_verification \
    --debug

# 3. Verify Output
if [ -d "$EXPERIENCE_DIR" ] && [ -f "$EXPERIENCE_DIR/experience_pool.jsonl" ]; then
    echo ">>> Success: Experience pool created at $EXPERIENCE_DIR"
else
    echo ">>> Error: Experience pool not found at $EXPERIENCE_DIR!"
    exit 1
fi
