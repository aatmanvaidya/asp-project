#!/bin/bash
#SBATCH --job-name=emotion_pipeline
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

# One array task per model, each on its own single A100, all writing under
# the same outputs/ dir. Keep --array=0-N above in sync with (length - 1) of
# this list.
MODELS=(
    wav2vec2-base
    hubert-xlarge
    wavlm-large
    distilhubert
    hubert-base
    wavlm-base
    wav2vec2-large
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================"
echo "Emotion Recognition Pipeline"
echo "============================================"
echo "Array Job : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node      : $SLURM_NODELIST"
echo "Model     : $MODEL"
echo "Start time: $(date)"
echo ""

PROJECT_ROOT="/home/tu/tu_tu/tu_zxord71/asp-project"
SCRIPT_DIR="$PROJECT_ROOT/experiments"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

echo "Loading modules..."
module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA Home : $CUDA_HOME"
echo "Python    : $(which python)"
echo ""

echo "Setting up project environment..."
cd "$PROJECT_ROOT" || exit 1
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

source "$PROJECT_ROOT/.venv/bin/activate"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r
echo "Python isolation enforced:"
echo "  PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
echo "  VIRTUAL_ENV=$VIRTUAL_ENV"
echo "  Python used: $(which python)"
echo ""

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Sourcing $PROJECT_ROOT/.env"
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Optuna HPO (single GPU, this model only)
#
# Runs 10 Optuna trials × 3 epochs per experiment to find the best
# learning rate, batch size, warmup ratio, and weight decay.
# Saves best_hyperparameters.json for each model/dataset combination.
# Adjust --hpo_trials to trade search quality against wall-clock time:
#   20 trials ≈ good coverage | 10 trials ≈ faster | 5 trials ≈ quick sanity
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================"
echo "Stage 1: Optuna HPO ($MODEL, 1 GPU, 10 trials each)"
echo "============================================"
echo ""

CUDA_VISIBLE_DEVICES=0 uv run "$SCRIPT_DIR/run_all.py" \
    --output_dir "$OUTPUT_DIR" \
    --models "$MODEL" \
    --hpo_trials 10 \
    --hpo_only \
    --seed 42

STAGE1_EXIT=$?
echo ""
echo "Stage 1 exit code: $STAGE1_EXIT"

if [ $STAGE1_EXIT -ne 0 ]; then
    echo "Stage 1 (HPO) failed. Aborting."
    exit $STAGE1_EXIT
fi

echo ""
# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — 5-fold CV training (single GPU, this model only)
#
# Loads the best_hyperparameters.json saved in Stage 1 (tuned once, not
# re-tuned per fold) and, for each experiment, trains + evaluates 5 fresh
# models via stratified K-fold CV, reporting F1-macro etc. as mean ± 95% CI
# across folds. --skip_done lets you safely re-submit the job if it was
# interrupted: completed folds (fold_*/metrics.json) and completed
# experiments (cv_metrics.json) are both skipped automatically.
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================"
echo "Stage 2: 5-fold CV training ($MODEL, 1 GPU)"
echo "============================================"
echo ""

CUDA_VISIBLE_DEVICES=0 uv run "$SCRIPT_DIR/run_all.py" \
    --output_dir "$OUTPUT_DIR" \
    --models "$MODEL" \
    --epochs 30 \
    --batch_size 32 \
    --k_folds 5 \
    --seed 42 \
    --skip_done

STAGE2_EXIT=$?

echo ""
echo "============================================"
echo "Job finished"
echo "============================================"
echo "Stage 1 exit : $STAGE1_EXIT"
echo "Stage 2 exit : $STAGE2_EXIT"
echo "End time     : $(date)"
echo ""

if [ $STAGE2_EXIT -eq 0 ]; then
    echo "Success. Outputs:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    if [ -f "$OUTPUT_DIR/report.md" ]; then
        echo "=== report.md (partial — this task's model ($MODEL) only;"
        echo "    every array task overwrites this file, so it does NOT"
        echo "    accumulate across tasks. Once ALL array tasks finish,"
        echo "    run this once (no --models filter) to regenerate the"
        echo "    full report from every completed cv_metrics.json:"
        echo "      uv run \"$SCRIPT_DIR/run_all.py\" --output_dir \"$OUTPUT_DIR\" --skip_done"
        echo "    ==="
    fi
else
    echo "Failure: Stage 2 exited with code $STAGE2_EXIT"
    echo "Check logs: logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
fi

exit $STAGE2_EXIT
