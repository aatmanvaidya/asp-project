#!/bin/bash
#SBATCH --job-name=emotion_pipeline
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

echo "============================================"
echo "Emotion Recognition Pipeline"
echo "============================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

PROJECT_ROOT="/home/tu/tu_tu/tu_zxoqp65/work/asp-project"
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
# Stage 1 — Optuna HPO (single GPU)
#
# Runs 20 Optuna trials × 3 epochs per experiment to find the best
# learning rate, batch size, warmup ratio, and weight decay.
# Saves best_hyperparameters.json for each model/dataset combination.
# Adjust --hpo_trials to trade search quality against wall-clock time:
#   20 trials ≈ good coverage | 10 trials ≈ faster | 5 trials ≈ quick sanity
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================"
echo "Stage 1: Optuna HPO (1 GPU, 20 trials each)"
echo "============================================"
echo ""

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/run_all.py" \
    --output_dir "$OUTPUT_DIR" \
    --datasets ravdess cameo \
    --hpo_trials 20 \
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
# Stage 2 — Full training (4-GPU DDP via torchrun)
#
# Loads the best_hyperparameters.json saved in Stage 1 for each experiment
# and trains the final model with early stopping. --skip_done lets you
# safely re-submit the job if it was interrupted: completed experiments
# (those with metrics.json) are skipped automatically.
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================"
echo "Stage 2: Full training (4-GPU DDP, torchrun)"
echo "============================================"
echo ""

torchrun \
    --standalone \
    --nproc_per_node=4 \
    "$SCRIPT_DIR/run_all.py" \
    --output_dir "$OUTPUT_DIR" \
    --datasets ravdess cameo \
    --epochs 30 \
    --batch_size 8 \
    --seed 42 \
    --skip_done \
    --include_mfcc \
    --mfcc_epochs 50

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
        echo "=== report.md ==="
        cat "$OUTPUT_DIR/report.md"
    fi
else
    echo "Failure: Stage 2 exited with code $STAGE2_EXIT"
    echo "Check logs: logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $STAGE2_EXIT
