#!/bin/bash
#SBATCH --job-name=wav2vec2_ravdess_emotion
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=0:29:55
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

echo "============================================"
echo "Wav2Vec2 RAVDESS Emotion Classification"
echo "============================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

PROJECT_ROOT="/home/tu/tu_tu/tu_zxord71/asp-project"
SCRIPT_DIR="$PROJECT_ROOT/experiments"
OUTPUT_DIR="$SCRIPT_DIR/wav2vec2-ravdess-output"

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

# HuggingFace cache — point to scratch to avoid quota issues on home
export HF_HOME="$PROJECT_ROOT/.hf_cache"
mkdir -p "$HF_HOME"
echo "HF_HOME   : $HF_HOME"

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Sourcing $PROJECT_ROOT/.env"
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

echo "=========================================="
echo "Starting wav2vec2_ravdess_emotion.py ..."
echo "=========================================="
echo ""

python "$SCRIPT_DIR/wav2vec2_ravdess_emotion.py" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed"
echo "=========================================="
echo "Exit code : $EXIT_CODE"
echo "End time  : $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Success: training completed."
    echo "Outputs in: $OUTPUT_DIR"
    echo "  confusion_matrix.png"
    echo "  training_curves.png"
    echo "  model checkpoints"
else
    echo "Failure: job exited with code $EXIT_CODE"
    echo "Check logs: logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
