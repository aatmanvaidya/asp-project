#!/bin/bash
#SBATCH --job-name=emotion_pipeline
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

# Single job, single A100 GPU, all models x datasets from config.py
# (CAMEO disabled there). run_all.py handles HPO + CV training end-to-end
# for every combo in one process — see its module docstring.
echo "============================================"
echo "Emotion Recognition Pipeline"
echo "============================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
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

echo "============================================"
echo "Running pipeline (1 GPU, HPO + CV training)"
echo "============================================"
echo ""

CUDA_VISIBLE_DEVICES=0 uv run "$SCRIPT_DIR/run_all.py" \
    --output_dir "$OUTPUT_DIR" \
    --hpo_trials 10 \
    --epochs 30 \
    --batch_size 32 \
    --k_folds 5 \
    --seed 42 \
    --skip_done

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Job finished"
echo "============================================"
echo "Exit code : $EXIT_CODE"
echo "End time  : $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Success. Outputs:"
    ls -lh "$OUTPUT_DIR"
else
    echo "Failure: exited with code $EXIT_CODE"
    echo "Check logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
