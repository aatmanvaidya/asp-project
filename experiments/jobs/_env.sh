#!/bin/bash
# Shared environment setup, sourced (not submitted) by every experiments/jobs/*.sbatch
# script. Not itself a SLURM script — no #SBATCH directives.
#
# Uses an absolute path rather than deriving PROJECT_ROOT from the sbatch
# script's own location: SLURM copies the submitted script into a per-job
# spool directory before executing it, so ${BASH_SOURCE[0]} inside a running
# job points at /var/spool/... , not the repo. This was the cause of one
# recurring failure ("Permission denied" cd-ing into a stale/wrong path baked
# into an earlier version of this script).
PROJECT_ROOT="/home/tu/tu_tu/tu_zxoqp65/work/asp-project"
SCRIPT_DIR="$PROJECT_ROOT/experiments"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

if [ -z "$MODEL" ]; then
    echo "MODEL is not set. Submit via experiments/submit_pipeline.sh <model_name>, not sbatch directly." >&2
    exit 1
fi

echo "============================================"
echo "Emotion Recognition Pipeline"
echo "============================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Array idx : ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Node      : $SLURM_NODELIST"
echo "Model     : $MODEL"
echo "Start time: $(date)"
echo ""

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
