#!/bin/bash
#SBATCH --job-name=snr_analysis
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000mb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

# CPU-only job: SNR estimation is plain audio decode + numpy/scipy, no GPU
# needed. Reads model predictions already produced by run_pipeline.sh, so
# this must run after that job (or against an existing outputs/ dir).
echo "============================================"
echo "SNR vs. Model Correctness — Correlation Analysis"
echo "============================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

PROJECT_ROOT="/home/tu/tu_tu/tu_zxord71/asp-project"
SCRIPT_DIR="$PROJECT_ROOT/experiments/snr_analysis"
OUTPUT_DIR="$PROJECT_ROOT/outputs"

echo "Loading modules..."
module load devel/python/3.13.3-llvm-19.1
echo "Python    : $(which python)"
echo ""

echo "Setting up project environment..."
cd "$PROJECT_ROOT" || exit 1
mkdir -p logs

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

echo "============================================"
echo "Running SNR correlation analysis (CPU only)"
echo "============================================"
echo ""

uv run "$SCRIPT_DIR/run_snr_analysis.py" \
    --predictions_root "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR/snr_analysis"

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
    ls -lh "$OUTPUT_DIR/snr_analysis"
else
    echo "Failure: exited with code $EXIT_CODE"
    echo "Check logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
