#!/bin/bash
#SBATCH --job-name=emo_hpo
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

# Stage 1 of 3 — Optuna HPO for one model, all datasets.
# Not meant to be sbatch'd directly: submit via experiments/submit_pipeline.sh
# <model_name>, which sets MODEL and chains stages 2 and 3 after this one.

source "/home/tu/tu_tu/tu_zxoqp65/work/asp-project/experiments/jobs/_env.sh"

echo "============================================"
echo "Stage 1: Optuna HPO ($MODEL, 1 GPU, 10 trials/experiment)"
echo "============================================"
echo ""

CUDA_VISIBLE_DEVICES=0 uv run "$SCRIPT_DIR/run_all.py" \
    --output_dir "$OUTPUT_DIR" \
    --models "$MODEL" \
    --hpo_trials 10 \
    --hpo_only \
    --seed 42

EXIT=$?
echo ""
echo "Stage 1 exit code: $EXIT"
exit $EXIT
