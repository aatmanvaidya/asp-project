#!/bin/bash
#SBATCH --job-name=emo_aggregate
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

# Stage 3 of 3 — aggregate this model's 5 completed folds into cv_metrics.json
# (mean +/- 95% CI) and regenerate outputs/report.md.
# Not meant to be sbatch'd directly: submit via experiments/submit_pipeline.sh
# <model_name>, which chains this after Stage 2's array job succeeds
# (afterok on an array job waits for every array task, not just the first).
#
# This step only reads fold_*/metrics.json and computes stats — no GPU work.
# Left on gpu_a100_il / --gres=gpu:1 to match the partition already known to
# work on this cluster; switch to a CPU partition here if one exists, to free
# up the GPU allocation for this ~1-minute step.
#
# NOTE (matches the prior single-job pipeline's behavior): report.md is
# regenerated from only the --models filtered here, so each model's Stage 3
# job overwrites it with that model's rows, it does NOT accumulate across
# models. Once every model's chain has completed, regenerate the full
# cross-model report once (no --models filter):
#   uv run experiments/run_all.py --output_dir outputs --skip_done

source "/home/tu/tu_tu/tu_zxoqp65/work/asp-project/experiments/jobs/_env.sh"

echo "============================================"
echo "Stage 3: aggregating CV folds ($MODEL)"
echo "============================================"
echo ""

uv run "$SCRIPT_DIR/run_all.py" \
    --output_dir "$OUTPUT_DIR" \
    --models "$MODEL" \
    --k_folds 5 \
    --aggregate_only \
    --skip_done

EXIT=$?
echo ""
echo "Stage 3 exit code: $EXIT"
if [ $EXIT -eq 0 ]; then
    echo "$MODEL's chain (HPO -> 5-fold CV -> aggregate) is complete."
fi
exit $EXIT
