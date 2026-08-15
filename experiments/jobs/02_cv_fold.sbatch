#!/bin/bash
#SBATCH --job-name=emo_cv
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_fold%a.out
#SBATCH --error=logs/%x_%A_fold%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aatman-vrundavan.vaidya@student.uni-tuebingen.de

# Stage 2 of 3 — one 5-fold-CV fold per array task, one model, all datasets.
# Not meant to be sbatch'd directly: submit via experiments/submit_pipeline.sh
# <model_name>.
#
# Each of the 5 array tasks is an independent SLURM job step: a crash in
# fold 2 (OOM, disk hiccup, node failure, ...) no longer takes the other 4
# folds down with it. Requires best_hyperparameters.json from Stage 1 to
# already exist (run_all.py fails fast otherwise rather than 5 array tasks
# each redundantly re-running Optuna).
#
# Re-running just the failed fold after a partial failure:
#   sbatch --export=ALL,MODEL=<model_name> --array=2 experiments/jobs/02_cv_fold.sbatch

source "/home/tu/tu_tu/tu_zxoqp65/work/asp-project/experiments/jobs/_env.sh"

echo "============================================"
echo "Stage 2: fold $SLURM_ARRAY_TASK_ID/5 CV training ($MODEL, 1 GPU)"
echo "============================================"
echo ""

CUDA_VISIBLE_DEVICES=0 uv run "$SCRIPT_DIR/run_all.py" \
    --output_dir "$OUTPUT_DIR" \
    --models "$MODEL" \
    --epochs 30 \
    --batch_size 32 \
    --k_folds 5 \
    --fold "$SLURM_ARRAY_TASK_ID" \
    --seed 42 \
    --skip_done

EXIT=$?
echo ""
echo "Stage 2 fold $SLURM_ARRAY_TASK_ID exit code: $EXIT"
exit $EXIT
