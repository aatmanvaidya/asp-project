#!/bin/bash
# Submit the full HPO -> 5-fold CV -> aggregate chain for one model, with each
# stage waiting for the previous one to fully succeed (sbatch --dependency).
#
# Usage (run from the project root, so the logs/ relative --output= paths in
# experiments/jobs/*.sbatch resolve correctly):
#   experiments/submit_pipeline.sh <model_name>
#   e.g. experiments/submit_pipeline.sh wav2vec2-base
#
# Run once per model in MODELS (experiments/pipeline/config.py). Replaces the
# old single sbatch job that ran HPO + all 5 folds for a model in one Python
# process: a crash partway through (disk quota, OOM, node failure, ...) used
# to silently drop every fold after the one that failed, for that whole
# model. Now:
#   - Stage 1 (HPO) and Stage 3 (aggregate) are single jobs.
#   - Stage 2 (CV) is a SLURM array of 5 independent tasks, one per fold, so a
#     failed fold can be resubmitted on its own instead of re-running (or
#     silently losing) the rest.
#   - --dependency=afterok chains the stages: Stage 2 only starts once Stage
#     1 exits 0, and afterok on an array job waits for every task in it, so
#     Stage 3 only starts once all 5 folds have succeeded.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_name>" >&2
    exit 1
fi
MODEL="$1"
JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/jobs" && pwd)"

mkdir -p logs

HPO_JOBID=$(sbatch --parsable --export=ALL,MODEL="$MODEL" "$JOBS_DIR/01_hpo.sbatch")
echo "Stage 1 (HPO)       : job $HPO_JOBID"

CV_JOBID=$(sbatch --parsable --export=ALL,MODEL="$MODEL" \
    --dependency=afterok:"$HPO_JOBID" "$JOBS_DIR/02_cv_fold.sbatch")
echo "Stage 2 (5-fold CV) : job $CV_JOBID (array 0-4)"

AGG_JOBID=$(sbatch --parsable --export=ALL,MODEL="$MODEL" \
    --dependency=afterok:"$CV_JOBID" "$JOBS_DIR/03_aggregate.sbatch")
echo "Stage 3 (aggregate) : job $AGG_JOBID"

echo ""
echo "Chain submitted for $MODEL: $HPO_JOBID -> $CV_JOBID (array 0-4) -> $AGG_JOBID"
echo "Check status: squeue -j $HPO_JOBID,$CV_JOBID,$AGG_JOBID"
echo ""
echo "If a fold fails, resubmit just that one after inspecting its log"
echo "(logs/emo_cv_${CV_JOBID}_fold<N>.err):"
echo "  sbatch --export=ALL,MODEL=$MODEL --array=<N> --dependency=afterok:$HPO_JOBID $JOBS_DIR/02_cv_fold.sbatch"
echo ""
echo "Once every model's chain has completed, regenerate the full cross-model"
echo "report once (no --models filter):"
echo "  uv run experiments/run_all.py --output_dir outputs --skip_done"
