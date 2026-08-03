#!/bin/bash
# Submit a hydro-LLM (Time-LLM identity) matrix job on UBELIX, free tier only.
#
# Drives experiments/hydro_llm/run_matrix.py, which invokes the VALIDATED swiss
# Time-LLM harness underneath (channel-independent N=1 path, same as the published
# n=3 cells) with the CLI-override + results.json fixes. Single seed by default:
# multi-seed sweeps are on HOLD and must be requested explicitly.
#
# Usage (from ~/codes/liulian-python on the cluster):
#   RUNTAG=hydro-1990 DATASETS="swiss-river-1990" \
#     MODES="none embedding random_embedding entity_description" SEEDS="2021" \
#     sbatch jobs/run_hydro_llm.sh
#
# Free-tier guardrails (never change without explicit authorisation):
#   --account=gratis  --qos=job_gratis  --gres=gpu:rtx4090:1  (max 2x4090 or 1xh100)

#SBATCH --job-name=hydrollm
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output=outputs/hydrollm.%x.o%J
#SBATCH --error=errors/hydrollm.%x.e%J

set -euo pipefail

RUNTAG="${RUNTAG:?set RUNTAG}"
DATASETS="${DATASETS:?set DATASETS}"
MODES="${MODES:?set MODES}"
SEEDS="${SEEDS:-2026}"
PHASE="${PHASE:-full}"   # full=Ray Tune HPO; dev=5ep no HPO (fast first baseline)

echo "=== hydro-llm job: tag=$RUNTAG datasets=[$DATASETS] modes=[$MODES] seeds=[$SEEDS] ==="
echo "node=$(hostname) date=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module load Python/3.12.3-GCCcore-13.3.0
# shellcheck disable=SC1091
source "$HOME/codes/liulian-python/.venv/bin/activate"

cd "$HOME/codes/liulian-python"
mkdir -p outputs errors

# HF weights are cached on the cluster; stay offline so a cell never blocks on a
# network fetch of GPT-2.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

# --resume: on a requeue after the wall clock, cells already ok in the manifest are
# skipped so only the unfinished ones re-run.
python experiments/hydro_llm/run_matrix.py \
  --phase "$PHASE" \
  --run-tag "$RUNTAG" \
  --datasets $DATASETS \
  --modes $MODES \
  --seeds $SEEDS \
  --resume

echo "=== done: $(date -Iseconds) ==="
