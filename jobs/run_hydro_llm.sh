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
#
# Memory-bound cells (LLAMA arm, llm_layers=12 retrains): override the GPU at submit
# time — CLI beats the header, account/qos stay gratis (1x H100 96 GB is gratis-legal):
#   sbatch --gres=gpu:h100:1 --cpus-per-task=8 jobs/run_hydro_llm.sh

#SBATCH --job-name=hydrollm
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
# MEASURED 2026-08-07: the gpu partition TIMELIMIT is 1-00:00:00 (sinfo -p gpu) and a
# 96:00:00 request is REJECTED at submit ("Requested time limit is invalid"). Long
# sweeps therefore run as 24h segments: --resume (manifest skip + Ray Tune resume)
# continues finished/partial cells, and submit an afterany-dependency successor job
# (sbatch --dependency=afterany:<jobid>) so the next segment starts automatically.
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output=outputs/hydrollm.%x.o%J
#SBATCH --error=errors/hydrollm.%x.e%J

set -euo pipefail

# Raise the open-file limit: DataLoader workers + a long Time-LLM run exhaust the default
# soft limit ("Too many open files"). Pairs with set_sharing_strategy('file_system') in the
# runner. Cap at the hard limit so we never exceed it.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true
echo "open-file limit: $(ulimit -n)"

RUNTAG="${RUNTAG:?set RUNTAG}"
DATASETS="${DATASETS:?set DATASETS}"
MODES="${MODES:?set MODES}"
SEEDS="${SEEDS:-2026}"
PHASE="${PHASE:-full}"   # full=Ray Tune HPO; dev=5ep no HPO (fast first baseline)
TRAIN_EPOCHS="${TRAIN_EPOCHS:-}"   # override the phase epoch cap; e.g. 30 for the paper baseline
LR="${LR:-}"                       # override learning rate (epoch diagnostic: 0.01 / 0.001)
PATIENCE="${PATIENCE:-}"           # override early-stop patience; >= epochs disables early stop
HPO_NUM_SAMPLES="${HPO_NUM_SAMPLES:-}"  # override Ray Tune trial count (phase-full default 50)
# EXPLICIT config (defense in depth): never rely on run_matrix's --config DEFAULT — a debug
# toggle leaking into the synced tree once flipped the default to the 64-sample debug.yaml
# and silently ran a whole Tier-0 with checkpoints disabled (job 11579994 cell 1, 2026-08-05).
CONFIG="${CONFIG:-experiments/hydro_llm/configs/timellm_config.yaml}"

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
EXTRA_ARGS=()
[ -n "$TRAIN_EPOCHS" ] && EXTRA_ARGS+=(--train-epochs "$TRAIN_EPOCHS")
[ -n "$LR" ] && EXTRA_ARGS+=(--learning-rate "$LR")
[ -n "$PATIENCE" ] && EXTRA_ARGS+=(--patience "$PATIENCE")
[ -n "$HPO_NUM_SAMPLES" ] && EXTRA_ARGS+=(--hpo-num-samples "$HPO_NUM_SAMPLES")

python experiments/hydro_llm/run_matrix.py \
  --config "$CONFIG" \
  --phase "$PHASE" \
  --run-tag "$RUNTAG" \
  --datasets $DATASETS \
  --modes $MODES \
  --seeds $SEEDS \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --resume

echo "=== done: $(date -Iseconds) ==="
