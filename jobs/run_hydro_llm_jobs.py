"""
run_hydro_llm_jobs — Python launcher for the hydro-LLM matrix on UBELIX.

Same style as jobs/run_jobs_ray_tune.py (the project's reference launcher): the
batch script is generated as a heredoc string and submitted with ``sbatch <<EOF``,
so everything — header, phases, dependency chains — lives in ONE Python file you
run on the login node. The plain-bash alternative (jobs/run_hydro_llm.sh submitted
manually with env vars) stays available; both produce identical jobs.

Usage (on the cluster login node, from the repo root):

    # submit the standard two-arm Tier-0 v2 plan (phases none -> +numeric_embedding
    # -> +text/soft/text_embedding, each phase as afterany-chained 24h segments):
    python3 jobs/run_hydro_llm_jobs.py

    # one arm only, custom phases/segments:
    python3 jobs/run_hydro_llm_jobs.py --arms promptfix --segments 2 \
        --phases "none" "none numeric_embedding"

    # dry run: print the scripts instead of submitting
    python3 jobs/run_hydro_llm_jobs.py --dry

Free-tier guardrails are hardcoded in the header (account=gratis, qos=job_gratis,
1x rtx4090, 24h — the gpu partition's HARD wall, measured 2026-08-07). Long phases
survive the wall through --resume + the afterany chain this launcher builds.
"""

import argparse
import os
import re
import subprocess

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CUR_PATH)

PREFIX_KW = 'hydrollm'

ARMS = {
    'promptfix': {
        'run_tag': 'hydro-t0v2-promptfix',
        'config': 'experiments/hydro_llm/configs/timellm_config.yaml',
    },
    'ettctrl': {
        'run_tag': 'hydro-t0v2-ettctrl',
        'config': 'experiments/hydro_llm/configs/tier0_ettcontrol.yaml',
    },
}

#: Phase order per the study design (2026-08-10): the injection-scheme comparison
#: runs FROZEN-backbone first; each later phase repeats the earlier modes so
#: --resume skips completed cells and only the new modes execute.
DEFAULT_PHASES = [
    'none',
    'none numeric_embedding',
    'none numeric_embedding entity_description soft_prompt text_embedding',
]

DATASETS = 'swiss-river-1990 swiss-river-2010 swiss-river-zurich'
SEEDS = '2026'


def get_job_script(job_name: str, run_tag: str, config: str, modes: str) -> str:
    """Build the full batch script (mirrors jobs/run_hydro_llm.sh, kept in sync)."""
    script = r"""#!/bin/bash
#SBATCH --job-name=""" + job_name + r"""
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jajupmochi@gmail.com
#SBATCH --output=outputs/""" + PREFIX_KW + r""".%x.o%J
#SBATCH --error=errors/""" + PREFIX_KW + r""".%x.e%J

set -euo pipefail
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true

module load Python/3.12.3-GCCcore-13.3.0
source "$HOME/codes/liulian-python/.venv/bin/activate"
cd "$HOME/codes/liulian-python"
mkdir -p outputs errors
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

python experiments/hydro_llm/run_matrix.py \
  --config """ + config + r""" \
  --phase full \
  --run-tag """ + run_tag + r""" \
  --datasets """ + DATASETS + r""" \
  --modes """ + modes + r""" \
  --seeds """ + SEEDS + r""" \
  --resume
"""
    script = re.sub('\n\t+', '\n', script)
    return script.strip() + '\n'


def submit(script: str, dependency: str | None, dry: bool) -> str:
    """sbatch the heredoc script; return the job id ('' on --dry)."""
    dep = f'--dependency=afterany:{dependency} ' if dependency else ''
    command = 'sbatch --parsable ' + dep + '<<EOF\n' + script + 'EOF'
    if dry:
        print(command)
        print('-' * 60)
        return ''
    out = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=REPO)
    job_id = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ''
    if not job_id.isdigit():
        raise RuntimeError(f'sbatch failed: stdout={out.stdout!r} stderr={out.stderr!r}')
    return job_id


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    p.add_argument('--phases', nargs='+', default=DEFAULT_PHASES,
                   help='mode lists, one string per phase (later phases should repeat earlier modes)')
    p.add_argument('--segments', type=int, default=4,
                   help='24h segments chained per phase (afterany; finished phases exit fast)')
    p.add_argument('--dry', action='store_true', help='print scripts, submit nothing')
    args = p.parse_args()

    for arm in args.arms:
        cfg = ARMS[arm]
        prev = None
        chain: list[str] = []
        for phase_i, modes in enumerate(args.phases):
            job_name = f'{PREFIX_KW}-{arm}-p{phase_i}'
            script = get_job_script(job_name, cfg['run_tag'], cfg['config'], modes)
            for _seg in range(args.segments):
                prev = submit(script, prev, args.dry) or prev
                if prev:
                    chain.append(prev)
        print(f'{arm} ({cfg["run_tag"]}): ' + (' '.join(chain) if chain else '(dry run)'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
