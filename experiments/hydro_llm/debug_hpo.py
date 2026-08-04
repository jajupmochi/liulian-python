#!/usr/bin/env python3
"""In-process HPO-trial debugger for the Time-LLM pipeline.

WHY THIS EXISTS
    The real HPO path runs each trial inside a Ray worker PROCESS, so PyCharm
    breakpoints set in the model / trainer / forward code do NOT hit when you run
    the Ray path (`python experiments/run.py --config .../debug.yaml`). On Ray 2.x
    `local_mode` only serialises trials, it does not move them into the driver.

    This script runs ONE HPO trial's core IN THE MAIN PROCESS, using the SAME
    building blocks a real trial uses:
      * the search space is resolved from search_spaces.yaml (resolve_search_space),
      * one hyper-parameter config is sampled from it and merged onto the base
        config — exactly what make_trainable's _trainable does (merge base+trial),
      * the model is built via the SAME liulian.pipeline.build_model (so every
        identity branch — none / embedding / coordinates / soft_prompt / ... — is
        exercised identically to a real run),
      * training runs through the SAME liulian.runtime.trainer.ForecastTrainer.fit
        loop the trial uses.
    So the model + trainer + forward internals are byte-for-byte the trial path;
    only Ray's per-epoch reporting wrapper is omitted (it is not model logic).

    To debug the Ray ORCHESTRATION itself (search-space build, ASHA, best-config
    selection, post-HPO rebuild+retrain) — which all run in the DRIVER process —
    use the Ray path instead and set breakpoints there:
        python experiments/run.py --config experiments/swiss_river/debug.yaml

USAGE
    python experiments/hydro_llm/debug_hpo.py
    python experiments/hydro_llm/debug_hpo.py --config experiments/swiss_river/debug.yaml
    python experiments/hydro_llm/debug_hpo.py --identifier-mode coordinates_embedding
    # In PyCharm: set this file as the Script, add HF_HUB_OFFLINE=1;TRANSFORMERS_OFFLINE=1
    # to the env, then Debug. Put breakpoints in timellm.forecast / trainer.fit / build_model.

SUGGESTED BREAKPOINTS
    liulian/pipeline.py            build_model()            — identity branch selection
    liulian/models/torch/timellm.py  Model.forecast()       — patch → reprogram → GPT-2 → head
    liulian/runtime/trainer.py     ForecastTrainer.fit()    — the training loop each trial runs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local GPT-2/BERT weights only; never hit the network in a debug session.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')


def _sample_trial_config(space: dict, seed: int) -> dict:
    """Draw ONE concrete hyper-parameter dict from a Ray Tune search space.

    Mirrors what Ray does per trial (sample each dimension), but deterministically
    from ``seed`` so a debug session is reproducible. Falls back to the sampler's
    own ``.sample()`` for any distribution type.
    """
    import random

    rng = random.Random(seed)
    out: dict = {}
    for name, sampler in space.items():
        cats = getattr(sampler, 'categories', None)
        if cats is not None:  # tune.choice
            out[name] = cats[rng.randrange(len(cats))]
        else:
            try:
                out[name] = sampler.sample()
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='experiments/swiss_river/debug.yaml',
                    help='debug config yaml (default: experiments/swiss_river/debug.yaml)')
    ap.add_argument('--identifier-mode', default=None,
                    help='override identifier_mode (e.g. none / embedding / coordinates_embedding / soft_prompt)')
    ap.add_argument('--arch', default=None, help='override model arch (timellm / gpt4ts / tempo / autotimes / calf)')
    ap.add_argument('--no-hpo-sample', action='store_true',
                    help='use the base config as-is (skip sampling from the search space)')
    args = ap.parse_args()

    from liulian.config import load_config
    from liulian.optim.search_spaces import resolve_search_space
    from liulian.pipeline import build_dataset, build_loaders, build_model, seed_everything
    from liulian.runtime.trainer import ForecastTrainer

    overrides: dict = {}
    if args.identifier_mode is not None:
        overrides['identifier_mode'] = args.identifier_mode
    if args.arch is not None:
        overrides['model'] = args.arch
    config = load_config(yaml_path=args.config, cli_overrides=overrides)

    # 1) Resolve the SAME search space HPO would use, and sample one trial config.
    if not args.no_hpo_sample and config.get('hpo', False):
        space = resolve_search_space(
            model=config.get('model', ''),
            data=config.get('data', ''),
            identifier_mode=config.get('identifier_mode', 'none'),
            id_integration=config.get('id_integration', 'concat_to_x'),
        )
        trial = _sample_trial_config(space, seed=int(config.get('seed', 2026)))
        print(f'[debug_hpo] sampled trial hypers: {trial}')
        config.update(trial)  # merge base + trial (what _trainable does)
    else:
        print('[debug_hpo] using base config (no HPO sampling)')

    seed_everything(int(config.get('seed', 2026)), deterministic=bool(config.get('deterministic', False)))

    # 2) Build the trial the SAME way the pipeline does (breakpoints in build_model).
    dataset = build_dataset(config)
    model = build_model(config, dataset)
    loaders = build_loaders(dataset, config)

    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ForecastTrainer(config=config, device=device)

    print(f'[debug_hpo] arch={config.get("model")} identifier_mode={config.get("identifier_mode")} '
          f'd_model={config.get("d_model")} d_ff={config.get("d_ff")} lr={config.get("learning_rate")} '
          f'llm_layers={config.get("llm_layers")}')

    # 3) Run the SAME ForecastTrainer.fit loop each HPO trial runs (breakpoints here).
    #    epochs come from config['train_epochs'] (read inside the trainer), matching a trial.
    summary = trainer.fit(
        model,
        loaders['train'],
        loaders['val'],
        test_loader=loaders.get('test'),
    )
    hist = summary.get('history') if isinstance(summary, dict) else None
    last = hist[-1] if isinstance(hist, list) and hist else summary
    print(f'[debug_hpo] DONE — last epoch record: {last}')


if __name__ == '__main__':
    main()
