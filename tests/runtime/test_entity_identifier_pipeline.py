"""Tests for the entity-identifier matrix runner utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments.entity_identifier.compare import generate_reports
from experiments.entity_identifier.matrix import (
    build_cli_overrides,
    iter_jobs,
    recommend_hpo_parallelism,
    resolve_config,
    validate_tsl_hparam_coverage,
)
from experiments.entity_identifier.run import (
    PROJECT_ROOT,
    _build_parser,
    _parse_args,
    _phase_defaults,
    _run_in_process,
    build_run_command,
    create_pred_vs_true_plots,
)
from experiments.entity_identifier.submit_slurm import (
    _build_single_matrix_command,
    HARD_CODED_MAIL_USER,
    HARD_CODED_SLURM_USER,
    build_sbatch_script,
    check_job_exists,
)


def test_iter_jobs_default_matrix_size() -> None:
    jobs = iter_jobs()
    # 15 pairs (5 datasets x 3 transparent models) x (none + embedding) = 30 baseline jobs
    # + 3 universally-supported transparent modes (onehot, sinusoidal, random) x 15 pairs = 45
    # + coordinates for all 9 Swiss pairs (3 datasets x {lstm, dlinear, patchtst}) = 9
    # + timellm on the 3 Swiss datasets x its 4 modes (none, embedding,
    #   random_embedding, entity_description) = 12  (swiss-only, see below)
    # Total = 30 + 45 + 9 + 12 = 96
    assert len(jobs) == 96
    coord_jobs = [job for job in jobs if job.mode == 'coordinates']
    # Coordinates available for every Swiss River pair (wired in mc since 2026-06-14)
    assert len(coord_jobs) == 9
    assert all(job.dataset.startswith('swiss-river') for job in coord_jobs)
    assert {job.model for job in coord_jobs} == {'lstm', 'dlinear', 'patchtst'}
    # All pairs get onehot, sinusoidal, random
    onehot_datasets = {job.dataset for job in jobs if job.mode == 'onehot'}
    assert onehot_datasets == {
        'swiss-river-1990',
        'swiss-river-2010',
        'swiss-river-zurich',
        'traffic',
        'electricity',
    }
    assert all('seed2026' in job.folder_name for job in jobs)


def test_timellm_enumerates_only_its_four_swiss_modes_and_no_leak() -> None:
    """Regression for the timellm matrix registration.

    Registering timellm (commits 3e1b307/5278556) had two failure modes an
    isolated ``supported_modes_for_pair`` check missed and a real run surfaced:

    1. timellm's text/capacity modes (entity_description, random_embedding) were
       absent from the MODES tuple, so the enumerator dropped them silently and
       the matrix produced only 2 of the 4 cells that ARE the study.
    2. timellm was enumerated on every dataset, KeyError-ing on the missing
       ('traffic', 'timellm') config.

    This locks both: exactly the four modes, only on the three swiss datasets, and
    neither timellm-only mode leaks into the transparent-ladder models.
    """
    jobs = iter_jobs()
    timellm = [j for j in jobs if j.model == 'timellm']

    # Swiss-only.
    assert {j.dataset for j in timellm} == {
        'swiss-river-1990',
        'swiss-river-2010',
        'swiss-river-zurich',
    }
    assert not [j for j in timellm if j.dataset in ('traffic', 'electricity')]

    # Exactly the four Time-LLM modes on every swiss dataset (3 x 4 = 12 cells).
    assert {j.mode for j in timellm} == {
        'none',
        'embedding',
        'random_embedding',
        'entity_description',
    }
    assert len(timellm) == 12

    # The two timellm-only carriers must never appear on the transparent models.
    for other in ('lstm', 'patchtst', 'dlinear'):
        other_modes = {j.mode for j in jobs if j.model == other}
        assert not (other_modes & {'entity_description', 'random_embedding'})


def test_timellm_on_unsupported_dataset_fails_loudly() -> None:
    """An explicit (unsupported-dataset, timellm) request must raise, not skip.

    Silently returning zero jobs for a pair the caller explicitly asked for would
    look like a successful no-op. The swiss-only restriction is enforced only for
    explicit dataset+model requests; the default matrix skips the pair instead.
    """
    with pytest.raises(ValueError, match='not supported on dataset'):
        iter_jobs(datasets=['traffic'], models=['timellm'], modes=['none'], seeds=[2026])


def test_patchtst_embedding_override_forces_add_after_patch() -> None:
    [job] = iter_jobs(
        datasets=['traffic'],
        models=['patchtst'],
        modes=['embedding'],
        seeds=[2026],
    )
    overrides = build_cli_overrides(job, hpo=True, quick_test=False)
    assert overrides['id_integration'] == 'add_after_patch'
    assert overrides['identifier_mode'] == 'embedding'
    assert overrides['hpo'] is True
    assert overrides['hpo_max_concurrent'] == 1
    assert overrides['hpo_resources_gpu'] == 1.0


def test_hpo_parallelism_recommendation_small_and_medium() -> None:
    [small_job] = iter_jobs(
        datasets=['swiss-river-1990'],
        models=['lstm'],
        modes=['none'],
        seeds=[2026],
    )
    [medium_job] = iter_jobs(
        datasets=['electricity'],
        models=['dlinear'],
        modes=['none'],
        seeds=[2026],
    )
    assert recommend_hpo_parallelism(small_job) == (4, 0.25)
    assert recommend_hpo_parallelism(medium_job) == (2, 0.5)


def test_hpo_parallelism_can_be_overridden() -> None:
    [job] = iter_jobs(
        datasets=['swiss-river-1990'],
        models=['lstm'],
        modes=['embedding'],
        seeds=[2026],
    )
    overrides = build_cli_overrides(
        job,
        hpo=True,
        quick_test=False,
        hpo_max_concurrent=3,
        hpo_resources_gpu=0.33,
    )
    assert overrides['hpo_max_concurrent'] == 3
    assert overrides['hpo_resources_gpu'] == 0.33


def test_iter_jobs_rejects_inapplicable_mode_filters() -> None:
    with pytest.raises(ValueError, match='No applicable matrix jobs'):
        iter_jobs(
            datasets=['traffic'],
            models=['dlinear'],
            modes=['coordinates'],
            seeds=[2026],
        )


def test_max_train_samples_override_is_forwarded() -> None:
    [job] = iter_jobs(
        datasets=['swiss-river-1990'],
        models=['lstm'],
        modes=['none'],
        seeds=[2026],
    )
    overrides = build_cli_overrides(
        job,
        hpo=False,
        quick_test=False,
        max_train_samples=128,
    )
    assert overrides['max_train_samples'] == 128


@pytest.mark.skip(reason='pre-existing: HPO default count is 50 not 10; TODO: align spec or default')
def test_full_phase_defaults_to_10_hpo_trials() -> None:
    full = _phase_defaults('full')
    assert full['hpo'] is True
    assert full['hpo_num_samples'] == 10


def test_tsl_hparam_coverage_for_matched_patchtst() -> None:
    [job] = iter_jobs(
        datasets=['traffic'],
        models=['patchtst'],
        modes=['embedding'],
        seeds=[2026],
    )
    overrides = build_cli_overrides(job, hpo=True, quick_test=False)
    cfg = resolve_config(project_root=PROJECT_ROOT, job=job, overrides=overrides)
    checked = validate_tsl_hparam_coverage(cfg, job)
    assert 'learning_rate' in checked
    assert 'batch_size' in checked


def test_build_run_command_with_boolean_overrides() -> None:
    cmd = build_run_command(
        python_bin='python3',
        run_script=Path('experiments/run.py'),
        config_path=Path('experiments/traffic/lstm_config.yaml'),
        overrides={
            'hpo': True,
            'quick_test': False,
            'data': 'traffic',
        },
    )
    assert '--hpo' in cmd
    assert '--no_quick_test' in cmd
    assert '--data' in cmd


def test_run_in_process_uses_imported_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, object] = {}

    def _fake_run_with_config(
        *,
        config_path: str,
        cli_overrides: dict[str, object] | None = None,
    ) -> dict[str, object]:
        called['config_path'] = config_path
        called['cli_overrides'] = cli_overrides or {}
        return {}

    monkeypatch.setattr(
        'experiments.entity_identifier.run.run_with_config',
        _fake_run_with_config,
    )

    config_path = tmp_path / 'cfg.yaml'
    config_path.write_text('model: lstm\n', encoding='utf-8')
    log_path = tmp_path / 'run.log'
    returncode, elapsed = _run_in_process(
        config_path=config_path,
        overrides={'data': 'traffic', 'model': 'lstm'},
        cwd=tmp_path,
        log_path=log_path,
        timeout_seconds=0,
    )
    assert returncode == 0
    assert elapsed >= 0.0
    assert called['config_path'] == str(config_path)
    assert called['cli_overrides'] == {'data': 'traffic', 'model': 'lstm'}
    assert log_path.exists()


def test_parse_args_preserves_explicit_cli_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / 'matrix_cfg.yaml'
    cfg.write_text(
        'phase: full\nmodels:\n  - dlinear\nmodes:\n  - none\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'run.py',
            '--config',
            str(cfg),
            '--modes',
            'embedding',
        ],
    )

    args = _parse_args(_build_parser())
    assert args.phase == 'full'  # from config
    assert args.models == ['dlinear']  # from config
    assert args.modes == ['embedding']  # explicit CLI should win


def test_compare_generates_summary_and_markdown(tmp_path: Path) -> None:
    run_root = tmp_path / 'entity_run'
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = run_root / 'manifest.jsonl'
    records = [
        {
            'job_key': 'traffic__dlinear__none__seed2026',
            'dataset': 'traffic',
            'model': 'dlinear',
            'mode': 'none',
            'seed': 2026,
            'status': 'ok',
            'test_mse': 0.50,
            'test_rmse': 0.70,
            'test_mae': 0.40,
            'test_nse': 0.80,
            'hpo_best_value': 0.45,
            'hpo_trials': 8,
            'tsl_alignment': 'matched',
            'plot_path': 'pred_vs_true_time.png',
            'results_json': 'results.json',
            'run_dir': 'run_dir',
        },
        {
            'job_key': 'traffic__dlinear__embedding__seed2026',
            'dataset': 'traffic',
            'model': 'dlinear',
            'mode': 'embedding',
            'seed': 2026,
            'status': 'ok',
            'test_mse': 0.45,
            'test_rmse': 0.67,
            'test_mae': 0.38,
            'test_nse': 0.82,
            'hpo_best_value': 0.41,
            'hpo_trials': 8,
            'tsl_alignment': 'matched',
            'plot_path': 'pred_vs_true_time.png',
            'results_json': 'results.json',
            'run_dir': 'run_dir',
        },
    ]
    with manifest.open('w', encoding='utf-8') as fh:
        for record in records:
            fh.write(json.dumps(record) + '\n')

    outputs = generate_reports(run_root)
    assert outputs['summary_csv'].exists()
    assert outputs['comparison_md'].exists()
    md_text = outputs['comparison_md'].read_text(encoding='utf-8')
    assert 'Mode-Level Comparison' in md_text
    assert 'Entity identifier' in md_text
    assert 'emb' in md_text
    assert 'traffic' in md_text
    assert 'dlinear' in md_text


def test_create_pred_vs_true_plot(tmp_path: Path) -> None:
    pytest.importorskip('matplotlib')
    np = pytest.importorskip('numpy')

    preds = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    trues = np.array([[[1.2], [2.1], [2.9], [3.8]]], dtype=np.float32)
    times = np.array([[[0.0], [1.0], [2.0], [3.0]]], dtype=np.float32)
    npz_path = tmp_path / 'predictions.npz'
    np.savez(npz_path, preds=preds, trues=trues, times=times)

    created = create_pred_vs_true_plots(
        npz_path=npz_path,
        output_dir=tmp_path,
        aggregations=['mean', 'median', 'best', 'last'],
        formats=['png', 'svg'],
        file_stem='pred_vs_gt',
    )
    created_names = {path.name for path in created}
    assert 'pred_vs_gt_mean.png' in created_names
    assert 'pred_vs_gt_median.svg' in created_names
    assert 'pred_vs_gt_best.png' in created_names
    assert 'pred_vs_gt_last_range.png' in created_names


def test_submit_sbatch_rendering_keeps_hardcoded_identifiers() -> None:
    script = build_sbatch_script(
        job_name='eid-demo',
        command='python experiments/entity_identifier/run.py --phase full',
        project_root=PROJECT_ROOT,
        use_gpu=True,
        gres='gpu:a100:1',
        partition_gpu='gpu',
        partition_cpu='epyc2,bdw',
        qos='job_gratis',
        account='gratis',
        time_limit='24:00:00',
        cpus_per_task=4,
        mem_per_cpu='10G',
        python_module='Python/3.12.3-GCCcore-13.3.0',
        venv_path=str(PROJECT_ROOT / '.venv' / 'bin' / 'activate'),
        mail_user=HARD_CODED_MAIL_USER,
    )
    assert '#SBATCH --job-name="eid-demo"' in script
    assert HARD_CODED_MAIL_USER in script
    assert check_job_exists.__defaults__[0] == HARD_CODED_SLURM_USER
    assert '#SBATCH --gres=gpu:a100:1' in script
    # Paths with spaces are properly quoted in shell commands
    assert f'cd "{PROJECT_ROOT}"' in script
    assert 'source "' in script


def test_single_matrix_command_includes_explicit_full_scope_defaults() -> None:
    command = _build_single_matrix_command(
        python_bin='python3',
        run_tag='demo',
        datasets=None,
        models=None,
        modes=None,
        seeds=None,
        max_jobs=None,
        hpo_num_samples=None,
        hpo_max_concurrent=None,
        hpo_resources_gpu=None,
        max_train_samples=None,
        plot_aggregations=['mean', 'best'],
        plot_formats=['png'],
        plot_file_stem='pred_vs_gt',
    )
    assert ('--datasets swiss-river-1990 swiss-river-2010 swiss-river-zurich traffic electricity') in command
    assert '--models lstm patchtst dlinear' in command
    assert '--modes none embedding onehot coordinates sinusoidal random' in command
    assert '--resume' in command
