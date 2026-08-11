"""Tests for the hydro-LLM matrix runner's cell enumeration guardrails.

The Time-LLM matrix sweeps modes x datasets. entity_description needs authored
per-station text; only swiss-river-1990 (and ETTh1) has it today. build_cells
must SKIP (dataset, entity_description) for datasets without descriptions rather
than schedule a cell that only raises at run time — otherwise a 3-dataset sweep
looks like it will produce 9 results when 2 of them are guaranteed failures.
"""

from __future__ import annotations

from types import SimpleNamespace


import pytest

from experiments.hydro_llm.run_matrix import build_cells, build_overrides
from liulian.pipeline import _load_entity_descriptions, has_entity_descriptions


def _args(**over):
    base = dict(
        datasets=['swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich'],
        modes=['none', 'entity_description', 'numeric_embedding'],
        a2=['learnable'],
        a1=['default'],
        tuning=['frozen'],
        backbones=['GPT2'],
        seeds=[2026],
        arch='timellm',
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestEntityDescriptionAvailability:
    def test_all_swiss_datasets_and_etth1_have_descriptions(self) -> None:
        # SEMANTICS CHANGED 2026-08-11: 2010 (63 BAFU stations, river/town from
        # the hydrodaten directory) and zurich (15 cantonal stations,
        # coordinate-descriptive text) were authored; before that only 1990 and
        # ETTh1 had text and this test asserted the others False.
        assert has_entity_descriptions('swiss-river-1990') is True
        assert has_entity_descriptions('ETTh1') is True
        assert has_entity_descriptions('swiss-river-2010') is True
        assert has_entity_descriptions('swiss-river-zurich') is True
        assert has_entity_descriptions('does-not-exist') is False


class TestBuildCellsGuardrail:
    def test_entity_description_runs_everywhere_now(self) -> None:
        # All three swiss datasets have authored descriptions since 2026-08-11,
        # so nothing skips with the REAL data files; the skip MECHANISM itself
        # is covered by TestTextEmbeddingSkipGuard via monkeypatch.
        cells = build_cells(_args())
        ed = [c for c in cells if c['mode'] == 'entity_description']
        assert {c['dataset'] for c in ed} == {
            'swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich'}

    def test_other_modes_run_on_all_datasets(self) -> None:
        cells = build_cells(_args())
        for mode in ('none', 'numeric_embedding'):
            got = {c['dataset'] for c in cells if c['mode'] == mode}
            assert got == {'swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich'}

    def test_total_cell_count_is_nine_since_descriptions_authored(self) -> None:
        # 3 datasets x (none + entity_description + numeric_embedding) = 9.
        # Was 7 while 2010/zurich lacked station text (their entity_description
        # cells skipped); descriptions authored 2026-08-11.
        assert len(build_cells(_args())) == 9

    def test_no_skip_when_only_1990_requested(self) -> None:
        cells = build_cells(_args(datasets=['swiss-river-1990']))
        assert {c['mode'] for c in cells} == {'none', 'entity_description', 'numeric_embedding'}
        assert len(cells) == 3


class TestPromptRichnessA1:
    """Level-A1 prompt richness for entity_description.

    default = authored rich station text; minimal = a bare positional
    identifier. The ablation asks whether richer text helps beyond a distinct
    id, so the two MUST differ and minimal must still be distinct per station.
    """

    def test_minimal_is_distinct_positional_text(self):
        out = _load_entity_descriptions({'data': 'swiss-river-1990', 'num_entities': 28, 'prompt_richness': 'minimal'})
        assert len(out) == 28
        assert out[0] != out[1]  # distinct per station
        assert 'station number 1' in out[0]

    def test_default_and_minimal_differ(self):
        common = {'data': 'swiss-river-1990', 'num_entities': 28}
        mn = _load_entity_descriptions({**common, 'prompt_richness': 'minimal'})
        df = _load_entity_descriptions({**common, 'prompt_richness': 'default'})
        assert mn != df

    def test_minimal_requires_num_entities(self):
        with pytest.raises(ValueError, match='num_entities'):
            _load_entity_descriptions({'data': 'swiss-river-1990', 'prompt_richness': 'minimal'})

    def test_unknown_richness_raises(self):
        with pytest.raises(ValueError, match='not implemented'):
            _load_entity_descriptions({'data': 'swiss-river-1990', 'num_entities': 28, 'prompt_richness': 'bogus'})

    def test_build_overrides_sets_prompt_richness_for_entity_description(self):
        from types import SimpleNamespace

        args = SimpleNamespace(
            phase='dry',
            hpo_num_samples=None,
            train_epochs=None,
            learning_rate=None,
            patience=None,
            max_train_samples=None,
        )
        cells = build_cells(
            _args(datasets=['swiss-river-1990'], modes=['entity_description'], a1=['default', 'minimal'])
        )
        by_sub = {c['sub']: build_overrides(c, args) for c in cells}
        assert by_sub['minimal']['prompt_richness'] == 'minimal'
        assert by_sub['default']['prompt_richness'] == 'default'

    def test_build_overrides_omits_prompt_richness_for_non_text_modes(self):
        from types import SimpleNamespace

        args = SimpleNamespace(
            phase='dry',
            hpo_num_samples=None,
            train_epochs=None,
            learning_rate=None,
            patience=None,
            max_train_samples=None,
        )
        cell = build_cells(_args(datasets=['swiss-river-1990'], modes=['none']))[0]
        assert 'prompt_richness' not in build_overrides(cell, args)


class TestPromptRichnessStats:
    """Level-A1 `stats` richness: positional id + per-station TRAIN-only temperature stats."""

    def test_stats_desc_formats_from_station_stats(self):
        from liulian.pipeline import _load_entity_descriptions

        stats = [{'mean': 0.4, 'std': 0.2, 'min': 0.0, 'max': 1.0}, {'mean': 0.6, 'std': 0.1, 'min': 0.1, 'max': 0.9}]
        out = _load_entity_descriptions({'prompt_richness': 'stats', 'station_stats': stats})
        assert len(out) == 2
        assert out[0] != out[1]  # distinct per station
        assert 'mean 0.40' in out[0] and 'mean 0.60' in out[1]

    def test_stats_requires_station_stats(self):
        from liulian.pipeline import _load_entity_descriptions

        with pytest.raises(ValueError, match='station_stats'):
            _load_entity_descriptions({'prompt_richness': 'stats'})

    def test_compute_station_train_stats_is_leakage_safe(self):
        # Stats must come from the TRAIN frame ONLY, never val/test.
        import numpy as np

        from liulian.config import load_config
        from liulian.pipeline import _compute_station_train_stats, build_dataset

        cfg = load_config(
            'experiments/hydro_llm/configs/timellm_config.yaml',
            cli_overrides={
                'data': 'swiss-river-1990',
                'identifier_mode': 'entity_description',
                'split_mode': 'per_entity',
            },
        )
        ds = build_dataset(cfg)
        st = _compute_station_train_stats(ds)
        assert st is not None and len(st) == len(ds.station_ids)
        # the train-only mean must differ from the all-splits mean (proves train-only, not全体)
        frames = ds._split_frames
        s0 = ds.station_ids[0]
        all_vals = np.concatenate([frames[k][f'{s0}_wt'].to_numpy(dtype=float) for k in frames])
        all_mean = float(np.nanmean(all_vals))
        assert abs(st[0]['mean'] - all_mean) > 1e-9  # train-only != all-splits

    def test_stats_in_implemented_a1(self):
        from experiments.hydro_llm.run_matrix import IMPLEMENTED_A1

        assert 'stats' in IMPLEMENTED_A1


class TestCoordinatesA2:
    """A2 coordinates for timellm: real per-station CH1903 coords from the graph .pth."""

    def test_coordinates_feature_is_real_and_distinct(self):

        from liulian.config import load_config
        from liulian.pipeline import build_dataset, build_model

        cfg = load_config(
            'experiments/hydro_llm/configs/timellm_config.yaml',
            cli_overrides={
                'data': 'swiss-river-1990',
                'identifier_mode': 'coordinates_embedding',
                'split_mode': 'per_entity',
                'llm_layers': 1,
            },
        )
        ds = build_dataset(cfg)
        # topology (with coords) must load for the coordinates_embedding mode
        assert ds.topology is not None and len(ds.topology.coordinates) == len(ds.station_ids)
        m = build_model(cfg, ds)
        feat = m.transparent_feat
        assert tuple(feat.shape) == (len(ds.station_ids), 2)
        assert not bool((feat == 0).all().item())  # not fake zeros
        assert len({tuple(r) for r in feat.tolist()}) == len(ds.station_ids)  # distinct per station

    def test_coordinates_in_implemented_a2(self):
        from experiments.hydro_llm.run_matrix import IMPLEMENTED_A2, _identifier_mode_for

        assert 'coordinates' in IMPLEMENTED_A2
        assert _identifier_mode_for('numeric_embedding', 'coordinates') == 'coordinates_embedding'


class TestConfigOverridesDefaults:
    """--config file params auto-override phase/argparse defaults; explicit CLI wins.

    Precedence: explicit CLI flag > --config file value > phase default.
    """

    def _cfg(self, tmp_path, **keys):
        import yaml

        p = tmp_path / 'dbg.yaml'
        p.write_text(yaml.safe_dump(keys), encoding='utf-8')
        return str(p)

    def test_config_hpo_survives_non_full_phase(self, tmp_path):
        # config sets hpo:true; phase 'dev' would default hpo off -> config must win
        from experiments.hydro_llm.run_matrix import build_overrides

        cfg = self._cfg(tmp_path, hpo=True, hpo_num_samples=3, train_epochs=2)
        cell = dict(
            arch='timellm',
            dataset='swiss-river-1990',
            mode='none',
            sub='default',
            tuning='frozen',
            backbone='GPT2',
            seed=2026,
            identifier_mode='none',
            job_key='k',
        )
        args = SimpleNamespace(
            phase='dev',
            config=cfg,
            hpo_num_samples=None,
            train_epochs=None,
            learning_rate=None,
            patience=None,
            max_train_samples=None,
        )
        ov = build_overrides(cell, args)
        # config keys are NOT injected as overrides (so the yaml base provides them, unclobbered)
        assert 'hpo' not in ov and 'train_epochs' not in ov and 'hpo_num_samples' not in ov

    def test_phase_default_applies_when_config_silent(self, tmp_path):
        from experiments.hydro_llm.run_matrix import build_overrides

        cfg = self._cfg(tmp_path, model='timellm')  # no hpo/train_epochs keys
        cell = dict(
            arch='timellm',
            dataset='swiss-river-1990',
            mode='none',
            sub='default',
            tuning='frozen',
            backbone='GPT2',
            seed=2026,
            identifier_mode='none',
            job_key='k',
        )
        args = SimpleNamespace(
            phase='full',
            config=cfg,
            hpo_num_samples=None,
            train_epochs=None,
            learning_rate=None,
            patience=None,
            max_train_samples=None,
        )
        ov = build_overrides(cell, args)
        assert ov.get('hpo') is True and ov.get('hpo_num_samples') == 50  # phase-full cap

    def test_explicit_cli_beats_config(self, tmp_path):
        from experiments.hydro_llm.run_matrix import build_overrides

        cfg = self._cfg(tmp_path, train_epochs=2)
        cell = dict(
            arch='timellm',
            dataset='swiss-river-1990',
            mode='none',
            sub='default',
            tuning='frozen',
            backbone='GPT2',
            seed=2026,
            identifier_mode='none',
            job_key='k',
        )
        args = SimpleNamespace(
            phase='dev',
            config=cfg,
            hpo_num_samples=None,
            train_epochs=17,
            learning_rate=None,
            patience=None,
            max_train_samples=None,
        )
        assert build_overrides(cell, args)['train_epochs'] == 17

    def _axis_args(self, cfg, **over):
        base = dict(
            config=cfg,
            phase=None,
            run_tag=None,
            timeout_seconds=None,
            resume=False,
            arch=None,
            datasets=None,
            modes=None,
            a2=None,
            a1=None,
            tuning=None,
            backbones=None,
            seeds=None,
        )
        base.update(over)
        return SimpleNamespace(**base)

    def test_apply_config_defaults_fills_omitted_axes(self, tmp_path):
        from experiments.hydro_llm.run_matrix import _apply_config_defaults

        cfg = self._cfg(
            tmp_path,
            model='timellm',
            data='swiss-river-2010',
            llm_model='GPT2',
            llm_tuning='lora',
            seed=7,
            identifier_mode='random_embedding',
        )
        args = self._axis_args(cfg)
        _apply_config_defaults(args)
        assert args.arch == 'timellm' and args.datasets == ['swiss-river-2010']
        assert args.tuning == ['lora'] and args.seeds == [7]
        # identifier_mode random_embedding reverse-maps to numeric_embedding + random
        assert args.modes == ['numeric_embedding'] and args.a2 == ['random']

    def test_explicit_axis_flag_not_overwritten_by_config(self, tmp_path):
        from experiments.hydro_llm.run_matrix import _apply_config_defaults

        cfg = self._cfg(tmp_path, llm_tuning='lora')
        args = self._axis_args(
            cfg,
            arch='gpt4ts',
            datasets=['swiss-river-1990'],
            modes=['none'],
            a2=['learnable'],
            a1=['default'],
            tuning=['frozen'],
            backbones=['GPT2'],
            seeds=[2026],
        )
        _apply_config_defaults(args)
        assert args.tuning == ['frozen'] and args.arch == 'gpt4ts'  # explicit wins

    def test_config_phase_drives_run(self, tmp_path):
        # config sets phase:full -> _apply_config_defaults fills it (so `run_matrix --config`
        # with no --phase actually RUNS instead of hitting the dry-run guard).
        from experiments.hydro_llm.run_matrix import _apply_config_defaults

        args = self._axis_args(self._cfg(tmp_path, phase='full', model='timellm'))
        _apply_config_defaults(args)
        assert args.phase == 'full'

    def test_config_phase_defaults_to_dry_when_silent(self, tmp_path):
        from experiments.hydro_llm.run_matrix import _apply_config_defaults

        args = self._axis_args(self._cfg(tmp_path, model='timellm'))
        _apply_config_defaults(args)
        assert args.phase == 'dry'

    def test_explicit_phase_beats_config(self, tmp_path):
        from experiments.hydro_llm.run_matrix import _apply_config_defaults

        args = self._axis_args(self._cfg(tmp_path, phase='full'), phase='smoke')
        _apply_config_defaults(args)
        assert args.phase == 'smoke'  # explicit --phase wins

    def test_invalid_config_phase_raises(self, tmp_path):
        from experiments.hydro_llm.run_matrix import _apply_config_defaults

        args = self._axis_args(self._cfg(tmp_path, phase='bogus'))
        with pytest.raises(SystemExit):
            _apply_config_defaults(args)


class TestDebugDefaultIsClusterSafe:
    """DEBUGGING must be module-scope (build_parser importable) and default OFF so a real
    `python run_matrix.py` (incl. the cluster's sbatch) does NOT default to the 64-sample
    debug.yaml. It was a NameError + a silent debug-config-in-production risk before the fix.
    """

    def test_build_parser_importable_and_defaults_to_base_config(self):
        import experiments.hydro_llm.run_matrix as m
        # DEBUGGING is defined at module scope (no NameError when build_parser runs on import)
        assert hasattr(m, 'DEBUGGING')
        args = m.build_parser().parse_args([])  # would NameError before the fix
        # without HYDRO_DEBUG, the default config is the aligned BASE_CONFIG, not debug.yaml
        if not m.DEBUGGING:
            assert args.config == str(m.BASE_CONFIG)
            assert 'debug' not in args.config


class TestPromptContentIsReal:
    """The swiss prompt_bank content must be the authored water-temperature text.

    Regression for two bugs found 2026-08-04: (a) wt-swiss-1990.txt was an AI placeholder
    ("This is just a sample text file..."); (b) prompt_domain: 0 made the model use
    Time-LLM's hardcoded ETT (electricity) description for river data.
    """

    def test_prompt_bank_content_is_authored_not_placeholder(self):
        from liulian.pipeline import _load_prompt_content

        for d in ('swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich'):
            c = _load_prompt_content({'data': d})
            assert 'sample text file' not in c  # the placeholder
            assert 'water temperature' in c     # the real domain
            assert len(c.split()) <= 110        # <=~100-token principle

    def test_config_enables_prompt_domain(self):
        from liulian.config import load_config

        cfg = load_config('experiments/hydro_llm/configs/timellm_config.yaml')
        assert cfg['prompt_domain'] == 1  # 0 = hardcoded ETT text (the bug)


class TestPromptVariantKnobs:
    """Generalized Level-A1 prompt-content knobs (prompt_variant x prompt_stats).

    Regression anchors: (a) full stats stays byte-identical to the upstream Time-LLM
    template; (b) the trial-rebuild path (SimpleNamespace(**config)) constructs with
    prompt_domain=1 — it crashed before content was written back into the config dict.
    """

    def test_variant_selects_bank_file(self):
        from liulian.pipeline import _load_prompt_content

        base = {'data': 'swiss-river-1990'}
        assert 'snowmelt' in _load_prompt_content({**base, 'prompt_variant': 'domain'})
        assert 'one day sample rate' in _load_prompt_content({**base, 'prompt_variant': 'minimal'})
        assert _load_prompt_content({**base, 'prompt_variant': 'none'}) == ''

    def test_missing_variant_file_raises(self):
        from liulian.pipeline import _load_prompt_content

        with pytest.raises(FileNotFoundError):
            _load_prompt_content({'data': 'swiss-river-2010', 'prompt_variant': 'minimal'})

    def test_full_stats_is_upstream_verbatim(self):
        from liulian.models.torch.timellm import Model

        p = Model._compose_prompt('D. ', None, '7', '90', '0.1', '0.9', '0.5', True, '[1, 2]', stats_mode='full')
        assert p == (
            '<|start_prompt|>Dataset description: D. '
            'Task description: forecast the next 7 steps given the previous 90 steps information; '
            'Input statistics: min value 0.1, max value 0.9, median value 0.5, '
            'the trend of input is upward, top 5 lags are : [1, 2]<|<end_prompt>|>'
        )

    def test_basic_drops_lags_none_drops_stats(self):
        from liulian.models.torch.timellm import Model

        basic = Model._compose_prompt('D. ', None, '7', '90', '0.1', '0.9', '0.5', True, '[1]', stats_mode='basic')
        none_ = Model._compose_prompt('D. ', None, '7', '90', '0.1', '0.9', '0.5', True, '[1]', stats_mode='none')
        assert 'top 5 lags' not in basic and 'min value' in basic
        assert 'Input statistics' not in none_ and 'Task description' in none_

    def test_content_written_back_into_config(self):
        # the HPO trial rebuilds the model from SimpleNamespace(**config); without content
        # in the dict, prompt_domain=1 crashed every trial (masked as "unexpected kwarg").
        from liulian.config import load_config
        from liulian.pipeline import build_dataset, build_model

        cfg = load_config(
            'experiments/hydro_llm/configs/timellm_config.yaml',
            cli_overrides={'data': 'swiss-river-1990', 'identifier_mode': 'none',
                           'split_mode': 'per_entity', 'llm_layers': 1},
        )
        ds = build_dataset(cfg)
        build_model(cfg, ds)
        assert 'content' in cfg and 'water temperature' in cfg['content']


class TestDistinguisherVsContentArms:
    """A1 `symbol` (text onehot: distinct, zero semantics) and `shuffled` (real rich text
    deranged between stations: distinct, WRONG content). Together with `minimal` and
    `default` they separate "the prompt works as a distinguisher" from "the factual
    content matters". Both are fixed-seed deterministic so all model seeds share them.
    """

    BASE = {'data': 'swiss-river-1990', 'num_entities': 28}

    def test_symbol_distinct_no_digits_deterministic(self):
        from liulian.pipeline import _load_entity_descriptions as L

        sym = L({**self.BASE, 'prompt_richness': 'symbol'})
        assert len(sym) == 28 and len(set(sym)) == 28          # distinct per station
        assert not any(ch.isdigit() for s in sym for ch in s)  # zero ordinal leakage
        assert sym == L({**self.BASE, 'prompt_richness': 'symbol'})  # deterministic

    def test_shuffled_is_derangement_of_default(self):
        from liulian.pipeline import _load_entity_descriptions as L

        shf = L({**self.BASE, 'prompt_richness': 'shuffled'})
        dflt = L({**self.BASE, 'prompt_richness': 'default'})
        assert sorted(shf) == sorted(dflt)                      # same content set (rich, real)
        assert all(a != b for a, b in zip(shf, dflt))           # no station keeps its true text
        assert shf == L({**self.BASE, 'prompt_richness': 'shuffled'})  # deterministic

    def test_symbol_requires_num_entities(self):
        from liulian.pipeline import _load_entity_descriptions as L

        with pytest.raises(ValueError, match='num_entities'):
            L({'data': 'swiss-river-1990', 'prompt_richness': 'symbol'})

    def test_arms_registered_in_a1(self):
        from experiments.hydro_llm.run_matrix import IMPLEMENTED_A1

        assert 'symbol' in IMPLEMENTED_A1 and 'shuffled' in IMPLEMENTED_A1


class TestIdentityModeConsistency:
    """The timellm identity-mode set exists in THREE places that must agree:

    1. pipeline._TIMELLM_IDENTITY_MODES (build_model: surfaces num_entities etc.)
    2. trainer pass_entity_ids (a mode missing there falls back to raw station numbers
       in x_mark and CUDA-asserts — the coordinates_embedding bug of 2026-08-04)
    3. run_matrix _A2_TO_IDENTIFIER values + prompt-family passthrough modes

    This test locks the three together so a future mode cannot drift.
    """

    EXPECTED = {
        'embedding', 'random_embedding', 'soft_prompt', 'entity_description',
        'text_embedding', 'onehot_embedding', 'sinusoidal_embedding', 'coordinates_embedding',
    }

    @staticmethod
    def _pipeline_set():
        import re

        import liulian.pipeline as pl

        src = open(pl.__file__).read()
        m = re.search(r"_TIMELLM_IDENTITY_MODES = frozenset\(\s*\{([^}]*)\}", src)
        return set(re.findall(r"'([a-z_]+)'", m.group(1)))

    @staticmethod
    def _trainer_set():
        import re

        import liulian.runtime.trainer as tr

        src = open(tr.__file__).read()
        m = re.search(r"or _idmode\s*\n\s*in \{([^}]*)\}", src)
        # 'embedding' reaches the model via use_entity_embedding, not this literal set
        return set(re.findall(r"'([a-z_]+)'", m.group(1))) | {'embedding'}

    @staticmethod
    def _matrix_set():
        from experiments.hydro_llm.run_matrix import _A2_TO_IDENTIFIER, IMPLEMENTED_MODES

        return set(_A2_TO_IDENTIFIER.values()) | {
            m for m in IMPLEMENTED_MODES if m not in ('none', 'numeric_embedding')
        }

    def test_three_sets_agree(self):
        p, t, x = self._pipeline_set(), self._trainer_set(), self._matrix_set()
        assert p == self.EXPECTED, f'pipeline drifted: {p ^ self.EXPECTED}'
        assert t == self.EXPECTED, f'trainer drifted: {t ^ self.EXPECTED}'
        assert x == self.EXPECTED, f'run_matrix drifted: {x ^ self.EXPECTED}'


class TestExplicitHpoBlockInConfigs:
    """The hydro-LLM configs pin the HPO EXECUTION settings explicitly (2026-08-07,
    project rule 3) instead of inheriting them invisibly from DEFAULT_CONFIG in
    liulian/config.py. Locks three contracts:
      (a) both configs carry the hpo_* keys with the Tier-0-verified values and do
          NOT set the `hpo` on/off switch (that stays phase-owned);
      (b) the config's hpo_num_samples (24) suppresses phase-full's 50 injection;
      (c) hpo_save_checkpoints stays true — false broke the post-HPO retrain
          ("Best checkpoint not found", job 11579994).
    """

    CONFIGS = (
        'experiments/hydro_llm/configs/timellm_config.yaml',
        'experiments/hydro_llm/configs/tier0_ettcontrol.yaml',
    )

    def test_hpo_block_present_and_switch_absent(self):
        import yaml

        for p in self.CONFIGS:
            cfg = yaml.safe_load(open(p, encoding='utf-8'))
            assert 'hpo' not in cfg, f'{p}: `hpo` must stay phase-owned'
            assert cfg['hpo_num_samples'] == 30, p
            assert cfg['hpo_scheduler'] == 'asha', p
            assert cfg['hpo_resources_gpu'] == 0.25, p
            assert cfg['hpo_save_checkpoints'] is True, p

    def test_config_num_samples_beats_phase_full_50(self):
        from types import SimpleNamespace

        from experiments.hydro_llm.run_matrix import build_overrides

        cell = {
            'arch': 'timellm', 'dataset': 'swiss-river-1990', 'mode': 'none',
            'sub': 'default', 'tuning': 'frozen', 'backbone': 'GPT2',
            'seed': 2026, 'identifier_mode': 'none',
        }
        args = SimpleNamespace(
            phase='full', config=self.CONFIGS[0], hpo_num_samples=None,
            train_epochs=None, learning_rate=None, patience=None,
            max_train_samples=None,
        )
        ov = build_overrides(cell, args)
        # phase full still turns HPO ON (config has no `hpo` key)...
        assert ov.get('hpo') is True
        # ...but must NOT inject the phase-50: the config's 24 is authoritative.
        assert 'hpo_num_samples' not in ov, (
            'phase default leaked past the config hpo_num_samples'
        )


class TestTextEmbeddingSkipGuard:
    """Regression (2026-08-11, cluster job 11912254): text_embedding on a
    dataset WITHOUT station descriptions errored at run time (loud ValueError
    in _load_entity_descriptions) because the build-time auto-skip guard
    covered only entity_description. Both text-source modes need the text, so
    both must be skipped at cell-build time for text-less datasets."""

    def test_text_modes_skipped_without_descriptions(self, monkeypatch):
        import experiments.hydro_llm.run_matrix as rm

        from types import SimpleNamespace

        monkeypatch.setattr(rm, 'has_entity_descriptions', lambda ds: ds == 'swiss-river-1990')
        cells = rm.build_cells(SimpleNamespace(
            datasets=['swiss-river-1990', 'swiss-river-2010'],
            modes=['none', 'entity_description', 'text_embedding'],
            a2=['learnable'], a1=['default'], tuning=['frozen'],
            backbones=['GPT2'], seeds=[2026], arch='timellm',
        ))
        combos = {(c['dataset'], c['mode']) for c in cells}
        assert ('swiss-river-1990', 'entity_description') in combos
        assert ('swiss-river-1990', 'text_embedding') in combos
        assert ('swiss-river-2010', 'entity_description') not in combos
        assert ('swiss-river-2010', 'text_embedding') not in combos
        assert ('swiss-river-2010', 'none') in combos
