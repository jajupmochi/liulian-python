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
    def test_only_1990_and_etth1_have_descriptions(self) -> None:
        assert has_entity_descriptions('swiss-river-1990') is True
        assert has_entity_descriptions('ETTh1') is True
        assert has_entity_descriptions('swiss-river-2010') is False
        assert has_entity_descriptions('swiss-river-zurich') is False
        assert has_entity_descriptions('does-not-exist') is False


class TestBuildCellsGuardrail:
    def test_entity_description_skipped_where_unavailable(self) -> None:
        cells = build_cells(_args())
        ed = [c for c in cells if c['mode'] == 'entity_description']
        # only swiss-river-1990 keeps its entity_description cell
        assert {c['dataset'] for c in ed} == {'swiss-river-1990'}

    def test_other_modes_run_on_all_datasets(self) -> None:
        cells = build_cells(_args())
        for mode in ('none', 'numeric_embedding'):
            got = {c['dataset'] for c in cells if c['mode'] == mode}
            assert got == {'swiss-river-1990', 'swiss-river-2010', 'swiss-river-zurich'}

    def test_total_cell_count_is_seven_not_nine(self) -> None:
        # 1990: none+entity_description+numeric_embedding (3);
        # 2010/zurich: none+numeric_embedding (2 each) => 3 + 2 + 2 = 7.
        assert len(build_cells(_args())) == 7

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
            'experiments/swiss_river/timellm_config.yaml',
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
        import torch

        from liulian.config import load_config
        from liulian.pipeline import build_dataset, build_model

        cfg = load_config(
            'experiments/swiss_river/timellm_config.yaml',
            cli_overrides={'data': 'swiss-river-1990', 'identifier_mode': 'coordinates_embedding',
                           'split_mode': 'per_entity', 'llm_layers': 1},
        )
        ds = build_dataset(cfg)
        # topology (with coords) must load for the coordinates_embedding mode
        assert ds.topology is not None and len(ds.topology.coordinates) == len(ds.station_ids)
        m = build_model(cfg, ds)
        feat = m.transparent_feat
        assert tuple(feat.shape) == (len(ds.station_ids), 2)
        assert not bool((feat == 0).all().item())          # not fake zeros
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
        cell = dict(arch='timellm', dataset='swiss-river-1990', mode='none', sub='default',
                    tuning='frozen', backbone='GPT2', seed=2026, identifier_mode='none', job_key='k')
        args = SimpleNamespace(phase='dev', config=cfg, hpo_num_samples=None, train_epochs=None,
                               learning_rate=None, patience=None, max_train_samples=None)
        ov = build_overrides(cell, args)
        # config keys are NOT injected as overrides (so the yaml base provides them, unclobbered)
        assert 'hpo' not in ov and 'train_epochs' not in ov and 'hpo_num_samples' not in ov

    def test_phase_default_applies_when_config_silent(self, tmp_path):
        from experiments.hydro_llm.run_matrix import build_overrides
        cfg = self._cfg(tmp_path, model='timellm')  # no hpo/train_epochs keys
        cell = dict(arch='timellm', dataset='swiss-river-1990', mode='none', sub='default',
                    tuning='frozen', backbone='GPT2', seed=2026, identifier_mode='none', job_key='k')
        args = SimpleNamespace(phase='full', config=cfg, hpo_num_samples=None, train_epochs=None,
                               learning_rate=None, patience=None, max_train_samples=None)
        ov = build_overrides(cell, args)
        assert ov.get('hpo') is True and ov.get('hpo_num_samples') == 50  # phase-full cap

    def test_explicit_cli_beats_config(self, tmp_path):
        from experiments.hydro_llm.run_matrix import build_overrides
        cfg = self._cfg(tmp_path, train_epochs=2)
        cell = dict(arch='timellm', dataset='swiss-river-1990', mode='none', sub='default',
                    tuning='frozen', backbone='GPT2', seed=2026, identifier_mode='none', job_key='k')
        args = SimpleNamespace(phase='dev', config=cfg, hpo_num_samples=None, train_epochs=17,
                               learning_rate=None, patience=None, max_train_samples=None)
        assert build_overrides(cell, args)['train_epochs'] == 17

    def test_apply_config_defaults_fills_omitted_axes(self, tmp_path):
        from experiments.hydro_llm.run_matrix import _apply_config_defaults
        cfg = self._cfg(tmp_path, model='timellm', data='swiss-river-2010',
                        llm_model='GPT2', llm_tuning='lora', seed=7, identifier_mode='random_embedding')
        args = SimpleNamespace(config=cfg, arch=None, datasets=None, modes=None, a2=None,
                               a1=None, tuning=None, backbones=None, seeds=None)
        _apply_config_defaults(args)
        assert args.arch == 'timellm' and args.datasets == ['swiss-river-2010']
        assert args.tuning == ['lora'] and args.seeds == [7]
        # identifier_mode random_embedding reverse-maps to numeric_embedding + random
        assert args.modes == ['numeric_embedding'] and args.a2 == ['random']

    def test_explicit_axis_flag_not_overwritten_by_config(self, tmp_path):
        from experiments.hydro_llm.run_matrix import _apply_config_defaults
        cfg = self._cfg(tmp_path, llm_tuning='lora')
        args = SimpleNamespace(config=cfg, arch='gpt4ts', datasets=['swiss-river-1990'],
                               modes=['none'], a2=['learnable'], a1=['default'],
                               tuning=['frozen'], backbones=['GPT2'], seeds=[2026])
        _apply_config_defaults(args)
        assert args.tuning == ['frozen'] and args.arch == 'gpt4ts'  # explicit wins
