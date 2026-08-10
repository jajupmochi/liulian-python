"""Unit tests for the optimizer module."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from liulian.optim.base import OptimizationResult
from liulian.optim.ray_optimizer import RayOptimizer, make_trainable
from liulian.optim.search_spaces import resolve_search_space


class TestTimeLLMSearchSpace:
    """Mode-aware composition of the timellm_swiss HPO space.

    Locks the dead-knob guard added when timellm_swiss was introduced: the
    Time-LLM numeric embedding width IS d_model (tuned in the model space), so
    a standalone ``embedding_size`` must never leak into the timellm space —
    the same class of guard as patchtst + add_after_patch. Without the guard,
    HPO would tune a parameter that cannot change the trained model.
    """

    CORE = {'learning_rate', 'd_model', 'd_ff', 'llm_layers'}

    def _keys(self, **kw):
        return set(resolve_search_space(**kw))

    def test_none_is_the_four_core_knobs(self) -> None:
        assert self._keys(model='timellm', data='swiss-river-1990', identifier_mode='none') == self.CORE

    def test_embedding_mode_drops_dead_embedding_size(self) -> None:
        # identifier_mode 'embedding' is what numeric_embedding+learnable maps to.
        keys = self._keys(model='timellm', data='swiss-river-1990', identifier_mode='embedding')
        assert 'embedding_size' not in keys, 'dead knob leaked into timellm space'
        assert keys == self.CORE

    def test_soft_prompt_adds_its_only_knob(self) -> None:
        keys = self._keys(model='timellm', data='swiss-river-1990', identifier_mode='soft_prompt')
        assert keys == self.CORE | {'soft_prompt_len'}

    def test_text_modes_add_nothing(self) -> None:
        for mode in ('entity_description', 'text_embedding', 'random_embedding', 'onehot_embedding'):
            assert self._keys(model='timellm', data='swiss-river-1990', identifier_mode=mode) == self.CORE, mode

    def test_lstm_embedding_still_has_embedding_size(self) -> None:
        # Guard must NOT regress the lstm path, which genuinely tunes it.
        assert 'embedding_size' in self._keys(model='lstm', data='swiss-river-1990', identifier_mode='embedding')

    def test_canonical_values_present_in_grid(self) -> None:
        space = resolve_search_space(model='timellm', data='swiss-river-1990')
        # ray.tune.choice objects expose .categories; skip the value-level
        # assertion if the sampler backend does not (keeps the test CPU-only
        # and backend-agnostic — the key-level guarantees above still run).
        for name, canonical in (('d_model', 32), ('d_ff', 128), ('learning_rate', 0.01), ('llm_layers', 6)):
            cats = getattr(space[name], 'categories', None)
            if cats is not None:
                assert canonical in list(cats), f'{name} grid missing canonical {canonical}'


class TestOptimizationResult:
    def test_creation(self) -> None:
        result = OptimizationResult(
            best_config={'lr': 0.01},
            best_value=0.42,
            n_trials=4,
        )
        assert result.best_value == pytest.approx(0.42)
        assert result.n_trials == 4
        assert result.trials_summary == []


class TestRayOptimizer:
    def test_default_config(self) -> None:
        opt = RayOptimizer()
        assert opt.config['num_samples'] == 4
        assert opt.config['mode'] == 'min'

    def test_custom_config(self) -> None:
        opt = RayOptimizer(config={'num_samples': 10, 'mode': 'max'})
        assert opt.config['num_samples'] == 10
        assert opt.config['mode'] == 'max'

    def test_merge_search_spaces(self) -> None:
        merged = RayOptimizer.merge_search_spaces(
            model_space={'lr': 0.01, 'hidden': 64},
            task_constraints={'hidden': 128},
            user_overrides={'lr': 0.001},
        )
        assert merged['lr'] == 0.001  # user wins
        assert merged['hidden'] == 128  # task wins over model

    def test_fallback_run(self) -> None:
        """Test grid-sweep fallback when Ray is not installed."""
        opt = RayOptimizer(config={'num_samples': 4, 'mode': 'min'})

        # Force fallback mode regardless of Ray availability
        opt._ray_available = False

        search_space = {
            'lr': [0.01, 0.001],
            'hidden': [32, 64],
        }
        result = opt.run(spec=None, search_space=search_space)

        assert isinstance(result, OptimizationResult)
        assert result.n_trials <= 4
        assert result.best_config  # must not be empty
        assert len(result.trials_summary) == result.n_trials

    def test_fallback_scalar_values(self) -> None:
        """Scalar values in search_space are auto-wrapped into lists."""
        opt = RayOptimizer(config={'num_samples': 1})
        opt._ray_available = False

        result = opt.run(spec=None, search_space={'lr': 0.01})
        assert result.n_trials == 1
        assert result.best_config['lr'] == 0.01

    def test_fallback_max_mode(self) -> None:
        opt = RayOptimizer(config={'num_samples': 2, 'mode': 'max'})
        opt._ray_available = False

        result = opt.run(spec=None, search_space={'x': [1, 2]})
        assert isinstance(result.best_value, float)

    def test_make_trainable_uses_trial_scoped_checkpoint_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ray trials should use unique checkpoint dirs to avoid collisions."""
        captured: dict[str, object] = {}

        class DummyTrainer:
            def __init__(
                self,
                config: dict[str, object],
                device: object = None,
                checkpoint_dir: str | None = None,
                exp_logger: object = None,
                inverse_transform: object = None,
            ) -> None:
                del device, exp_logger, inverse_transform
                captured['config'] = config
                captured['checkpoint_dir'] = checkpoint_dir

            def fit(self, *args: object, **kwargs: object) -> None:
                del args, kwargs

        class DummyModel:
            def __init__(self, args: SimpleNamespace) -> None:
                del args

            def float(self) -> 'DummyModel':
                return self

        class _FakeTuneContext:
            def get_trial_id(self) -> str:
                return 'abc123'

        class _FakeTuneModule:
            @staticmethod
            def get_context() -> _FakeTuneContext:
                return _FakeTuneContext()

        monkeypatch.setitem(sys.modules, 'ray', types.SimpleNamespace(tune=_FakeTuneModule))
        import liulian.runtime.trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, 'ForecastTrainer', DummyTrainer)

        _loaders = {'train': object(), 'val': object(), 'test': object()}
        trainable = make_trainable(
            model_cls=DummyModel,
            model_args=SimpleNamespace(),
            loaders=_loaders,
            base_config={},
        )
        # loaders are injected (tune.with_parameters) rather than closure-captured
        trainable({}, loaders=_loaders)

        assert captured['config'] == {'disable_early_stopping': False}
        assert captured['checkpoint_dir'] == 'checkpoints/trial_abc123'


class TestPerExperimentSearchSpaceFile:
    """`search_space_file` config key: per-experiment grid file with default
    fallback (2026-08-07). Locks four contracts:
      (a) a custom file's grid wins over the default;
      (b) no key -> the default liulian/optim/search_spaces.yaml (unchanged);
      (c) a configured-but-MISSING path raises (silent fallback would run the
          wrong grid unnoticed);
      (d) the shipped hydro-LLM file resolves the same knob set as the default
          does today (identical at fork time), incl. the soft_prompt extra —
          guarding the identifier_spaces-must-live-in-the-same-file trap.
    """

    def test_custom_file_wins(self, tmp_path):
        import yaml

        custom = tmp_path / 'space.yaml'
        custom.write_text(
            yaml.safe_dump({
                'model_spaces': {'tiny': {'learning_rate': {'dist': 'choice', 'values': [0.5]}}},
                'identifier_spaces': {'none': {}},
                'resolution': [{'data': 'swiss-river', 'model': 'timellm', 'space': 'tiny'}],
            }),
            encoding='utf-8',
        )
        space = resolve_search_space(
            model='timellm', data='swiss-river-1990', search_space_file=str(custom)
        )
        assert set(space) == {'learning_rate'}

    def test_no_key_uses_default_file(self):
        default = resolve_search_space(model='timellm', data='swiss-river-1990')
        explicit_none = resolve_search_space(
            model='timellm', data='swiss-river-1990', search_space_file=None
        )
        assert set(default) == set(explicit_none)
        assert {'learning_rate', 'd_model', 'd_ff', 'llm_layers'} <= set(default)

    def test_missing_configured_file_raises(self):
        with pytest.raises(FileNotFoundError, match='search_space_file'):
            resolve_search_space(
                model='timellm', data='swiss-river-1990',
                search_space_file='does/not/exist.yaml',
            )

    def test_hydro_llm_file_matches_default_at_fork(self):
        hydro = 'experiments/hydro_llm/configs/search_spaces.yaml'
        for mode in ('none', 'entity_description', 'random_embedding'):
            a = resolve_search_space(model='timellm', data='swiss-river-1990', identifier_mode=mode)
            b = resolve_search_space(
                model='timellm', data='swiss-river-1990', identifier_mode=mode,
                search_space_file=hydro,
            )
            assert set(a) == set(b) == {'learning_rate', 'd_model', 'd_ff', 'llm_layers'}, mode
        sp = resolve_search_space(
            model='timellm', data='swiss-river-1990', identifier_mode='soft_prompt',
            search_space_file=hydro,
        )
        assert 'soft_prompt_len' in sp, 'identifier_spaces entry lost in the per-experiment file'

    def test_hydro_configs_point_at_existing_file(self):
        import os

        import yaml

        for p in (
            'experiments/hydro_llm/configs/timellm_config.yaml',
            'experiments/hydro_llm/configs/tier0_ettcontrol.yaml',
        ):
            cfg = yaml.safe_load(open(p, encoding='utf-8'))
            f = cfg.get('search_space_file')
            assert f, f'{p}: search_space_file key missing'
            assert os.path.exists(f), f'{p}: search_space_file {f} does not exist'

    def test_debug_config_uses_dedicated_debug_grid(self):
        """debug.yaml points at search_spaces.debug.yaml (2026-08-07): same 4
        model knobs but d_model/d_ff/llm_layers pinned to single smallest values
        (fast trials) and lr keeps 2 values so the 2 debug trials differ; the
        soft_prompt extra survives (identifier entries not lost)."""
        import os

        import yaml

        cfg = yaml.safe_load(open('experiments/hydro_llm/configs/debug.yaml', encoding='utf-8'))
        f = cfg.get('search_space_file')
        assert f and f.endswith('search_spaces.debug.yaml') and os.path.exists(f)
        raw = yaml.safe_load(open(f, encoding='utf-8'))
        grid = raw['model_spaces']['timellm_swiss_debug']
        assert len(grid['learning_rate']['values']) == 2
        for pinned in ('d_model', 'd_ff', 'llm_layers'):
            assert len(grid[pinned]['values']) == 1, pinned
        sp = resolve_search_space(
            model='timellm', data='swiss-river-1990', identifier_mode='soft_prompt',
            search_space_file=f,
        )
        assert {'learning_rate', 'd_model', 'd_ff', 'llm_layers', 'soft_prompt_len'} == set(sp)


class TestHpoDebugAttach:
    """`hpo_debug_attach` (2026-08-07): opt-in reverse-connect from a Ray worker
    to a PyCharm Debug Server — the only way IDE breakpoints inside the
    trainable can hit (Ray 2.x removed local_mode; num_cpus=1 still uses
    worker processes). Locks: absent -> no-op; set-but-unreachable -> LOUD
    RuntimeError (silently running undebugged would defeat the point)."""

    def test_absent_key_is_noop(self):
        from liulian.optim.ray_optimizer import _maybe_attach_debugger

        _maybe_attach_debugger({})  # must not raise, must not try to connect
        _maybe_attach_debugger({'hpo_debug_attach': None})
        _maybe_attach_debugger({'hpo_debug_attach': False})

    def test_unreachable_server_raises_loudly(self):
        from liulian.optim.ray_optimizer import _maybe_attach_debugger

        # Port 1 on localhost is never a PyCharm debug server -> connection
        # refused -> our RuntimeError with the setup hint.
        with pytest.raises(RuntimeError, match='Debug Server'):
            _maybe_attach_debugger({'hpo_debug_attach': 'localhost:1'})

    def test_missing_package_raises_with_install_hint(self, monkeypatch):
        import builtins

        from liulian.optim import ray_optimizer as ro

        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == 'pydevd_pycharm':
                raise ImportError('not installed')
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, '__import__', fake_import)
        with pytest.raises(RuntimeError, match='pydevd-pycharm'):
            ro._maybe_attach_debugger({'hpo_debug_attach': True})

    def test_settrace_called_with_version_stable_kwargs_only(self, monkeypatch):
        """Regression (2026-08-10): pydevd-pycharm 2xx renamed the redirect
        kwargs (stdoutToServer -> stdout_to_server), so passing the old camelCase
        names raised TypeError BEFORE any connection attempt — and the broad
        except then misreported it as 'could not reach the server'. The helper
        must pass ONLY host/port/suspend (accepted by every version). The fake
        below has the NEW signature and NO **kwargs: the pre-fix call fails on
        it, the fixed call succeeds."""
        import sys
        import types

        calls = {}

        def fake_settrace(host=None, stdout_to_server=False, stderr_to_server=False,
                          port=5678, suspend=True, trace_only_current_thread=False,
                          overwrite_prev_trace=False, patch_multiprocessing=False,
                          stop_at_frame=None):
            calls['host'], calls['port'], calls['suspend'] = host, port, suspend

        fake_mod = types.ModuleType('pydevd_pycharm')
        fake_mod.settrace = fake_settrace
        monkeypatch.setitem(sys.modules, 'pydevd_pycharm', fake_mod)

        from liulian.optim.ray_optimizer import _maybe_attach_debugger

        _maybe_attach_debugger({'hpo_debug_attach': 'localhost:5678'})
        assert calls == {'host': 'localhost', 'port': 5678, 'suspend': False}
