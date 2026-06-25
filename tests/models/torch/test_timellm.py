"""
Tests for TimeLLM model adapter.

TimeLLM reprograms Large Language Models (LLMs) for time series forecasting
by converting time series data into text-like representations.
"""

import pytest
import numpy as np
from tests.models.torch.conftest import (
    check_torch_available,
    check_transformers_available,
    validate_forecast_output,
)


@pytest.fixture(scope='module', autouse=True)
def check_dependencies():
    """Check required dependencies are installed."""
    check_torch_available()
    check_transformers_available()


class TestTimeLLMForecast:
    """Test TimeLLM adapter for forecasting task.

    Note: TimeLLM only supports forecasting, not other tasks.
    """

    @pytest.fixture
    def forecast_config_gpt2(self):
        """Configuration for forecasting with GPT2."""
        return {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 32,
            'd_ff': 128,
            'patch_len': 16,
            'stride': 8,
            'llm_model': 'GPT2',
            'llm_dim': 768,
            'llm_layers': 6,
            'prompt_domain': False,
            'llm_path': 'gpt2',  # Will download from HuggingFace
        }

    @pytest.fixture
    def forecast_config_llama(self):
        """Configuration for forecasting with LLAMA."""
        return {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 32,
            'd_ff': 128,
            'patch_len': 16,
            'stride': 8,
            'llm_model': 'LLAMA',
            'llm_dim': 4096,
            'llm_layers': 32,
            'prompt_domain': False,
            'llm_path': 'huggyllama/llama-7b',  # Requires model access
        }

    @pytest.fixture
    def sample_inputs(self):
        """Sample inputs for forecasting."""
        return {
            'x_enc': np.random.randn(4, 96, 7).astype(np.float32),
            'x_mark_enc': np.random.randn(4, 96, 4).astype(np.float32),
            'x_dec': np.random.randn(4, 72, 7).astype(np.float32),
            'x_mark_dec': np.random.randn(4, 72, 4).astype(np.float32),
        }

    @pytest.mark.slow
    @pytest.mark.download
    def test_adapter_instantiation_gpt2(self, forecast_config_gpt2):
        """Test adapter can be instantiated with GPT2.

        Marked as slow because it may download GPT2 model (~500MB).
        """
        from liulian.models.torch.timellm import TimeLLMAdapter

        model = TimeLLMAdapter(forecast_config_gpt2)
        assert model is not None
        assert hasattr(model, 'run')

    @pytest.mark.slow
    @pytest.mark.download
    def test_forecast_run_gpt2(self, forecast_config_gpt2, sample_inputs):
        """Test forecasting with GPT2.

        Marked as slow due to LLM inference time.
        """
        from liulian.models.torch.timellm import TimeLLMAdapter

        model = TimeLLMAdapter(forecast_config_gpt2)
        outputs = model.run(sample_inputs)

        validate_forecast_output(
            outputs,
            batch_size=4,
            pred_len=forecast_config_gpt2['pred_len'],
            features=forecast_config_gpt2['c_out'],
        )

    @pytest.mark.skip(reason='Requires LLAMA model access token')
    def test_forecast_run_llama(self, forecast_config_llama, sample_inputs):
        """Test forecasting with LLAMA.

        Skipped by default because LLAMA requires:
        1. HuggingFace access token
        2. Model download (~13GB for 7B model)
        3. Significant GPU memory
        """
        from liulian.models.torch.timellm import TimeLLMAdapter

        model = TimeLLMAdapter(forecast_config_llama)
        outputs = model.run(sample_inputs)

        validate_forecast_output(outputs, 4, 24, 7)

    @pytest.mark.slow
    @pytest.mark.download
    def test_different_patch_configs(self, forecast_config_gpt2, sample_inputs):
        """Test different patch configurations."""
        from liulian.models.torch.timellm import TimeLLMAdapter

        patch_configs = [
            (16, 8),  # Default
            (8, 4),  # Smaller patches
            (24, 12),  # Larger patches
        ]

        for patch_len, stride in patch_configs:
            config = forecast_config_gpt2.copy()
            config['patch_len'] = patch_len
            config['stride'] = stride

            model = TimeLLMAdapter(config)
            outputs = model.run(sample_inputs)

            validate_forecast_output(outputs, 4, 24, 7)

    @pytest.mark.slow
    @pytest.mark.download
    def test_with_prompt_domain(self, forecast_config_gpt2, sample_inputs):
        """Test with domain-specific prompt."""
        from liulian.models.torch.timellm import TimeLLMAdapter

        config = forecast_config_gpt2.copy()
        config['prompt_domain'] = True

        model = TimeLLMAdapter(config)
        outputs = model.run(sample_inputs)

        validate_forecast_output(outputs, 4, 24, 7)

    @pytest.mark.slow
    @pytest.mark.download
    def test_bert_backbone(self):
        """Test with BERT as LLM backbone."""
        from liulian.models.torch.timellm import TimeLLMAdapter

        config = {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 32,
            'd_ff': 128,
            'patch_len': 16,
            'stride': 8,
            'llm_model': 'BERT',
            'llm_dim': 768,
            'llm_layers': 12,
            'prompt_domain': False,
            'llm_path': 'bert-base-uncased',
        }

        model = TimeLLMAdapter(config)

        inputs = {
            'x_enc': np.random.randn(4, 96, 7).astype(np.float32),
            'x_mark_enc': np.random.randn(4, 96, 4).astype(np.float32),
            'x_dec': np.random.randn(4, 72, 7).astype(np.float32),
            'x_mark_dec': np.random.randn(4, 72, 4).astype(np.float32),
        }

        outputs = model.run(inputs)
        validate_forecast_output(outputs, 4, 24, 7)


class TestTimeLLMEdgeCases:
    """Test edge cases and error handling for TimeLLM."""

    @pytest.fixture
    def forecast_config(self):
        return {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 32,
            'd_ff': 128,
            'patch_len': 16,
            'stride': 8,
            'llm_model': 'GPT2',
            'llm_dim': 768,
            'llm_layers': 6,
            'prompt_domain': False,
            'llm_path': 'gpt2',
        }

    @pytest.mark.slow
    @pytest.mark.download
    def test_missing_input_keys(self, forecast_config):
        """Test error handling for missing input keys."""
        from liulian.models.torch.timellm import TimeLLMAdapter

        model = TimeLLMAdapter(forecast_config)

        # Only x_enc provided — TimeLLM actually tolerates this since the
        # base adapter fills in None for missing optional keys and the
        # model can run with just x_enc.
        # Test with truly empty inputs to verify error handling.
        incomplete_inputs = {}

        with pytest.raises((KeyError, AttributeError, TypeError)):
            model.run(incomplete_inputs)


class TestTimeLLMDownload:
    """Tests related to model downloading.

    All tests here are skipped by default and require manual execution.
    """

    @pytest.mark.skip(reason='Manual test - downloads large models')
    def test_gpt2_download():
        """Test initial GPT2 model download.

        Run manually with: pytest tests/models/torch/test_timellm.py::test_gpt2_download -v

        Downloads ~500MB from HuggingFace.
        """
        from liulian.models.torch.timellm import TimeLLMAdapter

        config = {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 32,
            'd_ff': 128,
            'patch_len': 16,
            'stride': 8,
            'llm_model': 'GPT2',
            'llm_dim': 768,
            'llm_layers': 6,
            'prompt_domain': False,
            'llm_path': 'gpt2',
        }

        # First instantiation will download
        model = TimeLLMAdapter(config)
        assert model is not None

        # Subsequent instantiations should use cache
        model2 = TimeLLMAdapter(config)
        assert model2 is not None

    @pytest.mark.skip(reason='Manual test - requires LLAMA access')
    def test_llama_download():
        """Test LLAMA model download.

        Requires:
        1. HuggingFace account with LLAMA access
        2. Authentication token set in environment
        3. ~13GB disk space
        4. GPU with sufficient memory
        """
        from liulian.models.torch.timellm import TimeLLMAdapter

        config = {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 32,
            'd_ff': 128,
            'patch_len': 16,
            'stride': 8,
            'llm_model': 'LLAMA',
            'llm_dim': 4096,
            'llm_layers': 32,
            'prompt_domain': False,
            'llm_path': 'huggyllama/llama-7b',
        }

        model = TimeLLMAdapter(config)
        assert model is not None


class TestTimeLLMEntityDescriptionPrompt:
    """Fast (no-GPT2) tests for the H4 ``entity_description`` prompt injection.

    ``Model._compose_prompt`` is a staticmethod, so these exercise the real H4
    prompt-composition logic without instantiating the LLM (no download, no
    transformers forward). They lock two properties:

    * **Parity** — with ``entity_desc=None`` (or ``''``) the prompt is
      byte-identical to the original pre-H4 prompt, preserving the V1/V2
      bit-exact reproduction on the ``none`` path.
    * **Effect** — with a description set, that text actually appears in the
      prompt and changes it (no dead knob).
    """

    # Fixed sample stats shared across the cases below.
    _DESC = 'Swiss River Network water temperature. '
    _PRED, _SEQ = '24', '96'
    _MIN, _MAX, _MED = '0.1', '0.9', '0.5'
    _LAGS = '[1, 2, 3, 24, 25]'

    def _baseline(self, trend_up: bool = True) -> str:
        """The exact original (pre-H4) prompt for the shared sample stats."""
        return (
            f'<|start_prompt|>Dataset description: {self._DESC}'
            f'Task description: forecast the next {self._PRED} steps given the previous {self._SEQ} steps information; '
            'Input statistics: '
            f'min value {self._MIN}, '
            f'max value {self._MAX}, '
            f'median value {self._MED}, '
            f'the trend of input is {"upward" if trend_up else "downward"}, '
            f'top 5 lags are : {self._LAGS}<|<end_prompt>|>'
        )

    def _compose(self, entity_desc, trend_up: bool = True) -> str:
        from liulian.models.torch.timellm import Model

        return Model._compose_prompt(
            self._DESC,
            entity_desc,
            self._PRED,
            self._SEQ,
            self._MIN,
            self._MAX,
            self._MED,
            trend_up,
            self._LAGS,
        )

    def test_none_is_byte_identical_to_baseline(self):
        """entity_desc=None ⟹ byte-identical to the verified pre-H4 prompt."""
        assert self._compose(None) == self._baseline(trend_up=True)
        assert self._compose(None, trend_up=False) == self._baseline(trend_up=False)
        assert 'Entity description:' not in self._compose(None)

    def test_empty_string_falls_back_to_baseline(self):
        """Empty description is falsy ⟹ no injection (same as None)."""
        assert self._compose('') == self._baseline()

    def test_description_is_injected_and_changes_prompt(self):
        """A real description appears verbatim and changes the prompt (no dead knob)."""
        desc = 'River Aare at Bern, alpine river, elevation 502 m'
        out = self._compose(desc)
        baseline = self._baseline()
        assert f'Entity description: {desc};' in out
        assert out != baseline
        assert len(out) > len(baseline)
        # Injected segment sits between the task description and the statistics.
        assert out.index('Entity description:') < out.index('Input statistics:')

    def test_distinct_descriptions_yield_distinct_prompts(self):
        """Two entities ⟹ two different prompts (identity actually discriminates)."""
        a = self._compose('Station A: Rhine at Basel')
        b = self._compose('Station B: Ticino at Bellinzona')
        assert a != b
        assert 'Rhine at Basel' in a and 'Ticino at Bellinzona' in b

    def test_resolve_baseline_none(self):
        """entity_descriptions=None → all-None (baseline; byte-identical prompt)."""
        from liulian.models.torch.timellm import Model

        assert Model._resolve_entity_descs(None, None, None, 5, 1) == [None] * 5

    def test_resolve_per_sample_from_x_mark(self):
        """Per-sample path: id read from x_mark column maps each sample's desc."""
        import torch

        from liulian.models.torch.timellm import Model

        descs = ['A', 'B', 'C']
        # x_mark_enc: (B=4, T=2, mark=3); last column carries the per-sample id.
        xm = torch.zeros(4, 2, 3)
        xm[:, :, -1] = torch.tensor([2.0, 0.0, 1.0, 2.0]).unsqueeze(1)
        out = Model._resolve_entity_descs(descs, -1, xm, 4, 1)
        assert out == ['C', 'A', 'B', 'C']

    def test_resolve_per_sample_out_of_range_raises(self):
        """An id beyond the table fails loud (no silent wrong identity)."""
        import pytest
        import torch

        from liulian.models.torch.timellm import Model

        xm = torch.zeros(2, 2, 1)
        xm[:, :, -1] = torch.tensor([0.0, 9.0]).unsqueeze(1)  # 9 is out of range
        with pytest.raises(ValueError, match='out of range'):
            Model._resolve_entity_descs(['A', 'B'], -1, xm, 2, 1)

    def test_resolve_per_channel_mod_n(self):
        """Per-channel fallback (no x_mark col): b % N over the flattened series."""
        from liulian.models.torch.timellm import Model

        out = Model._resolve_entity_descs(['x', 'y', 'z'], None, None, 6, 3)
        assert out == ['x', 'y', 'z', 'x', 'y', 'z']

    def test_resolve_per_channel_length_mismatch_raises(self):
        """Per-channel path with wrong table length fails loud."""
        import pytest

        from liulian.models.torch.timellm import Model

        with pytest.raises(ValueError, match='one per channel'):
            Model._resolve_entity_descs(['x', 'y'], None, None, 7, 7)
