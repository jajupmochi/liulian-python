"""
Tests for TimeMoE model adapter.

TimeMoE is a pretrained Mixture of Experts model for zero-shot forecasting.
Requires transformers library (torch-models-full).
"""

import pytest
import numpy as np
from tests.models.torch.conftest import (
    check_torch_available,
    check_transformers_available,
    sample_forecast_inputs,
    validate_forecast_output,
)


@pytest.fixture(scope='module', autouse=True)
def check_dependencies():
    """Check required dependencies are installed."""
    check_torch_available()
    check_transformers_available()


class TestTimeMoEForecast:
    """Test TimeMoE adapter for forecasting task (zero-shot)."""

    @pytest.fixture
    def forecast_config(self):
        """Configuration for forecasting."""
        return {
            'model_name': 'Maple728/TimeMoE-50M',
            'seq_len': 96,
            'pred_len': 24,
            'd_ff': 2048,
            'd_model': 512,
            'top_k': 2,
            'n_heads': 8,
            'enc_in': 7,
        }

    @pytest.mark.slow
    @pytest.mark.download
    def test_adapter_instantiation(self, forecast_config):
        """Test adapter can be instantiated (may download ~200MB model on first run)."""
        from liulian.models.torch.timemoe import TimeMoEAdapter

        model = TimeMoEAdapter(forecast_config)
        assert model is not None
        assert hasattr(model, 'run')

    @pytest.mark.slow
    @pytest.mark.download
    @pytest.mark.xfail(
        reason='TimeMoE trust_remote_code model incompatible with transformers>=5.0 (DynamicCache API change)',
        raises=AttributeError,
    )
    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        """Test zero-shot forecasting task execution."""
        from liulian.models.torch.timemoe import TimeMoEAdapter

        model = TimeMoEAdapter(forecast_config)

        # TimeMoE only needs x_enc for zero-shot forecasting
        inputs = {'x_enc': sample_forecast_inputs['x_enc']}
        outputs = model.run(inputs)

        validate_forecast_output(
            outputs,
            batch_size=4,
            pred_len=forecast_config['pred_len'],
            features=forecast_config['enc_in'],
        )

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason='TimeMoE trust_remote_code model incompatible with transformers>=5.0 (DynamicCache API change)',
        raises=AttributeError,
    )
    def test_different_horizons(self, forecast_config):
        """Test different prediction horizons."""
        from liulian.models.torch.timemoe import TimeMoEAdapter

        for pred_len in [12, 24, 48, 96]:
            config = forecast_config.copy()
            config['pred_len'] = pred_len

            model = TimeMoEAdapter(config)
            inputs = {'x_enc': np.random.randn(4, 96, 7).astype(np.float32)}
            outputs = model.run(inputs)

            validate_forecast_output(outputs, 4, pred_len, 7)


class TestTimeMoEEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.slow
    def test_missing_input_keys(self):
        """Test error handling for missing input keys."""
        from liulian.models.torch.timemoe import TimeMoEAdapter

        config = {
            'model_name': 'Maple728/TimeMoE-50M',
            'seq_len': 96,
            'pred_len': 24,
            'd_ff': 2048,
            'd_model': 512,
            'top_k': 2,
            'n_heads': 8,
            'enc_in': 7,
        }

        model = TimeMoEAdapter(config)

        # Missing x_enc — model gets None and fails
        incomplete_inputs = {}

        with pytest.raises((KeyError, AttributeError, TypeError)):
            model.run(incomplete_inputs)


@pytest.mark.skip(reason='Model download required, run manually with --run-download')
class TestTimeMoEDownload:
    """Tests that require model download (run manually)."""

    def test_first_time_download(self):
        """Test first-time model download flow."""
        from liulian.models.torch.timemoe import TimeMoEAdapter

        config = {
            'model_name': 'Maple728/TimeMoE-50M',
            'seq_len': 96,
            'pred_len': 24,
            'd_ff': 2048,
            'd_model': 512,
            'top_k': 2,
            'n_heads': 8,
            'enc_in': 7,
        }

        # This will download ~200MB on first run
        model = TimeMoEAdapter(config)

        inputs = {'x_enc': np.random.randn(4, 96, 7).astype(np.float32)}
        outputs = model.run(inputs)

        validate_forecast_output(outputs, 4, 24, 7)
