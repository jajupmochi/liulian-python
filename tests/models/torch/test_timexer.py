"""
Tests for TimeXer model adapter.

TimeXer is an exogenous-variable-aware Transformer for time series
forecasting. It is **forecast-only** and returns None for other tasks.
"""

import pytest
from tests.models.torch.conftest import (
    check_torch_available,
    sample_forecast_inputs,
    validate_forecast_output,
)


@pytest.fixture(scope='module', autouse=True)
def check_dependencies():
    check_torch_available()


class TestTimeXerForecast:
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
            'd_model': 64,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 128,
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': 'h',
            'patch_len': 16,
            'features': 'M',
        }

    def test_adapter_instantiation(self, forecast_config):
        from liulian.models.torch.timexer import TimeXerAdapter

        model = TimeXerAdapter(forecast_config)
        assert model is not None

    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        from liulian.models.torch.timexer import TimeXerAdapter

        model = TimeXerAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)
        validate_forecast_output(outputs, 4, 24, 7)

    def test_different_patch_len(self, forecast_config, sample_forecast_inputs):
        from liulian.models.torch.timexer import TimeXerAdapter

        for pl in [8, 12, 16, 24, 32, 48]:
            config = {**forecast_config, 'patch_len': pl}
            model = TimeXerAdapter(config)
            outputs = model.run(sample_forecast_inputs)
            validate_forecast_output(outputs, 4, 24, 7)

    def test_ms_features(self, forecast_config, sample_forecast_inputs):
        """Test with multivariate-to-single forecasting (features='MS')."""
        from liulian.models.torch.timexer import TimeXerAdapter

        config = {**forecast_config, 'features': 'MS', 'c_out': 1}
        model = TimeXerAdapter(config)
        outputs = model.run(sample_forecast_inputs)
        validate_forecast_output(outputs, 4, 24, 1)

    def test_non_forecast_returns_none(self, forecast_config):
        """Non-forecast tasks should raise or return empty due to None output."""
        from liulian.models.torch.timexer import TimeXerAdapter

        config = {**forecast_config, 'task_name': 'imputation'}
        model = TimeXerAdapter(config)
        # TimeXer returns None for non-forecast, adapter raises TypeError
        with pytest.raises(TypeError):
            model.run({
                'x_enc': __import__('torch').randn(4, 96, 7),
                'x_mark_enc': __import__('torch').randn(4, 96, 4),
            })
