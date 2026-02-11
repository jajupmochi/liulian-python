"""
Tests for TimesNet model adapter.

TimesNet uses temporal 2-D variation modelling via Inception blocks and FFT
to capture intra‑ and inter‑period patterns (ICLR 2023).
"""

import pytest
from tests.models.torch.conftest import (
    check_torch_available,
    sample_forecast_inputs,
    sample_imputation_inputs,
    sample_anomaly_inputs,
    sample_classification_inputs,
    validate_forecast_output,
    validate_imputation_output,
    validate_anomaly_output,
    validate_classification_output,
)


@pytest.fixture(scope='module', autouse=True)
def check_dependencies():
    check_torch_available()


class TestTimesNetForecast:
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
            'd_ff': 32,
            'e_layers': 2,
            'top_k': 5,
            'num_kernels': 6,
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': 'h',
        }

    def test_adapter_instantiation(self, forecast_config):
        from liulian.models.torch.timesnet import TimesNetAdapter

        model = TimesNetAdapter(forecast_config)
        assert model is not None
        assert hasattr(model, 'run')

    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        from liulian.models.torch.timesnet import TimesNetAdapter

        model = TimesNetAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)
        validate_forecast_output(outputs, 4, 24, 7)

    def test_small_d_model(self, forecast_config, sample_forecast_inputs):
        from liulian.models.torch.timesnet import TimesNetAdapter

        config = forecast_config.copy()
        config['d_model'] = 16
        config['d_ff'] = 16
        model = TimesNetAdapter(config)
        outputs = model.run(sample_forecast_inputs)
        validate_forecast_output(outputs, 4, 24, 7)


class TestTimesNetOtherTasks:
    @pytest.fixture
    def base_config(self):
        return {
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 32,
            'd_ff': 32,
            'e_layers': 2,
            'top_k': 5,
            'num_kernels': 6,
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': 'h',
        }

    def test_imputation(self, base_config, sample_imputation_inputs):
        from liulian.models.torch.timesnet import TimesNetAdapter

        config = {**base_config, 'task_name': 'imputation'}
        model = TimesNetAdapter(config)
        outputs = model.run(sample_imputation_inputs)
        validate_imputation_output(outputs, 4, 96, 7)

    def test_anomaly_detection(self, base_config, sample_anomaly_inputs):
        from liulian.models.torch.timesnet import TimesNetAdapter

        config = {**base_config, 'task_name': 'anomaly_detection'}
        model = TimesNetAdapter(config)
        outputs = model.run(sample_anomaly_inputs)
        validate_anomaly_output(outputs, 4, 96, 7)

    def test_classification(self, base_config, sample_classification_inputs):
        from liulian.models.torch.timesnet import TimesNetAdapter

        config = {**base_config, 'task_name': 'classification', 'num_class': 10}
        model = TimesNetAdapter(config)
        outputs = model.run(sample_classification_inputs)
        validate_classification_output(outputs, 4, 10)
