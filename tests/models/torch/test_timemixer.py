"""
Tests for TimeMixer model adapter.

TimeMixer uses multi-scale mixing with seasonal–trend decomposition
for time series analysis (ICLR 2024).
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


class TestTimeMixerForecast:
    @pytest.fixture
    def forecast_config(self):
        # seq_len=96 must be divisible by window^layers = 2^2 = 4 -> 96/4=24 OK
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
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': 'h',
            'down_sampling_layers': 2,
            'down_sampling_window': 2,
            'down_sampling_method': 'avg',
            'channel_independence': True,
        }

    def test_adapter_instantiation(self, forecast_config):
        from liulian.models.torch.timemixer import TimeMixerAdapter

        model = TimeMixerAdapter(forecast_config)
        assert model is not None

    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        from liulian.models.torch.timemixer import TimeMixerAdapter

        model = TimeMixerAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)
        validate_forecast_output(outputs, 4, 24, 7)

    def test_channel_dependence(self, forecast_config, sample_forecast_inputs):
        from liulian.models.torch.timemixer import TimeMixerAdapter

        config = {**forecast_config, 'channel_independence': False}
        model = TimeMixerAdapter(config)
        outputs = model.run(sample_forecast_inputs)
        validate_forecast_output(outputs, 4, 24, 7)


class TestTimeMixerOtherTasks:
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
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': 'h',
            'down_sampling_layers': 2,
            'down_sampling_window': 2,
            'down_sampling_method': 'avg',
            'channel_independence': True,
        }

    def test_imputation(self, base_config, sample_imputation_inputs):
        from liulian.models.torch.timemixer import TimeMixerAdapter

        config = {**base_config, 'task_name': 'imputation'}
        model = TimeMixerAdapter(config)
        outputs = model.run(sample_imputation_inputs)
        validate_imputation_output(outputs, 4, 96, 7)

    def test_anomaly_detection(self, base_config, sample_anomaly_inputs):
        from liulian.models.torch.timemixer import TimeMixerAdapter

        config = {**base_config, 'task_name': 'anomaly_detection'}
        model = TimeMixerAdapter(config)
        outputs = model.run(sample_anomaly_inputs)
        validate_anomaly_output(outputs, 4, 96, 7)

    def test_classification(self, base_config, sample_classification_inputs):
        from liulian.models.torch.timemixer import TimeMixerAdapter

        config = {**base_config, 'task_name': 'classification', 'num_class': 10}
        model = TimeMixerAdapter(config)
        outputs = model.run(sample_classification_inputs)
        validate_classification_output(outputs, 4, 10)
