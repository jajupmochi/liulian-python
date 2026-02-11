"""
Tests for FEDformer model adapter.

FEDformer uses frequency-enhanced decomposed transformer architecture
for long-term time series forecasting (ICML 2022, Fourier mode).
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


class TestFEDformerForecast:
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
            'd_layers': 1,
            'd_ff': 128,
            'dropout': 0.05,
            'moving_avg': 25,
            'embed': 'timeF',
            'freq': 'h',
            'activation': 'gelu',
            'modes': 64,
            'mode_select': 'random',
        }

    def test_adapter_instantiation(self, forecast_config):
        from liulian.models.torch.fedformer import FEDformerAdapter

        model = FEDformerAdapter(forecast_config)
        assert model is not None

    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        from liulian.models.torch.fedformer import FEDformerAdapter

        model = FEDformerAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)
        validate_forecast_output(outputs, 4, 24, 7)

    def test_different_moving_avg(self, forecast_config, sample_forecast_inputs):
        from liulian.models.torch.fedformer import FEDformerAdapter

        for ma in [13, 25]:
            config = {**forecast_config, 'moving_avg': ma}
            model = FEDformerAdapter(config)
            outputs = model.run(sample_forecast_inputs)
            validate_forecast_output(outputs, 4, 24, 7)


class TestFEDformerOtherTasks:
    @pytest.fixture
    def base_config(self):
        return {
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 64,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 128,
            'dropout': 0.05,
            'moving_avg': 25,
            'embed': 'timeF',
            'freq': 'h',
            'activation': 'gelu',
            'modes': 64,
            'mode_select': 'random',
        }

    def test_imputation(self, base_config, sample_imputation_inputs):
        from liulian.models.torch.fedformer import FEDformerAdapter

        config = {**base_config, 'task_name': 'imputation'}
        model = FEDformerAdapter(config)
        outputs = model.run(sample_imputation_inputs)
        validate_imputation_output(outputs, 4, 96, 7)

    def test_anomaly_detection(self, base_config, sample_anomaly_inputs):
        from liulian.models.torch.fedformer import FEDformerAdapter

        config = {**base_config, 'task_name': 'anomaly_detection'}
        model = FEDformerAdapter(config)
        outputs = model.run(sample_anomaly_inputs)
        validate_anomaly_output(outputs, 4, 96, 7)

    def test_classification(self, base_config, sample_classification_inputs):
        from liulian.models.torch.fedformer import FEDformerAdapter

        config = {**base_config, 'task_name': 'classification', 'num_class': 10}
        model = FEDformerAdapter(config)
        outputs = model.run(sample_classification_inputs)
        validate_classification_output(outputs, 4, 10)
