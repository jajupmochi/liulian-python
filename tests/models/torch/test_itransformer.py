"""
Tests for iTransformer model adapter.

iTransformer applies attention across variates (not time), designed for
multivariate time series with strong inter-variate correlations.
"""

import pytest
import numpy as np
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
    """Check required dependencies are installed."""
    check_torch_available()


class TestiTransformerForecast:
    """Test iTransformer adapter for forecasting task."""

    @pytest.fixture
    def forecast_config(self):
        """Configuration for forecasting."""
        return {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 2048,
            'dropout': 0.1,
            'activation': 'gelu',
            'use_norm': True,
        }

    def test_adapter_instantiation(self, forecast_config):
        """Test adapter can be instantiated."""
        from liulian.models.torch.itransformer import iTransformerAdapter

        model = iTransformerAdapter(forecast_config)
        assert model is not None
        assert hasattr(model, 'run')

    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        """Test forecasting task execution."""
        from liulian.models.torch.itransformer import iTransformerAdapter

        model = iTransformerAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)

        validate_forecast_output(
            outputs,
            batch_size=4,
            pred_len=forecast_config['pred_len'],
            features=forecast_config['c_out'],
        )

    def test_with_without_norm(self, forecast_config, sample_forecast_inputs):
        """Test with/without non-stationary normalization."""
        from liulian.models.torch.itransformer import iTransformerAdapter

        # With normalization
        config_norm = forecast_config.copy()
        config_norm['use_norm'] = True
        model_norm = iTransformerAdapter(config_norm)
        outputs_norm = model_norm.run(sample_forecast_inputs)

        # Without normalization
        config_no_norm = forecast_config.copy()
        config_no_norm['use_norm'] = False
        model_no_norm = iTransformerAdapter(config_no_norm)
        outputs_no_norm = model_no_norm.run(sample_forecast_inputs)

        # Both should produce valid outputs
        validate_forecast_output(outputs_norm, 4, 24, 7)
        validate_forecast_output(outputs_no_norm, 4, 24, 7)

    def test_multivariate_focus(self):
        """Test iTransformer with many variates (its strength)."""
        from liulian.models.torch.itransformer import iTransformerAdapter

        # Configuration with many variates
        config = {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 21,  # Many variates
            'dec_in': 21,
            'c_out': 21,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 2048,
            'dropout': 0.1,
            'activation': 'gelu',
            'use_norm': True,
        }

        model = iTransformerAdapter(config)

        inputs = {
            'x_enc': np.random.randn(4, 96, 21).astype(np.float32),
            'x_mark_enc': np.random.randn(4, 96, 4).astype(np.float32),
            'x_dec': np.random.randn(4, 72, 21).astype(np.float32),
            'x_mark_dec': np.random.randn(4, 72, 4).astype(np.float32),
        }

        outputs = model.run(inputs)
        validate_forecast_output(outputs, 4, 24, 21)


class TestiTransformerOtherTasks:
    """Test iTransformer adapter for other tasks."""

    @pytest.fixture
    def imputation_config(self):
        return {
            'task_name': 'imputation',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 2048,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    @pytest.fixture
    def anomaly_config(self):
        return {
            'task_name': 'anomaly_detection',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 2048,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    @pytest.fixture
    def classification_config(self):
        return {
            'task_name': 'classification',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'num_class': 10,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 2048,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    def test_imputation(self, imputation_config, sample_imputation_inputs):
        """Test imputation task."""
        from liulian.models.torch.itransformer import iTransformerAdapter

        model = iTransformerAdapter(imputation_config)
        outputs = model.run(sample_imputation_inputs)
        validate_imputation_output(outputs, 4, 96, 7)

    def test_anomaly_detection(self, anomaly_config, sample_anomaly_inputs):
        """Test anomaly detection task."""
        from liulian.models.torch.itransformer import iTransformerAdapter

        model = iTransformerAdapter(anomaly_config)
        outputs = model.run(sample_anomaly_inputs)
        validate_anomaly_output(outputs, 4, 96, 7)

    def test_classification(self, classification_config, sample_classification_inputs):
        """Test classification task."""
        from liulian.models.torch.itransformer import iTransformerAdapter

        model = iTransformerAdapter(classification_config)
        outputs = model.run(sample_classification_inputs)
        validate_classification_output(outputs, 4, 10)
