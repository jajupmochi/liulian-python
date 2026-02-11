"""
Tests for Informer model adapter.

Informer uses ProbSparse attention for efficient long sequence forecasting
with O(L log L) complexity.
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


class TestInformerForecast:
    """Test Informer adapter for forecasting task."""

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
            'd_layers': 1,
            'd_ff': 2048,
            'factor': 5,
            'distil': True,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    def test_adapter_instantiation(self, forecast_config):
        """Test adapter can be instantiated."""
        from liulian.models.torch.informer import InformerAdapter

        model = InformerAdapter(forecast_config)
        assert model is not None
        assert hasattr(model, 'run')

    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        """Test forecasting task execution."""
        from liulian.models.torch.informer import InformerAdapter

        model = InformerAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)

        validate_forecast_output(
            outputs,
            batch_size=4,
            pred_len=forecast_config['pred_len'],
            features=forecast_config['c_out'],
        )

    def test_probsparse_factor(self, forecast_config, sample_forecast_inputs):
        """Test different ProbSparse sampling factors."""
        from liulian.models.torch.informer import InformerAdapter

        for factor in [1, 3, 5, 7]:
            config = forecast_config.copy()
            config['factor'] = factor

            model = InformerAdapter(config)
            outputs = model.run(sample_forecast_inputs)

            validate_forecast_output(outputs, 4, 24, 7)

    def test_with_without_distillation(self, forecast_config, sample_forecast_inputs):
        """Test with/without distillation (memory optimization)."""
        from liulian.models.torch.informer import InformerAdapter

        # With distillation (recommended)
        config_distil = forecast_config.copy()
        config_distil['distil'] = True
        model_distil = InformerAdapter(config_distil)
        outputs_distil = model_distil.run(sample_forecast_inputs)

        # Without distillation
        config_no_distil = forecast_config.copy()
        config_no_distil['distil'] = False
        model_no_distil = InformerAdapter(config_no_distil)
        outputs_no_distil = model_no_distil.run(sample_forecast_inputs)

        # Both should produce valid outputs
        validate_forecast_output(outputs_distil, 4, 24, 7)
        validate_forecast_output(outputs_no_distil, 4, 24, 7)

    def test_long_sequence(self):
        """Test with long input sequence (Informer's strength)."""
        from liulian.models.torch.informer import InformerAdapter

        config = {
            'task_name': 'long_term_forecast',
            'seq_len': 512,  # Long sequence
            'pred_len': 96,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 2048,
            'factor': 5,  # Higher factor for longer sequences
            'distil': True,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

        model = InformerAdapter(config)

        inputs = {
            'x_enc': np.random.randn(4, 512, 7).astype(np.float32),
            'x_mark_enc': np.random.randn(4, 512, 4).astype(np.float32),
            'x_dec': np.random.randn(4, 144, 7).astype(np.float32),
            'x_mark_dec': np.random.randn(4, 144, 4).astype(np.float32),
        }

        outputs = model.run(inputs)
        validate_forecast_output(outputs, 4, 96, 7)


class TestInformerOtherTasks:
    """Test Informer adapter for other tasks."""

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
            'd_layers': 1,
            'd_ff': 2048,
            'factor': 5,
            'distil': True,
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
            'd_layers': 1,
            'd_ff': 2048,
            'factor': 5,
            'distil': True,
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
            'd_layers': 1,
            'd_ff': 2048,
            'factor': 5,
            'distil': True,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    def test_imputation(self, imputation_config, sample_imputation_inputs):
        """Test imputation task."""
        from liulian.models.torch.informer import InformerAdapter

        model = InformerAdapter(imputation_config)
        outputs = model.run(sample_imputation_inputs)
        validate_imputation_output(outputs, 4, 96, 7)

    def test_anomaly_detection(self, anomaly_config, sample_anomaly_inputs):
        """Test anomaly detection task."""
        from liulian.models.torch.informer import InformerAdapter

        model = InformerAdapter(anomaly_config)
        outputs = model.run(sample_anomaly_inputs)
        validate_anomaly_output(outputs, 4, 96, 7)

    def test_classification(self, classification_config, sample_classification_inputs):
        """Test classification task."""
        from liulian.models.torch.informer import InformerAdapter

        model = InformerAdapter(classification_config)
        outputs = model.run(sample_classification_inputs)
        validate_classification_output(outputs, 4, 10)
