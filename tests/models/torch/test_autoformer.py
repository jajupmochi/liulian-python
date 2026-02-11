"""
Tests for Autoformer model adapter.

Autoformer uses decomposition architecture with Auto-Correlation mechanism
for long-term forecasting with trend-seasonal decomposition.
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


class TestAutoformerForecast:
    """Test Autoformer adapter for forecasting task."""

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
            'moving_avg': 25,
            'factor': 1,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    def test_adapter_instantiation(self, forecast_config):
        """Test adapter can be instantiated."""
        from liulian.models.torch.autoformer import AutoformerAdapter

        model = AutoformerAdapter(forecast_config)
        assert model is not None
        assert hasattr(model, 'run')

    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        """Test forecasting task execution."""
        from liulian.models.torch.autoformer import AutoformerAdapter

        model = AutoformerAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)

        validate_forecast_output(
            outputs,
            batch_size=4,
            pred_len=forecast_config['pred_len'],
            features=forecast_config['c_out'],
        )

    def test_different_moving_avg_windows(
        self, forecast_config, sample_forecast_inputs
    ):
        """Test different moving average window sizes for decomposition."""
        from liulian.models.torch.autoformer import AutoformerAdapter

        for window_size in [5, 13, 25, 49]:
            config = forecast_config.copy()
            config['moving_avg'] = window_size

            model = AutoformerAdapter(config)
            outputs = model.run(sample_forecast_inputs)

            validate_forecast_output(outputs, 4, 24, 7)

    def test_decomposition_quality(self, forecast_config, sample_forecast_inputs):
        """Test that decomposition produces reasonable outputs."""
        from liulian.models.torch.autoformer import AutoformerAdapter

        model = AutoformerAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)

        # Outputs should be valid
        validate_forecast_output(outputs, 4, 24, 7)

        # Check variance (not all zeros, not all same value)
        assert outputs['predictions'].std() > 1e-6, 'Output variance too low'

    def test_seasonal_patterns(self):
        """Test Autoformer with synthetic seasonal data."""
        from liulian.models.torch.autoformer import AutoformerAdapter

        config = {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 1,
            'dec_in': 1,
            'c_out': 1,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 2048,
            'moving_avg': 25,
            'factor': 1,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

        model = AutoformerAdapter(config)

        # Create synthetic seasonal data
        t = np.arange(96)
        seasonal = np.sin(2 * np.pi * t / 24)  # Daily cycle
        trend = 0.01 * t  # Linear trend
        data = (seasonal + trend).astype(np.float32)

        inputs = {
            'x_enc': data.reshape(1, 96, 1),
            'x_mark_enc': np.random.randn(1, 96, 4).astype(np.float32),
            'x_dec': np.random.randn(1, 72, 1).astype(np.float32),
            'x_mark_dec': np.random.randn(1, 72, 4).astype(np.float32),
        }

        outputs = model.run(inputs)
        validate_forecast_output(outputs, 1, 24, 1)


class TestAutoformerOtherTasks:
    """Test Autoformer adapter for other tasks."""

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
            'moving_avg': 25,
            'factor': 1,
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
            'moving_avg': 25,
            'factor': 1,
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
            'moving_avg': 25,
            'factor': 1,
            'dropout': 0.1,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    def test_imputation(self, imputation_config, sample_imputation_inputs):
        """Test imputation task."""
        from liulian.models.torch.autoformer import AutoformerAdapter

        model = AutoformerAdapter(imputation_config)
        outputs = model.run(sample_imputation_inputs)
        validate_imputation_output(outputs, 4, 96, 7)

    def test_anomaly_detection(self, anomaly_config, sample_anomaly_inputs):
        """Test anomaly detection task."""
        from liulian.models.torch.autoformer import AutoformerAdapter

        model = AutoformerAdapter(anomaly_config)
        outputs = model.run(sample_anomaly_inputs)
        validate_anomaly_output(outputs, 4, 96, 7)

    def test_classification(self, classification_config, sample_classification_inputs):
        """Test classification task."""
        from liulian.models.torch.autoformer import AutoformerAdapter

        model = AutoformerAdapter(classification_config)
        outputs = model.run(sample_classification_inputs)
        validate_classification_output(outputs, 4, 10)
