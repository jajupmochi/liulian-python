"""
Tests for PatchTST model adapter.

PatchTST uses patch-based tokenization for efficient long-term forecasting
with channel independence or shared projection.
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


class TestPatchTSTForecast:
    """Test PatchTST adapter for forecasting task."""

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
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'fc_dropout': 0.2,
            'head_dropout': 0.0,
            'patch_len': 16,
            'stride': 8,
            'individual': False,
            'revin': True,
            'affine': False,
            'subtract_last': False,
            'decomposition': False,
            'kernel_size': 25,
        }

    def test_adapter_instantiation(self, forecast_config):
        """Test adapter can be instantiated."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        model = PatchTSTAdapter(forecast_config)
        assert model is not None
        assert hasattr(model, 'run')

    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        """Test forecasting task execution."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        model = PatchTSTAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)

        validate_forecast_output(
            outputs,
            batch_size=4,
            pred_len=forecast_config['pred_len'],
            features=forecast_config['c_out'],
        )

    def test_different_patch_configs(self, forecast_config, sample_forecast_inputs):
        """Test different patch length and stride combinations."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        patch_configs = [
            (16, 8),  # Default
            (8, 4),  # Smaller patches
            (24, 12),  # Larger patches
            (12, 12),  # No overlap
        ]

        for patch_len, stride in patch_configs:
            config = forecast_config.copy()
            config['patch_len'] = patch_len
            config['stride'] = stride

            model = PatchTSTAdapter(config)
            outputs = model.run(sample_forecast_inputs)

            validate_forecast_output(outputs, 4, 24, 7)

    def test_individual_vs_shared(self, forecast_config, sample_forecast_inputs):
        """Test individual vs shared channel projection."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        # Individual projection per channel
        config_individual = forecast_config.copy()
        config_individual['individual'] = True
        model_individual = PatchTSTAdapter(config_individual)
        outputs_individual = model_individual.run(sample_forecast_inputs)

        # Shared projection across channels
        config_shared = forecast_config.copy()
        config_shared['individual'] = False
        model_shared = PatchTSTAdapter(config_shared)
        outputs_shared = model_shared.run(sample_forecast_inputs)

        # Both should produce valid outputs
        validate_forecast_output(outputs_individual, 4, 24, 7)
        validate_forecast_output(outputs_shared, 4, 24, 7)

    def test_revin_normalization(self, forecast_config, sample_forecast_inputs):
        """Test RevIN (Reversible Instance Normalization)."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        # With RevIN (recommended)
        config_revin = forecast_config.copy()
        config_revin['revin'] = True
        model_revin = PatchTSTAdapter(config_revin)
        outputs_revin = model_revin.run(sample_forecast_inputs)

        # Without RevIN
        config_no_revin = forecast_config.copy()
        config_no_revin['revin'] = False
        model_no_revin = PatchTSTAdapter(config_no_revin)
        outputs_no_revin = model_no_revin.run(sample_forecast_inputs)

        validate_forecast_output(outputs_revin, 4, 24, 7)
        validate_forecast_output(outputs_no_revin, 4, 24, 7)

    def test_with_decomposition(self, forecast_config, sample_forecast_inputs):
        """Test with series decomposition."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        config = forecast_config.copy()
        config['decomposition'] = True
        config['kernel_size'] = 25

        model = PatchTSTAdapter(config)
        outputs = model.run(sample_forecast_inputs)

        validate_forecast_output(outputs, 4, 24, 7)


class TestPatchTSTOtherTasks:
    """Test PatchTST adapter for other tasks."""

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
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'patch_len': 16,
            'stride': 8,
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
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'patch_len': 16,
            'stride': 8,
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
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'patch_len': 16,
            'stride': 8,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    def test_imputation(self, imputation_config, sample_imputation_inputs):
        """Test imputation task."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        model = PatchTSTAdapter(imputation_config)
        outputs = model.run(sample_imputation_inputs)
        validate_imputation_output(outputs, 4, 96, 7)

    def test_anomaly_detection(self, anomaly_config, sample_anomaly_inputs):
        """Test anomaly detection task."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        model = PatchTSTAdapter(anomaly_config)
        outputs = model.run(sample_anomaly_inputs)
        validate_anomaly_output(outputs, 4, 96, 7)

    def test_classification(self, classification_config, sample_classification_inputs):
        """Test classification task."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        model = PatchTSTAdapter(classification_config)
        outputs = model.run(sample_classification_inputs)
        validate_classification_output(outputs, 4, 10)
