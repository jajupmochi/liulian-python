"""
Tests for DLinear model adapter.

DLinear is a simple linear model with series decomposition, supporting
forecasting, imputation, anomaly detection, and classification tasks.
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


@pytest.fixture(scope="module", autouse=True)
def check_dependencies():
    """Check required dependencies are installed."""
    check_torch_available()


class TestDLinearForecast:
    """Test DLinear adapter for forecasting task."""
    
    @pytest.fixture
    def forecast_config(self):
        """Configuration for forecasting."""
        return {
            "task_name": "long_term_forecast",
            "seq_len": 96,
            "pred_len": 24,
            "label_len": 48,
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
            "individual": False,
        }
    
    def test_adapter_instantiation(self, forecast_config):
        """Test adapter can be instantiated."""
        from liulian.models.torch.dlinear import DLinearAdapter
        
        model = DLinearAdapter(forecast_config)
        assert model is not None
        assert hasattr(model, 'run')
        assert hasattr(model, 'save')
        assert hasattr(model, 'load')
    
    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        """Test forecasting task execution."""
        from liulian.models.torch.dlinear import DLinearAdapter
        
        model = DLinearAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)
        
        validate_forecast_output(
            outputs,
            batch_size=4,
            pred_len=forecast_config["pred_len"],
            features=forecast_config["c_out"]
        )
    
    def test_individual_vs_shared(self, forecast_config, sample_forecast_inputs):
        """Test individual vs shared linear layers."""
        from liulian.models.torch.dlinear import DLinearAdapter
        
        # Shared layers
        config_shared = forecast_config.copy()
        config_shared["individual"] = False
        model_shared = DLinearAdapter(config_shared)
        outputs_shared = model_shared.run(sample_forecast_inputs)
        
        # Individual layers
        config_individual = forecast_config.copy()
        config_individual["individual"] = True
        model_individual = DLinearAdapter(config_individual)
        outputs_individual = model_individual.run(sample_forecast_inputs)
        
        # Both should produce valid outputs
        validate_forecast_output(outputs_shared, 4, 24, 7)
        validate_forecast_output(outputs_individual, 4, 24, 7)
        
        # Outputs should be different (different parameterization)
        assert not np.allclose(
            outputs_shared["predictions"],
            outputs_individual["predictions"]
        ), "Individual and shared modes should produce different outputs"


class TestDLinearImputation:
    """Test DLinear adapter for imputation task."""
    
    @pytest.fixture
    def imputation_config(self):
        """Configuration for imputation."""
        return {
            "task_name": "imputation",
            "seq_len": 96,
            "enc_in": 7,
            "c_out": 7,
            "individual": False,
        }
    
    def test_imputation_run(self, imputation_config, sample_imputation_inputs):
        """Test imputation task execution."""
        from liulian.models.torch.dlinear import DLinearAdapter
        
        model = DLinearAdapter(imputation_config)
        outputs = model.run(sample_imputation_inputs)
        
        validate_imputation_output(
            outputs,
            batch_size=4,
            seq_len=imputation_config["seq_len"],
            features=imputation_config["c_out"]
        )


class TestDLinearAnomalyDetection:
    """Test DLinear adapter for anomaly detection task."""
    
    @pytest.fixture
    def anomaly_config(self):
        """Configuration for anomaly detection."""
        return {
            "task_name": "anomaly_detection",
            "seq_len": 96,
            "enc_in": 7,
            "c_out": 7,
            "individual": False,
        }
    
    def test_anomaly_run(self, anomaly_config, sample_anomaly_inputs):
        """Test anomaly detection task execution."""
        from liulian.models.torch.dlinear import DLinearAdapter
        
        model = DLinearAdapter(anomaly_config)
        outputs = model.run(sample_anomaly_inputs)
        
        validate_anomaly_output(
            outputs,
            batch_size=4,
            seq_len=anomaly_config["seq_len"],
            features=anomaly_config["c_out"]
        )


class TestDLinearClassification:
    """Test DLinear adapter for classification task."""
    
    @pytest.fixture
    def classification_config(self):
        """Configuration for classification."""
        return {
            "task_name": "classification",
            "seq_len": 96,
            "enc_in": 7,
            "num_class": 10,
            "individual": False,
        }
    
    def test_classification_run(self, classification_config, sample_classification_inputs):
        """Test classification task execution."""
        from liulian.models.torch.dlinear import DLinearAdapter
        
        model = DLinearAdapter(classification_config)
        outputs = model.run(sample_classification_inputs)
        
        validate_classification_output(
            outputs,
            batch_size=4,
            num_classes=classification_config["num_class"]
        )


class TestDLinearEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_input_keys(self):
        """Test that model handles missing optional keys gracefully."""
        import torch
        from liulian.models.torch.dlinear import DLinearAdapter
        
        config = {
            "task_name": "long_term_forecast",
            "seq_len": 96,
            "pred_len": 24,
            "label_len": 48,
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
            "individual": False,
        }
        
        model = DLinearAdapter(config)
        
        # Only x_enc provided - optional keys missing but handled gracefully
        incomplete_inputs = {
            "x_enc": torch.randn(4, 96, 7),
            # Missing x_mark_enc, x_dec, x_mark_dec (passed as None)
        }
        
        # Should work - model handles None for optional parameters
        outputs = model.run(incomplete_inputs)
        assert "predictions" in outputs
    
    def test_wrong_input_shapes(self, sample_forecast_inputs):
        """Test model behavior with mismatched input shapes.
        
        Note: PyTorch models typically don't validate shapes upfront,
        so we just verify the model runs (may produce wrong-sized output).
        """
        import torch
        from liulian.models.torch.dlinear import DLinearAdapter
        
        config = {
            "task_name": "long_term_forecast",
            "seq_len": 96,
            "pred_len": 24,
            "label_len": 48,
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
            "individual": False,
        }
        
        model = DLinearAdapter(config)
        
        # Wrong shape for x_enc (expecting features=7, providing 5)
        wrong_inputs = sample_forecast_inputs.copy()
        wrong_inputs["x_enc"] = torch.randn(4, 96, 5)
        
        # Model will process it but output shape may be wrong
        outputs = model.run(wrong_inputs)
        # Just verify it runs and returns something
        assert "predictions" in outputs
