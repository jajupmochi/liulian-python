"""
Shared test fixtures and utilities for PyTorch model adapters.
"""

import pytest
from typing import Dict

try:
    import torch
except ImportError:
    pytest.skip("torch not installed", allow_module_level=True)


@pytest.fixture
def sample_forecast_inputs():
    """Generate sample inputs for forecasting task."""
    batch_size = 4
    seq_len = 96
    pred_len = 24
    label_len = 48
    features = 7
    
    return {
        "x_enc": torch.randn(batch_size, seq_len, features),
        "x_mark_enc": torch.randn(batch_size, seq_len, 4),
        "x_dec": torch.randn(batch_size, label_len + pred_len, features),
        "x_mark_dec": torch.randn(batch_size, label_len + pred_len, 4),
    }


@pytest.fixture
def sample_imputation_inputs():
    """Generate sample inputs for imputation task."""
    batch_size = 4
    seq_len = 96
    features = 7
    
    # Create random mask (0.2 missing rate)
    mask = torch.rand(batch_size, seq_len, features) > 0.2
    
    return {
        "x_enc": torch.randn(batch_size, seq_len, features),
        "x_mark_enc": torch.randn(batch_size, seq_len, 4),
        "mask": mask.float(),
    }


@pytest.fixture
def sample_anomaly_inputs():
    """Generate sample inputs for anomaly detection task."""
    batch_size = 4
    seq_len = 96
    features = 7
    
    return {
        "x_enc": torch.randn(batch_size, seq_len, features),
    }


@pytest.fixture
def sample_classification_inputs():
    """Generate sample inputs for classification task."""
    batch_size = 4
    seq_len = 96
    features = 7
    
    return {
        "x_enc": torch.randn(batch_size, seq_len, features),
        "x_mark_enc": torch.randn(batch_size, seq_len, 4),
    }


def check_torch_available():
    """Check if torch is available, skip if not."""
    try:
        import torch
        return True
    except ImportError:
        pytest.skip("torch not installed (requires pip install -e '.[torch-models]')")


def check_transformers_available():
    """Check if transformers is available, skip if not."""
    try:
        import transformers
        return True
    except ImportError:
        pytest.skip("transformers not installed (requires pip install -e '.[torch-models-full]')")


def validate_forecast_output(outputs: Dict, batch_size: int, pred_len: int, features: int):
    """Validate forecasting task output format."""
    assert "predictions" in outputs, "Output must contain 'predictions' key"
    predictions = outputs["predictions"]
    
    assert isinstance(predictions, torch.Tensor), "Predictions must be torch Tensor"
    assert predictions.shape == (batch_size, pred_len, features), \
        f"Expected shape ({batch_size}, {pred_len}, {features}), got {predictions.shape}"
    assert predictions.dtype in [torch.float32, torch.float64], \
        f"Expected float dtype, got {predictions.dtype}"
    assert not torch.isnan(predictions).any(), "Predictions contain NaN values"
    assert not torch.isinf(predictions).any(), "Predictions contain Inf values"


def validate_imputation_output(outputs: Dict, batch_size: int, seq_len: int, features: int):
    """Validate imputation task output format."""
    assert "predictions" in outputs, "Output must contain 'predictions' key"
    predictions = outputs["predictions"]
    
    assert isinstance(predictions, torch.Tensor), "Predictions must be torch Tensor"
    assert predictions.shape == (batch_size, seq_len, features), \
        f"Expected shape ({batch_size}, {seq_len}, {features}), got {predictions.shape}"
    assert predictions.dtype in [torch.float32, torch.float64], \
        f"Expected float dtype, got {predictions.dtype}"
    assert not torch.isnan(predictions).any(), "Predictions contain NaN values"


def validate_anomaly_output(outputs: Dict, batch_size: int, seq_len: int, features: int):
    """Validate anomaly detection task output format."""
    assert "predictions" in outputs, "Output must contain 'predictions' key"
    predictions = outputs["predictions"]
    
    assert isinstance(predictions, torch.Tensor), "Predictions must be torch Tensor"
    assert predictions.shape == (batch_size, seq_len, features), \
        f"Expected shape ({batch_size}, {seq_len}, {features}), got {predictions.shape}"
    assert predictions.dtype in [torch.float32, torch.float64], \
        f"Expected float dtype, got {predictions.dtype}"


def validate_classification_output(outputs: Dict, batch_size: int, num_classes: int):
    """Validate classification task output format."""
    assert "predictions" in outputs, "Output must contain 'predictions' key"
    predictions = outputs["predictions"]
    
    assert isinstance(predictions, torch.Tensor), "Predictions must be torch Tensor"
    assert predictions.shape == (batch_size, num_classes), \
        f"Expected shape ({batch_size}, {num_classes}), got {predictions.shape}"
    assert predictions.dtype in [torch.float32, torch.float64], \
        f"Expected float dtype, got {predictions.dtype}"
