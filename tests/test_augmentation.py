"""
Tests for data augmentation module.
"""

import pytest

try:
    import torch
except ImportError:
    pytest.skip('torch not installed', allow_module_level=True)

import numpy as np

from liulian.utils.augmentation import (
    jitter,
    scaling,
    rotation,
    permutation,
    magnitude_warp,
    time_warp,
    window_slice,
    window_warp,
    apply_augmentations,
    random_augmentation,
)


@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    batch_size = 8
    seq_len = 96
    n_features = 7
    
    # Create deterministic data for reproducibility
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, n_features)
    
    return x


def test_jitter_shape(sample_data):
    """Test jitter preserves shape."""
    x_aug = jitter(sample_data, sigma=0.05)
    
    assert x_aug.shape == sample_data.shape
    assert x_aug.dtype == sample_data.dtype
    assert x_aug.device == sample_data.device


def test_jitter_adds_noise(sample_data):
    """Test jitter actually adds noise."""
    torch.manual_seed(42)
    x_aug = jitter(sample_data.clone(), sigma=0.05)
    
    # Should be different from original
    assert not torch.allclose(x_aug, sample_data)
    
    # Difference should be reasonable (most within 3 sigma, allow some outliers)
    diff = (x_aug - sample_data).abs()
    assert diff.mean() < 0.05  # Mean should be close to sigma
    assert diff.max() < 0.25  # Max allows for rare outliers (5 sigma)


def test_scaling_shape(sample_data):
    """Test scaling preserves shape."""
    x_aug = scaling(sample_data, sigma=0.1)
    
    assert x_aug.shape == sample_data.shape
    assert x_aug.dtype == sample_data.dtype


def test_scaling_changes_magnitude(sample_data):
    """Test scaling changes magnitude."""
    torch.manual_seed(42)
    x_aug = scaling(sample_data.clone(), sigma=0.2)
    
    # Should be different from original
    assert not torch.allclose(x_aug, sample_data)
    
    # Ratio should be around 1.0 on average
    ratio = (x_aug / (sample_data + 1e-8)).mean()
    assert 0.8 < ratio < 1.2


def test_rotation_shape(sample_data):
    """Test rotation preserves shape."""
    x_aug = rotation(sample_data)
    
    assert x_aug.shape == sample_data.shape
    assert x_aug.dtype == sample_data.dtype


def test_permutation_shape(sample_data):
    """Test permutation preserves shape."""
    x_aug = permutation(sample_data, max_segments=5, seg_mode="equal")
    
    assert x_aug.shape == sample_data.shape
    assert x_aug.dtype == sample_data.dtype


def test_permutation_equal_mode(sample_data):
    """Test equal-segment permutation."""
    torch.manual_seed(42)
    x_aug = permutation(sample_data.clone(), max_segments=4, seg_mode="equal")
    
    # Should be different order
    assert not torch.allclose(x_aug, sample_data)


def test_permutation_random_mode(sample_data):
    """Test random-segment permutation."""
    torch.manual_seed(42)
    x_aug = permutation(sample_data.clone(), max_segments=5, seg_mode="random")
    
    assert x_aug.shape == sample_data.shape


def test_magnitude_warp_shape(sample_data):
    """Test magnitude_warp preserves shape."""
    x_aug = magnitude_warp(sample_data, sigma=0.2, knot=4)
    
    assert x_aug.shape == sample_data.shape


def test_magnitude_warp_changes_amplitude(sample_data):
    """Test magnitude_warp changes amplitude."""
    torch.manual_seed(42)
    np.random.seed(42)
    x_aug = magnitude_warp(sample_data.clone(), sigma=0.3, knot=4)
    
    # Should be different from original
    assert not torch.allclose(x_aug, sample_data)


def test_time_warp_shape(sample_data):
    """Test time_warp preserves shape."""
    x_aug = time_warp(sample_data, sigma=0.2, knot=4)
    
    assert x_aug.shape == sample_data.shape


def test_window_slice_shape(sample_data):
    """Test window_slice preserves shape."""
    x_aug = window_slice(sample_data, reduce_ratio=0.9)
    
    assert x_aug.shape == sample_data.shape
    assert x_aug.dtype == sample_data.dtype


def test_window_slice_zooms_in(sample_data):
    """Test window_slice creates zoomed version."""
    torch.manual_seed(42)
    x_aug = window_slice(sample_data.clone(), reduce_ratio=0.5)
    
    # Should be different from original
    assert not torch.allclose(x_aug, sample_data)


def test_window_warp_shape(sample_data):
    """Test window_warp preserves shape."""
    x_aug = window_warp(sample_data, window_ratio=0.1, scales=(0.5, 2.0))
    
    assert x_aug.shape == sample_data.shape


def test_apply_augmentations_single(sample_data):
    """Test applying single augmentation."""
    torch.manual_seed(42)
    x_aug = apply_augmentations(sample_data.clone(), ["jitter"], jitter_sigma=0.05)
    
    assert x_aug.shape == sample_data.shape
    assert not torch.allclose(x_aug, sample_data)


def test_apply_augmentations_multiple(sample_data):
    """Test applying multiple augmentations."""
    torch.manual_seed(42)
    x_aug = apply_augmentations(
        sample_data.clone(),
        ["jitter", "scaling", "permutation"],
        jitter_sigma=0.05,
        scaling_sigma=0.1,
        permutation_segments=4
    )
    
    assert x_aug.shape == sample_data.shape
    assert not torch.allclose(x_aug, sample_data)


def test_apply_augmentations_all(sample_data):
    """Test applying all basic augmentations."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    aug_list = [
        "jitter",
        "scaling",
        "rotation",
        "permutation",
        "window_slice"
    ]
    
    x_aug = apply_augmentations(sample_data.clone(), aug_list)
    
    assert x_aug.shape == sample_data.shape
    assert not torch.allclose(x_aug, sample_data)


def test_random_augmentation(sample_data):
    """Test random augmentation selection."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    x_aug = random_augmentation(sample_data.clone(), num_augmentations=3)
    
    assert x_aug.shape == sample_data.shape
    assert not torch.allclose(x_aug, sample_data)


def test_random_augmentation_custom_list(sample_data):
    """Test random augmentation with custom list."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    available_augs = ["jitter", "scaling"]
    x_aug = random_augmentation(
        sample_data.clone(),
        num_augmentations=2,
        available_augs=available_augs
    )
    
    assert x_aug.shape == sample_data.shape


def test_augmentation_on_gpu():
    """Test augmentations work on GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    x = torch.randn(4, 96, 7, device='cuda')
    
    x_aug = jitter(x, sigma=0.05)
    assert x_aug.device.type == 'cuda'
    assert x_aug.shape == x.shape
    
    x_aug = scaling(x, sigma=0.1)
    assert x_aug.device.type == 'cuda'


def test_augmentation_with_single_sample():
    """Test augmentations work with single sample (batch_size=1)."""
    x = torch.randn(1, 96, 7)
    
    x_aug = jitter(x, sigma=0.05)
    assert x_aug.shape == (1, 96, 7)
    
    x_aug = permutation(x, max_segments=4)
    assert x_aug.shape == (1, 96, 7)


def test_augmentation_with_different_shapes():
    """Test augmentations work with different input shapes."""
    # Short sequence
    x_short = torch.randn(8, 24, 3)
    x_aug = apply_augmentations(x_short, ["jitter", "scaling"])
    assert x_aug.shape == x_short.shape
    
    # Long sequence
    x_long = torch.randn(8, 512, 10)
    x_aug = apply_augmentations(x_long, ["jitter", "permutation"])
    assert x_aug.shape == x_long.shape
    
    # Single feature
    x_univariate = torch.randn(8, 96, 1)
    x_aug = apply_augmentations(x_univariate, ["jitter", "window_slice"])
    assert x_aug.shape == x_univariate.shape


def test_window_slice_edge_cases():
    """Test window_slice with edge case ratios."""
    x = torch.randn(4, 96, 7)
    
    # Very small ratio
    x_aug = window_slice(x, reduce_ratio=0.3)
    assert x_aug.shape == x.shape
    
    # Ratio close to 1.0
    x_aug = window_slice(x, reduce_ratio=0.99)
    assert x_aug.shape == x.shape


def test_augmentation_reproducibility():
    """Test augmentations are reproducible with same seed."""
    x = torch.randn(4, 96, 7)
    
    # First run
    torch.manual_seed(123)
    np.random.seed(123)
    x_aug1 = apply_augmentations(x.clone(), ["jitter", "scaling", "permutation"])
    
    # Second run with same seed
    torch.manual_seed(123)
    np.random.seed(123)
    x_aug2 = apply_augmentations(x.clone(), ["jitter", "scaling", "permutation"])
    
    assert torch.allclose(x_aug1, x_aug2)


def test_augmentation_unknown_name(sample_data):
    """Test handling of unknown augmentation name."""
    import io
    import sys
    
    # Capture print output
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    x_aug = apply_augmentations(sample_data.clone(), ["unknown_aug", "jitter"])
    
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    
    assert "Warning: Unknown augmentation" in output
    assert x_aug.shape == sample_data.shape
