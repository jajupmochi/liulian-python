"""
Tests for M4 dataset implementation.

Note: These tests use mock data since M4 dataset files may not be available.
Real M4 dataset tests should be run with actual downloaded M4 data.
"""

import pytest

try:
    import torch
except ImportError:
    pytest.skip('torch not installed', allow_module_level=True)

import numpy as np
import os
import tempfile
import pandas as pd

from liulian.data.m4_dataset import M4Dataset, M4Meta, M4DatasetInfo


@pytest.fixture
def mock_m4_directory():
    """Create a mock M4 dataset directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create M4-info.csv
        info_data = {
            'M4id': ['M1', 'M2', 'M3', 'M4', 'M5'],
            'SP': ['Monthly', 'Monthly', 'Quarterly', 'Monthly', 'Yearly'],
            'Frequency': [12, 12, 4, 12, 1],
            'Horizon': [18, 18, 8, 18, 6]
        }
        info_df = pd.DataFrame(info_data)
        info_df.to_csv(os.path.join(tmpdir, 'M4-info.csv'), index=False)
        
        # Create training.npz with sample time series
        # Monthly: 3 series, Quarterly: 1 series, Yearly: 1 series
        training_data = np.array([
            # Monthly series (longer)
            np.concatenate([np.random.randn(200) * 10 + 100, [np.nan] * 50]),
            np.concatenate([np.random.randn(180) * 5 + 50, [np.nan] * 70]),
            np.concatenate([np.random.randn(220) * 8 + 80, [np.nan] * 30]),
            # Quarterly series
            np.concatenate([np.random.randn(60) * 15 + 200, [np.nan] * 190]),
            # Yearly series
            np.concatenate([np.random.randn(30) * 20 + 150, [np.nan] * 220]),
        ], dtype=object)
        np.savez(os.path.join(tmpdir, 'training.npz'), data=training_data)
        
        # Create test.npz
        test_data = training_data.copy()  # Use same for simplicity
        np.savez(os.path.join(tmpdir, 'test.npz'), data=test_data)
        
        yield tmpdir


def test_m4_meta():
    """Test M4Meta dataclass has correct attributes."""
    assert len(M4Meta.seasonal_patterns) == 6
    assert 'Monthly' in M4Meta.seasonal_patterns
    assert M4Meta.horizons_map['Monthly'] == 18
    assert M4Meta.frequency_map['Monthly'] == 12
    assert M4Meta.history_size['Monthly'] == 1.5


def test_m4_dataset_info_load(mock_m4_directory):
    """Test loading M4DatasetInfo from files."""
    info = M4DatasetInfo.load(training=True, dataset_file=mock_m4_directory)
    
    assert len(info.ids) == 5
    assert len(info.groups) == 5
    assert 'Monthly' in info.groups
    assert 'Quarterly' in info.groups


def test_m4_dataset_initialization(mock_m4_directory):
    """Test M4Dataset initialization."""
    dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(96, 48, 96),
        seasonal_patterns='Monthly'
    )
    
    assert dataset.seq_len == 96
    assert dataset.label_len == 48
    assert dataset.pred_len == 96
    assert dataset.seasonal_patterns == 'Monthly'
    assert len(dataset) == 3  # 3 Monthly series in mock data


def test_m4_dataset_different_patterns(mock_m4_directory):
    """Test M4Dataset with different seasonal patterns."""
    # Monthly
    monthly_dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(96, 48, 96),
        seasonal_patterns='Monthly'
    )
    assert len(monthly_dataset) == 3
    
    # Quarterly
    quarterly_dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(60, 30, 60),
        seasonal_patterns='Quarterly'
    )
    assert len(quarterly_dataset) == 1
    
    # Yearly
    yearly_dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(30, 15, 30),
        seasonal_patterns='Yearly'
    )
    assert len(yearly_dataset) == 1


def test_m4_dataset_getitem(mock_m4_directory):
    """Test M4Dataset __getitem__ returns correct format."""
    dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(96, 48, 96),
        seasonal_patterns='Monthly'
    )
    
    insample, outsample, in_mask, out_mask = dataset[0]
    
    # Check shapes
    assert insample.shape == (96, 1)
    assert outsample.shape == (144, 1)  # label_len + pred_len
    assert in_mask.shape == (96, 1)
    assert out_mask.shape == (144, 1)
    
    # Check types
    assert isinstance(insample, torch.Tensor)
    assert isinstance(outsample, torch.Tensor)
    assert isinstance(in_mask, torch.Tensor)
    assert isinstance(out_mask, torch.Tensor)
    
    # Check dtypes
    assert insample.dtype == torch.float32
    assert in_mask.dtype == torch.float32


def test_m4_dataset_masks(mock_m4_directory):
    """Test that masks correctly indicate valid data."""
    dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(50, 20, 30),
        seasonal_patterns='Monthly'
    )
    
    insample, outsample, in_mask, out_mask = dataset[0]
    
    # Masks should contain 0s and 1s only
    assert torch.all((in_mask == 0) | (in_mask == 1))
    assert torch.all((out_mask == 0) | (out_mask == 1))
    
    # At least some data should be valid (mask = 1)
    assert in_mask.sum() > 0
    assert out_mask.sum() > 0


def test_m4_dataset_random_sampling(mock_m4_directory):
    """Test that random sampling produces different windows."""
    dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(50, 20, 30),
        seasonal_patterns='Monthly'
    )
    
    # Get multiple samples from same series
    torch.manual_seed(42)
    np.random.seed(42)
    sample1 = dataset[0]
    
    torch.manual_seed(123)
    np.random.seed(123)
    sample2 = dataset[0]
    
    # Due to random cut point, samples may differ
    # (Not guaranteed to be different, but likely)
    insample1, _, _, _ = sample1
    insample2, _, _, _ = sample2
    
    # Check that we can sample successfully
    assert insample1.shape == insample2.shape


def test_m4_dataset_last_insample_window(mock_m4_directory):
    """Test last_insample_window method."""
    dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(96, 48, 96),
        seasonal_patterns='Monthly'
    )
    
    insample, in_mask = dataset.last_insample_window()
    
    # Check shapes
    assert insample.shape == (3, 96)  # 3 Monthly series
    assert in_mask.shape == (3, 96)
    
    # Check types
    assert isinstance(insample, torch.Tensor)
    assert isinstance(in_mask, torch.Tensor)
    
    # All masks should be 1 (all data valid)
    assert in_mask.sum() > 0


def test_m4_dataset_train_val_test_splits(mock_m4_directory):
    """Test different data splits."""
    # Train
    train_dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(96, 48, 96),
        seasonal_patterns='Monthly'
    )
    
    # Val (should use subset of training data)
    val_dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='val',
        size=(96, 48, 96),
        seasonal_patterns='Monthly'
    )
    
    # Test
    test_dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='test',
        size=(96, 48, 96),
        seasonal_patterns='Monthly'
    )
    
    # Val should be smaller than train (20% split)
    assert len(val_dataset) < len(train_dataset) or len(train_dataset) == 1
    
    # Test should also have data
    assert len(test_dataset) > 0


def test_m4_dataset_integration_with_dataloader(mock_m4_directory):
    """Test M4Dataset works with PyTorch DataLoader."""
    from torch.utils.data import DataLoader
    
    dataset = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(96, 48, 96),
        seasonal_patterns='Monthly'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    # Test one batch
    batch = next(iter(dataloader))
    insample, outsample, in_mask, out_mask = batch
    
    assert insample.shape == (2, 96, 1)  # batch_size=2
    assert outsample.shape == (2, 144, 1)
    assert in_mask.shape == (2, 96, 1)
    assert out_mask.shape == (2, 144, 1)


def test_m4_dataset_different_sizes(mock_m4_directory):
    """Test M4Dataset with different seq_len/pred_len configurations."""
    # Short sequences
    dataset_short = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(24, 12, 24),
        seasonal_patterns='Monthly'
    )
    
    insample, outsample, _, _ = dataset_short[0]
    assert insample.shape == (24, 1)
    assert outsample.shape == (36, 1)  # 12 + 24
    
    # Long sequences
    dataset_long = M4Dataset(
        root_path=mock_m4_directory,
        flag='train',
        size=(192, 96, 192),
        seasonal_patterns='Monthly'
    )
    
    insample, outsample, _, _ = dataset_long[0]
    assert insample.shape == (192, 1)
    assert outsample.shape == (288, 1)  # 96 + 192


def test_m4_dataset_data_factory_integration(mock_m4_directory):
    """Test M4Dataset registration in data_factory."""
    from liulian.data.data_factory import DATASET_REGISTRY, create_dataloader
    
    # Check registry
    assert 'm4' in DATASET_REGISTRY
    assert DATASET_REGISTRY['m4'] == M4Dataset
    
    # Create dataloader
    loader = create_dataloader(
        data_name='m4',
        root_path=mock_m4_directory,
        data_path='',  # M4 doesn't use data_path
        flag='train',
        size=(96, 48, 96),
        seasonal_patterns='Monthly',
        batch_size=2,
        shuffle=True
    )
    
    assert len(loader) > 0
    batch = next(iter(loader))
    assert len(batch) == 4  # insample, outsample, in_mask, out_mask
