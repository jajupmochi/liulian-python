"""
Tests for PyTorch data loaders and factory.

Tests verify:
- ETT datasets return torch tensors
- Custom CSV datasets work correctly
- Data factory creates proper DataLoaders
- Train/val/test splits are correct
- Tensor shapes match expectations
"""
import os
import tempfile
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    pytest.skip('torch not installed', allow_module_level=True)

import pandas as pd
import numpy as np

from liulian.data.torch_datasets import (
    ETTHourDataset,
    ETTMinuteDataset,
    CustomCSVDataset,
)
from liulian.data.data_factory import (
    create_dataloader,
    create_dataloaders,
    register_dataset,
)


@pytest.fixture
def ett_hour_data(tmp_path):
    """Create a synthetic ETT hour dataset."""
    # Generate enough hourly data for ETT splits (20 months = 14400 hours)
    # ETT splits: train(12 months), val(4 months), test(4 months)
    dates = pd.date_range('2020-01-01', periods=14400, freq='h')
    
    # 7 features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    data = {
        'date': dates,
        'HUFL': np.random.randn(14400),
        'HULL': np.random.randn(14400),
        'MUFL': np.random.randn(14400),
        'MULL': np.random.randn(14400),
        'LUFL': np.random.randn(14400),
        'LULL': np.random.randn(14400),
        'OT': np.random.randn(14400),
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = tmp_path / 'ETTh_test.csv'
    df.to_csv(csv_path, index=False)
    
    return str(tmp_path), 'ETTh_test.csv'


@pytest.fixture
def ett_minute_data(tmp_path):
    """Create a synthetic ETT minute dataset (15-min intervals)."""
    # Generate 10 days of 15-minute data (960 samples)
    dates = pd.date_range('2020-01-01', periods=960, freq='15min')
    
    # 7 features
    data = {
        'date': dates,
        'HUFL': np.random.randn(960),
        'HULL': np.random.randn(960),
        'MUFL': np.random.randn(960),
        'MULL': np.random.randn(960),
        'LUFL': np.random.randn(960),
        'LULL': np.random.randn(960),
        'OT': np.random.randn(960),
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = tmp_path / 'ETTm_test.csv'
    df.to_csv(csv_path, index=False)
    
    return str(tmp_path), 'ETTm_test.csv'


@pytest.fixture
def custom_csv_data(tmp_path):
    """Create a synthetic custom CSV dataset."""
    # Generate 60 days of hourly data (1440 samples) for 80/10/10 split
    dates = pd.date_range('2020-01-01', periods=1440, freq='h')
    
    data = {
        'date': dates,
        'feature1': np.random.randn(1440),
        'feature2': np.random.randn(1440),
        'feature3': np.random.randn(1440),
        'target_value': np.random.randn(1440),  # Custom target name
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = tmp_path / 'custom_test.csv'
    df.to_csv(csv_path, index=False)
    
    return str(tmp_path), 'custom_test.csv'


class TestETTHourDataset:
    """Tests for ETTHourDataset."""
    
    def test_basic_loading(self, ett_hour_data):
        """Test basic dataset loading."""
        root_path, data_path = ett_hour_data
        
        dataset = ETTHourDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(96, 48, 96),
            features='M',
        )
        
        # Check dataset length
        assert len(dataset) > 0
    
    def test_returns_torch_tensors(self, ett_hour_data):
        """Test that dataset returns torch tensors."""
        root_path, data_path = ett_hour_data
        
        dataset = ETTHourDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(96, 48, 96),
            features='M',
        )
        
        # Get a sample
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
        
        # Check all are torch tensors
        assert isinstance(seq_x, torch.Tensor)
        assert isinstance(seq_y, torch.Tensor)
        assert isinstance(seq_x_mark, torch.Tensor)
        assert isinstance(seq_y_mark, torch.Tensor)
    
    def test_tensor_shapes(self, ett_hour_data):
        """Test tensor shapes match specifications."""
        root_path, data_path = ett_hour_data
        
        seq_len, label_len, pred_len = 96, 48, 96
        dataset = ETTHourDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(seq_len, label_len, pred_len),
            features='M',
        )
        
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
        
        # Check shapes
        assert seq_x.shape[0] == seq_len
        assert seq_y.shape[0] == label_len + pred_len
        assert seq_x_mark.shape[0] == seq_len
        assert seq_y_mark.shape[0] == label_len + pred_len
        
        # Check features (7 columns in ETT data)
        assert seq_x.shape[1] == 7
        assert seq_y.shape[1] == 7
    
    def test_univariate_mode(self, ett_hour_data):
        """Test univariate (S) mode."""
        root_path, data_path = ett_hour_data
        
        dataset = ETTHourDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(96, 48, 96),
            features='S',
            target='OT',
        )
        
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
        
        # Should only have 1 feature
        assert seq_x.shape[1] == 1
        assert seq_y.shape[1] == 1
    
    def test_inverse_transform(self, ett_hour_data):
        """Test inverse transformation."""
        root_path, data_path = ett_hour_data
        
        dataset = ETTHourDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(96, 48, 96),
            scale=True,
        )
        
        seq_x, _, _, _ = dataset[0]
        
        # Inverse transform should work
        original = dataset.inverse_transform(seq_x)
        assert isinstance(original, torch.Tensor)
        assert original.shape == seq_x.shape
    
    def test_train_val_test_splits(self, ett_hour_data):
        """Test that train/val/test splits are different."""
        root_path, data_path = ett_hour_data
        
        train_dataset = ETTHourDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(96, 48, 96),
        )
        
        val_dataset = ETTHourDataset(
            root_path=root_path,
            data_path=data_path,
            flag='val',
            size=(96, 48, 96),
        )
        
        test_dataset = ETTHourDataset(
            root_path=root_path,
            data_path=data_path,
            flag='test',
            size=(96, 48, 96),
        )
        
        # All should have data
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        assert len(test_dataset) > 0


class TestETTMinuteDataset:
    """Tests for ETTMinuteDataset."""
    
    def test_basic_loading(self, ett_minute_data):
        """Test basic dataset loading."""
        root_path, data_path = ett_minute_data
        
        dataset = ETTMinuteDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(96, 48, 96),
            features='M',
        )
        
        assert len(dataset) > 0
    
    def test_returns_torch_tensors(self, ett_minute_data):
        """Test that dataset returns torch tensors."""
        root_path, data_path = ett_minute_data
        
        dataset = ETTMinuteDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(96, 48, 96),
        )
        
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
        
        # Check all are torch tensors
        assert isinstance(seq_x, torch.Tensor)
        assert isinstance(seq_y, torch.Tensor)
        assert isinstance(seq_x_mark, torch.Tensor)
        assert isinstance(seq_y_mark, torch.Tensor)
    
    def test_timeenc_categorical(self, ett_minute_data):
        """Test categorical time encoding (includes minute)."""
        root_path, data_path = ett_minute_data
        
        dataset = ETTMinuteDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            timeenc=0,  # Categorical
        )
        
        _, _, seq_x_mark, _ = dataset[0]
        
        # Should have 5 time features for minute data: month, day, weekday, hour, minute
        assert seq_x_mark.shape[1] == 5


class TestCustomCSVDataset:
    """Tests for CustomCSVDataset."""
    
    def test_basic_loading(self, custom_csv_data):
        """Test basic dataset loading."""
        root_path, data_path = custom_csv_data
        
        dataset = CustomCSVDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(96, 48, 96),
            target='target_value',
        )
        
        assert len(dataset) > 0
    
    def test_returns_torch_tensors(self, custom_csv_data):
        """Test that dataset returns torch tensors."""
        root_path, data_path = custom_csv_data
        
        dataset = CustomCSVDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            target='target_value',
        )
        
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
        
        assert isinstance(seq_x, torch.Tensor)
        assert isinstance(seq_y, torch.Tensor)
        assert isinstance(seq_x_mark, torch.Tensor)
        assert isinstance(seq_y_mark, torch.Tensor)
    
    def test_custom_split_ratios(self, custom_csv_data):
        """Test custom train/val/test ratios."""
        root_path, data_path = custom_csv_data
        
        # 80/10/10 split
        train_dataset = CustomCSVDataset(
            root_path=root_path,
            data_path=data_path,
            flag='train',
            target='target_value',
            train_ratio=0.8,
            test_ratio=0.1,
        )
        
        val_dataset = CustomCSVDataset(
            root_path=root_path,
            data_path=data_path,
            flag='val',
            target='target_value',
            train_ratio=0.8,
            test_ratio=0.1,
        )
        
        test_dataset = CustomCSVDataset(
            root_path=root_path,
            data_path=data_path,
            flag='test',
            target='target_value',
            train_ratio=0.8,
            test_ratio=0.1,
        )
        
        # All should have data
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        assert len(test_dataset) > 0


class TestDataFactory:
    """Tests for data factory functions."""
    
    def test_create_dataloader_ett(self, ett_hour_data):
        """Test creating DataLoader via factory."""
        root_path, data_path = ett_hour_data
        
        loader = create_dataloader(
            data_name='ETTh1',
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(96, 48, 96),
            batch_size=16,
        )
        
        # Check loader is created
        assert loader is not None
        
        # Get a batch
        batch = next(iter(loader))
        seq_x, seq_y, seq_x_mark, seq_y_mark = batch
        
        # Check batch dimensions
        assert seq_x.shape[0] == 16  # batch size
        assert seq_x.shape[1] == 96  # seq_len
    
    def test_create_dataloader_custom(self, custom_csv_data):
        """Test creating custom DataLoader via factory."""
        root_path, data_path = custom_csv_data
        
        loader = create_dataloader(
            data_name='custom',
            root_path=root_path,
            data_path=data_path,
            flag='train',
            target='target_value',
            batch_size=8,
        )
        
        assert loader is not None
        
        batch = next(iter(loader))
        seq_x, _, _, _ = batch
        assert seq_x.shape[0] == 8  # batch size
    
    def test_create_all_dataloaders(self, ett_hour_data):
        """Test creating all splits at once."""
        root_path, data_path = ett_hour_data
        
        loaders = create_dataloaders(
            data_name='ETTh1',
            root_path=root_path,
            data_path=data_path,
            size=(96, 48, 96),
            batch_size=16,
        )
        
        # Should have all three splits
        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders
        
        # All should be DataLoaders
        for loader in loaders.values():
            assert loader is not None
    
    def test_unknown_dataset_error(self):
        """Test error on unknown dataset name."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            create_dataloader(
                data_name='nonexistent_dataset',
                root_path='/tmp',
                data_path='data.csv',
            )
    
    def test_register_custom_dataset(self, custom_csv_data):
        """Test registering a custom dataset class."""
        root_path, data_path = custom_csv_data
        
        # Register under new name
        register_dataset('my_custom', CustomCSVDataset)
        
        # Should be able to create loader with new name
        loader = create_dataloader(
            data_name='my_custom',
            root_path=root_path,
            data_path=data_path,
            target='target_value',
            batch_size=4,
        )
        
        assert loader is not None


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_data_pipeline(self, ett_hour_data):
        """Test complete data loading pipeline."""
        root_path, data_path = ett_hour_data
        
        # Create DataLoader
        train_loader = create_dataloader(
            data_name='ETTh1',
            root_path=root_path,
            data_path=data_path,
            flag='train',
            size=(96, 48, 96),
            features='M',
            batch_size=16,
            shuffle=True,
        )
        
        # Iterate through a few batches
        for i, batch in enumerate(train_loader):
            seq_x, seq_y, seq_x_mark, seq_y_mark = batch
            
            # Verify tensors
            assert isinstance(seq_x, torch.Tensor)
            assert isinstance(seq_y, torch.Tensor)
            assert isinstance(seq_x_mark, torch.Tensor)
            assert isinstance(seq_y_mark, torch.Tensor)
            
            # Verify shapes
            assert seq_x.shape[0] <= 16  # batch size (may be smaller for last batch)
            assert seq_x.shape[1] == 96  # seq_len
            assert seq_y.shape[1] == 48 + 96  # label_len + pred_len
            
            # Only test a few batches
            if i >= 2:
                break
