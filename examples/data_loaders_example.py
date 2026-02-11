"""
Example: Using PyTorch Data Loaders

Demonstrates how to use the torch-native data loaders for time series forecasting.

All loaders return pure PyTorch tensors - no numpy conversion required.
"""
import torch
from liulian.data.data_factory import create_dataloader, create_dataloaders


def example_ett_dataset():
    """Example: Loading ETT hourly dataset."""
    print("\n=== Example 1: ETT Hour Dataset ===\n")
    
    # Create a DataLoader for ETTh1 dataset
    train_loader = create_dataloader(
        data_name='ETTh1',
        root_path='./data/ETT',
        data_path='ETTh1.csv',
        flag='train',
        size=(96, 48, 96),  # (seq_len, label_len, pred_len)
        features='M',  # Multivariate
        scale=True,
        batch_size=32,
        shuffle=True,
    )
    
    # Iterate through batches
    for batch_idx, batch in enumerate(train_loader):
        seq_x, seq_y, seq_x_mark, seq_y_mark = batch
        
        print(f"Batch {batch_idx}:")
        print(f"  seq_x shape: {seq_x.shape}")  # [batch_size, seq_len, features]
        print(f"  seq_y shape: {seq_y.shape}")  # [batch_size, label_len+pred_len, features]
        print(f"  seq_x dtype: {seq_x.dtype}")  # torch.float32
        print(f"  All torch tensors: {isinstance(seq_x, torch.Tensor)}")
        
        # Only show first batch
        break


def example_all_splits():
    """Example: Creating all splits at once."""
    print("\n=== Example 2: All Splits (Train/Val/Test) ===\n")
    
    # Create all three splits at once
    loaders = create_dataloaders(
        data_name='ETTm1',
        root_path='./data/ETT',
        data_path='ETTm1.csv',
        size=(96, 48, 96),
        features='M',
        batch_size=32,
    )
    
    # Access each split
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")


def example_custom_csv():
    """Example: Loading custom CSV dataset."""
    print("\n=== Example 3: Custom CSV Dataset ===\n")
    
    # Load custom CSV with any columns
    loader = create_dataloader(
        data_name='custom',
        root_path='./data',
        data_path='my_timeseries.csv',
        flag='train',
        size=(96, 48, 96),
        features='S',  # Univariate (single target)
        target='value',  # Your target column name
        scale=True,
        batch_size=16,
        train_ratio=0.8,  # Custom split ratio
        test_ratio=0.1,
    )
    
    # Get a batch
    batch = next(iter(loader))
    seq_x, seq_y, seq_x_mark, seq_y_mark = batch
    
    print(f"Custom CSV loaded:")
    print(f"  Input shape: {seq_x.shape}")
    print(f"  Target shape: {seq_y.shape}")


def example_univariate():
    """Example: Univariate forecasting mode."""
    print("\n=== Example 4: Univariate Mode ===\n")
    
    loader = create_dataloader(
        data_name='ETTh1',
        root_path='./data/ETT',
        data_path='ETTh1.csv',
        flag='train',
        features='S',  # Univariate: only target column
        target='OT',  # Target column name
        batch_size=32,
    )
    
    batch = next(iter(loader))
    seq_x, seq_y, seq_x_mark, seq_y_mark = batch
    
    print(f"Univariate mode:")
    print(f"  Features dimension: {seq_x.shape[-1]}")  # Should be 1


def example_with_model():
    """Example: Using with PyTorch model adapter."""
    print("\n=== Example 5: Integration with Model ===\n")
    
    # Create data loader
    loader = create_dataloader(
        data_name='ETTh1',
        root_path='./data/ETT',
        data_path='ETTh1.csv',
        flag='test',
        size=(96, 48, 96),
        batch_size=32,
        shuffle=False,
    )
    
    # Simulate model adapter usage
    for batch in loader:
        seq_x, seq_y, seq_x_mark, seq_y_mark = batch
        
        # Convert to model input format
        model_input = {
            'x_enc': seq_x,  # Already torch.Tensor!
            'x_mark_enc': seq_x_mark,
            'x_dec': seq_y[:, :48, :],  # First label_len steps
            'x_mark_dec': seq_y_mark,
        }
        
        print("Model input ready:")
        print(f"  x_enc: {model_input['x_enc'].shape}")
        print(f"  All torch tensors - NO numpy conversion needed!")
        
        # model_output = model_adapter.forward(model_input)
        break


if __name__ == '__main__':
    """
    Run all examples.
    
    Note: These examples assume you have ETT datasets downloaded.
    Download from: https://github.com/thuml/Time-Series-Library
    """
    print("=" * 60)
    print("PyTorch Data Loaders - Usage Examples")
    print("=" * 60)
    
    # Uncomment to run examples (requires actual data files)
    # example_ett_dataset()
    # example_all_splits()
    # example_custom_csv()
    # example_univariate()
    # example_with_model()
    
    print("\n" + "=" * 60)
    print("Key Features:")
    print("  ✓ Pure PyTorch tensors (no numpy conversion)")
    print("  ✓ StandardScaler normalization")
    print("  ✓ Train/Val/Test splits")
    print("  ✓ Categorical & time_features encoding")
    print("  ✓ Multivariate & Univariate modes")
    print("  ✓ Custom CSV support")
    print("  ✓ Factory pattern for easy creation")
    print("=" * 60)
