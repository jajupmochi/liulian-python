"""
Data factory for creating PyTorch DataLoaders with proper dataset selection.

Adapted from Time-Series-Library:
https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_factory.py

MIT License
"""
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from liulian.data.torch_datasets import (
    ETTHourDataset,
    ETTMinuteDataset,
    CustomCSVDataset,
)
from liulian.data.m4_dataset import M4Dataset


# Dataset registry mapping names to classes
DATASET_REGISTRY: Dict[str, type] = {
    'ETTh1': ETTHourDataset,
    'ETTh2': ETTHourDataset,
    'ETTm1': ETTMinuteDataset,
    'ETTm2': ETTMinuteDataset,
    'custom': CustomCSVDataset,
    'm4': M4Dataset,
}


def create_dataloader(
    data_name: str,
    root_path: str,
    data_path: str,
    flag: str = 'train',
    size: Optional[tuple] = None,
    features: str = 'M',
    target: str = 'OT',
    scale: bool = True,
    timeenc: int = 0,
    freq: str = 'h',
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create a PyTorch DataLoader for time series forecasting.
    
    Args:
        data_name: Dataset name (e.g., 'ETTh1', 'ETTm1', 'custom')
        root_path: Root directory containing the data file
        data_path: CSV filename
        flag: Split type - 'train', 'val', or 'test'
        size: Tuple of (seq_len, label_len, pred_len)
        features: Feature mode - 'M', 'S', or 'MS'
        target: Target column name
        scale: Whether to apply StandardScaler normalization
        timeenc: Time encoding mode (0 or 1)
        freq: Frequency string for time features
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle data (typically True for train, False for val/test)
        drop_last: Whether to drop the last incomplete batch
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        DataLoader configured for the specified dataset
        
    Raises:
        ValueError: If data_name is not recognized
        
    Examples:
        >>> train_loader = create_dataloader(
        ...     data_name='ETTh1',
        ...     root_path='./data/ETT',
        ...     data_path='ETTh1.csv',
        ...     flag='train',
        ...     size=(96, 48, 96),
        ...     batch_size=32,
        ...     shuffle=True
        ... )
        
        >>> val_loader = create_dataloader(
        ...     data_name='custom',
        ...     root_path='./data',
        ...     data_path='my_data.csv',
        ...     flag='val',
        ...     features='S',
        ...     target='value',
        ...     batch_size=32,
        ...     shuffle=False
        ... )
    """
    # Get dataset class from registry
    if data_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {data_name}. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    
    dataset_class = DATASET_REGISTRY[data_name]
    
    # Create dataset instance
    dataset = dataset_class(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=size,
        features=features,
        target=target,
        scale=scale,
        timeenc=timeenc,
        freq=freq,
        **kwargs
    )
    
    # Determine shuffle based on flag if not explicitly set
    if flag == 'train' and shuffle is None:
        shuffle = True
    elif flag in ['val', 'test'] and shuffle is None:
        shuffle = False
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    
    return dataloader


def create_dataloaders(
    data_name: str,
    root_path: str,
    data_path: str,
    size: Optional[tuple] = None,
    features: str = 'M',
    target: str = 'OT',
    scale: bool = True,
    timeenc: int = 0,
    freq: str = 'h',
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for a dataset.
    
    Convenience function to create all three splits at once with consistent settings.
    
    Args:
        data_name: Dataset name (e.g., 'ETTh1', 'ETTm1', 'custom')
        root_path: Root directory containing the data file
        data_path: CSV filename
        size: Tuple of (seq_len, label_len, pred_len)
        features: Feature mode - 'M', 'S', or 'MS'
        target: Target column name
        scale: Whether to apply StandardScaler normalization
        timeenc: Time encoding mode (0 or 1)
        freq: Frequency string for time features
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataLoaders
        
    Examples:
        >>> loaders = create_dataloaders(
        ...     data_name='ETTh1',
        ...     root_path='./data/ETT',
        ...     data_path='ETTh1.csv',
        ...     size=(96, 48, 96),
        ...     batch_size=32
        ... )
        >>> train_loader = loaders['train']
        >>> val_loader = loaders['val']
        >>> test_loader = loaders['test']
    """
    loaders = {}
    
    for flag in ['train', 'val', 'test']:
        # Train shuffles, val/test don't
        shuffle = (flag == 'train')
        
        loaders[flag] = create_dataloader(
            data_name=data_name,
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            size=size,
            features=features,
            target=target,
            scale=scale,
            timeenc=timeenc,
            freq=freq,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
            **kwargs
        )
    
    return loaders


def register_dataset(name: str, dataset_class: type):
    """
    Register a new dataset class in the registry.
    
    Allows extending the factory with custom dataset classes.
    
    Args:
        name: Name to register the dataset under
        dataset_class: Dataset class (must inherit from torch.utils.data.Dataset)
        
    Examples:
        >>> class MyCustomDataset(Dataset):
        ...     def __init__(self, root_path, data_path, flag, **kwargs):
        ...         pass
        >>> register_dataset('my_data', MyCustomDataset)
        >>> loader = create_dataloader('my_data', ...)
    """
    DATASET_REGISTRY[name] = dataset_class
