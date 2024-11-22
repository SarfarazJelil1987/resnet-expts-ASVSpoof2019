"""
ASVspoof2019 Dataset DataLoader

This module provides DataLoader functionality for the ASVspoof2019 dataset,
handling batch creation and data loading optimization for training and evaluation.

The module implements efficient data loading with features like:
- Automatic batching of data
- Multi-process data loading
- Memory pinning for faster GPU transfer
- Dynamic shuffling for training data
- Persistent workers for improved performance

Author: Sarfaraz Jelil
"""


from typing import Optional, Union
from torch.utils.data import DataLoader
from .dataset import ASVspoof2019Dataset

def get_dataloader(
    config: object,
    split: str = 'train'
) -> DataLoader:
    """
    Creates and returns a DataLoader for the ASVspoof2019 dataset.

    Args:
        config: Configuration object containing dataloader parameters
                Required attributes:
                - batch_size: int
                - num_workers: int
        split: Dataset split to use ('train', 'dev', or 'eval')
               Defaults to 'train'

    Returns:
        DataLoader: PyTorch DataLoader object configured for the specified split

    Raises:
        ValueError: If split is not one of 'train', 'dev', or 'eval'
        AttributeError: If config is missing required attributes
    """
    # Validate split parameter
    valid_splits = {'train', 'dev', 'eval'}
    if split not in valid_splits:
        raise ValueError(f"Split must be one of {valid_splits}, got {split}")

    # Validate config attributes
    required_attrs = {'batch_size', 'num_workers'}
    missing_attrs = required_attrs - set(dir(config))
    if missing_attrs:
        raise AttributeError(f"Config missing required attributes: {missing_attrs}")

    # Create dataset instance
    try:
        dataset = ASVspoof2019Dataset(config, split)
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset: {str(e)}")

    # Configure DataLoader with optimal settings
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train'),  # Shuffle only training data
        num_workers=config.num_workers,
        pin_memory=True,  # Speeds up data transfer to GPU
        drop_last=split == 'train',  # Drop incomplete batches during training
        persistent_workers=True if config.num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if config.num_workers > 0 else None,  # Prefetch next batches
    )

