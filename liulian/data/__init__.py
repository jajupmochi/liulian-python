"""Data layer — dataset abstractions, manifest management, and topology specs."""

from liulian.data.base import BaseDataset, DataSplit
from liulian.data.manifest import load_manifest, validate_manifest
from liulian.data.spec import FieldSpec, TopologySpec
from liulian.data.prompt_bank import load_content

# Torch-dependent modules — import lazily to avoid hard dependency
try:
    from liulian.data.swiss_river import SwissRiverDataset
    from liulian.data.dataset_custom import DatasetCustom
    from liulian.data.seq_dataset import (
        SequenceDataset,
        SequenceFullDataset,
        SequenceWindowedDataset,
        add_noise_to_array,
    )
except ImportError:  # torch not installed
    pass

__all__ = [
    'BaseDataset',
    'DataSplit',
    'FieldSpec',
    'TopologySpec',
    'SwissRiverDataset',
    'DatasetCustom',
    'SequenceDataset',
    'SequenceFullDataset',
    'SequenceWindowedDataset',
    'add_noise_to_array',
    'load_manifest',
    'validate_manifest',
    'load_content',
]
