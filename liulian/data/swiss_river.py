"""Swiss River dataset adapter for the liulian framework.

Wraps the Time-LLM reference project's Dataset_Swiss_1990 (ConcatDataset of
per-station SequenceWindowedDatasets) into liulian's BaseDataset / DataSplit
interface. Data loading is delegated to the reference project's data_provider.

Source data: refer_projects/Time-LLM_20260209_154911/dataset/swiss_river/
Data loader: refer_projects/Time-LLM_20260209_154911/data_provider/
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import DataLoader

from liulian.data.base import BaseDataset, DataSplit


# ---------------------------------------------------------------------------
# Locate and import reference project data_provider
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_TIMELLM_ROOT = os.path.join(
    _PROJECT_ROOT, 'refer_projects', 'Time-LLM_20260209_154911'
)


def _ensure_timellm_path() -> None:
    """Add Time-LLM reference project to sys.path if needed."""
    if _TIMELLM_ROOT not in sys.path:
        sys.path.insert(0, _TIMELLM_ROOT)


class SwissRiverDataset(BaseDataset):
    """Swiss River Network dataset using liulian's BaseDataset interface.

    Loads Swiss River station data via the Time-LLM reference project's
    data_provider, then materialises the PyTorch DataLoader batches into
    numpy arrays for the liulian DataSplit format.

    Attributes:
        domain: ``"hydrology"``
        version: ``"1.0"``

    Args:
        data_name: Dataset variant (``"swiss-river-1990"`` / ``"swiss-river-2010"``
            / ``"swiss-river-zurich"``).
        root_path: Path to the swiss_river data directory. If None, auto-detected
            from the reference project.
        seq_len: Input sequence length (default: 90).
        pred_len: Prediction horizon (default: 7).
        max_samples: Maximum samples to materialise per split (for quick tests).
            If None, all samples are loaded.

    Example::

        ds = SwissRiverDataset(data_name="swiss-river-1990", max_samples=1000)
        train_split = ds.get_split("train")
        X_batch, y_batch = train_split.get_batch(batch_size=32)
    """

    domain = 'hydrology'
    version = '1.0'

    def __init__(
        self,
        data_name: str = 'swiss-river-1990',
        root_path: Optional[str] = None,
        seq_len: int = 90,
        pred_len: int = 7,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.data_name = data_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.max_samples = max_samples

        # Auto-detect root_path
        if root_path is None:
            root_path = os.path.join(_TIMELLM_ROOT, 'dataset', 'swiss_river')
        self.root_path = root_path

        # Data path mapping
        self._data_path_map = {
            'swiss-river-1990': 'swiss-1990.csv',
            'swiss-river-2010': 'swiss-2010.csv',
            'swiss-river-zurich': 'zurich.csv',
        }

        # Cache loaded splits
        self._splits: Dict[str, DataSplit] = {}

    def _build_args(self) -> Any:
        """Build a simple namespace matching data_provider's expected args."""
        from types import SimpleNamespace

        return SimpleNamespace(
            data=self.data_name,
            root_path=self.root_path + '/',
            data_path=self._data_path_map.get(self.data_name, 'swiss-1990.csv'),
            features='M',
            target='OT',
            embed='timeF',
            freq='d',
            seq_len=self.seq_len,
            label_len=0,
            pred_len=self.pred_len,
            percent=100,
            batch_size=256,  # large batch for materialisation
            num_workers=0,
            seasonal_patterns='Monthly',
        )

    def _materialise_split(self, flag: str) -> DataSplit:
        """Load data via data_provider and convert to numpy arrays.

        Args:
            flag: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            DataSplit with X and y numpy arrays.
        """
        _ensure_timellm_path()

        # CWD must be Time-LLM root for relative paths in data_provider
        original_cwd = os.getcwd()
        os.chdir(_TIMELLM_ROOT)

        try:
            from data_provider.data_factory import data_provider

            args = self._build_args()
            dataset, loader = data_provider(args, flag)

            all_x, all_y = [], []
            count = 0
            for batch_x, batch_y, _, _ in loader:
                all_x.append(batch_x.numpy())
                all_y.append(batch_y.numpy())
                count += batch_x.shape[0]
                if self.max_samples is not None and count >= self.max_samples:
                    break

            X = np.concatenate(all_x, axis=0)
            y = np.concatenate(all_y, axis=0)

            if self.max_samples is not None:
                X = X[: self.max_samples]
                y = y[: self.max_samples]

            return DataSplit(X=X, y=y, name=flag)
        finally:
            os.chdir(original_cwd)

    def get_split(self, split_name: str) -> DataSplit:
        """Return the data split for the given partition.

        Args:
            split_name: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            A DataSplit with X [n, seq_len, features] and y [n, pred_len, features].

        Raises:
            KeyError: If split_name is not valid.
        """
        if split_name not in ('train', 'val', 'test'):
            raise KeyError(
                f"Unknown split: '{split_name}'. Use 'train', 'val', or 'test'."
            )

        if split_name not in self._splits:
            self._splits[split_name] = self._materialise_split(split_name)

        return self._splits[split_name]

    def info(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        return {
            'domain': self.domain,
            'version': self.version,
            'data_name': self.data_name,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'root_path': self.root_path,
        }

    def get_data_loaders(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> Dict[str, Any]:
        """Return PyTorch DataLoaders for train / val / test splits.

        This uses the Time-LLM reference project's ``data_provider`` to
        create proper ``DataLoader`` instances.  The result can be passed
        directly to :class:`~liulian.runtime.experiment.Experiment` via
        the ``data_loaders`` parameter.

        Args:
            batch_size: Batch size for all loaders.
            num_workers: Number of data-loading worker processes.

        Returns:
            Dict with ``"train"``, ``"val"``, and ``"test"``
            :class:`~torch.utils.data.DataLoader` instances.
        """
        _ensure_timellm_path()
        original_cwd = os.getcwd()
        os.chdir(_TIMELLM_ROOT)

        try:
            from data_provider.data_factory import data_provider

            args = self._build_args()
            args.batch_size = batch_size
            args.num_workers = num_workers

            _, train_loader = data_provider(args, 'train')
            _, val_loader = data_provider(args, 'val')
            _, test_loader = data_provider(args, 'test')

            return {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader,
            }
        finally:
            os.chdir(original_cwd)
