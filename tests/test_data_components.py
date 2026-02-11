"""
Tests for data components: DatasetCustom, prompt_bank, seq_dataset.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# -----------------------------------------------------------------------
# DatasetCustom  (requires torch)
# -----------------------------------------------------------------------


@pytest.mark.skipif(not _torch_available(), reason='torch not installed')
class TestDatasetCustom:
    """Tests for DatasetCustom CSV loader."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a tiny CSV for testing."""
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='h')
        df = pd.DataFrame(
            {
                'date': dates,
                'feature_1': np.random.randn(n),
                'feature_2': np.random.randn(n),
                'OT': np.random.randn(n),
            }
        )
        path = tmp_path / 'test_data.csv'
        df.to_csv(path, index=False)
        return str(path)

    def test_instantiation(self, sample_csv):
        from liulian.data.dataset_custom import DatasetCustom

        ds = DatasetCustom(
            root_path=os.path.dirname(sample_csv),
            data_path=os.path.basename(sample_csv),
            flag='train',
            size=(48, 24, 12),
        )
        assert len(ds) > 0

    def test_getitem(self, sample_csv):
        from liulian.data.dataset_custom import DatasetCustom

        ds = DatasetCustom(
            root_path=os.path.dirname(sample_csv),
            data_path=os.path.basename(sample_csv),
            flag='train',
            size=(48, 24, 12),
        )
        seq_x, seq_y, seq_x_mark, seq_y_mark = ds[0]
        assert seq_x.shape[0] == 48  # seq_len
        assert seq_y.shape[0] == 24 + 12  # label_len + pred_len

    def test_splits(self, sample_csv):
        from liulian.data.dataset_custom import DatasetCustom

        dirname = os.path.dirname(sample_csv)
        basename = os.path.basename(sample_csv)

        train = DatasetCustom(
            root_path=dirname, data_path=basename, flag='train', size=(24, 12, 6)
        )
        val = DatasetCustom(
            root_path=dirname, data_path=basename, flag='val', size=(24, 12, 6)
        )
        test = DatasetCustom(
            root_path=dirname, data_path=basename, flag='test', size=(24, 12, 6)
        )

        assert len(train) > 0
        assert len(val) >= 0
        assert len(test) > 0

    def test_get_data_loaders(self, sample_csv):
        from liulian.data.dataset_custom import DatasetCustom

        ds = DatasetCustom(
            root_path=os.path.dirname(sample_csv),
            data_path=os.path.basename(sample_csv),
            flag='train',
            size=(24, 12, 6),
        )
        loaders = ds.get_data_loaders(batch_size=4)
        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders


# -----------------------------------------------------------------------
# prompt_bank
# -----------------------------------------------------------------------


class TestPromptBank:
    """Tests for the prompt_bank module."""

    def test_load_builtin(self):
        from liulian.data.prompt_bank import load_content

        content = load_content('ETT')
        assert isinstance(content, str)
        assert len(content) > 20

    def test_load_swiss_river(self):
        from liulian.data.prompt_bank import load_content

        content = load_content('swiss_river')
        assert 'river' in content.lower() or 'water' in content.lower()

    def test_load_weather(self):
        from liulian.data.prompt_bank import load_content

        content = load_content('Weather')
        assert isinstance(content, str)
        assert len(content) > 10

    def test_unknown_returns_generic(self):
        from liulian.data.prompt_bank import load_content

        content = load_content('nonexistent_dataset_xyz')
        assert isinstance(content, str)
        assert len(content) > 0

    def test_load_from_txt_file(self, tmp_path):
        from liulian.data.prompt_bank import load_content

        (tmp_path / 'custom_ds.txt').write_text('Custom data prompt.')
        content = load_content('custom_ds', prompt_dir=str(tmp_path))
        assert content == 'Custom data prompt.'


# -----------------------------------------------------------------------
# seq_dataset
# -----------------------------------------------------------------------


@pytest.mark.skipif(not _torch_available(), reason='torch not installed')
class TestSequenceDatasets:
    """Tests for gap-aware sequence datasets."""

    @pytest.fixture
    def contiguous_df(self):
        """DataFrame with a single contiguous block of 100 days."""
        return pd.DataFrame(
            {
                'epoch_day': np.arange(100),
                'x1': np.random.randn(100),
                'y1': np.random.randn(100),
            }
        )

    @pytest.fixture
    def gapped_df(self):
        """DataFrame with two contiguous blocks separated by a gap."""
        days_a = np.arange(0, 50)
        days_b = np.arange(60, 110)  # 10-day gap
        days = np.concatenate([days_a, days_b])
        return pd.DataFrame(
            {
                'epoch_day': days,
                'x1': np.random.randn(100),
                'y1': np.random.randn(100),
            }
        )

    def test_full_contiguous(self, contiguous_df):
        from liulian.data.seq_dataset import SequenceFullDataset

        ds = SequenceFullDataset(
            contiguous_df,
            feature_cols=['x1'],
            target_cols=['y1'],
        )
        assert len(ds) == 1  # single contiguous block
        t, x, y = ds[0]
        assert t.shape[0] == 100
        assert x.shape == (100, 1)
        assert y.shape == (100, 1)

    def test_full_gapped(self, gapped_df):
        from liulian.data.seq_dataset import SequenceFullDataset

        ds = SequenceFullDataset(
            gapped_df,
            feature_cols=['x1'],
            target_cols=['y1'],
        )
        assert len(ds) == 2  # two blocks

    def test_windowed_contiguous(self, contiguous_df):
        from liulian.data.seq_dataset import SequenceWindowedDataset

        ds = SequenceWindowedDataset(
            window_len=20,
            df=contiguous_df,
            feature_cols=['x1'],
            target_cols=['y1'],
        )
        assert len(ds) == 100 - 20 + 1  # 81 windows
        t, x, y = ds[0]
        assert t.shape[0] == 20

    def test_windowed_gapped(self, gapped_df):
        from liulian.data.seq_dataset import SequenceWindowedDataset

        ds = SequenceWindowedDataset(
            window_len=20,
            df=gapped_df,
            feature_cols=['x1'],
            target_cols=['y1'],
        )
        # Two sequences of length 50 each → 31 windows each = 62 total
        assert len(ds) == 2 * (50 - 20 + 1)

    def test_windowed_dev_run(self, contiguous_df):
        from liulian.data.seq_dataset import SequenceWindowedDataset

        ds = SequenceWindowedDataset(
            window_len=10,
            df=contiguous_df,
            feature_cols=['x1'],
            target_cols=['y1'],
            dev_run=True,
        )
        assert len(ds) == 10

    def test_noise_injection(self, contiguous_df):
        from liulian.data.seq_dataset import SequenceFullDataset

        ds = SequenceFullDataset(
            contiguous_df,
            feature_cols=['x1'],
            target_cols=['y1'],
        )
        original_vals = ds.df['x1'].values.copy()
        ds.add_noise('gaussian', {'std': 1.0})
        assert not np.allclose(ds.df['x1'].values, original_vals)

    def test_add_noise_to_array(self):
        from liulian.data.seq_dataset import add_noise_to_array

        arr = np.zeros(100, dtype=np.float32)
        noisy = add_noise_to_array(arr, 'gaussian', {'std': 0.5})
        assert noisy.shape == arr.shape
        assert not np.allclose(noisy, arr)

        quantized = add_noise_to_array(
            np.linspace(0, 1, 100), 'quantization', {'levels': 10}
        )
        assert quantized.shape == (100,)

    def test_short_subsequence_drop(self):
        """Short windows are dropped when method='drop'."""
        from liulian.data.seq_dataset import SequenceWindowedDataset

        # Only 5 rows → window_len=10 → should be dropped
        df = pd.DataFrame(
            {
                'epoch_day': np.arange(5),
                'x1': np.random.randn(5),
            }
        )
        ds = SequenceWindowedDataset(
            window_len=10,
            df=df,
            feature_cols=['x1'],
            short_subsequence_method='drop',
        )
        assert len(ds) == 0
