"""Tests for items 1–8 of the training-gap adaptation round.

Covers:
- Item 1: disable_early_stopping under ASHA
- Item 2: accelerator module
- Item 3: configurable scalers
- Item 4: swiss-river LSTM / Transformer models
- Item 5: NaN masking in loss
- Item 6: teacher forcing / decoder input
- Item 7: normalized + de-normalized metrics
- Item 8: HPO search spaces and ASHA presets
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import sklearn  # noqa: F401

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# =====================================================================
# Item 1 — disable_early_stopping
# =====================================================================


class TestDisableEarlyStopping:
    """Ensure the trainer respects disable_early_stopping."""

    @staticmethod
    def _make_trainer(**overrides: Any):
        from liulian.runtime.trainer import ForecastTrainer

        cfg: Dict[str, Any] = {
            'pred_len': 3,
            'train_epochs': 5,
            'patience': 1,
            'learning_rate': 0.01,
            'loss': 'mse',
            'show_progress': False,
        }
        cfg.update(overrides)
        return ForecastTrainer(config=cfg)

    @staticmethod
    def _tiny_model(n_in: int = 4, pred_len: int = 3):
        """Return a minimal model with the 4-arg forward signature."""

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(n_in, pred_len)

            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                return self.fc(x_enc[:, -1, :]).unsqueeze(-1).expand(-1, pred_len, n_in)

        return M()

    @staticmethod
    def _make_loader(n: int = 20, seq: int = 10, feat: int = 4, pred: int = 3):
        """Create a trivial DataLoader."""
        x = torch.randn(n, seq, feat)
        y = torch.randn(n, pred, feat)
        xm = torch.zeros(n, seq, 1)
        ym = torch.zeros(n, pred, 1)
        return DataLoader(TensorDataset(x, y, xm, ym), batch_size=4)

    def test_early_stopping_fires(self):
        """With patience=1 and high LR, val loss oscillates → early stop."""
        trainer = self._make_trainer(patience=1, train_epochs=100, learning_rate=0.5)
        model = self._tiny_model()
        summary = trainer.fit(model, self._make_loader(), self._make_loader())
        assert summary['epochs_run'] < 100

    def test_disable_early_stopping(self):
        """With disable_early_stopping=True, all epochs must run."""
        trainer = self._make_trainer(
            patience=1,
            train_epochs=5,
            disable_early_stopping=True,
        )
        model = self._tiny_model()
        summary = trainer.fit(model, self._make_loader(), self._make_loader())
        assert summary['epochs_run'] == 5


# =====================================================================
# Item 2 — accelerator module
# =====================================================================


class TestAccelerator:
    """Verify the accelerator builder."""

    def test_returns_none_when_disabled(self):
        from liulian.runtime.accelerator import build_accelerator

        assert build_accelerator({'use_accelerator': False}) is None

    def test_returns_none_by_default(self):
        from liulian.runtime.accelerator import build_accelerator

        assert build_accelerator({}) is None

    def test_trainer_accelerator_default_none(self):
        from liulian.runtime.trainer import ForecastTrainer

        t = ForecastTrainer(config={'pred_len': 1, 'show_progress': False})
        assert t.accelerator is None


# =====================================================================
# Item 3 — configurable scalers
# =====================================================================


class TestScalers:
    """Test the scaler factory and implementations."""

    def test_no_scaler(self):
        from liulian.data.scalers import get_scaler

        s = get_scaler('none')
        x = np.array([[1, 2], [3, 4]], dtype=float)
        np.testing.assert_array_equal(s.fit_transform(x), x)
        np.testing.assert_array_equal(s.inverse_transform(x), x)

    @pytest.mark.skipif(
        not _HAS_SKLEARN,
        reason='scikit-learn not installed',
    )
    def test_standard_scaler(self):
        from liulian.data.scalers import get_scaler

        s = get_scaler('standard')
        x = np.random.randn(100, 3)
        s.fit(x)
        t = s.transform(x)
        assert abs(t.mean()) < 0.1
        inv = s.inverse_transform(t)
        np.testing.assert_allclose(inv, x, atol=1e-6)

    @pytest.mark.skipif(
        not _HAS_SKLEARN,
        reason='scikit-learn not installed',
    )
    def test_minmax_scaler(self):
        from liulian.data.scalers import get_scaler

        s = get_scaler('minmax')
        x = np.random.randn(100, 2)
        s.fit(x)
        t = s.transform(x)
        assert t.min() >= -0.01
        assert t.max() <= 1.01

    def test_unknown_raises(self):
        from liulian.data.scalers import get_scaler

        with pytest.raises(ValueError, match='Unknown scaler'):
            get_scaler('foobar')

    @pytest.mark.skipif(
        not _HAS_SKLEARN,
        reason='scikit-learn not installed',
    )
    def test_dim_split_scaler(self):
        from sklearn.preprocessing import MinMaxScaler as MMS

        from liulian.data.scalers import DimSplitScaler

        data = np.random.randn(50, 3)
        gs = MMS().fit(data)

        ds = DimSplitScaler(gs)
        col = data[:, 1:2]
        t = ds[1].transform(col)
        inv = ds[1].inverse_transform(t)
        np.testing.assert_allclose(inv, col, atol=1e-6)


# =====================================================================
# Item 4 — swiss-river LSTM / Transformer models
# =====================================================================


class TestSwissModels:
    """Smoke-test swiss-river model forward passes."""

    def test_swiss_lstm(self):
        from liulian.models.torch.swiss_lstm import SwissLstmModel

        m = SwissLstmModel(input_size=3, hidden_size=16, num_layers=1)
        x = torch.randn(2, 10, 3)
        out = m(x)
        assert out.shape == (2, 10, 1)
        assert torch.isfinite(out).all(), 'SwissLstm produced non-finite values'

    def test_extrapo_limo(self):
        from liulian.models.torch.swiss_lstm import ExtrapoLstmModelLIMO

        m = ExtrapoLstmModelLIMO(input_size=2, hidden_size=16, num_layers=1, future_steps=3)
        x = torch.randn(2, 13, 2)  # 10 + 3 future
        out = m(x)
        assert out.shape == (2, 3, 1)
        assert torch.isfinite(out).all(), 'ExtrapoLIMO produced non-finite values'

    def test_extrapo_fembed(self):
        from liulian.models.torch.swiss_lstm import ExtrapoLstmModelFEmbed

        m = ExtrapoLstmModelFEmbed(
            input_size=2,
            hidden_size=16,
            num_layers=1,
            future_steps=3,
            d_future_emb=8,
        )
        x = torch.randn(2, 13, 2)
        out = m(x)
        assert out.shape == (2, 3, 1)
        assert torch.isfinite(out).all(), 'ExtrapoFEmbed produced non-finite values'

    def test_lstm_embedding(self):
        from liulian.models.torch.swiss_lstm import LstmEmbeddingModel

        m = LstmEmbeddingModel(
            input_size=2,
            num_embeddings=10,
            embedding_size=4,
            hidden_size=16,
            num_layers=1,
        )
        e = torch.randint(0, 10, (2, 10))
        x = torch.randn(2, 10, 2)
        out = m(e, x)
        assert out.shape == (2, 10, 1)
        assert torch.isfinite(out).all(), 'LstmEmbedding produced non-finite values'

    def test_swiss_transformer_sinusoidal(self):
        from liulian.models.torch.swiss_transformer import SwissTransformerModel

        m = SwissTransformerModel(
            input_size=3,
            num_heads=2,
            num_layers=1,
            dim_feedforward=32,
            d_model=8,
            positional_encoding='sinusoidal',
        )
        x = torch.randn(2, 10, 3)
        out = m(x)
        assert out.shape == (2, 10, 1)
        assert torch.isfinite(out).all(), 'Transformer sinusoidal non-finite'

    def test_swiss_transformer_embedding(self):
        from liulian.models.torch.swiss_transformer import (
            SwissTransformerEmbeddingModel,
        )

        m = SwissTransformerEmbeddingModel(
            input_size=3,
            num_embeddings=5,
            embedding_size=4,
            num_heads=2,
            num_layers=1,
            dim_feedforward=32,
            d_model=8,
            positional_encoding='sinusoidal',
        )
        e = torch.randint(0, 5, (2, 10))
        x = torch.randn(2, 10, 3)
        out = m(e, x)
        assert out.shape == (2, 10, 1)
        assert torch.isfinite(out).all(), 'Transformer embedding non-finite'

    def test_swiss_lstm_adapter(self):
        from liulian.models.torch.swiss_lstm import SwissLSTMAdapter

        adapter = SwissLSTMAdapter({'enc_in': 3, 'd_model': 16, 'e_layers': 1})
        batch = {'x_enc': torch.randn(2, 10, 3)}
        out = adapter.forward(batch)
        assert 'predictions' in out
        assert out['predictions'].shape == (2, 10, 1)
        assert torch.isfinite(out['predictions']).all()

    def test_swiss_transformer_adapter(self):
        from liulian.models.torch.swiss_transformer import SwissTransformerAdapter

        adapter = SwissTransformerAdapter(
            {
                'enc_in': 3,
                'n_heads': 2,
                'e_layers': 1,
                'd_ff': 32,
                'd_model': 8,
                'positional_encoding': 'sinusoidal',
            }
        )
        batch = {'x_enc': torch.randn(2, 10, 3)}
        out = adapter.forward(batch)
        assert 'predictions' in out
        assert out['predictions'].shape == (2, 10, 1)
        assert torch.isfinite(out['predictions']).all()


# =====================================================================
# Generalized LSTM / Transformer — multi-channel & entity modes
# =====================================================================


class TestGeneralizedModels:
    """Test that LSTM and Transformer work with any dataset / entity mode."""

    # — Multi-channel output (traffic, ETT etc.) ——————————————————————

    def test_lstm_multi_channel_output(self):
        """LSTM with c_out=7 for ETT-style dataset."""
        from liulian.models.torch.swiss_lstm import SwissLstmModel

        m = SwissLstmModel(input_size=7, hidden_size=32, num_layers=1, c_out=7)
        x = torch.randn(2, 96, 7)
        out = m(x)
        assert out.shape == (2, 96, 7)
        assert torch.isfinite(out).all(), 'lstm_multi_channel produced non-finite values'

    def test_extrapo_limo_multi_channel(self):
        from liulian.models.torch.swiss_lstm import ExtrapoLstmModelLIMO

        m = ExtrapoLstmModelLIMO(
            input_size=7,
            hidden_size=32,
            num_layers=1,
            future_steps=24,
            c_out=7,
        )
        x = torch.randn(2, 96 + 24, 7)
        out = m(x)
        assert out.shape == (2, 24, 7)
        assert torch.isfinite(out).all(), 'extrapo_limo_multi produced non-finite values'

    def test_extrapo_fembed_multi_channel(self):
        from liulian.models.torch.swiss_lstm import ExtrapoLstmModelFEmbed

        m = ExtrapoLstmModelFEmbed(
            input_size=7,
            hidden_size=32,
            num_layers=1,
            future_steps=24,
            d_future_emb=16,
            c_out=7,
        )
        x = torch.randn(2, 96 + 24, 7)
        out = m(x)
        assert out.shape == (2, 24, 7)
        assert torch.isfinite(out).all(), 'extrapo_fembed_multi produced non-finite values'

    def test_lstm_embedding_multi_channel(self):
        from liulian.models.torch.swiss_lstm import LstmEmbeddingModel

        m = LstmEmbeddingModel(
            input_size=7,
            num_embeddings=20,
            embedding_size=8,
            hidden_size=32,
            num_layers=1,
            c_out=7,
        )
        e = torch.randint(0, 20, (2, 96))
        x = torch.randn(2, 96, 7)
        out = m(e, x)
        assert out.shape == (2, 96, 7)
        assert torch.isfinite(out).all(), 'lstm_embedding_multi produced non-finite values'

    def test_transformer_multi_channel(self):
        from liulian.models.torch.swiss_transformer import SwissTransformerModel

        m = SwissTransformerModel(
            input_size=7,
            num_heads=2,
            num_layers=1,
            dim_feedforward=32,
            d_model=8,
            positional_encoding='sinusoidal',
            c_out=7,
        )
        x = torch.randn(2, 96, 7)
        out = m(x)
        assert out.shape == (2, 96, 7)
        assert torch.isfinite(out).all(), 'transformer_multi produced non-finite values'

    def test_transformer_embedding_multi_channel(self):
        from liulian.models.torch.swiss_transformer import (
            SwissTransformerEmbeddingModel,
        )

        m = SwissTransformerEmbeddingModel(
            input_size=7,
            num_embeddings=10,
            embedding_size=4,
            num_heads=2,
            num_layers=1,
            dim_feedforward=32,
            d_model=8,
            positional_encoding='sinusoidal',
            c_out=7,
        )
        e = torch.randint(0, 10, (2, 96))
        x = torch.randn(2, 96, 7)
        out = m(e, x)
        assert out.shape == (2, 96, 7)
        assert torch.isfinite(out).all(), 'transformer_embedding_multi produced non-finite values'

    # — LSTMAdapter with different entity modes ——————————————————————

    def test_lstm_adapter_no_entity_traffic(self):
        """Traffic-like: 862 channels, no entities."""
        from liulian.models.torch.swiss_lstm import LSTMAdapter

        adapter = LSTMAdapter(
            {
                'enc_in': 10,
                'c_out': 10,
                'd_model': 16,
                'e_layers': 1,
            }
        )
        out = adapter.forward({'x_enc': torch.randn(2, 96, 10)})
        assert out['predictions'].shape == (2, 96, 10)
        assert torch.isfinite(out['predictions']).all(), 'lstm_adapter_no_entity produced non-finite'

    def test_lstm_adapter_embedding_mode(self):
        """Entity embedding mode — IDs from x_mark_enc."""
        from liulian.models.torch.swiss_lstm import LSTMAdapter

        adapter = LSTMAdapter(
            {
                'enc_in': 3,
                'c_out': 1,
                'd_model': 16,
                'e_layers': 1,
                'identifier_mode': 'embedding',
                'num_embeddings': 20,
                'embedding_size': 4,
            }
        )
        x_mark = torch.zeros(2, 10, 2)
        x_mark[:, :, 0] = torch.randint(0, 20, (2, 10)).float()
        out = adapter.forward(
            {
                'x_enc': torch.randn(2, 10, 3),
                'x_mark_enc': x_mark,
            }
        )
        assert out['predictions'].shape == (2, 10, 1)
        assert torch.isfinite(out['predictions']).all(), 'lstm_adapter_embedding produced non-finite'

    def test_lstm_adapter_feature_concat_mode(self):
        """feature_concat mode — separate entity_features tensor."""
        from liulian.models.torch.swiss_lstm import LSTMAdapter

        adapter = LSTMAdapter(
            {
                'enc_in': 3,
                'c_out': 1,
                'd_model': 16,
                'e_layers': 1,
                'identifier_mode': 'feature_concat',
                'entity_dim': 8,
            }
        )
        out = adapter.forward(
            {
                'x_enc': torch.randn(2, 10, 3),
                'entity_features': torch.randn(2, 10, 8),
            }
        )
        assert out['predictions'].shape == (2, 10, 1)
        assert torch.isfinite(out['predictions']).all(), 'lstm_adapter_feature_concat produced non-finite'

    def test_lstm_adapter_onehot_transparent(self):
        """One-hot entity features already in x_enc — transparent."""
        from liulian.models.torch.swiss_lstm import LSTMAdapter

        # 3 raw features + 5 one-hot = 8
        adapter = LSTMAdapter(
            {
                'enc_in': 8,
                'c_out': 1,
                'd_model': 16,
                'e_layers': 1,
                'identifier_mode': 'onehot',
            }
        )
        out = adapter.forward({'x_enc': torch.randn(2, 10, 8)})
        assert out['predictions'].shape == (2, 10, 1)
        assert torch.isfinite(out['predictions']).all(), 'lstm_adapter_onehot produced non-finite'

    # — TransformerEncoderAdapter with different entity modes ————————

    def test_transformer_adapter_multi_channel(self):
        """Multi-channel output for traffic/ETT."""
        from liulian.models.torch.swiss_transformer import TransformerEncoderAdapter

        adapter = TransformerEncoderAdapter(
            {
                'enc_in': 7,
                'c_out': 7,
                'n_heads': 2,
                'e_layers': 1,
                'd_ff': 32,
                'd_model': 8,
                'positional_encoding': 'sinusoidal',
            }
        )
        out = adapter.forward({'x_enc': torch.randn(2, 96, 7)})
        assert out['predictions'].shape == (2, 96, 7)
        assert torch.isfinite(out['predictions']).all(), 'transformer_adapter_multi produced non-finite'

    def test_transformer_adapter_embedding_mode(self):
        from liulian.models.torch.swiss_transformer import TransformerEncoderAdapter

        adapter = TransformerEncoderAdapter(
            {
                'enc_in': 3,
                'c_out': 1,
                'n_heads': 2,
                'e_layers': 1,
                'd_ff': 32,
                'd_model': 8,
                'identifier_mode': 'embedding',
                'num_embeddings': 20,
                'embedding_size': 4,
            }
        )
        x_mark = torch.zeros(2, 10, 2)
        x_mark[:, :, 0] = torch.randint(0, 20, (2, 10)).float()
        out = adapter.forward(
            {
                'x_enc': torch.randn(2, 10, 3),
                'x_mark_enc': x_mark,
            }
        )
        assert out['predictions'].shape == (2, 10, 1)
        assert torch.isfinite(out['predictions']).all(), 'transformer_adapter_embedding produced non-finite'

    def test_transformer_adapter_feature_concat(self):
        from liulian.models.torch.swiss_transformer import TransformerEncoderAdapter

        adapter = TransformerEncoderAdapter(
            {
                'enc_in': 3,
                'c_out': 1,
                'n_heads': 2,
                'e_layers': 1,
                'd_ff': 32,
                'd_model': 8,
                'identifier_mode': 'feature_concat',
                'entity_dim': 6,
            }
        )
        out = adapter.forward(
            {
                'x_enc': torch.randn(2, 10, 3),
                'entity_features': torch.randn(2, 10, 6),
            }
        )
        assert out['predictions'].shape == (2, 10, 1)
        assert torch.isfinite(out['predictions']).all(), 'transformer_adapter_feature_concat produced non-finite'

    # — Entity feature model (standalone) ———————————————————————————

    def test_lstm_entity_feature_model(self):
        from liulian.models.torch.swiss_lstm import LstmEntityFeatureModel

        m = LstmEntityFeatureModel(
            input_size=3,
            entity_dim=5,
            hidden_size=16,
            num_layers=1,
            c_out=1,
        )
        x = torch.randn(2, 10, 3)
        ef = torch.randn(2, 10, 5)
        out = m(x, ef)
        assert out.shape == (2, 10, 1)
        assert torch.isfinite(out).all(), 'lstm_entity_feature produced non-finite values'

    def test_transformer_entity_feature_model(self):
        from liulian.models.torch.swiss_transformer import TransformerEntityFeatureModel

        m = TransformerEntityFeatureModel(
            input_size=3,
            entity_dim=5,
            num_heads=2,
            num_layers=1,
            dim_feedforward=32,
            d_model=8,
            c_out=3,
        )
        x = torch.randn(2, 10, 3)
        ef = torch.randn(2, 10, 5)
        out = m(x, ef)
        assert out.shape == (2, 10, 3)
        assert torch.isfinite(out).all(), 'transformer_entity_feature produced non-finite values'

    # — ExtrapoLSTMAdapter ——————————————————————————————————————————

    def test_extrapo_adapter_multi_channel(self):
        from liulian.models.torch.swiss_lstm import ExtrapoLSTMAdapter

        adapter = ExtrapoLSTMAdapter(
            {
                'enc_in': 7,
                'c_out': 7,
                'd_model': 16,
                'e_layers': 1,
                'pred_len': 24,
            }
        )
        out = adapter.forward({'x_enc': torch.randn(2, 96, 7)})
        assert out['predictions'].shape == (2, 24, 7)
        assert torch.isfinite(out['predictions']).all(), 'extrapo_adapter_multi produced non-finite'

    # — Backward compatibility ——————————————————————————————————————

    def test_swiss_lstm_adapter_is_lstm_adapter(self):
        from liulian.models.torch.swiss_lstm import LSTMAdapter, SwissLSTMAdapter

        assert SwissLSTMAdapter is LSTMAdapter

    def test_swiss_transformer_adapter_is_encoder_adapter(self):
        from liulian.models.torch.swiss_transformer import (
            SwissTransformerAdapter,
            TransformerEncoderAdapter,
        )

        assert SwissTransformerAdapter is TransformerEncoderAdapter

    def test_swiss_embedding_adapter_sets_mode(self):
        from liulian.models.torch.swiss_lstm import SwissLSTMEmbeddingAdapter

        adapter = SwissLSTMEmbeddingAdapter(
            {
                'enc_in': 2,
                'd_model': 16,
                'e_layers': 1,
            }
        )
        assert adapter._entity_mode == 'embedding'

    def test_swiss_transformer_embedding_adapter_sets_mode(self):
        from liulian.models.torch.swiss_transformer import (
            SwissTransformerEmbeddingAdapter,
        )

        adapter = SwissTransformerEmbeddingAdapter(
            {
                'enc_in': 2,
                'n_heads': 2,
                'e_layers': 1,
                'd_ff': 32,
                'd_model': 8,
            }
        )
        assert adapter._entity_mode == 'embedding'

    # — Dropout support ————————————————————————————————————————————

    def test_lstm_dropout(self):
        from liulian.models.torch.swiss_lstm import SwissLstmModel

        m = SwissLstmModel(
            input_size=3,
            hidden_size=16,
            num_layers=3,
            c_out=1,
            dropout=0.2,
        )
        x = torch.randn(2, 10, 3)
        out = m(x)
        assert out.shape == (2, 10, 1)
        assert torch.isfinite(out).all(), 'lstm_dropout produced non-finite values'


# =====================================================================
# Item 5 — NaN masking in loss
# =====================================================================


class TestNanMasking:
    """Test the NaN-masked loss computation."""

    def test_masked_loss_skips_nan(self):
        from liulian.runtime.trainer import ForecastTrainer

        t = ForecastTrainer(
            config={'pred_len': 1, 'nan_mask_loss': True, 'show_progress': False},
        )
        criterion = nn.MSELoss()
        outputs = torch.tensor([[1.0, 2.0, 3.0]])
        targets = torch.tensor([[1.0, float('nan'), 3.0]])
        loss = t._masked_loss(criterion, outputs, targets)
        # Only non-NaN entries: MSE([1,3], [1,3]) = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_unmasked_loss_propagates_nan(self):
        from liulian.runtime.trainer import ForecastTrainer

        t = ForecastTrainer(
            config={'pred_len': 1, 'nan_mask_loss': False, 'show_progress': False},
        )
        criterion = nn.MSELoss()
        outputs = torch.tensor([[1.0, 2.0]])
        targets = torch.tensor([[1.0, float('nan')]])
        loss = t._masked_loss(criterion, outputs, targets)
        assert math.isnan(loss.item())


# =====================================================================
# Item 6 — teacher forcing / decoder input
# =====================================================================


class TestTeacherForcing:
    """Test decoder input construction."""

    def test_label_mode_with_label_len(self):
        from liulian.runtime.trainer import ForecastTrainer

        t = ForecastTrainer(
            config={'pred_len': 3, 'teacher_forcing': 'label', 'show_progress': False},
        )
        batch_y = torch.ones(2, 5, 4)  # label_len=2 + pred_len=3
        dec = t._build_decoder_input(batch_y, label_len=2, pred_len=3)
        assert dec.shape == (2, 5, 4)
        assert (dec[:, :2, :] == 1.0).all()  # label prefix is GT
        assert (dec[:, 2:, :] == 0.0).all()  # rest is zeros

    def test_zeros_mode(self):
        from liulian.runtime.trainer import ForecastTrainer

        t = ForecastTrainer(
            config={'pred_len': 3, 'teacher_forcing': 'zeros', 'show_progress': False},
        )
        batch_y = torch.ones(2, 5, 4)
        dec = t._build_decoder_input(batch_y, label_len=2, pred_len=3)
        assert dec.shape == (2, 3, 4)  # no label prefix
        assert (dec == 0.0).all()


# =====================================================================
# Item 7 — de-normalized metrics
# =====================================================================


class TestDenormMetrics:
    """Test that eval_denorm produces denorm_ prefixed metrics."""

    @staticmethod
    def _identity_inv(t: torch.Tensor) -> torch.Tensor:
        return t * 2  # simple scale transform for testing

    def test_denorm_metrics_present(self):
        from liulian.runtime.trainer import ForecastTrainer

        t = ForecastTrainer(
            config={
                'pred_len': 3,
                'eval_denorm': True,
                'show_progress': False,
                'loss': 'mse',
                'metrics': ['mse'],
                'teacher_forcing': 'zeros',
            },
            inverse_transform=self._identity_inv,
        )

        class M(nn.Module):
            def forward(self, x_enc, xm, xd, xmd):
                # Return tensor on same device as input
                return torch.zeros(x_enc.size(0), 3, x_enc.size(2), device=x_enc.device)

        model = M()
        x = torch.randn(8, 10, 2)
        y = torch.ones(8, 3, 2)
        xm = torch.zeros(8, 10, 1)
        ym = torch.zeros(8, 3, 1)
        loader = DataLoader(TensorDataset(x, y, xm, ym), batch_size=4)

        metrics = t.evaluate(model, loader, metric_names=['mse'])
        assert 'mse' in metrics
        assert 'denorm_mse' in metrics

    def test_no_denorm_by_default(self):
        from liulian.runtime.trainer import ForecastTrainer

        t = ForecastTrainer(
            config={
                'pred_len': 3,
                'show_progress': False,
                'loss': 'mse',
                'metrics': ['mse'],
                'teacher_forcing': 'zeros',
            },
        )

        class M(nn.Module):
            def forward(self, x_enc, xm, xd, xmd):
                return torch.zeros(x_enc.size(0), 3, x_enc.size(2), device=x_enc.device)

        model = M()
        x = torch.randn(8, 10, 2)
        y = torch.ones(8, 3, 2)
        xm = torch.zeros(8, 10, 1)
        ym = torch.zeros(8, 3, 1)
        loader = DataLoader(TensorDataset(x, y, xm, ym), batch_size=4)

        metrics = t.evaluate(model, loader, metric_names=['mse'])
        assert 'mse' in metrics
        assert 'denorm_mse' not in metrics

    def test_denorm_metrics_present_when_inverse_returns_none(self):
        from liulian.runtime.trainer import ForecastTrainer

        t = ForecastTrainer(
            config={
                'pred_len': 3,
                'eval_denorm': True,
                'show_progress': False,
                'loss': 'mse',
                'metrics': ['mse'],
                'teacher_forcing': 'zeros',
            },
            inverse_transform=lambda x: None,
        )

        class M(nn.Module):
            def forward(self, x_enc, xm, xd, xmd):
                return torch.zeros(x_enc.size(0), 3, x_enc.size(2), device=x_enc.device)

        model = M()
        x = torch.randn(8, 10, 2)
        y = torch.ones(8, 3, 2)
        xm = torch.zeros(8, 10, 1)
        ym = torch.zeros(8, 3, 1)
        loader = DataLoader(TensorDataset(x, y, xm, ym), batch_size=4)

        metrics = t.evaluate(model, loader, metric_names=['mse'])
        assert 'mse' in metrics
        assert 'denorm_mse' in metrics


# =====================================================================
# Item 8 — HPO search spaces & ASHA presets
# =====================================================================


class TestSearchSpaces:
    """Test the pre-defined search space registry."""

    @pytest.mark.parametrize(
        'name',
        [
            'swiss_lstm',
            'swiss_transformer',
            'swiss_lstm_embedding',
            'swiss_transformer_embedding',
            'swiss_stgnn',
            'timellm_etth1',
            'timellm_etth2',
            'timellm_ettm',
            'timellm_weather',
            'timellm_electricity',
            'timellm_swissriver',
        ],
    )
    def test_lookup(self, name: str):
        from liulian.optim.search_spaces import get_search_space

        space = get_search_space(name)
        assert isinstance(space, dict)
        assert len(space) > 0
        for v in space.values():
            # Values are ray.tune.* sample objects (choice, uniform, etc.)
            assert not isinstance(v, (int, float, str)), f'Expected ray.tune.* object, got {type(v)}'

    def test_unknown_raises(self):
        from liulian.optim.search_spaces import get_search_space

        with pytest.raises(ValueError, match='Unknown search space'):
            get_search_space('nonexistent_space')

    @pytest.mark.parametrize(
        'name',
        ['default', 'soft', 'single_soft', 'single_hard'],
    )
    def test_asha_preset(self, name: str):
        from liulian.optim.search_spaces import get_asha_preset

        preset = get_asha_preset(name)
        assert preset['scheduler'] == 'asha'
        assert 'max_epochs' in preset
        assert 'grace_period' in preset
        assert 'reduction_factor' in preset

    def test_asha_unknown_raises(self):
        from liulian.optim.search_spaces import get_asha_preset

        with pytest.raises(ValueError, match='Unknown ASHA preset'):
            get_asha_preset('bogus')


class TestFlattenEntityIdChunks:
    """Regression for the predict() entity_ids flatten (trainer.py:646).

    The eval loop collects each batch's entity_ids (batch[4]) into a list of
    chunks, then flattens them for the per-entity inverse transform. A chunk
    may be a tuple (some DataLoader collates yield tuples), and the old
    ``sum(chunks, [])`` raised ``TypeError: can only concatenate list (not
    "tuple") to list`` — crashing predict()+aggregate on those datasets
    (surfaced by the swiss-river e2e test). _flatten_id_chunks must handle
    both list and tuple chunks. Bug was data/collate-triggered latent
    fragility, not a regression.
    """

    def test_tuple_chunks_flatten_without_error(self):
        # This is the exact bug: tuple chunks. Old sum(...,[]) raised here.
        from liulian.runtime.trainer import _flatten_id_chunks

        assert _flatten_id_chunks([('a', 'b'), ('c',)]) == ['a', 'b', 'c']

    def test_old_sum_expression_would_raise_on_tuples(self):
        # Locks in WHY the fix is needed: the pre-fix expression is broken.
        with pytest.raises(TypeError):
            sum([('a', 'b'), ('c',)], [])  # noqa: RUF017 - documents the bug

    def test_list_chunks_still_flatten(self):
        from liulian.runtime.trainer import _flatten_id_chunks

        assert _flatten_id_chunks([['a', 'b'], ['c']]) == ['a', 'b', 'c']

    def test_mixed_list_and_tuple_chunks(self):
        from liulian.runtime.trainer import _flatten_id_chunks

        assert _flatten_id_chunks([['a'], ('b', 'c')]) == ['a', 'b', 'c']

    def test_empty_input(self):
        from liulian.runtime.trainer import _flatten_id_chunks

        assert _flatten_id_chunks([]) == []


class TestPassEntityIdsCoordinates:
    """Regression: coordinates_embedding must be in the trainer's pass_entity_ids set.

    timellm reads the per-sample station id from the entity_ids KWARG and indexes a
    (num_stations, 2) coord table with it. If coordinates_embedding is NOT in
    pass_entity_ids, the kwarg is not passed and the model falls back to the raw
    station number from x_mark (e.g. 2091) -> index-out-of-bounds -> CUDA
    device-side assert. This locks the fix (trainer.py) so coords cannot regress.
    """

    def test_coordinates_embedding_passes_entity_ids(self):
        from liulian.runtime.trainer import ForecastTrainer

        t = ForecastTrainer(config={
            'pred_len': 1, 'show_progress': False,
            'identifier_mode': 'coordinates_embedding',
        })
        assert t.pass_entity_ids is True

    def test_other_timellm_transparent_modes_pass_entity_ids(self):
        from liulian.runtime.trainer import ForecastTrainer

        for mode in ('onehot_embedding', 'sinusoidal_embedding', 'random_embedding'):
            t = ForecastTrainer(config={'pred_len': 1, 'show_progress': False, 'identifier_mode': mode})
            assert t.pass_entity_ids is True, mode

    def test_none_mode_does_not_pass_entity_ids(self):
        from liulian.runtime.trainer import ForecastTrainer

        t = ForecastTrainer(config={'pred_len': 1, 'show_progress': False, 'identifier_mode': 'none'})
        assert t.pass_entity_ids is False
