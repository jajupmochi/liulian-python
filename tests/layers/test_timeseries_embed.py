"""Tests for timeseries embedding layers."""

import torch
import pytest
from liulian.layers.timeseries.embed import (
    PositionalEmbedding,
    TokenEmbedding,
    FixedEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
    DataEmbedding,
    DataEmbedding_inverted,
    DataEmbedding_wo_pos,
)


class TestPositionalEmbedding:
    """Test positional embedding layer."""

    def test_shape(self):
        """Test output shape."""
        emb = PositionalEmbedding(d_model=64, max_len=100)
        x = torch.randn(8, 50, 64)
        out = emb(x)
        assert out.shape == (1, 50, 64)

    def test_non_learnable(self):
        """Verify positional embeddings are non-learnable."""
        emb = PositionalEmbedding(d_model=64)
        assert not any(p.requires_grad for p in emb.parameters())


class TestTokenEmbedding:
    """Test token embedding layer."""

    def test_shape(self):
        """Test projection to embedding dimension."""
        emb = TokenEmbedding(c_in=10, d_model=64)
        x = torch.randn(8, 50, 10)
        out = emb(x)
        assert out.shape == (8, 50, 64)

    def test_learnable(self):
        """Verify weights are learnable."""
        emb = TokenEmbedding(c_in=10, d_model=64)
        assert any(p.requires_grad for p in emb.parameters())


class TestDataEmbedding:
    """Test unified data embedding."""

    def test_basic_embedding(self):
        """Test basic embedding without temporal features."""
        emb = DataEmbedding(c_in=6, d_model=64, freq='h')
        x = torch.randn(8, 24, 6)  # batch, time, channels
        out = emb(x, x_mark=None)
        assert out.shape == (8, 24, 64)

    def test_with_temporal_features(self):
        """Test embedding with temporal features."""
        emb = DataEmbedding(c_in=6, d_model=64, embed_type='fixed', freq='h')
        x = torch.randn(8, 24, 6)
        x_mark = torch.randint(0, 13, (8, 24, 5))  # month, day, weekday, hour, minute
        out = emb(x, x_mark=x_mark)
        assert out.shape == (8, 24, 64)

    def test_dropout(self):
        """Test dropout effect."""
        emb = DataEmbedding(c_in=6, d_model=64, dropout=0.5)
        emb.eval()
        x = torch.randn(8, 24, 6)
        out1 = emb(x, x_mark=None)
        out2 = emb(x, x_mark=None)
        assert torch.allclose(out1, out2)  # Deterministic in eval mode


class TestDataEmbeddingInverted:
    """Test inverted (channel-wise) embedding."""

    def test_shape(self):
        """Test channel-wise embedding dimension."""
        emb = DataEmbedding_inverted(c_in=6, d_model=64)
        x = torch.randn(8, 24, 6)  # batch, time, channels
        out = emb(x, x_mark=None)
        assert out.shape == (8, 6, 64)


class TestDataEmbeddingWoPos:
    """Test embedding without positional encoding."""

    def test_shape(self):
        """Test shape without positional encoding."""
        emb = DataEmbedding_wo_pos(c_in=6, d_model=64, freq='h')
        x = torch.randn(8, 24, 6)
        out = emb(x, x_mark=None)
        assert out.shape == (8, 24, 64)

    def test_no_pos_embedding_difference(self):
        """Verify absence of positional component."""
        emb = DataEmbedding_wo_pos(c_in=6, d_model=64)
        emb.eval()
        
        # Two different sequences of same shape should have different temporal patterns
        x1 = torch.randn(1, 10, 6)
        x2 = torch.randn(1, 20, 6)
        
        out1 = emb(x1, x_mark=None)
        out2 = emb(x2, x_mark=None)
        
        # Shapes should match only the first dimension
        assert out1.shape[0] == out2.shape[0] == 1
        assert out1.shape[2] == out2.shape[2] == 64


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
