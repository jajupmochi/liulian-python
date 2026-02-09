"""Tests for AutoCorrelation layer."""

import torch
import pytest
from liulian.layers.timeseries.autocorrelation import (
    AutoCorrelation,
    AutoCorrelationLayer,
)


class TestAutoCorrelation:
    """Test AutoCorrelation mechanism."""

    def test_output_shape_training(self):
        """Test output shape during training."""
        acorr = AutoCorrelation(factor=1.0, output_attention=False)
        acorr.train()
        
        batch, length, n_heads, d_head = 8, 24, 4, 16
        queries = torch.randn(batch, length, n_heads, d_head)
        keys = torch.randn(batch, length, n_heads, d_head)
        values = torch.randn(batch, length, n_heads, d_head)
        
        output, attn = acorr(queries, keys, values, attn_mask=None)
        
        assert output.shape == values.shape
        assert attn is None

    def test_output_shape_inference(self):
        """Test output shape during inference."""
        acorr = AutoCorrelation(factor=1.0, output_attention=False)
        acorr.eval()
        
        batch, length, n_heads, d_head = 8, 24, 4, 16
        queries = torch.randn(batch, length, n_heads, d_head)
        keys = torch.randn(batch, length, n_heads, d_head)
        values = torch.randn(batch, length, n_heads, d_head)
        
        output, attn = acorr(queries, keys, values, attn_mask=None)
        
        assert output.shape == values.shape
        assert attn is None

    def test_attention_output(self):
        """Test that attention can be returned."""
        acorr = AutoCorrelation(factor=1.0, output_attention=True)
        acorr.eval()
        
        batch, length, n_heads, d_head = 4, 12, 2, 16
        queries = torch.randn(batch, length, n_heads, d_head)
        keys = torch.randn(batch, length, n_heads, d_head)
        values = torch.randn(batch, length, n_heads, d_head)
        
        output, attn = acorr(queries, keys, values, attn_mask=None)
        
        assert output.shape == values.shape
        assert attn is not None
        assert attn.shape[0] == batch  # batch dimension

    def test_sequence_length_mismatch(self):
        """Test handling of different sequence lengths."""
        acorr = AutoCorrelation(factor=1.0, output_attention=False)
        acorr.eval()
        
        batch, n_heads, d_head = 4, 2, 16
        queries = torch.randn(batch, 24, n_heads, d_head)  # Longer
        keys = torch.randn(batch, 12, n_heads, d_head)    # Shorter
        values = torch.randn(batch, 12, n_heads, d_head)  # Shorter
        
        output, _ = acorr(queries, keys, values, attn_mask=None)
        
        assert output.shape == queries.shape


class TestAutoCorrelationLayer:
    """Test AutoCorrelationLayer wrapper."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        d_model = 64
        n_heads = 4
        
        acorr = AutoCorrelation(factor=1.0)
        layer = AutoCorrelationLayer(acorr, d_model=d_model, n_heads=n_heads)
        
        batch, length = 8, 24
        queries = torch.randn(batch, length, d_model)
        keys = torch.randn(batch, length, d_model)
        values = torch.randn(batch, length, d_model)
        
        output, _ = layer(queries, keys,  values, attn_mask=None)
        
        assert output.shape == (batch, length, d_model)

    def test_projection_dimensions(self):
        """Test internal projections have correct dimensions."""
        d_model = 64
        n_heads = 4
        d_keys = 16
        d_values = 16
        
        acorr = AutoCorrelation(factor=1.0)
        layer = AutoCorrelationLayer(
            acorr,
            d_model=d_model,
            n_heads=n_heads,
            d_keys=d_keys,
            d_values=d_values,
        )
        
        # Verify projection matrices
        assert layer.query_projection.weight.shape == (d_keys * n_heads, d_model)
        assert layer.key_projection.weight.shape == (d_keys * n_heads, d_model)
        assert layer.value_projection.weight.shape == (d_values * n_heads, d_model)
        assert layer.out_projection.weight.shape == (d_model, d_values * n_heads)

    def test_gradient_flow(self):
        """Test gradients flow through layer."""
        d_model = 32
        n_heads = 2
        
        acorr = AutoCorrelation(factor=1.0)
        layer = AutoCorrelationLayer(acorr, d_model=d_model, n_heads=n_heads)
        
        batch, length = 2, 10
        queries = torch.randn(batch, length, d_model, requires_grad=True)
        keys = torch.randn(batch, length, d_model, requires_grad=True)
        values = torch.randn(batch, length, d_model, requires_grad=True)
        
        output, _ = layer(queries, keys, values)
        loss = output.sum()
        loss.backward()
        
        assert queries.grad is not None
        assert keys.grad is not None
        assert values.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
