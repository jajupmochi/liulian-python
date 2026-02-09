"""Tests for Fourier correlation layer."""

import torch
import pytest
from liulian.layers.timeseries.fourier_correlation import FourierBlock


class TestFourierBlock:
    """Test FourierBlock layer."""

    def test_forward_shape(self):
        """Test output shape."""
        block = FourierBlock(in_channels=16, out_channels=16, n_heads=4, seq_len=24, modes=8)
        q = torch.randn(8, 24, 4, 4)  # (B, L, H, E)
        out, attn = block(q, None, None, None)
        assert out.shape == (8, 24, 4, 4)
        assert attn is None

    def test_complex_multiplication(self):
        """Test complex number multiplication."""
        block = FourierBlock(in_channels=16, out_channels=16, n_heads=4, seq_len=24, modes=8)
        x = torch.complex(torch.randn(2, 4, 4, 12), torch.randn(2, 4, 4, 12))
        w = torch.complex(torch.randn(4, 4, 4, 8), torch.randn(4, 4, 4, 8))
        result = block.compl_mul1d("bhi,hio->bho", x, w)
        assert torch.is_complex(result)

    def test_gradient_flow(self):
        """Test gradients flow through layer."""
        block = FourierBlock(in_channels=16, out_channels=16, n_heads=4, seq_len=24, modes=8)
        q = torch.randn(2, 24, 4, 4, requires_grad=True)
        out, _ = block(q, None, None, None)
        loss = out.sum()
        loss.backward()
        assert q.grad is not None
