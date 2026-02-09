"""Fourier-based correlation layer for frequency-domain processing.

Performs representation learning on frequency domain via FFT, linear
transformation, and inverse FFT.

Adapted from Time-Series-Library.
"""

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(
    seq_len: int,
    modes: int = 64,
    mode_select_method: str = 'random'
) -> list:
    """Get frequency domain modes for Fourier transformation.
    
    Args:
        seq_len: Sequence length
        modes: Number of modes to select
        mode_select_method: 'random' or 'fixed' (lowest modes)
        
    Returns:
        List of selected frequency indices
    """
    modes = min(modes, seq_len // 2)
    
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    
    index.sort()
    return index


class FourierBlock(nn.Module):
    """Fourier-based correlation block using FFT and frequency-domain transformations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int,
        seq_len: int,
        modes: int = 64,
        mode_select_method: str = 'random'
    ):
        """Initialize Fourier block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            n_heads: Number of heads for multi-head operations
            seq_len: Sequence length
            modes: Number of frequency modes
            mode_select_method: Mode selection strategy
        """
        super().__init__()
        
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        self.n_heads = n_heads
        self.scale = 1.0 / (in_channels * out_channels)
        
        # Learnable weights for complex multiplication
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                self.n_heads,
                in_channels // self.n_heads,
                out_channels // self.n_heads,
                len(self.index),
                dtype=torch.float
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                self.n_heads,
                in_channels // self.n_heads,
                out_channels // self.n_heads,
                len(self.index),
                dtype=torch.float
            )
        )

    def compl_mul1d(self, order: str, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in frequency domain.
        
        Args:
            order: Einstein summation order
            x: Input tensor
            weights: Weight tensor
            
        Returns:
            Complex-valued result
        """
        # Convert to complex if needed
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            weights = torch.complex(weights, torch.zeros_like(weights).to(weights.device))
        
        # Complex multiplication
        real_part = (
            torch.einsum(order, x.real, weights.real) - 
            torch.einsum(order, x.imag, weights.imag)
        )
        imag_part = (
            torch.einsum(order, x.real, weights.imag) + 
            torch.einsum(order, x.imag, weights.real)
        )
        
        return torch.complex(real_part, imag_part)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """Forward pass with Fourier transformation.
        
        Args:
            q: Query (batch, length, heads, dim)
            k: Key (unused, included for API consistency)
            v: Value (unused, included for API consistency)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)  # (B, H, E, L)
        
        # Compute FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Initialize output in frequency domain
        out_ft = torch.zeros(
            B, H, E, L // 2 + 1,
            device=x.device,
            dtype=torch.cfloat
        )
        
        # Apply learnable transformations in frequency domain
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[3] or wi >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, wi] = self.compl_mul1d(
                "bhi,hio->bho",
                x_ft[:, :, :, i],
                torch.complex(self.weights1, self.weights2)[:, :, :, wi]
            )
        
        # Inverse FFT back to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        
        return (x.permute(0, 3, 1, 2), None)


__all__ = ['FourierBlock', 'get_frequency_modes']
