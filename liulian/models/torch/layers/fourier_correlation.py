"""Fourier Correlation Layers for FEDformer

Adapted from Time-Series-Library:
    Source: https://github.com/thuml/Time-Series-Library
    File: layers/FourierCorrelation.py

Performs attention mechanism on frequency domain achieving O(N) complexity.
Paper: https://proceedings.mlr.press/v162/zhou22g.html
"""

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """Get modes on frequency domain.

    Args:
        seq_len: Sequence length
        modes: Number of modes to select
        mode_select_method: 'random' for random sampling, else lowest modes

    Returns:
        Sorted list of frequency mode indices
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
    """1D Fourier block for frequency-domain representation learning.

    Performs FFT, linear transform, and Inverse FFT.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_heads,
        seq_len,
        modes=0,
        mode_select_method='random',
    ):
        super(FourierBlock, self).__init__()
        self.index = get_frequency_modes(
            seq_len, modes=modes, mode_select_method=mode_select_method
        )
        self.n_heads = n_heads
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                self.n_heads,
                in_channels // self.n_heads,
                out_channels // self.n_heads,
                len(self.index),
                dtype=torch.float,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                self.n_heads,
                in_channels // self.n_heads,
                out_channels // self.n_heads,
                len(self.index),
                dtype=torch.float,
            )
        )

    def compl_mul1d(self, order, x, weights):
        """Complex multiplication."""
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(
                weights, torch.zeros_like(weights).to(weights.device)
            )
        if x_flag or w_flag:
            return torch.complex(
                torch.einsum(order, x.real, weights.real)
                - torch.einsum(order, x.imag, weights.imag),
                torch.einsum(order, x.real, weights.imag)
                + torch.einsum(order, x.imag, weights.real),
            )
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            if i >= x_ft.shape[3] or wi >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, wi] = self.compl_mul1d(
                'bhi,hio->bho',
                x_ft[:, :, :, i],
                torch.complex(self.weights1, self.weights2)[:, :, :, wi],
            )
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


class FourierCrossAttention(nn.Module):
    """1D Fourier Cross Attention layer.

    Performs FFT, linear transform, attention mechanism and Inverse FFT.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        seq_len_q,
        seq_len_kv,
        modes=64,
        mode_select_method='random',
        activation='tanh',
        policy=0,
        num_heads=8,
    ):
        super(FourierCrossAttention, self).__init__()
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index_q = get_frequency_modes(
            seq_len_q, modes=modes, mode_select_method=mode_select_method
        )
        self.index_kv = get_frequency_modes(
            seq_len_kv, modes=modes, mode_select_method=mode_select_method
        )
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                num_heads,
                in_channels // num_heads,
                out_channels // num_heads,
                len(self.index_q),
                dtype=torch.float,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                num_heads,
                in_channels // num_heads,
                out_channels // num_heads,
                len(self.index_q),
                dtype=torch.float,
            )
        )

    def compl_mul1d(self, order, x, weights):
        """Complex multiplication."""
        x_flag = True
        w_flag = True
        if not torch.is_complex(x):
            x_flag = False
            x = torch.complex(x, torch.zeros_like(x).to(x.device))
        if not torch.is_complex(weights):
            w_flag = False
            weights = torch.complex(
                weights, torch.zeros_like(weights).to(weights.device)
            )
        if x_flag or w_flag:
            return torch.complex(
                torch.einsum(order, x.real, weights.real)
                - torch.einsum(order, x.imag, weights.imag),
                torch.einsum(order, x.real, weights.imag)
                + torch.einsum(order, x.imag, weights.real),
            )
        else:
            return torch.einsum(order, x.real, weights.real)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(
            B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat
        )
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            if j >= xq_ft.shape[3]:
                continue
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(
            B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat
        )
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            if j >= xk_ft.shape[3]:
                continue
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # Perform attention mechanism on frequency domain
        xqk_ft = self.compl_mul1d('bhex,bhey->bhxy', xq_ft_, xk_ft_)
        if self.activation == 'tanh':
            xqk_ft = torch.complex(xqk_ft.real.tanh(), xqk_ft.imag.tanh())
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception(
                '{} activation function is not implemented'.format(self.activation)
            )
        xqkv_ft = self.compl_mul1d('bhxy,bhey->bhex', xqk_ft, xk_ft_)
        xqkvw = self.compl_mul1d(
            'bhex,heox->bhox',
            xqkv_ft,
            torch.complex(self.weights1, self.weights2),
        )
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            if i >= xqkvw.shape[3] or j >= out_ft.shape[3]:
                continue
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(
            out_ft / self.in_channels / self.out_channels, n=xq.size(-1)
        )
        return (out, None)
