"""Tests for the TEMPO-style decomposition + frozen-GPT-2 model.

TEMPO decomposes the series (trend + seasonal) and runs each component through a
shared frozen GPT-2, summing the component predictions. It shares the study's
additive entity-identity plumbing with GPT4TS and runs on the same pipeline. The
tests assert the forward contract, that the decomposition reconstructs the input,
that identity is not a dead knob, and that prompt-only modes are rejected.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')


def _cfg(**kw):
    base = dict(
        task_name='long_term_forecast',
        pred_len=7,
        seq_len=90,
        enc_in=1,
        patch_len=16,
        moving_avg=25,
        llm_layers=1,
        cache_dir=None,
        identifier_mode='none',
        num_entities=28,
        sinusoidal_dim=16,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(scope='module')
def _x():
    torch.manual_seed(0)
    return torch.randn(4, 90, 1), torch.zeros(4, 90, 1)


class TestTempoModel:
    def test_forward_shape(self, _x):
        from liulian.models.torch.tempo import Model

        x, xm = _x
        m = Model(_cfg()).eval()
        with torch.no_grad():
            y = m(x, xm, None, None)
        assert tuple(y.shape) == (4, 7, 1)

    def test_decomposition_reconstructs_input(self, _x):
        from liulian.models.torch.layers.decomposition import series_decomp

        x, _ = _x
        sea, trend = series_decomp(25)(x)
        assert torch.allclose(sea + trend, x, atol=1e-5)

    def test_identity_is_not_a_dead_knob(self, _x):
        from liulian.models.torch.tempo import Model

        x, xm = _x
        m = Model(_cfg(identifier_mode='embedding')).eval()
        e0 = torch.zeros(4, 1).long()
        e1 = torch.ones(4, 1).long()
        with torch.no_grad():
            y0 = m(x, xm, None, None, entity_ids=e0)
            y1 = m(x, xm, None, None, entity_ids=e1)
        # different station ids -> different predictions (identity actually flows)
        assert (y0 - y1).abs().mean() > 0

    def test_random_embedding_is_frozen(self):
        from liulian.models.torch.tempo import Model

        m = Model(_cfg(identifier_mode='random_embedding'))
        assert m.entity_embedding is not None
        assert m.entity_embedding.weight.requires_grad is False

    @pytest.mark.parametrize('mode', ['soft_prompt', 'entity_description', 'text_embedding'])
    def test_prompt_only_modes_rejected(self, mode):
        from liulian.models.torch.tempo import Model

        with pytest.raises(ValueError, match='additive'):
            Model(_cfg(identifier_mode=mode))

    def test_gpt2_backbone_mostly_frozen(self):
        from liulian.models.torch.tempo import Model

        m = Model(_cfg())
        # GPT-2 blocks frozen except LayerNorm; wpe + ln_f trainable.
        for name, p in m.gpt2.h.named_parameters():
            if 'ln' not in name:
                assert p.requires_grad is False, name
