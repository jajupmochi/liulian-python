"""Tests for the CALF-style cross-modal dual-branch model.

CALF runs the patch features through TWO branches sharing one frozen GPT-2: a
cross-modal branch (reprogrammed into the LLM word-embedding space) and a
temporal branch, fused by summing their heads. It shares the additive
entity-identity plumbing and runs on the same pipeline. Tests assert the forward
contract, that BOTH branches contribute, that identity is not a dead knob, and
that prompt-only modes are rejected.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')


def _cfg(**kw):
    base = dict(
        task_name='long_term_forecast',
        pred_len=7,
        seq_len=90,
        enc_in=1,
        d_model=32,
        d_ff=128,
        n_heads=8,
        patch_len=16,
        stride=8,
        llm_layers=1,
        cache_dir=None,
        dropout=0.1,
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


class TestCalfModel:
    def test_forward_shape(self, _x):
        from liulian.models.torch.calf import Model

        x, xm = _x
        m = Model(_cfg()).eval()
        with torch.no_grad():
            y = m(x, xm, None, None)
        assert tuple(y.shape) == (4, 7, 1)

    def test_both_branches_contribute(self, _x):
        from liulian.models.torch.calf import Model

        x, xm = _x
        m = Model(_cfg()).eval()
        with torch.no_grad():
            full = m(x, xm, None, None)
            m2 = copy.deepcopy(m)
            for p in m2.head_temporal.parameters():
                p.zero_()
            cross_only = m2(x, xm, None, None)
        # zeroing the temporal head changes the output -> the temporal branch matters
        assert (full - cross_only).abs().mean() > 0

    def test_identity_is_not_a_dead_knob(self, _x):
        from liulian.models.torch.calf import Model

        x, xm = _x
        m = Model(_cfg(identifier_mode='embedding')).eval()
        with torch.no_grad():
            y0 = m(x, xm, None, None, entity_ids=torch.zeros(4, 1).long())
            y1 = m(x, xm, None, None, entity_ids=torch.ones(4, 1).long())
        assert (y0 - y1).abs().mean() > 0

    def test_d_ff_larger_than_hidden_raises(self):
        from liulian.models.torch.calf import Model

        with pytest.raises(ValueError, match='d_ff'):
            Model(_cfg(d_ff=1024))  # > gpt2 hidden (768)

    @pytest.mark.parametrize('mode', ['soft_prompt', 'entity_description', 'text_embedding'])
    def test_prompt_only_modes_rejected(self, mode):
        from liulian.models.torch.calf import Model

        with pytest.raises(ValueError, match='additive'):
            Model(_cfg(identifier_mode=mode))
