"""Tests for the AutoTimes-style autoregressive frozen-GPT-2 model.

AutoTimes segments the series into time tokens and autoregresses with a frozen
GPT-2, predicting the next segment (= forecast horizon) from the last token. It
shares the study's additive entity-identity plumbing and runs on the same
pipeline. Tests assert the forward contract, the token segmentation, that
identity is not a dead knob, and that prompt-only modes are rejected.
"""

from __future__ import annotations

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
        patch_len=16,
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


class TestAutoTimesModel:
    def test_forward_shape(self, _x):
        from liulian.models.torch.autotimes import Model

        x, xm = _x
        m = Model(_cfg()).eval()
        with torch.no_grad():
            y = m(x, xm, None, None)
        assert tuple(y.shape) == (4, 7, 1)

    def test_token_count_is_floor_div(self):
        from liulian.models.torch.autotimes import Model

        # token_len defaults to pred_len (7); seq_len 90 -> 12 tokens (84 points kept).
        assert Model(_cfg()).n_tokens == 90 // 7

    def test_custom_token_len(self):
        from liulian.models.torch.autotimes import Model

        assert Model(_cfg(token_len=10)).n_tokens == 9  # 90 // 10

    def test_token_len_larger_than_seq_raises(self):
        from liulian.models.torch.autotimes import Model

        with pytest.raises(ValueError, match='at least one input time token'):
            Model(_cfg(seq_len=5, token_len=7))

    def test_identity_is_not_a_dead_knob(self, _x):
        from liulian.models.torch.autotimes import Model

        x, xm = _x
        m = Model(_cfg(identifier_mode='embedding')).eval()
        with torch.no_grad():
            y0 = m(x, xm, None, None, entity_ids=torch.zeros(4, 1).long())
            y1 = m(x, xm, None, None, entity_ids=torch.ones(4, 1).long())
        assert (y0 - y1).abs().mean() > 0

    def test_random_embedding_is_frozen(self):
        from liulian.models.torch.autotimes import Model

        m = Model(_cfg(identifier_mode='random_embedding'))
        assert m.entity_embedding.weight.requires_grad is False

    @pytest.mark.parametrize('mode', ['soft_prompt', 'entity_description', 'text_embedding'])
    def test_prompt_only_modes_rejected(self, mode):
        from liulian.models.torch.autotimes import Model

        with pytest.raises(ValueError, match='additive'):
            Model(_cfg(identifier_mode=mode))
