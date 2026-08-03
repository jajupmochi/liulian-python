"""
AutoTimes-style autoregressive frozen-LLM forecaster.

Paper (idea): AutoTimes — Autoregressive Time Series Forecasters via Large
Language Models, https://arxiv.org/abs/2402.02370

FROM-SCRATCH adaptation (not a vendored port), on the SAME entry + pipeline as
timellm/gpt4ts/tempo (only the architecture differs, per the study design).

AutoTimes' central idea kept here: cut the series into non-overlapping SEGMENTS
("time tokens") of length ``token_len``, embed each into the LLM space, run them
through a frozen GPT-2 whose CAUSAL attention models the token sequence
autoregressively, and predict the NEXT segment from the last token's hidden
state. With ``token_len == pred_len`` the next segment IS the forecast horizon.

Differences from the paper, stated honestly: single-step next-segment decode
(one horizon = one token), GPT-2 backbone (LayerNorm + wpe trainable, like
GPT4TS), no timestamp/text token embeddings. Identity is additive only (shared
plumbing with gpt4ts/tempo); prompt-only modes are rejected.
"""

from os import PathLike
from typing import Union

import torch
import torch.nn as nn


class Model(nn.Module):
    """AutoTimes-style: segment the series into time tokens, autoregress with a frozen GPT-2."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        # One "time token" = a segment of the series. Default token_len == pred_len
        # so the autoregressive next-token prediction IS the forecast horizon.
        self.token_len = int(getattr(configs, 'token_len', self.pred_len))
        self.n_tokens = self.seq_len // self.token_len
        if self.n_tokens < 1:
            raise ValueError(
                f'seq_len ({self.seq_len}) must be >= token_len ({self.token_len}); '
                'AutoTimes needs at least one input time token.'
            )
        self.gpt_layers = getattr(configs, 'gpt_layers', getattr(configs, 'llm_layers', 6))
        self.cache_dir: Union[str, PathLike, None] = getattr(configs, 'cache_dir', None)

        from transformers import GPT2Config, GPT2Model

        gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
        gpt2_config.num_hidden_layers = self.gpt_layers
        gpt2_config.output_attentions = True
        gpt2_config.output_hidden_states = True
        try:
            self.gpt2 = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                local_files_only=True,
                config=gpt2_config,
            )
        except EnvironmentError:
            print('Local GPT-2 files not found. Downloading from HuggingFace...')
            self.gpt2 = GPT2Model.from_pretrained(
                'openai-community/gpt2',
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                local_files_only=False,
                config=gpt2_config,
            )
        self.gpt2_dim = self.gpt2.config.n_embd
        for param in self.gpt2.parameters():
            param.requires_grad = False
        for layer in self.gpt2.h:
            for name, param in layer.named_parameters():
                if 'ln' in name:
                    param.requires_grad = True
        for param in self.gpt2.ln_f.parameters():
            param.requires_grad = True
        self.gpt2.wpe.weight.requires_grad = True

        self.in_layer = nn.Linear(self.token_len, self.gpt2_dim)
        self.out_layer = nn.Linear(self.gpt2_dim, self.pred_len)

        # -- Additive entity identity (same plumbing/rules as gpt4ts/tempo) ----
        import math as _math

        self.entity_id_mark_col: Union[int, None] = None
        self.identifier_mode: str = getattr(configs, 'identifier_mode', 'none')
        _prompt_only = {'entity_description', 'soft_prompt', 'text_embedding'}
        if self.identifier_mode in _prompt_only:
            raise ValueError(
                f'autotimes supports additive identity only; identifier_mode='
                f'{self.identifier_mode!r} is prompt-only. Use none/embedding/'
                f'random_embedding/onehot_embedding/sinusoidal_embedding, or run '
                f'prompt modes on timellm.'
            )
        self.entity_embedding: Union[nn.Embedding, None] = None
        self.transparent_proj: Union[nn.Linear, None] = None
        if self.identifier_mode in ('embedding', 'random_embedding'):
            _n = int(getattr(configs, 'num_entities', 0)) or None
            if _n is None:
                raise ValueError(f'identifier_mode={self.identifier_mode!r} requires configs.num_entities')
            self.entity_embedding = nn.Embedding(_n, self.gpt2_dim)
            if self.identifier_mode == 'random_embedding':
                self.entity_embedding.weight.requires_grad_(False)
        elif self.identifier_mode in ('onehot_embedding', 'sinusoidal_embedding'):
            _n = int(getattr(configs, 'num_entities', 0)) or None
            if _n is None:
                raise ValueError(f'identifier_mode={self.identifier_mode!r} requires configs.num_entities')
            if self.identifier_mode == 'onehot_embedding':
                feat = torch.eye(_n)
            else:
                dim = int(getattr(configs, 'sinusoidal_dim', 16))
                pos = torch.arange(_n).unsqueeze(1).float()
                div = torch.exp(torch.arange(0, dim, 2).float() * (-_math.log(10000.0) / dim))
                feat = torch.zeros(_n, dim)
                feat[:, 0::2] = torch.sin(pos * div)
                feat[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('transparent_feat', feat)
            self.transparent_proj = nn.Linear(feat.shape[1], self.gpt2_dim)

    def _resolve_station_ids(self, x_mark_enc, entity_ids):
        if entity_ids is not None:
            return entity_ids.reshape(-1).long()
        if self.entity_id_mark_col is not None and x_mark_enc is not None:
            return x_mark_enc[:, 0, self.entity_id_mark_col].long()
        return None

    def _inject_identity(self, x, station_ids):
        if station_ids is None:
            return x
        if self.entity_embedding is not None:
            x = x + self.entity_embedding(station_ids).unsqueeze(1)
        if self.transparent_proj is not None:
            x = x + self.transparent_proj(self.transparent_feat[station_ids]).unsqueeze(1)
        return x

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        B, L, C = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        # Keep the most recent n_tokens * token_len points, segment into time tokens.
        keep = self.n_tokens * self.token_len
        x = x[:, -keep:, :]  # [B, keep, C]
        x = x.permute(0, 2, 1).reshape(B * C, self.n_tokens, self.token_len)

        x = self.in_layer(x)  # [B*C, n_tokens, gpt2_dim]
        x = self._inject_identity(x, self._resolve_station_ids(x_mark_enc, entity_ids))
        # GPT-2 causal attention autoregressively models the token sequence.
        outputs = self.gpt2(inputs_embeds=x).last_hidden_state  # [B*C, n_tokens, gpt2_dim]

        # Predict the NEXT segment from the LAST token's hidden state.
        pred = self.out_layer(outputs[:, -1, :])  # [B*C, pred_len]
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, pred_len, C]
        return pred * stdev + means

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, entity_ids=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=entity_ids)
            return dec_out[:, -self.pred_len :, :]
        return None
