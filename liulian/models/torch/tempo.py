"""
TEMPO-style decomposition + frozen-LLM forecaster.

Paper (idea): TEMPO — Prompt-based Generative Pre-trained Transformer for Time
Series Forecasting, https://arxiv.org/abs/2310.04948

This is a FROM-SCRATCH adaptation, not a vendored port of the official repo. It
keeps TEMPO's central idea — decompose the series into interpretable components
and process each with a SHARED frozen GPT-2 — while reusing this project's
GPT4TS backbone code and the study's additive entity-identity plumbing, so it
runs on the SAME pipeline as timellm/gpt4ts (only the "backbone"/architecture
differs, per the study design).

Design (differences from the paper are stated honestly):
* Decomposition: ``series_decomp`` (moving-average) into TREND + SEASONAL (2
  components). The paper uses an STL trend/seasonal/residual split; the residual
  is folded into the seasonal component here. moving_avg kernel = ``moving_avg``
  config (default 25).
* Each component has its OWN patch in/out projection but shares ONE frozen GPT-2
  (LayerNorm + positional embedding trainable, like GPT4TS). Per-component
  instance norm; the component predictions are summed to reconstruct the signal.
* Identity: additive only (same as GPT4TS) — the per-station vector is added to
  each component's patch embeddings. Prompt-only modes are rejected; soft-prompt
  identity is a documented extension, not implemented yet.
"""

from os import PathLike
from typing import Union

import torch
import torch.nn as nn

from liulian.models.torch.layers.decomposition import series_decomp


class Model(nn.Module):
    """TEMPO-style: seasonal-trend decomposition, each component through a shared frozen GPT-2."""

    #: Components the series is decomposed into (index-stable; drives per-component layers).
    COMPONENTS = ('trend', 'seasonal')

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        self.patch_size = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'gpt4ts_stride', self.patch_size)
        self.gpt_layers = getattr(configs, 'gpt_layers', getattr(configs, 'llm_layers', 6))

        kernel = int(getattr(configs, 'moving_avg', 25))
        if kernel % 2 == 0:  # series_decomp padding assumes an odd kernel
            kernel += 1
        self.decomp = series_decomp(kernel)

        self.cache_dir: Union[str, PathLike, None] = getattr(configs, 'cache_dir', None)

        # -- Shared frozen GPT-2 backbone (LayerNorm + wpe trainable) --------
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

        # -- Per-component patch in/out projections -------------------------
        self.num_patches = (self.seq_len - self.patch_size) // self.stride + 1
        self.in_layers = nn.ModuleDict({c: nn.Linear(self.patch_size, self.gpt2_dim) for c in self.COMPONENTS})
        self.out_layers = nn.ModuleDict(
            {c: nn.Linear(self.gpt2_dim * self.num_patches, self.pred_len) for c in self.COMPONENTS}
        )

        # -- Additive entity identity (same plumbing/rules as GPT4TS) -------
        import math as _math

        self.entity_id_mark_col: Union[int, None] = None
        self.identifier_mode: str = getattr(configs, 'identifier_mode', 'none')
        _prompt_only = {'entity_description', 'soft_prompt', 'text_embedding'}
        if self.identifier_mode in _prompt_only:
            raise ValueError(
                f'tempo (decomposition adapter) supports additive identity only; '
                f'identifier_mode={self.identifier_mode!r} is prompt-only. Use '
                f'none/embedding/random_embedding/onehot_embedding/sinusoidal_embedding, '
                f'or run prompt modes on timellm. (soft-prompt for tempo is a planned extension.)'
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

    def _forecast_component(self, comp_name, x_comp, station_ids):
        """One decomposed component [B, L, C] -> its prediction [B, pred_len, C]."""
        B, L, C = x_comp.shape
        means = x_comp.mean(1, keepdim=True).detach()
        x = x_comp - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev

        x = x.permute(0, 2, 1)  # [B, C, L]
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = x.reshape(B * C, self.num_patches, self.patch_size)

        x = self.in_layers[comp_name](x)  # [B*C, num_patches, gpt2_dim]
        x = self._inject_identity(x, station_ids)
        outputs = self.gpt2(inputs_embeds=x).last_hidden_state
        outputs = outputs.reshape(B * C, -1)
        outputs = self.out_layers[comp_name](outputs)  # [B*C, pred_len]
        outputs = outputs.reshape(B, C, -1).permute(0, 2, 1)  # [B, pred_len, C]
        return outputs * stdev + means

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        seasonal, trend = self.decomp(x_enc)  # each [B, L, C]
        station_ids = self._resolve_station_ids(x_mark_enc, entity_ids)
        components = {'trend': trend, 'seasonal': seasonal}
        pred = None
        for c in self.COMPONENTS:
            p = self._forecast_component(c, components[c], station_ids)
            pred = p if pred is None else pred + p
        return pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, entity_ids=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=entity_ids)
            return dec_out[:, -self.pred_len :, :]
        return None
