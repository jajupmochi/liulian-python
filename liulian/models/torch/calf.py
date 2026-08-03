"""
CALF-style cross-modal (dual-branch) frozen-LLM forecaster.

Paper (idea): CALF — Cross-modal LLM Fine-Tuning for Time Series Forecasting,
https://arxiv.org/abs/2403.07300

FROM-SCRATCH adaptation (not a vendored port), on the SAME entry + pipeline as
timellm/gpt4ts/tempo/autotimes (only the architecture differs, per the study
design). CALF's central idea kept here: DUAL awareness — a TEMPORAL branch that
processes the raw patch features, and a CROSS-MODAL branch that reprograms the
patch features into the frozen LLM's WORD-EMBEDDING space (via cross-attention,
reusing timellm's ReprogrammingLayer) — both through the SAME frozen GPT-2, then
fused (summed heads).

Differences from the paper, stated honestly: CALF's feature/output/gradient
ALIGNMENT LOSSES are NOT implemented here — they belong in the TASK layer (tasks
own losses; models own only forward), so they are a task-side extension, not
part of this forward. This module provides the dual-branch cross-modal forward
only. Identity is additive (shared plumbing with the other adapters); the
reprogramming is the always-on architecture, not an identity mode, so prompt-only
identity modes are rejected.
"""

from os import PathLike
from typing import Union

import torch
import torch.nn as nn

from liulian.models.torch.layers.embed import TimeLLMPatchEmbedding
from liulian.models.torch.layers.standard_norm import Normalize
from liulian.models.torch.timellm import FlattenHead, ReprogrammingLayer


class Model(nn.Module):
    """CALF-style: temporal branch + cross-modal (word-embedding reprogrammed) branch, fused."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        if self.task_name not in ('long_term_forecast', 'short_term_forecast'):
            raise NotImplementedError('calf implements forecasting only.')
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = int(getattr(configs, 'd_model', 32))
        self.d_ff = int(getattr(configs, 'd_ff', 128))
        self.n_heads = int(getattr(configs, 'n_heads', 8))
        self.patch_len = int(getattr(configs, 'patch_len', 16))
        self.stride = int(getattr(configs, 'stride', 8))
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
        self.d_llm = self.gpt2.config.n_embd  # 768
        if self.d_ff > self.d_llm:
            raise ValueError(f'd_ff ({self.d_ff}) must be <= gpt2 hidden ({self.d_llm}).')
        for param in self.gpt2.parameters():
            param.requires_grad = False
        for layer in self.gpt2.h:
            for name, param in layer.named_parameters():
                if 'ln' in name:
                    param.requires_grad = True
        for param in self.gpt2.ln_f.parameters():
            param.requires_grad = True
        self.gpt2.wpe.weight.requires_grad = True

        # Cross-modal anchor: the frozen LLM word embeddings mapped to a small token set.
        self.word_embeddings = self.gpt2.get_input_embeddings().weight  # [vocab, d_llm]
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.patch_embedding = TimeLLMPatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.stride, float(getattr(configs, 'dropout', 0.1))
        )
        self.patch_nums = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # Cross-modal branch: reprogram patch (d_model) -> word-embedding space (d_llm).
        self.reprogramming_layer = ReprogrammingLayer(self.d_model, self.n_heads, self.d_ff, self.d_llm)
        # Temporal branch: project the raw patch (d_model) straight into the LLM input space.
        self.temporal_proj = nn.Linear(self.d_model, self.d_llm)

        self.head_cross = FlattenHead(
            self.enc_in, self.head_nf, self.pred_len, head_dropout=float(getattr(configs, 'dropout', 0.1))
        )
        self.head_temporal = FlattenHead(
            self.enc_in, self.head_nf, self.pred_len, head_dropout=float(getattr(configs, 'dropout', 0.1))
        )
        self.normalize_layers = Normalize(self.enc_in, affine=False)

        # -- Additive entity identity (d_model space, like timellm's numeric id) --
        import math as _math

        self.entity_id_mark_col: Union[int, None] = None
        self.identifier_mode: str = getattr(configs, 'identifier_mode', 'none')
        self._station_ids = None
        _prompt_only = {'entity_description', 'soft_prompt', 'text_embedding'}
        if self.identifier_mode in _prompt_only:
            raise ValueError(
                f'calf supports additive identity only; identifier_mode='
                f'{self.identifier_mode!r} is prompt-only. The cross-modal reprogramming '
                f'is the always-on architecture, not an identity mode. Use none/embedding/'
                f'random_embedding/onehot_embedding/sinusoidal_embedding, or run prompt '
                f'modes on timellm.'
            )
        self.entity_embedding: Union[nn.Embedding, None] = None
        self.transparent_proj: Union[nn.Linear, None] = None
        if self.identifier_mode in ('embedding', 'random_embedding'):
            _n = int(getattr(configs, 'num_entities', 0)) or None
            if _n is None:
                raise ValueError(f'identifier_mode={self.identifier_mode!r} requires configs.num_entities')
            self.entity_embedding = nn.Embedding(_n, self.d_model)
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
            self.transparent_proj = nn.Linear(feat.shape[1], self.d_model)

    def _resolve_station_ids(self, x_mark_enc, entity_ids):
        if entity_ids is not None:
            return entity_ids.reshape(-1).long()
        if self.entity_id_mark_col is not None and x_mark_enc is not None:
            return x_mark_enc[:, 0, self.entity_id_mark_col].long()
        return None

    def _inject_identity(self, enc_out, station_ids):
        """Add the per-station vector (d_model) to the patch embeddings before both branches."""
        if station_ids is None:
            return enc_out
        if self.entity_embedding is not None:
            enc_out = enc_out + self.entity_embedding(station_ids).unsqueeze(1)
        if self.transparent_proj is not None:
            enc_out = enc_out + self.transparent_proj(self.transparent_feat[station_ids]).unsqueeze(1)
        return enc_out

    def _branch_head(self, branch_input, n_vars, head):
        """[B*N, patch_nums, d_llm] -> [B, pred_len, N] via frozen GPT-2 + a FlattenHead."""
        out = self.gpt2(inputs_embeds=branch_input).last_hidden_state  # [B*N, patch_nums, d_llm]
        out = out[:, :, : self.d_ff]  # [B*N, patch_nums, d_ff]
        out = out.reshape(-1, n_vars, out.shape[-2], out.shape[-1])  # [B, N, patch_nums, d_ff]
        out = out.permute(0, 1, 3, 2).contiguous()  # [B, N, d_ff, patch_nums]
        out = head(out)  # [B, N, pred_len]
        return out.permute(0, 2, 1).contiguous()  # [B, pred_len, N]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        x = self.normalize_layers(x_enc, 'norm')  # [B, T, N]
        B, T, N = x.shape
        x = x.permute(0, 2, 1).contiguous()  # [B, N, T]
        enc_out, n_vars = self.patch_embedding(x)  # [B*N, patch_nums, d_model]
        enc_out = self._inject_identity(enc_out, self._resolve_station_ids(x_mark_enc, entity_ids))

        # Cross-modal branch: reprogram into the word-embedding space.
        source = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)  # [num_tokens, d_llm]
        cross_in = self.reprogramming_layer(enc_out, source, source)  # [B*N, patch_nums, d_llm]
        cross_out = self._branch_head(cross_in, n_vars, self.head_cross)

        # Temporal branch: raw patch features straight into the LLM.
        temporal_in = self.temporal_proj(enc_out)  # [B*N, patch_nums, d_llm]
        temporal_out = self._branch_head(temporal_in, n_vars, self.head_temporal)

        out = cross_out + temporal_out
        return self.normalize_layers(out, 'denorm')

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, entity_ids=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=entity_ids)
        return dec_out[:, -self.pred_len :, :]
