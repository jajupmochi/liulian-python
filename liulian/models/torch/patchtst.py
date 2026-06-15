"""PatchTST with optional entity integration modes.

Paper: https://arxiv.org/pdf/2211.14730.pdf
Original Implementation: Time-Series-Library
https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py

Supported entity settings for ``model: patchtst``:

* ``identifier_mode: none`` — plain PatchTST.
* ``identifier_mode: embedding`` + ``id_integration: concat_to_x``
    — legacy wrapper-based integration at the time-step level.
* ``identifier_mode: embedding`` + ``id_integration: add_after_patch``
    — add a learned per-channel embedding in ``d_model`` space after
        patching and patch projection.
* Transparent modes (``onehot`` / ``sinusoidal`` / ``random`` /
    ``coordinates`` / ``numeric_id``) + ``id_integration: add_after_patch``
    — project the FIXED per-channel identifier table ``(N, D)`` to
        ``d_model`` via a learned ``nn.Linear`` and add it in patch-token
        space, downstream of the per-channel instance-norm. This lets the
        transparent identifiers survive instance normalization (which would
        otherwise subtract away any pre-norm per-channel constant), exactly
        like the learned ``embedding`` path.
"""

import torch
from torch import nn
from typing import Dict, Any
from liulian.models.torch.layers.transformer_blocks import Encoder, EncoderLayer
from liulian.models.torch.layers.attention import FullAttention, AttentionLayer
from liulian.models.torch.layers.embed import PatchEmbedding
from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.entity_mixin import EntityAwareMixin, _build_channel_features


# Transparent identifier modes that can also use ``add_after_patch``.
# These build a FIXED ``(N, D)`` table (no learned lookup) projected to
# ``d_model`` and added in patch-token space.
_TRANSPARENT_ADD_AFTER_PATCH_MODES = frozenset({'onehot', 'sinusoidal', 'random', 'coordinates', 'numeric_id'})


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.identifier_mode = getattr(configs, 'identifier_mode', 'none')
        self.id_integration = getattr(configs, 'id_integration', 'concat_to_x')
        padding = stride

        # ``add_after_patch`` has two flavours, both injecting a per-channel
        # vector in ``d_model`` patch-token space (downstream of the
        # instance-norm in ``forecast``):
        #   * embedding  → learned ``nn.Embedding`` lookup.
        #   * transparent → FIXED ``(N, D)`` identifier table projected to
        #     ``d_model`` via a learned ``nn.Linear`` (the table itself is a
        #     non-learned buffer; only the projection is trained).
        self._use_embedding_after_patch = (
            self.identifier_mode == 'embedding' and self.id_integration == 'add_after_patch'
        )
        self._use_transparent_after_patch = (
            self.identifier_mode in _TRANSPARENT_ADD_AFTER_PATCH_MODES and self.id_integration == 'add_after_patch'
        )
        # Union flag — gates the early return in ``_inject_entity_after_patch``
        # and the multi_channel requirement shared by both flavours.
        self._use_add_after_patch = self._use_embedding_after_patch or self._use_transparent_after_patch
        if self._use_add_after_patch and getattr(configs, 'split_mode', None) != 'multi_channel':
            raise ValueError("PatchTST only supports id_integration='add_after_patch' in split_mode='multi_channel'.")

        # patching and embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)
        if self._use_embedding_after_patch:
            num_stations = getattr(configs, 'enc_in', 1)
            self.entity_embedding = nn.Embedding(num_stations, configs.d_model)
        elif self._use_transparent_after_patch:
            num_stations = getattr(configs, 'enc_in', 1)
            # Build the FIXED per-channel identifier table once, reusing the
            # canonical feature formulas from entity_mixin (onehot/sinusoidal/
            # random/coordinates/numeric_id). Registered as a buffer so it
            # moves with the model (.to(device)) and lands in state_dict.
            id_table = _build_channel_features(
                self.identifier_mode,
                num_stations,
                sinusoidal_dim=int(getattr(configs, 'sinusoidal_dim', 16)),
                random_dim=int(getattr(configs, 'random_identifier_dim', 16)),
                random_seed=int(getattr(configs, 'random_identifier_seed', 2026)),
                coordinates=getattr(configs, 'coordinates', None),
                station_ids=getattr(configs, 'station_ids', None),
            )
            self.register_buffer('id_table', id_table)  # (N, D)
            self.id_proj = nn.Linear(id_table.shape[1], configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)),
        )

        # Prediction Head
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.pred_len,
                head_dropout=configs.dropout,
            )
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(
                configs.enc_in,
                self.head_nf,
                configs.seq_len,
                head_dropout=configs.dropout,
            )
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def _inject_entity_after_patch(
        self,
        enc_out: torch.Tensor,
        n_vars: int,
    ) -> torch.Tensor:
        """Add per-channel entity vectors in patch-token space.

        ``enc_out`` is ``(batch * n_vars, patch_num, d_model)`` with the
        channels interleaved as ``arange(n_vars).repeat(batch)``. Both the
        learned-embedding and the transparent-projection flavours produce a
        per-channel ``(n_vars, d_model)`` matrix that is indexed by channel
        and broadcast over the ``patch_num`` axis.
        """
        if not self._use_add_after_patch:
            return enc_out
        batch_size = enc_out.shape[0] // n_vars
        ids = torch.arange(n_vars, device=enc_out.device).repeat(batch_size)
        if self._use_transparent_after_patch:
            # (N, d_model) projection of the fixed identifier table.
            per_channel = self.id_proj(self.id_table)  # (N, d_model)
            emb = per_channel[ids]  # (batch * n_vars, d_model)
        else:
            emb = self.entity_embedding(ids)  # (batch * n_vars, d_model)
        return enc_out + emb.unsqueeze(1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self._inject_entity_after_patch(enc_out, n_vars)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self._inject_entity_after_patch(enc_out, n_vars)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self._inject_entity_after_patch(enc_out, n_vars)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self._inject_entity_after_patch(enc_out, n_vars)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None


class PatchTSTAdapter(EntityAwareMixin, TorchModelAdapter):
    """
    Adapter for PatchTST model to liulian ExecutableModel interface.

    Expected config parameters:
        - seq_len: Input sequence length
        - pred_len: Prediction sequence length
        - enc_in: Number of input features/variates
        - d_model: Model dimension (default: 128)
        - n_heads: Number of attention heads (default: 16)
        - e_layers: Number of encoder layers (default: 3)
        - d_ff: Feed-forward dimension (default: 256)
        - dropout: Dropout rate (default: 0.2)
        - activation: Activation function (default: 'gelu')
        - factor: Attention factor (default: 1)
        - patch_len: Patch length (default: 16)
        - stride: Patch stride (default: 8)
        - task_name: Task type (default: 'long_term_forecast')
    """

    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'activation': 'gelu',
            'factor': 1,
            'patch_len': 16,
            'stride': 8,
            'task_name': 'long_term_forecast',
        }
        default_config.update(config)

        model_cfg = self._entity_model_config(default_config)
        model = Model(
            self._dict_to_namespace(model_cfg),
            patch_len=model_cfg['patch_len'],
            stride=model_cfg['stride'],
        )
        super().__init__(model, default_config)
        self._init_entity_support(default_config)

    def _prepare_model_inputs(self, inputs: Dict[str, torch.Tensor]) -> tuple:
        """Prepare inputs for PatchTST forward pass"""
        x_enc = inputs['x_enc']
        batch_size, seq_len, n_features = x_enc.shape

        x_mark_enc = inputs.get('x_mark_enc', torch.zeros(batch_size, seq_len, 1, device=x_enc.device))
        x_dec = inputs.get(
            'x_dec',
            torch.zeros(batch_size, self.config['pred_len'], n_features, device=x_enc.device),
        )
        x_mark_dec = inputs.get(
            'x_mark_dec',
            torch.zeros(batch_size, self.config['pred_len'], 1, device=x_enc.device),
        )

        return (x_enc, x_mark_enc, x_dec, x_mark_dec)
