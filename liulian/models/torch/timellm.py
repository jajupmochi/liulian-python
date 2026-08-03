"""
Time-LLM: Time Series Forecasting by Reprogramming Large Language Models

Paper: https://arxiv.org/abs/2310.01728
Original Implementation: Time-LLM
https://github.com/KimMeen/Time-LLM

Time-LLM reprograms frozen LLMs (LLAMA, GPT2, BERT) for time series forecasting
using patch-based embeddings and cross-attention reprogramming.
"""

from math import sqrt
from os import PathLike
from typing import Dict, Any, Union

import torch
import torch.nn as nn

from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.entity_mixin import EntityAwareMixin
from liulian.models.torch.layers.embed import TimeLLMPatchEmbedding
from liulian.models.torch.layers.standard_norm import Normalize


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        # target_embedding is the time series patch embeddings,
        # source_embedding and value_embedding are the LLM vocabulary embeddings.
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape  # S: num_tokens for text protypes embeddings (from LLM vocabulary)
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)  # [B, L, H * D_v]

        return self.out_projection(out)  # [B, L, D_llm]

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1.0 / sqrt(E)

        scores = torch.einsum('blhe,she->bhls', target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum('bhls,she->blhe', A, value_embedding)

        return reprogramming_embedding


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.cache_dir: Union[str, PathLike, None] = getattr(configs, 'cache_dir', None)

        # H4 entity_description (Time-LLM-only text identity): optional list of
        # per-channel natural-language descriptions injected into the LLM prompt,
        # one entry per channel for multi_channel (channel = entity). ``None`` →
        # baseline prompt, byte-identical to the verified V1/V2 path. Set by the
        # experiment harness/adapter; see forecast() for the b % N injection.
        self.entity_descriptions: Union[list, None] = None
        # H4 per-sample path: column index in x_mark_enc that carries the
        # per-sample entity id (which channel/station this CI sample is). Set by
        # the harness when the loader exposes it; None -> per-channel b % N path.
        self.entity_id_mark_col: Union[int, None] = None

        # H2 embedding injection (NUMERIC identity): a learned per-entity vector
        # ADDED to the patch embeddings (post-norm) — the numeric counterpart of
        # the H4 text identity. Built only for identifier_mode='embedding' (needs
        # configs.num_entities); None otherwise so other modes are unaffected.
        self.identifier_mode: str = getattr(configs, 'identifier_mode', 'none')
        if self.identifier_mode in ('embedding', 'random_embedding'):
            _n_ent = getattr(configs, 'num_entities', None)
            if _n_ent is None:
                raise ValueError(f'identifier_mode={self.identifier_mode!r} requires configs.num_entities')
            self.entity_embedding: Union[nn.Embedding, None] = nn.Embedding(int(_n_ent), int(configs.d_model))
            if self.identifier_mode == 'random_embedding':
                # Capacity-matched control: FIXED random per-entity vectors (0
                # learnable params) with the same architecture as 'embedding'.
                # If this helps too, the gain is per-entity DISTINCTNESS
                # (identity), not the learnable capacity of 'embedding'.
                self.entity_embedding.weight.requires_grad_(False)
        else:
            self.entity_embedding = None

        # Level-A `soft_prompt` (LEARNED identity, PREFIX position): a per-entity
        # block of learnable continuous tokens prepended to the LLM input sequence.
        # This is the "learned × prefix" cell of the representation×position 2x2 —
        # the counterpart of entity_description (text × prefix) and numeric_embedding
        # (learned × additive). Built only for this mode; None otherwise.
        self.soft_prompt_len: int = int(getattr(configs, 'soft_prompt_len', 8))
        if self.identifier_mode == 'soft_prompt':
            _n_ent = getattr(configs, 'num_entities', None)
            if _n_ent is None:
                raise ValueError("identifier_mode='soft_prompt' requires configs.num_entities")
            # (num_entities, soft_prompt_len, d_llm); small init like a prompt embedding.
            self.soft_prompt = nn.Parameter(torch.randn(int(_n_ent), self.soft_prompt_len, int(configs.llm_dim)) * 0.02)
        else:
            self.soft_prompt = None

        # Import transformers here to make it optional
        from transformers import (
            LlamaConfig,
            LlamaModel,
            LlamaTokenizer,
            GPT2Config,
            GPT2Model,
            GPT2Tokenizer,
            BertConfig,
            BertModel,
            BertTokenizer,
            AutoConfig,
            AutoModel,
            AutoTokenizer,
            AutoModelForCausalLM,
        )

        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            # # todo: use this?
            # self.llama_config = LlamaConfig.from_pretrained('meta-llama/Llama-2-7b-hf')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print('Local model files not found. Attempting to download...')
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print('Local tokenizer files not found. Attempting to download them..')
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print('Local model files not found. Attempting to download...')
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print('Local tokenizer files not found. Attempting to download them..')
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print('Local model files not found. Attempting to download...')
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print('Local tokenizer files not found. Attempting to download them..')
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                )

        elif configs.llm_model == 'TINYLLAMA':
            self.llm_config = AutoConfig.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
            self.llm_config.num_hidden_layers = configs.llm_layers
            self.llm_config.attn_implementation = 'eager'
            self.llm_config.output_attentions = True
            self.llm_config.output_hidden_states = True
            try:
                self.llm_model = AutoModel.from_pretrained(
                    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llm_config,
                )
            except Exception as e:
                print(f'Failed to load local TinyLLaMA: {e}')
                print('Attempting to download...')
                self.llm_model = AutoModel.from_pretrained(
                    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llm_config,
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as e:
                print(f'Local TinyLLaMA tokenizer not found: {e}')
                print('Attempting to download...')
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                    trust_remote_code=True,
                    local_files_only=False,
                )

        elif configs.llm_model == 'QWEN':
            self.llm_config = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-7B-Chat').config
            self.llm_config.num_hidden_layers = configs.llm_layers
            self.llm_config.output_attentions = True
            self.llm_config.output_hidden_states = True
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    'Qwen/Qwen-7B-Chat',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as e:
                print(f'Failed to load local Qwen: {e}')
                print('Attempting to download...')
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    'Qwen/Qwen-7B-Chat',
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=False,
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'Qwen/Qwen-7B-Chat', trust_remote_code=True, local_files_only=True
                )
            except Exception as e:
                print(f'Local Qwen tokenizer not found: {e}')
                print('Attempting to download...')
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'Qwen/Qwen-7B-Chat', trust_remote_code=True, local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:  # todo: what is this?
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = TimeLLMPatchEmbedding(
            int(configs.d_model),
            int(self.patch_len),
            int(self.stride),
            int(self.stride),
            float(configs.dropout),
        )

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(
                configs.enc_in,
                self.head_nf,
                self.pred_len,
                head_dropout=configs.dropout,
            )
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, entity_ids=None):
        # entity_ids: (B,) per-sample station index. The pipeline trainer passes it as a
        # kwarg (batch element), NOT inside x_mark. The harness instead put it in an
        # x_mark column (entity_id_mark_col). forecast() accepts either and prefers the
        # explicit kwarg.
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=entity_ids)
            return dec_out[:, -self.pred_len :, :]
        return None

    @staticmethod
    def _resolve_entity_descs(
        entity_descriptions: Union[list, None],
        entity_id_mark_col: Union[int, None],
        x_mark_enc: Any,
        n_series: int,
        n_channels: int,
        station_ids: Any = None,
    ) -> list:
        """Resolve the H4 entity description for each of the ``n_series`` flattened
        series (returns a length-``n_series`` list of ``str`` or ``None``).

        Three cases, in order:

        - ``entity_descriptions is None`` -> all ``None`` (baseline; the prompt is
          byte-identical to the verified pre-H4 path).
        - ``entity_id_mark_col is not None`` -> **per-sample** id read from
          ``x_mark_enc[:, 0, entity_id_mark_col]``. This is the correct path for
          this repo's **channel-independent** loaders (each sample is one
          channel/station; ``n_channels == 1``, ``n_series == batch``).
        - else -> **per-channel** ``b % n_channels`` (a true multi_channel layout
          where channels live in the feature axis).

        Fails loud on a length / out-of-range id mismatch rather than silently
        injecting the wrong identity.
        """
        if entity_descriptions is None:
            return [None] * n_series
        n_desc = len(entity_descriptions)
        # Prefer the unified per-sample ids (station_ids = entity_ids kwarg or x_mark,
        # resolved once in forecast). Fall back to the legacy x_mark read for callers
        # that still pass only entity_id_mark_col.
        if station_ids is not None:
            ids = station_ids.reshape(-1).long().tolist()
            if len(ids) != n_series:
                raise ValueError(f'per-sample entity ids ({len(ids)}) != n_series ({n_series}).')
            resolved = []
            for i in ids:
                if not 0 <= i < n_desc:
                    raise ValueError(f'entity id {i} out of range for {n_desc} descriptions.')
                resolved.append(entity_descriptions[i])
            return resolved
        if entity_id_mark_col is not None:
            ids = x_mark_enc[:, 0, entity_id_mark_col].long().tolist()
            if len(ids) != n_series:
                raise ValueError(
                    f'per-sample entity ids ({len(ids)}) != n_series ({n_series}); '
                    'the entity_id_mark_col path expects a channel-independent '
                    'layout (one series per sample).'
                )
            resolved = []
            for i in ids:
                if not 0 <= i < n_desc:
                    raise ValueError(f'entity id {i} out of range for {n_desc} descriptions.')
                resolved.append(entity_descriptions[i])
            return resolved
        if n_desc != n_channels:
            raise ValueError(
                f'entity_descriptions has {n_desc} entries but the model sees '
                f'N={n_channels} channels; the per-channel path needs one per channel.'
            )
        return [entity_descriptions[b % n_channels] for b in range(n_series)]

    @staticmethod
    def _compose_prompt(
        description: str,
        entity_desc: Union[str, None],
        pred_len: str,
        seq_len: str,
        min_v: str,
        max_v: str,
        median_v: str,
        trend_up: bool,
        lags_str: str,
    ) -> str:
        """Build one channel's Time-LLM text prompt.

        ``entity_desc`` is the H4 per-channel natural-language identity. When it
        is ``None`` or empty the inserted segment is the empty string, so the
        returned prompt is byte-identical to the original (pre-H4) prompt — this
        preserves the V1/V2 bit-exact reproduction on the ``none`` path.
        """
        entity_str = f'Entity description: {entity_desc}; ' if entity_desc else ''
        return (
            f'<|start_prompt|>Dataset description: {description}'
            f'Task description: forecast the next {pred_len} steps given the previous {seq_len} steps information; '
            f'{entity_str}'
            'Input statistics: '
            f'min value {min_v}, '
            f'max value {max_v}, '
            f'median value {median_v}, '
            f'the trend of input is {"upward" if trend_up else "downward"}, '
            f'top 5 lags are : {lags_str}<|<end_prompt>|>'
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, entity_ids=None):
        # Unified per-sample station id source: prefer the explicit entity_ids kwarg
        # (pipeline trainer path); fall back to the x_mark column (harness path). Either
        # way _station_ids is a Long (B,) tensor or None (baseline / no id available).
        self._station_ids = None
        if entity_ids is not None:
            self._station_ids = entity_ids.reshape(-1).long()
        elif self.entity_id_mark_col is not None and x_mark_enc is not None:
            self._station_ids = x_mark_enc[:, 0, self.entity_id_mark_col].long()

        x_enc = self.normalize_layers(x_enc, 'norm')  # x_enc: [B, T, N_f]

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        # H4: resolve a per-series entity description (or None for the baseline).
        # This harness is channel-independent (N=1, one channel/station per
        # sample), so identity comes PER-SAMPLE from an x_mark column wired by the
        # loader (entity_id_mark_col); b % N is the fallback for a true
        # multi_channel layout. None -> byte-identical to the verified prompt.
        entity_desc_for = self._resolve_entity_descs(
            self.entity_descriptions,
            self.entity_id_mark_col,
            x_mark_enc,
            x_enc.shape[0],
            N,
            station_ids=self._station_ids,
        )
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            entity_desc = entity_desc_for[b]
            prompt_ = self._compose_prompt(
                self.description,
                entity_desc,
                str(self.pred_len),
                str(self.seq_len),
                min_values_str,
                max_values_str,
                median_values_str,
                bool(trends[b] > 0),
                lags_values_str,
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        if self.tokenizer is None or not callable(self.tokenizer):
            raise RuntimeError(
                'Tokenizer is not properly initialized or not callable. Please check LLM model and tokenizer setup.'
            )
        prompt_output = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=2048)
        if not hasattr(prompt_output, 'input_ids'):
            raise RuntimeError(
                f"Tokenizer output does not have 'input_ids'. Type: {type(prompt_output)}. Please check tokenizer type and initialization."
            )
        prompt = prompt_output.input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        # [num_tokens, d_vocab]:
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # x_enc: [B, N_f, T]， enc_out: [B*N_f, num_patches, d_model]:
        # enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))  # todo: maybe needed autocast as in timellm?
        enc_out, n_vars = self.patch_embedding(x_enc)  # keep as float32, do not cast to bfloat16
        # H2: add the learned per-entity embedding to the patch embeddings
        # (post-norm numeric identity). enc_out is [B*N, num_patches, d_model];
        # this harness is channel-independent (N=1) so B*N == B and the per-sample
        # ids (B,) read from the x_mark column align row-for-row.
        if self.entity_embedding is not None and self._station_ids is not None:
            enc_out = enc_out + self.entity_embedding(self._station_ids).unsqueeze(1)
        enc_out = self.reprogramming_layer(
            enc_out, source_embeddings, source_embeddings
        )  # enc_out: [B*N_f, num_patches, d_llm]
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # Level-A soft_prompt (LEARNED × PREFIX): prepend this station's learnable
        # prefix tokens to the LLM input. Prepending (not appending) is safe because
        # the output head below slices the LAST patch_nums positions, so the patch
        # tokens keep their place. Per-sample id from the same x_mark column as the
        # numeric identity; N=1 channel-independent so ids (B,) align row-for-row.
        if self.soft_prompt is not None and self._station_ids is not None:
            soft_tokens = self.soft_prompt[self._station_ids]  # (B, soft_prompt_len, d_llm)
            llama_enc_out = torch.cat([soft_tokens, llama_enc_out], dim=1)
        # dec_out = self.llm_model(inputs_embeds=llama_enc_out.to(
        #     self.llm_model.dtype if hasattr(self.llm_model, 'dtype') else llama_enc_out.dtype
        # )).last_hidden_state  # [B*N_f, prompt_len + num_patches, d_llm]
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # dec_out = dec_out.float()
        dec_out = dec_out[:, :, : self.d_ff]  # [B*N_f, prompt_len + num_patches, d_ff]

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()  # [B, N_f, d_ff, prompt_len + num_patches]

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums :])  # [B, N_f, pred_len]
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class TimeLLMAdapter(EntityAwareMixin, TorchModelAdapter):
    """
    Adapter for Time-LLM model to liulian ExecutableModel interface.

    Expected config parameters:
        - seq_len: Input sequence length
        - pred_len: Prediction sequence length
        - enc_in: Number of input features/variates
        - d_model: Patch embedding dimension (default: 32)
        - d_ff: Feed-forward dimension (default: 128)
        - n_heads: Number of attention heads (default: 8)
        - llm_model: LLM backbone - 'LLAMA', 'GPT2', or 'BERT' (default: 'GPT2')
        - llm_dim: LLM hidden dimension - 4096 for LLAMA, 768 for GPT2/BERT (default: 768)
        - llm_layers: Number of LLM layers to use (default: 6)
        - patch_len: Patch length (default: 16)
        - stride: Patch stride (default: 8)
        - dropout: Dropout rate (default: 0.1)
        - prompt_domain: Use domain-specific prompt (default: False)
        - content: Custom dataset description (default: ETT description)
        - task_name: Task type (default: 'long_term_forecast')

    Note: Requires transformers package. Downloads pretrained LLM on first use.
    """

    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'd_model': 32,
            'd_ff': 128,
            'n_heads': 8,
            'llm_model': 'GPT2',
            'llm_dim': 768,  # 768 for GPT2/BERT, 4096 for LLAMA
            'llm_layers': 6,
            'patch_len': 16,
            'stride': 8,
            'dropout': 0.1,
            'prompt_domain': False,
            'content': 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.',
            'task_name': 'long_term_forecast',
        }
        default_config.update(config)

        model = Model(
            self._dict_to_namespace(default_config),
            patch_len=default_config['patch_len'],
            stride=default_config['stride'],
        )
        super().__init__(model, default_config)
        self._init_entity_support(default_config)

    def _prepare_model_inputs(self, inputs: Dict[str, torch.Tensor]) -> tuple:
        """Prepare inputs for Time-LLM forward pass"""
        x_enc = inputs['x_enc']
        batch_size, seq_len, n_features = x_enc.shape

        x_mark_enc = inputs.get('x_mark_enc', torch.zeros(batch_size, seq_len, 1, device=x_enc.device))
        x_dec = inputs.get(
            'x_dec',
            torch.zeros(batch_size, self._config['pred_len'], n_features, device=x_enc.device),
        )
        x_mark_dec = inputs.get(
            'x_mark_dec',
            torch.zeros(batch_size, self._config['pred_len'], 1, device=x_enc.device),
        )

        return (x_enc, x_mark_enc, x_dec, x_mark_dec)
