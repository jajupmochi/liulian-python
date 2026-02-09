"""
TimeLLM model adapter for liulian framework.

TimeLLM: Time Series Forecasting with Large Language Models
Paper: Time-LLM (ICLR 2024)

Adapted from Time-LLM:
https://github.com/KimMeen/Time-LLM/blob/main/models/TimeLLM.py
"""

from typing import Dict, Any
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    LlamaConfig, LlamaModel,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
)

from liulian.models.torch.base_adapter import TorchModelAdapter
from liulian.models.torch.layers.embed import PatchEmbedding
from liulian.models.torch.layers.normalization import Normalize


class FlattenHead(nn.Module):
    """Flatten head for TimeLLM predictions."""

    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0):
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
    """
    Reprogramming layer for aligning time series patches with LLM vocabulary space.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_llm: int = None,
        attention_dropout: float = 0.1
    ):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        Reprogram target embeddings using source embeddings.

        Args:
            target_embedding: Patch embeddings [B, L, d_model]
            source_embedding: LLM word embeddings [vocab_size, d_llm]
            value_embedding: LLM word embeddings [vocab_size, d_llm]

        Returns:
            Reprogrammed embeddings [B, L, d_llm]
        """
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        """Compute reprogramming attention."""
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class TimeLLMCore(nn.Module):
    """
    TimeLLM core model.

    TimeLLM reprograms frozen LLMs for time series forecasting by converting
    time series patches into the LLM's embedding space and using text prompts
    to describe the forecasting task and input statistics.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        d_ff: int = 128,
        n_heads: int = 8,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        llm_model: str = 'GPT2',
        llm_dim: int = 768,
        llm_layers: int = 6,
        prompt_domain: bool = False,
        content: str = None,
    ):
        """
        Initialize TimeLLM model.

        Args:
            seq_len: Input sequence length
            pred_len: Prediction horizon
            enc_in: Number of input channels
            d_model: Patch embedding dimension
            d_ff: Feed-forward dimension
            n_heads: Number of attention heads
            patch_len: Length of each patch
            stride: Stride for patching
            dropout: Dropout rate
            llm_model: LLM to use ('GPT2', 'BERT', or 'LLAMA')
            llm_dim: LLM hidden dimension
            llm_layers: Number of LLM layers to use
            prompt_domain: Whether to use domain-specific prompt
            content: Domain description content
        """
        super(TimeLLMCore, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.d_ff = d_ff
        self.d_llm = llm_dim
        self.patch_len = patch_len
        self.stride = stride
        self.top_k = 5

        # Initialize LLM and tokenizer
        if llm_model == 'GPT2':
            config = GPT2Config.from_pretrained('openai-community/gpt2')
            config.num_hidden_layers = llm_layers
            config.output_attentions = True
            config.output_hidden_states = True
            
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=config,
                )
            except:
                print("Downloading GPT2 model...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except:
                print("Downloading GPT2 tokenizer...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
                
        elif llm_model == 'BERT':
            config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            config.num_hidden_layers = llm_layers
            config.output_attentions = True
            config.output_hidden_states = True
            
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=config,
                )
            except:
                print("Downloading BERT model...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except:
                print("Downloading BERT tokenizer...")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise NotImplementedError(f"LLM model {llm_model} not implemented")

        # Set padding token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # Freeze LLM parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # Task description
        if prompt_domain and content:
            self.description = content
        else:
            self.description = (
                'The Electricity Transformer Temperature (ETT) is a crucial '
                'indicator in the electric power long-term deployment.'
            )

        self.dropout = nn.Dropout(dropout)

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            d_model, self.patch_len, self.stride, dropout
        )

        # Word embedding mapping
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # Reprogramming layer
        self.reprogramming_layer = ReprogrammingLayer(
            d_model, n_heads, d_ff, self.d_llm
        )

        # Output projection
        self.patch_nums = int((seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums
        self.output_projection = FlattenHead(
            enc_in, self.head_nf, self.pred_len, head_dropout=dropout
        )

        # Normalization
        self.normalize_layers = Normalize(enc_in, affine=False)

    def calculate_lags(self, x_enc):
        """Calculate autocorrelation lags using FFT."""
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def forward(self, x_enc):
        """
        Forward pass for forecasting.

        Args:
            x_enc: Encoder input [batch, seq_len, enc_in]

        Returns:
            Predictions [batch, pred_len, enc_in]
        """
        # Normalize
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc_reshaped = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # Compute statistics for prompt
        min_values = torch.min(x_enc_reshaped, dim=1)[0]
        max_values = torch.max(x_enc_reshaped, dim=1)[0]
        medians = torch.median(x_enc_reshaped, dim=1).values
        lags = self.calculate_lags(x_enc_reshaped)
        trends = x_enc_reshaped.diff(dim=1).sum(dim=1)

        # Create prompts
        prompt = []
        for b in range(x_enc_reshaped.shape[0]):
            min_val = str(min_values[b].tolist()[0])
            max_val = str(max_values[b].tolist()[0])
            median_val = str(medians[b].tolist()[0])
            lags_val = str(lags[b].tolist())
            trend = 'upward' if trends[b] > 0 else 'downward'
            
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {self.pred_len} steps "
                f"given the previous {self.seq_len} steps information; "
                f"Input statistics: min value {min_val}, max value {max_val}, "
                f"median value {median_val}, the trend of input is {trend}, "
                f"top 5 lags are: {lags_val}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        x_enc = x_enc_reshaped.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # Tokenize prompts
        prompt_tokens = self.tokenizer(
            prompt, return_tensors="pt", padding=True, 
            truncation=True, max_length=2048
        ).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(
            prompt_tokens.to(x_enc.device)
        )

        # Map word embeddings
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)

        # Patch embedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        
        # Reprogramming
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # Concatenate prompt and patches
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        
        # LLM forward
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        # Reshape and project
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # Denormalize
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out


class TimeLLMAdapter(TorchModelAdapter):
    """
    Adapter for TimeLLM model to liulian framework.
    """

    def _create_model(self, **model_params) -> nn.Module:
        """
        Create TimeLLM core model.

        Args:
            **model_params: Model parameters including:
                - seq_len: Input sequence length
                - pred_len: Prediction horizon
                - enc_in: Number of input channels
                - d_model: Patch embedding dimension (default: 32)
                - d_ff: Feed-forward dimension (default: 128)
                - n_heads: Number of attention heads (default: 8)
                - patch_len: Patch length (default: 16)
                - stride: Stride for patching (default: 8)
                - dropout: Dropout rate (default: 0.1)
                - llm_model: LLM to use (default: 'GPT2')
                - llm_dim: LLM dimension (default: 768 for GPT2)
                - llm_layers: Number of LLM layers (default: 6)
                - prompt_domain: Use domain prompt (default: False)
                - content: Domain description (optional)

        Returns:
            TimeLLMCore model instance
        """
        return TimeLLMCore(
            seq_len=model_params['seq_len'],
            pred_len=model_params['pred_len'],
            enc_in=model_params['enc_in'],
            d_model=model_params.get('d_model', 32),
            d_ff=model_params.get('d_ff', 128),
            n_heads=model_params.get('n_heads', 8),
            patch_len=model_params.get('patch_len', 16),
            stride=model_params.get('stride', 8),
            dropout=model_params.get('dropout', 0.1),
            llm_model=model_params.get('llm_model', 'GPT2'),
            llm_dim=model_params.get('llm_dim', 768),
            llm_layers=model_params.get('llm_layers', 6),
            prompt_domain=model_params.get('prompt_domain', False),
            content=model_params.get('content', None),
        )

    def forward(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        Forward pass through TimeLLM model.

        Args:
            batch: Dictionary containing:
                - x_enc: Encoder input [batch, seq_len, enc_in]

        Returns:
            Predictions as numpy array [batch, pred_len, enc_in]
        """
        # Convert input to torch
        x_enc = self._numpy_to_torch(batch['x_enc'])

        # Forward pass
        output = self.model(x_enc)

        # Convert back to numpy
        return self._torch_to_numpy(output)
