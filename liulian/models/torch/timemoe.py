"""
TimeMoE: Mixture of Time Series Experts

Paper: https://arxiv.org/abs/2409.16040
Original Implementation: Time-Series-Library
https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMoE.py

Uses pretrained Maple728/TimeMoE-50M model for zero-shot forecasting.
"""

import torch
from torch import nn
import numpy as np
from typing import Dict, Any
from liulian.models.torch.base_adapter import TorchModelAdapter


class Model(nn.Module):
    """TimeMoE model using pretrained Mixture-of-Experts"""
    
    def __init__(self, configs):
        """
        Initializes TimeMoE with pretrained model
        """
        super().__init__()
        # Import here to make it optional
        from transformers import AutoModelForCausalLM
        
        self.model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', trust_remote_code=True)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        x_enc = torch.reshape(x_enc, (B*C, L))
        output = self.model.generate(x_enc, max_new_tokens=self.pred_len)
        dec_out = torch.reshape(output, (B, output.shape[-1], C))
        dec_out = dec_out[:,-self.pred_len:, :]
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None


class TimeMoEAdapter(TorchModelAdapter):
    """
    Adapter for TimeMoE model to liulian ExecutableModel interface.
    
    Expected config parameters:
        - seq_len: Input sequence length
        - pred_len: Prediction sequence length
        - enc_in: Number of input features/variates
        - task_name: 'zero_shot_forecast'
        
    Note: Requires transformers package and downloads Maple728/TimeMoE-50M model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        default_config = {
            'task_name': 'zero_shot_forecast',
        }
        default_config.update(config)
        
        model = Model(self._dict_to_namespace(default_config))
        super().__init__(model, default_config)
    
    def _prepare_model_inputs(self, inputs: Dict[str, torch.Tensor]) -> tuple:
        """Prepare inputs for TimeMoE forward pass"""
        x_enc = inputs['x_enc']
        batch_size, seq_len, n_features = x_enc.shape
        
        x_mark_enc = inputs.get('x_mark_enc', torch.zeros(batch_size, seq_len, 1, device=x_enc.device))
        x_dec = inputs.get('x_dec', torch.zeros(batch_size, self.config['pred_len'], n_features, device=x_enc.device))
        x_mark_dec = inputs.get('x_mark_dec', torch.zeros(batch_size, self.config['pred_len'], 1, device=x_enc.device))
        
        return (x_enc, x_mark_enc, x_dec, x_mark_dec)
