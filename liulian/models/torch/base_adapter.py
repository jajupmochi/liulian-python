"""PyTorch model adapter base class

This module provides infrastructure for adapting PyTorch models to the liulian
ExecutableModel interface. All PyTorch model adapters should inherit from
TorchModelAdapter base class.

The adapter is responsible for:
1. API mapping between dict-based interface and PyTorch model parameters
2. Configuration format conversion (Dict → Namespace/Config)
3. Device management (CPU/GPU)
4. Model execution and inference

Note: This adapter works directly with PyTorch tensors. No numpy conversion is performed.
"""

from __future__ import annotations

from typing import Any, Dict, Union
from types import SimpleNamespace

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        'PyTorch models require torch to be installed. '
        'Install with: pip install liulian[torch-models]'
    )

from liulian.models.base import ExecutableModel


class TorchModelAdapter(ExecutableModel):
    """Adapter base class from PyTorch models to ExecutableModel interface

    This base class provides common adaptation logic for wrapping PyTorch models.
    Subclasses are responsible for building the model and passing it to __init__.

    Attributes:
        device: PyTorch device (cuda/cpu)
        _model: Internal PyTorch model instance
        _config: Model configuration
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]) -> None:
        """Initialize the adapter with a model and configuration

        Args:
            model: PyTorch model instance (nn.Module)
            config: Model configuration dictionary
        """
        super().__init__()
        self.device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._model: nn.Module = model.to(self.device)
        self._config: Dict[str, Any] = config

    def configure(self, task: Any, config: Dict[str, Any]) -> None:
        """Update model configuration (optional, for runtime reconfiguration)

        Note: The model is already configured during __init__. This method
        is provided for compatibility with the ExecutableModel interface.

        Args:
            task: BaseTask instance describing the experiment task
            config: Hyperparameter configuration dictionary
        """
        # Update config if provided
        if config:
            self._config.update(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute forward pass with PyTorch tensors

        Args:
            batch: Input batch containing PyTorch tensors
                - "x_enc" or "X": Encoder input [batch_size, seq_len, n_features]
                - "x_mark_enc" or "X_mark": Encoder time marks (optional)
                - "x_dec": Decoder input (optional)
                - "x_mark_dec": Decoder time marks (optional)

        Returns:
            Dictionary containing prediction results:
                - "predictions": PyTorch tensor predictions
                - Additional outputs may vary by model
        """
        if self._model is None:
            raise RuntimeError('Model not properly initialized.')

        # Forward pass in evaluation mode
        self._model.eval()
        with torch.no_grad():
            output = self._forward_torch_model(batch)

        return output

    def run(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convenience alias for forward() method

        This provides a more intuitive API for users who expect a run() method.

        Args:
            batch: Input batch containing PyTorch tensors

        Returns:
            Dictionary containing prediction results
        """
        return self.forward(batch)

    def _forward_torch_model(
        self, torch_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Call PyTorch model's forward method

        This method handles different calling conventions for different models.
        Subclasses can override this method to adapt to specific model APIs.

        Args:
            torch_batch: PyTorch Tensor batch with keys like 'x_enc', 'x_mark_enc', etc.

        Returns:
            Dictionary with at least a 'predictions' key containing output tensor
        """

        # Helper: convert numpy → torch if needed, move to device
        def _to_tensor(v):
            if v is None:
                return None
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if isinstance(v, torch.Tensor):
                return v.float().to(self.device)
            return v

        # Helper: get first non-None value from dict by multiple key names
        def _get(*keys):
            for k in keys:
                v = torch_batch.get(k)
                if v is not None:
                    return v
            return None

        # Extract common time series model inputs (support both naming conventions)
        x_enc = _to_tensor(_get('x_enc', 'X'))
        x_mark_enc = _to_tensor(_get('x_mark_enc', 'X_mark'))
        x_dec = _to_tensor(_get('x_dec', 'X_dec'))
        x_mark_dec = _to_tensor(_get('x_mark_dec', 'X_mark_dec'))
        mask = _to_tensor(_get('mask'))

        # Call model - most time series models expect 4 arguments + optional mask
        # Pass all parameters including None - let the model handle them
        output = self._model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

        # Normalize output to dictionary format
        if isinstance(output, torch.Tensor):
            return {'predictions': output}
        elif isinstance(output, dict):
            # Ensure 'predictions' key exists
            if 'predictions' not in output and len(output) > 0:
                # Use first tensor as predictions
                first_tensor = next(
                    (v for v in output.values() if isinstance(v, torch.Tensor)), None
                )
                if first_tensor is not None:
                    output['predictions'] = first_tensor
            return output
        else:
            raise TypeError(f'Model returned unsupported type: {type(output)}')

    def _dict_to_namespace(self, config: Dict[str, Any]) -> SimpleNamespace:
        """Convert configuration dictionary to Namespace

        Many PyTorch time series models use argparse.Namespace format configs.
        This method recursively converts nested dictionaries.

        Args:
            config: Configuration dictionary

        Returns:
            SimpleNamespace object
        """
        namespace = SimpleNamespace()
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(namespace, key, self._dict_to_namespace(value))
            else:
                setattr(namespace, key, value)
        return namespace

    def save(self, path: str) -> None:
        """Save model weights

        Args:
            path: Save path
        """
        if self._model is not None:
            torch.save(
                {
                    'model_state_dict': self._model.state_dict(),
                    'config': self._config,
                },
                path,
            )

    def load(self, path: str) -> None:
        """Load model weights

        Args:
            path: Model file path

        Raises:
            RuntimeError: If model not initialized before calling load
        """
        if self._model is None:
            raise RuntimeError(
                'Model must be initialized before loading weights. '
                'The adapter requires a model instance to be provided during __init__.'
            )

        checkpoint = torch.load(path, map_location=self.device)
        self._model.load_state_dict(checkpoint['model_state_dict'])

        # Update config if present
        if 'config' in checkpoint:
            self._config = checkpoint['config']

    def capabilities(self) -> Dict[str, bool]:
        """Return model capabilities

        Returns:
            Capabilities dictionary
        """
        return {
            'supports_training': False,  # Most adapters are for inference only
            'supports_incremental': False,
            'requires_gpu': torch.cuda.is_available(),
            'framework': 'pytorch',
        }
