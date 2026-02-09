"""PyTorch model adapter base class

This module provides infrastructure for adapting PyTorch models to the liulian 
ExecutableModel interface. All PyTorch model adapters should inherit from 
TorchModelAdapter base class.

The adapter is responsible for:
1. Data type conversion between NumPy and PyTorch Tensors
2. API mapping between dict-based interface and PyTorch model parameters
3. Configuration format conversion (Dict → Namespace/Config)
4. Device management (CPU/GPU)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Union
from types import SimpleNamespace

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch models require torch to be installed. "
        "Install with: pip install liulian[torch-models]"
    )

from liulian.models.base import ExecutableModel


class TorchModelAdapter(ExecutableModel):
    """Adapter base class from PyTorch models to ExecutableModel interface
    
    This base class provides common adaptation logic. Subclasses only need to 
    implement the _build_model() method to construct specific PyTorch model instances.
    
    Attributes:
        device: PyTorch device (cuda/cpu)
        _model: Internal PyTorch model instance
        _config: Model configuration
    """
    
    def __init__(self) -> None:
        """Initialize the adapter"""
        super().__init__()
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: Optional[nn.Module] = None
        self._config: Optional[Dict[str, Any]] = None
        
    def configure(self, task: Any, config: Dict[str, Any]) -> None:
        """Configure the model
        
        Args:
            task: BaseTask instance describing the experiment task
            config: Hyperparameter configuration dictionary
        """
        self._config = config
        
        # Convert config format (Dict → Namespace)
        torch_config = self._dict_to_namespace(config)
        
        # Build PyTorch model
        self._model = self._build_model(torch_config)
        
        # Move to device
        if self._model is not None:
            self._model = self._model.to(self.device)
            
    @abstractmethod
    def _build_model(self, config: SimpleNamespace) -> nn.Module:
        """Build concrete PyTorch model instance
        
        Subclasses must implement this method to create the specific model.
        
        Args:
            config: Model configuration (Namespace format)
            
        Returns:
            PyTorch model instance
        """
        pass
        
    def forward(self, batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Execute forward pass
        
        Args:
            batch: Input batch containing NumPy arrays
                - "X": Input features [batch_size, seq_len, n_features]
                - "X_mark": Time marks (optional)
                - "y": Target values (during training)
                - "y_mark": Target time marks (optional)
                
        Returns:
            Dictionary containing prediction results:
                - "predictions": NumPy array predictions
                - "diagnostics": Diagnostic information (optional)
        """
        if self._model is None:
            raise RuntimeError("Model not configured. Call configure() first.")
            
        # NumPy → PyTorch
        torch_batch = self._numpy_to_torch(batch)
        
        # Forward pass
        self._model.eval()
        with torch.no_grad():
            # Call internal model's forward
            output = self._forward_torch_model(torch_batch)
            
        # PyTorch → NumPy
        result = self._torch_to_numpy(output)
        
        return result
        
    def _forward_torch_model(
        self, 
        torch_batch: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Call PyTorch model's forward method
        
        This method handles different calling conventions for different models.
        Subclasses can override this method to adapt to specific model APIs.
        
        Args:
            torch_batch: PyTorch Tensor batch
            
        Returns:
            Model output (Tensor or Dict)
        """
        # Extract common time series model inputs
        x_enc = torch_batch.get("X")  # encoder input
        x_mark_enc = torch_batch.get("X_mark")  # encoder time marks
        x_dec = torch_batch.get("X_dec")  # decoder input
        x_mark_dec = torch_batch.get("X_mark_dec")  # decoder time marks
        
        # Most models use this calling convention
        if x_dec is not None and x_mark_enc is not None:
            output = self._model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif x_mark_enc is not None:
            output = self._model(x_enc, x_mark_enc, None, None)
        else:
            output = self._model(x_enc)
            
        return output
        
    def _numpy_to_torch(
        self, 
        batch: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Convert NumPy batch to PyTorch Tensors
        
        Args:
            batch: Dictionary of NumPy arrays
            
        Returns:
            Dictionary of PyTorch Tensors
        """
        torch_batch = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).float().to(self.device)
                torch_batch[key] = tensor
            else:
                torch_batch[key] = value
        return torch_batch
        
    def _torch_to_numpy(
        self, 
        output: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """Convert PyTorch output to NumPy format
        
        Args:
            output: PyTorch Tensor or dictionary of Tensors
            
        Returns:
            Dictionary containing prediction results
        """
        if isinstance(output, torch.Tensor):
            # Simple Tensor output
            predictions = output.cpu().detach().numpy()
            return {"predictions": predictions}
        elif isinstance(output, dict):
            # Dictionary output
            result = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.cpu().detach().numpy()
                else:
                    result[key] = value
            # Ensure predictions key exists
            if "predictions" not in result and len(result) > 0:
                # Use first Tensor as predictions
                first_array = next(
                    (v for v in result.values() if isinstance(v, np.ndarray)), 
                    None
                )
                if first_array is not None:
                    result["predictions"] = first_array
            return result
        else:
            raise TypeError(f"Unsupported output type: {type(output)}")
            
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
            torch.save({
                'model_state_dict': self._model.state_dict(),
                'config': self._config,
            }, path)
            
    def load(self, path: str) -> None:
        """Load model weights
        
        Args:
            path: Model file path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if self._model is None:
            # Need to configure model first
            if 'config' in checkpoint:
                config_ns = self._dict_to_namespace(checkpoint['config'])
                self._model = self._build_model(config_ns)
                self._model = self._model.to(self.device)
            else:
                raise RuntimeError(
                    "Cannot load model without configuration. "
                    "Call configure() first or ensure checkpoint contains config."
                )
                
        self._model.load_state_dict(checkpoint['model_state_dict'])
        
    def capabilities(self) -> Dict[str, bool]:
        """Return model capabilities
        
        Returns:
            Capabilities dictionary
        """
        return {
            "supports_training": False,  # Most adapters are for inference only
            "supports_incremental": False,
            "requires_gpu": torch.cuda.is_available(),
            "framework": "pytorch",
        }
