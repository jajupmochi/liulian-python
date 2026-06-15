"""
Tests for PatchTST model adapter.

PatchTST uses patch-based tokenization for efficient long-term forecasting
with channel independence or shared projection.
"""

import pytest
from tests.models.torch.conftest import (
    check_torch_available,
    validate_forecast_output,
    validate_imputation_output,
    validate_anomaly_output,
    validate_classification_output,
)


@pytest.fixture(scope='module', autouse=True)
def check_dependencies():
    """Check required dependencies are installed."""
    check_torch_available()


class TestPatchTSTForecast:
    """Test PatchTST adapter for forecasting task."""

    @pytest.fixture
    def forecast_config(self):
        """Configuration for forecasting."""
        return {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'fc_dropout': 0.2,
            'head_dropout': 0.0,
            'patch_len': 16,
            'stride': 8,
            'individual': False,
            'revin': True,
            'affine': False,
            'subtract_last': False,
            'decomposition': False,
            'kernel_size': 25,
        }

    def test_adapter_instantiation(self, forecast_config):
        """Test adapter can be instantiated."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        model = PatchTSTAdapter(forecast_config)
        assert model is not None
        assert hasattr(model, 'run')

    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        """Test forecasting task execution."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        model = PatchTSTAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)

        validate_forecast_output(
            outputs,
            batch_size=4,
            pred_len=forecast_config['pred_len'],
            features=forecast_config['c_out'],
        )

    def test_different_patch_configs(self, forecast_config, sample_forecast_inputs):
        """Test different patch length and stride combinations."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        patch_configs = [
            (16, 8),  # Default
            (8, 4),  # Smaller patches
            (24, 12),  # Larger patches
            (12, 12),  # No overlap
        ]

        for patch_len, stride in patch_configs:
            config = forecast_config.copy()
            config['patch_len'] = patch_len
            config['stride'] = stride

            model = PatchTSTAdapter(config)
            outputs = model.run(sample_forecast_inputs)

            validate_forecast_output(outputs, 4, 24, 7)

    def test_individual_vs_shared(self, forecast_config, sample_forecast_inputs):
        """Test individual vs shared channel projection."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        # Individual projection per channel
        config_individual = forecast_config.copy()
        config_individual['individual'] = True
        model_individual = PatchTSTAdapter(config_individual)
        outputs_individual = model_individual.run(sample_forecast_inputs)

        # Shared projection across channels
        config_shared = forecast_config.copy()
        config_shared['individual'] = False
        model_shared = PatchTSTAdapter(config_shared)
        outputs_shared = model_shared.run(sample_forecast_inputs)

        # Both should produce valid outputs
        validate_forecast_output(outputs_individual, 4, 24, 7)
        validate_forecast_output(outputs_shared, 4, 24, 7)

    def test_revin_normalization(self, forecast_config, sample_forecast_inputs):
        """Test RevIN (Reversible Instance Normalization)."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        # With RevIN (recommended)
        config_revin = forecast_config.copy()
        config_revin['revin'] = True
        model_revin = PatchTSTAdapter(config_revin)
        outputs_revin = model_revin.run(sample_forecast_inputs)

        # Without RevIN
        config_no_revin = forecast_config.copy()
        config_no_revin['revin'] = False
        model_no_revin = PatchTSTAdapter(config_no_revin)
        outputs_no_revin = model_no_revin.run(sample_forecast_inputs)

        validate_forecast_output(outputs_revin, 4, 24, 7)
        validate_forecast_output(outputs_no_revin, 4, 24, 7)

    def test_with_decomposition(self, forecast_config, sample_forecast_inputs):
        """Test with series decomposition."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        config = forecast_config.copy()
        config['decomposition'] = True
        config['kernel_size'] = 25

        model = PatchTSTAdapter(config)
        outputs = model.run(sample_forecast_inputs)

        validate_forecast_output(outputs, 4, 24, 7)


class TestPatchTSTOtherTasks:
    """Test PatchTST adapter for other tasks."""

    @pytest.fixture
    def imputation_config(self):
        return {
            'task_name': 'imputation',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'patch_len': 16,
            'stride': 8,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    @pytest.fixture
    def anomaly_config(self):
        return {
            'task_name': 'anomaly_detection',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'patch_len': 16,
            'stride': 8,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    @pytest.fixture
    def classification_config(self):
        return {
            'task_name': 'classification',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'num_class': 10,
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'dropout': 0.2,
            'patch_len': 16,
            'stride': 8,
            'activation': 'gelu',
            'embed': 'timeF',
            'freq': 'h',
        }

    def test_imputation(self, imputation_config, sample_imputation_inputs):
        """Test imputation task."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        model = PatchTSTAdapter(imputation_config)
        outputs = model.run(sample_imputation_inputs)
        validate_imputation_output(outputs, 4, 96, 7)

    def test_anomaly_detection(self, anomaly_config, sample_anomaly_inputs):
        """Test anomaly detection task."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        model = PatchTSTAdapter(anomaly_config)
        outputs = model.run(sample_anomaly_inputs)
        validate_anomaly_output(outputs, 4, 96, 7)

    def test_classification(self, classification_config, sample_classification_inputs):
        """Test classification task."""
        from liulian.models.torch.patchtst import PatchTSTAdapter

        model = PatchTSTAdapter(classification_config)
        outputs = model.run(sample_classification_inputs)
        validate_classification_output(outputs, 4, 10)


class TestPatchTSTTransparentAddAfterPatch:
    """PatchTST transparent identifiers via ``add_after_patch``.

    Transparent modes (onehot/sinusoidal/random/coordinates/numeric_id) can be
    injected in d_model patch-token space (downstream of the per-channel
    instance-norm) via a learned projection of a FIXED identifier table, instead
    of the pre-norm ``concat_to_x`` path. This bypasses the instance-norm that
    would otherwise erase a per-channel time-invariant identifier.
    """

    N = 6  # channels / stations

    def _make_ns(self, mode, integration='add_after_patch', split_mode='multi_channel'):
        from types import SimpleNamespace

        return SimpleNamespace(
            task_name='long_term_forecast',
            seq_len=64,
            pred_len=8,
            enc_in=self.N,
            dec_in=self.N,
            c_out=self.N,
            d_model=16,
            n_heads=2,
            e_layers=1,
            d_ff=32,
            dropout=0.0,
            activation='gelu',
            factor=1,
            split_mode=split_mode,
            identifier_mode=mode,
            id_integration=integration,
            sinusoidal_dim=8,
            random_identifier_dim=8,
            random_identifier_seed=2026,
            coordinates={str(i): (float(i), float(2 * i)) for i in range(self.N)},
            station_ids=[str(i) for i in range(self.N)],
        )

    @pytest.mark.parametrize(
        'mode,expected_dim',
        [
            ('onehot', N),
            ('numeric_id', 1),
            ('sinusoidal', 8),
            ('random', 8),
            ('coordinates', 2),
        ],
    )
    def test_builds_projection_with_correct_dims(self, mode, expected_dim):
        """The fixed table + learned projection are built with matching dims."""
        from liulian.models.torch.patchtst import Model

        model = Model(self._make_ns(mode), patch_len=16, stride=8)
        assert model._use_transparent_after_patch is True
        assert model._use_embedding_after_patch is False
        assert model._use_add_after_patch is True
        assert hasattr(model, 'id_table')
        assert hasattr(model, 'id_proj')
        assert tuple(model.id_table.shape) == (self.N, expected_dim)
        assert model.id_proj.in_features == expected_dim
        assert model.id_proj.out_features == 16
        # The fixed table must NOT be a trainable parameter (it is a buffer);
        # only the projection is learned.
        param_names = {n for n, _ in model.named_parameters()}
        assert not any('id_table' in n for n in param_names)
        assert any('id_proj' in n for n in param_names)

    def test_forward_runs_and_is_finite(self):
        """Forward pass produces a finite, correctly shaped forecast."""
        import torch

        from liulian.models.torch.patchtst import Model

        model = Model(self._make_ns('onehot'), patch_len=16, stride=8).eval()
        B = 4
        x = torch.randn(B, 64, self.N)
        mark = torch.zeros(B, 64, 1)
        dec = torch.zeros(B, 8, self.N)
        with torch.no_grad():
            out = model(x, mark, dec, mark)
        assert out.shape == (B, 8, self.N)
        assert torch.isfinite(out).all()

    def test_identifier_survives_instance_norm(self):
        """The injected identifier changes the output (survives instance-norm).

        Contrast with a per-channel constant added pre-norm, which instance-norm
        erases. Zeroing the projection turns the injection into a no-op; the
        output must differ from the trained (non-zero) projection.
        """
        import torch

        from liulian.models.torch.patchtst import Model

        torch.manual_seed(0)
        model = Model(self._make_ns('onehot'), patch_len=16, stride=8).eval()
        B = 4
        x = torch.randn(B, 64, self.N)
        mark = torch.zeros(B, 64, 1)
        dec = torch.zeros(B, 8, self.N)
        with torch.no_grad():
            out_with_id = model(x, mark, dec, mark)
            model.id_proj.weight.zero_()
            model.id_proj.bias.zero_()
            out_no_id = model(x, mark, dec, mark)
        # Non-trivial difference => the post-patch identifier is NOT erased.
        assert (out_with_id - out_no_id).abs().max().item() > 1e-4

    def test_embedding_path_unchanged(self):
        """The learned ``embedding`` add_after_patch path is untouched."""
        from liulian.models.torch.patchtst import Model

        model = Model(self._make_ns('embedding'), patch_len=16, stride=8)
        assert model._use_embedding_after_patch is True
        assert model._use_transparent_after_patch is False
        assert hasattr(model, 'entity_embedding')
        assert not hasattr(model, 'id_proj')
        assert not hasattr(model, 'id_table')

    def test_requires_multi_channel(self):
        """add_after_patch is rejected outside multi_channel split."""
        from liulian.models.torch.patchtst import Model

        with pytest.raises(ValueError, match='multi_channel'):
            Model(self._make_ns('onehot', split_mode='per_entity'), patch_len=16, stride=8)

    def test_concat_to_x_does_not_build_projection(self):
        """Transparent + concat_to_x keeps the legacy (no internal) path."""
        from liulian.models.torch.patchtst import Model

        model = Model(self._make_ns('onehot', integration='concat_to_x'), patch_len=16, stride=8)
        assert model._use_transparent_after_patch is False
        assert model._use_add_after_patch is False
        assert not hasattr(model, 'id_proj')
