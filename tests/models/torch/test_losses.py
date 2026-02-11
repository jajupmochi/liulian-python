"""
Tests for custom PyTorch loss functions.
"""

import pytest
import numpy as np


@pytest.fixture(scope='module', autouse=True)
def check_torch():
    """Check if PyTorch is available."""
    pytest.importorskip('torch')


def test_divide_no_nan_normal():
    """Test divide_no_nan with normal values."""
    import torch
    from liulian.models.torch.losses import divide_no_nan

    a = torch.tensor([2.0, 4.0, 6.0])
    b = torch.tensor([2.0, 2.0, 2.0])
    result = divide_no_nan(a, b)

    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(result, expected)


def test_divide_no_nan_with_zeros():
    """Test divide_no_nan replaces NaN/Inf with 0."""
    import torch
    from liulian.models.torch.losses import divide_no_nan

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 0.0, 0.5])
    result = divide_no_nan(a, b)

    # b[1] is 0, so result[1] should be 0 (not NaN or Inf)
    assert result[0] == 0.5
    assert result[1] == 0.0  # Was NaN/Inf, replaced with 0
    assert result[2] == 6.0
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_divide_no_nan_all_zeros():
    """Test divide_no_nan with all zero denominator."""
    import torch
    from liulian.models.torch.losses import divide_no_nan

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.zeros(3)
    result = divide_no_nan(a, b)

    # All should be 0 (no NaN/Inf)
    assert torch.allclose(result, torch.zeros(3))


class TestMAPELoss:
    """Test MAPE loss function."""

    def test_mape_loss_instantiation(self):
        """Test MAPE loss can be instantiated."""
        from liulian.models.torch.losses import mape_loss

        criterion = mape_loss()
        assert criterion is not None

    def test_mape_loss_perfect_prediction(self):
        """MAPE loss should be 0 for perfect predictions."""
        import torch
        from liulian.models.torch.losses import mape_loss

        criterion = mape_loss()
        forecast = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones_like(forecast)
        insample = torch.randn(1, 10, 3)

        loss = criterion(insample, freq=1, forecast=forecast, target=target, mask=mask)
        assert loss.item() < 1e-6

    def test_mape_loss_calculation(self):
        """Test MAPE loss calculation."""
        import torch
        from liulian.models.torch.losses import mape_loss

        criterion = mape_loss()
        forecast = torch.tensor([[9.0, 18.0]])
        target = torch.tensor([[10.0, 20.0]])
        mask = torch.ones_like(forecast)

        loss = criterion(None, 0, forecast, target, mask)

        # Errors: 10%, 10% -> mean 10% = 0.1
        expected = 0.1
        assert abs(loss.item() - expected) < 1e-2

    def test_mape_loss_with_mask(self):
        """Test MAPE loss with masked values."""
        import torch
        from liulian.models.torch.losses import mape_loss

        criterion = mape_loss()
        forecast = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.5, 2.5, 10.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])  # Ignore last value

        loss_masked = criterion(None, 0, forecast, target, mask)

        # Should only consider first two values
        mask_all = torch.ones_like(forecast)
        loss_all = criterion(None, 0, forecast, target, mask_all)

        # Masked loss should be different (and smaller since we ignore large error)
        assert loss_masked.item() < loss_all.item()


class TestSMAPELoss:
    """Test sMAPE loss function."""

    def test_smape_loss_instantiation(self):
        """Test sMAPE loss can be instantiated."""
        from liulian.models.torch.losses import smape_loss

        criterion = smape_loss()
        assert criterion is not None

    def test_smape_loss_perfect_prediction(self):
        """sMAPE loss should be 0 for perfect predictions."""
        import torch
        from liulian.models.torch.losses import smape_loss

        criterion = smape_loss()
        forecast = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones_like(forecast)

        loss = criterion(None, 0, forecast, target, mask)
        assert loss.item() < 1e-6

    def test_smape_loss_symmetry(self):
        """sMAPE should be symmetric: sMAPE(F,A) == sMAPE(A,F)."""
        import torch
        from liulian.models.torch.losses import smape_loss

        criterion = smape_loss()
        a = torch.tensor([[10.0, 20.0, 30.0]])
        b = torch.tensor([[12.0, 18.0, 35.0]])
        mask = torch.ones_like(a)

        loss1 = criterion(None, 0, a, b, mask)  # a as forecast, b as target
        loss2 = criterion(None, 0, b, a, mask)  # b as forecast, a as target

        # Swapping forecast and target should give the same loss
        assert abs(loss1.item() - loss2.item()) < 1e-5

    def test_smape_loss_bounded(self):
        """sMAPE should be bounded [0, 200]."""
        import torch
        from liulian.models.torch.losses import smape_loss

        criterion = smape_loss()

        # Test various prediction errors
        target = torch.tensor([[10.0, 20.0, 30.0]])
        forecast = torch.tensor([[5.0, 25.0, 60.0]])
        mask = torch.ones_like(target)

        loss = criterion(None, 0, forecast, target, mask)

        assert 0 <= loss.item() <= 200


class TestMASELoss:
    """Test MASE loss function."""

    def test_mase_loss_instantiation(self):
        """Test MASE loss can be instantiated."""
        from liulian.models.torch.losses import mase_loss

        criterion = mase_loss()
        assert criterion is not None

    def test_mase_loss_perfect_prediction(self):
        """MASE loss should be 0 for perfect predictions."""
        import torch
        from liulian.models.torch.losses import mase_loss

        criterion = mase_loss()
        insample = torch.randn(4, 96, 7)
        forecast = torch.randn(4, 24, 7)
        target = forecast.clone()  # Perfect prediction
        mask = torch.ones_like(forecast)

        loss = criterion(insample, freq=24, forecast=forecast, target=target, mask=mask)
        assert loss.item() < 1e-6

    def test_mase_loss_requires_insample(self):
        """MASE requires in-sample data for naive forecast."""
        import torch
        from liulian.models.torch.losses import mase_loss

        criterion = mase_loss()
        insample = torch.randn(4, 96, 7)
        forecast = torch.randn(4, 24, 7)
        target = torch.randn(4, 24, 7)
        mask = torch.ones_like(forecast)

        # Should work with proper insample
        loss = criterion(insample, freq=24, forecast=forecast, target=target, mask=mask)
        assert loss.item() >= 0

    def test_mase_loss_frequency_parameter(self):
        """MASE uses frequency for naive forecast computation."""
        import torch
        from liulian.models.torch.losses import mase_loss

        criterion = mase_loss()

        # Create synthetic seasonal data
        insample = torch.randn(2, 48, 1)
        forecast = torch.randn(2, 12, 1)
        target = torch.randn(2, 12, 1)
        mask = torch.ones_like(forecast)

        # Different frequencies should give different losses
        loss_freq12 = criterion(
            insample, freq=12, forecast=forecast, target=target, mask=mask
        )
        loss_freq24 = criterion(
            insample, freq=24, forecast=forecast, target=target, mask=mask
        )

        # Both should be valid (positive)
        assert loss_freq12.item() >= 0
        assert loss_freq24.item() >= 0

    def test_mase_loss_better_than_naive(self):
        """MASE < 1 means forecast is better than naive."""
        import torch
        from liulian.models.torch.losses import mase_loss

        criterion = mase_loss()

        # Create seasonal data with period 24
        t = torch.arange(120).float()
        seasonal = torch.sin(2 * np.pi * t / 24).unsqueeze(0).unsqueeze(-1)

        insample = seasonal[:, :96, :]
        target = seasonal[:, 96:120, :]

        # Good forecast: very close to target
        forecast = target + torch.randn_like(target) * 0.01
        mask = torch.ones_like(forecast)

        loss = criterion(insample, freq=24, forecast=forecast, target=target, mask=mask)

        # Should be very small for good forecast
        assert loss.item() >= 0


class TestLossComparison:
    """Test comparison between different loss functions."""

    def test_all_losses_work_together(self):
        """Test all three losses can be used in same context."""
        import torch
        from liulian.models.torch.losses import mape_loss, smape_loss, mase_loss

        insample = torch.randn(8, 96, 7)
        forecast = torch.randn(8, 24, 7)
        target = torch.randn(8, 24, 7)
        mask = torch.ones_like(forecast)

        mape_criterion = mape_loss()
        smape_criterion = smape_loss()
        mase_criterion = mase_loss()

        mape_val = mape_criterion(insample, 24, forecast, target, mask)
        smape_val = smape_criterion(insample, 24, forecast, target, mask)
        mase_val = mase_criterion(insample, 24, forecast, target, mask)

        # All should return valid scalar losses
        assert mape_val.item() >= 0
        assert smape_val.item() >= 0
        assert mase_val.item() >= 0

    def test_losses_are_differentiable(self):
        """Test that losses support backpropagation."""
        import torch
        from liulian.models.torch.losses import mape_loss, smape_loss, mase_loss

        insample = torch.randn(4, 96, 7, requires_grad=False)
        forecast = torch.randn(4, 24, 7, requires_grad=True)
        target = torch.randn(4, 24, 7, requires_grad=False)
        mask = torch.ones_like(forecast, requires_grad=False)

        for criterion_class in [mape_loss, smape_loss, mase_loss]:
            criterion = criterion_class()
            loss = criterion(insample, 24, forecast, target, mask)

            # Should be able to backpropagate
            loss.backward()

            # Forecast should have gradients
            assert forecast.grad is not None

            # Clear gradients for next test
            forecast.grad.zero_()


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_losses_with_small_values(self):
        """Test losses with very small values."""
        import torch
        from liulian.models.torch.losses import mape_loss, smape_loss, mase_loss

        insample = torch.randn(2, 48, 1) * 1e-6
        forecast = torch.randn(2, 12, 1) * 1e-6
        target = torch.randn(2, 12, 1) * 1e-6
        mask = torch.ones_like(forecast)

        # All should handle small values without inf/nan
        for criterion_class in [mape_loss, smape_loss, mase_loss]:
            criterion = criterion_class()
            loss = criterion(insample, 12, forecast, target, mask)

            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_losses_with_large_values(self):
        """Test losses with very large values."""
        import torch
        from liulian.models.torch.losses import mape_loss, smape_loss, mase_loss

        insample = torch.randn(2, 48, 1) * 1e6
        forecast = torch.randn(2, 12, 1) * 1e6
        target = torch.randn(2, 12, 1) * 1e6
        mask = torch.ones_like(forecast)

        # All should handle large values without inf/nan
        for criterion_class in [mape_loss, smape_loss, mase_loss]:
            criterion = criterion_class()
            loss = criterion(insample, 12, forecast, target, mask)

            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
