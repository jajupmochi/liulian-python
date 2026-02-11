"""
Tests for time series metrics module.
"""

import pytest
import numpy as np
from liulian.utils.metrics import RSE, CORR, MAE, MSE, RMSE, MAPE, MSPE, metric


class TestBasicMetrics:
    """Test basic metrics (MAE, MSE, RMSE)."""

    def test_mae_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        assert MAE(pred, true) == 0.0

    def test_mae_calculation(self):
        """Test MAE calculation."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.1, 2.1, 2.9])
        expected = np.mean([0.1, 0.1, 0.1])
        assert abs(MAE(pred, true) - expected) < 1e-6

    def test_mse_perfect_prediction(self):
        """MSE should be 0 for perfect predictions."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([1.0, 2.0, 3.0])
        assert MSE(pred, true) == 0.0

    def test_mse_calculation(self):
        """Test MSE calculation."""
        pred = np.array([1.0, 2.0, 3.0])
        true = np.array([2.0, 3.0, 4.0])
        expected = 1.0  # Each error is 1, squared is 1, mean is 1
        assert abs(MSE(pred, true) - expected) < 1e-6

    def test_rmse_is_sqrt_of_mse(self):
        """RMSE should equal sqrt(MSE)."""
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([1.5, 2.5, 3.5, 4.5])
        mse = MSE(pred, true)
        rmse = RMSE(pred, true)
        assert abs(rmse - np.sqrt(mse)) < 1e-6

    def test_multidimensional_arrays(self):
        """Test metrics work with multidimensional arrays."""
        pred = np.random.randn(10, 5, 3)
        true = pred + 0.1  # Small error

        mae = MAE(pred, true)
        mse = MSE(pred, true)
        rmse = RMSE(pred, true)

        assert mae > 0
        assert mse > 0
        assert rmse > 0
        assert rmse == np.sqrt(mse)


class TestPercentageMetrics:
    """Test percentage-based metrics (MAPE, MSPE)."""

    def test_mape_calculation(self):
        """Test MAPE calculation."""
        pred = np.array([9.0, 18.0, 27.0])
        true = np.array([10.0, 20.0, 30.0])
        # Errors: 10%, 10%, 10% -> mean 10%
        expected = 0.1
        assert abs(MAPE(pred, true) - expected) < 1e-6

    def test_mspe_calculation(self):
        """Test MSPE calculation."""
        pred = np.array([9.0, 18.0])
        true = np.array([10.0, 20.0])
        # Percentage errors: (1/10)^2, (2/20)^2 = 0.01, 0.01
        expected = 0.01
        assert abs(MSPE(pred, true) - expected) < 1e-6

    def test_mape_with_zeros_gives_inf(self):
        """MAPE with zero true values produces inf/nan."""
        pred = np.array([1.0, 2.0])
        true = np.array([0.0, 2.0])

        with pytest.warns(RuntimeWarning):
            result = MAPE(pred, true)
            assert np.isinf(result) or np.isnan(result)


class TestRelativeMetrics:
    """Test relative metrics (RSE, CORR)."""

    def test_rse_perfect_prediction(self):
        """RSE should be 0 for perfect predictions."""
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true = pred.copy()
        assert RSE(pred, true) == 0.0

    def test_rse_calculation(self):
        """Test RSE scales with prediction error."""
        true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred_good = true + 0.1
        pred_bad = true + 1.0

        rse_good = RSE(pred_good, true)
        rse_bad = RSE(pred_bad, true)

        assert rse_bad > rse_good

    def test_corr_perfect_prediction(self):
        """CORR should be 1 for perfect predictions."""
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        true = pred.copy()
        assert abs(CORR(pred, true) - 1.0) < 1e-6

    def test_corr_perfect_negative_correlation(self):
        """CORR should be -1 for perfectly negative correlation."""
        true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = 6.0 - true  # Perfect negative correlation
        assert abs(CORR(pred, true) - (-1.0)) < 1e-1

    def test_corr_no_correlation(self):
        """CORR should be near 0 for uncorrelated data."""
        np.random.seed(42)
        true = np.random.randn(100)
        pred = np.random.randn(100)
        corr = CORR(pred, true)
        assert abs(corr) < 0.5  # Should be close to 0


class TestMetricFunction:
    """Test the combined metric() function."""

    def test_metric_returns_all_five(self):
        """metric() should return tuple of 5 values."""
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([1.1, 2.1, 3.1, 4.1])

        result = metric(pred, true)
        assert len(result) == 5
        mae, mse, rmse, mape, mspe = result

        # All should be positive for non-zero error
        assert mae > 0
        assert mse > 0
        assert rmse > 0
        assert mape > 0
        assert mspe > 0

    def test_metric_consistency(self):
        """metric() results should match individual function calls."""
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        true = np.array([1.5, 2.5, 3.5, 4.5])

        mae_comb, mse_comb, rmse_comb, mape_comb, mspe_comb = metric(pred, true)

        assert abs(mae_comb - MAE(pred, true)) < 1e-6
        assert abs(mse_comb - MSE(pred, true)) < 1e-6
        assert abs(rmse_comb - RMSE(pred, true)) < 1e-6
        assert abs(mape_comb - MAPE(pred, true)) < 1e-6
        assert abs(mspe_comb - MSPE(pred, true)) < 1e-6

    def test_metric_with_multidimensional(self):
        """metric() should work with multidimensional arrays."""
        pred = np.random.randn(10, 24, 7)
        true = pred + np.random.randn(10, 24, 7) * 0.1

        result = metric(pred, true)
        assert len(result) == 5
        assert all(isinstance(v, float) for v in result)


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_arrays(self):
        """Test with zero arrays."""
        pred = np.zeros(10)
        true = np.zeros(10)

        assert MAE(pred, true) == 0.0
        assert MSE(pred, true) == 0.0
        assert RMSE(pred, true) == 0.0

    def test_large_values(self):
        """Test with large values."""
        pred = np.array([1e6, 2e6, 3e6])
        true = np.array([1.1e6, 2.1e6, 3.1e6])

        mae = MAE(pred, true)
        assert mae > 0
        assert not np.isinf(mae)
        assert not np.isnan(mae)

    def test_small_values(self):
        """Test with small values."""
        pred = np.array([1e-6, 2e-6, 3e-6])
        true = np.array([1.1e-6, 2.1e-6, 3.1e-6])

        mae = MAE(pred, true)
        assert mae > 0
        assert not np.isinf(mae)
        assert not np.isnan(mae)

    def test_negative_values(self):
        """Test with negative values."""
        pred = np.array([-1.0, -2.0, -3.0])
        true = np.array([-1.1, -2.1, -2.9])

        mae = MAE(pred, true)
        mse = MSE(pred, true)
        rmse = RMSE(pred, true)

        assert mae > 0
        assert mse > 0
        assert rmse > 0
