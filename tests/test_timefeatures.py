"""
Tests for time features extraction module.
"""

import pytest

# Check for pandas availability
pandas = pytest.importorskip('pandas')
pd = pandas  # alias used throughout tests

import numpy as np
from liulian.utils.timefeatures import (
    TimeFeature,
    SecondOfMinute,
    MinuteOfHour,
    HourOfDay,
    DayOfWeek,
    DayOfMonth,
    DayOfYear,
    MonthOfYear,
    WeekOfYear,
    time_features_from_frequency_str,
    time_features,
)


class TestIndividualFeatures:
    """Test individual time feature extractors."""

    def test_hour_of_day_range(self):
        """HourOfDay should return values in [-0.5, 0.5]."""
        dates = pandas.date_range('2024-01-01', periods=24, freq='1h')
        feat = HourOfDay()
        values = feat(dates)

        assert len(values) == 24
        assert values.min() >= -0.5
        assert values.max() <= 0.5

    def test_hour_of_day_midnight(self):
        """HourOfDay(midnight) should be -0.5."""
        dates = pd.DatetimeIndex(['2024-01-01 00:00:00'])
        feat = HourOfDay()
        assert abs(feat(dates)[0] - (-0.5)) < 1e-6

    def test_hour_of_day_noon(self):
        """HourOfDay(noon) should be ~0."""
        dates = pd.DatetimeIndex(['2024-01-01 12:00:00'])
        feat = HourOfDay()
        expected = 12 / 23.0 - 0.5
        assert abs(feat(dates)[0] - expected) < 1e-6

    def test_day_of_week_range(self):
        """DayOfWeek should return values in [-0.5, 0.5]."""
        dates = pd.date_range('2024-01-01', periods=7, freq='1D')
        feat = DayOfWeek()
        values = feat(dates)

        assert len(values) == 7
        assert values.min() >= -0.5
        assert values.max() <= 0.5

    def test_month_of_year_range(self):
        """MonthOfYear should return values in [-0.5, 0.5]."""
        dates = pd.date_range('2024-01-01', periods=12, freq='1ME')
        feat = MonthOfYear()
        values = feat(dates)

        assert len(values) == 12
        assert values.min() >= -0.5
        assert values.max() <= 0.5

    def test_month_of_year_january(self):
        """MonthOfYear(January) should be -0.5."""
        dates = pd.DatetimeIndex(['2024-01-15'])
        feat = MonthOfYear()
        expected = (1 - 1) / 11.0 - 0.5
        assert abs(feat(dates)[0] - expected) < 1e-6

    def test_month_of_year_december(self):
        """MonthOfYear(December) should be 0.5."""
        dates = pd.DatetimeIndex(['2024-12-15'])
        feat = MonthOfYear()
        expected = (12 - 1) / 11.0 - 0.5
        assert abs(feat(dates)[0] - expected) < 1e-6

    def test_minute_of_hour_range(self):
        """MinuteOfHour should return values in [-0.5, 0.5]."""
        dates = pd.date_range('2024-01-01', periods=60, freq='1min')
        feat = MinuteOfHour()
        values = feat(dates)

        assert len(values) == 60
        assert values.min() >= -0.5
        assert values.max() <= 0.5

    def test_second_of_minute_range(self):
        """SecondOfMinute should return values in [-0.5, 0.5]."""
        dates = pd.date_range('2024-01-01', periods=60, freq='1s')
        feat = SecondOfMinute()
        values = feat(dates)

        assert len(values) == 60
        assert values.min() >= -0.5
        assert values.max() <= 0.5


class TestFeatureRepresentation:
    """Test that features are properly represented."""

    def test_feature_repr(self):
        """Test __repr__ method."""
        feat = HourOfDay()
        assert repr(feat) == 'HourOfDay()'

        feat = DayOfWeek()
        assert repr(feat) == 'DayOfWeek()'


class TestFrequencyMapping:
    """Test time_features_from_frequency_str function."""

    def test_hourly_frequency(self):
        """Hourly frequency should return appropriate features."""
        features = time_features_from_frequency_str('1h')
        assert len(features) == 4

        # Should include HourOfDay, DayOfWeek, DayOfMonth, DayOfYear
        feature_types = [type(f).__name__ for f in features]
        assert 'HourOfDay' in feature_types
        assert 'DayOfWeek' in feature_types
        assert 'DayOfMonth' in feature_types
        assert 'DayOfYear' in feature_types

    def test_daily_frequency(self):
        """Daily frequency should return appropriate features."""
        features = time_features_from_frequency_str('1D')
        assert len(features) == 3

        feature_types = [type(f).__name__ for f in features]
        assert 'DayOfWeek' in feature_types
        assert 'DayOfMonth' in feature_types
        assert 'DayOfYear' in feature_types

    def test_monthly_frequency(self):
        """Monthly frequency should return MonthOfYear."""
        features = time_features_from_frequency_str('1ME')
        assert len(features) == 1
        assert isinstance(features[0], MonthOfYear)

    def test_minutely_frequency(self):
        """Minutely frequency should return 5 features."""
        features = time_features_from_frequency_str('1min')
        assert len(features) == 5

        feature_types = [type(f).__name__ for f in features]
        assert 'MinuteOfHour' in feature_types
        assert 'HourOfDay' in feature_types

    def test_secondly_frequency(self):
        """Secondly frequency should return 6 features (most granular)."""
        features = time_features_from_frequency_str('1s')
        assert len(features) == 6

        feature_types = [type(f).__name__ for f in features]
        assert 'SecondOfMinute' in feature_types
        assert 'MinuteOfHour' in feature_types
        assert 'HourOfDay' in feature_types

    def test_weekly_frequency(self):
        """Weekly frequency should return 2 features."""
        features = time_features_from_frequency_str('1W')
        assert len(features) == 2

        feature_types = [type(f).__name__ for f in features]
        assert 'DayOfMonth' in feature_types
        assert 'WeekOfYear' in feature_types

    def test_yearly_frequency(self):
        """Yearly frequency should return empty list."""
        features = time_features_from_frequency_str('1YE')
        assert len(features) == 0

    def test_frequency_aliases(self):
        """Test common frequency aliases."""
        # 'h' and '1h' should be equivalent
        features_h = time_features_from_frequency_str('h')
        features_1h = time_features_from_frequency_str('1h')
        assert len(features_h) == len(features_1h)

        # 'min' and '1min' should be equivalent
        features_min = time_features_from_frequency_str('min')
        features_1min = time_features_from_frequency_str('1min')
        assert len(features_min) == len(features_1min)

    def test_unsupported_frequency(self):
        """Unsupported frequency should raise (RuntimeError, ValueError)."""
        with pytest.raises((RuntimeError, ValueError)):
            time_features_from_frequency_str('xyz')


class TestTimeFeatures:
    """Test the convenience time_features() function."""

    def test_time_features_shape(self):
        """time_features() should return (n_features, n_timestamps)."""
        dates = pd.date_range('2024-01-01', periods=24, freq='1h')
        features = time_features(dates, freq='h')

        # Hourly frequency has 4 features
        assert features.shape == (4, 24)

    def test_time_features_default_frequency(self):
        """time_features() should use 'h' as default frequency."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        features = time_features(dates)  # No freq specified

        # Should have 4 features for hourly frequency
        assert features.shape[0] == 4

    def test_time_features_values_in_range(self):
        """All time features should be in [-0.5, 0.5]."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        features = time_features(dates, freq='h')

        assert features.min() >= -0.5
        assert features.max() <= 0.5

    def test_time_features_daily(self):
        """Test time_features with daily frequency."""
        dates = pd.date_range('2024-01-01', periods=365, freq='1D')
        features = time_features(dates, freq='d')

        # Daily frequency has 3 features
        assert features.shape == (3, 365)

    def test_time_features_monthly(self):
        """Test time_features with monthly frequency."""
        dates = pd.date_range('2024-01-01', periods=12, freq='1ME')
        features = time_features(dates, freq='ME')

        # Monthly frequency has 1 feature
        assert features.shape == (1, 12)


class TestIntegration:
    """Test integration with typical use cases."""

    def test_typical_forecasting_workflow(self):
        """Simulate typical forecasting workflow."""
        # Create hourly data for one week
        dates = pd.date_range('2024-01-01', periods=168, freq='1h')

        # Extract time features
        features = time_features(dates, freq='h')

        # Features should be ready for model input (after transpose)
        assert features.shape == (4, 168)

        # Transpose for typical model format: (timesteps, features)
        features_transposed = features.T
        assert features_transposed.shape == (168, 4)

    def test_multiple_frequency_consistency(self):
        """Same dates with same frequency should give same features."""
        dates1 = pd.date_range('2024-01-01', periods=24, freq='1h')
        dates2 = pd.date_range('2024-02-01', periods=24, freq='1h')

        features1 = time_features(dates1, freq='h')
        features2 = time_features(dates2, freq='h')

        # Should have same shape
        assert features1.shape == features2.shape

        # Same hours should have same hour-of-day features
        # (First row is HourOfDay)
        assert np.array_equal(features1[0, :], features2[0, :])

    def test_seasonal_pattern_capture(self):
        """Time features should capture seasonal patterns."""
        # Create data spanning multiple weeks
        dates = pd.date_range('2024-01-01', periods=168 * 4, freq='1h')
        features = time_features(dates, freq='h')

        # Hour of day should repeat every 24 hours
        hour_feature = features[0, :]  # First feature is HourOfDay

        # Check that pattern repeats (approximately, accounting for day rollover)
        for i in range(0, len(hour_feature) - 24, 24):
            segment1 = hour_feature[i : i + 24]
            segment2 = hour_feature[i + 24 : i + 48]
            np.testing.assert_array_almost_equal(segment1, segment2, decimal=6)
