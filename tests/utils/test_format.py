"""The shared T/G/M/K parameter-count formatter (liulian.utils.format).

Single formatter used at EVERY model-parameter statistics site (pipeline
build_model log, print_experiment_info, timellm ln_only/lora prints) so
magnitudes read consistently. Thresholds: >=1e12 T, >=1e9 G, >=1e6 M (2
decimals), >=1e3 K (1 decimal), else plain int; sign preserved.
"""

import pytest

from liulian.utils.format import format_param_count


@pytest.mark.parametrize(
    'n, expected',
    [
        (6_700_000_000_000, '6.70T'),
        (7_240_000_000, '7.24G'),
        (124_439_808, '124.44M'),   # GPT-2 small
        (19_968, '20.0K'),          # ln_only unfrozen count
        (999, '999'),
        (0, '0'),
        (-1_500_000, '-1.50M'),
        (1_000, '1.0K'),
        (1_000_000, '1.00M'),
    ],
)
def test_format_param_count(n, expected):
    assert format_param_count(n) == expected


def test_pipeline_wrapper_delegates():
    from liulian.pipeline import _format_param_count

    assert _format_param_count(124_439_808) == '124.44M'
