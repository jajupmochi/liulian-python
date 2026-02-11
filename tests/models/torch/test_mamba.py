"""
Tests for Mamba model adapter.

Mamba is a selective state-space model. Requires the external ``mamba-ssm``
package which is typically CUDA-only, so tests are skipped when unavailable.
"""

import pytest
from tests.models.torch.conftest import (
    check_torch_available,
    sample_forecast_inputs,
    validate_forecast_output,
)


def _mamba_available() -> bool:
    try:
        import mamba_ssm  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope='module', autouse=True)
def check_dependencies():
    check_torch_available()


@pytest.mark.skipif(not _mamba_available(), reason='mamba-ssm not installed')
class TestMambaForecast:
    @pytest.fixture
    def forecast_config(self):
        return {
            'task_name': 'long_term_forecast',
            'seq_len': 96,
            'pred_len': 24,
            'label_len': 48,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 64,
            'd_ff': 16,
            'e_layers': 2,
            'd_conv': 4,
            'expand': 2,
            'dropout': 0.1,
            'embed': 'timeF',
            'freq': 'h',
        }

    def test_adapter_instantiation(self, forecast_config):
        from liulian.models.torch.mamba_model import MambaAdapter

        model = MambaAdapter(forecast_config)
        assert model is not None

    def test_forecast_run(self, forecast_config, sample_forecast_inputs):
        from liulian.models.torch.mamba_model import MambaAdapter

        model = MambaAdapter(forecast_config)
        outputs = model.run(sample_forecast_inputs)
        validate_forecast_output(outputs, 4, 24, 7)
