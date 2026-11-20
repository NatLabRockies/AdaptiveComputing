"""log_expected_improvement returns -log(EI + 1e-30) with the incumbent taken
at the fidelity being evaluated."""
import numpy as np

from adaptive_computing.samplers.acquisition_functions import acq_func_map
from adaptive_computing.samplers.acquisition_functions.expected_improvement import expected_improvement
from adaptive_computing.samplers.acquisition_functions.log_expected_improvement import log_expected_improvement


class _Surrogate:
    i_output = 0

    def predict_values(self, x, fidelity_level=0):
        return np.sin(3 * x[:, :1]) + fidelity_level

    def predict_variances(self, x, fidelity_level=0):
        return 0.01 + 0.1 * x[:, :1] ** 2


class _Dataset:
    """Fidelity 0 has a lower incumbent (0.0) than fidelity 1 (0.5)."""
    y = [np.array([[0.0], [2.0]]), np.array([[0.5], [1.5]])]

    def get_unmasked_data(self, i_fidelity=None, i_output=None):
        return None, self.y[i_fidelity]


class _OnlyFidelity:
    """Presents one fidelity's data as fidelity 0 (expected_improvement reads fidelity 0)."""

    def __init__(self, y):
        self.y = y

    def get_unmasked_data(self, i_fidelity=None, i_output=None):
        return None, self.y


def test_registered():
    assert acq_func_map["log_expected_improvement"] is log_expected_improvement


def test_equals_minus_log_ei_with_own_fidelity_incumbent():
    x = np.linspace(0, 1, 51)[:, None]
    s, d = _Surrogate(), _Dataset()
    for fid in (0, 1):
        val = log_expected_improvement(x, s, d, fid)
        # expected_improvement with this fidelity's data, evaluated at this fidelity
        ei = -expected_improvement(x, s, _OnlyFidelity(d.y[fid]), fid)
        assert np.allclose(val, -np.log(np.maximum(ei, 0.0) + 1e-30))
        assert np.argmin(val) == np.argmax(ei)
