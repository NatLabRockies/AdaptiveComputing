"""Checks of MFKPosterior, the full cross-level posterior that MF-KG and MF-MES
(adaptive_computing/drivers/hero_mf_information.py) build on SMT's MFK model.

1. Means and variances at both levels match SMT's predictions.
2. The joint covariance over (level, point) pairs is symmetric and PSD.
3. Fantasy updates: retrain SMT with frozen hyperparameters after adding one LF
   observation (a) or one HF observation at an LF-only point (b), and compare
   the change in SMT's HF mean with the Gaussian update from the covariance.
4. The MES helpers return finite, nonnegative values.
"""
import numpy as np
import pytest
from smt.applications import MFK

from adaptive_computing.drivers.hero_mf_information import (MFKPosterior, _mes_correlated, _mes_direct,
                                                            _sample_min_values)

_A = np.array([[10., 3.0, 17., 3.5, 1.7, 8.0], [0.05, 10., 17., 0.1, 8.0, 14.],
               [3.0, 3.5, 1.7, 10., 17., 8.0], [17., 8.0, 0.05, 10., 0.1, 14.]])
_P = 1.e-4 * np.array([[1312., 1696., 5569., 124., 8283., 5886.], [2329., 4135., 8307., 3736., 1004., 9991.],
                       [2348., 1451., 3522., 2883., 3047., 6650.], [4047., 8828., 8732., 5743., 1091., 381.0]])


def f_hf(X):
    return -np.exp(-np.einsum("kd,nkd->nk", _A, (X[:, None, :] - _P) ** 2)) @ np.array([1.0, 1.2, 3.0, 3.2])[:, None]


def f_lf(X):  # Park, Haftka & Kim (2016) Hartmann LF
    t = -np.einsum("kd,nkd->nk", _A, (X[:, None, :] - _P) ** 2)
    g = (np.exp(-4. / 9.) + np.exp(-4. / 9.) * (t + 4.) / 9.) ** 9
    return -g @ np.array([0.5, 0.5, 2.0, 4.0])[:, None]


class FrozenMFK(MFK):
    """MFK with fixed per-level, per-dimension theta and fixed data scaling."""
    frozen_theta = None
    frozen_scaling = None

    def _check_param(self):
        super()._check_param()
        self._theta0 = np.array(self.frozen_theta, dtype=float)

    def _new_train_init(self):
        super()._new_train_init()
        self.X_offset, self.X_scale, self.y_mean, self.y_std = self.frozen_scaling
        self.X_norma_all = [(x - self.X_offset) / self.X_scale for x in self.X]
        self.y_norma_all = [(f - self.y_mean) / self.y_std for f in self.y]


def _fit(X0, y0, X1, y1, frozen=None):
    if frozen is None:
        m = MFK(print_global=False)
    else:
        m = FrozenMFK(print_global=False, hyper_opt="NoOp")
        m.frozen_theta, m.frozen_scaling = frozen
    m.set_training_values(X0, y0, name=0)
    m.set_training_values(X1, y1)
    m.train()
    return m


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    X1 = rng.random((8, 6))
    X0 = np.vstack([X1, rng.random((22, 6))])
    y0, y1 = f_lf(X0), f_hf(X1)
    m = _fit(X0, y0, X1, y1)
    frozen = (np.array(m.optimal_theta), (m.X_offset, m.X_scale, m.y_mean, m.y_std))
    return dict(rng=rng, X0=X0, X1=X1, y0=y0, y1=y1, m=m, post=MFKPosterior(m), T=rng.random((200, 6)),
                frozen=frozen)


def test_marginals_match_smt(setup):
    m, post, T = setup["m"], setup["post"], setup["T"]
    mu0, mu1, v0, v1, _ = post.marginals(T)
    mse, _ = m.predict_variances_all_levels(T)
    assert np.abs(mu1 - m.predict_values(T).ravel()).max() < 1e-8
    assert np.abs(mu0 - m._predict_intermediate_values(T, 1).ravel()).max() < 1e-8
    assert np.abs(v0 - mse[:, 0]).max() / mse[:, 0].max() < 1e-6
    assert np.abs(v1 - mse[:, 1]).max() / mse[:, 1].max() < 1e-6


def test_joint_covariance_psd(setup):
    post, T = setup["post"], setup["T"]
    S = T[:40]
    tS = post.terms(S)
    K00, K01, K10, K11 = post.cross(tS, tS)
    _, _, _, v1, _ = post.marginals(S)
    assert np.abs(np.diag(K11) - v1).max() / v1.max() < 1e-6
    J = np.block([[K00, K01], [K10, K11]])
    assert np.abs(J - J.T).max() / np.abs(J).max() < 1e-8
    ev = np.linalg.eigvalsh(0.5 * (J + J.T))
    assert ev.min() / ev.max() > -1e-8


def test_fantasy_updates_match_frozen_refit(setup):
    s = setup
    post, T, rng = s["post"], s["T"], s["rng"]
    mu0, mu1, _, _, _ = post.marginals(T)
    tT = post.terms(T)

    # (a) one LF observation at a new point
    x = rng.random((1, 6))
    y = f_lf(x)
    mf = _fit(np.vstack([s["X0"], x]), np.vstack([s["y0"], y]), s["X1"], s["y1"], s["frozen"])
    _, _, K10x, _ = post.cross(tT, post.terms(x))
    m0x, _, v0x, _, _ = post.marginals(x)
    pred1 = mu1 + K10x[:, 0] / v0x[0] * (y[0, 0] - m0x[0])
    new1 = mf.predict_values(T).ravel()
    assert np.abs(new1 - pred1).max() / np.abs(new1 - mu1).max() < 1e-4

    # (b) one HF observation at an existing LF-only point (a promotion)
    x = s["X0"][8:9]
    y = f_hf(x)
    mf = _fit(s["X0"], s["y0"], np.vstack([s["X1"], x]), np.vstack([s["y1"], y]), s["frozen"])
    _, _, _, K11x = post.cross(tT, post.terms(x))
    _, m1x, _, v1x, _ = post.marginals(x)
    pred1 = mu1 + K11x[:, 0] / v1x[0] * (y[0, 0] - m1x[0])
    new1 = mf.predict_values(T).ravel()
    assert np.abs(new1 - pred1).max() / np.abs(new1 - mu1).max() < 1e-4


def test_mes_helpers(setup):
    post, T, X1, y1 = setup["post"], setup["T"], setup["X1"], setup["y1"]
    _, mug, _, vg, _ = post.marginals(np.vstack([T, X1]))
    fs = _sample_min_values(mug, np.sqrt(vg), 16)
    assert np.all(np.isfinite(fs)) and fs.max() <= y1.min() + 1e-6
    g = np.array([[-2.0, 0.0, 2.0]])
    direct = _mes_direct(g)
    assert np.all(np.isfinite(direct)) and np.all(direct >= -1e-12)
    # an uncorrelated LF carries no information about f*; a nearly perfect one carries almost all
    assert np.allclose(_mes_correlated(g, np.array([0.0])), 0.0, atol=1e-12)
    assert np.all(_mes_correlated(g, np.array([0.999])) <= direct + 1e-9)
