"""Two-fidelity benchmark problems for multi-fidelity Bayesian optimization.

Every function takes x in the unit cube [0, 1]^D, maps it to the problem's box,
and returns a (1, 1) array.  ``PROBLEMS`` lists, per problem, the HF function,
the available LFs with their HF:LF cost ratios, the default dimension and the
known HF minimum (used to report simple regret).

hartmann    6D Hartmann (HF) with the Park, Haftka & Kim (2016) low fidelity
            ("phk").  Hard: within each of the two deep basins the LF tracks HF
            almost perfectly, but with different scale factors (about 3.4 vs
            1.8), so constant-rho co-kriging ranks the basins backwards.
            Two easier LFs whose minimum is at the HF minimizer: "close"
            (first Hartmann weight 1.0 -> 0.9, as in BoTorch's
            AugmentedHartmann) and "forrester" (0.5 f + a linear trend, the 6D
            analogue of the Forrester (2007) LF).  No published cost ratio; 10
            is used by default.
rosenbrock  Rosenbrock, box [-2, 2]^D, min 0 at x = 1 (u = 0.75).  LFs "f2"
            (cost 0.5) and "f3" (cost 0.1).
rastrigin   Shifted-rotated Rastrigin, box [-0.1, 0.2]^D, min 0 at x = 0.1
            (u = 2/3).  LFs "f2" (cost 1/16) and "f3" (cost 1/256).

Rosenbrock and Rastrigin are the NATO AVT-331 L1 benchmarks MF2 and MF3 from
Mainini et al. (2022), "Analytical benchmark problems and methodological
framework for the assessment and comparison of multifidelity optimization
methods" (arXiv:2204.07867), transcribed from the authors' reference code
(gitlab.com/qudo046/avt-331-l1-benchmarks, Python/L1Problems.py), with the
paper's relative costs (Table 2).  Where paper and code differ the code is
followed: Rosenbrock f3 divides by 10 + 0.25 * sum(x) (the paper prints x_1 in
the sum), and Rastrigin rotates only the last two coordinates by 0.2 rad.  The
reference code defines D = 2, 5 and 10 with identical formulas; any D >= 2
works here.
"""
import numpy as np

# ------------------------------------------------------------------ Hartmann 6D

_A = np.array([[10., 3.0, 17., 3.5, 1.7, 8.0],
               [0.05, 10., 17., 0.1, 8.0, 14.],
               [3.0, 3.5, 1.7, 10., 17., 8.0],
               [17., 8.0, 0.05, 10., 0.1, 14.]])
_P = 1.e-4 * np.array([[1312., 1696., 5569., 124., 8283., 5886.],
                       [2329., 4135., 8307., 3736., 1004., 9991.],
                       [2348., 1451., 3522., 2883., 3047., 6650.],
                       [4047., 8828., 8732., 5743., 1091., 381.0]])


def _hartmann(x, alpha, g=np.exp):
    # Summed term by term (not vectorized) so results are bitwise identical to
    # the runs in the README; the GP fits are sensitive to the last bit.
    x = np.asarray(x, dtype=float).reshape(-1)
    assert x.size == 6, f"hartmann is 6D, got {x.size} inputs"
    y = np.zeros(1)
    for i in range(len(alpha)):
        y[0] += -1.0 * alpha[i] * g(-1.0 * np.inner(_A[i, :], (x - _P[i, :]) ** 2.))
    return y.reshape(-1, 1)


def hartmann_hf(x):
    return _hartmann(x, np.array([1.0, 1.2, 3.0, 3.2]))


def _fexp(t):
    return (np.exp(-4. / 9.) + (np.exp(-4. / 9.) * (t + 4.) / 9.)) ** 9


def hartmann_lf_phk(x):
    """Park, Haftka & Kim (2016) LF: different weights, exp replaced by a
    9th-power polynomial approximation."""
    return _hartmann(x, np.array([0.5, 0.5, 2.0, 4.0]), _fexp)


def hartmann_lf_close(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    alpha = np.array([0.9, 1.2, 3.0, 3.2])
    return np.array([[-np.sum(alpha * np.exp(-np.sum(_A * (x - _P) ** 2, axis=1)))]])


def hartmann_lf_forrester(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.array([[0.5 * hartmann_hf(x).item() + 0.5 * (np.mean(x) - 0.5) - 0.5]])


# ------------------------------------------------------------------ AVT-331 L1

def _unit(x, lo, hi):
    x = np.asarray(x, dtype=float).reshape(-1)
    return lo + x * (hi - lo)


def _rosen(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rosenbrock_f1(x):
    return np.array([[_rosen(_unit(x, -2.0, 2.0))]])


def rosenbrock_f2(x):
    x = _unit(x, -2.0, 2.0)
    y = np.sum(50.0 * (x[1:] - x[:-1] ** 2) ** 2 + (-2.0 - x[:-1]) ** 2) - 0.5 * np.sum(x)
    return np.array([[y]])


def rosenbrock_f3(x):
    x = _unit(x, -2.0, 2.0)
    return np.array([[(_rosen(x) - 4.0 - 0.5 * np.sum(x)) / (10.0 + 0.25 * np.sum(x))]])


def _rot(d, theta=0.2):
    """Identity except a rotation by theta of the last two coordinates."""
    r = np.eye(d)
    r[-2:, -2:] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    return r


def _rastrigin(x, phi):
    u = _unit(x, -0.1, 0.2)
    z = _rot(u.size) @ (u - 0.1)
    f = np.sum(z ** 2 + 1.0 - np.cos(10.0 * np.pi * z))
    th = 1.0 - 0.0001 * phi
    return np.array([[f + np.sum(th * np.cos(10.0 * np.pi * th * z + 0.5 * np.pi * th + np.pi) ** 2)]])


def rastrigin_f1(x):
    return _rastrigin(x, 10000.0)


def rastrigin_f2(x):
    return _rastrigin(x, 5000.0)


def rastrigin_f3(x):
    return _rastrigin(x, 2500.0)


# ------------------------------------------------------------------ registry
# lfs: name -> (function, HF:LF cost ratio).  For the AVT-331 problems the ratio
# is the paper's; Hartmann has none published.
PROBLEMS = {
    "hartmann": dict(hf=hartmann_hf, ndim=6, fixed_ndim=True, y_min=-3.32237, default_lf="phk",
                     lfs={"phk": (hartmann_lf_phk, 10.0), "close": (hartmann_lf_close, 10.0),
                          "forrester": (hartmann_lf_forrester, 10.0)}),
    "rosenbrock": dict(hf=rosenbrock_f1, ndim=5, fixed_ndim=False, y_min=0.0, default_lf="f3",
                       lfs={"f2": (rosenbrock_f2, 2.0), "f3": (rosenbrock_f3, 10.0)}),
    "rastrigin": dict(hf=rastrigin_f1, ndim=5, fixed_ndim=False, y_min=0.0, default_lf="f2",
                      lfs={"f2": (rastrigin_f2, 16.0), "f3": (rastrigin_f3, 256.0)}),
}
