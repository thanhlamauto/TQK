"""Low-complexity shared reliability surfaces (README section 4).

Two surfaces are fit over the 57 policies.  Both use the same three features:

    q     = t / 64          (denoising depth of the pruning checkpoint)
    logM  = log(M)          (initial pool size)
    ratio = K / M           (survivor fraction)

Miss probability ``p_miss``
---------------------------
Logistic model with a *monotone non-increasing* effect of denoising depth::

    logit(p_miss) = b0 + b1*logM + b2*ratio + sum_j w_j * max(0, knot_j - q)

with ``w_j >= 0``.  Each hinge ``max(0, knot_j - q)`` is non-increasing in ``q``,
so the fitted miss probability cannot increase as denoising depth grows.  This is
the pre-declared constraint from the README ("miss probability constrained not to
increase with denoising depth").

Expected conditional regret ``ell``
-----------------------------------
Positive Gamma regression with a log link::

    ell = exp(c0 + c1*logM + c2*ratio + c3*q + c4*q^2)

The log link guarantees ``ell > 0``.

Both models are fit by minimising the (weighted) binomial / Gamma deviance with a
small L2 penalty on the non-intercept coefficients.  Weights are the number of
underlying subset observations: ``n`` for ``p_miss`` and the number of observed
misses ``n_miss`` for ``ell``.  All constants (knots, penalty, optimiser bounds)
are frozen before the calibration rewards are inspected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

#: Fixed hinge knots on the q = t/64 grid (t in 10..28 -> q in 0.156..0.4375).
Q_KNOTS: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40)
L2_MISS = 1e-3
L2_ELL = 1e-3
ELL_FLOOR = 1e-4
PROB_EPS = 1e-9
COEF_BOUND = 30.0


def _hinge(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.maximum(0.0, np.asarray(Q_KNOTS)[None, :] - q[:, None])


@dataclass
class MissModel:
    theta: np.ndarray

    def predict(self, q, log_m, ratio):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        b0, b1, b2 = self.theta[:3]
        w = self.theta[3:]
        eta = b0 + b1 * log_m + b2 * ratio + _hinge(q) @ w
        return 1.0 / (1.0 + np.exp(-eta))


@dataclass
class EllModel:
    theta: np.ndarray

    def predict(self, q, log_m, ratio):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        c0, c1, c2, c3, c4 = self.theta
        eta = c0 + c1 * log_m + c2 * ratio + c3 * q + c4 * q * q
        return np.exp(eta)


def fit_miss(q, log_m, ratio, target, weight) -> MissModel:
    q = np.asarray(q, dtype=float)
    log_m = np.asarray(log_m, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    target = np.clip(np.asarray(target, dtype=float), 0.0, 1.0)
    weight = np.asarray(weight, dtype=float)
    hinges = _hinge(q)
    n_knots = hinges.shape[1]

    def negative_log_likelihood(theta: np.ndarray) -> float:
        eta = theta[0] + theta[1] * log_m + theta[2] * ratio + hinges @ theta[3:]
        # stable weight * [y*eta - log(1+exp(eta))]
        log_sigmoid = -np.logaddexp(0.0, -eta)
        log_one_minus = -np.logaddexp(0.0, eta)
        ll = target * log_sigmoid + (1.0 - target) * log_one_minus
        return float(-np.sum(weight * ll)) + L2_MISS * float(np.sum(theta[3:] ** 2))

    x0 = np.zeros(3 + n_knots)
    bounds = [(-COEF_BOUND, COEF_BOUND)] * 3 + [(0.0, COEF_BOUND)] * n_knots
    result = minimize(
        negative_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
    )
    return MissModel(result.x)


def fit_ell(q, log_m, ratio, target, weight) -> EllModel:
    q = np.asarray(q, dtype=float)
    log_m = np.asarray(log_m, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    target = np.maximum(np.asarray(target, dtype=float), ELL_FLOOR)
    weight = np.asarray(weight, dtype=float)

    def gamma_deviance(theta: np.ndarray) -> float:
        eta = (
            theta[0]
            + theta[1] * log_m
            + theta[2] * ratio
            + theta[3] * q
            + theta[4] * q * q
        )
        mu = np.exp(eta)
        per_obs = (target - mu) / mu - np.log(target / mu)
        return float(2.0 * np.sum(weight * per_obs)) + L2_ELL * float(np.sum(theta[1:] ** 2))

    x0 = np.array([np.log(max(float(np.average(target, weights=weight)), ELL_FLOOR)), 0.0, 0.0, 0.0, 0.0])
    bounds = [(-COEF_BOUND, COEF_BOUND)] * 5
    result = minimize(
        gamma_deviance,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
    )
    return EllModel(result.x)


def predict_miss_ell(model_miss: MissModel, model_ell: EllModel, q, log_m, ratio) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(model_miss.predict(q, log_m, ratio), PROB_EPS, 1.0 - PROB_EPS)
    ell = np.maximum(model_ell.predict(q, log_m, ratio), 0.0)
    return p, ell
