"""Maximum-likelihood estimation of GEV parameters via scipy.optimize.

Uses Nelder–Mead (derivative-free, robust) on the log-scale parametrization
``θ = (μ, log σ, ξ)`` to keep σ strictly positive without bounds. Method-of-
moments starting values (Gumbel approximation) keep the optimizer in the
feasible region for typical climate data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distribution import gev_log_likelihood


@dataclass(frozen=True)
class GEVFit:
    location: float       # μ, in display units
    scale: float          # σ, > 0
    shape: float          # ξ; > 0 Fréchet, < 0 Weibull, ≈ 0 Gumbel
    log_likelihood: float
    converged: bool
    n_observations: int


def _starting_values(maxima: np.ndarray) -> tuple[float, float, float]:
    """Method-of-moments starting values from the Gumbel (ξ=0) approximation.

    For Gumbel: σ = std·√6/π, μ = mean − γ·σ where γ ≈ 0.5772 (Euler–Mascheroni).
    """
    mean_x = float(np.mean(maxima))
    std_x = float(np.std(maxima, ddof=1)) if len(maxima) > 1 else 1.0
    if std_x == 0:
        std_x = 1.0
    scale0 = std_x * np.sqrt(6.0) / np.pi
    loc0 = mean_x - 0.5772 * scale0
    return loc0, max(scale0, 1e-6), 0.0


def fit_gev(
    maxima: np.ndarray,
    *,
    init: tuple[float, float, float] | None = None,
) -> GEVFit:
    """MLE fit. Raises ValueError if ``maxima`` has fewer than 5 observations.

    Returns a :class:`GEVFit` with the point estimate and convergence flag.
    """
    from scipy.optimize import minimize  # local import keeps unit tests light

    x = np.asarray(maxima, dtype=float)
    if x.size < 5:
        raise ValueError(f"Need >= 5 observations to fit GEV, got {x.size}.")

    loc0, scale0, shape0 = init or _starting_values(x)

    def neg_log_lik(theta: np.ndarray) -> float:
        loc, log_scale, shape = theta
        scale = float(np.exp(log_scale))
        ll = gev_log_likelihood(x, loc, scale, shape)
        if not np.isfinite(ll):
            return 1.0e10
        return -ll

    result = minimize(
        neg_log_lik,
        np.array([loc0, np.log(scale0), shape0]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000, "adaptive": True},
    )

    loc, log_scale, shape = result.x
    scale = float(np.exp(log_scale))
    return GEVFit(
        location=float(loc),
        scale=scale,
        shape=float(shape),
        log_likelihood=-float(result.fun),
        converged=bool(result.success),
        n_observations=int(x.size),
    )
