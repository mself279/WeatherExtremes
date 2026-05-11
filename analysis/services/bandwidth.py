"""Bandwidth selection by minimizing a cross-validation MISE estimator.

For each candidate bandwidth ``h`` we compute the score

    M(h) = integral lambda_hat(t)^2 dt  -  2 * sum_i lambda_hat_{-i}(T_i)

where the second term uses the "delete-one" estimator. Local minima of
``M(h)`` are candidate optimal bandwidths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate

from .kde import occurrence_rate, reflect_events


@dataclass(frozen=True)
class BandwidthScore:
    bandwidth_years: float
    integrated_rate_squared: float
    delete_one_sum: float
    score: float  # MISE-like criterion (lower is better)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Numpy-2.0-safe trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))  # noqa: NPY201 - fallback for numpy<2


def _simpson(y: np.ndarray, x: np.ndarray) -> float:
    """Simpson's rule via SciPy with backwards-compat naming."""
    if hasattr(integrate, "simpson"):
        return float(integrate.simpson(y, x=x))
    return float(integrate.simps(y, x))  # type: ignore[attr-defined]


def score_bandwidth(
    grid: np.ndarray,
    event_dates: np.ndarray,
    bandwidth_years: float,
    bias_window_years: float | None = None,
    observation_factor: float = 1.0,  # kept for API compat; unused now
) -> BandwidthScore:
    """Compute the cross-validation score for a single bandwidth.

    ``bias_window_years`` controls the event-reflection width near each
    boundary; pass ``None`` to disable bias correction. The integral is
    always taken over the original observation grid.
    """
    if bias_window_years and bias_window_years > 0:
        events_for_kde = reflect_events(
            event_dates, grid[0], grid[-1], bias_window_years
        )
    else:
        events_for_kde = event_dates

    rate = occurrence_rate(grid, events_for_kde, bandwidth_years, delete_one=False)
    integral = _simpson(rate ** 2, grid)

    # LOO rate at each ORIGINAL event, using all events (incl. reflected) as
    # neighbors except the event itself.
    rate_at_events = occurrence_rate(
        event_dates, events_for_kde, bandwidth_years, delete_one=True
    )
    delete_one_sum = float(rate_at_events.sum())

    return BandwidthScore(
        bandwidth_years=float(bandwidth_years),
        integrated_rate_squared=integral,
        delete_one_sum=delete_one_sum,
        score=integral - 2.0 * delete_one_sum,
    )


def select_bandwidth(
    grid: np.ndarray,
    event_dates: np.ndarray,
    bandwidths_years: np.ndarray,
    bias_window_factor: float = 3.0,
    observation_factor: float = 1.0,
) -> tuple[list[BandwidthScore], BandwidthScore]:
    """Score every candidate bandwidth and return the best (lowest score).

    ``bias_window_factor`` scales the bias-correction reflection width as a
    multiple of each candidate bandwidth (set to 0 to disable correction).
    Returns the full list of scores plus the chosen one.
    """
    if len(bandwidths_years) == 0:
        raise ValueError("bandwidths_years must contain at least one candidate")

    scores: list[BandwidthScore] = []
    for h in bandwidths_years:
        bw = (
            bias_window_factor * h
            if bias_window_factor and bias_window_factor > 0
            else None
        )
        scores.append(
            score_bandwidth(
                grid=grid,
                event_dates=event_dates,
                bandwidth_years=float(h),
                bias_window_years=bw,
                observation_factor=observation_factor,
            )
        )

    best = min(scores, key=lambda s: s.score)
    return scores, best
