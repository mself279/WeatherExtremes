"""Numerical correctness checks for the KDE service."""
from __future__ import annotations

import numpy as np

from analysis.services.kde import (
    build_time_grid,
    occurrence_rate,
    reflect_events,
    reflect_pseudo_data,
)


def test_build_time_grid_daily():
    grid = build_time_grid(2000.0, 2001.0, 1.0 / 365.0)
    assert grid[0] == 2000.0
    assert grid.size >= 364
    # Spacing rounded to 4 decimals, so 0.0027 vs 0.002739..., tolerance ~5e-5.
    assert abs((grid[1] - grid[0]) - 1.0 / 365.0) < 1e-4


def test_occurrence_rate_integrates_to_event_count():
    """For a wide kernel, the integral of lambda(t) should approximate N."""
    grid = build_time_grid(1990.0, 2010.0, 1.0 / 365.0)
    events = np.array([1995.0, 2000.0, 2005.0])
    rate = occurrence_rate(grid, events, bandwidth_years=2.0)
    integral = np.trapezoid(rate, grid) if hasattr(np, "trapezoid") else np.trapz(rate, grid)
    # Should be close to the number of events, minus what falls outside the window.
    assert 2.5 < integral < 3.0


def test_delete_one_zeros_diagonal():
    grid = np.array([1990.0, 1995.0, 2000.0])
    events = np.array([1990.0, 1995.0, 2000.0])
    full = occurrence_rate(grid, events, bandwidth_years=1.0, delete_one=False)
    leave_one = occurrence_rate(grid, events, bandwidth_years=1.0, delete_one=True)
    # Leave-one-out drops the t==T_i term so the rate is strictly smaller.
    assert np.all(leave_one < full)


def test_reflect_pseudo_data_extends_grid():
    grid = build_time_grid(2000.0, 2002.0, 0.5)
    extended = reflect_pseudo_data(grid, bias_window=2)
    assert extended[0] < grid[0]
    assert extended[-1] > grid[-1]
    # Original points should still be present (in the same order).
    assert all(g in extended for g in grid)


def test_reflect_events_mirrors_at_boundaries():
    events = np.array([1990.5, 1995.0, 2000.0, 2023.5])  # near both edges
    reflected = reflect_events(events, 1990.0, 2024.0, bias_window_years=5.0)
    # The 1990.5 event (within 5y of left boundary) reflects to 1989.5
    # The 2023.5 event (within 5y of right boundary) reflects to 2024.5
    left_refls = reflected[reflected < 1990.0]
    right_refls = reflected[reflected > 2024.0]
    assert np.isclose(left_refls.min(), 1989.5)
    assert np.isclose(right_refls.max(), 2024.5)
    # Original events still present
    for e in events:
        assert any(np.isclose(reflected, e))


def test_reflect_events_corrects_boundary_bias():
    """A constant-rate process should produce a flat estimate after correction."""
    rng = np.random.default_rng(42)
    events = np.sort(rng.uniform(2000, 2020, size=400))
    grid = build_time_grid(2000.0, 2020.0, 1.0 / 365)
    h = 5.0
    rate_uncorr = occurrence_rate(grid, events, h)
    reflected = reflect_events(events, 2000.0, 2020.0, 3 * h)
    rate_corr = occurrence_rate(grid, reflected, h)
    # Uncorrected: edges should be far below center.
    center = rate_uncorr[len(grid) // 2]
    assert rate_uncorr[100] < 0.7 * center
    # Corrected: edges should be close to center.
    center_corr = rate_corr[len(grid) // 2]
    assert 0.85 < rate_corr[100] / center_corr < 1.15
    assert 0.85 < rate_corr[-100] / center_corr < 1.15
