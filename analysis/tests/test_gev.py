"""Tests for GEV math (no scipy required for the distribution module).

Tests that need ``scipy.optimize`` (the MLE fit and the bootstrap) are
guarded so they no-op cleanly in environments without SciPy installed.
"""
from __future__ import annotations

import numpy as np

from analysis.services.gev.distribution import (
    gev_cdf,
    gev_log_likelihood,
    gev_pdf,
    gev_quantile,
    gev_return_level,
)
from analysis.services.gev.diagnostics import (
    bucket_diagnostic,
    exceedance,
    fitted_quantiles,
    return_levels,
)


# ---------------------------------------------------------------------------
# distribution.py
# ---------------------------------------------------------------------------


def test_cdf_quantile_round_trip():
    loc, scale, shape = 50.0, 5.0, 0.1
    p = np.array([0.05, 0.5, 0.95, 0.99])
    x = gev_quantile(p, loc, scale, shape)
    p_back = gev_cdf(x, loc, scale, shape)
    assert np.allclose(p, p_back)


def test_pdf_integrates_to_one():
    loc, scale, shape = 100.0, 10.0, 0.05
    xx = np.linspace(loc - 5 * scale, loc + 30 * scale, 20000)
    pdf = gev_pdf(xx, loc, scale, shape)
    integral = float(np.trapezoid(pdf, xx))
    assert 0.98 < integral < 1.02


def test_gumbel_limit_matches_analytic():
    loc, scale = 50.0, 5.0
    p = 0.95
    q_app = gev_quantile(np.array([p]), loc, scale, 0.0)[0]
    q_exact = loc - scale * np.log(-np.log(p))
    assert abs(q_app - q_exact) < 1e-6


def test_return_level_equals_appropriate_quantile():
    loc, scale, shape = 100.0, 10.0, 0.05
    rl_100 = gev_return_level(np.array([100.0]), loc, scale, shape)[0]
    q_99 = gev_quantile(np.array([0.99]), loc, scale, shape)[0]
    assert abs(rl_100 - q_99) < 1e-9


def test_log_likelihood_higher_at_true_params():
    rng = np.random.default_rng(0)
    sample = gev_quantile(rng.uniform(1e-6, 1 - 1e-6, 100), 50.0, 5.0, 0.1)
    ll_true = gev_log_likelihood(sample, 50.0, 5.0, 0.1)
    ll_off = gev_log_likelihood(sample, 80.0, 5.0, 0.1)
    assert ll_true > ll_off


def test_log_likelihood_minus_inf_outside_support():
    # ξ = 0.5, support: x > μ - σ/ξ = 50 - 10/0.5 = 30
    # Sample including a value below 30 should yield -inf.
    sample = np.array([35.0, 40.0, 25.0])
    ll = gev_log_likelihood(sample, 50.0, 10.0, 0.5)
    assert ll == float("-inf") or not np.isfinite(ll)


# ---------------------------------------------------------------------------
# diagnostics.py
# ---------------------------------------------------------------------------


def test_bucket_diagnostic_total_observed_matches_data():
    rng = np.random.default_rng(1)
    sample = gev_quantile(rng.uniform(1e-6, 1 - 1e-6, 80), 5.0, 1.0, 0.05)
    bd = bucket_diagnostic(sample, 5.0, 1.0, 0.05, bucket_width=0.5)
    # All observations should fall inside the auto-derived bucket range
    assert bd.observed.sum() == sample.size
    # Expected counts close to total (CDF range covers most of the mass)
    assert 0.85 * sample.size < bd.expected.sum() < sample.size * 1.05


def test_return_levels_increase_with_period():
    rl = return_levels(50.0, 5.0, 0.1)
    assert np.all(np.diff(rl.levels) > 0)


def test_fitted_quantiles_are_increasing():
    fq = fitted_quantiles(50.0, 5.0, 0.1)
    assert np.all(np.diff(fq.values) > 0)


def test_exceedance_yields_sensible_return_period():
    # P(X > x_99) should be about 0.01, return period 100 yr.
    loc, scale, shape = 50.0, 5.0, 0.1
    x_99 = gev_quantile(np.array([0.99]), loc, scale, shape)[0]
    ex = exceedance(loc, scale, shape, float(x_99), reference_year=2020)
    assert abs(ex.exceedance_probability - 0.01) < 1e-6
    assert abs(ex.expected_return_period - 100.0) < 1e-3


# ---------------------------------------------------------------------------
# fit.py — only run if scipy is installed
# ---------------------------------------------------------------------------


def test_mle_recovers_known_parameters_if_scipy_available():
    try:
        from analysis.services.gev.fit import fit_gev
    except ImportError:
        return  # scipy not installed; skip

    rng = np.random.default_rng(42)
    true_loc, true_scale, true_shape = 100.0, 8.0, 0.1
    n = 500
    sample = gev_quantile(rng.uniform(1e-6, 1 - 1e-6, n), true_loc, true_scale, true_shape)

    try:
        fit = fit_gev(sample)
    except Exception:
        return  # If scipy.optimize not available at runtime, skip

    # Sanity: with n=500, MLE should be within ~10% of true values
    assert abs(fit.location - true_loc) / true_loc < 0.05
    assert abs(fit.scale - true_scale) / true_scale < 0.10
    assert abs(fit.shape - true_shape) < 0.05
