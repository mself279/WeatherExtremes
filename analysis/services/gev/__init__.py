"""Generalized Extreme Value (GEV) analysis on annual block maxima.

The GEV distribution arises as the limiting distribution of suitably normalized
maxima of i.i.d. random variables (Fisher–Tippett–Gnedenko theorem). For a
sequence of annual maxima ``M_1, ..., M_n`` the three-parameter family

    F(x; μ, σ, ξ) = exp{-(1 + ξ(x-μ)/σ)^(-1/ξ)}    (ξ ≠ 0)
    F(x; μ, σ, 0) = exp{-exp(-(x-μ)/σ)}             (Gumbel limit)

models the CDF of ``M``. We fit (μ, σ, ξ) by maximum likelihood and quantify
parameter uncertainty by parametric + nonparametric bootstrap.

Modules
-------
* :mod:`distribution` — PDF, CDF, quantile, log-likelihood (numpy only)
* :mod:`fit`          — scipy.optimize MLE
* :mod:`diagnostics`  — bucketed observed-vs-expected, return periods, exceedance
* :mod:`confidence`   — parametric + nonparametric bootstrap
* :mod:`extract`      — annual maxima from a GHCN station's cached daily file
* :mod:`plots`        — Plotly figure builders
* :mod:`pipeline`     — orchestrator (the only thing the view needs to import)
"""
