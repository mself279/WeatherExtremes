"""End-to-end orchestration of the analysis pipeline.

Inputs: a parsed dataset + an :class:`AnalysisParameters` configuration.
Output: an :class:`AnalysisResult` containing summary statistics and Plotly
figure JSON. The result object is JSON-serializable (no numpy arrays leak
out) so it can be stored in ``models.JSONField`` and re-rendered later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import autocorr, bandwidth, bootstrap, kde, plots
from .io import event_dates


@dataclass(frozen=True)
class AnalysisParameters:
    indicator_column: str = "Over3"
    observation_start: float = 1938.0
    observation_end: float = 2024.0
    observation_factor: float = 1.0 / 365.0
    bandwidth_years: float = 20.0
    bias_correction: bool = True
    bootstrap_iterations: int = 100
    # Search defaults tuned for tractable runtime on dense-event datasets
    # (e.g. ~1500 TMAX>95°F events at STL): 10 candidates instead of 30, and
    # min=3 because h=1 is rarely meaningful for multi-decade analyses.
    bandwidth_search_min: float = 3.0
    bandwidth_search_max: float = 30.0
    bandwidth_search_step: float = 3.0
    additional_indicators: tuple[str, ...] = ()
    decluster_tau: int = 0
    rng_seed: int | None = 12345

    def candidate_bandwidths(self) -> np.ndarray:
        return np.arange(
            self.bandwidth_search_min,
            self.bandwidth_search_max + self.bandwidth_search_step / 2,
            self.bandwidth_search_step,
        )


@dataclass
class AnalysisResult:
    """Everything the UI needs to render a finished run."""

    parameters: dict[str, Any]
    summary: dict[str, Any]
    figures: dict[str, str] = field(default_factory=dict)
    series: dict[str, list[float]] = field(default_factory=dict)


def _build_indicator(
    grid: np.ndarray, dates: np.ndarray, observation_factor: float
) -> np.ndarray:
    """Mark grid points whose (rounded) value is in ``dates``."""
    rounded_grid = np.round(grid, 4)
    rounded_dates = np.round(dates, 4)
    return np.isin(rounded_grid, rounded_dates).astype(int)


def run_analysis(
    df,
    params: AnalysisParameters,
) -> AnalysisResult:
    """Execute the full pipeline and return a JSON-serializable result.

    Parameters
    ----------
    df:
        DataFrame from :func:`analysis.services.io.parse_input`.
    params:
        :class:`AnalysisParameters` instance.
    """
    rng = np.random.default_rng(params.rng_seed)

    grid = kde.build_time_grid(
        params.observation_start,
        params.observation_end,
        params.observation_factor,
    )

    dates = event_dates(df, params.indicator_column)
    indicator = _build_indicator(grid, dates, params.observation_factor)

    if params.decluster_tau and params.decluster_tau > 0:
        indicator = autocorr.decluster(indicator, params.decluster_tau)
        dates = grid[indicator == 1]

    # Helper: reflect events at the boundaries when bias correction is on.
    def _events_for(bw_years: float, evts: np.ndarray) -> np.ndarray:
        if not params.bias_correction:
            return evts
        return kde.reflect_events(
            evts,
            params.observation_start,
            params.observation_end,
            bias_window_years=3.0 * bw_years,
        )

    # ---- 1. Smoothed occurrence rate at the chosen bandwidth + comparisons.
    bw_compare = sorted({params.bandwidth_years, 10.0, 15.0, 20.0, 25.0})
    rates_by_bandwidth: dict[float, np.ndarray] = {}
    for bw in bw_compare:
        rates_by_bandwidth[float(bw)] = kde.occurrence_rate(
            grid=grid,
            event_dates=_events_for(float(bw), dates),
            bandwidth_years=float(bw),
            delete_one=False,
        )

    point_estimate = rates_by_bandwidth[float(params.bandwidth_years)]

    # ---- 2. Multi-threshold comparison (e.g. Over3 vs Over4).
    rates_by_threshold: dict[str, np.ndarray] = {
        params.indicator_column: point_estimate
    }
    for col in params.additional_indicators:
        if col == params.indicator_column or col not in df.columns:
            continue
        col_dates = event_dates(df, col)
        rates_by_threshold[col] = kde.occurrence_rate(
            grid=grid,
            event_dates=_events_for(float(params.bandwidth_years), col_dates),
            bandwidth_years=float(params.bandwidth_years),
            delete_one=False,
        )

    # ---- 3. Bandwidth selection.
    candidate_h = params.candidate_bandwidths()
    scores, chosen = bandwidth.select_bandwidth(
        grid=grid,
        event_dates=dates,
        bandwidths_years=candidate_h,
        bias_window_factor=3.0 if params.bias_correction else 0.0,
        observation_factor=params.observation_factor,
    )

    # ---- 4. Bootstrap confidence band at the chosen-by-user bandwidth.
    # Bootstrap resamples the original event list (preserves the empirical
    # distribution); we reflect each resampled set before the KDE.
    bootstrap_curves = bootstrap.bootstrap_rates(
        grid=grid,
        event_dates=dates,
        bandwidth_years=float(params.bandwidth_years),
        iterations=int(params.bootstrap_iterations),
        rng=rng,
        reflect_for_bias=params.bias_correction,
        observation_start=params.observation_start,
        observation_end=params.observation_end,
    )
    band = bootstrap.confidence_band(grid, point_estimate, bootstrap_curves)

    # ---- 5. Build figures.
    figures = {
        "occurrence_rate": plots.occurrence_rate_figure(
            grid=grid,
            rates_by_bandwidth=rates_by_bandwidth,
            event_dates=dates,
            title=(
                f"Occurrence rate of '{params.indicator_column}' events "
                f"({params.observation_start:g}–{params.observation_end:g})"
            ),
        ),
        "confidence_band": plots.confidence_band_figure(
            band=band,
            bandwidth_years=float(params.bandwidth_years),
        ),
        "bandwidth_score": plots.bandwidth_score_figure(
            scores=scores,
            chosen=chosen,
        ),
    }
    if len(rates_by_threshold) > 1:
        figures["multi_threshold"] = plots.multi_threshold_figure(
            grid=grid,
            rates_by_threshold=rates_by_threshold,
            title="Occurrence rate by threshold",
        )

    # ---- 6. Summary numbers + serializable series for cross-run comparison.
    annual_mean = float(point_estimate.mean()) if point_estimate.size else 0.0
    peak_index = int(np.argmax(point_estimate)) if point_estimate.size else 0
    peak_year = float(grid[peak_index]) if grid.size else 0.0

    summary = {
        "event_count": int(dates.size),
        "annual_mean_rate": annual_mean,
        "peak_year": peak_year,
        "peak_rate": float(point_estimate[peak_index]) if point_estimate.size else 0.0,
        "chosen_bandwidth_years": float(chosen.bandwidth_years),
        "chosen_bandwidth_score": float(chosen.score),
    }

    series = {
        "grid": grid.tolist(),
        "point_estimate": point_estimate.tolist(),
        "p10": band.p10.tolist(),
        "p90": band.p90.tolist(),
        "event_dates": dates.tolist(),
    }

    return AnalysisResult(
        parameters=asdict(params) if not isinstance(params, dict) else dict(params),
        summary=summary,
        figures=figures,
        series=series,
    )
