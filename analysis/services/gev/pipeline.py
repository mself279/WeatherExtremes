"""End-to-end GEV pipeline: dataset → fit → diagnostics → CIs → figures."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import confidence, diagnostics, plots
from .distribution import gev_return_level
from .extract import AnnualMaxima, extract_annual_maxima
from .fit import GEVFit, fit_gev


@dataclass(frozen=True)
class GEVParameters:
    element: str
    direction: str = "max"            # "max" or "min"
    start_year: int | None = None
    end_year: int | None = None
    drop_quality_failed: bool = True
    min_obs_per_year: int = 200
    bootstrap_method: str = "both"    # "parametric" / "nonparametric" / "both" / "none"
    bootstrap_iterations: int = 500
    reference_year: int | None = None  # for exceedance probability; defaults to last year
    reference_value: float | None = None  # in display units; defaults to that year's max
    bucket_width: float = 0.5
    rng_seed: int | None = 12345


@dataclass
class GEVResult:
    parameters: dict[str, Any]
    summary: dict[str, Any]
    figures: dict[str, str] = field(default_factory=dict)
    series: dict[str, list] = field(default_factory=dict)
    confidence_intervals: dict[str, Any] = field(default_factory=dict)


def _do_fit(am: AnnualMaxima) -> GEVFit:
    if am.direction == "max":
        return fit_gev(am.display)
    # For 'min', fit GEV to negated data (max of -X), then negate location.
    fit = fit_gev(-am.display)
    return GEVFit(
        location=-fit.location,
        scale=fit.scale,
        shape=fit.shape,
        log_likelihood=fit.log_likelihood,
        converged=fit.converged,
        n_observations=fit.n_observations,
    )


def run_gev(
    *,
    station_id: str,
    cache_dir: Path,
    params: GEVParameters,
) -> GEVResult:
    rng = np.random.default_rng(params.rng_seed)

    # ---- 1. Annual maxima
    am = extract_annual_maxima(
        station_id=station_id,
        element=params.element,
        cache_dir=cache_dir,
        direction=params.direction,
        start_year=params.start_year,
        end_year=params.end_year,
        drop_quality_failed=params.drop_quality_failed,
        min_obs_per_year=params.min_obs_per_year,
    )

    # ---- 2. MLE
    fit = _do_fit(am)

    # ---- 3. Diagnostics
    full_diag = diagnostics.bucket_diagnostic(
        am.display, fit.location, fit.scale, fit.shape,
        bucket_width=params.bucket_width,
        direction=am.direction,
    )
    # Tail diagnostic: from the 75th percentile up
    tail_min = float(np.quantile(am.display, 0.75))
    tail_diag = diagnostics.bucket_diagnostic(
        am.display, fit.location, fit.scale, fit.shape,
        bucket_width=params.bucket_width,
        edge_min=tail_min,
        direction=am.direction,
    )
    rl = diagnostics.return_levels(
        fit.location, fit.scale, fit.shape, direction=am.direction,
    )

    # Reference year/value (defaults: last year of record + that year's max)
    ref_year = params.reference_year if params.reference_year is not None else int(am.years[-1])
    if params.reference_value is not None:
        ref_value = float(params.reference_value)
    else:
        idx = int(np.argmax(am.years == ref_year))
        ref_value = float(am.display[idx])
    ex = diagnostics.exceedance(
        fit.location, fit.scale, fit.shape, ref_value, ref_year,
        direction=am.direction,
    )

    fq = diagnostics.fitted_quantiles(
        fit.location, fit.scale, fit.shape, direction=am.direction,
    )

    # ---- 4. Bootstrap CIs
    parametric_res = None
    nonparametric_res = None
    method = params.bootstrap_method
    if method in ("parametric", "both"):
        parametric_res = confidence.parametric_bootstrap(
            fit, n_iter=params.bootstrap_iterations,
            direction=am.direction, rng=rng,
        )
    if method in ("nonparametric", "both"):
        nonparametric_res = confidence.nonparametric_bootstrap(
            am.display, n_iter=params.bootstrap_iterations,
            direction=am.direction, rng=rng,
        )

    # ---- 5. Return-level CI from parametric bootstrap (point-wise across T)
    rl_ci = None
    if parametric_res is not None and parametric_res.samples.size:
        # For each bootstrap parameter set, compute return levels at the same T.
        # Direction-aware via the same diagnostics helper so min-direction fits
        # produce cold-extreme return levels, not their warm-side equivalent.
        T = rl.return_periods
        n_boot = parametric_res.samples.shape[0]
        boot_levels = np.empty((n_boot, len(T)))
        for k in range(n_boot):
            l, s, sh = parametric_res.samples[k]
            try:
                boot_levels[k] = diagnostics.return_levels(
                    float(l), float(s), float(sh),
                    return_periods=T,
                    direction=am.direction,
                ).levels
            except Exception:  # noqa: BLE001
                boot_levels[k] = np.nan
        with np.errstate(invalid="ignore"):
            lo = np.nanquantile(boot_levels, 0.025, axis=0)
            hi = np.nanquantile(boot_levels, 0.975, axis=0)
        rl_ci = (lo, hi)

    # ---- 6. Figures
    title_prefix = f"{params.element} annual {am.direction} (n={am.years.size})"
    figures: dict[str, str] = {
        "annual_maxima": plots.annual_maxima_with_fit_figure(
            years=am.years,
            maxima=am.display,
            location=fit.location, scale=fit.scale, shape=fit.shape,
            display_unit=am.spec.display_unit,
            title=f"{title_prefix} with fitted percentiles",
        ),
        "histogram_pdf": plots.histogram_pdf_figure(
            maxima=am.display,
            location=fit.location, scale=fit.scale, shape=fit.shape,
            display_unit=am.spec.display_unit,
            title=f"{title_prefix} — empirical vs GEV PDF",
            direction=am.direction,
            nonparam_mean=(
                nonparametric_res.means
                if nonparametric_res is not None and nonparametric_res.n_successful > 0
                else None
            ),
        ),
        "bucket_full": plots.bucket_figure(
            full_diag, am.spec.display_unit,
            title=f"Bucket diagnostic ({params.bucket_width}-{am.spec.display_unit} bins, full range)",
        ),
        "bucket_tail": plots.bucket_figure(
            tail_diag, am.spec.display_unit,
            title=f"Bucket diagnostic — upper tail (≥ p75)",
        ),
        "return_level": plots.return_level_figure(
            rl, am.spec.display_unit,
            parametric_ci=rl_ci,
            title="Return-level plot with 95% parametric CI",
        ),
    }
    if parametric_res or nonparametric_res:
        figures["bootstrap_distributions"] = plots.bootstrap_distributions_figure(
            parametric=parametric_res,
            nonparametric=nonparametric_res,
            point_estimate=(fit.location, fit.scale, fit.shape),
        )

    # ---- 7. Summary + serializable output
    summary = {
        "n_years": int(am.years.size),
        "year_range": [int(am.years[0]), int(am.years[-1])],
        "location": fit.location,
        "scale": fit.scale,
        "shape": fit.shape,
        "log_likelihood": fit.log_likelihood,
        "converged": fit.converged,
        "exceedance_reference_year": ex.reference_year,
        "exceedance_reference_value": ex.reference_value,
        "exceedance_probability": ex.exceedance_probability,
        "expected_return_period_years": ex.expected_return_period,
        "bucket_full_chi_square": full_diag.chi_square,
        "bucket_full_sse": full_diag.sum_squared_error,
        "bucket_tail_chi_square": tail_diag.chi_square,
        "bucket_tail_sse": tail_diag.sum_squared_error,
        "fitted_quantiles": {
            f"p{int(p*100)}": float(v) for p, v in zip(fq.probabilities, fq.values)
        },
        "return_levels": {
            int(t): float(v) for t, v in zip(rl.return_periods, rl.levels)
        },
        "display_unit": am.spec.display_unit,
        "element_label": am.spec.label,
    }

    cis: dict[str, Any] = {}
    if parametric_res is not None:
        cis["parametric"] = {
            "n_iterations": parametric_res.n_iterations,
            "n_successful": parametric_res.n_successful,
            "parameters": parametric_res.parameter_dict(),
        }
    if nonparametric_res is not None:
        cis["nonparametric"] = {
            "n_iterations": nonparametric_res.n_iterations,
            "n_successful": nonparametric_res.n_successful,
            "parameters": nonparametric_res.parameter_dict(),
        }

    series = {
        "years": am.years.tolist(),
        "maxima_display": am.display.tolist(),
        "maxima_raw": am.raw.tolist(),
    }

    return GEVResult(
        parameters=asdict(params) if not isinstance(params, dict) else dict(params),
        summary=summary,
        figures=figures,
        series=series,
        confidence_intervals=cis,
    )
