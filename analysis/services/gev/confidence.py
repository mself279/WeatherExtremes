"""Bootstrap confidence intervals for GEV parameters.

Two methods, both following the patterns described in the user's workbook:

* **Parametric** (Coles 2001 §2.6.5). Generate ``n_iter`` samples of size
  ``n`` from the fitted GEV(μ̂, σ̂, ξ̂); refit each. Reflects only sampling
  uncertainty assuming the GEV model is correct.
* **Nonparametric** (ISL §5.2). Resample with replacement from the observed
  annual maxima; refit each. Robust to model misspecification.

Both yield a ``BootstrapResult`` with the per-parameter mean / SD / 95% CI
plus the raw refit array for downstream visualization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distribution import gev_quantile
from .fit import GEVFit, fit_gev


@dataclass(frozen=True)
class BootstrapResult:
    method: str                                  # "parametric" or "nonparametric"
    n_iterations: int
    n_successful: int                            # refits that converged
    samples: np.ndarray                          # (n_successful, 3) of (loc, scale, shape)
    means: tuple[float, float, float]
    sds: tuple[float, float, float]
    ci_lower: tuple[float, float, float]         # 2.5th percentile
    ci_upper: tuple[float, float, float]         # 97.5th percentile

    def parameter_dict(self) -> dict[str, dict[str, float]]:
        names = ("location", "scale", "shape")
        return {
            n: {
                "mean": self.means[i],
                "sd": self.sds[i],
                "ci_lower": self.ci_lower[i],
                "ci_upper": self.ci_upper[i],
            }
            for i, n in enumerate(names)
        }


def _summarize(samples: np.ndarray, method: str, n_iter: int) -> BootstrapResult:
    if samples.size == 0:
        nans = (float("nan"), float("nan"), float("nan"))
        return BootstrapResult(
            method=method,
            n_iterations=n_iter,
            n_successful=0,
            samples=samples,
            means=nans,
            sds=nans,
            ci_lower=nans,
            ci_upper=nans,
        )
    means = tuple(float(x) for x in samples.mean(axis=0))
    sds = tuple(float(x) for x in samples.std(axis=0, ddof=1))
    lo = tuple(float(x) for x in np.quantile(samples, 0.025, axis=0))
    hi = tuple(float(x) for x in np.quantile(samples, 0.975, axis=0))
    return BootstrapResult(
        method=method,
        n_iterations=n_iter,
        n_successful=int(samples.shape[0]),
        samples=samples,
        means=means,  # type: ignore[arg-type]
        sds=sds,      # type: ignore[arg-type]
        ci_lower=lo,  # type: ignore[arg-type]
        ci_upper=hi,  # type: ignore[arg-type]
    )


def parametric_bootstrap(
    fit: GEVFit,
    *,
    n_iter: int,
    direction: str = "max",
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Sample ``n_iter`` synthetic datasets from the fitted GEV; refit each.

    For ``direction == "min"``, ``fit.location`` is the displayed (negated)
    location; the underlying GEV was fit to -X. We sample from the underlying
    fit, negate to X-space, then negate again and refit so the returned
    location is in the same negated-back convention as the original fit.
    """
    rng = rng or np.random.default_rng()
    n = fit.n_observations
    # Underlying location used for sampling: for min-direction, recover μ_fit
    # (the location parameter for -X) by negating the displayed location.
    underlying_loc = -fit.location if direction == "min" else fit.location

    rows: list[tuple[float, float, float]] = []
    for _ in range(int(n_iter)):
        u = rng.uniform(1e-9, 1.0 - 1e-9, size=n)
        sample_underlying = gev_quantile(u, underlying_loc, fit.scale, fit.shape)
        try:
            if direction == "min":
                # sample_underlying lives in -X space; negate to get X samples,
                # then negate again for fitting in -X space, and negate location
                # back when storing.
                refit = fit_gev(sample_underlying)  # already in -X space
                if refit.converged and np.isfinite(refit.log_likelihood):
                    rows.append((-refit.location, refit.scale, refit.shape))
            else:
                refit = fit_gev(sample_underlying)
                if refit.converged and np.isfinite(refit.log_likelihood):
                    rows.append((refit.location, refit.scale, refit.shape))
        except Exception:  # noqa: BLE001 - non-converging fits skipped
            continue
    return _summarize(np.array(rows), method="parametric", n_iter=int(n_iter))


def nonparametric_bootstrap(
    maxima: np.ndarray,
    *,
    n_iter: int,
    direction: str = "max",
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Resample observed values with replacement; refit each.

    For ``direction == "min"``, negate samples before fitting (so the GEV is
    fit to -X, matching the original MLE convention) and negate the location
    back for storage. This makes the nonparametric mean / CI directly
    comparable to the MLE point estimate.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(maxima, dtype=float)
    n = x.size
    rows: list[tuple[float, float, float]] = []
    for _ in range(int(n_iter)):
        sample = rng.choice(x, size=n, replace=True)
        try:
            if direction == "min":
                refit = fit_gev(-sample)
                if refit.converged and np.isfinite(refit.log_likelihood):
                    rows.append((-refit.location, refit.scale, refit.shape))
            else:
                refit = fit_gev(sample)
                if refit.converged and np.isfinite(refit.log_likelihood):
                    rows.append((refit.location, refit.scale, refit.shape))
        except Exception:  # noqa: BLE001
            continue
    return _summarize(np.array(rows), method="nonparametric", n_iter=int(n_iter))
