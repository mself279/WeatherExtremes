"""Plotly figure builders for the GEV detail page."""
from __future__ import annotations

import numpy as np

from .confidence import BootstrapResult
from .diagnostics import BucketDiagnostic, ReturnLevels
from .distribution import gev_pdf


_LAYOUT_DEFAULTS = dict(
    margin=dict(l=50, r=20, t=50, b=50),
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def annual_maxima_with_fit_figure(
    years: np.ndarray,
    maxima: np.ndarray,
    location: float,
    scale: float,
    shape: float,
    display_unit: str,
    title: str,
) -> str:
    """Annual maxima time series with fitted GEV percentile lines overlaid."""
    import plotly.graph_objects as go

    from .diagnostics import fitted_quantiles

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=maxima, mode="markers+lines",
        line=dict(color="rgb(31,119,180)", width=1),
        marker=dict(size=6),
        name=f"Annual maximum ({display_unit})",
    ))
    fq = fitted_quantiles(location, scale, shape)
    for prob, val in zip(fq.probabilities, fq.values):
        fig.add_hline(
            y=float(val),
            line_dash="dot",
            line_color="rgba(150,150,150,0.6)",
            annotation_text=f"p{int(prob*100)}={val:.2f}",
            annotation_position="right",
        )
    fig.update_layout(
        title=title, xaxis_title="Year",
        yaxis_title=f"Annual maximum ({display_unit})",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def histogram_pdf_figure(
    maxima: np.ndarray,
    location: float, scale: float, shape: float,
    display_unit: str,
    title: str,
    direction: str = "max",
    nonparam_mean: tuple[float, float, float] | None = None,
) -> str:
    """Empirical histogram (density-normalized) with fitted GEV PDF overlay.

    For ``direction == "min"``, ``location`` is the displayed/negated location;
    the underlying fit is to -X. The PDF of X at point x is then
    ``gev_pdf(-x; -location, scale, shape)``. If ``nonparam_mean`` is supplied
    we plot a second dashed curve so users can visually compare the MLE and
    nonparametric bootstrap mean fits.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=maxima, histnorm="probability density",
        nbinsx=20,
        marker=dict(color="rgba(31,119,180,0.55)", line=dict(width=0.5, color="rgb(31,119,180)")),
        name="Observed",
    ))
    x_min = float(np.min(maxima)) - 1.5 * np.std(maxima)
    x_max = float(np.max(maxima)) + 1.5 * np.std(maxima)
    xx = np.linspace(x_min, x_max, 400)

    def _pdf_for_direction(loc: float, scl: float, shp: float) -> np.ndarray:
        if direction == "max":
            return gev_pdf(xx, loc, scl, shp)
        # Min-direction: GEV was fit to -X, displayed location is -μ_fit.
        # PDF of X at x is gev_pdf(-x; μ_fit, σ, ξ) = gev_pdf(-x; -loc, σ, ξ).
        return gev_pdf(-xx, -loc, scl, shp)

    pdf_mle = _pdf_for_direction(location, scale, shape)
    fig.add_trace(go.Scatter(
        x=xx, y=pdf_mle, mode="lines",
        line=dict(color="red", width=2.5),
        name=f"MLE: GEV(μ={location:.2f}, σ={scale:.2f}, ξ={shape:.3f})",
    ))

    if nonparam_mean is not None:
        np_loc, np_scale, np_shape = nonparam_mean
        if np.all(np.isfinite([np_loc, np_scale, np_shape])) and np_scale > 0:
            pdf_np = _pdf_for_direction(np_loc, np_scale, np_shape)
            fig.add_trace(go.Scatter(
                x=xx, y=pdf_np, mode="lines",
                line=dict(color="rgb(255,127,14)", width=2.5, dash="dash"),
                name=(
                    f"Nonparametric mean: GEV(μ={np_loc:.2f}, "
                    f"σ={np_scale:.2f}, ξ={np_shape:.3f})"
                ),
            ))

    extreme_word = "maximum" if direction == "max" else "minimum"
    fig.update_layout(
        title=title,
        xaxis_title=f"Annual {extreme_word} ({display_unit})",
        yaxis_title="Density",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def bucket_figure(diag: BucketDiagnostic, display_unit: str, title: str) -> str:
    """Side-by-side observed vs expected counts per bucket.

    The x-axis is explicitly set to cover the full edge range so isolated
    outlier bars (e.g., a single 1954-style maximum at the right edge) are
    always visible — Plotly's auto-range can clip them when most counts
    cluster in a narrow region.
    """
    import plotly.graph_objects as go
    import numpy as np

    centers = (diag.edges[:-1] + diag.edges[1:]) / 2.0
    bucket_width = float(diag.edges[1] - diag.edges[0]) if len(diag.edges) >= 2 else 0.5

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=diag.observed,
        name="Observed",
        marker_color="rgb(31,119,180)",
        hovertemplate="bucket %{x:.2f}<br>observed=%{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=centers, y=diag.expected,
        name="Expected (GEV)",
        marker_color="rgba(220,60,60,0.7)",
        hovertemplate="bucket %{x:.2f}<br>expected=%{y:.2f}<extra></extra>",
    ))

    x_lo = float(diag.edges[0]) - bucket_width / 2.0
    x_hi = float(diag.edges[-1]) + bucket_width / 2.0
    y_max = float(max(np.max(diag.observed) if diag.observed.size else 1,
                      np.max(diag.expected) if diag.expected.size else 1))

    fig.update_layout(
        title=title,
        xaxis=dict(title=f"Bucket center ({display_unit})", range=[x_lo, x_hi]),
        yaxis=dict(title="Count", range=[0, y_max * 1.1 + 0.5]),
        barmode="group",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def return_level_figure(
    rl: ReturnLevels,
    display_unit: str,
    parametric_ci: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Return level plot",
) -> str:
    """Return level vs return period (log-x), with optional CI band."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if parametric_ci is not None:
        lo, hi = parametric_ci
        fig.add_trace(go.Scatter(
            x=np.concatenate([rl.return_periods, rl.return_periods[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself",
            fillcolor="rgba(120,120,120,0.18)",
            line=dict(width=0),
            hoverinfo="skip",
            name="95% CI (parametric)",
        ))
    fig.add_trace(go.Scatter(
        x=rl.return_periods, y=rl.levels,
        mode="markers+lines",
        line=dict(color="rgb(31,119,180)", width=2),
        marker=dict(size=8),
        name="Return level",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Return period (years, log scale)",
        yaxis_title=f"Return level ({display_unit})",
        xaxis_type="log",
        **_LAYOUT_DEFAULTS,
    )
    return fig.to_json()


def bootstrap_distributions_figure(
    parametric: BootstrapResult | None,
    nonparametric: BootstrapResult | None,
    point_estimate: tuple[float, float, float],
    title: str = "Bootstrap parameter distributions",
) -> str:
    """Three-panel histogram of bootstrap parameter distributions."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = ("Location (μ)", "Scale (σ)", "Shape (ξ)")
    fig = make_subplots(rows=1, cols=3, subplot_titles=names)
    for i, name in enumerate(names, start=1):
        if parametric is not None and parametric.samples.size:
            fig.add_trace(
                go.Histogram(
                    x=parametric.samples[:, i - 1],
                    nbinsx=30, opacity=0.55,
                    marker_color="rgb(31,119,180)",
                    name="Parametric" if i == 1 else None,
                    showlegend=(i == 1),
                ),
                row=1, col=i,
            )
        if nonparametric is not None and nonparametric.samples.size:
            fig.add_trace(
                go.Histogram(
                    x=nonparametric.samples[:, i - 1],
                    nbinsx=30, opacity=0.55,
                    marker_color="rgb(255,127,14)",
                    name="Nonparametric" if i == 1 else None,
                    showlegend=(i == 1),
                ),
                row=1, col=i,
            )
        fig.add_vline(
            x=float(point_estimate[i - 1]),
            row=1, col=i,
            line_dash="dash", line_color="black",
            annotation_text="MLE",
            annotation_position="top",
        )
    fig.update_layout(title=title, barmode="overlay",
                      template="plotly_white",
                      margin=dict(l=40, r=20, t=70, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1))
    return fig.to_json()
