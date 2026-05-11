"""End-to-end smoke test for the analysis pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.services.pipeline import AnalysisParameters, run_analysis


def _synthetic_dataset(seed: int = 42) -> pd.DataFrame:
    """1990–2010 daily grid with a non-stationary event rate."""
    rng = np.random.default_rng(seed)
    grid = np.arange(1990.0, 2010.0, 1.0 / 365.0)
    # Linearly increasing probability of an event over time.
    p = np.linspace(0.005, 0.05, num=len(grid))
    flag = (rng.random(len(grid)) < p).astype(int)
    return pd.DataFrame(
        {"Event_Date": np.round(grid, 4), "Over3": flag, "Over4": flag * (rng.random(len(grid)) < 0.3)}
    )


def test_pipeline_produces_figures_and_summary():
    df = _synthetic_dataset()
    params = AnalysisParameters(
        indicator_column="Over3",
        observation_start=1990.0,
        observation_end=2010.0,
        observation_factor=1.0 / 365.0,
        bandwidth_years=3.0,
        bias_correction=True,
        bootstrap_iterations=20,
        bandwidth_search_min=1.0,
        bandwidth_search_max=6.0,
        bandwidth_search_step=1.0,
        additional_indicators=("Over4",),
        decluster_tau=0,
        rng_seed=7,
    )
    result = run_analysis(df, params)

    assert "occurrence_rate" in result.figures
    assert "confidence_band" in result.figures
    assert "bandwidth_score" in result.figures
    assert "multi_threshold" in result.figures

    assert result.summary["event_count"] > 0
    assert 1990.0 <= result.summary["peak_year"] <= 2010.0
    # Non-stationary input should produce a higher rate later in the window.
    grid = np.array(result.series["grid"])
    rate = np.array(result.series["point_estimate"])
    early = rate[grid < 1995].mean()
    late = rate[grid > 2005].mean()
    assert late > early
