"""GHCN-Daily ingestion services.

This package handles everything specific to NOAA's Global Historical
Climatology Network — Daily dataset:

* :mod:`elements`  — unit specs, default thresholds, conversions
* :mod:`formats`   — parsers for ghcnd-stations.txt, ghcnd-states.txt,
                     ghcnd-inventory.txt, and per-station CSVs
* :mod:`fetcher`   — HTTP downloads with on-disk cache + max-age policy
* :mod:`transform` — raw daily values → indicator-column CSV that the
                     existing analysis pipeline already understands

The package has no Django imports so it stays unit-testable in isolation
and is the natural place to factor a generic ``DataSourceAdapter`` later
when we add USGS streamflow / NOAA tides / SPC tornado / IBTrACS / etc.
"""
