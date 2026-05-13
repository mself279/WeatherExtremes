"""NOAA Storm Events Database ingestion + exploration.

Pulls filtered event lists from NCEI's ``stormevents/csv`` endpoint, parses
the returned CSV, and produces exploration summaries + Plotly figures.

Unlike the GHCN ingestion, this package does **not** maintain a local bulk
mirror of the entire dataset. Each user query goes directly to NCEI with
the user's filters; the response (typically <1 MB) is cached on disk by a
hash of the filter parameters so identical re-queries skip the network.

Modules
-------
* :mod:`states`      — FIPS code → state name lookup (static)
* :mod:`event_types` — curated event types with their NCEI URL values
* :mod:`fetcher`     — build URL, GET CSV from NCEI, disk-cache the response
* :mod:`formats`     — parse the returned CSV into typed records
* :mod:`summarize`   — per-year / per-month / per-magnitude / per-county summaries
                       plus Plotly figure builders for the exploration page
"""
