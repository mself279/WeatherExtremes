# WeatherExtremes

A Django web app for non-stationary occurrence-rate analysis of extreme weather
events using Gaussian kernel density estimation, with bootstrap confidence
bands and bandwidth selection via cross-validation. Refactored from the
original `Statistical Analysis - STL Precipitation.py` script.

## Features

- Upload event-time CSVs (e.g. days exceeding a precipitation threshold).
- Configure the analysis: observation window, bandwidth, bias correction,
  bootstrap iterations.
- Run the full pipeline: smoothed occurrence rate, multi-bandwidth comparison,
  bandwidth selection (MISE), bootstrap confidence interval, autocorrelation
  declustering.
- Interactive Plotly charts in the browser (zoom, pan, hover).
- Persist runs in the database and compare results across runs side by side.
- Built-in STL Precipitation sample dataset (`fixtures/stl_precipitation_sample.csv`).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python manage.py migrate
python manage.py load_sample          # loads the STL Precipitation dataset
python manage.py createsuperuser      # optional
python manage.py runserver
```

Then visit <http://127.0.0.1:8000/>.

## Project layout

```
WeatherExtremes/
├── config/                 # Django project (settings, root urls, wsgi/asgi)
├── analysis/               # Main app
│   ├── models.py           # Dataset, AnalysisRun
│   ├── forms.py            # Upload + parameter forms
│   ├── views.py            # Class-based views
│   ├── urls.py
│   ├── services/           # Pure-Python analysis library (no Django imports)
│   │   ├── kde.py          # Gaussian kernel occurrence-rate estimator
│   │   ├── bandwidth.py    # Bandwidth selection (MISE / cross-validation)
│   │   ├── bootstrap.py    # Resampling for confidence intervals
│   │   ├── autocorr.py     # Tau-based event declustering
│   │   ├── plots.py        # Plotly figure builders
│   │   ├── io.py           # CSV parsing
│   │   └── pipeline.py     # Orchestrator: dataset + params -> results
│   ├── management/commands/load_sample.py
│   └── tests/
└── fixtures/stl_precipitation_sample.csv
```

The `services/` package has no Django dependencies, so it's directly testable
and trivially callable from a Celery / Django-Q task later when the analysis
needs to move off the request thread.

## Input file format

A CSV with a numeric `Event_Date` column expressed as fractional years
(`1938.2466` ≈ early April 1938) and one or more 0/1 indicator columns
(e.g. `Over3`, `Over4`) marking days where the event occurred. The original
STL Precipitation file is the reference example.

## Statistical method

Following Mudelsee, *Climate Time Series Analysis* (2nd ed.), Chapter 6:

1. Build a regular time grid on `[observation_start, observation_end]`.
2. Mark event dates with a 0/1 indicator.
3. Estimate the occurrence-rate λ(t) with a Gaussian kernel of bandwidth `h`.
4. Reduce boundary bias by reflecting `3h` of data outside both ends.
5. Pick `h` by minimizing MISE via the leave-one-out estimator.
6. Build pointwise (1−α) confidence bands by bootstrapping the event list.
7. Optionally decluster using tau from the event-indicator autocorrelation.

## Roadmap

- [ ] Move long-running runs to a background worker (Celery + Redis).
- [ ] Multi-user accounts; share datasets with permissions.
- [ ] REST API (DRF) so other apps can submit analysis jobs.
- [ ] GEV / POT extreme value modelling alongside the rate analysis.
