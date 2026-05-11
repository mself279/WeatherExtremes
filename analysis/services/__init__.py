"""
Pure-Python analysis library for extreme-event occurrence rates.

This package has no Django imports so it can be unit-tested in isolation
and re-invoked from a background task queue (Celery, Django-Q, etc.) when
the analysis moves off the request thread.

Public entry point:

    from analysis.services.pipeline import run_analysis
"""
