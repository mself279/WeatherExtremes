"""HTTP download with on-disk cache and max-age policy.

We use ``urllib.request`` so the package adds no extra third-party
dependency. NOAA's ``/pub/data/ghcn/daily/`` is plain HTTPS with no auth.
A polite User-Agent identifies the app.

The cache layout under ``cache_dir`` is::

    ghcnd-stations.txt
    ghcnd-states.txt
    ghcnd-inventory.txt
    by_station/USW00013994.csv.gz
    by_station/...
"""
from __future__ import annotations

import shutil
import time
import urllib.request
from datetime import timedelta
from pathlib import Path

GHCN_BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/"
USER_AGENT = (
    "WeatherExtremes/0.1 (educational Django project; "
    "github.com/matthewself/WeatherExtremes)"
)
DEFAULT_TIMEOUT = 60.0  # seconds


class FetchError(RuntimeError):
    """Raised when a download fails or the response is empty."""


def _is_fresh(path: Path, max_age: timedelta | None) -> bool:
    if not path.exists():
        return False
    if max_age is None:
        return True
    age = time.time() - path.stat().st_mtime
    return age < max_age.total_seconds()


def download(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    max_age: timedelta | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Path:
    """Download ``url`` to ``dest``; skip if cached and fresh.

    Returns the destination path. Raises :class:`FetchError` on failure.
    """
    if not force and _is_fresh(dest, max_age):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - we control URL
            with tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
    except Exception as exc:  # noqa: BLE001 - we wrap into FetchError
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise FetchError(f"Could not download {url}: {exc}") from exc

    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"Empty response from {url}")

    tmp.replace(dest)
    return dest


def fetch_metadata(
    cache_dir: Path,
    *,
    force: bool = False,
    max_age: timedelta | None = timedelta(days=30),
) -> dict[str, Path]:
    """Fetch the three metadata files; returns a name → path mapping."""
    out: dict[str, Path] = {}
    for name in ("ghcnd-states.txt", "ghcnd-stations.txt", "ghcnd-inventory.txt"):
        out[name] = download(
            GHCN_BASE + name,
            cache_dir / name,
            force=force,
            max_age=max_age,
        )
    return out


def fetch_station(
    station_id: str,
    cache_dir: Path,
    *,
    force: bool = False,
    max_age: timedelta | None = timedelta(days=7),
) -> Path:
    """Fetch the per-station ``.csv.gz`` file; returns its cached path."""
    if not (len(station_id) == 11 and station_id.isalnum()):
        raise ValueError(f"Invalid GHCN station id: {station_id!r}")
    fname = f"{station_id}.csv.gz"
    return download(
        GHCN_BASE + f"by_station/{fname}",
        cache_dir / "by_station" / fname,
        force=force,
        max_age=max_age,
    )
