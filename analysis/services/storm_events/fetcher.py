"""HTTP fetch + on-disk cache for NCEI Storm Events CSV.

We hit NCEI's interactive ``stormevents/csv`` endpoint with the user's
filter parameters. The same query always returns the same data (historical
events don't change once recorded), so we cache aggressively on a SHA256
hash of the parameter dict. Cache TTL defaults to 30 days; pass
``force=True`` to bypass.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

NCEI_CSV_ENDPOINT = "https://www.ncei.noaa.gov/stormevents/csv"
USER_AGENT = (
    "WeatherExtremes/0.1 (educational Django project; "
    "github.com/matthewself/WeatherExtremes)"
)
DEFAULT_TIMEOUT = 60.0  # seconds


class FetchError(RuntimeError):
    """Raised when the request fails or NCEI returns something unparseable."""


@dataclass(frozen=True)
class FilterParams:
    """User-facing filter values; transformed into NCEI URL params by ``build_url``."""

    statefips: str                  # e.g. "29,MISSOURI" or "-999,ALL"
    event_type_ncei: str            # e.g. "(C) Tornado"
    begin_year: int
    end_year: int
    county: str = "ALL"             # NCEI county name; "ALL" = no county filter
    tornfilter: str = "0"           # "0".."5"
    hailfilter: str = "0.00"        # e.g. "0.00", "1.00", "2.00"
    windfilter: str = "000"         # e.g. "000", "050", "075", "100"

    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "endpoint": NCEI_CSV_ENDPOINT,
                **{k: getattr(self, k) for k in self.__dataclass_fields__},
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def build_url(p: FilterParams) -> str:
    """Translate ``FilterParams`` into the NCEI URL.

    NCEI's interactive endpoint expects all date components broken out
    (``beginDate_mm``, ``beginDate_dd``, ``beginDate_yyyy``, …) and the
    state encoded as ``<FIPS>,<NAME>``.
    """
    params = [
        ("eventType", p.event_type_ncei),
        ("beginDate_mm", "01"),
        ("beginDate_dd", "01"),
        ("beginDate_yyyy", str(int(p.begin_year))),
        ("endDate_mm", "12"),
        ("endDate_dd", "31"),
        ("endDate_yyyy", str(int(p.end_year))),
        ("county", p.county),
        ("hailfilter", p.hailfilter),
        ("tornfilter", p.tornfilter),
        ("windfilter", p.windfilter),
        ("sort", "DT"),
        ("submitbutton", "Search"),
        ("statefips", p.statefips),
    ]
    return f"{NCEI_CSV_ENDPOINT}?" + urllib.parse.urlencode(params)


def _is_fresh(path: Path, max_age: timedelta | None) -> bool:
    if not path.exists():
        return False
    if max_age is None:
        return True
    return (time.time() - path.stat().st_mtime) < max_age.total_seconds()


def fetch_csv(
    params: FilterParams,
    cache_dir: Path,
    *,
    force: bool = False,
    max_age: timedelta | None = timedelta(days=30),
    timeout: float = DEFAULT_TIMEOUT,
) -> bytes:
    """Return the raw CSV bytes, using disk cache when possible.

    Raises :class:`FetchError` on HTTP failure, non-CSV response, or empty
    bytes. An empty result set (header only, zero data rows) is **not** an
    error — the parser handles that.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{params.cache_key()}.csv"

    if not force and _is_fresh(cache_path, max_age):
        return cache_path.read_bytes()

    url = build_url(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = resp.read()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"Could not fetch {url}: {exc}") from exc

    if not data:
        raise FetchError(f"Empty response from {url}")

    # NCEI sometimes returns HTML (error page or session redirect). Treat
    # that as a fetch failure rather than letting it through to the parser.
    head = data[:200].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        raise FetchError(
            f"NCEI returned HTML rather than CSV for {url}. "
            "The endpoint may have changed or the query may be invalid."
        )

    # Write the cache file atomically.
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(cache_path)
    return data
