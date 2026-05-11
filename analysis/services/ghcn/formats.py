"""Parsers for GHCN-Daily metadata files and per-station CSVs.

All column ranges in the docs are 1-based and inclusive
(`ID: cols 1-11`); Python slicing is 0-based and exclusive
(`line[0:11]`). Both perspectives appear in the comments below so the
mapping is easy to verify against the NOAA readme.
"""
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Iterator

import pandas as pd

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationRecord:
    id: str
    latitude: float
    longitude: float
    elevation: float | None  # meters; None when raw value was -999.9
    state: str
    name: str
    network_code: str         # 3rd char of ID (e.g. 'W' for WBAN)
    gsn_flag: str
    hcn_flag: str
    wmo_id: str


@dataclass(frozen=True)
class StateRecord:
    code: str
    name: str


@dataclass(frozen=True)
class InventoryRecord:
    station_id: str
    element: str
    first_year: int
    last_year: int


# ---------------------------------------------------------------------------
# ghcnd-stations.txt
# ---------------------------------------------------------------------------

# Cols 1-11 = ID, 13-20 = LAT, 22-30 = LON, 32-37 = ELEV (m, missing=-999.9),
# 39-40 = STATE, 42-71 = NAME, 73-75 = GSN_FLAG, 77-79 = HCN_FLAG, 81-85 = WMO_ID

def parse_stations(
    lines: Iterable[str], *, us_only: bool = True
) -> Iterator[StationRecord]:
    for raw in lines:
        line = raw.rstrip("\n")
        if len(line) < 41:  # ID + at least state field present
            continue
        sid = line[0:11].strip()
        if not sid:
            continue
        if us_only and not sid.startswith("US"):
            continue
        try:
            lat = float(line[12:20])
            lon = float(line[21:30])
        except ValueError:
            continue
        try:
            elev_raw = float(line[31:37])
            elevation: float | None = None if elev_raw <= -999.0 else elev_raw
        except ValueError:
            elevation = None
        state = line[38:40].strip() if len(line) >= 40 else ""
        name = line[41:71].strip() if len(line) >= 41 else ""
        gsn = line[72:75].strip() if len(line) >= 75 else ""
        hcn = line[76:79].strip() if len(line) >= 79 else ""
        wmo = line[80:85].strip() if len(line) >= 85 else ""
        network_code = sid[2] if len(sid) >= 3 else ""
        yield StationRecord(
            id=sid,
            latitude=lat,
            longitude=lon,
            elevation=elevation,
            state=state,
            name=name,
            network_code=network_code,
            gsn_flag=gsn,
            hcn_flag=hcn,
            wmo_id=wmo,
        )


# ---------------------------------------------------------------------------
# ghcnd-states.txt
# ---------------------------------------------------------------------------

# Cols 1-2 = CODE, 4-50 = NAME

def parse_states(lines: Iterable[str]) -> Iterator[StateRecord]:
    for raw in lines:
        line = raw.rstrip("\n")
        if len(line) < 4:
            continue
        code = line[0:2].strip()
        name = line[3:50].strip()
        if code and name:
            yield StateRecord(code=code, name=name)


# ---------------------------------------------------------------------------
# ghcnd-inventory.txt
# ---------------------------------------------------------------------------

# Cols 1-11 = ID, 13-20 = LAT, 22-30 = LON, 32-35 = ELEMENT,
# 37-40 = FIRSTYEAR, 42-45 = LASTYEAR

def parse_inventory(
    lines: Iterable[str],
    *,
    us_only: bool = True,
    keep_elements: set[str] | None = None,
) -> Iterator[InventoryRecord]:
    for raw in lines:
        line = raw.rstrip("\n")
        if len(line) < 45:
            continue
        sid = line[0:11].strip()
        if us_only and not sid.startswith("US"):
            continue
        element = line[31:35].strip()
        if keep_elements is not None and element not in keep_elements:
            continue
        try:
            first_year = int(line[36:40])
            last_year = int(line[41:45])
        except ValueError:
            continue
        yield InventoryRecord(
            station_id=sid,
            element=element,
            first_year=first_year,
            last_year=last_year,
        )


# ---------------------------------------------------------------------------
# by_station/<id>.csv(.gz)
# ---------------------------------------------------------------------------

# The per-station CSV has one row per (station, date, element). NCEI's
# /pub/data/ghcn/daily/by_station/ files do NOT have a header row; the AWS
# NODD mirror sometimes does. We sniff the first line and adapt.

_STATION_COLUMNS = ["ID", "DATE", "ELEMENT", "VALUE", "M_FLAG", "Q_FLAG", "S_FLAG", "OBS_TIME"]


def read_station_csv(path: str | Path | IO[bytes]) -> pd.DataFrame:
    """Load a per-station GHCN file (csv or csv.gz) into a normalized DataFrame.

    Returned columns: ID, DATE (string YYYYMMDD), ELEMENT, VALUE (int),
    M_FLAG, Q_FLAG, S_FLAG, OBS_TIME.
    """
    if isinstance(path, (str, Path)):
        path = Path(path)
        if path.suffix == ".gz":
            opener = lambda: gzip.open(path, "rb")  # noqa: E731
        else:
            opener = lambda: path.open("rb")  # noqa: E731
        with opener() as fh:
            data = fh.read()
    else:
        data = path.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        # Try gzip-magic detection on the buffer.
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)

    text = data.decode("utf-8", errors="replace")
    if not text.strip():
        return pd.DataFrame(columns=_STATION_COLUMNS)

    first_line = text.split("\n", 1)[0]
    has_header = first_line.upper().startswith(("ID,", "STATION,"))

    buf = io.StringIO(text)
    if has_header:
        df = pd.read_csv(buf, dtype=str, keep_default_na=False)
        df.columns = [c.strip().upper() for c in df.columns]
        if "STATION" in df.columns and "ID" not in df.columns:
            df = df.rename(columns={"STATION": "ID"})
    else:
        df = pd.read_csv(buf, header=None, names=_STATION_COLUMNS, dtype=str, keep_default_na=False)

    # Normalize to the canonical column set; missing optional cols become "".
    for col in _STATION_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[_STATION_COLUMNS].copy()

    # VALUE is always an integer per the spec; missing values may exist as "".
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    return df
