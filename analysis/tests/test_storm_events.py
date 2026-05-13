"""Tests for the Storm Events service layer (no network required)."""
from __future__ import annotations

import pandas as pd

from analysis.services.storm_events.event_types import (
    EVENT_TYPES,
    by_code,
    event_type_choices,
)
from analysis.services.storm_events.fetcher import FilterParams, build_url
from analysis.services.storm_events.formats import parse_events_csv
from analysis.services.storm_events.states import (
    ALL_STATES_FIPS,
    display_name,
    state_choices,
)
from analysis.services.storm_events.summarize import (
    annual_counts_figure,
    magnitude_breakdown_figure,
    summarize,
)


# ---------------------------------------------------------------------------
# states.py
# ---------------------------------------------------------------------------


def test_state_choices_includes_all_and_missouri():
    choices = state_choices()
    assert (ALL_STATES_FIPS, "All states") in choices
    assert ("29,MISSOURI", "Missouri") in choices
    # All 50 states + DC + PR = 52 + "All states" sentinel = 53
    assert len(choices) >= 52


def test_state_display_name_round_trip():
    assert display_name("29,MISSOURI") == "Missouri"
    assert display_name(ALL_STATES_FIPS) == "All states"
    assert display_name("99,FOO") == "99,FOO"  # falls back to the raw value


# ---------------------------------------------------------------------------
# event_types.py
# ---------------------------------------------------------------------------


def test_event_types_have_distinct_codes():
    codes = [s.code for s in EVENT_TYPES]
    assert len(codes) == len(set(codes))


def test_event_type_choices_starts_with_tornado():
    choices = event_type_choices()
    assert choices[0] == ("tornado", "Tornado")


def test_tornado_event_type_has_tornado_magnitude_filter():
    spec = by_code("tornado")
    assert spec is not None
    assert spec.ncei_value == "(C) Tornado"
    assert spec.magnitude_field == "tornfilter"


# ---------------------------------------------------------------------------
# fetcher.py — URL building (no network)
# ---------------------------------------------------------------------------


def test_build_url_includes_all_required_params():
    p = FilterParams(
        statefips="29,MISSOURI",
        event_type_ncei="(C) Tornado",
        begin_year=2020,
        end_year=2026,
    )
    url = build_url(p)
    assert "eventType=%28C%29+Tornado" in url
    assert "statefips=29%2CMISSOURI" in url
    assert "beginDate_yyyy=2020" in url
    assert "endDate_yyyy=2026" in url
    assert "submitbutton=Search" in url


def test_cache_key_stable_and_param_sensitive():
    p1 = FilterParams("29,MISSOURI", "(C) Tornado", 2020, 2026)
    p2 = FilterParams("29,MISSOURI", "(C) Tornado", 2020, 2026)
    p3 = FilterParams("29,MISSOURI", "(C) Tornado", 2021, 2026)
    assert p1.cache_key() == p2.cache_key()
    assert p1.cache_key() != p3.cache_key()


# ---------------------------------------------------------------------------
# formats.py — CSV parsing
# ---------------------------------------------------------------------------


def _csv(rows: list[str], header: str | None = None) -> bytes:
    h = header or (
        "EVENT_ID,EVENT_TYPE,STATE,STATE_FIPS,CZ_NAME,CZ_FIPS,"
        "BEGIN_DATE_TIME,END_DATE_TIME,YEAR,MAGNITUDE,TOR_F_SCALE,"
        "INJURIES_DIRECT,DEATHS_DIRECT,DAMAGE_PROPERTY,DAMAGE_CROPS,"
        "BEGIN_LAT,BEGIN_LON,END_LAT,END_LON"
    )
    return ("\n".join([h] + rows) + "\n").encode("utf-8")


def test_parse_empty_csv_returns_empty_df():
    parsed = parse_events_csv(b"")
    assert parsed.df.empty


def test_parse_header_only_returns_empty_df():
    parsed = parse_events_csv(_csv([]))
    assert parsed.df.empty
    assert parsed.raw_rows == 0


def test_parse_three_tornado_rows():
    rows = [
        "12345,Tornado,MISSOURI,29,GREENE,77,11-MAY-2011 14:50:00,11-MAY-2011 15:12:00,2011,,EF3,12,3,$25.0M,$1.5M,37.21,-93.30,37.30,-93.10",
        "12346,Tornado,MISSOURI,29,JASPER,97,22-MAY-2011 17:34:00,22-MAY-2011 18:12:00,2011,,EF5,1150,158,$2.8B,$0,37.06,-94.51,37.10,-94.30",
        "12347,Tornado,MISSOURI,29,GREENE,77,03-APR-2020 21:00:00,03-APR-2020 21:18:00,2020,,EF1,0,0,$50K,$0,37.20,-93.28,37.25,-93.20",
    ]
    parsed = parse_events_csv(_csv(rows))
    df = parsed.df
    assert parsed.raw_rows == 3
    assert df.iloc[1]["injuries_direct"] == 1150
    assert df.iloc[0]["damage_property_usd"] == 25_000_000.0
    assert df.iloc[1]["damage_property_usd"] == 2_800_000_000.0
    assert df.iloc[2]["damage_property_usd"] == 50_000.0


def test_parse_handles_missing_optional_columns():
    # NCEI sometimes omits lat/lon and tor_f_scale for non-tornado queries.
    header = "EVENT_ID,EVENT_TYPE,STATE,STATE_FIPS,CZ_NAME,BEGIN_DATE_TIME,YEAR,MAGNITUDE,DAMAGE_PROPERTY"
    rows = ["999,Hail,KANSAS,20,SEDGWICK,15-JUN-2010 16:00:00,2010,2.50,$10K"]
    parsed = parse_events_csv(_csv(rows, header=header))
    df = parsed.df
    assert parsed.raw_rows == 1
    assert df.iloc[0]["magnitude"] == 2.50
    # Missing columns should be NaN or empty
    assert pd.isna(df.iloc[0]["begin_lat"])


def test_parse_handles_interactive_endpoint_format():
    """Regression: the interactive search CSV uses different column names than the
    bulk archive. Both must parse correctly with the same code path."""
    header = (
        "EVENT_ID,CZ_NAME_STR,BEGIN_LOCATION,BEGIN_DATE,BEGIN_TIME,EVENT_TYPE,"
        "MAGNITUDE,TOR_F_SCALE,DEATHS_DIRECT,INJURIES_DIRECT,"
        "DAMAGE_PROPERTY_NUM,DAMAGE_CROPS_NUM,STATE_ABBR,CZ_TIMEZONE,"
        "MAGNITUDE_TYPE,EPISODE_ID,CZ_TYPE,CZ_FIPS,WFO,INJURIES_INDIRECT,"
        "DEATHS_INDIRECT,SOURCE,FLOOD_CAUSE,TOR_LENGTH,TOR_WIDTH,BEGIN_RANGE,"
        "BEGIN_AZIMUTH,END_RANGE,END_AZIMUTH,END_LOCATION,END_DATE,END_TIME,"
        "BEGIN_LAT,BEGIN_LON,END_LAT,END_LON,EVENT_NARRATIVE,EPISODE_NARRATIVE,"
        "ABSOLUTE_ROWNUMBER"
    )
    rows = [
        # Three real Missouri tornado rows from the NCEI interactive CSV.
        "10063615,ST. LOUIS CO.,,01/03/1950,1100,Tornado,0,F3,0,3,2500000,0,MO,CST,,,C,189,,0,0,,,6.2,150,,,,,,01/03/1950,1100,38.77,-90.22,38.82,-90.12,,,1",
        "10063626,NEW MADRID CO.,,11/13/1951,1330,Tornado,0,F3,0,1,25000,0,MO,CST,,,C,143,,0,0,,,1,27,,,,,,11/13/1951,1330,36.62,-89.75,,,,,2",
        "10063632,PEMISCOT CO.,,03/21/1952,2000,Tornado,0,F4,17,100,2500000,0,MO,CST,,,C,155,,0,0,,,6.5,880,,,,,,03/21/1952,2000,36.05,-89.82,36.07,-89.70,,,3",
    ]
    parsed = parse_events_csv(_csv(rows, header=header))
    df = parsed.df
    assert parsed.raw_rows == 3

    # All dates parsed successfully — none NaT.
    assert df["year"].notna().all()
    assert list(df["year"].astype(int)) == [1950, 1951, 1952]
    # Months too (1, 11, 3 from the begin dates).
    assert list(df["month"].astype(int)) == [1, 11, 3]

    # Counties found via CZ_NAME_STR fallback.
    assert df["cz_name"].iloc[0] == "ST. LOUIS CO."
    assert df["cz_name"].iloc[2] == "PEMISCOT CO."

    # Damage values come from DAMAGE_PROPERTY_NUM (raw integers).
    assert df["damage_property_usd"].iloc[0] == 2_500_000.0
    assert df["damage_property_usd"].iloc[1] == 25_000.0
    assert df["damage_property_usd"].iloc[2] == 2_500_000.0

    # Casualty fields still work (unchanged column names).
    assert df["deaths_direct"].iloc[2] == 17
    assert df["injuries_direct"].iloc[2] == 100


def test_parse_handles_damage_edge_cases():
    rows = [
        "1,Tornado,MO,29,X,1,01-JAN-2020 00:00:00,01-JAN-2020 00:30:00,2020,,EF2,0,0,$1.5K,$0,0,0,0,0",
        "2,Tornado,MO,29,X,1,01-JAN-2020 00:00:00,01-JAN-2020 00:30:00,2020,,EF2,0,0,,,0,0,0,0",
        "3,Tornado,MO,29,X,1,01-JAN-2020 00:00:00,01-JAN-2020 00:30:00,2020,,EF2,0,0,$0,$0,0,0,0,0",
    ]
    parsed = parse_events_csv(_csv(rows))
    df = parsed.df
    assert df.iloc[0]["damage_property_usd"] == 1500.0
    assert pd.isna(df.iloc[1]["damage_property_usd"])
    assert df.iloc[2]["damage_property_usd"] == 0.0


# ---------------------------------------------------------------------------
# summarize.py
# ---------------------------------------------------------------------------


def test_summarize_zero_events():
    parsed = parse_events_csv(_csv([]))
    s = summarize(parsed.df)
    assert s.n_events == 0
    assert s.year_first is None
    assert s.annual_counts == []
    assert s.magnitude_breakdown == {}


def test_summarize_three_events_aggregates_correctly():
    rows = [
        "1,Tornado,MO,29,GREENE,77,01-JAN-2011 00:00:00,01-JAN-2011 00:30:00,2011,,EF3,12,3,$25M,$0,0,0,0,0",
        "2,Tornado,MO,29,JASPER,97,22-MAY-2011 17:34:00,22-MAY-2011 18:12:00,2011,,EF5,1150,158,$2.8B,$0,0,0,0,0",
        "3,Tornado,MO,29,GREENE,77,03-APR-2020 21:00:00,03-APR-2020 21:18:00,2020,,EF1,0,0,$50K,$0,0,0,0,0",
    ]
    parsed = parse_events_csv(_csv(rows))
    s = summarize(parsed.df)
    assert s.n_events == 3
    assert s.year_first == 2011 and s.year_last == 2020
    assert s.total_deaths == 161   # 3 + 158 + 0
    assert s.total_injuries == 1162  # 12 + 1150 + 0
    # Annual counts fill in the gap years as zeros
    yr_map = dict(s.annual_counts)
    assert yr_map[2011] == 2
    assert yr_map[2012] == 0
    assert yr_map[2020] == 1
    # Top counties is sorted desc and casts to Title Case
    assert s.top_counties[0] == ("Greene", 2)
    # F/EF scale breakdown
    assert s.magnitude_breakdown == {"EF1": 1, "EF3": 1, "EF5": 1}


def test_summarize_annual_damage_property_aggregates_by_year_full_range():
    # Two 2011 tornadoes and one 2020 — annual_damage_property should fill
    # 2012-2019 as zeros so the chart shows the gap visibly.
    rows = [
        "1,Tornado,MO,29,A,1,01-JAN-2011 00:00:00,01-JAN-2011 00:30:00,2011,,EF3,0,0,$25M,$0,0,0,0,0",
        "2,Tornado,MO,29,B,1,02-JAN-2011 00:00:00,02-JAN-2011 00:30:00,2011,,EF3,0,0,$10M,$0,0,0,0,0",
        "3,Tornado,MO,29,C,1,03-APR-2020 00:00:00,03-APR-2020 00:30:00,2020,,EF1,0,0,$1M,$0,0,0,0,0",
    ]
    parsed = parse_events_csv(_csv(rows))
    s = summarize(parsed.df)
    by_year = dict(s.annual_damage_property)
    assert by_year[2011] == 35_000_000.0
    assert by_year[2012] == 0.0
    assert by_year[2019] == 0.0
    assert by_year[2020] == 1_000_000.0
    assert s.total_damage_property_usd == 36_000_000.0


def test_summarize_handles_all_nan_damage_with_zero_total():
    # No damage values populated — total_damage stays at 0 (the view uses this
    # to suppress the chart entirely).
    rows = [
        "1,Lightning,MO,29,A,1,01-JAN-2020 00:00:00,01-JAN-2020 00:01:00,2020,,,0,0,,,0,0,0,0",
    ]
    parsed = parse_events_csv(_csv(rows))
    s = summarize(parsed.df)
    assert s.total_damage_property_usd == 0.0
    # annual_damage_property is still computed (all zeros); view checks total > 0
    # to decide whether to render the figure.
    assert s.annual_damage_property == [(2020, 0.0)]


def test_summarize_numeric_magnitude_bucketing_for_hail():
    # No TOR_F_SCALE → use numeric magnitude bucketing
    header = "EVENT_ID,EVENT_TYPE,STATE,STATE_FIPS,CZ_NAME,BEGIN_DATE_TIME,YEAR,MAGNITUDE,DAMAGE_PROPERTY,DAMAGE_CROPS"
    rows = [
        "1,Hail,KS,20,A,15-JUN-2010 16:00:00,2010,1.00,$0,$0",
        "2,Hail,KS,20,B,15-JUN-2010 16:00:00,2010,1.00,$0,$0",
        "3,Hail,KS,20,C,15-JUN-2010 16:00:00,2010,2.00,$0,$0",
    ]
    parsed = parse_events_csv(_csv(rows, header=header))
    s = summarize(parsed.df)
    assert s.magnitude_breakdown == {"1": 2, "2": 1}


def test_figure_builders_return_valid_json_strings():
    try:
        import plotly  # noqa: F401
    except ImportError:
        return  # plotly only required at runtime; skip when not installed
    rows = [
        "1,Tornado,MO,29,GREENE,77,01-JAN-2011 00:00:00,01-JAN-2011 00:30:00,2011,,EF3,12,3,$25M,$0,0,0,0,0",
    ]
    parsed = parse_events_csv(_csv(rows))
    s = summarize(parsed.df)
    import json
    for fn, kwargs in [
        (annual_counts_figure, {"title": "Annual"}),
        (magnitude_breakdown_figure, {"title": "Magnitude", "unit": "scale"}),
    ]:
        out = fn(s, **kwargs)
        parsed_json = json.loads(out)
        assert "data" in parsed_json
        assert "layout" in parsed_json
