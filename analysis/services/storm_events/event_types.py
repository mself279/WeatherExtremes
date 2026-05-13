"""Curated subset of NCEI Storm Events event types.

NCEI's URL parameter uses ``(C) <Name>`` strings — the ``(C)`` prefix is part
of the value the form posts. We keep a curated list of the most-asked-for
events; the full taxonomy has ~50 entries but most are rarely queried.

Each entry tracks the magnitude field that's relevant for that event type so
the UI can show the right filter (tornado scale vs hail size vs wind speed)
and the exploration summaries can label charts appropriately.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventTypeSpec:
    code: str                # short label used in our UI
    ncei_value: str          # exact string NCEI expects in the eventType param
    label: str               # display label
    magnitude_field: str     # "tornfilter" / "hailfilter" / "windfilter" / "" (none)
    magnitude_label: str     # what to call the magnitude axis on charts
    magnitude_unit: str      # e.g. "EF scale", "in (diameter)", "kt"


EVENT_TYPES: tuple[EventTypeSpec, ...] = (
    EventTypeSpec(
        code="tornado",
        ncei_value="(C) Tornado",
        label="Tornado",
        magnitude_field="tornfilter",
        magnitude_label="Tornado scale (F/EF)",
        magnitude_unit="scale",
    ),
    EventTypeSpec(
        code="hail",
        ncei_value="(C) Hail",
        label="Hail",
        magnitude_field="hailfilter",
        magnitude_label="Hail diameter",
        magnitude_unit="in",
    ),
    EventTypeSpec(
        code="thunderstorm_wind",
        ncei_value="(C) Thunderstorm Wind",
        label="Thunderstorm Wind",
        magnitude_field="windfilter",
        magnitude_label="Wind speed",
        magnitude_unit="kt",
    ),
    EventTypeSpec(
        code="high_wind",
        ncei_value="(C) High Wind",
        label="High Wind",
        magnitude_field="windfilter",
        magnitude_label="Wind speed",
        magnitude_unit="kt",
    ),
    EventTypeSpec(
        code="flash_flood",
        ncei_value="(C) Flash Flood",
        label="Flash Flood",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="flood",
        ncei_value="(C) Flood",
        label="Flood",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="winter_storm",
        ncei_value="(C) Winter Storm",
        label="Winter Storm",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="blizzard",
        ncei_value="(C) Blizzard",
        label="Blizzard",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="heavy_snow",
        ncei_value="(C) Heavy Snow",
        label="Heavy Snow",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="ice_storm",
        ncei_value="(C) Ice Storm",
        label="Ice Storm",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="heat",
        ncei_value="(C) Heat",
        label="Heat",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="excessive_heat",
        ncei_value="(C) Excessive Heat",
        label="Excessive Heat",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="drought",
        ncei_value="(C) Drought",
        label="Drought",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="lightning",
        ncei_value="(C) Lightning",
        label="Lightning",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="wildfire",
        ncei_value="(C) Wildfire",
        label="Wildfire",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
    EventTypeSpec(
        code="hurricane",
        ncei_value="(C) Hurricane",
        label="Hurricane",
        magnitude_field="",
        magnitude_label="Saffir-Simpson category",
        magnitude_unit="category",
    ),
    EventTypeSpec(
        code="tropical_storm",
        ncei_value="(C) Tropical Storm",
        label="Tropical Storm",
        magnitude_field="",
        magnitude_label="",
        magnitude_unit="",
    ),
)


def event_type_choices() -> list[tuple[str, str]]:
    """Django ChoiceField pairs (value=code, display=label)."""
    return [(spec.code, spec.label) for spec in EVENT_TYPES]


def by_code(code: str) -> EventTypeSpec | None:
    for spec in EVENT_TYPES:
        if spec.code == code:
            return spec
    return None
