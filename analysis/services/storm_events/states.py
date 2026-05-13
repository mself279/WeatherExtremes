"""US state / territory FIPS codes used by NCEI Storm Events.

NCEI sends ``statefips=<NUM>,<NAME>`` (e.g., ``29,MISSOURI``) in their
filter URLs. We mirror that exact format because the endpoint is
strict — both the numeric code and the uppercase name must be present.
"""
from __future__ import annotations

# (FIPS, uppercase name, title-case display name)
_STATES: tuple[tuple[str, str, str], ...] = (
    ("1", "ALABAMA", "Alabama"),
    ("2", "ALASKA", "Alaska"),
    ("4", "ARIZONA", "Arizona"),
    ("5", "ARKANSAS", "Arkansas"),
    ("6", "CALIFORNIA", "California"),
    ("8", "COLORADO", "Colorado"),
    ("9", "CONNECTICUT", "Connecticut"),
    ("10", "DELAWARE", "Delaware"),
    ("11", "DISTRICT OF COLUMBIA", "District of Columbia"),
    ("12", "FLORIDA", "Florida"),
    ("13", "GEORGIA", "Georgia"),
    ("15", "HAWAII", "Hawaii"),
    ("16", "IDAHO", "Idaho"),
    ("17", "ILLINOIS", "Illinois"),
    ("18", "INDIANA", "Indiana"),
    ("19", "IOWA", "Iowa"),
    ("20", "KANSAS", "Kansas"),
    ("21", "KENTUCKY", "Kentucky"),
    ("22", "LOUISIANA", "Louisiana"),
    ("23", "MAINE", "Maine"),
    ("24", "MARYLAND", "Maryland"),
    ("25", "MASSACHUSETTS", "Massachusetts"),
    ("26", "MICHIGAN", "Michigan"),
    ("27", "MINNESOTA", "Minnesota"),
    ("28", "MISSISSIPPI", "Mississippi"),
    ("29", "MISSOURI", "Missouri"),
    ("30", "MONTANA", "Montana"),
    ("31", "NEBRASKA", "Nebraska"),
    ("32", "NEVADA", "Nevada"),
    ("33", "NEW HAMPSHIRE", "New Hampshire"),
    ("34", "NEW JERSEY", "New Jersey"),
    ("35", "NEW MEXICO", "New Mexico"),
    ("36", "NEW YORK", "New York"),
    ("37", "NORTH CAROLINA", "North Carolina"),
    ("38", "NORTH DAKOTA", "North Dakota"),
    ("39", "OHIO", "Ohio"),
    ("40", "OKLAHOMA", "Oklahoma"),
    ("41", "OREGON", "Oregon"),
    ("42", "PENNSYLVANIA", "Pennsylvania"),
    ("44", "RHODE ISLAND", "Rhode Island"),
    ("45", "SOUTH CAROLINA", "South Carolina"),
    ("46", "SOUTH DAKOTA", "South Dakota"),
    ("47", "TENNESSEE", "Tennessee"),
    ("48", "TEXAS", "Texas"),
    ("49", "UTAH", "Utah"),
    ("50", "VERMONT", "Vermont"),
    ("51", "VIRGINIA", "Virginia"),
    ("53", "WASHINGTON", "Washington"),
    ("54", "WEST VIRGINIA", "West Virginia"),
    ("55", "WISCONSIN", "Wisconsin"),
    ("56", "WYOMING", "Wyoming"),
    ("72", "PUERTO RICO", "Puerto Rico"),
)


# Sentinel value NCEI uses for "all states".
ALL_STATES_FIPS = "-999,ALL"


def state_choices() -> list[tuple[str, str]]:
    """List of ``(value, display)`` pairs for a Django ChoiceField.

    ``value`` is the NCEI URL-ready string ``"<FIPS>,<NAME>"``.
    """
    out: list[tuple[str, str]] = [(ALL_STATES_FIPS, "All states")]
    for fips, upper, title in _STATES:
        out.append((f"{fips},{upper}", title))
    return out


def display_name(value: str) -> str:
    """Convert the URL value back to a human-readable name."""
    if value == ALL_STATES_FIPS:
        return "All states"
    for fips, upper, title in _STATES:
        if f"{fips},{upper}" == value:
            return title
    return value
