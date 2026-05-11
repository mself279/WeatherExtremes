"""Load the bundled STL Precipitation sample dataset.

Usage::

    python manage.py load_sample
"""
from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand

from analysis.models import Dataset
from analysis.services.io import parse_input


class Command(BaseCommand):
    help = "Load the bundled STL Precipitation sample CSV as a Dataset."

    DEFAULT_NAME = "STL Precipitation (sample)"
    DEFAULT_DESCRIPTION = (
        "Daily precipitation event indicators for St. Louis, 1938–2024. "
        "Indicator columns mark days with precipitation exceeding 3, 4, 5, "
        "or 6 inches. Originally analyzed in the 'Statistical Analysis - "
        "STL Precipitation.py' script."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            default=str(settings.BASE_DIR / "fixtures" / "stl_precipitation_sample.csv"),
            help="Path to the CSV to load.",
        )
        parser.add_argument(
            "--name",
            default=self.DEFAULT_NAME,
            help="Dataset name shown in the UI.",
        )

    def handle(self, *args, **options):
        path = Path(options["path"])
        name = options["name"]

        if not path.exists():
            self.stderr.write(self.style.ERROR(f"Sample file not found: {path}"))
            return

        if Dataset.objects.filter(name=name).exists():
            self.stdout.write(
                self.style.WARNING(f"Dataset '{name}' already exists; skipping.")
            )
            return

        with path.open("rb") as fh:
            parsed = parse_input(fh)

        dataset = Dataset(
            name=name,
            description=self.DEFAULT_DESCRIPTION,
            indicator_columns=parsed.indicator_columns,
            row_count=int(len(parsed.raw)),
            min_year=float(parsed.raw["Event_Date"].min()),
            max_year=float(parsed.raw["Event_Date"].max()),
        )

        with path.open("rb") as fh:
            dataset.file.save(path.name, File(fh), save=True)

        self.stdout.write(
            self.style.SUCCESS(
                f"Loaded {dataset.row_count:,} rows; "
                f"indicators: {', '.join(parsed.indicator_columns)}; "
                f"range {dataset.min_year:.2f}–{dataset.max_year:.2f}."
            )
        )
