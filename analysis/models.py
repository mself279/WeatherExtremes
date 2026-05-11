"""Persistence layer for datasets and analysis runs."""
from __future__ import annotations

import uuid
from pathlib import Path

from django.db import models
from django.urls import reverse
from django.utils import timezone


def dataset_upload_path(instance: "Dataset", filename: str) -> str:
    name = Path(filename).name
    return f"datasets/{instance.id or uuid.uuid4()}/{name}"


class Dataset(models.Model):
    """An uploaded or imported CSV of fractional-year event indicators."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to=dataset_upload_path)
    indicator_columns = models.JSONField(
        default=list,
        help_text="0/1 indicator columns detected in the file (e.g. ['Over3', 'Over4']).",
    )
    row_count = models.PositiveIntegerField(default=0)
    min_year = models.FloatField(null=True, blank=True)
    max_year = models.FloatField(null=True, blank=True)
    # Provenance fields populated when the dataset was derived from an
    # external source (currently GHCN; later USGS, NOAA CO-OPS, SPC, etc.).
    source_station = models.ForeignKey(
        "GhcnStation",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="datasets",
    )
    source_metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Element, thresholds, operator, raw range, etc. for derived datasets.",
    )
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("analysis:dataset_detail", args=[str(self.id)])


class AnalysisRun(models.Model):
    """One execution of the pipeline against a Dataset with a given config."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="runs",
    )
    label = models.CharField(
        max_length=200,
        blank=True,
        help_text="Optional label to distinguish runs in the comparison view.",
    )
    parameters = models.JSONField()
    summary = models.JSONField(default=dict, blank=True)
    figures = models.JSONField(default=dict, blank=True)
    series = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.PENDING,
    )
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self) -> str:
        return self.label or f"Run {self.id.hex[:8]}"

    def get_absolute_url(self) -> str:
        return reverse("analysis:run_detail", args=[str(self.id)])

    @property
    def is_finished(self) -> bool:
        return self.status in (self.Status.SUCCESS, self.Status.FAILED)

    @property
    def display_label(self) -> str:
        return self.label or f"{self.dataset.name} – {self.created_at:%Y-%m-%d %H:%M}"


# ---------------------------------------------------------------------------
# GHCN catalog (populated by the `sync_ghcn` management command)
# ---------------------------------------------------------------------------


class GhcnState(models.Model):
    """U.S. state / Canadian province lookup from ghcnd-states.txt."""

    code = models.CharField(max_length=2, primary_key=True)
    name = models.CharField(max_length=64)

    class Meta:
        ordering = ("name",)
        verbose_name = "GHCN state/province"
        verbose_name_plural = "GHCN states/provinces"

    def __str__(self) -> str:
        return f"{self.code} – {self.name}"


class GhcnStation(models.Model):
    """A station from ghcnd-stations.txt (limited to US stations on sync)."""

    id = models.CharField(max_length=11, primary_key=True)
    name = models.CharField(max_length=64)
    state = models.CharField(max_length=2, blank=True, db_index=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    elevation = models.FloatField(null=True, blank=True)  # meters
    network_code = models.CharField(
        max_length=1,
        db_index=True,
        help_text="3rd character of the ID: W=WBAN, C=Cooperative, 1=CoCoRaHS, etc.",
    )
    gsn_flag = models.CharField(max_length=3, blank=True)
    hcn_flag = models.CharField(max_length=3, blank=True)
    wmo_id = models.CharField(max_length=5, blank=True)

    class Meta:
        ordering = ("state", "name")
        indexes = [
            models.Index(fields=("state", "name")),
        ]
        verbose_name = "GHCN station"

    def __str__(self) -> str:
        return f"{self.id} – {self.name}"

    def get_absolute_url(self) -> str:
        return reverse("analysis:ghcn_configure", args=[self.id])


class GEVRun(models.Model):
    """One execution of the GEV pipeline against a GHCN-derived Dataset.

    Structurally a sibling of :class:`AnalysisRun` (separate model so the
    parameters and outputs can evolve independently): different inputs
    (annual maxima vs event indicators), different outputs (parameter
    estimates and return periods vs rate curves).
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="gev_runs",
    )
    label = models.CharField(max_length=200, blank=True)
    parameters = models.JSONField()
    summary = models.JSONField(default=dict, blank=True)
    figures = models.JSONField(default=dict, blank=True)
    series = models.JSONField(default=dict, blank=True)
    confidence_intervals = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.PENDING,
    )
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-created_at",)
        verbose_name = "GEV run"

    def __str__(self) -> str:
        return self.label or f"GEV run {self.id.hex[:8]}"

    def get_absolute_url(self) -> str:
        return reverse("analysis:gev_detail", args=[str(self.id)])

    @property
    def display_label(self) -> str:
        return self.label or f"{self.dataset.name} – GEV {self.created_at:%Y-%m-%d %H:%M}"


class GhcnElementAvailability(models.Model):
    """Per-station element coverage (FIRSTYEAR/LASTYEAR) from ghcnd-inventory.txt."""

    station = models.ForeignKey(
        GhcnStation, on_delete=models.CASCADE, related_name="elements"
    )
    element = models.CharField(max_length=4, db_index=True)
    first_year = models.IntegerField()
    last_year = models.IntegerField()

    class Meta:
        unique_together = ("station", "element")
        ordering = ("station", "element")
        indexes = [
            models.Index(fields=("element", "first_year", "last_year")),
            models.Index(fields=("element", "last_year")),
        ]
        verbose_name = "GHCN element availability"
        verbose_name_plural = "GHCN element availabilities"

    def __str__(self) -> str:
        return f"{self.station_id} {self.element} {self.first_year}–{self.last_year}"
