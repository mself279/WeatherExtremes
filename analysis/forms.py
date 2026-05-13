"""Forms for dataset upload and analysis configuration."""
from __future__ import annotations

from typing import Any

from django import forms
from django.conf import settings

from .models import AnalysisRun, Dataset, GhcnStation
from .services.ghcn.elements import (
    ELEMENTS,
    OPERATOR_CHOICES,
)
from .services.io import parse_input


class DatasetUploadForm(forms.ModelForm):
    """Validates the CSV at upload time and captures schema metadata."""

    class Meta:
        model = Dataset
        fields = ("name", "description", "file")
        widgets = {
            "description": forms.Textarea(attrs={"rows": 3}),
        }

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        upload = cleaned.get("file")
        if not upload:
            return cleaned
        upload.seek(0)
        try:
            parsed = parse_input(upload)
        except Exception as exc:
            raise forms.ValidationError(f"Could not parse uploaded file: {exc}")
        upload.seek(0)
        if not parsed.indicator_columns:
            raise forms.ValidationError(
                "No 0/1 indicator columns were detected in the file. "
                "Add at least one column like 'Over3' with 0/1 values."
            )
        # Stash for the view to write into the model after save.
        self._parsed_metadata = {
            "indicator_columns": parsed.indicator_columns,
            "row_count": int(len(parsed.raw)),
            "min_year": float(parsed.raw["Event_Date"].min()),
            "max_year": float(parsed.raw["Event_Date"].max()),
        }
        return cleaned

    def save(self, commit: bool = True) -> Dataset:  # type: ignore[override]
        instance = super().save(commit=False)
        meta = getattr(self, "_parsed_metadata", None)
        if meta:
            instance.indicator_columns = meta["indicator_columns"]
            instance.row_count = meta["row_count"]
            instance.min_year = meta["min_year"]
            instance.max_year = meta["max_year"]
        if commit:
            instance.save()
        return instance


class AnalysisRunForm(forms.Form):
    """Configure a single analysis run against a Dataset."""

    label = forms.CharField(
        max_length=200,
        required=False,
        help_text="Optional label to identify this run.",
    )
    indicator_column = forms.ChoiceField(
        help_text="Which 0/1 column marks the events you want to analyze.",
    )
    additional_indicators = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        help_text="Add other thresholds to overlay on the chart for comparison.",
    )
    observation_start = forms.FloatField()
    observation_end = forms.FloatField()
    observation_factor = forms.FloatField(
        initial=1.0 / 365.0,
        help_text="Spacing between time-grid points in years (1/365 = daily).",
    )
    bandwidth_years = forms.FloatField(
        initial=20.0,
        min_value=0.5,
        help_text="Smoothing bandwidth h, in years.",
    )
    bias_correction = forms.BooleanField(
        required=False,
        initial=True,
        help_text="Reflect 3h of data outside the window to reduce edge bias.",
    )
    bootstrap_iterations = forms.IntegerField(
        initial=settings.ANALYSIS_DEFAULT_BOOTSTRAP_ITER,
        min_value=10,
        max_value=settings.ANALYSIS_MAX_BOOTSTRAP_ITER,
        help_text="Number of bootstrap resamples for the confidence band.",
    )
    bandwidth_search_min = forms.FloatField(initial=3.0, min_value=0.5)
    bandwidth_search_max = forms.FloatField(initial=30.0, min_value=1.0)
    bandwidth_search_step = forms.FloatField(initial=3.0, min_value=0.1)
    decluster_tau = forms.IntegerField(
        initial=0,
        min_value=0,
        help_text="Tau (in observation steps) for declustering. 0 disables it.",
    )
    rng_seed = forms.IntegerField(
        initial=12345,
        required=False,
        help_text="Optional seed for reproducible bootstrap results.",
    )

    def __init__(self, *args: Any, dataset: Dataset, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        choices = [(c, c) for c in dataset.indicator_columns]
        self.fields["indicator_column"].choices = choices
        self.fields["additional_indicators"].choices = choices
        if dataset.min_year is not None:
            self.fields["observation_start"].initial = float(int(dataset.min_year))
        if dataset.max_year is not None:
            self.fields["observation_end"].initial = float(
                int(dataset.max_year) + 1
            )

    def clean(self) -> dict[str, Any]:
        cleaned = super().clean()
        start = cleaned.get("observation_start")
        end = cleaned.get("observation_end")
        if start is not None and end is not None and end <= start:
            raise forms.ValidationError(
                "observation_end must be greater than observation_start."
            )
        bw_min = cleaned.get("bandwidth_search_min")
        bw_max = cleaned.get("bandwidth_search_max")
        if bw_min is not None and bw_max is not None and bw_max <= bw_min:
            raise forms.ValidationError(
                "bandwidth_search_max must be greater than bandwidth_search_min."
            )
        return cleaned

    def to_parameters(self):  # noqa: ANN201 - returns AnalysisParameters
        from .services.pipeline import AnalysisParameters

        d = self.cleaned_data
        return AnalysisParameters(
            indicator_column=d["indicator_column"],
            observation_start=float(d["observation_start"]),
            observation_end=float(d["observation_end"]),
            observation_factor=float(d["observation_factor"]),
            bandwidth_years=float(d["bandwidth_years"]),
            bias_correction=bool(d.get("bias_correction")),
            bootstrap_iterations=int(d["bootstrap_iterations"]),
            bandwidth_search_min=float(d["bandwidth_search_min"]),
            bandwidth_search_max=float(d["bandwidth_search_max"]),
            bandwidth_search_step=float(d["bandwidth_search_step"]),
            additional_indicators=tuple(d.get("additional_indicators") or ()),
            decluster_tau=int(d["decluster_tau"]),
            rng_seed=d.get("rng_seed"),
        )


class GhcnImportForm(forms.Form):
    """Configure conversion of a GHCN station file into a Dataset."""

    name = forms.CharField(max_length=200)
    description = forms.CharField(
        required=False, widget=forms.Textarea(attrs={"rows": 3})
    )
    element = forms.ChoiceField()
    operator = forms.ChoiceField(choices=OPERATOR_CHOICES)
    thresholds = forms.CharField(
        help_text=(
            "Comma-separated threshold values in the element's display unit "
            "(e.g. inches for PRCP, °F for TMAX/TMIN, mph for wind)."
        )
    )
    start_year = forms.IntegerField()
    end_year = forms.IntegerField()
    drop_quality_failed = forms.BooleanField(
        required=False,
        initial=True,
        help_text="Drop observations whose Q_FLAG indicates a failed quality check.",
    )
    refresh = forms.BooleanField(
        required=False,
        initial=False,
        help_text="Re-download the station file even if a recent copy is cached.",
    )

    def __init__(self, *args: Any, station: GhcnStation, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.station = station

        availability = list(station.elements.order_by("element"))
        # Build choices, preferring elements we know how to interpret.
        labeled: list[tuple[str, str]] = []
        for a in availability:
            spec = ELEMENTS.get(a.element)
            if spec is not None:
                labeled.append(
                    (
                        a.element,
                        f"{a.element} – {spec.label} "
                        f"({a.first_year}–{a.last_year}, {spec.display_unit})",
                    )
                )
            else:
                labeled.append(
                    (a.element, f"{a.element} (raw, {a.first_year}–{a.last_year})")
                )
        self.fields["element"].choices = labeled

        # If the caller passed initial={"element": ...}, honor it as the basis
        # for per-element defaults (used when redirecting from explore page).
        initial_data = self.initial or {}
        requested_elem = initial_data.get("element")

        default_elem = None
        if requested_elem and any(a.element == requested_elem for a in availability):
            default_elem = requested_elem
        if default_elem is None:
            default_elem = next(
                (a.element for a in availability if a.element == "PRCP"),
                availability[0].element if availability else None,
            )

        if default_elem and default_elem in ELEMENTS:
            spec = ELEMENTS[default_elem]
            avail = next(a for a in availability if a.element == default_elem)
            # Only set field-level initial if NOT already overridden via the
            # caller's initial dict.
            if "element" not in initial_data:
                self.fields["element"].initial = default_elem
            if "operator" not in initial_data:
                self.fields["operator"].initial = spec.default_operator
            if "thresholds" not in initial_data:
                self.fields["thresholds"].initial = ", ".join(
                    str(int(t) if t == int(t) else t) for t in spec.default_thresholds
                )
            if "start_year" not in initial_data:
                self.fields["start_year"].initial = avail.first_year
            if "end_year" not in initial_data:
                self.fields["end_year"].initial = avail.last_year
            if "name" not in initial_data:
                self.fields["name"].initial = (
                    f"{station.name.title()} – {default_elem}"
                )

    def clean_thresholds(self):
        raw = (self.cleaned_data.get("thresholds") or "").strip()
        if not raw:
            raise forms.ValidationError("Provide at least one threshold value.")
        try:
            values = [float(t.strip()) for t in raw.split(",") if t.strip()]
        except ValueError as exc:
            raise forms.ValidationError(
                f"Could not parse thresholds: {exc}. Use comma-separated numbers."
            )
        if not values:
            raise forms.ValidationError("Provide at least one threshold value.")
        return values

    def clean(self):
        cleaned = super().clean()
        sy = cleaned.get("start_year")
        ey = cleaned.get("end_year")
        if sy is not None and ey is not None and ey < sy:
            raise forms.ValidationError("end_year must be >= start_year.")
        elem = cleaned.get("element")
        if elem and elem not in ELEMENTS:
            raise forms.ValidationError(
                f"Element {elem!r} is not currently supported by the import "
                "form. (Supported: PRCP, TMAX, TMIN, TAVG, AWND, WSF2, WSF5, "
                "SNOW, SNWD.)"
            )
        return cleaned


class StormEventsBrowseForm(forms.Form):
    """Filter inputs for an NCEI Storm Events query."""

    from .services.storm_events.event_types import event_type_choices
    from .services.storm_events.states import state_choices

    statefips = forms.ChoiceField(
        choices=state_choices(),
        label="State",
        initial="29,MISSOURI",
    )
    event_type = forms.ChoiceField(
        choices=event_type_choices(),
        label="Event type",
        initial="tornado",
    )
    begin_year = forms.IntegerField(
        label="Start year", min_value=1950, max_value=2100, initial=1950,
    )
    end_year = forms.IntegerField(
        label="End year", min_value=1950, max_value=2100, initial=2024,
    )
    tornfilter = forms.ChoiceField(
        choices=(("0", "All"), ("1", "F1/EF1+"), ("2", "F2/EF2+"),
                 ("3", "F3/EF3+"), ("4", "F4/EF4+"), ("5", "F5/EF5+")),
        label="Tornado scale (only applies to tornadoes)",
        initial="3",
        required=False,
    )
    hailfilter = forms.ChoiceField(
        choices=(("0.00", "All"), ("0.75", "0.75 in+"), ("1.00", "1.00 in+"),
                 ("1.50", "1.50 in+"), ("2.00", "2.00 in+"), ("3.00", "3.00 in+")),
        label="Hail diameter (only applies to hail)",
        initial="0.00",
        required=False,
    )
    windfilter = forms.ChoiceField(
        choices=(("000", "All"), ("050", "50 kt+"), ("065", "65 kt+"),
                 ("075", "75 kt+"), ("100", "100 kt+")),
        label="Wind speed (only applies to wind events)",
        initial="000",
        required=False,
    )

    def clean(self):
        cleaned = super().clean()
        by = cleaned.get("begin_year")
        ey = cleaned.get("end_year")
        if by is not None and ey is not None and ey < by:
            raise forms.ValidationError("End year must be >= start year.")
        return cleaned


class GEVRunForm(forms.Form):
    """Configure a GEV analysis on a GHCN-derived Dataset."""

    DIRECTION_CHOICES = (("max", "Annual maximum"), ("min", "Annual minimum"))
    BOOTSTRAP_CHOICES = (
        ("both", "Parametric + Nonparametric"),
        ("parametric", "Parametric only"),
        ("nonparametric", "Nonparametric only"),
        ("none", "Skip confidence intervals (point estimate only)"),
    )

    label = forms.CharField(
        max_length=200, required=False,
        help_text="Optional label to identify this run.",
    )
    element = forms.ChoiceField(
        help_text="Which element to use; for v1 the dataset's source element is suggested.",
    )
    direction = forms.ChoiceField(
        choices=DIRECTION_CHOICES,
        initial="max",
        help_text="Block extreme — max for PRCP/TMAX/wind, min for TMIN.",
    )
    start_year = forms.IntegerField(
        required=False,
        help_text="First year to include (blank = use all available).",
    )
    end_year = forms.IntegerField(
        required=False,
        help_text="Last year to include (blank = use all available).",
    )
    min_obs_per_year = forms.IntegerField(
        initial=200, min_value=1, max_value=366,
        help_text="Skip years with fewer than this many observations (incomplete years skew block extremes).",
    )
    drop_quality_failed = forms.BooleanField(
        required=False, initial=True,
        help_text="Drop observations whose Q_FLAG indicates a failed quality check.",
    )
    bootstrap_method = forms.ChoiceField(
        choices=BOOTSTRAP_CHOICES, initial="both",
    )
    bootstrap_iterations = forms.IntegerField(
        initial=500, min_value=50, max_value=5000,
        help_text="Number of bootstrap refits per method.",
    )
    bucket_width = forms.FloatField(
        initial=0.5, min_value=0.01,
        help_text="Bucket width (display units) for the observed-vs-expected diagnostic.",
    )
    reference_year = forms.IntegerField(
        required=False,
        help_text="Year for the exceedance probability (defaults to last year of record).",
    )
    reference_value = forms.FloatField(
        required=False,
        help_text="Reference value (display units) for exceedance. Blank = that year's max.",
    )
    rng_seed = forms.IntegerField(
        initial=12345, required=False,
        help_text="Optional RNG seed for reproducible bootstraps.",
    )

    def __init__(self, *args: Any, dataset: Dataset, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dataset = dataset

        # Element choices: from the source station's available elements,
        # restricted to those we know how to interpret.
        if dataset.source_station is not None:
            specs = ELEMENTS
            avail = list(dataset.source_station.elements.order_by("element"))
            valid = [a for a in avail if a.element in specs]
            self.fields["element"].choices = [
                (a.element, f"{a.element} – {specs[a.element].label}") for a in valid
            ]
            # Source element from the dataset, if present.
            src_elem = (dataset.source_metadata or {}).get("element")
            if src_elem and any(a.element == src_elem for a in valid):
                self.fields["element"].initial = src_elem
                spec = specs[src_elem]
                self.fields["direction"].initial = (
                    "max" if spec.default_operator in (">", ">=") else "min"
                )
                self.fields["bucket_width"].initial = spec.default_bucket_width
                self.fields["label"].initial = (
                    f"{dataset.source_station.name.title()} – {src_elem} GEV"
                )
            elif valid:
                first = valid[0].element
                self.fields["element"].initial = first
                self.fields["bucket_width"].initial = specs[first].default_bucket_width

    def clean(self):
        cleaned = super().clean()
        sy = cleaned.get("start_year")
        ey = cleaned.get("end_year")
        if sy is not None and ey is not None and ey < sy:
            raise forms.ValidationError("end_year must be >= start_year.")
        return cleaned

    def to_parameters(self):  # noqa: ANN201 - returns GEVParameters
        from .services.gev.pipeline import GEVParameters
        d = self.cleaned_data
        return GEVParameters(
            element=d["element"],
            direction=d["direction"],
            start_year=d.get("start_year"),
            end_year=d.get("end_year"),
            drop_quality_failed=bool(d.get("drop_quality_failed")),
            min_obs_per_year=int(d["min_obs_per_year"]),
            bootstrap_method=d["bootstrap_method"],
            bootstrap_iterations=int(d["bootstrap_iterations"]),
            reference_year=d.get("reference_year"),
            reference_value=d.get("reference_value"),
            bucket_width=float(d["bucket_width"]),
            rng_seed=d.get("rng_seed"),
        )


class RunComparisonForm(forms.Form):
    """Pick 2-N runs to overlay on the comparison chart."""

    runs = forms.ModelMultipleChoiceField(
        queryset=AnalysisRun.objects.filter(status=AnalysisRun.Status.SUCCESS),
        widget=forms.CheckboxSelectMultiple,
    )

    def clean_runs(self):
        runs = self.cleaned_data["runs"]
        if runs.count() < 2:
            raise forms.ValidationError("Pick at least two runs to compare.")
        return runs
