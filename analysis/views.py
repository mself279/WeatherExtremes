"""Class-based views for the WeatherExtremes UI."""
from __future__ import annotations

import io
import logging
import traceback
from dataclasses import asdict
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.core.files.base import ContentFile
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.views.generic import (
    CreateView,
    DetailView,
    FormView,
    ListView,
    TemplateView,
    View,
)

from .forms import (
    AnalysisRunForm,
    DatasetUploadForm,
    GEVRunForm,
    GhcnImportForm,
    RunComparisonForm,
)
from .models import (
    AnalysisRun,
    Dataset,
    GEVRun,
    GhcnElementAvailability,
    GhcnState,
    GhcnStation,
)
from .services import plots
from .services.gev.pipeline import run_gev as run_gev_pipeline
from .services.ghcn import elements as element_specs
from .services.ghcn import fetcher as ghcn_fetcher
from .services.ghcn import summarize as ghcn_summarize
from .services.ghcn.formats import read_station_csv
from .services.ghcn.transform import transform_to_indicators
from .services.io import parse_input
from .services.pipeline import run_analysis

logger = logging.getLogger(__name__)


class HomeView(TemplateView):
    template_name = "analysis/home.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["dataset_count"] = Dataset.objects.count()
        ctx["run_count"] = AnalysisRun.objects.count()
        ctx["recent_runs"] = AnalysisRun.objects.select_related("dataset")[:5]
        return ctx


class MethodologyView(TemplateView):
    """Static page describing the math, parameters, and outputs."""

    template_name = "analysis/methodology.html"


# ---------------------------------------------------------------------------
# GHCN browse + import
# ---------------------------------------------------------------------------


class GhcnBrowseView(ListView):
    """Filterable list of US GHCN stations with element availability."""

    model = GhcnStation
    template_name = "analysis/ghcn_browse.html"
    context_object_name = "stations"
    paginate_by = 50

    def _filters(self) -> dict:
        g = self.request.GET
        try:
            first_by = int(g.get("first_by")) if g.get("first_by") else None
        except (TypeError, ValueError):
            first_by = None
        try:
            last_after = int(g.get("last_after")) if g.get("last_after") else None
        except (TypeError, ValueError):
            last_after = None
        return {
            "state": g.get("state", "").strip(),
            "q": g.get("q", "").strip(),
            "elements": [e for e in g.getlist("element") if e],
            "network_code": g.get("network_code", "").strip(),
            "first_by": first_by,
            "last_after": last_after,
            "hcn_only": g.get("hcn_only") == "1",
        }

    def get_queryset(self):
        f = self._filters()
        qs = GhcnStation.objects.all()

        if f["state"]:
            qs = qs.filter(state=f["state"])
        if f["q"]:
            qs = qs.filter(
                Q(name__icontains=f["q"]) | Q(id__icontains=f["q"])
            )
        if f["network_code"]:
            qs = qs.filter(network_code=f["network_code"])
        if f["hcn_only"]:
            qs = qs.exclude(hcn_flag="")

        # Element/year filters: each requested element must be present at this
        # station and must satisfy the year bounds. Implemented as one EXISTS
        # subquery per requested element so the AND semantics are correct.
        for elem in f["elements"]:
            cond = Q(elements__element=elem)
            if f["first_by"] is not None:
                cond &= Q(elements__first_year__lte=f["first_by"])
            if f["last_after"] is not None:
                cond &= Q(elements__last_year__gte=f["last_after"])
            qs = qs.filter(cond)

        return qs.distinct().prefetch_related("elements")

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["filters"] = self._filters()
        ctx["states"] = GhcnState.objects.all()
        # Available elements with counts (helps users see what's worth picking).
        ctx["element_options"] = (
            GhcnElementAvailability.objects.values("element")
            .annotate(n=Count("station", distinct=True))
            .order_by("element")
        )
        ctx["element_specs"] = element_specs.ELEMENTS
        ctx["station_count"] = GhcnStation.objects.count()
        ctx["network_options"] = (
            GhcnStation.objects.values_list("network_code", flat=True)
            .distinct()
            .order_by("network_code")
        )
        return ctx


class GhcnConfigureView(View):
    """Station landing page: shows station info, datasets already derived from it,
    and a form to configure another import. POST creates a new Dataset."""

    template_name = "analysis/ghcn_configure.html"

    def _station(self, station_id: str) -> GhcnStation:
        return get_object_or_404(GhcnStation, pk=station_id)

    def _context(self, station: GhcnStation, form: GhcnImportForm) -> dict:
        existing = (
            station.datasets.all()
            .order_by("-created_at")
            .annotate(run_count=Count("runs"))
        )
        return {
            "form": form,
            "station": station,
            "elements": list(station.elements.order_by("element")),
            "specs": element_specs.ELEMENTS,
            "existing_datasets": existing,
        }

    def get(self, request, station_id):
        station = self._station(station_id)
        # Honor query params from the explore page so the form arrives pre-filled.
        initial = {}
        for key in ("element", "operator", "thresholds", "start_year", "end_year", "name"):
            if request.GET.get(key):
                initial[key] = request.GET[key]
        form = GhcnImportForm(initial=initial or None, station=station)
        return render(request, self.template_name, self._context(station, form))

    def post(self, request, station_id):
        station = self._station(station_id)
        form = GhcnImportForm(request.POST, station=station)
        if not form.is_valid():
            return render(request, self.template_name, self._context(station, form))

        cache_dir = Path(settings.MEDIA_ROOT) / "ghcn_cache"
        try:
            raw_path = ghcn_fetcher.fetch_station(
                station.id,
                cache_dir=cache_dir,
                force=form.cleaned_data.get("refresh", False),
            )
            raw_df = read_station_csv(raw_path)
            result = transform_to_indicators(
                raw_df,
                element=form.cleaned_data["element"],
                thresholds=form.cleaned_data["thresholds"],
                operator=form.cleaned_data["operator"],
                start_year=form.cleaned_data["start_year"],
                end_year=form.cleaned_data["end_year"],
                drop_quality_failed=bool(form.cleaned_data.get("drop_quality_failed")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("GHCN import failed", extra={"station": station.id})
            messages.error(request, f"GHCN import failed: {exc}")
            return render(request, self.template_name, self._context(station, form))

        # Persist the indicator-column CSV on the Dataset.
        out_buf = io.StringIO()
        result.df.to_csv(out_buf, index=False)
        csv_bytes = out_buf.getvalue().encode("utf-8")

        dataset = Dataset(
            name=form.cleaned_data["name"],
            description=form.cleaned_data.get("description") or "",
            indicator_columns=result.threshold_columns,
            row_count=int(len(result.df)),
            min_year=float(result.df["Event_Date"].min()),
            max_year=float(result.df["Event_Date"].max()),
            source_station=station,
            source_metadata=result.metadata,
        )
        fname = (
            f"{station.id}_{result.metadata['element']}_"
            f"{form.cleaned_data['start_year']}-{form.cleaned_data['end_year']}.csv"
        )
        dataset.file.save(fname, ContentFile(csv_bytes), save=True)

        messages.success(
            request,
            f"Imported {station.name.title()} – {result.metadata['element']}: "
            f"{result.metadata['observations_with_value']:,} observations, "
            f"{sum(result.metadata['events_per_threshold'].values()):,} threshold events.",
        )
        return redirect(dataset.get_absolute_url())


class DatasetListView(ListView):
    model = Dataset
    template_name = "analysis/dataset_list.html"
    context_object_name = "datasets"


class DatasetUploadView(CreateView):
    model = Dataset
    form_class = DatasetUploadForm
    template_name = "analysis/dataset_upload.html"
    success_url = reverse_lazy("analysis:dataset_list")

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(
            self.request,
            f"Dataset '{self.object.name}' uploaded "
            f"({self.object.row_count:,} rows; "
            f"columns: {', '.join(self.object.indicator_columns)}).",
        )
        return response


class DatasetDetailView(DetailView):
    model = Dataset
    template_name = "analysis/dataset_detail.html"
    context_object_name = "dataset"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["runs"] = self.object.runs.all()
        ctx["gev_runs"] = self.object.gev_runs.all()
        return ctx


class AnalysisRunCreateView(View):
    """Render the form on GET; execute the analysis on POST."""

    template_name = "analysis/run_create.html"

    def _get_dataset(self, pk):
        return Dataset.objects.get(pk=pk)

    def get(self, request, pk):
        dataset = self._get_dataset(pk)
        form = AnalysisRunForm(dataset=dataset)
        return render(
            request, self.template_name, {"form": form, "dataset": dataset}
        )

    def post(self, request, pk):
        dataset = self._get_dataset(pk)
        form = AnalysisRunForm(request.POST, dataset=dataset)
        if not form.is_valid():
            return render(
                request, self.template_name, {"form": form, "dataset": dataset}
            )

        params = form.to_parameters()
        run = AnalysisRun.objects.create(
            dataset=dataset,
            label=form.cleaned_data.get("label", ""),
            parameters=asdict(params),
            status=AnalysisRun.Status.RUNNING,
        )
        try:
            with dataset.file.open("rb") as fh:
                parsed = parse_input(fh)
            result = run_analysis(parsed.raw, params)
        except Exception as exc:
            run.status = AnalysisRun.Status.FAILED
            run.error = f"{exc}\n\n{traceback.format_exc()}"
            run.completed_at = timezone.now()
            run.save()
            logger.exception("Analysis run failed", extra={"run_id": str(run.id)})
            messages.error(request, f"Analysis failed: {exc}")
            return redirect("analysis:run_detail", pk=run.id)

        run.summary = result.summary
        run.figures = result.figures
        run.series = result.series
        run.status = AnalysisRun.Status.SUCCESS
        run.completed_at = timezone.now()
        run.save()

        messages.success(request, "Analysis complete.")
        return redirect("analysis:run_detail", pk=run.id)


class AnalysisRunListView(ListView):
    """Runs index — shows both kernel-rate runs and GEV runs in two sections."""

    model = AnalysisRun
    template_name = "analysis/run_list.html"
    context_object_name = "runs"
    paginate_by = 50

    def get_queryset(self):
        return AnalysisRun.objects.select_related("dataset").all()

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["gev_runs"] = (
            GEVRun.objects.select_related("dataset", "dataset__source_station").all()
        )
        return ctx


class AnalysisRunDetailView(DetailView):
    model = AnalysisRun
    template_name = "analysis/run_detail.html"
    context_object_name = "run"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["figures"] = self.object.figures or {}
        return ctx


class RunComparisonView(FormView):
    template_name = "analysis/run_compare.html"
    form_class = RunComparisonForm

    def get_initial(self):
        ids = self.request.GET.getlist("run")
        if ids:
            return {"runs": ids}
        return super().get_initial()

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        runs_qs = ctx["form"].fields["runs"].queryset.select_related("dataset")
        ctx["all_runs"] = runs_qs
        ctx["figure_json"] = None

        # Render the comparison if the user has selected runs (via GET or POST)
        ids = self.request.GET.getlist("run") or self.request.POST.getlist("runs")
        if ids:
            selected = list(runs_qs.filter(pk__in=ids))
            if len(selected) >= 2:
                series = []
                for run in selected:
                    grid = run.series.get("grid") or []
                    estimate = run.series.get("point_estimate") or []
                    if grid and estimate:
                        series.append((run.display_label, grid, estimate))
                if series:
                    ctx["figure_json"] = plots.comparison_figure(
                        series=series,
                        title="Smoothed occurrence rate – run comparison",
                    )
                ctx["selected_runs"] = selected
        return ctx

    def form_valid(self, form):
        runs = form.cleaned_data["runs"]
        url = reverse("analysis:run_compare")
        params = "&".join(f"run={r.pk}" for r in runs)
        return redirect(f"{url}?{params}")


# ---------------------------------------------------------------------------
# GHCN data exploration
# ---------------------------------------------------------------------------


class GhcnExploreView(View):
    """Pre-analysis exploration: completeness, distribution, annual extremes,
    threshold-crossing frequency. No Dataset is created — this informs the
    user's threshold/range choices before they hit the configure page."""

    template_name = "analysis/ghcn_explore.html"

    def _station(self, station_id: str) -> GhcnStation:
        return get_object_or_404(GhcnStation, pk=station_id)

    def _parse_thresholds(self, raw: str) -> list[float] | None:
        if not raw:
            return None
        try:
            return [float(t.strip()) for t in raw.split(",") if t.strip()]
        except ValueError:
            return None

    def get(self, request, station_id):
        station = self._station(station_id)
        availability = list(station.elements.order_by("element"))
        valid_codes = [a.element for a in availability if a.element in element_specs.ELEMENTS]

        # Pick element from query, defaulting to PRCP if available.
        requested = (request.GET.get("element") or "").strip()
        if requested in valid_codes:
            element = requested
        elif "PRCP" in valid_codes:
            element = "PRCP"
        elif valid_codes:
            element = valid_codes[0]
        else:
            element = None

        spec = element_specs.ELEMENTS.get(element) if element else None

        # Operator + thresholds (defaulting to per-element specs).
        operator = (request.GET.get("operator") or "").strip()
        thresholds_str = (request.GET.get("thresholds") or "").strip()

        if spec is not None:
            if not operator:
                operator = spec.default_operator
            user_thresholds = self._parse_thresholds(thresholds_str)
            if user_thresholds:
                thresholds = user_thresholds
            else:
                thresholds = list(spec.default_thresholds)
                thresholds_str = ", ".join(
                    str(int(t) if t == int(t) else t) for t in thresholds
                )
        else:
            thresholds = []

        # Generate summaries + figures.
        figures: dict[str, str] = {}
        summary_blob: dict[str, dict] = {}
        load_error: str | None = None

        if element:
            try:
                cache_dir = Path(settings.MEDIA_ROOT) / "ghcn_cache"
                raw_path = ghcn_fetcher.fetch_station(
                    station.id, cache_dir=cache_dir
                )
                raw_df = read_station_csv(raw_path)

                comp = ghcn_summarize.data_completeness(raw_df, element)
                dist = ghcn_summarize.value_distribution(
                    raw_df, element, thresholds=thresholds, operator=operator
                )
                ann = ghcn_summarize.annual_extremes(raw_df, element)
                freq = ghcn_summarize.threshold_frequencies(
                    raw_df, element, thresholds=thresholds, operator=operator
                )

                figures = {
                    "completeness": ghcn_summarize.completeness_figure(comp),
                    "distribution": ghcn_summarize.distribution_figure(dist, operator),
                    "annual": ghcn_summarize.annual_extremes_figure(ann),
                    "frequency": ghcn_summarize.threshold_frequency_figure(freq),
                }
                summary_blob = {
                    "completeness": {
                        "first_year": comp.years[0] if comp.years else None,
                        "last_year": comp.years[-1] if comp.years else None,
                        "total_obs": int(sum(comp.counts)) if comp.counts else 0,
                        "years_at_or_above_95pct": sum(1 for p in comp.pct if p >= 95),
                        "years_below_70pct": sum(1 for p in comp.pct if p < 70),
                    },
                    "distribution": {
                        "n": dist.n,
                        "quantiles": dist.quantiles,
                        "threshold_counts": dist.threshold_counts,
                        "display_unit": dist.spec.display_unit,
                    },
                    "annual": {
                        "direction": ann.direction,
                        "trend_slope": ann.trend_slope,
                        "trend_intercept": ann.trend_intercept,
                        "n_years": len(ann.years),
                        "display_unit": ann.spec.display_unit,
                    },
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("GHCN explore failed", extra={"station": station.id})
                load_error = str(exc)

        # Build query string for "Use these in Configure" link.
        configure_params: dict[str, str] = {}
        if element:
            configure_params["element"] = element
            configure_params["operator"] = operator
            configure_params["thresholds"] = ", ".join(
                str(int(t) if t == int(t) else t) for t in thresholds
            )
        configure_query = "&".join(f"{k}={v}" for k, v in configure_params.items())

        return render(request, self.template_name, {
            "station": station,
            "elements": availability,
            "valid_codes": valid_codes,
            "element": element,
            "operator": operator,
            "thresholds_str": thresholds_str,
            "spec": spec,
            "specs": element_specs.ELEMENTS,
            "operator_choices": element_specs.OPERATOR_CHOICES,
            "figures": figures,
            "summary_blob": summary_blob,
            "configure_query": configure_query,
            "load_error": load_error,
        })


# ---------------------------------------------------------------------------
# GEV runs (Generalized Extreme Value analysis)
# ---------------------------------------------------------------------------


class GEVRunCreateView(View):
    """GET: render form prefilled from the dataset's source. POST: run pipeline."""

    template_name = "analysis/gev_create.html"

    def _dataset(self, pk):
        return get_object_or_404(Dataset, pk=pk)

    def _check_eligible(self, dataset: Dataset) -> str | None:
        if dataset.source_station is None:
            return (
                "GEV analysis currently requires a GHCN-derived dataset "
                "(needs the raw daily values). Import a station via "
                "'Import from GHCN' first."
            )
        return None

    def get(self, request, pk):
        dataset = self._dataset(pk)
        ineligible = self._check_eligible(dataset)
        if ineligible:
            messages.warning(request, ineligible)
            return redirect(dataset.get_absolute_url())
        form = GEVRunForm(dataset=dataset)
        return render(
            request, self.template_name, {"form": form, "dataset": dataset}
        )

    def post(self, request, pk):
        dataset = self._dataset(pk)
        ineligible = self._check_eligible(dataset)
        if ineligible:
            messages.warning(request, ineligible)
            return redirect(dataset.get_absolute_url())

        form = GEVRunForm(request.POST, dataset=dataset)
        if not form.is_valid():
            return render(
                request, self.template_name, {"form": form, "dataset": dataset}
            )

        params = form.to_parameters()
        run = GEVRun.objects.create(
            dataset=dataset,
            label=form.cleaned_data.get("label", ""),
            parameters=asdict(params),
            status=GEVRun.Status.RUNNING,
        )
        try:
            cache_dir = Path(settings.MEDIA_ROOT) / "ghcn_cache"
            result = run_gev_pipeline(
                station_id=dataset.source_station.id,
                cache_dir=cache_dir,
                params=params,
            )
        except Exception as exc:  # noqa: BLE001
            run.status = GEVRun.Status.FAILED
            run.error = f"{exc}\n\n{traceback.format_exc()}"
            run.completed_at = timezone.now()
            run.save()
            logger.exception("GEV run failed", extra={"run_id": str(run.id)})
            messages.error(request, f"GEV analysis failed: {exc}")
            return redirect("analysis:gev_detail", pk=run.id)

        run.summary = result.summary
        run.figures = result.figures
        run.series = result.series
        run.confidence_intervals = result.confidence_intervals
        run.status = GEVRun.Status.SUCCESS
        run.completed_at = timezone.now()
        run.save()

        messages.success(request, "GEV fit complete.")
        return redirect("analysis:gev_detail", pk=run.id)


class GEVRunDetailView(DetailView):
    model = GEVRun
    template_name = "analysis/gev_detail.html"
    context_object_name = "run"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["figures"] = self.object.figures or {}
        return ctx
