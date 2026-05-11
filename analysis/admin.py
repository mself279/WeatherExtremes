from django.contrib import admin

from .models import (
    AnalysisRun,
    Dataset,
    GEVRun,
    GhcnElementAvailability,
    GhcnState,
    GhcnStation,
)


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "row_count", "min_year", "max_year", "source_station", "created_at")
    search_fields = ("name", "description", "source_station__id")
    readonly_fields = (
        "indicator_columns",
        "row_count",
        "min_year",
        "max_year",
        "source_station",
        "source_metadata",
        "created_at",
    )


@admin.register(AnalysisRun)
class AnalysisRunAdmin(admin.ModelAdmin):
    list_display = ("display_label", "dataset", "status", "created_at", "completed_at")
    list_filter = ("status", "dataset")
    search_fields = ("label", "dataset__name")
    readonly_fields = (
        "dataset",
        "parameters",
        "summary",
        "figures",
        "series",
        "status",
        "error",
        "created_at",
        "completed_at",
    )


@admin.register(GhcnState)
class GhcnStateAdmin(admin.ModelAdmin):
    list_display = ("code", "name")
    search_fields = ("code", "name")


@admin.register(GhcnStation)
class GhcnStationAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "state", "network_code", "latitude", "longitude", "elevation")
    list_filter = ("state", "network_code", "hcn_flag", "gsn_flag")
    search_fields = ("id", "name")


@admin.register(GEVRun)
class GEVRunAdmin(admin.ModelAdmin):
    list_display = ("display_label", "dataset", "status", "created_at", "completed_at")
    list_filter = ("status", "dataset")
    search_fields = ("label", "dataset__name")
    readonly_fields = (
        "dataset",
        "parameters",
        "summary",
        "figures",
        "series",
        "confidence_intervals",
        "status",
        "error",
        "created_at",
        "completed_at",
    )


@admin.register(GhcnElementAvailability)
class GhcnElementAvailabilityAdmin(admin.ModelAdmin):
    list_display = ("station", "element", "first_year", "last_year")
    list_filter = ("element",)
    search_fields = ("station__id", "station__name")
