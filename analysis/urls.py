"""URL config for the analysis app."""
from django.urls import path

from . import views

app_name = "analysis"

urlpatterns = [
    path("", views.HomeView.as_view(), name="home"),
    path("datasets/", views.DatasetListView.as_view(), name="dataset_list"),
    path("datasets/upload/", views.DatasetUploadView.as_view(), name="dataset_upload"),
    path(
        "datasets/from-ghcn/",
        views.GhcnBrowseView.as_view(),
        name="ghcn_browse",
    ),
    path(
        "datasets/from-ghcn/<str:station_id>/",
        views.GhcnConfigureView.as_view(),
        name="ghcn_configure",
    ),
    path(
        "datasets/from-ghcn/<str:station_id>/explore/",
        views.GhcnExploreView.as_view(),
        name="ghcn_explore",
    ),
    path(
        "datasets/<uuid:pk>/",
        views.DatasetDetailView.as_view(),
        name="dataset_detail",
    ),
    path(
        "datasets/<uuid:pk>/delete/",
        views.DatasetDeleteView.as_view(),
        name="dataset_delete",
    ),
    path(
        "datasets/<uuid:pk>/analyze/",
        views.AnalysisRunCreateView.as_view(),
        name="run_create",
    ),
    path("runs/", views.AnalysisRunListView.as_view(), name="run_list"),
    path("runs/<uuid:pk>/", views.AnalysisRunDetailView.as_view(), name="run_detail"),
    path(
        "runs/<uuid:pk>/delete/",
        views.AnalysisRunDeleteView.as_view(),
        name="run_delete",
    ),
    path("runs/compare/", views.RunComparisonView.as_view(), name="run_compare"),
    path("methodology/", views.MethodologyView.as_view(), name="methodology"),
    path("use-cases/", views.UseCasesView.as_view(), name="use_cases"),
    path(
        "storm-events/",
        views.StormEventsBrowseView.as_view(),
        name="storm_events_browse",
    ),
    path(
        "storm-events/save/",
        views.StormEventsSaveDatasetView.as_view(),
        name="storm_events_save",
    ),
    path(
        "datasets/<uuid:pk>/gev/",
        views.GEVRunCreateView.as_view(),
        name="gev_create",
    ),
    path("gev/<uuid:pk>/", views.GEVRunDetailView.as_view(), name="gev_detail"),
    path(
        "gev/<uuid:pk>/delete/",
        views.GEVRunDeleteView.as_view(),
        name="gev_delete",
    ),
]
