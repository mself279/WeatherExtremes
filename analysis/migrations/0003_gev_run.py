"""Add GEVRun model for Generalized Extreme Value analyses."""
import django.db.models.deletion
import django.utils.timezone
import uuid

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("analysis", "0002_ghcn_catalog"),
    ]

    operations = [
        migrations.CreateModel(
            name="GEVRun",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("label", models.CharField(blank=True, max_length=200)),
                ("parameters", models.JSONField()),
                ("summary", models.JSONField(blank=True, default=dict)),
                ("figures", models.JSONField(blank=True, default=dict)),
                ("series", models.JSONField(blank=True, default=dict)),
                ("confidence_intervals", models.JSONField(blank=True, default=dict)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("pending", "Pending"),
                            ("running", "Running"),
                            ("success", "Success"),
                            ("failed", "Failed"),
                        ],
                        default="pending",
                        max_length=16,
                    ),
                ),
                ("error", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                (
                    "dataset",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="gev_runs",
                        to="analysis.dataset",
                    ),
                ),
            ],
            options={
                "verbose_name": "GEV run",
                "ordering": ("-created_at",),
            },
        ),
    ]
