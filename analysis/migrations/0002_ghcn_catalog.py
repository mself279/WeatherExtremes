"""Add GHCN catalog tables and link Dataset to its source station."""
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("analysis", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="GhcnState",
            fields=[
                ("code", models.CharField(max_length=2, primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=64)),
            ],
            options={
                "verbose_name": "GHCN state/province",
                "verbose_name_plural": "GHCN states/provinces",
                "ordering": ("name",),
            },
        ),
        migrations.CreateModel(
            name="GhcnStation",
            fields=[
                (
                    "id",
                    models.CharField(max_length=11, primary_key=True, serialize=False),
                ),
                ("name", models.CharField(max_length=64)),
                ("state", models.CharField(blank=True, db_index=True, max_length=2)),
                ("latitude", models.FloatField()),
                ("longitude", models.FloatField()),
                ("elevation", models.FloatField(blank=True, null=True)),
                (
                    "network_code",
                    models.CharField(
                        db_index=True,
                        help_text=(
                            "3rd character of the ID: W=WBAN, C=Cooperative, "
                            "1=CoCoRaHS, etc."
                        ),
                        max_length=1,
                    ),
                ),
                ("gsn_flag", models.CharField(blank=True, max_length=3)),
                ("hcn_flag", models.CharField(blank=True, max_length=3)),
                ("wmo_id", models.CharField(blank=True, max_length=5)),
            ],
            options={
                "verbose_name": "GHCN station",
                "ordering": ("state", "name"),
                "indexes": [
                    models.Index(
                        fields=["state", "name"],
                        name="analysis_gh_state_3c8ecc_idx",
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="GhcnElementAvailability",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("element", models.CharField(db_index=True, max_length=4)),
                ("first_year", models.IntegerField()),
                ("last_year", models.IntegerField()),
                (
                    "station",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="elements",
                        to="analysis.ghcnstation",
                    ),
                ),
            ],
            options={
                "verbose_name": "GHCN element availability",
                "verbose_name_plural": "GHCN element availabilities",
                "ordering": ("station", "element"),
                "indexes": [
                    models.Index(
                        fields=["element", "first_year", "last_year"],
                        name="analysis_gh_element_b8c8a3_idx",
                    ),
                    models.Index(
                        fields=["element", "last_year"],
                        name="analysis_gh_element_5e1d4f_idx",
                    ),
                ],
                "unique_together": {("station", "element")},
            },
        ),
        migrations.AddField(
            model_name="dataset",
            name="source_station",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="datasets",
                to="analysis.ghcnstation",
            ),
        ),
        migrations.AddField(
            model_name="dataset",
            name="source_metadata",
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text=(
                    "Element, thresholds, operator, raw range, etc. for derived "
                    "datasets."
                ),
            ),
        ),
    ]
