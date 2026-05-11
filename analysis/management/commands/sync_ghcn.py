"""Download/refresh GHCN-Daily metadata files and populate the catalog tables.

Usage::

    python manage.py sync_ghcn                # uses 30-day cache, US-only
    python manage.py sync_ghcn --refresh      # force re-download
    python manage.py sync_ghcn --max-age-days 7
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from analysis.models import GhcnElementAvailability, GhcnState, GhcnStation
from analysis.services.ghcn import elements as element_specs
from analysis.services.ghcn import fetcher
from analysis.services.ghcn.formats import (
    parse_inventory,
    parse_states,
    parse_stations,
)


_BULK_CHUNK = 5000


class Command(BaseCommand):
    help = (
        "Download GHCN-Daily metadata (stations, states, inventory) and "
        "populate the GhcnState / GhcnStation / GhcnElementAvailability tables. "
        "US stations only."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Force re-download of metadata files even if cached.",
        )
        parser.add_argument(
            "--max-age-days",
            type=int,
            default=30,
            help="Skip downloading files newer than this many days (default 30).",
        )
        parser.add_argument(
            "--cache-dir",
            default=None,
            help="Override the metadata cache directory.",
        )
        parser.add_argument(
            "--keep-all-elements",
            action="store_true",
            help=(
                "Keep all elements in the inventory (default keeps only the "
                "supported subset: PRCP, TMAX, TMIN, TAVG, AWND, WSF2, WSF5, "
                "SNOW, SNWD)."
            ),
        )

    def handle(self, *args, **options):
        cache_dir = Path(options["cache_dir"]) if options["cache_dir"] else (
            Path(settings.MEDIA_ROOT) / "ghcn_cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.stdout.write(f"Cache directory: {cache_dir}")

        max_age = timedelta(days=int(options["max_age_days"]))
        force = bool(options["refresh"])

        try:
            files = fetcher.fetch_metadata(cache_dir, force=force, max_age=max_age)
        except fetcher.FetchError as exc:
            self.stderr.write(self.style.ERROR(f"Download failed: {exc}"))
            return

        for name, path in files.items():
            mb = path.stat().st_size / (1024 * 1024)
            self.stdout.write(f"  {name}: {mb:.1f} MB at {path}")

        keep_elements = (
            None
            if options["keep_all_elements"]
            else set(element_specs.supported_codes())
        )

        with transaction.atomic():
            n_states = self._sync_states(files["ghcnd-states.txt"])
            n_stations = self._sync_stations(files["ghcnd-stations.txt"])
            n_inventory = self._sync_inventory(
                files["ghcnd-inventory.txt"], keep_elements=keep_elements
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Synced {n_states} states, {n_stations} US stations, "
                f"{n_inventory} element-availability rows."
            )
        )

    # ------------------------------------------------------------------
    def _sync_states(self, path: Path) -> int:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            records = list(parse_states(fh))

        existing = {s.code: s for s in GhcnState.objects.all()}
        to_create = []
        to_update = []
        for rec in records:
            obj = existing.get(rec.code)
            if obj is None:
                to_create.append(GhcnState(code=rec.code, name=rec.name))
            elif obj.name != rec.name:
                obj.name = rec.name
                to_update.append(obj)
        GhcnState.objects.bulk_create(to_create, batch_size=_BULK_CHUNK)
        if to_update:
            GhcnState.objects.bulk_update(to_update, fields=["name"], batch_size=_BULK_CHUNK)
        return len(records)

    # ------------------------------------------------------------------
    def _sync_stations(self, path: Path) -> int:
        # US only by design (per the user spec).
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            records = list(parse_stations(fh, us_only=True))

        existing_ids = set(GhcnStation.objects.values_list("id", flat=True))
        to_create = []
        to_update = []
        for rec in records:
            data = dict(
                name=rec.name,
                state=rec.state,
                latitude=rec.latitude,
                longitude=rec.longitude,
                elevation=rec.elevation,
                network_code=rec.network_code,
                gsn_flag=rec.gsn_flag,
                hcn_flag=rec.hcn_flag,
                wmo_id=rec.wmo_id,
            )
            if rec.id in existing_ids:
                to_update.append(GhcnStation(id=rec.id, **data))
            else:
                to_create.append(GhcnStation(id=rec.id, **data))

        GhcnStation.objects.bulk_create(to_create, batch_size=_BULK_CHUNK)
        if to_update:
            GhcnStation.objects.bulk_update(
                to_update,
                fields=[
                    "name",
                    "state",
                    "latitude",
                    "longitude",
                    "elevation",
                    "network_code",
                    "gsn_flag",
                    "hcn_flag",
                    "wmo_id",
                ],
                batch_size=_BULK_CHUNK,
            )
        return len(records)

    # ------------------------------------------------------------------
    def _sync_inventory(
        self, path: Path, *, keep_elements: set[str] | None
    ) -> int:
        # Easiest correctness model: drop and re-insert. Inventory rarely
        # changes in shape, and the table is only a few hundred thousand rows
        # for US-only with the supported-element filter.
        GhcnElementAvailability.objects.all().delete()

        # Validate referenced station ids exist in our table; the file contains
        # rows for stations we've filtered out (international) when --us-only is
        # set on stations parse, but parse_inventory(us_only=True) handles that.
        valid_ids = set(GhcnStation.objects.values_list("id", flat=True))

        with path.open("r", encoding="utf-8", errors="replace") as fh:
            batch: list[GhcnElementAvailability] = []
            count = 0
            for rec in parse_inventory(
                fh, us_only=True, keep_elements=keep_elements
            ):
                if rec.station_id not in valid_ids:
                    continue
                batch.append(
                    GhcnElementAvailability(
                        station_id=rec.station_id,
                        element=rec.element,
                        first_year=rec.first_year,
                        last_year=rec.last_year,
                    )
                )
                if len(batch) >= _BULK_CHUNK:
                    GhcnElementAvailability.objects.bulk_create(batch)
                    count += len(batch)
                    batch.clear()
            if batch:
                GhcnElementAvailability.objects.bulk_create(batch)
                count += len(batch)
        return count
