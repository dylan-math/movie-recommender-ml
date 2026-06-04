"""Download TMDB daily exports and rebuild the item catalog snap (scheduled or manual)."""

from __future__ import annotations

import gzip
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from catalog_builder import build_tmdb_catalog_bundle, default_output_dir

log = logging.getLogger("tmdb-catalog-sync")

DEFAULT_INTERVAL_SECONDS = 86400.0  # 1 day
EXPORT_BASE_URL = "https://files.tmdb.org/p/exports"


@dataclass
class CatalogSyncResult:
    snap_dir: Path
    n_items: int
    movie_export: Path
    tv_export: Path


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def catalog_sync_enabled() -> bool:
    return os.getenv("TMDB_CATALOG_SYNC_ENABLED", "0").lower() in ("1", "true", "yes")


def catalog_sync_interval_seconds() -> float:
    days = os.getenv("TMDB_CATALOG_SYNC_INTERVAL_DAYS")
    if days is not None:
        try:
            return max(3600.0, float(days) * 86400.0)
        except ValueError:
            pass
    return max(3600.0, env_float("TMDB_CATALOG_SYNC_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS))


def export_date_label(*, lag_days: int = 1) -> str:
    """TMDB export filename date (MM_DD_YYYY), usually yesterday UTC."""
    day = datetime.now(timezone.utc).date() - timedelta(days=lag_days)
    return f"{day.month:02d}_{day.day:02d}_{day.year}"


def _export_urls(date_label: str) -> tuple[str, str]:
    return (
        f"{EXPORT_BASE_URL}/movie_ids_{date_label}.jsonl.gz",
        f"{EXPORT_BASE_URL}/tv_series_ids_{date_label}.jsonl.gz",
    )


def _auth_headers() -> dict[str, str]:
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def download_export(url: str, destination: Path, *, timeout_seconds: float = 900.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = _auth_headers()
    log.info("Downloading %s", url)
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 403:
                raise PermissionError(
                    f"TMDB export denied (403) for {url}. "
                    "Set TMDB_API_KEY in .env or download files manually into TMDB_EXPORT_DIR."
                )
            response.raise_for_status()
            with destination.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
    log.info("Saved %s (%s bytes)", destination, destination.stat().st_size)


def gunzip_file(gz_path: Path) -> Path:
    out_path = gz_path.with_suffix("")
    with gzip.open(gz_path, "rb") as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return out_path


def download_daily_exports(
    export_dir: Path,
    *,
    date_label: str | None = None,
) -> tuple[Path, Path]:
    label = date_label or export_date_label()
    movie_url, tv_url = _export_urls(label)
    movie_gz = export_dir / f"movie_ids_{label}.jsonl.gz"
    tv_gz = export_dir / f"tv_series_ids_{label}.jsonl.gz"
    download_export(movie_url, movie_gz)
    download_export(tv_url, tv_gz)
    return gunzip_file(movie_gz), gunzip_file(tv_gz)


def run_catalog_sync(
    *,
    base_artifact_dir: Path,
    output_root: Path,
    export_dir: Path | None = None,
    movie_export: Path | None = None,
    tv_export: Path | None = None,
    links_csv: Path | None = None,
    snap_name: str | None = None,
    download: bool = True,
) -> CatalogSyncResult:
    """Full sync: optional download → build snap → return path (activate separately)."""
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    exports = Path(export_dir or os.getenv("TMDB_EXPORT_DIR", "train_data/tmdb/exports"))

    if download:
        if not api_key:
            if movie_export is None or tv_export is None:
                raise RuntimeError(
                    "TMDB_API_KEY is missing and export files were not provided. "
                    "Set TMDB_API_KEY or pass movie_export/tv_export paths."
                )
        else:
            movie_export, tv_export = download_daily_exports(exports)

    if movie_export is None or tv_export is None:
        raise FileNotFoundError("movie_export and tv_export are required for catalog sync")

    out_dir = default_output_dir(output_root, snap_name)
    built = build_tmdb_catalog_bundle(
        base_artifact_dir=base_artifact_dir,
        output_dir=out_dir,
        movie_export=movie_export,
        tv_export=tv_export,
        links_csv=links_csv,
        model_version=out_dir.name,
    )
    n_items = len(load_item_ids_only(built))
    return CatalogSyncResult(
        snap_dir=built,
        n_items=n_items,
        movie_export=movie_export,
        tv_export=tv_export,
    )


def load_item_ids_only(artifact_dir: Path) -> list[str]:
    import numpy as np

    ids = np.load(artifact_dir / "movie_ids.npy", allow_pickle=True)
    return [str(x) for x in np.asarray(ids).reshape(-1)]


def activate_catalog_snap(
    snap_dir: Path,
    *,
    recommender_url: str,
    reset_user_vectors: bool = False,
) -> dict:
    payload = {
        "artifact_dir": str(snap_dir),
        "reset_user_vectors": reset_user_vectors,
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{recommender_url.rstrip('/')}/v1/admin/model/activate",
            json=payload,
        )
        response.raise_for_status()
        return response.json()
