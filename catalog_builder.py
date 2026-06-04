"""Build recommender item catalog from TMDB exports + optional MovieLens factor carry-over."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from als_runtime import load_artifact_bundle
from media_types import MediaType
from utils import encode_tmdb_id_into_my_id

log = logging.getLogger("catalog-builder")


def _utc_snap_name(prefix: str = "snap-tmdb") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def iter_tmdb_export_jsonl(path: Path, media_type: MediaType) -> Iterator[tuple[str, str | None]]:
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            tmdb_id = int(row["id"])
            title = row.get("title") or row.get("name") or row.get("original_title")
            yield encode_tmdb_id_into_my_id(tmdb_id, media_type), str(title) if title else None


def load_movielens_tmdb_links(path: Path) -> dict[str, int]:
    """Map plotwise item_id token -> MovieLens movieId when a links file is provided."""
    frame = pd.read_csv(path)
    tmdb_col = next((c for c in frame.columns if "tmdb" in c.lower() and "id" in c.lower()), None)
    movie_col = next((c for c in frame.columns if c.lower() in ("movieid", "movie_id", "movielens_id")), None)
    if tmdb_col is None or movie_col is None:
        raise ValueError(f"{path}: need tmdbId and movieId columns")
    mapping: dict[str, int] = {}
    for _, row in frame.iterrows():
        try:
            movie_id = int(row[movie_col])
            tmdb_id = int(row[tmdb_col])
        except (TypeError, ValueError):
            continue
        token = encode_tmdb_id_into_my_id(tmdb_id, "movie")
        mapping.setdefault(token, movie_id)
    return mapping


def build_tmdb_catalog_bundle(
    *,
    base_artifact_dir: Path,
    output_dir: Path,
    movie_export: Path | None = None,
    tv_export: Path | None = None,
    links_csv: Path | None = None,
    max_items: int | None = None,
    model_version: str | None = None,
) -> Path:
    """Expand item catalog with all TMDB ids; cold-start factors = mean ALS item vector."""
    base_dir = Path(base_artifact_dir)
    base = load_artifact_bundle(base_dir)
    cold_factor = base.item_factors.mean(axis=0).astype(np.float32)

    ml_by_token: dict[str, int] = {}
    if links_csv is not None and links_csv.exists():
        ml_by_token = load_movielens_tmdb_links(links_csv)
        log.info("Loaded %s TMDB→MovieLens links", len(ml_by_token))

    item_ids: list[str] = []
    factors: list[np.ndarray] = []
    titles: list[str | None] = []
    seen: set[str] = set()

    def add_item(token: str, title: str | None) -> None:
        if token in seen:
            return
        if max_items is not None and len(seen) >= max_items:
            return
        seen.add(token)
        movie_id = ml_by_token.get(token)
        if movie_id is not None and int(movie_id) in base.movie_id_to_idx:
            vec = base.item_factors[base.movie_id_to_idx[int(movie_id)]]
        else:
            vec = cold_factor
        item_ids.append(token)
        factors.append(vec)
        titles.append(title)

    sources: list[tuple[Path, MediaType]] = []
    if movie_export is not None:
        sources.append((movie_export, "movie"))
    if tv_export is not None:
        sources.append((tv_export, "tv"))
    if not sources:
        raise ValueError("Provide at least one of movie_export or tv_export")

    for path, media_type in sources:
        if not path.exists():
            raise FileNotFoundError(path)
        log.info("Reading %s (%s)", path, media_type)
        for token, title in iter_tmdb_export_jsonl(path, media_type):
            add_item(token, title)
            if max_items is not None and len(seen) >= max_items:
                break

    if not item_ids:
        raise ValueError("Catalog is empty — check export paths")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    item_factors = np.stack(factors).astype(np.float32)
    np.save(out / "item_factors.npy", item_factors)
    np.save(out / "movie_ids.npy", np.array(item_ids, dtype=np.str_))

    user_factors_path = base_dir / "user_factors.npy"
    if user_factors_path.exists():
        shutil.copy2(user_factors_path, out / "user_factors.npy")
        shutil.copy2(base_dir / "user_ids.npy", out / "user_ids.npy")

    metadata = pd.DataFrame(
        {
            "item_idx": np.arange(len(item_ids), dtype=np.int64),
            "item_id": item_ids,
            "title": [t or "" for t in titles],
        }
    )
    metadata.to_csv(out / "movie_metadata.csv", index=False)

    version = model_version or out.name
    config_path = base_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config.update(
        {
            "model_type": "explicit_als",
            "model_version": version,
            "item_id_format": "plotwise_tmdb",
            "n_items": len(item_ids),
            "catalog_source": "tmdb_export",
            "base_artifact_dir": str(base_dir),
        }
    )
    with open(out / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    log.info("Wrote catalog snap %s with %s items", out, len(item_ids))
    return out


def default_output_dir(output_root: Path, name: str | None = None) -> Path:
    snap = name or _utc_snap_name()
    return output_root / snap
