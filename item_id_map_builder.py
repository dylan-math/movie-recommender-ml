"""Build plotwise item_id (base64 TMDB token) ↔ MovieLens movieId map from GroupLens links.csv."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from utils import encode_tmdb_id_into_my_id

log = logging.getLogger("item-id-map")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MOVIELENS_DIR = REPO_ROOT / "train_data/movielens"

MOVIELENS_LINKS_URLS = {
    "32m": ("https://files.grouplens.org/datasets/movielens/ml-32m.zip", "ml-32m.zip"),
    "25m": ("https://files.grouplens.org/datasets/movielens/ml-25m.zip", "ml-25m.zip"),
    "latest": ("https://files.grouplens.org/datasets/movielens/ml-latest.zip", "ml-latest.zip"),
    "small": (
        "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "ml-latest-small.zip",
    ),
}


def read_links_csv(*, links_path: Path | None = None, zip_path: Path | None = None) -> pd.DataFrame:
    if links_path is not None:
        frame = pd.read_csv(links_path)
    elif zip_path is not None:
        with zipfile.ZipFile(zip_path) as archive:
            links_name = next(name for name in archive.namelist() if name.endswith("links.csv"))
            frame = pd.read_csv(io.BytesIO(archive.read(links_name)))
    else:
        raise ValueError("Provide links_path or zip_path")
    return frame


def load_snap_movie_ids(artifact_dir: Path) -> set[int]:
    ids_path = artifact_dir / "movie_ids.npy"
    metadata_path = artifact_dir / "movie_metadata.csv"
    if ids_path.exists():
        flat = np.load(ids_path, allow_pickle=True).reshape(-1)
        out: set[int] = set()
        for raw in flat:
            text = str(raw).strip()
            if text.isdigit():
                out.add(int(text))
        return out
    if metadata_path.exists():
        frame = pd.read_csv(metadata_path)
        col = "movieId" if "movieId" in frame.columns else "movie_idx"
        return {int(x) for x in frame[col].tolist()}
    raise FileNotFoundError(f"No movie_ids.npy or movie_metadata.csv under {artifact_dir}")


def build_item_id_map_frame(
    links: pd.DataFrame,
    *,
    movie_ids: set[int] | None = None,
) -> pd.DataFrame:
    tmdb_col = next((c for c in links.columns if "tmdb" in c.lower() and "id" in c.lower()), None)
    movie_col = next((c for c in links.columns if c.lower() in ("movieid", "movie_id")), None)
    if tmdb_col is None or movie_col is None:
        raise ValueError(f"links.csv needs movieId and tmdbId columns, got {list(links.columns)}")

    rows: list[dict[str, object]] = []
    for _, row in links.iterrows():
        try:
            movie_id = int(row[movie_col])
            tmdb_raw = row[tmdb_col]
        except (TypeError, ValueError):
            continue
        if pd.isna(tmdb_raw):
            continue
        try:
            tmdb_id = int(float(tmdb_raw))
        except (TypeError, ValueError):
            continue
        if tmdb_id <= 0:
            continue
        if movie_ids is not None and movie_id not in movie_ids:
            continue
        item_id = encode_tmdb_id_into_my_id(tmdb_id, "movie")
        rows.append({"item_id": item_id, "movieId": movie_id, "tmdbId": tmdb_id})

    if not rows:
        raise ValueError("No TMDB links produced — check links.csv and snap movie ids")

    frame = pd.DataFrame(rows).drop_duplicates(subset=["movieId"], keep="first")
    return frame


def write_item_id_map(
    output_path: Path,
    *,
    links_path: Path | None = None,
    zip_path: Path | None = None,
    artifact_dir: Path | None = None,
) -> Path:
    links = read_links_csv(links_path=links_path, zip_path=zip_path)
    movie_ids = load_snap_movie_ids(artifact_dir) if artifact_dir is not None else None
    frame = build_item_id_map_frame(links, movie_ids=movie_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    log.info(
        "Wrote %s rows to %s (snap movies=%s, linked=%s)",
        len(frame),
        output_path,
        len(movie_ids) if movie_ids is not None else "?",
        len(frame),
    )
    return output_path


def ensure_item_id_map_for_snap(artifact_dir: Path) -> Path | None:
    """Create ``<snap>/item_id_map.csv`` from local ml-32m.zip when missing."""
    target = artifact_dir / "item_id_map.csv"
    if target.exists():
        return target
    for zip_name in ("ml-32m.zip", "ml-25m.zip", "ml-latest.zip"):
        zip_path = DEFAULT_MOVIELENS_DIR / zip_name
        if not zip_path.exists():
            continue
        try:
            write_item_id_map(target, zip_path=zip_path, artifact_dir=artifact_dir)
            return target
        except Exception as exc:
            log.warning("Failed to build item_id_map from %s: %s", zip_path, exc)
    return None
