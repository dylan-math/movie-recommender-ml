"""Shared helpers for anime train-data ingest (offline, not runtime)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
ANIMEDATA_DIR = REPO_ROOT / "train_data" / "animedata"
RAW_DIR = ANIMEDATA_DIR / "raw"
PROCESSED_DIR = ANIMEDATA_DIR / "processed"

# MovieLens 32M max movieId ~ 84k; anime synthetic ids start here.
ANIME_MOVIE_ID_OFFSET = 10_000_000


def anime_rating_to_plotwise(rating: float) -> float:
    """Map anime scale 1..10 to Plotwise/MovieLens explicit scale 0.5..5."""
    value = float(rating)
    if not 1.0 <= value <= 10.0:
        raise ValueError(f"anime rating must be in [1, 10], got {rating}")
    converted = 0.5 + (value - 1.0) * 4.5 / 9.0
    return round(converted * 2.0) / 2.0  # snap to half-star steps


def synthetic_movie_id(anime_id: int | str) -> int:
    return ANIME_MOVIE_ID_OFFSET + int(anime_id)


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in frame.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def read_anime_catalog(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    anime_col = _pick_column(frame, ("anime_id", "animeId", "mal_id", "id"))
    title_col = _pick_column(frame, ("title", "name", "anime_title"))
    if anime_col is None or title_col is None:
        raise ValueError(f"{path}: need anime_id and title columns, got {list(frame.columns)}")

    out = pd.DataFrame(
        {
            "anime_id": frame[anime_col].astype(int),
            "title": frame[title_col].astype(str).str.strip(),
        }
    )
    for optional, aliases in (
        ("title_english", ("title_english", "english", "title_en", "english name")),
        ("year", ("year", "start_year", "aired_year")),
        ("type", ("type", "media_type", "format")),
    ):
        col = _pick_column(frame, aliases)
        if col is not None:
            out[optional] = frame[col]
    out["movieId"] = out["anime_id"].map(synthetic_movie_id)
    return out.drop_duplicates(subset=["anime_id"], keep="first")


def _rating_column_names(path: Path) -> tuple[str, str, str]:
    header = pd.read_csv(path, nrows=0)
    user_col = _pick_column(header, ("user_id", "userId", "username"))
    anime_col = _pick_column(header, ("anime_id", "animeId", "mal_id"))
    rating_col = _pick_column(header, ("rating", "score", "Rating"))
    if user_col is None or anime_col is None or rating_col is None:
        raise ValueError(f"{path}: need user_id, anime_id, rating; got {list(header.columns)}")
    return user_col, anime_col, rating_col


def _transform_rating_chunk(frame: pd.DataFrame, user_col: str, anime_col: str, rating_col: str) -> pd.DataFrame:
    anime_ids = frame[anime_col].astype("int32")
    rating_anime = frame[rating_col].astype("float32")
    converted = 0.5 + (rating_anime - 1.0) * 4.5 / 9.0
    rating = (converted * 2.0).round() / 2.0
    return pd.DataFrame(
        {
            "user_id": frame[user_col].astype(str),
            "anime_id": anime_ids,
            "item_id": anime_ids + ANIME_MOVIE_ID_OFFSET,
            "rating": rating.astype("float32"),
            "timestamp": 0,
        }
    )


def iter_anime_rating_chunks(path: Path, *, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    user_col, anime_col, rating_col = _rating_column_names(path)
    reader = pd.read_csv(path, usecols=[user_col, anime_col, rating_col], chunksize=chunksize)
    for chunk in reader:
        yield _transform_rating_chunk(chunk, user_col, anime_col, rating_col)


def write_interactions_parquet(
    ratings_path: Path,
    catalog: pd.DataFrame,
    output_path: Path,
    *,
    chunksize: int = 500_000,
) -> tuple[int, int, int]:
    """Stream ratings CSV → interactions.parquet; returns (rows, dropped, unique_users)."""
    known_anime = set(catalog["anime_id"].tolist())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    dropped_rows = 0
    users: set[str] = set()

    for chunk in iter_anime_rating_chunks(ratings_path, chunksize=chunksize):
        before = len(chunk)
        chunk = chunk[chunk["anime_id"].isin(known_anime)]
        dropped_rows += before - len(chunk)
        if chunk.empty:
            continue

        interactions = chunk[["user_id", "item_id", "rating", "timestamp"]].copy()
        users.update(interactions["user_id"].unique())
        table = pa.Table.from_pandas(interactions, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        total_rows += len(interactions)

    if writer is not None:
        writer.close()
    else:
        empty = pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"])
        empty.to_parquet(output_path, index=False)

    return total_rows, dropped_rows, len(users)
