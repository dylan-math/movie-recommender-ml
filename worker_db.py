"""Read-only PostgreSQL access for Worker (plotwise ``user_titles``).

Does not persist raw interactions — only runs SQL and returns rows to the caller.
Poll/retrain cursors live in ``WorkerState`` (RAM), not here.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from utils import normalize_bot_item_id

log = logging.getLogger("worker-db")

_engine: Engine | None = None


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is required")
    return url


def database_url_masked() -> str | None:
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    return make_url(url).render_as_string(hide_password=True)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(database_url(), pool_pre_ping=True, pool_size=5, max_overflow=5)
    return _engine


@contextmanager
def connect() -> Iterator[Any]:
    with get_engine().connect() as conn:
        yield conn


def load_external_item_map(artifact_dir: Path | None) -> dict[str, int]:
    path_raw = os.getenv("ITEM_ID_MAP_PATH")
    repo_root = Path(__file__).resolve().parent
    paths: list[Path] = []
    if path_raw:
        paths.append(Path(path_raw))
    if artifact_dir is not None:
        paths.append(artifact_dir / "item_id_map.csv")
    paths.append(repo_root / "train_data/movielens/item_id_map.csv")

    mapping: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            import pandas as pd

            frame = pd.read_csv(path)
        except Exception as exc:
            log.warning("Failed to load item id map %s: %s", path, exc)
            continue
        ext_col = next((c for c in frame.columns if c.lower() in ("external_id", "item_id", "tmdb_id")), None)
        movie_col = next((c for c in frame.columns if c.lower() in ("movieid", "movie_id", "movielens_id")), None)
        if ext_col is None or movie_col is None:
            continue
        for _, row in frame.iterrows():
            external = str(row[ext_col]).strip()
            try:
                movie_id = int(row[movie_col])
            except (TypeError, ValueError):
                continue
            public_id = normalize_bot_item_id(external) or external
            mapping[public_id] = movie_id
    return mapping


def fetch_user_rating_snapshot(user_id: str) -> tuple[list[tuple[str, float]], datetime | None]:
    sql = os.getenv(
        "DB_RATINGS_SQL",
        """
        SELECT item_id, rating, updated_at FROM user_titles
        WHERE user_id = :user_id AND rating IS NOT NULL
        ORDER BY updated_at
        """,
    ).strip()
    with connect() as conn:
        rows = conn.execute(text(sql), {"user_id": str(user_id)}).mappings().all()
    ratings: list[tuple[str, float]] = []
    max_updated: datetime | None = None
    for row in rows:
        if row.get("item_id") is None or row.get("rating") is None:
            continue
        ratings.append((str(row["item_id"]), float(row["rating"])))
        updated = row.get("updated_at")
        if isinstance(updated, datetime) and (max_updated is None or updated > max_updated):
            max_updated = updated
    return ratings, max_updated


def fetch_user_ratings(user_id: str) -> list[tuple[str, float]]:
    ratings, _ = fetch_user_rating_snapshot(user_id)
    return ratings


def fetch_users_with_new_ratings(
    *,
    last_refreshed_by_user: dict[str, datetime],
    limit: int,
) -> list[tuple[str, datetime | None]]:
    """Users whose ``MAX(updated_at)`` is newer than their per-user watermark."""
    sql = os.getenv(
        "DB_PENDING_USERS_SQL",
        """
        SELECT user_id::text AS user_id, MAX(updated_at) AS max_updated
        FROM user_titles
        WHERE rating IS NOT NULL
        GROUP BY user_id
        """,
    ).strip()
    with connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    pending: list[tuple[str, datetime]] = []
    for row in rows:
        user_id = row.get("user_id")
        if user_id is None:
            continue
        updated = row.get("max_updated")
        if not isinstance(updated, datetime):
            continue
        uid = str(user_id)
        since = last_refreshed_by_user.get(uid)
        if since is None or updated > since:
            pending.append((uid, updated))
    pending.sort(key=lambda item: item[1])
    cap = max(1, min(limit, 10_000))
    return pending[:cap]


def fetch_interaction_count() -> int:
    sql = os.getenv(
        "DB_INTERACTION_COUNT_SQL",
        "SELECT COUNT(*) AS total FROM user_titles WHERE rating IS NOT NULL",
    ).strip()
    with connect() as conn:
        row = conn.execute(text(sql)).mappings().first()
    return int(row["total"]) if row else 0
