"""Read-only PostgreSQL access for Worker (plotwise ``user_titles``).

Does not persist raw interactions — only runs SQL and returns rows to the caller.
Poll/retrain cursors live in ``WorkerState`` (RAM), not here.
"""

from __future__ import annotations

import hashlib
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


def build_ratings_fingerprint(parts: list[tuple[str, str]]) -> str:
    """Stable hash of (item_id, rating) pairs; rating text must match Postgres ``rating::text``."""
    payload = ",".join(
        f"{item_id}:{rating_text}"
        for item_id, rating_text in sorted(parts, key=lambda item: item[0])
    )
    return hashlib.md5(payload.encode()).hexdigest()


def fetch_user_rating_snapshot(user_id: str) -> tuple[list[tuple[str, float]], datetime | None, str]:
    sql = os.getenv(
        "DB_RATINGS_SQL",
        """
        SELECT item_id::text AS item_id, rating, rating::text AS rating_text, updated_at
        FROM user_titles
        WHERE user_id = :user_id AND rating IS NOT NULL
        ORDER BY item_id
        """,
    ).strip()
    with connect() as conn:
        rows = conn.execute(text(sql), {"user_id": str(user_id)}).mappings().all()
    ratings: list[tuple[str, float]] = []
    fingerprint_parts: list[tuple[str, str]] = []
    max_updated: datetime | None = None
    for row in rows:
        if row.get("item_id") is None or row.get("rating") is None:
            continue
        item_id = str(row["item_id"])
        ratings.append((item_id, float(row["rating"])))
        rating_text = row.get("rating_text")
        if rating_text is not None:
            fingerprint_parts.append((item_id, str(rating_text)))
        updated = row.get("updated_at")
        if isinstance(updated, datetime) and (max_updated is None or updated > max_updated):
            max_updated = updated
    return ratings, max_updated, build_ratings_fingerprint(fingerprint_parts)


def fetch_user_ratings(user_id: str) -> list[tuple[str, float]]:
    ratings, _, _ = fetch_user_rating_snapshot(user_id)
    return ratings


def fetch_users_with_changed_ratings(
    *,
    last_fingerprint_by_user: dict[str, str],
    limit: int,
) -> list[str]:
    """Users whose current ratings fingerprint differs from the last refreshed snapshot."""
    sql = os.getenv(
        "DB_PENDING_USERS_SQL",
        """
        SELECT user_id::text AS user_id, item_id::text AS item_id,
               rating::text AS rating_text, updated_at
        FROM user_titles
        WHERE rating IS NOT NULL
        ORDER BY user_id, item_id
        """,
    ).strip()
    with connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()

    parts_by_user: dict[str, list[tuple[str, str]]] = {}
    max_updated_by_user: dict[str, datetime] = {}
    for row in rows:
        user_id = row.get("user_id")
        item_id = row.get("item_id")
        rating_text = row.get("rating_text")
        if user_id is None or item_id is None or rating_text is None:
            continue
        uid = str(user_id)
        parts_by_user.setdefault(uid, []).append((str(item_id), str(rating_text)))
        updated = row.get("updated_at")
        if isinstance(updated, datetime):
            prev = max_updated_by_user.get(uid)
            if prev is None or updated > prev:
                max_updated_by_user[uid] = updated

    pending: list[tuple[str, datetime | None]] = []
    for uid, parts in parts_by_user.items():
        fingerprint = build_ratings_fingerprint(parts)
        if last_fingerprint_by_user.get(uid) != fingerprint:
            pending.append((uid, max_updated_by_user.get(uid)))

    pending.sort(key=lambda item: item[1] or datetime.min.replace(tzinfo=None))
    cap = max(1, min(limit, 10_000))
    return [uid for uid, _ in pending[:cap]]


def fetch_interaction_count() -> int:
    sql = os.getenv(
        "DB_INTERACTION_COUNT_SQL",
        "SELECT COUNT(*) AS total FROM user_titles WHERE rating IS NOT NULL",
    ).strip()
    with connect() as conn:
        row = conn.execute(text(sql)).mappings().first()
    return int(row["total"]) if row else 0
