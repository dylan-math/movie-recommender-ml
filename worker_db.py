"""Read-only PostgreSQL access for Worker (plotwise ``user_titles``).

Does not persist raw interactions — only runs SQL and returns rows to the caller.
Poll/retrain cursors live in ``WorkerState`` (RAM), not here.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

log = logging.getLogger("worker-db")

DEFAULT_DATABASE_URL = "postgresql+psycopg2://app_user:app_password@postgres:5432/app_db"
TMDB_ITEM_RE = re.compile(r"^tmdb_(?:movie|tv)_(\d+)$", re.IGNORECASE)

_engine: Engine | None = None


def database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(database_url(), pool_pre_ping=True, pool_size=5, max_overflow=5)
    return _engine


@contextmanager
def connect() -> Iterator[Any]:
    with get_engine().connect() as conn:
        yield conn


def decode_item_token(raw: str) -> str | None:
    value = str(raw).strip()
    if not value:
        return None
    if value.isdigit():
        return value
    try:
        pad = (-len(value)) % 4
        return base64.b64decode(value + ("=" * pad)).decode("utf-8")
    except Exception:
        return value


def load_external_item_map(artifact_dir: Path | None) -> dict[str, int]:
    path_raw = os.getenv("ITEM_ID_MAP_PATH")
    paths: list[Path] = []
    if path_raw:
        paths.append(Path(path_raw))
    if artifact_dir is not None:
        paths.append(artifact_dir / "item_id_map.csv")

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
            mapping[external] = movie_id
            decoded = decode_item_token(external)
            if decoded:
                mapping[decoded] = movie_id
    return mapping


def resolve_movie_id(
    raw_item_id: object,
    *,
    movie_id_to_idx: dict[int, int],
    external_map: dict[str, int],
) -> int | None:
    token = str(raw_item_id).strip()
    candidates = [token]
    decoded = decode_item_token(token)
    if decoded and decoded not in candidates:
        candidates.append(decoded)
    for candidate in candidates:
        if candidate in external_map:
            movie_id = external_map[candidate]
            if movie_id in movie_id_to_idx:
                return movie_id
        if candidate.isdigit():
            movie_id = int(candidate)
            if movie_id in movie_id_to_idx:
                return movie_id
        match = TMDB_ITEM_RE.match(candidate)
        if match and int(match.group(1)) in movie_id_to_idx:
            return int(match.group(1))
    return None


def fetch_user_ratings(user_id: str) -> list[tuple[str, float]]:
    sql = os.getenv(
        "DB_RATINGS_SQL",
        """
        SELECT item_id, rating FROM user_titles
        WHERE user_id = :user_id AND rating IS NOT NULL
        ORDER BY updated_at
        """,
    ).strip()
    with connect() as conn:
        rows = conn.execute(text(sql), {"user_id": str(user_id)}).mappings().all()
    return [
        (str(row["item_id"]), float(row["rating"]))
        for row in rows
        if row.get("item_id") is not None and row.get("rating") is not None
    ]


def fetch_pending_users(*, since: datetime | None, limit: int) -> list[tuple[str, datetime | None]]:
    sql = os.getenv(
        "DB_PENDING_USERS_SQL",
        """
        SELECT user_id, MAX(updated_at) AS max_updated FROM user_titles
        WHERE rating IS NOT NULL AND (:since IS NULL OR updated_at > :since)
        GROUP BY user_id ORDER BY max_updated LIMIT :limit
        """,
    ).strip()
    with connect() as conn:
        rows = conn.execute(
            text(sql),
            {"since": since, "limit": max(1, min(limit, 10_000))},
        ).mappings().all()
    pending: list[tuple[str, datetime | None]] = []
    for row in rows:
        if row.get("user_id") is None:
            continue
        updated = row.get("max_updated")
        pending.append(
            (
                str(row["user_id"]),
                updated if isinstance(updated, datetime) else None,
            )
        )
    return pending


def fetch_interaction_count() -> int:
    sql = os.getenv(
        "DB_INTERACTION_COUNT_SQL",
        "SELECT COUNT(*) AS total FROM user_titles WHERE rating IS NOT NULL",
    ).strip()
    with connect() as conn:
        row = conn.execute(text(sql)).mappings().first()
    return int(row["total"]) if row else 0
