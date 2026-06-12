#!/usr/bin/env python3
"""Match anime titles to TMDB ids (no tmdb in source — search API)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from animedata_lib import ANIMEDATA_DIR, PROCESSED_DIR, RAW_DIR, read_anime_catalog  # noqa: E402
from utils import encode_tmdb_id_into_my_id  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("match-animedata-tmdb")


def _load_env_file() -> None:
    env_path = ANIMEDATA_DIR.parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


TMDB_SEARCH = "https://api.themoviedb.org/3/search/multi"
DEFAULT_RPS = 20


class RateLimiter:
    """Sliding 1s window — at most *max_per_second* calls per second."""

    def __init__(self, max_per_second: int) -> None:
        self.max_per_second = max(1, max_per_second)
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 1.0:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_second:
            sleep_for = 1.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] >= 1.0:
                self._timestamps.popleft()
        self._timestamps.append(time.monotonic())


def _search_tmdb(
    client: httpx.Client,
    *,
    api_key: str,
    query: str,
    year: int | None,
    prefer_type: str,
) -> dict | None:
    params: dict[str, str | int] = {"api_key": api_key, "query": query, "include_adult": "false"}
    if year is not None:
        params["year"] = year
    response = client.get(TMDB_SEARCH, params=params, timeout=20.0)
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None

    def score(item: dict) -> tuple[int, float]:
        media = str(item.get("media_type") or "")
        type_match = 1 if media == prefer_type else 0
        popularity = float(item.get("popularity") or 0.0)
        return (type_match, popularity)

    best = max(results, key=score)
    if str(best.get("media_type")) not in ("tv", "movie"):
        return None
    return best


def _proxy_label(proxy: str) -> str:
    parsed = urlparse(proxy)
    if parsed.hostname:
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.hostname}{port}"
    return "configured"


def _make_tmdb_client(proxy: str | None) -> httpx.Client:
    if proxy:
        log.info("TMDB proxy: %s", _proxy_label(proxy))
    return httpx.Client(proxy=proxy, trust_env=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="TMDB title search for anime catalog")
    parser.add_argument("--catalog", type=Path, default=PROCESSED_DIR / "anime_catalog.csv")
    parser.add_argument("--raw-anime", type=Path, default=RAW_DIR / "anime.csv")
    parser.add_argument("--overrides", type=Path, default=RAW_DIR / "tmdb_overrides.csv")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--rps",
        type=int,
        default=DEFAULT_RPS,
        help=f"max TMDB API requests per second (default: {DEFAULT_RPS})",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="HTTP(S) proxy URL (default: TMDB_PROXY from .env)",
    )
    args = parser.parse_args()

    _load_env_file()
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set TMDB_API_KEY in .env or environment")

    proxy = (args.proxy or os.getenv("TMDB_PROXY", "")).strip() or None

    limiter = RateLimiter(args.rps)
    log.info("TMDB rate limit: %s req/s", args.rps)

    if args.catalog.is_file():
        catalog = pd.read_csv(args.catalog)
    elif args.raw_anime.is_file():
        catalog = read_anime_catalog(args.raw_anime)
    else:
        raise SystemExit(f"Need {args.catalog} or {args.raw_anime}")

    overrides: dict[int, tuple[int, str]] = {}
    if args.overrides.is_file():
        ov = pd.read_csv(args.overrides)
        for _, row in ov.iterrows():
            overrides[int(row["anime_id"])] = (int(row["tmdb_id"]), str(row.get("media_type") or "tv"))

    matched: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []

    with _make_tmdb_client(proxy) as client:
        for _, row in catalog.iterrows():
            anime_id = int(row["anime_id"])
            movie_id = int(row["movieId"])
            title = str(row["title"])
            title_en = str(row["title_english"]) if "title_english" in row and pd.notna(row["title_english"]) else ""
            prefer = str(row["type"]).strip().lower() if "type" in row and pd.notna(row["type"]) else "tv"
            if prefer not in ("tv", "movie"):
                prefer = "tv"
            year_val = row.get("year")
            year = int(year_val) if pd.notna(year_val) else None

            if anime_id in overrides:
                tmdb_id, media_type = overrides[anime_id]
                item_id = encode_tmdb_id_into_my_id(tmdb_id, media_type)
                matched.append(
                    {
                        "item_id": item_id,
                        "movieId": movie_id,
                        "tmdbId": tmdb_id,
                        "media_type": media_type,
                        "anime_id": anime_id,
                        "title": title,
                        "match_source": "override",
                    }
                )
                continue

            hit = None
            for query in (title_en, title):
                if not query:
                    continue
                try:
                    limiter.wait()
                    hit = _search_tmdb(
                        client,
                        api_key=api_key,
                        query=query,
                        year=year,
                        prefer_type=prefer,
                    )
                except httpx.HTTPError as exc:
                    log.warning("TMDB error anime_id=%s: %s", anime_id, exc)
                    hit = None
                if hit is not None:
                    break

            if hit is None:
                unmatched.append({"anime_id": anime_id, "movieId": movie_id, "title": title})
                continue

            media_type = str(hit["media_type"])
            tmdb_id = int(hit["id"])
            item_id = encode_tmdb_id_into_my_id(tmdb_id, media_type)
            matched.append(
                {
                    "item_id": item_id,
                    "movieId": movie_id,
                    "tmdbId": tmdb_id,
                    "media_type": media_type,
                    "anime_id": anime_id,
                    "title": title,
                    "tmdb_title": hit.get("title") or hit.get("name"),
                    "match_source": "search",
                }
            )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if matched:
        map_frame = pd.DataFrame(matched)[["item_id", "movieId", "tmdbId", "title"]]
        map_frame.to_csv(out_dir / "item_id_map.csv", index=False)
    if unmatched:
        pd.DataFrame(unmatched).to_csv(out_dir / "unmatched_anime.csv", index=False)

    log.info("Matched %s / %s anime → %s", len(matched), len(catalog), out_dir / "item_id_map.csv")
    if unmatched:
        log.info("Unmatched %s → %s", len(unmatched), out_dir / "unmatched_anime.csv")


if __name__ == "__main__":
    main()
