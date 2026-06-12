#!/usr/bin/env python3
"""Build animedata/processed/interactions.parquet from raw CSV files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from animedata_lib import (  # noqa: E402
    PROCESSED_DIR,
    RAW_DIR,
    read_anime_catalog,
    write_interactions_parquet,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build-animedata")

RATINGS_FILENAMES = ("ratings.csv", "rating_complete.csv")


def resolve_ratings_path(raw_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    for name in RATINGS_FILENAMES:
        path = raw_dir / name
        if path.is_file():
            if name != "ratings.csv":
                log.info("Using %s (ratings.csv not found)", path.name)
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert animedata raw CSV → interactions.parquet")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--ratings", type=Path, default=None, help="default: <raw-dir>/ratings.csv")
    parser.add_argument("--anime", type=Path, default=None, help="default: <raw-dir>/anime.csv")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    ratings_path = resolve_ratings_path(raw_dir, args.ratings)
    anime_path = args.anime or raw_dir / "anime.csv"
    if ratings_path is None or not ratings_path.is_file():
        names = ", ".join(RATINGS_FILENAMES)
        raise SystemExit(f"Missing ratings file in {raw_dir}/ — drop one of: {names}")
    if not anime_path.is_file():
        raise SystemExit(f"Missing {anime_path} — drop anime.csv into {raw_dir}/")

    catalog = read_anime_catalog(anime_path)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_dir / "anime_catalog.csv", index=False)

    total_rows, dropped, n_users = write_interactions_parquet(
        ratings_path,
        catalog,
        out_dir / "interactions.parquet",
    )
    if dropped:
        log.warning("Dropped %s ratings with unknown anime_id", dropped)

    log.info(
        "Wrote %s interactions, %s anime, %s users → %s",
        total_rows,
        len(catalog),
        n_users,
        out_dir,
    )


if __name__ == "__main__":
    main()
