#!/usr/bin/env python3
"""Merge MovieLens + animedata interactions into one parquet for retrain."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("merge-train-interactions")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MOVIELENS = REPO_ROOT / "train_data" / "movielens" / "interactions.parquet"
DEFAULT_ANIME = REPO_ROOT / "train_data" / "animedata" / "processed" / "interactions.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "train_data" / "combined" / "interactions.parquet"


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"user_id", "item_id", "rating"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: need columns {required}, got {list(frame.columns)}")
    out = frame[list(required)].copy()
    if "timestamp" in frame.columns:
        out["timestamp"] = frame["timestamp"]
    else:
        out["timestamp"] = 0
    out["user_id"] = out["user_id"].astype(str)
    out["item_id"] = out["item_id"].astype("int32")
    out["rating"] = out["rating"].astype("float32")
    out["timestamp"] = out["timestamp"].astype("int64")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Concat movielens + animedata interactions")
    parser.add_argument("--movielens", type=Path, default=DEFAULT_MOVIELENS)
    parser.add_argument("--anime", type=Path, default=DEFAULT_ANIME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anime-only", action="store_true", help="skip movielens (anime path only)")
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    if not args.anime_only:
        frames.append(_load(args.movielens))
    if args.anime.is_file():
        frames.append(_load(args.anime))
    elif not args.anime_only:
        log.warning("No anime interactions at %s — writing MovieLens only", args.anime)
    else:
        raise SystemExit(f"Missing {args.anime}")

    merged = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    log.info(
        "Wrote %s rows (%s users, %s items) → %s",
        len(merged),
        merged["user_id"].nunique(),
        merged["item_id"].nunique(),
        args.output,
    )


if __name__ == "__main__":
    main()
