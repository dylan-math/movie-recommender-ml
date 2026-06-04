#!/usr/bin/env python3
"""Build item_id_map.csv (plotwise tokens ↔ MovieLens movieId) from MovieLens links.csv."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from item_id_map_builder import MOVIELENS_LINKS_URLS, write_item_id_map

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build-item-id-map")

DEFAULT_SNAP = ROOT / "artifacts/registry/snap-phase-a-20260602T225455Z"
DEFAULT_MOVIELENS_DIR = ROOT / "train_data/movielens"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build plotwise item_id_map.csv from MovieLens links.")
    parser.add_argument(
        "--snap",
        type=Path,
        default=DEFAULT_SNAP,
        help="ALS snap dir (writes item_id_map.csv here by default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <snap>/item_id_map.csv)",
    )
    parser.add_argument("--links-csv", type=Path, help="Path to links.csv (skip zip)")
    parser.add_argument(
        "--dataset",
        choices=sorted(MOVIELENS_LINKS_URLS),
        default="32m",
        help="MovieLens zip to read links from if --links-csv omitted",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="MovieLens zip path (default: train_data/movielens/<dataset>.zip)",
    )
    args = parser.parse_args()

    out = args.output or (args.snap / "item_id_map.csv")
    zip_path = args.zip
    if args.links_csv is None:
        if zip_path is None:
            _, zip_name = MOVIELENS_LINKS_URLS[args.dataset]
            zip_path = DEFAULT_MOVIELENS_DIR / zip_name
        if not zip_path.exists():
            url, name = MOVIELENS_LINKS_URLS[args.dataset]
            raise SystemExit(
                f"Missing {zip_path}. Download MovieLens zip first, e.g.\n"
                f"  python3 offline/scripts/build_movielens_interactions.py --dataset {args.dataset}\n"
                f"  or wget {url} -O {DEFAULT_MOVIELENS_DIR / name}"
            )

    write_item_id_map(
        out,
        links_path=args.links_csv,
        zip_path=zip_path if args.links_csv is None else None,
        artifact_dir=args.snap,
    )
    log.info("Done: %s", out)


if __name__ == "__main__":
    main()
