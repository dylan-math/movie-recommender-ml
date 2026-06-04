"""Shared ALS runtime utilities for recommender and worker services."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parent / "artifacts" / "registry" / "snap-phase-a-20260602T225455Z"
)


def user_ids_to_storage_array(user_ids: np.ndarray) -> np.ndarray:
    """Save user ids without pickle (NumPy 2+ safe). MovieLens ids → int64."""
    flat = np.asarray(user_ids).reshape(-1)
    if flat.dtype == object:
        flat = np.array([str(value) for value in flat.tolist()], dtype=np.str_)
    if flat.dtype.kind in ("U", "S"):
        try:
            return flat.astype(np.int64)
        except ValueError:
            return flat.astype("U32")
    if np.issubdtype(flat.dtype, np.integer):
        return flat.astype(np.int64)
    return np.array([str(value) for value in flat.tolist()], dtype="U32")


def load_user_ids_npy(path: Path | str) -> np.ndarray:
    """Load user_ids.npy written as int64/unicode or legacy object arrays."""
    target = Path(path)
    try:
        array = np.load(target)
    except ValueError as exc:
        if "allow_pickle" not in str(exc):
            raise
        array = np.load(target, allow_pickle=True)
    flat = np.asarray(array).reshape(-1)
    if flat.dtype == object:
        flat = np.array([str(value) for value in flat.tolist()], dtype=np.str_)
    return flat


def user_id_key(user_id: object) -> str:
    if isinstance(user_id, (np.integer, int)):
        return str(int(user_id))
    return str(user_id)


@dataclass
class ArtifactBundle:
    artifact_dir: Path
    model_version: str
    factors: int
    regularization: float
    global_mean: float
    item_factors: np.ndarray
    movie_ids: np.ndarray
    movie_metadata: pd.DataFrame
    movie_id_to_idx: dict[int, int]


def load_artifact_bundle(artifact_dir: str | Path | None = None) -> ArtifactBundle:
    target_dir = Path(artifact_dir or DEFAULT_ARTIFACT_DIR)
    config_path = target_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)

    if config.get("model_type") != "explicit_als":
        raise ValueError(f"Unsupported model_type: {config.get('model_type')}")

    item_factors = np.load(target_dir / "item_factors.npy")
    movie_ids = np.load(target_dir / "movie_ids.npy")
    movie_metadata = pd.read_csv(target_dir / "movie_metadata.csv")
    if "movieId" not in movie_metadata.columns and "index" in movie_metadata.columns:
        movie_metadata = movie_metadata.rename(columns={"index": "movieId"})

    movie_id_to_idx = {int(movie_id): int(idx) for idx, movie_id in enumerate(movie_ids)}
    model_version = str(config.get("model_version") or config.get("trained_at") or target_dir.name)

    return ArtifactBundle(
        artifact_dir=target_dir,
        model_version=model_version,
        factors=int(config["factors"]),
        regularization=float(config["regularization"]),
        global_mean=float(config["global_mean"]),
        item_factors=item_factors.astype(np.float32),
        movie_ids=movie_ids,
        movie_metadata=movie_metadata,
        movie_id_to_idx=movie_id_to_idx,
    )


def fit_user_vector(
    *,
    item_factors: np.ndarray,
    global_mean: float,
    regularization: float,
    item_indices: np.ndarray,
    ratings_values: np.ndarray,
) -> np.ndarray:
    if item_indices.size == 0:
        raise ValueError("No known item indices to fit user vector.")
    selected_item_factors = item_factors[item_indices]
    centered_ratings = ratings_values.astype(np.float32) - float(global_mean)
    factor_count = item_factors.shape[1]
    lhs = selected_item_factors.T @ selected_item_factors
    lhs += float(regularization) * np.eye(factor_count, dtype=np.float32)
    rhs = selected_item_factors.T @ centered_ratings
    return np.linalg.solve(lhs, rhs).astype(np.float32)
