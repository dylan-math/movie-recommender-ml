"""Train-runner for phase A (MovieLens interactions only)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from als_runtime import (
    load_artifact_bundle,
    load_user_ids_npy,
    user_ids_to_storage_array,
)
from als_trainer import TrainResult, train_als_explicit

DEFAULT_BACKEND = os.environ.get("TRAIN_BACKEND", "cupy")
DEFAULT_USE_GPU = os.environ.get("TRAIN_USE_GPU", "1").lower() in ("1", "true", "yes")

DEFAULT_INTERACTIONS_PATH = Path(__file__).resolve().parent / "train_data" / "movielens" / "interactions.parquet"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "artifacts" / "registry"
DEFAULT_MOVIES_CSV = Path(__file__).resolve().parent / "train_data" / "movielens" / "movies.csv"


@dataclass
class TrainConfig:
    backend: str = DEFAULT_BACKEND  # cupy | naive
    use_gpu: bool = DEFAULT_USE_GPU
    factors: int = 64
    regularization: float | None = None
    iterations: int = 10
    train_window_mode: str = "all"
    train_window_days: int | None = None


def _default_regularization(backend: str) -> float:
    return 10.0


def _run_training(
    *,
    interactions: sparse.csr_matrix,
    config: TrainConfig,
) -> TrainResult:
    reg = config.regularization if config.regularization is not None else _default_regularization(config.backend)
    if config.backend == "cupy":
        from als_cupy_trainer import train_als_cupy

        return train_als_cupy(
            interactions=interactions,
            factors=config.factors,
            regularization=reg,
            iterations=config.iterations,
            use_gpu=config.use_gpu,
        )
    if config.backend == "naive":
        return train_als_explicit(
            interactions=interactions,
            factors=config.factors,
            regularization=reg,
            iterations=config.iterations,
        )
    raise ValueError(f"Unknown backend={config.backend!r}. Use cupy or naive.")


@dataclass
class TrainRunResult:
    model_version: str
    output_dir: Path
    factors: int
    regularization: float
    global_mean: float
    iterations: int
    n_users: int
    n_items: int
    n_interactions: int


def _read_interactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing interactions file: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    user_col = next((c for c in ("user_id", "userId", "user_idx") if c in frame.columns), None)
    item_col = next((c for c in ("item_id", "movieId", "movie_id", "item_idx") if c in frame.columns), None)
    rating_col = next((c for c in ("rating", "Rating", "score") if c in frame.columns), None)
    if user_col is None or item_col is None or rating_col is None:
        raise ValueError("Interactions must contain user, item and rating columns.")

    normalized = frame[[user_col, item_col, rating_col]].copy()
    normalized.columns = ["user_id", "item_id", "rating"]
    normalized["user_id"] = normalized["user_id"].astype(str)
    normalized["item_id"] = normalized["item_id"].astype(int)
    normalized["rating"] = normalized["rating"].astype(np.float32)
    return normalized


def _interactions_from_frame(frame: pd.DataFrame) -> tuple[sparse.csr_matrix, dict[str, int], dict[int, int], np.ndarray, np.ndarray]:
    user_ids = frame["user_id"].astype(str).unique()
    item_ids = np.sort(frame["item_id"].unique()).astype(np.int64)
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_id_to_idx = {int(mid): idx for idx, mid in enumerate(item_ids)}

    rows = frame["user_id"].map(user_id_to_idx).astype(np.int32).to_numpy()
    cols = frame["item_id"].map(movie_id_to_idx).astype(np.int32).to_numpy()
    data = frame["rating"].astype(np.float32).to_numpy()
    interactions = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return interactions, user_id_to_idx, movie_id_to_idx, item_ids, user_ids


def _interactions_from_artifacts(artifact_dir: Path) -> tuple[sparse.csr_matrix, dict[str, int], dict[int, int], np.ndarray, np.ndarray]:
    bundle = load_artifact_bundle(artifact_dir)
    user_items_path = artifact_dir / "user_items_centered.npz"
    user_ids_path = artifact_dir / "user_ids.npy"
    if not user_items_path.exists() or not user_ids_path.exists():
        raise FileNotFoundError(
            f"Missing training artifacts in {artifact_dir}. "
            "Provide train_data/movielens/interactions.parquet for phase A."
        )

    user_items = sparse.load_npz(user_items_path).tocsr()
    user_ids_arr = load_user_ids_npy(user_ids_path)
    user_id_to_idx = {str(uid): idx for idx, uid in enumerate(user_ids_arr)}

    centered = user_items.copy()
    centered.data = centered.data.astype(np.float32) + np.float32(bundle.global_mean)
    return centered, user_id_to_idx, bundle.movie_id_to_idx, bundle.movie_ids, user_ids_arr


def load_interactions_phase_a(
    *,
    interactions_path: Path | None = None,
    artifact_dir: Path | None = None,
) -> tuple[sparse.csr_matrix, dict[str, int], dict[int, int], np.ndarray, np.ndarray]:
    """
    Load phase-A interactions.

    Priority:
    1) explicit interactions parquet/csv
    2) fallback: reconstruct from artifacts/user_items_centered.npz + global_mean
    """
    if interactions_path is not None:
        frame = _normalize_interactions(_read_interactions(Path(interactions_path)))
        return _interactions_from_frame(frame)

    path = DEFAULT_INTERACTIONS_PATH
    if path.exists():
        frame = _normalize_interactions(_read_interactions(path))
        return _interactions_from_frame(frame)

    if artifact_dir is not None and Path(artifact_dir).exists():
        return _interactions_from_artifacts(Path(artifact_dir))

    raise FileNotFoundError(
        "Phase A needs train_data/movielens/interactions.parquet "
        "(run offline/scripts/build_movielens_interactions.py) or a full artifact bundle."
    )


def _build_movie_metadata(movie_ids: np.ndarray, movies_csv: Path | None = None) -> pd.DataFrame:
    titles: dict[int, str] = {}
    genres_map: dict[int, str] = {}
    if movies_csv is not None and movies_csv.exists():
        movies = pd.read_csv(movies_csv)
        id_col = "movieId" if "movieId" in movies.columns else "movie_id"
        title_col = "title" if "title" in movies.columns else "name"
        genres_col = "genres" if "genres" in movies.columns else None
        for row in movies.itertuples(index=False):
            movie_id = int(getattr(row, id_col))
            titles[movie_id] = str(getattr(row, title_col))
            if genres_col is not None:
                genres_map[movie_id] = str(getattr(row, genres_col))

    return pd.DataFrame(
        {
            "movie_idx": np.arange(len(movie_ids), dtype=np.int32),
            "movieId": movie_ids.astype(np.int64),
            "title": [titles.get(int(mid), f"movie-{mid}") for mid in movie_ids],
            "genres": [genres_map.get(int(mid), "unknown") for mid in movie_ids],
        }
    )


def save_bundle(
    *,
    output_dir: Path,
    train_result: TrainResult,
    interactions: sparse.csr_matrix,
    movie_ids: np.ndarray,
    user_ids: np.ndarray,
    model_version: str,
    config: TrainConfig,
    movies_csv: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "item_factors.npy", train_result.item_factors)
    np.save(output_dir / "user_factors.npy", train_result.user_factors)
    np.save(output_dir / "movie_ids.npy", movie_ids)
    np.save(output_dir / "user_ids.npy", user_ids_to_storage_array(user_ids))

    centered = interactions.copy()
    centered.data = centered.data.astype(np.float32) - np.float32(train_result.global_mean)
    sparse.save_npz(output_dir / "user_items_centered.npz", centered.tocsr())

    metadata = _build_movie_metadata(movie_ids, movies_csv)
    metadata.to_csv(output_dir / "movie_metadata.csv", index=False)

    reg = config.regularization if config.regularization is not None else _default_regularization(config.backend)
    config_payload = {
        "model_type": "explicit_als",
        "trainer_backend": config.backend,
        "train_use_gpu": bool(config.use_gpu),
        "factors": int(config.factors),
        "regularization": float(reg),
        "iterations": int(config.iterations),
        "global_mean": float(train_result.global_mean),
        "model_version": model_version,
        "data_source": "movie_lens",
        "train_window_mode": config.train_window_mode,
        "train_window_days": config.train_window_days,
        "max_ratings": None,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)

    title_to_idx = {str(title).casefold(): int(idx) for idx, title in enumerate(metadata["title"])}
    with open(output_dir / "title_to_idx.json", "w", encoding="utf-8") as file:
        json.dump(title_to_idx, file, ensure_ascii=False)

    return output_dir


def run_training_phase_a(
    *,
    interactions_path: Path | None = None,
    artifact_dir: Path | None = None,
    output_root: Path | None = None,
    output_dir: Path | None = None,
    config: TrainConfig | None = None,
    movies_csv: Path | None = None,
) -> TrainRunResult:
    cfg = config or TrainConfig()
    interactions, _user_id_to_idx, _movie_id_to_idx, movie_ids, user_ids = load_interactions_phase_a(
        interactions_path=interactions_path,
        artifact_dir=artifact_dir,
    )

    result = _run_training(interactions=interactions, config=cfg)

    out_root = output_root or DEFAULT_OUTPUT_ROOT
    if output_dir is not None:
        out_dir = Path(output_dir)
        model_version = out_dir.name
    else:
        model_version = f"snap-phase-a-{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        out_dir = out_root / model_version
    save_bundle(
        output_dir=out_dir,
        train_result=result,
        interactions=interactions,
        movie_ids=movie_ids,
        user_ids=user_ids,
        model_version=model_version,
        config=cfg,
        movies_csv=movies_csv or DEFAULT_MOVIES_CSV,
    )

    return TrainRunResult(
        model_version=model_version,
        output_dir=out_dir,
        factors=cfg.factors,
        regularization=cfg.regularization if cfg.regularization is not None else _default_regularization(cfg.backend),
        global_mean=result.global_mean,
        iterations=result.iterations,
        n_users=interactions.shape[0],
        n_items=interactions.shape[1],
        n_interactions=int(interactions.nnz),
    )
