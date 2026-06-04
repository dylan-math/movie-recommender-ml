"""Load train_config.yaml for Worker retrain and offline train scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from trainer_runner import DEFAULT_BACKEND, DEFAULT_USE_GPU, TrainConfig

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CONFIG_PATH = REPO_ROOT / "train_config.yaml"


@dataclass(frozen=True)
class TrainConfigFile:
    path: Path
    backend: str
    use_gpu: bool
    factors: int
    regularization: float
    iterations: int
    train_window_mode: str
    train_window_days: int | None
    cv_source: str | None = None
    cv_best_name: str | None = None


def resolve_train_config_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.getenv("TRAIN_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_TRAIN_CONFIG_PATH


def load_train_config_file(path: Path | str | None = None) -> TrainConfigFile:
    config_path = resolve_train_config_path(path)
    with open(config_path, encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    als = raw.get("als") or {}
    window = raw.get("train_window") or {}
    mode = str(window.get("mode", "all"))
    sliding_days = window.get("sliding_days")
    train_window_days: int | None
    if sliding_days is None:
        train_window_days = None
    else:
        train_window_days = int(sliding_days)

    return TrainConfigFile(
        path=config_path,
        backend=str(als.get("backend", DEFAULT_BACKEND)),
        use_gpu=bool(als.get("use_gpu", DEFAULT_USE_GPU)),
        factors=int(als.get("factors", 64)),
        regularization=float(als.get("regularization", 10.0)),
        iterations=int(als.get("iterations", 10)),
        train_window_mode=mode,
        train_window_days=train_window_days,
        cv_source=raw.get("cv_source"),
        cv_best_name=raw.get("cv_best_name"),
    )


def train_config_from_file(path: Path | str | None = None) -> TrainConfig:
    cfg = load_train_config_file(path)
    return TrainConfig(
        backend=cfg.backend,
        use_gpu=cfg.use_gpu,
        factors=cfg.factors,
        regularization=cfg.regularization,
        iterations=cfg.iterations,
        train_window_mode=cfg.train_window_mode,
        train_window_days=cfg.train_window_days,
    )
