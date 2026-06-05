"""Resolve or bootstrap ALS artifact snaps under artifacts/registry/."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from service_persistence import load_active_model_pointer
from train_config_loader import train_config_from_file
from trainer_runner import TrainConfig, run_training_phase_a

log = logging.getLogger("artifact-resolve")

REGISTRY_ACTIVE_FILE = "active_model.json"
DEFAULT_REGISTRY = Path("artifacts/registry")


def snap_ready(artifact_dir: Path) -> bool:
    return (artifact_dir / "config.json").is_file()


def registry_dir() -> Path:
    """Canonical registry root — snaps are subdirectories; config.json lives inside a snap."""
    raw = os.getenv("RECOM_ARTIFACT_REGISTRY") or os.getenv("TRAIN_OUTPUT_ROOT") or str(DEFAULT_REGISTRY)
    return Path(raw)


def ensure_registry(registry: Path | None = None) -> Path:
    """Create registry root if missing (safe to call on every startup)."""
    root = registry or registry_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_ready_snaps(registry: Path | None = None) -> list[Path]:
    root = ensure_registry(registry)
    ready = [p for p in root.iterdir() if p.is_dir() and snap_ready(p)]
    return sorted(ready, key=lambda p: p.stat().st_mtime, reverse=True)


def save_registry_active_pointer(
    registry: Path,
    *,
    artifact_dir: str | Path,
    model_version: str,
) -> None:
    root = ensure_registry(registry)
    payload = {
        "artifact_dir": str(artifact_dir),
        "model_version": model_version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(root / REGISTRY_ACTIVE_FILE, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_registry_active_pointer(registry: Path | None = None) -> dict[str, str] | None:
    path = ensure_registry(registry) / REGISTRY_ACTIVE_FILE
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    artifact_dir = data.get("artifact_dir")
    if not artifact_dir:
        return None
    return {
        "artifact_dir": str(artifact_dir),
        "model_version": str(data.get("model_version") or Path(artifact_dir).name),
    }


def _pointer_dirs() -> list[Path]:
    dirs: list[Path] = []
    for key in ("WORKER_STATE_DIR", "RECOM_STATE_DIR"):
        raw = os.getenv(key)
        if raw:
            dirs.append(Path(raw))
    return dirs


def _optional_env_snap() -> Path | None:
    raw = (os.getenv("RECOM_ARTIFACT_DIR") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    registry = ensure_registry()
    if path == registry:
        return None
    return path


def resolve_artifact_dir(
    *,
    env_dir: str | Path | None = None,
    registry: Path | None = None,
    allow_bootstrap: bool = False,
) -> Path:
    """Pick active snap: optional env pin (if ready) → registry pointer → latest → bootstrap."""
    root = ensure_registry(registry)
    env_path = Path(env_dir) if env_dir else _optional_env_snap()

    if env_path is not None and snap_ready(env_path):
        return env_path

    candidates: list[Path] = []

    registry_pointer = load_registry_active_pointer(root)
    if registry_pointer is not None:
        candidates.append(Path(registry_pointer["artifact_dir"]))

    for state_dir in _pointer_dirs():
        pointer = load_active_model_pointer(state_dir)
        if pointer is not None and pointer.get("artifact_dir"):
            candidates.append(Path(str(pointer["artifact_dir"])))

    candidates.extend(list_ready_snaps(root))

    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if snap_ready(path):
            if env_path is not None:
                log.info(
                    "Using artifact snap %s (RECOM_ARTIFACT_DIR=%s not ready; see %s)",
                    path,
                    env_path,
                    root / REGISTRY_ACTIVE_FILE,
                )
            return path

    if allow_bootstrap:
        return bootstrap_artifact_snap(registry=root)

    hints = [
        f"ensure {root / REGISTRY_ACTIVE_FILE} points to a snap or enable WORKER_BOOTSTRAP_ON_START",
        f"place a bundle under {root}/<snap>/config.json",
        "optional: RECOM_ARTIFACT_DIR=<snap-path> when pinning a specific version",
    ]
    if env_path is not None:
        hints.insert(0, f"RECOM_ARTIFACT_DIR={env_path} is not ready")
    raise FileNotFoundError("No ALS artifact snap found. " + "; ".join(hints) + ".")


def bootstrap_artifact_snap(*, registry: Path | None = None) -> Path:
    """Train Phase-A MovieLens model and write a new snap under registry."""
    root = ensure_registry(registry)

    interactions_raw = os.getenv("TRAIN_INTERACTIONS_PATH")
    if not interactions_raw:
        raise FileNotFoundError(
            "Cannot bootstrap model: TRAIN_INTERACTIONS_PATH is not set "
            "(need MovieLens interactions.parquet)."
        )
    interactions_path = Path(interactions_raw)
    if not interactions_path.is_file():
        raise FileNotFoundError(
            f"Cannot bootstrap model: missing train data at {interactions_path}. "
            "Mount train_data/ or copy interactions.parquet."
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = root / f"snap-phase-a-{stamp}"
    log.warning(
        "No artifact snap found; bootstrap training → %s (this may take several minutes)",
        output_dir,
    )

    train_cfg = train_config_from_file()
    result = run_training_phase_a(
        interactions_path=interactions_path,
        output_root=root,
        output_dir=output_dir,
        config=train_cfg,
    )

    if not snap_ready(result.output_dir):
        raise RuntimeError(f"Bootstrap training finished but snap is incomplete: {result.output_dir}")

    save_registry_active_pointer(
        root,
        artifact_dir=result.output_dir,
        model_version=result.model_version,
    )
    log.info(
        "Bootstrap training completed: model_version=%s users=%s items=%s interactions=%s",
        result.model_version,
        result.n_users,
        result.n_items,
        result.n_interactions,
    )
    return result.output_dir
