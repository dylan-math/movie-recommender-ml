"""Disk persistence for recommender/worker user-vector replicas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PersistedUserVector:
    user_id: str
    vector: np.ndarray
    stale: bool
    updated_at: str


def save_user_vectors(
    state_dir: Path,
    *,
    model_version: str,
    records: dict[str, PersistedUserVector],
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    user_ids = sorted(records.keys())
    if not user_ids:
        return
    vectors = np.stack([records[uid].vector for uid in user_ids]).astype(np.float32)
    np.save(state_dir / "user_vectors.npy", vectors)
    meta = {
        "model_version": model_version,
        "user_ids": user_ids,
        "stale": {uid: records[uid].stale for uid in user_ids},
        "updated_at": {uid: records[uid].updated_at for uid in user_ids},
    }
    with open(state_dir / "user_vectors_meta.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)


def load_user_vectors(state_dir: Path) -> tuple[str | None, dict[str, PersistedUserVector]]:
    meta_path = state_dir / "user_vectors_meta.json"
    vectors_path = state_dir / "user_vectors.npy"
    if not meta_path.exists() or not vectors_path.exists():
        return None, {}

    with open(meta_path, encoding="utf-8") as file:
        meta = json.load(file)
    model_version = str(meta.get("model_version", ""))
    user_ids = [str(uid) for uid in meta["user_ids"]]
    vectors = np.load(vectors_path)
    stale_map = meta.get("stale", {})
    updated_map = meta.get("updated_at", {})

    records: dict[str, PersistedUserVector] = {}
    for idx, uid in enumerate(user_ids):
        records[uid] = PersistedUserVector(
            user_id=uid,
            vector=vectors[idx],
            stale=bool(stale_map.get(uid, False)),
            updated_at=str(updated_map.get(uid, "")),
        )
    return model_version, records


def save_active_model_pointer(state_dir: Path, *, artifact_dir: str, model_version: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {"artifact_dir": artifact_dir, "model_version": model_version}
    with open(state_dir / "active_model.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_active_model_pointer(state_dir: Path) -> dict[str, str] | None:
    path = state_dir / "active_model.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_persisted_user_embedding_version(state_dir: Path) -> str | None:
    meta_path = state_dir / "user_vectors_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as file:
        meta = json.load(file)
    version = meta.get("model_version")
    return str(version) if version else None
