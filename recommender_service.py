"""Recommender service with local user-vector replica (push model)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from als_runtime import ArtifactBundle, load_artifact_bundle
from service_persistence import (
    PersistedUserVector,
    load_active_model_pointer,
    load_persisted_user_embedding_version,
    load_user_vectors,
    save_active_model_pointer,
    save_user_vectors,
)

log = logging.getLogger("recommender-service")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecommendRequest(BaseModel):
    user_id: int | str
    n: int = Field(default=10, ge=1, le=100)
    seen_item_ids: list[str] = Field(default_factory=list)


class RecommendItem(BaseModel):
    item_id: str
    score: float


class RecommendResponse(BaseModel):
    items: list[RecommendItem]


class SyncUserVector(BaseModel):
    user_id: int | str
    user_vector: list[float]
    stale: bool = False
    updated_at: str | None = None


class UsersSyncRequest(BaseModel):
    model_version: str
    users: list[SyncUserVector]


class ActivateRequest(BaseModel):
    artifact_dir: str | None = None
    reset_user_vectors: bool = False


@dataclass
class UserVectorRecord:
    vector: np.ndarray
    stale: bool
    updated_at: str


class RecommenderState:
    def __init__(self) -> None:
        self._lock = RLock()
        self.bundle: ArtifactBundle | None = None
        self.user_vectors: dict[str, UserVectorRecord] = {}
        self.user_embedding_version: str | None = None
        self.state_dir = Path(os.getenv("RECOM_STATE_DIR", "data/recommender_state"))

    def load_bundle(self, artifact_dir: str | Path | None = None, *, reset_vectors: bool = True) -> ArtifactBundle:
        bundle = load_artifact_bundle(artifact_dir)
        with self._lock:
            self.bundle = bundle
            if reset_vectors:
                self.user_vectors = {}
                self.user_embedding_version = None
        save_active_model_pointer(
            self.state_dir,
            artifact_dir=str(bundle.artifact_dir),
            model_version=bundle.model_version,
        )
        return bundle

    def _embedding_versions_unlocked(self) -> dict[str, Any]:
        bundle = self.bundle
        item_version = None if bundle is None else bundle.model_version
        in_memory_user_version = self.user_embedding_version
        user_count = len(self.user_vectors)
        stale_count = sum(1 for rec in self.user_vectors.values() if rec.stale)
        persisted_user_version = load_persisted_user_embedding_version(self.state_dir)
        effective_user_version = in_memory_user_version or persisted_user_version
        aligned = (
            item_version is not None
            and effective_user_version is not None
            and item_version == effective_user_version
        )
        return {
            "item_embedding_version": item_version,
            "user_embedding_version": effective_user_version,
            "user_embedding_version_in_memory": in_memory_user_version,
            "user_embedding_version_persisted": persisted_user_version,
            "versions_aligned": aligned,
            "user_vectors_count": user_count,
            "user_vectors_stale_count": stale_count,
        }

    def embedding_versions(self) -> dict[str, Any]:
        """Item (Y) and user (x_u replica) versions — may diverge during activate vs bulk push."""
        with self._lock:
            return self._embedding_versions_unlocked()

    def persist_user_vectors(self) -> None:
        with self._lock:
            bundle = self.bundle
            if bundle is None:
                return
            records = {
                uid: PersistedUserVector(
                    user_id=uid,
                    vector=rec.vector,
                    stale=rec.stale,
                    updated_at=rec.updated_at,
                )
                for uid, rec in self.user_vectors.items()
            }
        save_user_vectors(self.state_dir, model_version=bundle.model_version, records=records)

    def restore_user_vectors_from_disk(self) -> int:
        model_version, records = load_user_vectors(self.state_dir)
        if not records:
            return 0
        with self._lock:
            bundle = self.bundle
            if bundle is None or model_version != bundle.model_version:
                return 0
            for uid, persisted in records.items():
                self.user_vectors[uid] = UserVectorRecord(
                    vector=persisted.vector,
                    stale=persisted.stale,
                    updated_at=persisted.updated_at,
                )
            self.user_embedding_version = model_version
        return len(records)

    def upsert_users(self, model_version: str, users: list[SyncUserVector]) -> tuple[int, int]:
        with self._lock:
            if self.bundle is None:
                raise RuntimeError("Artifacts are not loaded.")
            if model_version != self.bundle.model_version:
                return 0, len(users)

            applied = 0
            rejected = 0
            expected_dim = self.bundle.factors
            for payload in users:
                if len(payload.user_vector) != expected_dim:
                    rejected += 1
                    continue
                self.user_vectors[str(payload.user_id)] = UserVectorRecord(
                    vector=np.array(payload.user_vector, dtype=np.float32),
                    stale=payload.stale,
                    updated_at=payload.updated_at or utc_now_iso(),
                )
                applied += 1
            if applied > 0:
                self.user_embedding_version = model_version
        self.persist_user_vectors()
        return applied, rejected

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        with self._lock:
            bundle = self.bundle
            if bundle is None:
                raise RuntimeError("Artifacts are not loaded.")
            record = self.user_vectors.get(str(request.user_id))
            if record is None:
                raise KeyError(f"Missing user vector for user_id={request.user_id}")

            scores = bundle.global_mean + bundle.item_factors @ record.vector
            scores = scores.copy()
            seen_indices = [
                bundle.item_id_to_idx[str(item_id)]
                for item_id in request.seen_item_ids
                if str(item_id) in bundle.item_id_to_idx
            ]
            if seen_indices:
                scores[seen_indices] = -np.inf

            candidate_count = min(request.n, scores.shape[0])
            if candidate_count <= 0:
                return RecommendResponse(items=[])

            top_idx = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            items = [
                RecommendItem(
                    item_id=str(bundle.movie_ids[idx]),
                    score=round(float(scores[idx]), 2),
                )
                for idx in top_idx
            ]
            return RecommendResponse(items=items)


app = FastAPI(title="recommender-service")
state = RecommenderState()

WORKER_URL = os.getenv("WORKER_URL", "http://localhost:8002")
SYNC_FROM_WORKER_ON_START = os.getenv("RECOM_SYNC_FROM_WORKER_ON_START", "1").lower() in (
    "1",
    "true",
    "yes",
)


async def sync_from_worker() -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{WORKER_URL.rstrip('/')}/v1/internal/recommender/resync",
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()


@app.on_event("startup")
async def startup() -> None:
    artifact_dir = os.getenv("RECOM_ARTIFACT_DIR")
    pointer = load_active_model_pointer(state.state_dir)
    if artifact_dir is None and pointer is not None:
        artifact_dir = pointer.get("artifact_dir")
    state.load_bundle(artifact_dir, reset_vectors=False)
    restored = state.restore_user_vectors_from_disk()
    log.info("Restored %s user vectors from disk", restored)

    if SYNC_FROM_WORKER_ON_START and (restored == 0 or os.getenv("RECOM_FORCE_WORKER_SYNC", "0") == "1"):
        try:
            payload = await sync_from_worker()
            log.info("Synced from worker: %s", payload)
        except Exception as exc:
            log.warning("Worker sync on startup failed: %s", exc)


@app.get("/health")
def health() -> dict[str, Any]:
    bundle = state.bundle
    versions = state.embedding_versions()
    return {
        "status": "ok",
        "model_version": None if bundle is None else bundle.model_version,
        "artifact_dir": None if bundle is None else str(bundle.artifact_dir),
        "state_dir": str(state.state_dir),
        **versions,
    }


@app.get("/v1/internal/model")
def get_model() -> dict[str, Any]:
    bundle = state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_version": bundle.model_version,
        "artifact_dir": str(bundle.artifact_dir),
        "factors": bundle.factors,
        "global_mean": bundle.global_mean,
        "regularization": bundle.regularization,
        "n_items": int(bundle.item_factors.shape[0]),
        **state.embedding_versions(),
    }


@app.post("/v1/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    try:
        return state.recommend(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/internal/users/sync")
def sync_users(request: UsersSyncRequest) -> dict[str, Any]:
    try:
        applied, rejected = state.upsert_users(request.model_version, request.users)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if state.bundle is not None and request.model_version != state.bundle.model_version:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "model_version_mismatch",
                "expected": state.bundle.model_version,
                "got": request.model_version,
                "applied": applied,
                "rejected": rejected,
            },
        )
    return {
        "applied": applied,
        "rejected": rejected,
        "model_version": request.model_version,
    }


@app.post("/v1/admin/model/activate")
def activate_model(request: ActivateRequest) -> dict[str, Any]:
    bundle = state.load_bundle(request.artifact_dir, reset_vectors=request.reset_user_vectors)
    return {
        "status": "activated",
        "model_version": bundle.model_version,
        "artifact_dir": str(bundle.artifact_dir),
        "item_id_format": bundle.item_id_format,
        "n_items": len(bundle.item_id_to_idx),
        "reset_user_vectors": request.reset_user_vectors,
    }


@app.post("/v1/admin/sync-from-worker")
async def admin_sync_from_worker() -> dict[str, Any]:
    return await sync_from_worker()
