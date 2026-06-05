"""Recommender service with local user-vector replica (push model)."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from artifact_resolve import resolve_artifact_dir
from plotwise_catalog import PlotwiseItemCatalog, load_plotwise_catalog
from service_persistence import (
    PersistedUserVector,
    load_active_model_pointer,
    load_persisted_user_embedding_version,
    load_user_vectors,
    save_active_model_pointer,
    save_user_vectors,
)

log = logging.getLogger("recommender-service")


def _configure_service_logging() -> None:
    """Ensure app logs appear in docker compose logs (uvicorn does not configure this logger)."""
    if log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


_configure_service_logging()

_current_http_request: contextvars.ContextVar[Request | None] = contextvars.ContextVar(
    "current_http_request",
    default=None,
)


class UserNotReadyError(Exception):
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        super().__init__(user_id)


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


class SyncItemVector(BaseModel):
    item_id: str
    item_vector: list[float]


class ItemsAppendRequest(BaseModel):
    model_version: str
    items: list[SyncItemVector]


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
        self.catalog: PlotwiseItemCatalog | None = None
        self.user_vectors: dict[str, UserVectorRecord] = {}
        self.user_embedding_version: str | None = None
        self.state_dir = Path(os.getenv("RECOM_STATE_DIR", "data/recommender_state"))

    def load_catalog(self, artifact_dir: str | Path | None = None, *, reset_vectors: bool = True) -> PlotwiseItemCatalog:
        catalog = load_plotwise_catalog(artifact_dir)
        with self._lock:
            self.catalog = catalog
            if reset_vectors:
                self.user_vectors = {}
                self.user_embedding_version = None
        save_active_model_pointer(
            self.state_dir,
            artifact_dir=str(catalog.artifact_dir),
            model_version=catalog.model_version,
        )
        return catalog

    def _embedding_versions_unlocked(self) -> dict[str, Any]:
        catalog = self.catalog
        item_version = None if catalog is None else catalog.model_version
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
            catalog = self.catalog
            if catalog is None:
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
            model_version = catalog.model_version
        save_user_vectors(self.state_dir, model_version=model_version, records=records)

    def restore_user_vectors_from_disk(self) -> int:
        model_version, records = load_user_vectors(self.state_dir)
        if not records:
            return 0
        with self._lock:
            catalog = self.catalog
            if catalog is None or model_version != catalog.model_version:
                return 0
            for uid, persisted in records.items():
                self.user_vectors[uid] = UserVectorRecord(
                    vector=persisted.vector,
                    stale=persisted.stale,
                    updated_at=persisted.updated_at,
                )
            self.user_embedding_version = model_version
        return len(records)

    def append_items(self, model_version: str, items: list[SyncItemVector]) -> tuple[int, int]:
        with self._lock:
            if self.catalog is None:
                raise RuntimeError("Plotwise item catalog is not loaded.")
            if model_version != self.catalog.model_version:
                return 0, len(items)
            applied = 0
            rejected = 0
            for payload in items:
                try:
                    self.catalog.append_item_factor(payload.item_id, np.array(payload.item_vector, dtype=np.float32))
                    applied += 1
                except ValueError:
                    rejected += 1
        if applied > 0:
            self.catalog.persist()
        return applied, rejected

    def upsert_users(self, model_version: str, users: list[SyncUserVector]) -> tuple[int, int]:
        with self._lock:
            if self.catalog is None:
                raise RuntimeError("Plotwise item catalog is not loaded.")
            if model_version != self.catalog.model_version:
                return 0, len(users)

            applied = 0
            rejected = 0
            expected_dim = self.catalog.factors
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
            catalog = self.catalog
            if catalog is None:
                raise RuntimeError("Plotwise item catalog is not loaded.")
            record = self.user_vectors.get(str(request.user_id))
            if record is None:
                raise UserNotReadyError(str(request.user_id))

            scores = catalog.global_mean + catalog.item_factors @ record.vector
            scores = scores.copy()
            seen_indices = [
                catalog.item_id_to_idx[str(item_id)]
                for item_id in request.seen_item_ids
                if str(item_id) in catalog.item_id_to_idx
            ]
            if seen_indices:
                scores[seen_indices] = -np.inf

            ranked_idx = np.argsort(-scores)
            items: list[RecommendItem] = []
            for idx in ranked_idx:
                if len(items) >= request.n:
                    break
                public_id = catalog.public_item_id_at(int(idx))
                if public_id is None:
                    continue
                items.append(
                    RecommendItem(
                        item_id=public_id,
                        score=round(float(scores[idx]), 2),
                    )
                )
            return RecommendResponse(items=items)


app = FastAPI(title="recommender-service")
state = RecommenderState()


def _user_id_from_validation(exc: RequestValidationError) -> str:
    body = exc.body
    if isinstance(body, dict) and body.get("user_id") is not None:
        return str(body["user_id"])
    if isinstance(body, bytes):
        try:
            import json

            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("user_id") is not None:
                return str(parsed["user_id"])
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return "?"


def _log_empty_recommend(user_id: str, code: str, message: str, **extra: Any) -> None:
    """Log why POST /v1/recommend returns an empty items list (client still gets 200)."""
    suffix = f" {extra}" if extra else ""
    log.warning(
        "POST /v1/recommend: empty items user_id=%s [%s] %s%s",
        user_id,
        code,
        message,
        suffix,
    )


EMPTY_RECOMMEND_RESPONSE = RecommendResponse(items=[])


@app.middleware("http")
async def bind_http_request(request: Request, call_next):
    token = _current_http_request.set(request)
    try:
        return await call_next(request)
    finally:
        _current_http_request.reset(token)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path.rstrip("/") == "/v1/recommend":
        log.warning(
            "POST /v1/recommend: invalid request user_id=%s errors=%s",
            _user_id_from_validation(exc),
            exc.errors(),
        )
    return await request_validation_exception_handler(request, exc)

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
    artifact_path = await asyncio.to_thread(
        resolve_artifact_dir,
        env_dir=os.getenv("RECOM_ARTIFACT_DIR"),
        allow_bootstrap=False,
    )
    catalog = state.load_catalog(str(artifact_path), reset_vectors=False)
    log.info(
        "Plotwise item data: %s base items, %s overlay items",
        catalog.base.item_factors.shape[0],
        catalog.overlay_count,
    )
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
    catalog = state.catalog
    versions = state.embedding_versions()
    return {
        "status": "ok",
        "model_version": None if catalog is None else catalog.model_version,
        "artifact_dir": None if catalog is None else str(catalog.artifact_dir),
        "plotwise_overlay_items": None if catalog is None else catalog.overlay_count,
        "state_dir": str(state.state_dir),
        **versions,
    }


@app.get("/v1/internal/model")
def get_model() -> dict[str, Any]:
    catalog = state.catalog
    if catalog is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_version": catalog.model_version,
        "artifact_dir": str(catalog.artifact_dir),
        "factors": catalog.factors,
        "global_mean": catalog.global_mean,
        "regularization": catalog.regularization,
        "n_items": int(catalog.item_factors.shape[0]),
        "plotwise_overlay_items": catalog.overlay_count,
        **state.embedding_versions(),
    }


WORKER_REQUEST_TIMEOUT = float(os.getenv("RECOM_WORKER_REQUEST_TIMEOUT", "60"))
WORKER_CONNECT_RETRIES = max(1, int(os.getenv("RECOM_WORKER_CONNECT_RETRIES", "15")))
WORKER_CONNECT_RETRY_DELAY = float(os.getenv("RECOM_WORKER_CONNECT_RETRY_DELAY", "1.0"))


def _recommender_vector_updated_at(user_id: str) -> str | None:
    with state._lock:
        record = state.user_vectors.get(user_id)
        return None if record is None else record.updated_at


async def ensure_user_vector_via_worker(user_id: str) -> bool:
    """Ask Worker to ensure vector is current and sync into this Recommender.

    Returns True when the user vector is ready in this service; False otherwise
    (reason logged — caller should return an empty recommend list).
    """
    base = WORKER_URL.rstrip("/")
    timeout = httpx.Timeout(WORKER_REQUEST_TIMEOUT)
    last_error: str | None = None
    recommender_updated_at = _recommender_vector_updated_at(user_id)

    for attempt in range(1, WORKER_CONNECT_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                readiness = await client.get(f"{base}/v1/internal/users/{user_id}/recommend-readiness")
                readiness.raise_for_status()
                snapshot = readiness.json()
                if not snapshot.get("has_ratings"):
                    _log_empty_recommend(
                        user_id,
                        "no_ratings",
                        "нет оценок",
                        rating_count=int(snapshot.get("rating_count") or 0),
                    )
                    return False

                refresh = await client.post(
                    f"{base}/v1/internal/users/{user_id}/refresh-now",
                    json={"recommender_updated_at": recommender_updated_at},
                )
                refresh.raise_for_status()
                result = refresh.json()
        except httpx.HTTPStatusError as exc:
            _log_empty_recommend(
                user_id,
                "worker_error",
                f"worker HTTP {exc.response.status_code}",
            )
            return False
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_error = str(exc)
            log.warning(
                "Worker unreachable attempt %s/%s user_id=%s: %s",
                attempt,
                WORKER_CONNECT_RETRIES,
                user_id,
                exc,
            )
            if attempt < WORKER_CONNECT_RETRIES:
                await asyncio.sleep(WORKER_CONNECT_RETRY_DELAY)
                continue
            _log_empty_recommend(user_id, "worker_unavailable", last_error)
            return False
        except httpx.TimeoutException as exc:
            last_error = str(exc)
            log.warning("Worker timeout user_id=%s: %s", user_id, exc)
            _log_empty_recommend(user_id, "worker_unavailable", last_error)
            return False

        refresh_status = str(result.get("refresh_status") or "")
        if refresh_status == "no_ratings":
            _log_empty_recommend(
                user_id,
                "no_ratings",
                "нет оценок",
                rating_count=int(result.get("rating_count") or 0),
            )
            return False
        if refresh_status not in ("ok", "pushed", "unchanged") or not result.get("has_vector"):
            _log_empty_recommend(
                user_id,
                "no_usable_ratings",
                "оценки есть, но ни одна не в каталоге",
                rating_count=int(result.get("rating_count") or 0),
                refresh_status=refresh_status,
            )
            return False
        log.info(
            "Worker refresh-now completed for user_id=%s status=%s",
            user_id,
            refresh_status,
        )
        return True

    _log_empty_recommend(user_id, "worker_unavailable", last_error or "unknown")
    return False


@app.post("/v1/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest, http_request: Request) -> RecommendResponse:
    user_id = str(request.user_id)
    http_request.state.recommend_user_id = user_id
    if not await ensure_user_vector_via_worker(user_id):
        return EMPTY_RECOMMEND_RESPONSE
    try:
        return state.recommend(request)
    except UserNotReadyError:
        _log_empty_recommend(
            user_id,
            "vector_sync_failed",
            "worker отработал, но вектор в recommender не появился",
        )
        return EMPTY_RECOMMEND_RESPONSE
    except RuntimeError as exc:
        _log_empty_recommend(user_id, "model_unavailable", str(exc))
        return EMPTY_RECOMMEND_RESPONSE


@app.post("/v1/internal/users/sync")
def sync_users(request: UsersSyncRequest) -> dict[str, Any]:
    try:
        applied, rejected = state.upsert_users(request.model_version, request.users)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if state.catalog is not None and request.model_version != state.catalog.model_version:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "model_version_mismatch",
                "expected": state.catalog.model_version,
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


@app.post("/v1/internal/items/append")
def append_items(request: ItemsAppendRequest) -> dict[str, Any]:
    try:
        applied, rejected = state.append_items(request.model_version, request.items)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if state.catalog is not None and request.model_version != state.catalog.model_version:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "model_version_mismatch",
                "expected": state.catalog.model_version,
                "got": request.model_version,
                "applied": applied,
                "rejected": rejected,
            },
        )
    return {
        "applied": applied,
        "rejected": rejected,
        "model_version": request.model_version,
        "n_items": len(state.catalog.item_id_to_idx) if state.catalog else 0,
    }


@app.post("/v1/admin/model/activate")
def activate_model(request: ActivateRequest) -> dict[str, Any]:
    catalog = state.load_catalog(request.artifact_dir, reset_vectors=request.reset_user_vectors)
    return {
        "status": "activated",
        "model_version": catalog.model_version,
        "artifact_dir": str(catalog.artifact_dir),
        "item_id_format": catalog.item_id_format,
        "n_items": len(catalog.item_id_to_idx),
        "plotwise_overlay_items": catalog.overlay_count,
        "reset_user_vectors": request.reset_user_vectors,
    }


@app.post("/v1/admin/sync-from-worker")
async def admin_sync_from_worker() -> dict[str, Any]:
    return await sync_from_worker()
