"""Worker service: refresh queue + coalescing + user-vector push to recommender.

Reads user ratings from PostgreSQL (plotwise ``user_titles``). Pushes vectors to Recommender over HTTP.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from als_runtime import ArtifactBundle, fit_user_vector, load_user_ids_npy, user_id_key
from artifact_resolve import resolve_artifact_dir, save_registry_active_pointer
from plotwise_catalog import PlotwiseItemCatalog, load_plotwise_catalog
from service_persistence import (
    PersistedUserVector,
    load_active_model_pointer,
    load_user_vectors,
    save_active_model_pointer,
    save_user_vectors,
)
from train_config_loader import load_train_config_file, train_config_from_file
from trainer_runner import TrainConfig, run_training_phase_a
from worker_db import (
    database_url_masked,
    fetch_interaction_count,
    fetch_pending_users,
    fetch_user_ratings,
    load_external_item_map,
)

log = logging.getLogger("worker-service")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RefreshRequest(BaseModel):
    user_id: int | str


class RetrainHyperparameters(BaseModel):
    backend: str = "cupy"
    use_gpu: bool = True
    factors: int = 64
    regularization: float = 10.0
    iterations: int = 15


def default_retrain_hyperparameters() -> RetrainHyperparameters:
    cfg = load_train_config_file()
    return RetrainHyperparameters(
        backend=cfg.backend,
        use_gpu=cfg.use_gpu,
        factors=cfg.factors,
        regularization=cfg.regularization,
        iterations=cfg.iterations,
    )


class RetrainRequest(BaseModel):
    reason: str | None = None
    data_source: str = "movie_lens"
    artifact_dir: str | None = None
    interactions_path: str | None = None
    output_root: str | None = None
    hyperparameters: RetrainHyperparameters | None = None


class ActivateModelRequest(BaseModel):
    artifact_dir: str
    reset_user_vectors: bool = False


class JobStatus(BaseModel):
    job_id: str
    status: str
    detail: str | None = None
    model_version: str | None = None
    updated_at: str


@dataclass
class UserVectorRecord:
    vector: np.ndarray
    stale: bool
    updated_at: str
    model_version: str


class WorkerState:
    def __init__(self) -> None:
        self.catalog: PlotwiseItemCatalog | None = None
        self.user_vectors: dict[str, UserVectorRecord] = {}
        self.pending_deadline: dict[str, float] = {}
        self.enqueued: set[str] = set()
        self.refresh_queue: asyncio.Queue[str] = asyncio.Queue()
        self.jobs: dict[str, JobStatus] = {}
        self.retrain_running = False
        self.lock = asyncio.Lock()
        self.stop_event = asyncio.Event()
        self.background_tasks: list[asyncio.Task[Any]] = []
        self.state_dir = Path(os.getenv("WORKER_STATE_DIR", "data/worker_state"))
        self.external_item_map: dict[str, int] = {}
        # In-memory only: no raw interactions persisted (vectors go to service_persistence).
        self.poll_watermark: datetime | None = None
        self.retrain_interaction_checkpoint: int = 0

    def load_catalog(self, artifact_dir: str | Path | None = None) -> PlotwiseItemCatalog:
        self.catalog = load_plotwise_catalog(artifact_dir)
        save_active_model_pointer(
            self.state_dir,
            artifact_dir=str(self.catalog.artifact_dir),
            model_version=self.catalog.model_version,
        )
        return self.catalog

    def persist_user_vectors(self) -> None:
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
        save_user_vectors(self.state_dir, model_version=catalog.model_version, records=records)

    def restore_user_vectors_from_disk(self) -> int:
        model_version, records = load_user_vectors(self.state_dir)
        if not records or self.catalog is None:
            return 0
        if model_version != self.catalog.model_version:
            return 0
        for uid, persisted in records.items():
            self.user_vectors[uid] = UserVectorRecord(
                vector=persisted.vector,
                stale=persisted.stale,
                updated_at=persisted.updated_at,
                model_version=model_version,
            )
        return len(records)


app = FastAPI(title="worker-service")
state = WorkerState()


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


COALESCE_SECONDS = env_float("WORKER_COALESCE_SECONDS", 15.0)
REFRESH_WORKERS = max(1, int(env_float("WORKER_REFRESH_WORKERS", 2)))
DB_POLL_INTERVAL = env_float(
    "DB_POLL_INTERVAL_SECONDS",
    env_float("BOT_POLL_INTERVAL_SECONDS", 30.0),
)
DB_POLL_LIMIT = max(1, int(env_float("DB_POLL_LIMIT", 200)))
RETRAIN_THRESHOLD = int(env_float("RETRAIN_THRESHOLD", 0))
RETRAIN_CHECK_INTERVAL = env_float("RETRAIN_CHECK_INTERVAL_SECONDS", 60.0)
RECOMMENDER_URL = os.getenv("RECOMMENDER_URL", "http://localhost:8001")
TRAIN_INTERACTIONS_PATH = os.getenv("TRAIN_INTERACTIONS_PATH")
TRAIN_OUTPUT_ROOT = os.getenv("TRAIN_OUTPUT_ROOT")
USER_VECTOR_SYNC_BATCH = max(1, int(env_float("USER_VECTOR_SYNC_BATCH", 256)))
PUSH_RECOMMENDER_ON_START = os.getenv("WORKER_PUSH_RECOMMENDER_ON_START", "1").lower() in (
    "1",
    "true",
    "yes",
)
WORKER_BOOTSTRAP_ON_START = os.getenv("WORKER_BOOTSTRAP_ON_START", "1").lower() in (
    "1",
    "true",
    "yes",
)

async def fetch_interaction_stats() -> dict[str, int]:
    total = await asyncio.to_thread(fetch_interaction_count)
    async with state.lock:
        checkpoint = state.retrain_interaction_checkpoint
    return {
        "total_interactions": total,
        "new_since_checkpoint": max(0, total - checkpoint),
    }


async def acknowledge_retrain_checkpoint() -> None:
    total = await asyncio.to_thread(fetch_interaction_count)
    async with state.lock:
        state.retrain_interaction_checkpoint = total


def project_known_items(
    catalog: PlotwiseItemCatalog,
    ratings: list[tuple[str, float]],
    *,
    external_map: dict[str, int],
    register_unknown: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if register_unknown:
        catalog.ensure_items((item_id for item_id, _ in ratings), external_map=external_map)
    item_indices: list[int] = []
    values: list[float] = []
    for item_id, rating in ratings:
        idx = catalog.resolve_index(item_id, external_map=external_map)
        if idx is None:
            continue
        item_indices.append(int(idx))
        values.append(float(rating))
    if not item_indices:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    return np.array(item_indices, dtype=np.int32), np.array(values, dtype=np.float32)


async def push_user_vectors_batch(
    client: httpx.AsyncClient,
    *,
    model_version: str,
    users: list[tuple[str, UserVectorRecord]],
) -> None:
    if not users:
        return
    payload = {
        "model_version": model_version,
        "users": [
            {
                "user_id": user_id,
                "user_vector": record.vector.tolist(),
                "stale": record.stale,
                "updated_at": record.updated_at,
            }
            for user_id, record in users
        ],
    }
    response = await client.post(f"{RECOMMENDER_URL.rstrip('/')}/v1/internal/users/sync", json=payload, timeout=30.0)
    response.raise_for_status()


async def push_user_vector(client: httpx.AsyncClient, user_id: str, record: UserVectorRecord) -> None:
    await push_user_vectors_batch(client, model_version=record.model_version, users=[(user_id, record)])


async def push_plotwise_items_batch(
    client: httpx.AsyncClient,
    *,
    model_version: str,
    items: list[tuple[str, int]],
) -> None:
    catalog = state.catalog
    if catalog is None or not items:
        return
    payload = {
        "model_version": model_version,
        "items": [
            {
                "item_id": item_id,
                "item_vector": catalog.item_factors[idx].tolist(),
            }
            for item_id, idx in items
        ],
    }
    response = await client.post(
        f"{RECOMMENDER_URL.rstrip('/')}/v1/internal/items/append",
        json=payload,
        timeout=60.0,
    )
    response.raise_for_status()


async def process_refresh_user(user_id: str, client: httpx.AsyncClient) -> str:
    """Compute user vector from DB ratings and push to Recommender.

    Returns ``ok``, ``no_ratings``, or ``no_usable_ratings``.
    """
    catalog = state.catalog
    if catalog is None:
        raise RuntimeError("Plotwise item catalog is not loaded.")

    user_id = str(user_id)
    raw_ratings = await asyncio.to_thread(fetch_user_ratings, user_id)
    if not raw_ratings:
        log.debug("refresh skip user_id=%s: no ratings", user_id)
        return "no_ratings"

    async with state.lock:
        new_items = catalog.ensure_items(
            (item_id for item_id, _ in raw_ratings),
            external_map=state.external_item_map,
        )
        if new_items:
            catalog.persist()
    if new_items:
        await push_plotwise_items_batch(
            client,
            model_version=catalog.model_version,
            items=new_items,
        )

    item_indices, rating_values = project_known_items(
        catalog,
        raw_ratings,
        external_map=state.external_item_map,
        register_unknown=False,
    )
    if item_indices.size == 0:
        log.debug("refresh skip user_id=%s: no usable ratings", user_id)
        return "no_usable_ratings"

    vector = fit_user_vector(
        item_factors=catalog.item_factors,
        global_mean=catalog.global_mean,
        regularization=catalog.regularization,
        item_indices=item_indices,
        ratings_values=rating_values,
    )
    record = UserVectorRecord(
        vector=vector,
        stale=False,
        updated_at=utc_now_iso(),
        model_version=catalog.model_version,
    )
    async with state.lock:
        state.user_vectors[user_id] = record
    state.persist_user_vectors()
    await push_user_vector(client, user_id, record)
    return "ok"


async def coalescer_loop() -> None:
    while not state.stop_event.is_set():
        now = asyncio.get_running_loop().time()
        ready: list[str] = []
        async with state.lock:
            for user_id, deadline in list(state.pending_deadline.items()):
                if deadline <= now and user_id not in state.enqueued:
                    state.enqueued.add(user_id)
                    ready.append(user_id)
                    del state.pending_deadline[user_id]
        for user_id in ready:
            await state.refresh_queue.put(user_id)
        await asyncio.sleep(0.5)


async def refresh_worker_loop() -> None:
    async with httpx.AsyncClient() as client:
        while not state.stop_event.is_set():
            try:
                user_id = await asyncio.wait_for(state.refresh_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                await process_refresh_user(user_id, client)
            except Exception as exc:
                log.warning("refresh failed user_id=%s: %s", user_id, exc)
            finally:
                async with state.lock:
                    state.enqueued.discard(user_id)
                state.refresh_queue.task_done()


async def retrain_threshold_loop() -> None:
    """Start full retrain when enough new DB interactions accumulated (Worker-side threshold)."""
    if RETRAIN_THRESHOLD <= 0:
        return

    while not state.stop_event.is_set():
        try:
            stats = await fetch_interaction_stats()
            new_count = stats["new_since_checkpoint"]
            async with state.lock:
                busy = state.retrain_running
            if not busy and new_count >= RETRAIN_THRESHOLD:
                log.info(
                    "retrain threshold reached: new_interactions=%s threshold=%s",
                    new_count,
                    RETRAIN_THRESHOLD,
                )
                await start_retrain_job(
                    RetrainRequest(reason="threshold", data_source="movie_lens"),
                    mark_running=True,
                )
        except Exception as exc:
            log.debug("retrain threshold check: %s", exc)
        await asyncio.sleep(RETRAIN_CHECK_INTERVAL)


async def db_poll_loop() -> None:
    """Poll DB for updated ratings; enqueue refresh. Watermark lives in RAM only."""
    while not state.stop_event.is_set():
        try:
            async with state.lock:
                since = state.poll_watermark
            pending = await asyncio.to_thread(fetch_pending_users, since=since, limit=DB_POLL_LIMIT)
            max_updated: datetime | None = since
            for user_id, updated_at in pending:
                await refresh(request=RefreshRequest(user_id=user_id))
                if updated_at is not None and (max_updated is None or updated_at > max_updated):
                    max_updated = updated_at
            if max_updated is not None and max_updated != since:
                async with state.lock:
                    state.poll_watermark = max_updated
        except Exception as exc:
            log.warning("db poll: %s", exc)
        await asyncio.sleep(DB_POLL_INTERVAL)


def _bulk_load_user_vectors(bundle: ArtifactBundle) -> dict[str, UserVectorRecord]:
    user_factors_path = bundle.artifact_dir / "user_factors.npy"
    user_ids_path = bundle.artifact_dir / "user_ids.npy"
    if not user_factors_path.exists() or not user_ids_path.exists():
        return {}

    user_factors = np.load(user_factors_path)
    user_ids = load_user_ids_npy(user_ids_path)
    now = utc_now_iso()
    records: dict[str, UserVectorRecord] = {}
    for user_id, vector in zip(user_ids, user_factors):
        records[user_id_key(user_id)] = UserVectorRecord(
            vector=vector.astype(np.float32),
            stale=False,
            updated_at=now,
            model_version=bundle.model_version,
        )
    return records


async def push_all_vectors_to_recommender(client: httpx.AsyncClient) -> dict[str, Any]:
    catalog = state.catalog
    if catalog is None:
        raise RuntimeError("Plotwise item catalog is not loaded.")

    response = await client.post(
        f"{RECOMMENDER_URL.rstrip('/')}/v1/admin/model/activate",
        json={"artifact_dir": str(catalog.artifact_dir)},
        timeout=20.0,
    )
    response.raise_for_status()

    async with state.lock:
        users = list(state.user_vectors.items())
    if catalog._overlay_ids:
        overlay_pairs = [(item_id, catalog.item_id_to_idx[item_id]) for item_id in catalog._overlay_ids]
        await push_plotwise_items_batch(
            client,
            model_version=catalog.model_version,
            items=overlay_pairs,
        )

    pushed = 0
    for offset in range(0, len(users), USER_VECTOR_SYNC_BATCH):
        batch = users[offset : offset + USER_VECTOR_SYNC_BATCH]
        await push_user_vectors_batch(client, model_version=catalog.model_version, users=batch)
        pushed += len(batch)

    return {"model_version": catalog.model_version, "users_pushed": pushed}


async def start_retrain_job(request: RetrainRequest, *, mark_running: bool = False) -> str:
    job_id = f"retrain-{uuid.uuid4().hex[:12]}"
    status = JobStatus(job_id=job_id, status="pending", updated_at=utc_now_iso())
    state.jobs[job_id] = status
    if mark_running:
        state.retrain_running = True
    asyncio.create_task(retrain_job(job_id, request))
    return job_id


async def retrain_job(job_id: str, request: RetrainRequest) -> None:
    status = state.jobs[job_id]
    async with state.lock:
        state.retrain_running = True
    status.status = "training"
    status.updated_at = utc_now_iso()
    try:
        if request.data_source != "movie_lens":
            raise ValueError(f"Unsupported data_source={request.data_source!r}. Phase A supports movie_lens only.")

        hyper = request.hyperparameters or default_retrain_hyperparameters()
        window_cfg = train_config_from_file()
        interactions_path = request.interactions_path or TRAIN_INTERACTIONS_PATH
        output_root = request.output_root or TRAIN_OUTPUT_ROOT

        train_result = await asyncio.to_thread(
            run_training_phase_a,
            interactions_path=Path(interactions_path) if interactions_path else None,
            artifact_dir=request.artifact_dir,
            output_root=Path(output_root) if output_root else None,
            config=TrainConfig(
                backend=hyper.backend,
                use_gpu=hyper.use_gpu,
                factors=hyper.factors,
                regularization=hyper.regularization,
                iterations=hyper.iterations,
                train_window_mode=window_cfg.train_window_mode,
                train_window_days=window_cfg.train_window_days,
            ),
        )

        status.status = "activating"
        status.updated_at = utc_now_iso()
        catalog = state.load_catalog(str(train_result.output_dir))

        async with state.lock:
            state.user_vectors = _bulk_load_user_vectors(catalog.base)
        state.persist_user_vectors()

        async with httpx.AsyncClient() as client:
            payload = await push_all_vectors_to_recommender(client)
            await acknowledge_retrain_checkpoint()

        status.status = "completed"
        status.model_version = catalog.model_version
        status.detail = (
            f"trained users={train_result.n_users} items={train_result.n_items} "
            f"interactions={train_result.n_interactions}; pushed={payload['users_pushed']}"
        )
        status.updated_at = utc_now_iso()
    except Exception as exc:
        status.status = "failed"
        status.detail = str(exc)
        status.updated_at = utc_now_iso()
    finally:
        async with state.lock:
            state.retrain_running = False


@app.on_event("startup")
async def startup() -> None:
    artifact_path = await asyncio.to_thread(
        resolve_artifact_dir,
        env_dir=os.getenv("RECOM_ARTIFACT_DIR"),
        allow_bootstrap=WORKER_BOOTSTRAP_ON_START,
    )
    catalog = state.load_catalog(str(artifact_path))
    save_registry_active_pointer(
        artifact_path.parent,
        artifact_dir=catalog.artifact_dir,
        model_version=catalog.model_version,
    )
    if WORKER_BOOTSTRAP_ON_START:
        log.info("Artifact snap ready at %s (bootstrap_on_start=1)", artifact_path)
    state.external_item_map = load_external_item_map(catalog.artifact_dir)
    if state.external_item_map:
        log.info("Loaded %s external item_id mappings", len(state.external_item_map))
    log.info(
        "Plotwise item data: %s base items, %s overlay items",
        catalog.base.item_factors.shape[0],
        catalog.overlay_count,
    )

    restored = state.restore_user_vectors_from_disk()
    if restored == 0:
        if state.catalog is not None:
            state.user_vectors = _bulk_load_user_vectors(state.catalog.base)
            state.persist_user_vectors()
            log.info("Hydrated %s user vectors from training bundle", len(state.user_vectors))
    else:
        log.info("Restored %s user vectors from worker state", restored)

    state.stop_event.clear()
    state.background_tasks.append(asyncio.create_task(coalescer_loop()))
    for _ in range(REFRESH_WORKERS):
        state.background_tasks.append(asyncio.create_task(refresh_worker_loop()))
    if DB_POLL_INTERVAL > 0:
        state.background_tasks.append(asyncio.create_task(db_poll_loop()))
    if RETRAIN_THRESHOLD > 0:
        state.background_tasks.append(asyncio.create_task(retrain_threshold_loop()))
        log.info("Retrain threshold enabled: %s interactions", RETRAIN_THRESHOLD)

    if PUSH_RECOMMENDER_ON_START:
        try:
            async with httpx.AsyncClient() as client:
                payload = await push_all_vectors_to_recommender(client)
                log.info("Pushed vectors to recommender on startup: %s", payload)
        except Exception as exc:
            log.warning("Recommender push on startup failed: %s", exc)


@app.on_event("shutdown")
async def shutdown() -> None:
    state.stop_event.set()
    for task in state.background_tasks:
        task.cancel()
    if state.background_tasks:
        await asyncio.gather(*state.background_tasks, return_exceptions=True)


@app.get("/health")
async def health() -> dict[str, Any]:
    catalog = state.catalog
    train_cfg = load_train_config_file()
    return {
        "status": "ok",
        "model_version": None if catalog is None else catalog.model_version,
        "artifact_dir": None if catalog is None else str(catalog.artifact_dir),
        "item_id_format": None if catalog is None else catalog.item_id_format,
        "n_items": None if catalog is None else len(catalog.item_id_to_idx),
        "plotwise_overlay_items": None if catalog is None else catalog.overlay_count,
        "queued_users": state.refresh_queue.qsize(),
        "known_user_vectors": len(state.user_vectors),
        "db_poll_interval_sec": DB_POLL_INTERVAL,
        "database_url": database_url_masked(),
        "retrain_threshold": RETRAIN_THRESHOLD,
        "retrain_check_interval_sec": RETRAIN_CHECK_INTERVAL,
        "retrain_running": state.retrain_running,
        "train_config_path": str(train_cfg.path),
        "train_hyperparameters": {
            "backend": train_cfg.backend,
            "use_gpu": train_cfg.use_gpu,
            "factors": train_cfg.factors,
            "regularization": train_cfg.regularization,
            "iterations": train_cfg.iterations,
            "cv_best_name": train_cfg.cv_best_name,
        },
    }


@app.get("/v1/internal/model")
async def get_model() -> dict[str, Any]:
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
        "known_user_vectors": len(state.user_vectors),
    }


@app.post("/v1/internal/users/{user_id}/refresh-now")
async def refresh_user_now(user_id: str) -> dict[str, Any]:
    """Synchronous ridge refresh + push to Recommender (for on-demand recommend)."""
    async with httpx.AsyncClient() as client:
        refresh_status = await process_refresh_user(str(user_id), client)
    async with state.lock:
        has_vector = str(user_id) in state.user_vectors
    rating_count = len(await asyncio.to_thread(fetch_user_ratings, user_id))
    return {
        "user_id": str(user_id),
        "refresh_status": refresh_status,
        "has_vector": has_vector,
        "rating_count": rating_count,
        "has_ratings": rating_count > 0,
    }


@app.get("/v1/internal/users/{user_id}/recommend-readiness")
async def recommend_readiness(user_id: str) -> dict[str, Any]:
    """Whether the user has DB ratings and a computed vector (for Recommender error messages)."""
    ratings = await asyncio.to_thread(fetch_user_ratings, user_id)
    async with state.lock:
        has_vector = str(user_id) in state.user_vectors
    rating_count = len(ratings)
    return {
        "user_id": str(user_id),
        "rating_count": rating_count,
        "has_ratings": rating_count > 0,
        "has_vector": has_vector,
    }


@app.get("/v1/internal/users/{user_id}/vector")
async def get_user_vector(user_id: str) -> dict[str, Any]:
    async with state.lock:
        record = state.user_vectors.get(str(user_id))
        catalog = state.catalog
    if record is None or catalog is None:
        raise HTTPException(status_code=404, detail=f"No vector for user_id={user_id}")
    return {
        "user_id": user_id,
        "model_version": record.model_version,
        "user_vector": record.vector.tolist(),
        "stale": record.stale,
        "updated_at": record.updated_at,
    }


@app.post("/v1/internal/recommender/resync")
async def recommender_resync() -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        return await push_all_vectors_to_recommender(client)


@app.post("/v1/admin/model/activate")
async def activate_model(request: ActivateModelRequest) -> dict[str, Any]:
    """Load item catalog snap on Worker and activate same bundle on Recommender (no bulk user push)."""
    catalog = state.load_catalog(request.artifact_dir)
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RECOMMENDER_URL.rstrip('/')}/v1/admin/model/activate",
            json={
                "artifact_dir": str(catalog.artifact_dir),
                "reset_user_vectors": request.reset_user_vectors,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        payload = response.json()
    return {
        "status": "activated",
        "model_version": catalog.model_version,
        "artifact_dir": str(catalog.artifact_dir),
        "item_id_format": catalog.item_id_format,
        "n_items": len(catalog.item_id_to_idx),
        "plotwise_overlay_items": catalog.overlay_count,
        "recommender": payload,
    }


@app.post("/v1/refresh")
async def refresh(request: RefreshRequest) -> dict[str, Any]:
    user_id = str(request.user_id)
    async with state.lock:
        state.pending_deadline[user_id] = asyncio.get_running_loop().time() + COALESCE_SECONDS
    return {"status": "accepted", "user_id": user_id, "coalesce_seconds": COALESCE_SECONDS}


@app.post("/v1/admin/retrain")
async def retrain(request: RetrainRequest) -> dict[str, Any]:
    """Manual or scheduled retrain (same job as threshold-triggered)."""
    async with state.lock:
        if state.retrain_running:
            raise HTTPException(status_code=409, detail="Retrain job already running")
    job_id = await start_retrain_job(request, mark_running=True)
    return {"job_id": job_id, "status": "accepted"}


@app.get("/v1/admin/retrain/{job_id}")
async def get_retrain_status(job_id: str) -> dict[str, Any]:
    status = state.jobs.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id={job_id}")
    return status.model_dump()
