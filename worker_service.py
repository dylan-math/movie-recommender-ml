"""Worker service: refresh queue + coalescing + user-vector push to recommender."""

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

from als_runtime import ArtifactBundle, fit_user_vector, load_artifact_bundle, load_user_ids_npy, user_id_key
from service_persistence import (
    PersistedUserVector,
    load_active_model_pointer,
    load_user_vectors,
    save_active_model_pointer,
    save_user_vectors,
)
from train_config_loader import load_train_config_file, train_config_from_file
from trainer_runner import TrainConfig, run_training_phase_a

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
        self.bundle: ArtifactBundle | None = None
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

    def load_bundle(self, artifact_dir: str | Path | None = None) -> ArtifactBundle:
        self.bundle = load_artifact_bundle(artifact_dir)
        save_active_model_pointer(
            self.state_dir,
            artifact_dir=str(self.bundle.artifact_dir),
            model_version=self.bundle.model_version,
        )
        return self.bundle

    def persist_user_vectors(self) -> None:
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
        if not records or self.bundle is None:
            return 0
        if model_version != self.bundle.model_version:
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
BOT_POLL_INTERVAL = env_float("BOT_POLL_INTERVAL_SECONDS", 30.0)
BOT_BACKEND_URL = os.getenv("BOT_BACKEND_URL", "http://localhost:9000")
BOT_RATINGS_PATH_TEMPLATE = os.getenv("BOT_RATINGS_PATH_TEMPLATE", "/v1/internal/users/{user_id}/ratings")
BOT_PENDING_REFRESH_PATH = os.getenv("BOT_PENDING_REFRESH_PATH", "/v1/internal/users/pending_refresh")
BOT_INTERACTION_STATS_PATH = os.getenv(
    "BOT_INTERACTION_STATS_PATH", "/v1/internal/interactions/stats"
)
BOT_RETRAIN_CHECKPOINT_PATH = os.getenv(
    "BOT_RETRAIN_CHECKPOINT_PATH", "/v1/internal/retrain/checkpoint"
)
BOT_REFRESH_ACK_PATH_TEMPLATE = os.getenv(
    "BOT_REFRESH_ACK_PATH_TEMPLATE", "/v1/internal/users/{user_id}/refresh_ack"
)
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


async def fetch_bot_interaction_stats(client: httpx.AsyncClient) -> dict[str, int]:
    response = await client.get(
        f"{BOT_BACKEND_URL.rstrip('/')}{BOT_INTERACTION_STATS_PATH}",
        timeout=10.0,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "total_interactions": int(payload.get("total_interactions", 0)),
        "new_since_checkpoint": int(payload.get("new_since_checkpoint", 0)),
    }


async def acknowledge_bot_retrain_checkpoint(client: httpx.AsyncClient) -> None:
    response = await client.post(f"{BOT_BACKEND_URL.rstrip('/')}{BOT_RETRAIN_CHECKPOINT_PATH}", timeout=10.0)
    response.raise_for_status()


async def fetch_user_ratings(client: httpx.AsyncClient, user_id: str) -> list[tuple[int, float]]:
    path = BOT_RATINGS_PATH_TEMPLATE.format(user_id=user_id)
    response = await client.get(f"{BOT_BACKEND_URL.rstrip('/')}{path}", timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    ratings = payload.get("ratings", payload if isinstance(payload, list) else [])
    result: list[tuple[int, float]] = []
    for row in ratings:
        if not isinstance(row, dict):
            continue
        item_id = row.get("item_id") or row.get("movieId") or row.get("movie_id")
        rating = row.get("rating")
        if item_id is None or rating is None:
            continue
        result.append((int(item_id), float(rating)))
    return result


def project_known_items(bundle: ArtifactBundle, ratings: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    item_indices: list[int] = []
    values: list[float] = []
    for item_id, rating in ratings:
        idx = bundle.movie_id_to_idx.get(int(item_id))
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


async def process_refresh_user(user_id: str, client: httpx.AsyncClient) -> None:
    bundle = state.bundle
    if bundle is None:
        raise RuntimeError("Bundle is not loaded.")

    ratings = await fetch_user_ratings(client, user_id)
    item_indices, rating_values = project_known_items(bundle, ratings)
    if item_indices.size == 0:
        return

    vector = fit_user_vector(
        item_factors=bundle.item_factors,
        global_mean=bundle.global_mean,
        regularization=bundle.regularization,
        item_indices=item_indices,
        ratings_values=rating_values,
    )
    record = UserVectorRecord(
        vector=vector,
        stale=False,
        updated_at=utc_now_iso(),
        model_version=bundle.model_version,
    )
    async with state.lock:
        state.user_vectors[user_id] = record
    state.persist_user_vectors()
    await push_user_vector(client, user_id, record)

    ack_path = BOT_REFRESH_ACK_PATH_TEMPLATE.format(user_id=user_id)
    try:
        await client.post(f"{BOT_BACKEND_URL.rstrip('/')}{ack_path}", timeout=5.0)
    except Exception:
        pass


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
    """Start full retrain when enough new bot interactions accumulated (Worker-side threshold)."""
    if RETRAIN_THRESHOLD <= 0:
        return

    async with httpx.AsyncClient() as client:
        while not state.stop_event.is_set():
            try:
                stats = await fetch_bot_interaction_stats(client)
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


async def bot_poll_loop() -> None:
    async with httpx.AsyncClient() as client:
        while not state.stop_event.is_set():
            try:
                response = await client.get(
                    f"{BOT_BACKEND_URL.rstrip('/')}{BOT_PENDING_REFRESH_PATH}",
                    params={"limit": 200},
                    timeout=10.0,
                )
                response.raise_for_status()
                user_ids = response.json().get("user_ids", [])
                for user_id in user_ids:
                    await refresh(request=RefreshRequest(user_id=user_id))
            except Exception as exc:
                log.debug("bot poll: %s", exc)
            await asyncio.sleep(BOT_POLL_INTERVAL)


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
    bundle = state.bundle
    if bundle is None:
        raise RuntimeError("Bundle is not loaded.")

    response = await client.post(
        f"{RECOMMENDER_URL.rstrip('/')}/v1/admin/model/activate",
        json={"artifact_dir": str(bundle.artifact_dir)},
        timeout=20.0,
    )
    response.raise_for_status()

    async with state.lock:
        users = list(state.user_vectors.items())

    pushed = 0
    for offset in range(0, len(users), USER_VECTOR_SYNC_BATCH):
        batch = users[offset : offset + USER_VECTOR_SYNC_BATCH]
        await push_user_vectors_batch(client, model_version=bundle.model_version, users=batch)
        pushed += len(batch)

    return {"model_version": bundle.model_version, "users_pushed": pushed}


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
        bundle = state.load_bundle(str(train_result.output_dir))

        async with state.lock:
            state.user_vectors = _bulk_load_user_vectors(bundle)
        state.persist_user_vectors()

        async with httpx.AsyncClient() as client:
            payload = await push_all_vectors_to_recommender(client)
            await acknowledge_bot_retrain_checkpoint(client)

        status.status = "completed"
        status.model_version = bundle.model_version
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
    artifact_dir = os.getenv("RECOM_ARTIFACT_DIR")
    pointer = load_active_model_pointer(state.state_dir)
    if artifact_dir is None and pointer is not None:
        artifact_dir = pointer.get("artifact_dir")
    state.load_bundle(artifact_dir)

    restored = state.restore_user_vectors_from_disk()
    if restored == 0:
        bundle = state.bundle
        if bundle is not None:
            state.user_vectors = _bulk_load_user_vectors(bundle)
            state.persist_user_vectors()
            log.info("Hydrated %s user vectors from training bundle", len(state.user_vectors))
    else:
        log.info("Restored %s user vectors from worker state", restored)

    state.stop_event.clear()
    state.background_tasks.append(asyncio.create_task(coalescer_loop()))
    for _ in range(REFRESH_WORKERS):
        state.background_tasks.append(asyncio.create_task(refresh_worker_loop()))
    if BOT_POLL_INTERVAL > 0:
        state.background_tasks.append(asyncio.create_task(bot_poll_loop()))
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
    bundle = state.bundle
    train_cfg = load_train_config_file()
    return {
        "status": "ok",
        "model_version": None if bundle is None else bundle.model_version,
        "artifact_dir": None if bundle is None else str(bundle.artifact_dir),
        "queued_users": state.refresh_queue.qsize(),
        "known_user_vectors": len(state.user_vectors),
        "bot_poll_interval_sec": BOT_POLL_INTERVAL,
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
        "known_user_vectors": len(state.user_vectors),
    }


@app.get("/v1/internal/users/{user_id}/vector")
async def get_user_vector(user_id: str) -> dict[str, Any]:
    async with state.lock:
        record = state.user_vectors.get(str(user_id))
        bundle = state.bundle
    if record is None or bundle is None:
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


@app.post("/v1/refresh")
async def refresh(request: RefreshRequest) -> dict[str, Any]:
    user_id = str(request.user_id)
    async with state.lock:
        state.pending_deadline[user_id] = asyncio.get_running_loop().time() + COALESCE_SECONDS
    return {"status": "accepted", "user_id": user_id, "coalesce_seconds": COALESCE_SECONDS}


@app.post("/v1/admin/retrain")
async def retrain(request: RetrainRequest) -> dict[str, Any]:
    """Manual or scheduled retrain (same job as threshold-triggered). Bot does not call this."""
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
