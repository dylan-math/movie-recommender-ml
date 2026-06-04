"""Explicit ALS on GPU via CuPy (GTX 1060 / CUDA 12, sm_61)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from als_trainer import TrainResult


@dataclass
class CupyGpuStatus:
    available: bool
    device_name: str | None
    message: str


def cupy_gpu_status() -> CupyGpuStatus:
    try:
        import cupy as cp
    except ImportError:
        return CupyGpuStatus(False, None, "Install: pip install cupy-cuda12x")

    try:
        if not cp.cuda.is_available():
            return CupyGpuStatus(False, None, "CuPy installed but no CUDA device")
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        _ = cp.zeros(1)
        return CupyGpuStatus(True, name, f"GPU ready: {name}")
    except Exception as exc:
        return CupyGpuStatus(False, None, f"CuPy CUDA error: {exc}")


def train_als_cupy(
    *,
    interactions: sparse.csr_matrix,
    factors: int,
    regularization: float,
    iterations: int,
    use_gpu: bool = True,
    show_progress: bool = True,
) -> TrainResult:
    if interactions.format != "csr":
        interactions = interactions.tocsr()

    n_users, n_items = interactions.shape
    if n_users == 0 or n_items == 0:
        raise ValueError("Empty interaction matrix.")

    global_mean = float(interactions.data.mean()) if interactions.nnz else 0.0
    centered = interactions.copy()
    centered.data = centered.data.astype(np.float32) - np.float32(global_mean)
    centered_csc = centered.tocsc()

    if use_gpu:
        status = cupy_gpu_status()
        if not status.available:
            raise RuntimeError(f"GPU requested but unavailable: {status.message}")
        import cupy as cp

        xp = cp
    else:
        xp = np

    user_factors = xp.random.normal(scale=0.01, size=(n_users, factors)).astype(np.float32)
    item_factors = xp.random.normal(scale=0.01, size=(n_items, factors)).astype(np.float32)
    reg = np.float32(regularization)
    eye = xp.eye(factors, dtype=np.float32)

    iterator = range(iterations)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="ALS (cupy)" if use_gpu else "ALS (numpy)")

    for _ in iterator:
        gram_items = item_factors.T @ item_factors + reg * eye
        inv_gram_items = xp.linalg.inv(gram_items)

        for user_idx in range(n_users):
            start = centered.indptr[user_idx]
            end = centered.indptr[user_idx + 1]
            if start == end:
                continue
            cols = centered.indices[start:end]
            ratings = xp.asarray(centered.data[start:end], dtype=np.float32)
            item_block = item_factors[cols]
            user_factors[user_idx] = inv_gram_items @ (item_block.T @ ratings)

        gram_users = user_factors.T @ user_factors + reg * eye
        inv_gram_users = xp.linalg.inv(gram_users)

        for item_idx in range(n_items):
            start = centered_csc.indptr[item_idx]
            end = centered_csc.indptr[item_idx + 1]
            if start == end:
                continue
            rows = centered_csc.indices[start:end]
            ratings = xp.asarray(centered_csc.data[start:end], dtype=np.float32)
            user_block = user_factors[rows]
            item_factors[item_idx] = inv_gram_users @ (user_block.T @ ratings)

    if use_gpu:
        user_factors = cp.asnumpy(user_factors)
        item_factors = cp.asnumpy(item_factors)

    return TrainResult(
        user_factors=user_factors.astype(np.float32),
        item_factors=item_factors.astype(np.float32),
        global_mean=global_mean,
        iterations=iterations,
    )
