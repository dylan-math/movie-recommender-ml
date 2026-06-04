"""Explicit-feedback ALS training (alternating least squares)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class TrainResult:
    user_factors: np.ndarray
    item_factors: np.ndarray
    global_mean: float
    iterations: int


def train_als_explicit(
    *,
    interactions: sparse.csr_matrix,
    factors: int,
    regularization: float,
    iterations: int,
) -> TrainResult:
    """Train explicit ALS on a sparse user-item rating matrix."""
    if interactions.format != "csr":
        interactions = interactions.tocsr()

    n_users, n_items = interactions.shape
    if n_users == 0 or n_items == 0:
        raise ValueError("Empty interaction matrix.")

    global_mean = float(interactions.data.mean()) if interactions.nnz else 0.0
    centered = interactions.copy()
    centered.data = centered.data.astype(np.float32) - np.float32(global_mean)
    centered_csc = centered.tocsc()

    user_factors = np.random.normal(scale=0.01, size=(n_users, factors)).astype(np.float32)
    item_factors = np.random.normal(scale=0.01, size=(n_items, factors)).astype(np.float32)

    reg = np.float32(regularization)
    eye = np.eye(factors, dtype=np.float32)

    for _ in range(iterations):
        gram_items = item_factors.T @ item_factors + reg * eye
        inv_gram_items = np.linalg.inv(gram_items)
        for user_idx in range(n_users):
            start = centered.indptr[user_idx]
            end = centered.indptr[user_idx + 1]
            if start == end:
                continue
            cols = centered.indices[start:end]
            ratings = centered.data[start:end]
            item_block = item_factors[cols]
            user_factors[user_idx] = inv_gram_items @ (item_block.T @ ratings)

        gram_users = user_factors.T @ user_factors + reg * eye
        inv_gram_users = np.linalg.inv(gram_users)
        for item_idx in range(n_items):
            start = centered_csc.indptr[item_idx]
            end = centered_csc.indptr[item_idx + 1]
            if start == end:
                continue
            rows = centered_csc.indices[start:end]
            ratings = centered_csc.data[start:end]
            user_block = user_factors[rows]
            item_factors[item_idx] = inv_gram_users @ (user_block.T @ ratings)

    return TrainResult(
        user_factors=user_factors.astype(np.float32),
        item_factors=item_factors.astype(np.float32),
        global_mean=global_mean,
        iterations=iterations,
    )
