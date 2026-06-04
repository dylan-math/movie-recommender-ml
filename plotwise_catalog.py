"""Runtime plotwise item catalog: base ALS snap + overlay persisted under data/plotwise_item_data/."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from als_runtime import ArtifactBundle, load_artifact_bundle, resolve_item_index
from utils import normalize_bot_item_id

log = logging.getLogger("plotwise-catalog")

META_FILE = "plotwise_item_meta.json"
OVERLAY_IDS_FILE = "plotwise_overlay_item_ids.npy"
OVERLAY_FACTORS_FILE = "plotwise_overlay_item_factors.npy"


def plotwise_catalog_dir() -> Path:
    """Directory for overlay files (env ``PLOTWISE_ITEM_DATA_DIR``, default ``data/plotwise_item_data``)."""
    return Path(os.getenv("PLOTWISE_ITEM_DATA_DIR", "data/plotwise_item_data"))


@dataclass
class PlotwiseItemCatalog:
    """Merged view: trained base snap + plotwise overlay (bot tokens learned at runtime)."""

    base: ArtifactBundle
    item_factors: np.ndarray
    item_ids: np.ndarray
    item_id_to_idx: dict[str, int]
    item_id_format: str
    _overlay_ids: list[str]
    _overlay_factors: np.ndarray
    _data_dir: Path

    @property
    def artifact_dir(self) -> Path:
        return self.base.artifact_dir

    @property
    def model_version(self) -> str:
        return self.base.model_version

    @property
    def factors(self) -> int:
        return self.base.factors

    @property
    def regularization(self) -> float:
        return self.base.regularization

    @property
    def global_mean(self) -> float:
        return self.base.global_mean

    @property
    def movie_ids(self) -> np.ndarray:
        return self.item_ids

    @property
    def movie_metadata(self) -> pd.DataFrame:
        return self.base.movie_metadata

    @property
    def overlay_count(self) -> int:
        return len(self._overlay_ids)

    @classmethod
    def from_base(cls, base: ArtifactBundle, *, data_dir: Path | None = None) -> PlotwiseItemCatalog:
        target = data_dir or plotwise_catalog_dir()
        overlay_ids: list[str] = []
        overlay_factors = np.zeros((0, base.factors), dtype=np.float32)
        meta_path = target / META_FILE
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as file:
                meta = json.load(file)
            if str(meta.get("base_model_version")) == base.model_version:
                ids_path = target / OVERLAY_IDS_FILE
                factors_path = target / OVERLAY_FACTORS_FILE
                if ids_path.exists() and factors_path.exists():
                    loaded_ids = np.load(ids_path, allow_pickle=True).reshape(-1)
                    overlay_ids = [str(x) for x in loaded_ids]
                    overlay_factors = np.load(factors_path).astype(np.float32)
                    if overlay_factors.shape[0] != len(overlay_ids):
                        log.warning("plotwise overlay size mismatch; ignoring overlay")
                        overlay_ids = []
                        overlay_factors = np.zeros((0, base.factors), dtype=np.float32)
            else:
                log.info(
                    "plotwise overlay base version mismatch (%s vs %s); starting fresh overlay",
                    meta.get("base_model_version"),
                    base.model_version,
                )

        return cls._merge(base, overlay_ids, overlay_factors, data_dir=target)

    @classmethod
    def _merge(
        cls,
        base: ArtifactBundle,
        overlay_ids: list[str],
        overlay_factors: np.ndarray,
        *,
        data_dir: Path,
    ) -> PlotwiseItemCatalog:
        base_ids = [str(x) for x in np.asarray(base.movie_ids).reshape(-1)]
        base_idx = dict(base.item_id_to_idx)
        merged_ids = list(base_ids)
        merged_factors = base.item_factors.astype(np.float32)
        merged_idx = dict(base_idx)

        cold = base.item_factors.mean(axis=0).astype(np.float32)
        clean_overlay_ids: list[str] = []
        clean_overlay_factors: list[np.ndarray] = []
        for token, factor in zip(overlay_ids, overlay_factors):
            if token in merged_idx:
                continue
            row = factor if factor.shape == (base.factors,) else cold
            merged_idx[token] = len(merged_ids)
            merged_ids.append(token)
            merged_factors = np.vstack([merged_factors, row.reshape(1, -1)])
            clean_overlay_ids.append(token)
            clean_overlay_factors.append(row.astype(np.float32))

        overlay_stack = (
            np.stack(clean_overlay_factors).astype(np.float32)
            if clean_overlay_factors
            else np.zeros((0, base.factors), dtype=np.float32)
        )
        item_format = "plotwise" if clean_overlay_ids else base.item_id_format
        return cls(
            base=base,
            item_factors=merged_factors,
            item_ids=np.array(merged_ids, dtype=object),
            item_id_to_idx=merged_idx,
            item_id_format=item_format,
            _overlay_ids=clean_overlay_ids,
            _overlay_factors=overlay_stack,
            _data_dir=data_dir,
        )

    def cold_start_vector(self) -> np.ndarray:
        return self.base.item_factors.mean(axis=0).astype(np.float32)

    def canonical_item_id(self, raw_item_id: object) -> str | None:
        token = normalize_bot_item_id(str(raw_item_id))
        if token is not None:
            return token
        value = str(raw_item_id).strip()
        return value or None

    def resolve_index(
        self,
        raw_item_id: object,
        *,
        external_map: dict[str, int] | None = None,
    ) -> int | None:
        return resolve_item_index(self, raw_item_id, external_map=external_map)

    def ensure_item(self, raw_item_id: object, *, external_map: dict[str, int] | None = None) -> int | None:
        """Register plotwise item if missing; return row index."""
        existing = self.resolve_index(raw_item_id, external_map=external_map)
        if existing is not None:
            return existing

        token = self.canonical_item_id(raw_item_id)
        if token is None:
            return None

        cold = self.cold_start_vector()
        self._overlay_ids.append(token)
        self._overlay_factors = np.vstack([self._overlay_factors, cold.reshape(1, -1)])
        idx = len(self.item_ids)
        self.item_id_to_idx[token] = idx
        self.item_ids = np.append(self.item_ids, token)
        self.item_factors = np.vstack([self.item_factors, cold.reshape(1, -1)])
        self.item_id_format = "plotwise"
        log.info("plotwise catalog: added item_id=%s index=%s", token[:16], idx)
        return idx

    def append_item_factor(self, item_id: str, factor: np.ndarray) -> int:
        """Append or return index (used when Recommender receives pushed item rows)."""
        token = str(item_id).strip()
        if token in self.item_id_to_idx:
            return self.item_id_to_idx[token]
        vector = np.asarray(factor, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.factors:
            raise ValueError(f"factor dim {vector.shape[0]} != {self.factors}")
        if token not in self._overlay_ids:
            self._overlay_ids.append(token)
            self._overlay_factors = np.vstack([self._overlay_factors, vector.reshape(1, -1)])
        idx = len(self.item_ids)
        self.item_id_to_idx[token] = idx
        self.item_ids = np.append(self.item_ids, token)
        self.item_factors = np.vstack([self.item_factors, vector.reshape(1, -1)])
        self.item_id_format = "plotwise"
        return idx

    def ensure_items(
        self,
        raw_item_ids: Iterable[object],
        *,
        external_map: dict[str, int] | None = None,
    ) -> list[tuple[str, int]]:
        added: list[tuple[str, int]] = []
        seen: set[str] = set()
        for raw in raw_item_ids:
            token = self.canonical_item_id(raw)
            if token is None or token in seen:
                continue
            seen.add(token)
            if token in self.item_id_to_idx:
                continue
            idx = self.ensure_item(raw, external_map=external_map)
            if idx is not None:
                added.append((token, idx))
        return added

    def reload_base(self, base: ArtifactBundle) -> None:
        """After model activate/retrain: rebuild merge, refresh cold vectors for overlay rows."""
        cold = base.item_factors.mean(axis=0).astype(np.float32)
        overlay_ids = list(self._overlay_ids)
        overlay_factors = np.stack([cold] * len(overlay_ids)).astype(np.float32) if overlay_ids else np.zeros(
            (0, base.factors), dtype=np.float32
        )
        merged = self._merge(base, overlay_ids, overlay_factors, data_dir=self._data_dir)
        self.base = merged.base
        self.item_factors = merged.item_factors
        self.item_ids = merged.item_ids
        self.item_id_to_idx = merged.item_id_to_idx
        self.item_id_format = merged.item_id_format
        self._overlay_ids = merged._overlay_ids
        self._overlay_factors = merged._overlay_factors

    def persist(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "base_model_version": self.model_version,
            "base_artifact_dir": str(self.artifact_dir),
            "overlay_count": len(self._overlay_ids),
        }
        with open(self._data_dir / META_FILE, "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)
        if self._overlay_ids:
            np.save(self._data_dir / OVERLAY_IDS_FILE, np.array(self._overlay_ids, dtype=object))
            np.save(self._data_dir / OVERLAY_FACTORS_FILE, self._overlay_factors)
        else:
            for path in (OVERLAY_IDS_FILE, OVERLAY_FACTORS_FILE):
                p = self._data_dir / path
                if p.exists():
                    p.unlink()


def load_plotwise_catalog(artifact_dir: str | Path | None = None) -> PlotwiseItemCatalog:
    base = load_artifact_bundle(artifact_dir)
    return PlotwiseItemCatalog.from_base(base)
