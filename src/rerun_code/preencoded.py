from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import read_table


VECTOR_PRIORITY = (
    "image_embeddings_cache.npy", "image_embeddings.npy", "embeddings.npy", "features.npy", "vectors.npy",
    "gallery_embeddings.npy", "query_embeddings.npy",
)
FAISS_PRIORITY = (
    "faiss_image.index", "image.index", "gallery.index", "faiss_image.faiss",
)
METADATA_PRIORITY = (
    "metadata.jsonl", "image_metadata.jsonl", "metadata.json", "data.jsonl",
    "gallery_metadata.jsonl", "query_metadata.jsonl", "metadata.csv",
)


def inventory_preencoded(root: str | Path) -> dict[str, list[str]]:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(base)
    return {
        "npy": [str(p) for p in sorted(base.rglob("*.npy"))],
        "faiss": [str(p) for p in sorted([*base.rglob("*.index"), *base.rglob("*.faiss")])],
        "metadata": [str(p) for suffix in ("*.jsonl", "*.json", "*.csv", "*.tsv", "*.parquet") for p in sorted(base.rglob(suffix))],
    }


def _select_named(root: Path, names: Sequence[str], candidates: Sequence[str], kind: str) -> Path:
    by_name = {Path(path).name: Path(path) for path in candidates}
    for name in names:
        if name in by_name:
            return by_name[name]
    if len(candidates) == 1:
        return Path(candidates[0])
    raise ValueError(
        f"Cannot choose one {kind} file below {root}. Candidates={list(candidates)}. "
        f"Set the explicit file in the notebook after reviewing the inventory."
    )


def _load_faiss_vectors(path: Path) -> np.ndarray:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("Install a compatible FAISS build to read a legacy index") from exc
    index = faiss.read_index(str(path))
    if not hasattr(index, "reconstruct_n"):
        raise TypeError(f"FAISS index cannot reconstruct vectors: {path}")
    values = index.reconstruct_n(0, index.ntotal)
    return np.asarray(values, dtype=np.float32)


def load_preencoded(
    root: str | Path,
    *,
    vector_file: str | Path | None = None,
    metadata_file: str | Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    base = Path(root)
    inventory = inventory_preencoded(base)
    if vector_file:
        vector_path = Path(vector_file)
    elif inventory["npy"]:
        vector_path = _select_named(base, VECTOR_PRIORITY, inventory["npy"], "NumPy vector")
    else:
        vector_path = _select_named(base, FAISS_PRIORITY, inventory["faiss"], "image FAISS index")
    if "text" in vector_path.name.lower():
        raise ValueError(
            f"Refusing to use text embeddings for image retrieval: {vector_path}. "
            "Select image_embeddings_cache.npy or faiss_image.index."
        )
    if metadata_file:
        metadata_path = Path(metadata_file)
    else:
        metadata_path = _select_named(base, METADATA_PRIORITY, inventory["metadata"], "metadata")
    vectors = np.load(vector_path).astype(np.float32) if vector_path.suffix == ".npy" else _load_faiss_vectors(vector_path)
    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix: {vector_path}")
    metadata = read_table(metadata_path)
    if len(vectors) != len(metadata):
        raise AssertionError(f"Vector/metadata count mismatch: {len(vectors)} != {len(metadata)}")
    provenance = {
        "root": str(base), "vector_file": str(vector_path), "metadata_file": str(metadata_path),
        "n_vectors": int(len(vectors)), "dimension": int(vectors.shape[1]), "inventory": inventory,
    }
    return vectors, metadata, provenance


def _tokens(row: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in ("record_id", "image_sha256", "image_id", "uid", "study_id", "file_name", "filename", "image_path", "path"):
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        result.add(f"{key}:{value}")
        if key in {"image_path", "path", "file_name", "filename"}:
            path = Path(value)
            result.add(f"basename:{path.name}")
            result.add(f"stem:{path.stem}")
    return result


def align_manifest_to_preencoded(
    manifest: pd.DataFrame,
    vectors: np.ndarray,
    metadata: pd.DataFrame,
    *,
    allow_positional: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    if len(vectors) != len(metadata):
        raise AssertionError("Vector and metadata rows differ")
    token_index: dict[str, set[int]] = defaultdict(set)
    for position, row in metadata.iterrows():
        for token in _tokens(row):
            token_index[token].add(int(position))
    positions: list[int] = []
    failures: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        matches: set[int] = set()
        for token in _tokens(row):
            matches.update(token_index.get(token, set()))
        if len(matches) != 1:
            failures.append({"record_id": row.get("record_id", ""), "n_matches": len(matches), "matches": sorted(matches)[:10]})
            positions.append(-1)
        else:
            positions.append(next(iter(matches)))
    if failures:
        if allow_positional and len(manifest) == len(metadata):
            positions = list(range(len(manifest)))
            mode = "explicitly_allowed_positional"
        else:
            preview = json.dumps(failures[:10], indent=2)
            raise AssertionError(
                f"Could not uniquely align {len(failures)} manifest rows to pre-encoded metadata. "
                f"Positional alignment is disabled. Examples:\n{preview}"
            )
    else:
        mode = "identifier"
    if len(set(positions)) != len(positions):
        raise AssertionError("Two manifest rows aligned to the same pre-encoded vector")
    aligned_metadata = metadata.iloc[positions].copy().reset_index(drop=True)
    aligned_vectors = np.asarray(vectors[positions], dtype=np.float32)
    return aligned_vectors, aligned_metadata, {"alignment_mode": mode, "n_aligned": len(positions)}
