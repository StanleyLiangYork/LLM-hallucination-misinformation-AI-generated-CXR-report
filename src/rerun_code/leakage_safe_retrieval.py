"""FAISS bundle and defense-in-depth retrieval helpers.

FAISS is imported lazily so the manifest notebook can run without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .leakage_safe_data import hamming_distance


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("Install faiss-cpu or a compatible GPU FAISS build on Biowulf") from exc
    return faiss


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def write_metadata(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return output


def read_metadata(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_gallery_bundle(
    embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> Path:
    faiss = _faiss()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vectors = l2_normalize(embeddings)
    if len(vectors) != len(metadata):
        raise ValueError("Embedding and metadata row counts differ")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(output / "gallery.index"))
    np.save(output / "gallery_embeddings.npy", vectors)
    write_metadata(metadata, output / "gallery_metadata.jsonl")
    return output


def save_query_bundle(
    embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    vectors = l2_normalize(embeddings)
    if len(vectors) != len(metadata):
        raise ValueError("Embedding and metadata row counts differ")
    np.save(output / "query_embeddings.npy", vectors)
    write_metadata(metadata, output / "query_metadata.jsonl")
    return output


def load_gallery_bundle(path: str | Path):
    faiss = _faiss()
    root = Path(path)
    index = faiss.read_index(str(root / "gallery.index"))
    metadata = read_metadata(root / "gallery_metadata.jsonl")
    if index.ntotal != len(metadata):
        raise AssertionError("Gallery index and metadata row counts differ")
    return index, metadata


def load_query_bundle(path: str | Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    root = Path(path)
    vectors = np.load(root / "query_embeddings.npy").astype(np.float32)
    metadata = read_metadata(root / "query_metadata.jsonl")
    if len(vectors) != len(metadata):
        raise AssertionError("Query embeddings and metadata row counts differ")
    return vectors, metadata


def ineligible_neighbor_reasons(
    query: Mapping[str, Any],
    neighbor: Mapping[str, Any],
    *,
    phash_threshold: int = 4,
) -> list[str]:
    reasons: list[str] = []
    for field in ("record_id", "study_id", "image_id", "image_sha256", "report_sha256"):
        qvalue = str(query.get(field, "") or "")
        nvalue = str(neighbor.get(field, "") or "")
        if qvalue and nvalue and qvalue == nvalue:
            reasons.append(f"same_{field}")
    patient_field = "patient_key" if query.get("patient_key") or neighbor.get("patient_key") else "patient_id"
    qpatient = str(query.get(patient_field, "") or "")
    npatient = str(neighbor.get(patient_field, "") or "")
    if qpatient and npatient and qpatient == npatient:
        reasons.append("same_patient_or_group")
    qhash = str(query.get("image_phash", "") or "")
    nhash = str(neighbor.get("image_phash", "") or "")
    if qhash and nhash and hamming_distance(qhash, nhash) <= phash_threshold:
        reasons.append(f"image_phash_distance<={phash_threshold}")
    return reasons


def safe_search(
    index,
    gallery_metadata: Sequence[Mapping[str, Any]],
    query_vector: np.ndarray,
    query_metadata: Mapping[str, Any],
    *,
    k: int = 10,
    phash_threshold: int = 4,
) -> list[dict[str, Any]]:
    """Search, filter all forbidden matches, and fail if fewer than k remain."""
    if k <= 0:
        raise ValueError("k must be positive")
    total = int(index.ntotal)
    if total < k:
        raise ValueError(f"Gallery contains {total} entries but k={k}")
    vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    vector = l2_normalize(vector)
    requested = min(total, max(k + 64, k * 5))
    eligible: list[dict[str, Any]] = []
    seen: set[int] = set()
    while True:
        scores, indices = index.search(vector, requested)
        for score, position in zip(scores[0], indices[0]):
            position = int(position)
            if position < 0 or position in seen:
                continue
            seen.add(position)
            neighbor = dict(gallery_metadata[position])
            reasons = ineligible_neighbor_reasons(
                query_metadata, neighbor, phash_threshold=phash_threshold
            )
            if reasons:
                continue
            neighbor["gallery_position"] = position
            neighbor["cosine_score"] = float(score)
            eligible.append(neighbor)
            if len(eligible) == k:
                return eligible
        if requested == total:
            break
        requested = min(total, requested * 2)
    raise AssertionError(
        f"Only {len(eligible)} eligible neighbors found for query "
        f"{query_metadata.get('record_id', '<unknown>')} (k={k})"
    )


def assert_neighbor_records(
    query: Mapping[str, Any],
    neighbors: Sequence[Mapping[str, Any]],
    *,
    expected_k: int,
    phash_threshold: int = 4,
) -> None:
    if len(neighbors) != expected_k:
        raise AssertionError(f"Expected {expected_k} neighbors, found {len(neighbors)}")
    for rank, neighbor in enumerate(neighbors, start=1):
        reasons = ineligible_neighbor_reasons(query, neighbor, phash_threshold=phash_threshold)
        if reasons:
            raise AssertionError(f"Ineligible neighbor at rank {rank}: {reasons}")
