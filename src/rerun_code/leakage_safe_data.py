"""Manifest construction and leakage checks for the JAMIA CXR rerun.

This module deliberately has no FAISS, PyTorch, or Hugging Face dependency so
that cohort and leakage audits can run before a GPU job is scheduled.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image


LABELS_13 = (
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged cardiomediastinum",
    "fracture",
    "lung lesion",
    "lung opacity",
    "pleural effusion",
    "pleural other",
    "pneumonia",
    "pneumothorax",
    "support devices",
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
_SPACE = re.compile(r"\s+")
_NEGATION = re.compile(
    r"\b(?:no|not|without|absent|negative\s+for|free\s+of|no\s+evidence\s+of)\b"
)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return _SPACE.sub(" ", str(value)).strip()


def normalize_report_text(text: Any) -> str:
    """Normalize text only for duplicate detection, not clinical labeling."""
    text = _clean(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return _SPACE.sub(" ", text).strip()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: Any) -> str:
    normalized = normalize_report_text(text)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def perceptual_hash(path: str | Path, size: int = 32, hash_size: int = 8) -> str:
    """Compute a deterministic 64-bit pHash without the imagehash package."""
    with Image.open(path) as image:
        arr = np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    indices = np.arange(size, dtype=np.float64)
    basis = np.cos((math.pi / size) * (indices[:, None] + 0.5) * indices[None, :])
    basis[:, 0] *= 1.0 / math.sqrt(2.0)
    basis *= math.sqrt(2.0 / size)
    dct = basis.T @ arr @ basis
    low = dct[:hash_size, :hash_size].copy()
    median = float(np.median(low.ravel()[1:]))
    bits = (low >= median).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    if not hash_a or not hash_b:
        return 10**9
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def normalize_labels(labels: Any) -> list[str]:
    if labels is None:
        return []
    if isinstance(labels, str):
        labels = [labels]
    mapping = {label.lower(): label for label in LABELS_13}
    normalized: list[str] = []
    for value in labels:
        key = _clean(value).lower().replace("_", " ")
        if key in mapping and key not in normalized:
            normalized.append(key)
    return normalized


def negated_source_label_flags(report_text: str, labels: Sequence[str]) -> list[str]:
    """Flag suspicious raw labels; this is not a clinical report labeler."""
    text = normalize_report_text(report_text)
    flagged: list[str] = []
    for label in labels:
        for match in re.finditer(rf"\b{re.escape(label)}\b", text):
            prefix = text[max(0, match.start() - 80) : match.start()]
            tokens = prefix.split()[-8:]
            if _NEGATION.search(" ".join(tokens)):
                flagged.append(label)
                break
    return sorted(set(flagged))


def _first(metadata: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = _clean(metadata.get(key))
        if value:
            return value
    return ""


def _report_fields(metadata: Mapping[str, Any]) -> tuple[str, str, str, str]:
    findings = _first(metadata, ("FINDINGS", "findings"))
    impression = _first(metadata, ("IMPRESSION", "impression"))
    caption = _first(metadata, ("Caption", "caption"))
    report = _first(metadata, ("report", "REPORT", "report_text"))
    if not report:
        report = _clean(" ".join(part for part in (findings, impression) if part))
    if not report:
        report = caption
    return findings, impression, caption, report


def _paired_json(image_path: Path) -> Path | None:
    direct = image_path.with_suffix(".json")
    if direct.exists():
        return direct
    candidates = list(image_path.parent.glob(f"{image_path.stem}.json"))
    return candidates[0] if candidates else None


def _identifiers(
    metadata: Mapping[str, Any], dataset: str, image_path: Path
) -> tuple[str, str, str, str, bool]:
    dataset_key = dataset.lower()
    image_id = _first(metadata, ("image_id", "image_name", "file_name")) or image_path.stem
    if dataset_key in {"mimic", "mimic-cxr", "mimic_cxr"}:
        patient_id = _first(metadata, ("subject_id", "patient_id"))
        study_id = _first(metadata, ("study_id", "report_id")) or image_path.stem
        source = "subject_id" if metadata.get("subject_id") is not None else "patient_id"
        reliable = bool(patient_id)
    else:
        patient_id = _first(metadata, ("patient_id", "subject_id"))
        if patient_id:
            source = "patient_id" if metadata.get("patient_id") is not None else "subject_id"
            reliable = True
        else:
            patient_id = _first(metadata, ("uid",))
            source = "uid_assumed_group" if patient_id else "missing"
            reliable = False
        study_id = _first(metadata, ("study_id", "uid", "report_id")) or image_path.stem
    return patient_id, study_id, image_id, source, reliable


def build_manifest(
    dataset_root: str | Path,
    dataset: str,
    split: str,
    *,
    compute_phash: bool = True,
    strict_pairs: bool = True,
) -> pd.DataFrame:
    """Build one row per image/JSON pair from a flat or nested dataset root."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    image_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No supported images found below {root}")

    records: list[dict[str, Any]] = []
    missing_json: list[str] = []
    for image_path in image_paths:
        json_path = _paired_json(image_path)
        if json_path is None:
            missing_json.append(str(image_path))
            continue
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Cannot parse metadata {json_path}: {exc}") from exc
        findings, impression, caption, report = _report_fields(metadata)
        patient_id, study_id, image_id, patient_source, patient_reliable = _identifiers(
            metadata, dataset, image_path
        )
        raw_labels = metadata.get("labels", [])
        if isinstance(raw_labels, str):
            raw_labels = [raw_labels]
        raw_labels = [_clean(value) for value in (raw_labels or []) if _clean(value)]
        labels_13 = normalize_labels(raw_labels)
        normal_value = _clean(metadata.get("normal")).lower()
        normal = normal_value if normal_value in {"yes", "no"} else ("yes" if not labels_13 else "no")
        records.append(
            {
                "dataset": dataset,
                "source_split": split,
                "patient_id": patient_id,
                "patient_key": f"{dataset}:{patient_id}" if patient_id else "",
                "patient_id_source": patient_source,
                "patient_id_reliable": bool(patient_reliable),
                "study_id": study_id,
                "image_id": image_id,
                "image_path": str(image_path),
                "json_path": str(json_path.resolve()),
                "view_position": _first(metadata, ("view_position", "ViewPosition", "view")),
                "normal": normal,
                "labels_raw": raw_labels,
                "labels_13": labels_13,
                "findings": findings,
                "impression": impression,
                "caption": caption,
                "report_text": report,
                "image_sha256": sha256_file(image_path),
                "image_phash": perceptual_hash(image_path) if compute_phash else "",
                "report_sha256": sha256_text(report),
                "negated_source_label_flags": negated_source_label_flags(report, labels_13),
            }
        )
    if strict_pairs and missing_json:
        preview = "\n".join(missing_json[:10])
        raise ValueError(f"{len(missing_json)} images have no paired JSON metadata. Examples:\n{preview}")
    if not records:
        raise ValueError(f"No image/JSON pairs found below {root}")
    frame = pd.DataFrame.from_records(records)
    frame.insert(0, "record_id", [f"{dataset}:{split}:{i:08d}" for i in range(len(frame))])
    return frame


def _nonempty_set(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame:
        return set()
    return {str(value) for value in frame[column].tolist() if _clean(value)}


def _reliable_patient_set(frame: pd.DataFrame) -> set[str]:
    if "patient_id" not in frame:
        return set()
    mask = frame.get("patient_id_reliable", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    column = "patient_key" if "patient_key" in frame else "patient_id"
    return {str(value) for value in frame.loc[mask, column].tolist() if _clean(value)}


def _examples(values: Iterable[str], limit: int = 20) -> list[str]:
    return sorted({str(value) for value in values})[:limit]


def overlap_audit(
    gallery: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    phash_threshold: int = 4,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Return exact and perceptual overlap counts between gallery and queries."""
    fields = ("study_id", "image_id", "image_sha256", "report_sha256")
    audit: dict[str, Any] = {
        "n_gallery": int(len(gallery)),
        "n_queries": int(len(queries)),
        "phash_threshold": int(phash_threshold),
    }
    patient_overlap = _reliable_patient_set(gallery) & _reliable_patient_set(queries)
    audit["reliable_patient_overlap_count"] = len(patient_overlap)
    audit["reliable_patient_overlap_examples"] = _examples(patient_overlap, max_examples)
    for field in fields:
        overlap = _nonempty_set(gallery, field) & _nonempty_set(queries, field)
        audit[f"{field}_overlap_count"] = len(overlap)
        audit[f"{field}_overlap_examples"] = _examples(overlap, max_examples)

    query_hashes = [value for value in queries.get("image_phash", []) if _clean(value)]
    near_pairs: list[dict[str, Any]] = []
    if query_hashes and "image_phash" in gallery:
        query_by_hash: dict[str, list[str]] = defaultdict(list)
        for _, row in queries.iterrows():
            if _clean(row.get("image_phash")):
                query_by_hash[str(row["image_phash"])].append(str(row.get("record_id", "")))
        for _, grow in gallery.iterrows():
            gallery_hash = _clean(grow.get("image_phash"))
            if not gallery_hash:
                continue
            for query_hash, query_ids in query_by_hash.items():
                distance = hamming_distance(gallery_hash, query_hash)
                if distance <= phash_threshold:
                    near_pairs.append(
                        {
                            "gallery_record_id": str(grow.get("record_id", "")),
                            "query_record_id": query_ids[0],
                            "hamming_distance": int(distance),
                        }
                    )
                    if len(near_pairs) >= max_examples:
                        break
            if len(near_pairs) >= max_examples:
                break
    audit["near_image_overlap_at_least_count"] = len(near_pairs)
    audit["near_image_overlap_examples"] = near_pairs
    audit["near_image_overlap_count_is_capped"] = len(near_pairs) >= max_examples
    return audit


def remove_query_conflicts_from_gallery(
    gallery: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    phash_threshold: int = 4,
    remove_assumed_patient_groups: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the fixed query cohort and exclude every conflicting gallery row."""
    query_sets = {
        field: _nonempty_set(queries, field)
        for field in ("study_id", "image_id", "image_sha256", "report_sha256")
    }
    reliable_query_patients = _reliable_patient_set(queries)
    patient_column = "patient_key" if "patient_key" in queries and "patient_key" in gallery else "patient_id"
    any_query_patients = _nonempty_set(queries, patient_column)
    query_phashes = [str(value) for value in queries.get("image_phash", []) if _clean(value)]

    keep_indices: list[int] = []
    removed: list[dict[str, Any]] = []
    for index, row in gallery.iterrows():
        reasons: list[str] = []
        patient_id = _clean(row.get("patient_id"))
        patient_key = _clean(row.get(patient_column))
        reliable = bool(row.get("patient_id_reliable", False))
        if patient_key and reliable and patient_key in reliable_query_patients:
            reasons.append("reliable_patient_overlap")
        elif patient_key and remove_assumed_patient_groups and patient_key in any_query_patients:
            reasons.append("assumed_patient_group_overlap")
        for field, query_values in query_sets.items():
            value = _clean(row.get(field))
            if value and value in query_values:
                reasons.append(f"{field}_overlap")
        gallery_phash = _clean(row.get("image_phash"))
        if gallery_phash and query_phashes:
            minimum = min(hamming_distance(gallery_phash, value) for value in query_phashes)
            if minimum <= phash_threshold:
                reasons.append(f"image_phash_distance<={phash_threshold}")
        if reasons:
            removed.append(
                {
                    "record_id": str(row.get("record_id", "")),
                    "image_id": str(row.get("image_id", "")),
                    "patient_id": patient_id,
                    "reasons": sorted(set(reasons)),
                }
            )
        else:
            keep_indices.append(index)
    clean = gallery.loc[keep_indices].copy().reset_index(drop=True)
    return clean, pd.DataFrame.from_records(removed)


def assert_leakage_free(
    gallery: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    phash_threshold: int = 4,
) -> dict[str, Any]:
    audit = overlap_audit(gallery, queries, phash_threshold=phash_threshold, max_examples=1)
    failures = []
    for key in (
        "reliable_patient_overlap_count",
        "study_id_overlap_count",
        "image_id_overlap_count",
        "image_sha256_overlap_count",
        "report_sha256_overlap_count",
        "near_image_overlap_at_least_count",
    ):
        if int(audit.get(key, 0)):
            failures.append(f"{key}={audit[key]}")
    if failures:
        raise AssertionError("Leakage gate failed: " + ", ".join(failures))
    return audit


def write_jsonl(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in frame.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def read_jsonl(path: str | Path) -> pd.DataFrame:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame.from_records(records)


def manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)


def cohort_summary(frame: pd.DataFrame) -> dict[str, Any]:
    patient_values = _nonempty_set(frame, "patient_id")
    reliable_mask = frame.get(
        "patient_id_reliable", pd.Series(False, index=frame.index, dtype=bool)
    ).fillna(False).astype(bool)
    reliable_patient_values = {
        str(value)
        for value in frame.loc[reliable_mask, "patient_id"].tolist()
        if _clean(value)
    }
    label_counts = Counter(label for labels in frame.get("labels_13", []) for label in labels)
    return {
        "n_images": int(len(frame)),
        "n_unique_image_ids": int(len(_nonempty_set(frame, "image_id"))),
        "n_unique_studies": int(len(_nonempty_set(frame, "study_id"))),
        "n_patients_or_groups": int(len(patient_values)),
        "n_unique_reliable_patients": int(len(reliable_patient_values)),
        "n_missing_patient_id": int(sum(not _clean(value) for value in frame.get("patient_id", []))),
        "n_records_with_reliable_patient_id": int(
            frame.get("patient_id_reliable", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "n_normal": int((frame.get("normal", pd.Series(dtype=str)) == "yes").sum()),
        "n_abnormal": int((frame.get("normal", pd.Series(dtype=str)) == "no").sum()),
        "n_label_quality_flags": int(sum(bool(value) for value in frame.get("negated_source_label_flags", []))),
        "label_prevalence_counts": dict(sorted(label_counts.items())),
    }


def describe_removals(removals: pd.DataFrame) -> dict[str, Any]:
    if removals.empty:
        return {"n_removed": 0, "reason_counts": {}}
    counts = Counter(reason for reasons in removals["reasons"] for reason in reasons)
    return {"n_removed": int(len(removals)), "reason_counts": dict(sorted(counts.items()))}
