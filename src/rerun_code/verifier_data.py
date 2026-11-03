from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


_SPACE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    text = _SPACE.sub(" ", str(value or "")).strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def text_sha256(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _nested_first(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    value = _first(row, keys)
    if value:
        return value
    for container_key in ("meta", "metadata", "provenance", "source"):
        container = row.get(container_key)
        if isinstance(container, Mapping):
            value = _first(container, keys)
            if value:
                return value
    return ""


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, Mapping):
                pieces.append(_content_text(part.get("text", part.get("content"))))
            else:
                pieces.append(_content_text(part))
        return "\n".join(piece for piece in pieces if piece).strip()
    if isinstance(content, Mapping):
        return _content_text(content.get("text", content.get("content")))
    return str(content).strip()


def _messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        harmony = row.get("harmony_prompt")
        if isinstance(harmony, str):
            try:
                decoded = json.loads(harmony)
                raw_messages = decoded if isinstance(decoded, list) else []
            except json.JSONDecodeError:
                raw_messages = []
        elif isinstance(harmony, list):
            raw_messages = harmony
        else:
            raw_messages = []
    result = []
    for message in raw_messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", message.get("from", "")) or "").strip().lower()
        if role in {"human", "prompt"}:
            role = "user"
        elif role in {"gpt", "bot", "model"}:
            role = "assistant"
        content = _content_text(message.get("content"))
        if role and content:
            result.append({"role": role, "content": content})
    return result


def _verdict_value(row: Mapping[str, Any], messages: Sequence[Mapping[str, str]]) -> tuple[Any, str]:
    for key in ("label", "verdict", "answer", "output"):
        if key in row and row[key] is not None:
            return row[key], key
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, Mapping):
        for key in ("answer", "label", "verdict", "value"):
            if key in ground_truth and ground_truth[key] is not None:
                return ground_truth[key], f"ground_truth.{key}"
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content"), "messages.assistant"
    return None, "missing"


def _parse_verdict(value: Any, *, row_index: int, source: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    token = "" if value is None else str(value).strip().lower()
    positive = {"true", "1", "yes", "correct", "consistent", "supported"}
    negative = {"false", "0", "no", "incorrect", "inconsistent", "unsupported"}
    if token in positive:
        return True
    if token in negative:
        return False
    match = re.fullmatch(r"(?:answer|label|verdict)\s*[:=]\s*(true|false)", token)
    if match:
        return match.group(1) == "true"
    raise ValueError(
        f"Unrecognized verifier label at row {row_index} from {source}: {value!r}. "
        "Expected ground_truth.answer, a top-level binary target, or an assistant TRUE/FALSE response."
    )


def standardize_verifier_records(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        row = dict(raw)
        messages = _messages(row)
        report = _nested_first(row, ("report", "report_text", "candidate_report", "text", "input"))
        if not report:
            user_messages = [message["content"] for message in messages if message["role"] == "user"]
            report = user_messages[-1] if user_messages else ""
        if not report:
            raise ValueError(f"Verifier row {index} contains no usable user/evidence-statement text")
        verdict_raw, verdict_source = _verdict_value(row, messages)
        verdict = _parse_verdict(verdict_raw, row_index=index, source=verdict_source)
        patient_id = _nested_first(row, ("patient_id", "subject_id", "patient", "uid"))
        source_id = _nested_first(
            row,
            ("source_record_id", "source_report_id", "report_id", "record_id", "image_id", "study_id", "file_name", "uid"),
        )
        group_key = patient_id or source_id
        if not group_key:
            raise ValueError(f"Verifier row {index} lacks patient/source provenance")
        original_id = _first(row, ("id", "example_id"))
        output.append({
            "verifier_record_id": original_id or f"verifier:{index:08d}",
            "patient_or_source_group": group_key,
            "grouping_level": "patient" if patient_id else "source_report_or_image",
            "patient_id": patient_id,
            "source_record_id": source_id,
            "report_text": report,
            "image_path": _nested_first(row, ("image_path", "path", "image")),
            "report_sha256": text_sha256(report),
            "verdict": verdict,
            "verdict_source": verdict_source,
            "task": _first(row, ("task",)),
            "raw": row,
        })
    frame = pd.DataFrame(output)
    if frame.empty:
        raise ValueError("No verifier records")
    return frame


def remove_evaluation_overlap(frame: pd.DataFrame, evaluation_records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation_patients = set()
    for field in ("patient_key", "patient_id"):
        for value in evaluation_records.get(field, []):
            cleaned = str(value or "").strip()
            if cleaned:
                evaluation_patients.add(cleaned)
                evaluation_patients.add(cleaned.split(":", 1)[-1])
    evaluation_sources = set(str(v) for field in ("record_id", "study_id", "image_id") for v in evaluation_records.get(field, []) if str(v or "").strip())
    evaluation_reports = set(str(v) for v in evaluation_records.get("report_sha256", []) if str(v or "").strip())
    reasons: list[list[str]] = []
    for _, row in frame.iterrows():
        row_reasons: list[str] = []
        group = str(row["patient_or_source_group"])
        patient = str(row.get("patient_id", "") or "")
        if group in evaluation_patients or patient in evaluation_patients:
            row_reasons.append("evaluation_patient_or_group")
        if str(row["source_record_id"]) in evaluation_sources:
            row_reasons.append("evaluation_source_id")
        if str(row["report_sha256"]) in evaluation_reports:
            row_reasons.append("evaluation_report_hash")
        reasons.append(row_reasons)
    mask = pd.Series([bool(value) for value in reasons], index=frame.index)
    excluded = frame.loc[mask].copy()
    if not excluded.empty:
        excluded["exclusion_reasons"] = [value for value, keep in zip(reasons, mask) if keep]
    return frame.loc[~mask].copy().reset_index(drop=True), excluded.reset_index(drop=True)


def grouped_stratified_split(
    frame: pd.DataFrame,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    *,
    seed: int = 20260831,
) -> pd.DataFrame:
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must be three values summing to one")
    groups = []
    for key, group in frame.groupby("patient_or_source_group", sort=True):
        groups.append((str(key), int(len(group)), float(group["verdict"].mean())))
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda item: item[1], reverse=True)
    target = [len(frame) * f for f in fractions]
    assignments: dict[str, str] = {}
    counts = [0, 0, 0]
    names = ["train", "validation", "test"]
    for key, size, rate in groups:
        remaining = [target[idx] - counts[idx] for idx in range(3)]
        fitting = [idx for idx in range(3) if remaining[idx] >= size]
        pool = fitting or list(range(3))
        # Fill the split with the largest absolute remaining capacity. This
        # respects 80/10/10 even when source groups contain multiple examples.
        chosen = max(pool, key=lambda idx: (remaining[idx], -counts[idx], -idx))
        assignments[key] = names[chosen]
        counts[chosen] += size
    result = frame.copy()
    result["split"] = result["patient_or_source_group"].map(assignments)
    assert_group_disjoint(result)
    return result


def assert_group_disjoint(frame: pd.DataFrame) -> None:
    seen: dict[str, str] = {}
    for _, row in frame.iterrows():
        group, split = str(row["patient_or_source_group"]), str(row["split"])
        if group in seen and seen[group] != split:
            raise AssertionError(f"Group {group!r} appears in {seen[group]} and {split}")
        seen[group] = split


def split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, group in frame.groupby("split"):
        result[str(split)] = {
            "n_examples": int(len(group)),
            "n_groups": int(group["patient_or_source_group"].nunique()),
            "positive": int(group["verdict"].sum()),
            "negative": int((~group["verdict"].astype(bool)).sum()),
            "grouping_levels": frame.loc[group.index, "grouping_level"].value_counts().to_dict() if "grouping_level" in frame else {},
        }
    return result


def verifier_prompt(report: str, evidence: str | None = None) -> str:
    report = report.strip()
    if evidence is not None:
        body = f"EVIDENCE\n{evidence.strip()}\n\nSTATEMENT:\n{report}"
    elif "EVIDENCE" in report.upper() and "STATEMENT" in report.upper():
        body = report
    else:
        body = f"STATEMENT:\n{report}"
    return (
        "Decide whether the STATEMENT is supported by the EVIDENCE. "
        "Output exactly TRUE or FALSE.\n\n"
        f"{body}\n\nANSWER:"
    )
