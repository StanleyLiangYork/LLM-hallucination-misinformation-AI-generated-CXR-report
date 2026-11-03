"""Post-rerun analyses requested during peer review.

The functions in this module operate only on the frozen Notebook 07 cohort.
They verify its checksum before analysis, preserve patient/source clustering,
and expose deterministic helpers used by Notebooks 11 and 12.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import sha256_path
from .report_labeler import LABELS_13
from .statistics import (
    _cluster_sufficient_statistics,
    _metrics_from_statistics,
    _multinomial_cluster_weights,
    _sum_statistics,
    _weighted_statistics,
    holm_adjust,
)


POST_RERUN_API_VERSION = 1
PRIMARY_METRICS = ("fer_abnormal", "omission", "macro_f1", "radgraph_f1")
CONDITIONS = (
    "A_single_pass",
    "B_unconditional_4pass",
    "C_pretrained_gate",
    "D_corrected_lora_gate",
)


def verified_per_study_path(paths: Mapping[str, Path]) -> tuple[Path, dict[str, object]]:
    """Return the Notebook 07 per-study file after provenance verification."""
    status07_path = Path(paths["metrics"]) / "notebook07_status.json"
    status08_path = Path(paths["statistics"]) / "notebook08_status.json"
    if not status07_path.exists() or not status08_path.exists():
        raise FileNotFoundError("Run Notebooks 07 and 08 before post-rerun analysis.")
    status07 = json.loads(status07_path.read_text(encoding="utf-8"))
    status08 = json.loads(status08_path.read_text(encoding="utf-8"))
    if not status07.get("ready") or not status08.get("ready"):
        raise RuntimeError(
            "Post-rerun analysis requires ready=true in both Notebook 07 and 08 status files."
        )
    source = Path(str(status07.get("per_study_metrics", "")))
    expected = str(status07.get("per_study_metrics_sha256", ""))
    if not source.exists() or not expected:
        raise FileNotFoundError("Notebook 07 per-study metrics or checksum is unavailable.")
    actual = sha256_path(source)
    if actual != expected:
        raise AssertionError(
            f"Notebook 07 per-study checksum mismatch: expected {expected}, observed {actual}."
        )
    if str(status08.get("upstream_per_study_metrics_sha256", "")) != actual:
        raise AssertionError("Notebook 08 is stale relative to the current Notebook 07 cohort.")
    return source, {
        "notebook07_status": str(status07_path),
        "notebook08_status": str(status08_path),
        "per_study_metrics": str(source),
        "per_study_metrics_sha256": actual,
    }


def read_jsonl_fields(path: str | Path, fields: Sequence[str]) -> pd.DataFrame:
    """Read selected JSONL fields and fail on malformed or incomplete rows."""
    records: list[dict[str, object]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {source}:{line_number}: {exc}") from exc
            missing = [field for field in fields if field not in row]
            if missing:
                raise KeyError(f"Missing fields at {source}:{line_number}: {missing}")
            records.append({field: row[field] for field in fields})
    frame = pd.DataFrame(records)
    if "generation_record_id" in frame and frame["generation_record_id"].duplicated().any():
        duplicate = frame.loc[frame["generation_record_id"].duplicated(), "generation_record_id"].iloc[0]
        raise AssertionError(f"Duplicate generation_record_id: {duplicate}")
    return frame


def load_statistical_frame(path: str | Path) -> pd.DataFrame:
    fields = (
        "generation_record_id",
        "query_record_id",
        "model_key",
        "bundle",
        "source_dataset",
        "condition",
        "patient_key",
        "reference_vector",
        "prediction_vector",
        "radgraph_f1",
    )
    frame = read_jsonl_fields(path, fields)
    frame["radgraph_f1"] = pd.to_numeric(frame["radgraph_f1"], errors="coerce")
    observed = set(frame["condition"].unique())
    if observed != set(CONDITIONS):
        raise AssertionError(f"Unexpected condition set: {sorted(observed)}")
    for column in ("reference_vector", "prediction_vector"):
        bad = frame[column].map(lambda value: len(value) != len(LABELS_13))
        if bad.any():
            raise AssertionError(f"{column} is not a {len(LABELS_13)}-element vector")
    return frame


def paired_cluster_bootstrap_difference(
    frame_b: pd.DataFrame,
    frame_c: pd.DataFrame,
    *,
    metrics: Sequence[str] = PRIMARY_METRICS,
    replicates: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 20260831,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Paired cluster-bootstrap confidence intervals for C minus B."""
    shared = ["query_record_id", "reference_vector", "prediction_vector", "radgraph_f1", "patient_key"]
    merged = frame_b[shared].merge(
        frame_c[shared], on="query_record_id", suffixes=("_b", "_c"), validate="one_to_one"
    )
    if len(merged) != len(frame_b) or len(merged) != len(frame_c):
        raise AssertionError("B-C bootstrap requires identical query sets")
    reference_b = np.asarray(merged["reference_vector_b"].tolist(), dtype=int)
    reference_c = np.asarray(merged["reference_vector_c"].tolist(), dtype=int)
    if not np.array_equal(reference_b, reference_c):
        raise AssertionError("B and C disagree on reference vectors")
    patient_b = merged["patient_key_b"].fillna(merged["query_record_id"]).astype(str)
    patient_c = merged["patient_key_c"].fillna(merged["query_record_id"]).astype(str)
    if not patient_b.equals(patient_c):
        raise AssertionError("B and C disagree on patient/source cluster assignments")
    codes, cluster_ids = pd.factorize(patient_b, sort=False)
    prediction_b = np.asarray(merged["prediction_vector_b"].tolist(), dtype=int)
    prediction_c = np.asarray(merged["prediction_vector_c"].tolist(), dtype=int)
    optional_b = {"radgraph_f1": merged["radgraph_f1_b"].to_numpy(dtype=float)}
    optional_c = {"radgraph_f1": merged["radgraph_f1_c"].to_numpy(dtype=float)}
    stats_b = _cluster_sufficient_statistics(reference_b, prediction_b, codes, len(cluster_ids), optional_b)
    stats_c = _cluster_sufficient_statistics(reference_b, prediction_c, codes, len(cluster_ids), optional_c)
    point_b = _metrics_from_statistics(_sum_statistics(stats_b))
    point_c = _metrics_from_statistics(_sum_statistics(stats_c))
    draws = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        weights = _multinomial_cluster_weights(rng, size, len(cluster_ids))
        values_b = _metrics_from_statistics(_weighted_statistics(stats_b, weights))
        values_c = _metrics_from_statistics(_weighted_statistics(stats_c, weights))
        for metric in metrics:
            draws[metric].append(values_c[metric] - values_b[metric])
    alpha = (1.0 - confidence_level) / 2.0
    rows = []
    for metric in metrics:
        values = np.concatenate(draws[metric])
        valid = values[np.isfinite(values)]
        rows.append({
            "metric": metric,
            "difference_c_minus_b": float(point_c[metric][0] - point_b[metric][0]),
            "difference_ci_low": float(np.quantile(valid, alpha)) if len(valid) else np.nan,
            "difference_ci_high": float(np.quantile(valid, 1.0 - alpha)) if len(valid) else np.nan,
            "valid_bootstrap_replicates": int(len(valid)),
            "n_clusters": int(len(cluster_ids)),
        })
    return pd.DataFrame(rows)


def _add_at(target: np.ndarray, codes: np.ndarray, values: np.ndarray) -> None:
    np.add.at(target, codes, values)


def bootstrap_cluster_counts(
    cluster_counts: np.ndarray,
    formulas: Mapping[str, tuple[int, int] | tuple[int, int, int]],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    batch_size: int = 256,
) -> dict[str, tuple[float, float, float, int]]:
    totals = cluster_counts.sum(axis=0)
    point: dict[str, float] = {}
    for name, formula in formulas.items():
        if len(formula) == 2:
            numerator, denominator = formula
            point[name] = totals[numerator] / totals[denominator] if totals[denominator] else np.nan
        else:
            positive, negative, denominator = formula
            point[name] = (
                (totals[positive] - totals[negative]) / totals[denominator]
                if totals[denominator] else np.nan
            )
    draws = {name: [] for name in formulas}
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        weights = _multinomial_cluster_weights(rng, size, len(cluster_counts))
        sampled = weights @ cluster_counts
        for name, formula in formulas.items():
            if len(formula) == 2:
                numerator, denominator = formula
                values = np.divide(
                    sampled[:, numerator], sampled[:, denominator],
                    out=np.full(size, np.nan), where=sampled[:, denominator] != 0,
                )
            else:
                positive, negative, denominator = formula
                values = np.divide(
                    sampled[:, positive] - sampled[:, negative], sampled[:, denominator],
                    out=np.full(size, np.nan), where=sampled[:, denominator] != 0,
                )
            draws[name].append(values)
    alpha = (1.0 - confidence_level) / 2.0
    output: dict[str, tuple[float, float, float, int]] = {}
    for name, pieces in draws.items():
        values = np.concatenate(pieces)
        valid = values[np.isfinite(values)]
        output[name] = (
            float(point[name]),
            float(np.quantile(valid, alpha)) if len(valid) else np.nan,
            float(np.quantile(valid, 1.0 - alpha)) if len(valid) else np.nan,
            int(len(valid)),
        )
    return output


def clustered_sign_flip_p(
    cluster_net: np.ndarray, *, replicates: int, seed: int, batch_size: int = 512
) -> float:
    values = np.asarray(cluster_net, dtype=float)
    observed = abs(values.sum())
    rng = np.random.default_rng(seed)
    exceed = valid = 0
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        signs = rng.integers(0, 2, size=(size, len(values))) * 2 - 1
        exceed += int((np.abs(signs @ values) >= observed - 1e-15).sum())
        valid += size
    return (exceed + 1) / (valid + 1)


def make_joint_inputs(group: pd.DataFrame) -> dict[str, object]:
    parts = {}
    for condition in CONDITIONS[1:]:
        parts[condition] = (
            group[group["condition"] == condition]
            .sort_values("query_record_id")
            .reset_index(drop=True)
        )
    b, c, d = (parts[condition] for condition in CONDITIONS[1:])
    if not (b["query_record_id"].equals(c["query_record_id"]) and b["query_record_id"].equals(d["query_record_id"])):
        raise AssertionError("Joint analysis requires identical query order in B, C, and D")
    reference = np.asarray(b["reference_vector"].tolist(), dtype=int)
    for arm in (c, d):
        if not np.array_equal(reference, np.asarray(arm["reference_vector"].tolist(), dtype=int)):
            raise AssertionError("Joint-analysis arms disagree on reference vectors")
    predictions = [np.asarray(arm["prediction_vector"].tolist(), dtype=int) for arm in (b, c, d)]
    wrong_b, wrong_c, wrong_d = [prediction != reference for prediction in predictions]
    patient = b["patient_key"].fillna(b["query_record_id"]).astype(str)
    codes, cluster_ids = pd.factorize(patient, sort=False)
    return {
        "base": b,
        "wrong_b": wrong_b,
        "wrong_c": wrong_c,
        "wrong_d": wrong_d,
        "rescue": wrong_b & wrong_c & ~wrong_d,
        "harm": ~wrong_b & ~wrong_c & wrong_d,
        "err_b": wrong_b.sum(axis=1),
        "err_c": wrong_c.sum(axis=1),
        "err_d": wrong_d.sum(axis=1),
        "codes": codes,
        "cluster_ids": cluster_ids,
    }


JOINT_COUNT_COLUMNS = (
    "event_rescue_n", "event_harm_n", "joint_wrong_event_opportunities",
    "joint_correct_event_opportunities", "n_label_events", "strict_study_rescue_n",
    "strict_study_harm_n", "joint_wrong_study_opportunities",
    "joint_correct_study_opportunities", "dominant_study_rescue_n",
    "dominant_study_harm_n", "n_records",
)

JOINT_FORMULAS = {
    "event_rescue_rate_all": (0, 4),
    "event_harm_rate_all": (1, 4),
    "event_rescue_rate_opportunity": (0, 2),
    "event_harm_rate_opportunity": (1, 3),
    "event_net_rate_all": (0, 1, 4),
    "strict_study_rescue_rate_all": (5, 11),
    "strict_study_harm_rate_all": (6, 11),
    "strict_study_rescue_rate_opportunity": (5, 7),
    "strict_study_harm_rate_opportunity": (6, 8),
    "strict_study_net_rate_all": (5, 6, 11),
    "dominant_study_rescue_rate_all": (9, 11),
    "dominant_study_harm_rate_all": (10, 11),
}


def joint_stratum_statistics(
    inputs: Mapping[str, object], *, replicates: int, confidence_level: float, seed: int
) -> tuple[dict[str, object], np.ndarray]:
    rescue = np.asarray(inputs["rescue"], dtype=bool)
    harm = np.asarray(inputs["harm"], dtype=bool)
    wrong_b = np.asarray(inputs["wrong_b"], dtype=bool)
    wrong_c = np.asarray(inputs["wrong_c"], dtype=bool)
    err_b = np.asarray(inputs["err_b"], dtype=int)
    err_c = np.asarray(inputs["err_c"], dtype=int)
    err_d = np.asarray(inputs["err_d"], dtype=int)
    n_records = len(err_b)
    row_counts = np.column_stack([
        rescue.sum(axis=1), harm.sum(axis=1),
        (wrong_b & wrong_c).sum(axis=1), (~wrong_b & ~wrong_c).sum(axis=1),
        np.full(n_records, len(LABELS_13)),
        (err_b > 0) & (err_c > 0) & (err_d == 0),
        (err_b == 0) & (err_c == 0) & (err_d > 0),
        (err_b > 0) & (err_c > 0), (err_b == 0) & (err_c == 0),
        err_d < np.minimum(err_b, err_c), err_d > np.maximum(err_b, err_c),
        np.ones(n_records),
    ]).astype(float)
    codes = np.asarray(inputs["codes"], dtype=int)
    cluster_counts = np.zeros((len(inputs["cluster_ids"]), row_counts.shape[1]), dtype=float)
    _add_at(cluster_counts, codes, row_counts)
    estimates = bootstrap_cluster_counts(
        cluster_counts, JOINT_FORMULAS,
        replicates=replicates, confidence_level=confidence_level, seed=seed,
    )
    totals = cluster_counts.sum(axis=0)
    row: dict[str, object] = {
        "n_clusters": int(len(cluster_counts)),
        **{name: int(value) for name, value in zip(JOINT_COUNT_COLUMNS, totals)},
        "event_net_signflip_p": clustered_sign_flip_p(
            cluster_counts[:, 0] - cluster_counts[:, 1], replicates=replicates, seed=seed
        ),
        "strict_study_net_signflip_p": clustered_sign_flip_p(
            cluster_counts[:, 5] - cluster_counts[:, 6], replicates=replicates, seed=seed
        ),
    }
    for name, (estimate, low, high, valid) in estimates.items():
        row[name] = estimate
        row[f"{name}_ci_low"] = low
        row[f"{name}_ci_high"] = high
        row[f"{name}_valid_bootstrap"] = valid
    return row, row_counts


def pooled_cluster_counts(frame: pd.DataFrame, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clusters = frame["patient_key"].fillna(frame["query_record_id"]).astype(str)
    codes, ids = pd.factorize(clusters, sort=False)
    output = np.zeros((len(ids), values.shape[1]), dtype=float)
    _add_at(output, codes, values)
    return output, ids


def summarize_pooled_joint(
    subset: pd.DataFrame,
    scope: str,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    counts, cluster_ids = pooled_cluster_counts(
        subset, subset[list(JOINT_COUNT_COLUMNS)].to_numpy(dtype=float)
    )
    estimates = bootstrap_cluster_counts(
        counts, JOINT_FORMULAS,
        replicates=replicates, confidence_level=confidence_level, seed=seed,
    )
    totals = counts.sum(axis=0)
    result: dict[str, object] = {
        "scope": scope,
        "n_clusters": int(len(cluster_ids)),
        **{name: int(value) for name, value in zip(JOINT_COUNT_COLUMNS, totals)},
        "event_net_signflip_p": clustered_sign_flip_p(
            counts[:, 0] - counts[:, 1], replicates=replicates, seed=seed
        ),
        "strict_study_net_signflip_p": clustered_sign_flip_p(
            counts[:, 5] - counts[:, 6], replicates=replicates, seed=seed
        ),
    }
    for name, (estimate, low, high, valid) in estimates.items():
        result[name] = estimate
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
        result[f"{name}_valid_bootstrap"] = valid
    return result


def qualitative_case_sample(
    cases: pd.DataFrame, *, sample_size: int = 60, seed: int = 20260831
) -> pd.DataFrame:
    """Select a deterministic, rescue/harm-balanced and diversity-oriented sample."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    work = cases.copy()
    tags = work["transition_tags"].fillna("").astype(str)
    work["review_direction"] = np.where(tags.str.contains("rescue"), "rescue", "harm")
    work["strict_priority"] = np.where(tags.str.contains("strict_"), 0, 1)
    rng = np.random.default_rng(seed)
    work["random_order"] = rng.random(len(work))
    selected = []
    rescue_target = min((sample_size + 1) // 2, int((work["review_direction"] == "rescue").sum()))
    harm_target = min(sample_size - rescue_target, int((work["review_direction"] == "harm").sum()))
    for direction, target in (("rescue", rescue_target), ("harm", harm_target)):
        pool = work[work["review_direction"] == direction].copy()
        pool["diversity_rank"] = pool.groupby(
            ["model_key", "source_dataset"], dropna=False
        )["random_order"].rank(method="first")
        pool = pool.sort_values(
            ["strict_priority", "diversity_rank", "model_key", "source_dataset", "random_order"]
        )
        selected.append(pool.head(target))
    result = pd.concat(selected, ignore_index=True)
    if len(result) < sample_size:
        used = set(result["case_key"])
        fill = work[~work["case_key"].isin(used)].sort_values(
            ["strict_priority", "random_order"]
        ).head(sample_size - len(result))
        result = pd.concat([result, fill], ignore_index=True)
    result = result.sort_values(["review_direction", "strict_priority", "random_order"]).reset_index(drop=True)
    result["blinded_case_id"] = [f"QR-{index:04d}" for index in range(1, len(result) + 1)]
    return result


def build_blinded_qualitative_materials(
    per_study_path: str | Path,
    selected_cases: pd.DataFrame,
    output_dir: str | Path,
    *,
    seed: int = 20260831,
) -> dict[str, str]:
    """Create two blinded case-review forms and an analyst-only crosswalk."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_keys = set(selected_cases["case_key"].astype(str))
    required_conditions = set(CONDITIONS[1:])
    records: dict[tuple[str, str], dict[str, object]] = {}
    with Path(per_study_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            condition = str(row.get("condition", ""))
            if condition not in required_conditions:
                continue
            case_key = "|".join(str(row.get(name, "")) for name in (
                "model_key", "bundle", "source_dataset", "query_record_id"
            ))
            if case_key in selected_keys:
                records[(case_key, condition)] = row
    expected = {(key, condition) for key in selected_keys for condition in required_conditions}
    missing = sorted(expected - set(records))
    if missing:
        raise AssertionError(f"Missing selected qualitative records: {missing[:5]}")
    rng = np.random.default_rng(seed)
    form_rows: list[dict[str, object]] = []
    crosswalk_rows: list[dict[str, object]] = []
    for _, case in selected_cases.sort_values("blinded_case_id").iterrows():
        case_key = str(case["case_key"])
        condition_order = list(CONDITIONS[1:])
        rng.shuffle(condition_order)
        for candidate_code, condition in zip(("X", "Y", "Z"), condition_order):
            row = records[(case_key, condition)]
            form_rows.append({
                "blinded_case_id": case["blinded_case_id"],
                "source_dataset": row["source_dataset"],
                "reference_report": row["reference_report"],
                "reference_labels": "; ".join(
                    label for label, value in zip(LABELS_13, row["reference_vector"]) if int(value) == 1
                ) or "No positive label among the 13 evaluated findings",
                "candidate_code": candidate_code,
                "candidate_report": row["final_report"],
                "fabrication_present_0_or_1": "",
                "omission_present_0_or_1": "",
                "clinically_important_error_0_or_1": "",
                "overall_acceptable_0_or_1": "",
                "severity_none_minor_major_critical": "",
                "reviewer_id": "",
                "notes": "",
            })
            crosswalk_rows.append({
                "blinded_case_id": case["blinded_case_id"],
                "candidate_code": candidate_code,
                "condition": condition,
                "model_key": row["model_key"],
                "bundle": row["bundle"],
                "source_dataset": row["source_dataset"],
                "query_record_id": row["query_record_id"],
                "generation_record_id": row["generation_record_id"],
                "transition_tags": case["transition_tags"],
                "b_label_error_count": case["b_label_error_count"],
                "c_label_error_count": case["c_label_error_count"],
                "d_label_error_count": case["d_label_error_count"],
                "jointly_rescued_labels": case["jointly_rescued_labels"],
                "jointly_harmed_labels": case["jointly_harmed_labels"],
            })
    forms = pd.DataFrame(form_rows)
    crosswalk = pd.DataFrame(crosswalk_rows)
    reviewer1 = output / "qualitative_reviewer1_template.csv"
    reviewer2 = output / "qualitative_reviewer2_template.csv"
    crosswalk_path = output / "qualitative_case_crosswalk_ANALYST_ONLY.csv"
    forms.to_csv(reviewer1, index=False)
    forms.to_csv(reviewer2, index=False)
    crosswalk.to_csv(crosswalk_path, index=False)
    instructions = output / "QUALITATIVE_REVIEW_INSTRUCTIONS.txt"
    instructions.write_text(
        "Each reviewer must work independently and must not receive the analyst-only crosswalk.\n"
        "For every candidate report, fill the four 0/1 fields, severity, one reviewer_id, and notes.\n"
        "Use 1 for fabrication/omission/error when present; use 1 for overall_acceptable when acceptable.\n"
        "Allowed severity values are none, minor, major, or critical.\n"
        "Save completed files as qualitative_reviewer1_completed.csv and "
        "qualitative_reviewer2_completed.csv in this folder.\n",
        encoding="utf-8",
    )
    return {
        "reviewer1_template": str(reviewer1),
        "reviewer2_template": str(reviewer2),
        "analyst_only_crosswalk": str(crosswalk_path),
        "instructions": str(instructions),
    }


def hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    return {str(Path(path)): sha256_path(path) for path in paths}
