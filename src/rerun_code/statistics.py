from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import aggregate_label_metrics
from .report_labeler import LABELS_13


# Version 3 uses numeric cluster-level sufficient statistics instead of
# rebuilding pandas DataFrames for every resample.
STATISTICS_API_VERSION = 3
_OPTIONAL_METRICS = ("radgraph_f1", "cider", "bertscore_f1", "rouge_l", "runtime_seconds")


def cluster_column(frame: pd.DataFrame) -> str:
    for candidate in ("patient_key", "patient_or_source_group", "record_id", "generation_record_id"):
        if candidate in frame and frame[candidate].fillna("").astype(str).ne("").all():
            return candidate
    raise ValueError("No complete cluster identifier")


def _sum_by_cluster(values: np.ndarray, codes: np.ndarray, n_clusters: int) -> np.ndarray:
    """Sum row-level scalar or vector values within each resampling cluster."""
    result = np.zeros((n_clusters, *values.shape[1:]), dtype=float)
    np.add.at(result, codes, values)
    return result


def _cluster_sufficient_statistics(
    reference: np.ndarray,
    prediction: np.ndarray,
    codes: np.ndarray,
    n_clusters: int,
    optional: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Create additive cluster-level quantities for all reported outcomes."""
    if reference.shape != prediction.shape or reference.ndim != 2 or reference.shape[1] != len(LABELS_13):
        raise ValueError(f"Expected matching Nx{len(LABELS_13)} reference and prediction vectors")
    reference = reference.astype(int, copy=False)
    prediction = prediction.astype(int, copy=False)
    reference_abnormal = reference.sum(axis=1) > 0
    prediction_positive = prediction.sum(axis=1)
    prediction_normal = prediction_positive == 0
    fields: dict[str, np.ndarray] = {
        "tp": _sum_by_cluster((reference == 1) & (prediction == 1), codes, n_clusters),
        "fp": _sum_by_cluster((reference == 0) & (prediction == 1), codes, n_clusters),
        "fn": _sum_by_cluster((reference == 1) & (prediction == 0), codes, n_clusters),
        "n_records": np.bincount(codes, minlength=n_clusters).astype(float),
        "matching_labels": _sum_by_cluster((reference == prediction).sum(axis=1), codes, n_clusters),
        "normal_abnormal_correct": _sum_by_cluster(
            ((~reference_abnormal) == prediction_normal), codes, n_clusters
        ),
        "fp_abnormal": _sum_by_cluster(
            (((reference == 0) & (prediction == 1)) * reference_abnormal[:, None]).sum(axis=1),
            codes,
            n_clusters,
        ),
        "predicted_positive_abnormal": _sum_by_cluster(
            prediction_positive * reference_abnormal, codes, n_clusters
        ),
    }
    optional_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, values in (optional or {}).items():
        numeric = np.asarray(values, dtype=float).reshape(-1)
        if len(numeric) != len(reference):
            raise ValueError(f"Optional metric {name!r} does not align with label vectors")
        finite = np.isfinite(numeric)
        optional_stats[name] = (
            _sum_by_cluster(np.where(finite, numeric, 0.0), codes, n_clusters),
            _sum_by_cluster(finite.astype(float), codes, n_clusters),
        )
    return {"fields": fields, "optional": optional_stats}


def _sum_statistics(stats: Mapping[str, object]) -> dict[str, object]:
    return {
        "fields": {
            name: np.asarray(value, dtype=float).sum(axis=0, keepdims=True)
            for name, value in stats["fields"].items()
        },
        "optional": {
            name: (
                np.asarray(pair[0], dtype=float).sum(axis=0, keepdims=True),
                np.asarray(pair[1], dtype=float).sum(axis=0, keepdims=True),
            )
            for name, pair in stats["optional"].items()
        },
    }


def _weighted_statistics(stats: Mapping[str, object], weights: np.ndarray) -> dict[str, object]:
    return {
        "fields": {name: weights @ np.asarray(value, dtype=float) for name, value in stats["fields"].items()},
        "optional": {
            name: (weights @ np.asarray(pair[0], dtype=float), weights @ np.asarray(pair[1], dtype=float))
            for name, pair in stats["optional"].items()
        },
    }


def _mix_statistics(
    base: Mapping[str, object], other: Mapping[str, object], swaps: np.ndarray
) -> dict[str, object]:
    """Swap complete clusters from ``other`` into ``base`` for each draw."""
    return {
        "fields": {
            name: np.asarray(value, dtype=float).sum(axis=0, keepdims=True)
            + swaps @ (np.asarray(other["fields"][name], dtype=float) - np.asarray(value, dtype=float))
            for name, value in base["fields"].items()
        },
        "optional": {
            name: tuple(
                np.asarray(value[index], dtype=float).sum(axis=0, keepdims=True)
                + swaps @ (np.asarray(other["optional"][name][index], dtype=float) - np.asarray(value[index], dtype=float))
                for index in (0, 1)
            )
            for name, value in base["optional"].items()
        },
    }


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    result = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _metrics_from_statistics(stats: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Calculate existing estimands from additive cluster-level totals."""
    values = stats["fields"]
    tp, fp, fn = (np.asarray(values[name], dtype=float) for name in ("tp", "fp", "fn"))
    predicted_positive = tp.sum(axis=1) + fp.sum(axis=1)
    reference_positive = tp.sum(axis=1) + fn.sum(axis=1)
    f1 = _divide(2 * tp, 2 * tp + fp + fn)
    with np.errstate(invalid="ignore"):
        macro_f1 = np.nanmean(f1, axis=1)
    micro_tp, micro_fp, micro_fn = tp.sum(axis=1), fp.sum(axis=1), fn.sum(axis=1)
    n_records = np.asarray(values["n_records"], dtype=float)
    result: dict[str, np.ndarray] = {
        "fer": _divide(fp.sum(axis=1), predicted_positive),
        "fer_abnormal": _divide(values["fp_abnormal"], values["predicted_positive_abnormal"]),
        "omission": _divide(fn.sum(axis=1), reference_positive),
        "macro_f1": macro_f1,
        "micro_f1": _divide(2 * micro_tp, 2 * micro_tp + micro_fp + micro_fn),
        "hamming_accuracy": _divide(values["matching_labels"], n_records * len(LABELS_13)),
        "normal_abnormal_accuracy": _divide(values["normal_abnormal_correct"], n_records),
        "false_positive_events": fp.sum(axis=1),
        "predicted_positive_events": predicted_positive,
        "false_negative_events": fn.sum(axis=1),
        "reference_positive_events": reference_positive,
    }
    for index, label in enumerate(LABELS_13):
        result[f"f1_{label.replace(' ', '_')}"] = f1[:, index]
    for name, (numerator, denominator) in stats["optional"].items():
        result[name] = _divide(numerator, denominator)
    return result


def _multinomial_cluster_weights(
    rng: np.random.Generator, n_draws: int, n_clusters: int
) -> np.ndarray:
    """Draw bootstrap cluster multiplicities without materializing data frames."""
    selected = rng.integers(0, n_clusters, size=(n_draws, n_clusters))
    weights = np.zeros((n_draws, n_clusters), dtype=float)
    np.add.at(
        weights,
        (np.repeat(np.arange(n_draws), n_clusters), selected.reshape(-1)),
        1.0,
    )
    return weights


def _optional_metric_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        for name in _OPTIONAL_METRICS
        if name in frame.columns and frame[name].notna().any()
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    replicates: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 20260831,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Patient/source-cluster bootstrap using vectorized sufficient statistics.

    Every draw samples clusters with replacement. Label outcomes are recomputed
    from resampled event counts and direct text metrics from resampled valid
    scores. Batching controls memory while retaining the configured draws.
    """
    if replicates <= 0 or batch_size <= 0:
        raise ValueError("replicates and batch_size must be positive")
    column = cluster_column(frame)
    point = aggregate_label_metrics(frame)
    metric_names = [
        name for name, value in point.items()
        if isinstance(value, (int, float)) and not name.startswith("n_")
    ]
    fallback = frame["generation_record_id"] if "generation_record_id" in frame else frame.index.astype(str)
    codes, cluster_ids = pd.factorize(frame[column].fillna(fallback).astype(str), sort=False)
    n_clusters = len(cluster_ids)
    if n_clusters == 0:
        raise ValueError("No clusters")
    reference = np.asarray(frame["reference_vector"].tolist(), dtype=int)
    prediction = np.asarray(frame["prediction_vector"].tolist(), dtype=int)
    stats = _cluster_sufficient_statistics(reference, prediction, codes, n_clusters, _optional_metric_arrays(frame))
    draws = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        weights = _multinomial_cluster_weights(rng, min(batch_size, replicates - start), n_clusters)
        estimates = _metrics_from_statistics(_weighted_statistics(stats, weights))
        for name in metric_names:
            draws[name].append(estimates[name])
    alpha = (1 - confidence_level) / 2
    rows = []
    for name in metric_names:
        values = np.concatenate(draws[name]).astype(float, copy=False)
        valid = values[np.isfinite(values)]
        rows.append({
            "metric": name,
            "estimate": point[name],
            "ci_low": float(np.quantile(valid, alpha)) if len(valid) else np.nan,
            "ci_high": float(np.quantile(valid, 1 - alpha)) if len(valid) else np.nan,
            "n_valid_replicates": int(len(valid)),
            "n_clusters": int(n_clusters),
        })
    return pd.DataFrame(rows)


def paired_cluster_permutation(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    id_column: str = "query_record_id",
    metrics: Sequence[str] = ("fer_abnormal", "omission", "macro_f1", "radgraph_f1"),
    replicates: int = 10000,
    seed: int = 20260831,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Two-sided paired patient/source-cluster permutation test.

    Each randomization swaps complete clusters between matched arms. Numeric
    mixing is algebraically equivalent to row swapping followed by metric
    recomputation, but avoids repeated DataFrame copies.
    """
    if replicates <= 0 or batch_size <= 0:
        raise ValueError("replicates and batch_size must be positive")
    shared = [id_column, "reference_vector", "prediction_vector"]
    extras = [name for name in (*_OPTIONAL_METRICS, "patient_key") if name in frame_a and name in frame_b]
    a = frame_a[shared + extras].copy()
    b = frame_b[shared + extras].copy()
    merged = a.merge(b, on=id_column, suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise AssertionError("Paired tests require identical record sets")
    reference_a = np.asarray(merged["reference_vector_a"].tolist(), dtype=int)
    reference_b = np.asarray(merged["reference_vector_b"].tolist(), dtype=int)
    if not np.array_equal(reference_a, reference_b):
        raise AssertionError("Paired tests require identical reference vectors for every query")
    if "patient_key_a" in merged and "patient_key_b" in merged:
        patient_a = merged["patient_key_a"].fillna(merged[id_column]).astype(str)
        patient_b = merged["patient_key_b"].fillna(merged[id_column]).astype(str)
        if not patient_a.equals(patient_b):
            raise AssertionError("Paired test arms disagree on patient/source cluster assignments")
    clusters = merged["patient_key_a"].fillna(merged[id_column]).astype(str) if "patient_key_a" in merged else merged[id_column].astype(str)
    codes, cluster_ids = pd.factorize(clusters, sort=False)
    n_clusters = len(cluster_ids)
    prediction_a = np.asarray(merged["prediction_vector_a"].tolist(), dtype=int)
    prediction_b = np.asarray(merged["prediction_vector_b"].tolist(), dtype=int)
    optional_a = {
        name: pd.to_numeric(merged[f"{name}_a"], errors="coerce").to_numpy(dtype=float)
        for name in _OPTIONAL_METRICS if f"{name}_a" in merged and f"{name}_b" in merged
    }
    optional_b = {
        name: pd.to_numeric(merged[f"{name}_b"], errors="coerce").to_numpy(dtype=float)
        for name in optional_a
    }
    stats_a = _cluster_sufficient_statistics(reference_a, prediction_a, codes, n_clusters, optional_a)
    stats_b = _cluster_sufficient_statistics(reference_a, prediction_b, codes, n_clusters, optional_b)
    point_a = _metrics_from_statistics(_sum_statistics(stats_a))
    point_b = _metrics_from_statistics(_sum_statistics(stats_b))
    observed = {
        name: float(point_b.get(name, np.array([np.nan]))[0] - point_a.get(name, np.array([np.nan]))[0])
        for name in metrics
    }
    exceed = {name: 0 for name in metrics}
    valid = {name: 0 for name in metrics}
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        n_draws = min(batch_size, replicates - start)
        swaps = rng.integers(0, 2, size=(n_draws, n_clusters)).astype(float)
        permuted_a = _metrics_from_statistics(_mix_statistics(stats_a, stats_b, swaps))
        permuted_b = _metrics_from_statistics(_mix_statistics(stats_b, stats_a, swaps))
        for name in metrics:
            delta = permuted_b.get(name, np.full(n_draws, np.nan)) - permuted_a.get(name, np.full(n_draws, np.nan))
            finite = np.isfinite(delta) & np.isfinite(observed[name])
            valid[name] += int(finite.sum())
            exceed[name] += int((np.abs(delta[finite]) >= abs(observed[name]) - 1e-15).sum())
    rows = []
    for name in metrics:
        p_value = (exceed[name] + 1) / (valid[name] + 1) if valid[name] else np.nan
        rows.append({
            "metric": name,
            "arm_a": float(point_a.get(name, np.array([np.nan]))[0]),
            "arm_b": float(point_b.get(name, np.array([np.nan]))[0]),
            "difference_b_minus_a": observed[name],
            "p_value": p_value,
            "valid_permutations": valid[name],
        })
    return pd.DataFrame(rows)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.where(np.isfinite(values))[0]
    order = finite[np.argsort(values[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()
