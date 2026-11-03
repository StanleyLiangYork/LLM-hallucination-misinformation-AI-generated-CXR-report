"""Paired direct-image effects with the original, vectorized cluster estimators."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .modality_ablation import ARMS, complete_frame, read_json, digest, atomic_json
from .metrics import aggregate_label_metrics
from .report_labeler import LABELS_13
from .statistics import (
    _cluster_sufficient_statistics, _weighted_statistics, _metrics_from_statistics,
    _multinomial_cluster_weights, cluster_bootstrap, paired_cluster_permutation,
    holm_adjust,
)


def align_pairs(a, b):
    """Never silently intersect records or pair different ground truths/clusters."""
    key = "query_record_id"
    if a[key].duplicated().any() or b[key].duplicated().any() or set(a[key]) != set(b[key]):
        raise AssertionError("Paired arms require identical unique query IDs")
    a, b = (x.sort_values(key).reset_index(drop=True) for x in (a, b))
    for field in ("patient_key", "reference_vector", "evidence_sha256"):
        if a[field].tolist() != b[field].tolist():
            raise AssertionError(f"Paired arms disagree on {field}")
    return a, b


def paired_difference_ci(a, b, metrics, *, replicates=10000, seed=20261029, batch_size=256):
    """Same resampled clusters in both arms; recompute nonlinear ratios from counts."""
    a, b = align_pairs(a, b)
    codes, groups = pd.factorize(a["patient_key"], sort=False)
    stats = []
    for frame in (a, b):
        stats.append(_cluster_sufficient_statistics(
            np.asarray(frame["reference_vector"].tolist()), np.asarray(frame["prediction_vector"].tolist()),
            codes, len(groups), {"radgraph_f1": frame["radgraph_f1"].to_numpy(float)}))
    draws = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, batch_size):
        weights = _multinomial_cluster_weights(rng, min(batch_size, replicates-start), len(groups))
        first, second = [_metrics_from_statistics(_weighted_statistics(s, weights)) for s in stats]
        for metric in metrics:
            draws[metric].append(second[metric] - first[metric])
    result = []
    for metric, chunks in draws.items():
        values = np.concatenate(chunks)
        values = values[np.isfinite(values)]
        result.append(dict(metric=metric, difference_ci_low=float(np.quantile(values, .025)) if len(values) else np.nan,
                           difference_ci_high=float(np.quantile(values, .975)) if len(values) else np.nan,
                           valid_difference_bootstrap_replicates=len(values), n_clusters=len(groups)))
    return pd.DataFrame(result)


def transitions(a, b):
    a, b = align_pairs(a, b)
    truth = np.asarray(a["reference_vector"].tolist())
    first, second = (np.asarray(x["prediction_vector"].tolist()) for x in (a, b))
    ea, eb = (x != truth for x in (first, second))
    return pd.DataFrame(dict(query_record_id=a["query_record_id"], patient_key=a["patient_key"],
                             errors_text=ea.sum(1), errors_image=eb.sum(1),
                             errors_resolved=(ea & ~eb).sum(1), errors_introduced=(~ea & eb).sum(1),
                             rescued=eb.sum(1) < ea.sum(1), harmed=eb.sum(1) > ea.sum(1),
                             unchanged_error_count=eb.sum(1) == ea.sum(1),
                             prediction_vector_changed=(first != second).any(1)))


def analyze(root, config, run_name):
    out, plan, original = complete_frame(root, config, run_name)
    saved = read_json(out / "scored_records.json")
    protocol = read_json(out / "scoring_protocol.json")
    if saved["protocol_sha256"] != digest(protocol):
        raise AssertionError("Scoring provenance differs")
    frame = pd.DataFrame(saved["rows"])
    if (frame["generation_record_id"].duplicated().any()
            or set(frame["generation_record_id"]) != set(original["generation_record_id"])):
        raise AssertionError("Scored cohort differs from complete generation cohort")
    expected_payload = original.set_index("generation_record_id")["payload_sha256"].to_dict()
    for row in saved["rows"]:
        unscored = {k: v for k, v in row.items() if k not in {"prediction_vector", "radgraph_f1", "payload_sha256"}}
        if digest(unscored) != expected_payload[row["generation_record_id"]]:
            raise AssertionError("Scored input report or reference changed")
    for field in ("prediction_vector", "reference_vector"):
        values = np.asarray(frame[field].tolist())
        if values.shape != (len(frame), 13) or not np.isin(values, [0, 1]).all():
            raise ValueError(f"Invalid binary vectors: {field}")
    if not np.isfinite(frame["radgraph_f1"].to_numpy(float)).all():
        raise ValueError("All four primary outcomes require complete RadGraph scores")
    destination = out / ("smoke_statistics" if plan["smoke_n"] is not None else "statistics")
    destination.mkdir(exist_ok=True)
    group_fields = ["model_key", "bundle", "source_dataset", "condition"]
    point_rows, comparisons, transition_frames, counts = [], [], [], []
    # Technical runs deliberately use fewer draws and can never export publication text.
    reps = 100 if plan["smoke_n"] is not None else plan["bootstrap_replicates"]
    perm = 100 if plan["smoke_n"] is not None else plan["permutation_replicates"]
    metrics = plan["primary_metrics"]
    for keys, group in frame.groupby(group_fields, sort=True):
        context = dict(zip(group_fields, keys))
        seed = int(digest([plan["seed"], list(keys)])[:8], 16)
        a, b = align_pairs(*(group[group.arm == arm] for arm in ARMS))
        for arm, arm_frame in zip(ARMS, (a, b)):
            estimates = aggregate_label_metrics(arm_frame)
            defined = sum(np.isfinite(estimates[f"f1_{label.replace(' ', '_')}"]) for label in LABELS_13)
            ci = cluster_bootstrap(arm_frame, replicates=reps, seed=seed)
            for row in ci[ci.metric.isin(metrics)].to_dict("records"):
                point_rows.append({**context, "arm": arm, "n_records": len(arm_frame),
                                   "n_defined_f1_labels": defined,
                                   "strict_all13_macro_f1": estimates["macro_f1"] if defined == 13 else np.nan,
                                   **row})
        test = paired_cluster_permutation(a, b, metrics=metrics, replicates=perm, seed=seed)
        ci = paired_difference_ci(a, b, metrics, replicates=reps, seed=seed)
        merged = test.merge(ci, on="metric", validate="one_to_one")
        comparisons.extend({**context, "n_records": len(a), **row} for row in merged.to_dict("records"))
        change = transitions(a, b).assign(**context)
        transition_frames.append(change)
        counts.append({**context, "n_pairs": len(change),
                       **{name: int(change[name].sum()) for name in
                          ("rescued", "harmed", "unchanged_error_count", "prediction_vector_changed",
                           "errors_resolved", "errors_introduced")}})
        print("Analyzed", "/".join(keys), flush=True)
    contrasts = pd.DataFrame(comparisons)
    contrasts["family"] = np.where(contrasts.condition.eq("A_single_pass"), "primary_single_pass", "secondary_gated_or_revision")
    # One family per metric: all selected backbones x source/gallery strata.
    # Secondary families additionally include ALL selected B/C/D strategies.
    for _, indices in contrasts.groupby(["family", "metric"]).groups.items():
        p = contrasts.loc[indices, "p_value"]
        adjusted = np.asarray(holm_adjust(p.fillna(1).tolist()))
        adjusted[p.isna().to_numpy()] = np.nan
        contrasts.loc[indices, "p_holm"] = adjusted
        contrasts.loc[indices, "holm_family_size"] = len(indices)
    contrasts["significant_holm_005"] = contrasts.p_holm < .05
    contrasts["positive_difference_means"] = np.where(
        contrasts.metric.isin(["fer_abnormal", "omission"]), "worse", "better")
    contrasts["difference_percentage_points"] = 100 * contrasts.difference_b_minus_a
    pd.DataFrame(point_rows).to_csv(destination / "Table_S20a_arm_estimates.csv", index=False)
    contrasts.to_csv(destination / "Table_S20b_paired_image_effects.csv", index=False)
    pd.DataFrame(counts).to_csv(destination / "Table_S20c_rescue_harm.csv", index=False)
    pd.concat(transition_frames).to_csv(destination / "case_level_image_effects.csv", index=False)
    summary = dict(experiment_id=plan["experiment_id"], n_final_reports=len(frame),
                   n_unique_queries=frame.query_record_id.nunique(), smoke_only=plan["smoke_n"] is not None,
                   bootstrap_replicates=reps, permutation_replicates=perm,
                   interpretation="image minus no-image within backbone; image-conditioned retrieval fixed",
                   macro_f1_policy="Matches existing metrics.py: mean over defined findings; count reported. "
                                   "Strict all-13 point estimate also supplied; do not hide undefined findings.",
                   timing_note="Runtime is descriptive, not a synchronized latency benchmark.")
    atomic_json(destination / "analysis_status.json", summary)
    if plan["smoke_n"] is None:
        primary = contrasts[contrasts.family.eq("primary_single_pass")]
        lines = ["SUPPLEMENTARY SECTION S6.17 AND TABLE S20 DRAFT", "",
                 "Confirm section/table numbering against the final assembled supplement.", "",
                 "Methods", "We compared each multimodal backbone with and without direct query-image input, "
                 "holding the retrieved reports, their order and scores, backbone, decoding settings and "
                 "modality-neutral textual instructions fixed. Both arms were generated anew. "
                 "Retrieval remained image-conditioned in both arms. Single-pass RAG was the primary "
                 "contrast; iterative and gated strategies were secondary. In the no-image arm, "
                 "pixels were withheld from both report generation and verification. Phi verification "
                 "remained text-only in both arms; MedGemma gated contrasts therefore include changes "
                 "to both generator and verifier inputs. No adapter was retrained.", "",
                 "We computed the four primary outcomes using the original dataset references and "
                 "frozen CheXbert mapping. Paired patient/source-group bootstraps and cluster-swap "
                 "tests used 10,000 draws. Difference intervals are unadjusted 95% intervals. "
                 "Holm correction was applied across model-by-source/gallery contrasts separately "
                 "for each outcome, with primary and secondary strategies in distinct families. "
                 "IUHN groups are source identifiers, not verified patients. Macro F1 matches the "
                 "primary pipeline's mean over defined finding F1 values; the number defined and "
                 "the strict all-13 point estimate are exported.", "",
                 f"Results: {len(frame):,} new final reports from {frame.query_record_id.nunique():,} unique query records.",
                 "Table S20a gives arm estimates; S20b paired effects; S20c study-level rescue and harm.", ""]
        for metric, group in primary.groupby("metric"):
            lines.append(f"Single-pass {metric}: {int(group.significant_holm_005.sum())} of {len(group)} "
                         f"contrasts met Holm-adjusted P<0.05. Image-minus-text-only estimates ranged "
                         f"from {group.difference_b_minus_a.min():+.4f} to {group.difference_b_minus_a.max():+.4f}.")
        lines += ["", "Interpretation limits", "These post hoc ablations estimate direct-image access within "
                  "the tested backbones and prompts, not a general effect of model modality, scale or "
                  "retrieval. The no-image condition may be outside the model's preferred use distribution. "
                  "Prompt changes mean this experiment is not a replacement for the original tables. "
                  "Reference-label and automated-extraction limitations remain; the earlier 200-reference "
                  "clinician assessment does not validate these new generated reports.", "",
                  "Main manuscript insertion plan", "Methods, after Models and generation strategies: "
                  "We additionally compared the two multimodal backbones with and without direct "
                  "image input while holding retrieved evidence and textual instructions fixed "
                  "(Supplementary Table S20). Single-pass RAG was primary and gated strategies were secondary.",
                  "Results: summarize the direction and uncertainty of the single-pass effects above "
                  "in one or two sentences; cite Table S20. Do not select only favorable outcomes.",
                  "Discussion: interpret the conditional within-backbone result; retain the caveat "
                  "that cross-model comparisons cannot isolate architecture or scale. Replace only "
                  "the limitation saying no same-backbone image ablation was performed. Keep the "
                  "absence of a no-retrieval arm and lack of prospective clinical validation."]
        recovery = read_json(out / "empty_output_recovery_summary.json")
        n_retried = sum(group["records_retried"] for group in recovery["by_stratum"].values())
        lines += ["", "Empty-output recovery and protocol amendment",
                  f"{n_retried} final-report records required at least one empty-output retry. "
                  "The identical prompt, retrieved evidence and image routing were used, with "
                  "min_new_tokens=32 on one retry only. The rule applied to both models and input arms. "
                  "It does not guarantee useful clinical content. Unresolved empty outputs were not scored.",
                  ("This rule was introduced after an observed Phi failure. Existing successful records "
                   "were retained byte-for-byte because the first-attempt branch was unchanged and those "
                   "records had no empty passes. Disclose this as a post hoc protocol amendment."
                   if recovery["post_hoc_amendment"] else "The retry rule was part of this run's frozen protocol."),
                  "Report model/strategy/arm-specific retry counts from empty_output_recovery_summary.json; "
                  "do not describe all outputs as obtained under the original unamended decoding rule."]
        (destination / "supplement_S20_and_manuscript_insertion.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return destination, contrasts
