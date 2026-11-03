"""Generate post-rerun Notebooks 10-13 from readable source cells."""

from __future__ import annotations

import textwrap
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "notebooks"
ROOT.mkdir(parents=True, exist_ok=True)


def md(value: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(value).strip().splitlines(keepends=True),
    }


def code(value: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(value).strip().splitlines(keepends=True),
    }


def save(name: str, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / name).write_text(json.dumps(notebook, indent=1), encoding="utf-8")


BOOTSTRAP = """
from pathlib import Path
import json, os, sys

def find_rerun_dir():
    candidates = [
        Path(os.environ.get("JAMIA_RERUN_DIR", "")),
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if str(candidate) and (candidate / "rerun_config.json").exists():
            return candidate.resolve()
    raise FileNotFoundError("Set JAMIA_RERUN_DIR to the folder containing rerun_config.json")

RERUN_DIR = find_rerun_dir()
implementation_dir = (RERUN_DIR / "src" / "rerun_code").resolve()
sys.path[:] = [
    entry for entry in sys.path
    if Path(entry or ".").resolve() != implementation_dir
]
sys.path.insert(0, str(RERUN_DIR / "src"))
from rerun_code.config import load_config, output_paths
RERUN_DIR, CONFIG = load_config(RERUN_DIR)
PATHS = output_paths(CONFIG)
POST_ROOT = PATHS["root"] / "post_rerun"
POST_ROOT.mkdir(parents=True, exist_ok=True)
print("Code:", RERUN_DIR)
print("Output:", POST_ROOT)
"""


def notebook_10():
    save("10_sampling_provenance_audit.ipynb", [
        md("""
        # 10 — Historical sampling-provenance audit

        This notebook addresses Reviewer 1, Major Comment 1. It verifies the
        observed composition of the frozen 400-study IUHN and 634-study MIMIC
        cohorts, checks whether every paired file in each frozen test directory
        entered the rerun, and inventories contemporaneous files that might
        document the upstream selection. Balance is not treated as proof of
        stratified sampling. If no historical selection record is found, the
        output explicitly records that the original sampling rationale remains
        unrecoverable from the supplied artifacts.
        """),
        code(BOOTSTRAP),
        code("""
        import hashlib, os, re
        from datetime import datetime, timezone
        import pandas as pd
        from rerun_code.common import read_jsonl, write_json
        from rerun_code.report_labeler import LABELS_13
        from rerun_code.config import sha256_path

        OUTPUT = POST_ROOT / "sampling_provenance"
        OUTPUT.mkdir(parents=True, exist_ok=True)
        composition_rows = []
        directory_rows = []
        frozen_manifests = {}

        for dataset in ("mimic", "iuhn"):
            manifest_path = PATHS["manifests"] / dataset / "fixed_queries.jsonl"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Run Notebook 01 first: {manifest_path}")
            manifest = pd.DataFrame(read_jsonl(manifest_path))
            frozen_manifests[dataset] = manifest
            normal = manifest["labels_13"].map(len).eq(0)
            row = {
                "dataset": dataset,
                "n_studies": int(len(manifest)),
                "n_patient_or_uid_groups": int(manifest["patient_key"].nunique()),
                "patient_id_reliable": bool(manifest["patient_id_reliable"].all()),
                "n_normal": int(normal.sum()),
                "n_abnormal": int((~normal).sum()),
            }
            for label in LABELS_13:
                row[f"n_{label.replace(' ', '_')}"] = int(
                    manifest["labels_13"].map(lambda labels: label in labels).sum()
                )
            composition_rows.append(row)

            test_root = Path(CONFIG["datasets"][dataset]["test"])
            if not test_root.exists():
                raise FileNotFoundError(test_root)
            image_suffixes = {".jpg", ".jpeg", ".png"}
            images = [p for p in test_root.rglob("*") if p.is_file() and p.suffix.lower() in image_suffixes]
            json_files = [p for p in test_root.rglob("*.json") if p.is_file()]
            image_keys = {str(p.relative_to(test_root).with_suffix("")) for p in images}
            json_keys = {str(p.relative_to(test_root).with_suffix("")) for p in json_files}
            directory_pairs = image_keys & json_keys
            manifest_image_names = set(manifest["image_path"].map(lambda value: Path(str(value)).name))
            directory_image_names = {p.name for p in images if str(p.relative_to(test_root).with_suffix("")) in directory_pairs}
            directory_rows.append({
                "dataset": dataset,
                "test_directory": str(test_root),
                "n_image_files": len(images),
                "n_json_files": len(json_files),
                "n_complete_image_json_pairs": len(directory_pairs),
                "n_frozen_manifest_records": len(manifest),
                "all_directory_pair_image_names_in_manifest": directory_image_names <= manifest_image_names,
                "all_manifest_image_names_in_directory_pairs": manifest_image_names <= directory_image_names,
                "manifest_sha256": sha256_path(manifest_path),
            })

        composition = pd.DataFrame(composition_rows)
        directory_audit = pd.DataFrame(directory_rows)
        composition.to_csv(OUTPUT / "frozen_cohort_composition.csv", index=False)
        directory_audit.to_csv(OUTPUT / "frozen_test_directory_audit.csv", index=False)
        display(composition)
        display(directory_audit)
        """),
        code("""
        # Inventory likely provenance artifacts. This is an evidence inventory,
        # not an automatic assertion about the historical selection method.
        keywords = re.compile(r"sample|split|cohort|manifest|random|stratif|readme|build_config", re.I)
        roots = [RERUN_DIR]
        for dataset in ("mimic", "iuhn"):
            roots.extend([
                Path(CONFIG["datasets"][dataset]["test"]),
                Path(CONFIG["datasets"][dataset]["preencoded_test"]),
            ])
        supplied = [Path(value) for value in os.environ.get("JAMIA_SAMPLING_EVIDENCE", "").split(",") if value.strip()]
        roots.extend(path for path in supplied if path.is_dir())
        candidates = {path.resolve() for path in supplied if path.is_file()}
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            root_depth = len(root.parts)
            for current, directories, files in os.walk(root):
                current_path = Path(current)
                if len(current_path.parts) - root_depth >= 4:
                    directories[:] = []
                directories[:] = [name for name in directories if not name.startswith(".")]
                for name in files:
                    path = current_path / name
                    if keywords.search(name):
                        candidates.add(path.resolve())

        evidence_rows = []
        for path in sorted(candidates):
            size = path.stat().st_size
            evidence_rows.append({
                "path": str(path),
                "size_bytes": size,
                "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_path(path) if size <= 50 * 1024 * 1024 else "not_hashed_over_50MB",
                "explicitly_supplied": path in {item.resolve() for item in supplied if item.exists()},
                "automatic_sampling_claim_supported": False,
                "review_note": "Inspect manually; filename presence does not establish the sampling rule.",
            })
        evidence = pd.DataFrame(evidence_rows)
        evidence.to_csv(OUTPUT / "sampling_evidence_inventory.csv", index=False)

        template_path = OUTPUT / "historical_sampling_provenance_template.csv"
        pd.DataFrame([
            {
                "dataset": dataset,
                "source_population": "",
                "eligibility_criteria": "",
                "selection_method": "",
                "stratification_variables": "",
                "random_seed": "",
                "selection_date_or_version": "",
                "responsible_investigator": "",
                "evidence_file": "",
                "manuscript_wording": "",
            }
            for dataset in ("mimic", "iuhn")
        ]).to_csv(template_path, index=False)

        completed_path = Path(os.environ.get(
            "JAMIA_COMPLETED_SAMPLING_PROVENANCE",
            OUTPUT / "historical_sampling_provenance_completed.csv",
        ))
        historical_resolved = False
        validation_errors = []
        if completed_path.exists():
            completed = pd.read_csv(completed_path, dtype=str).fillna("")
            required = [
                "dataset", "source_population", "eligibility_criteria", "selection_method",
                "evidence_file", "manuscript_wording",
            ]
            missing = [name for name in required if name not in completed]
            if missing:
                validation_errors.append(f"missing columns: {missing}")
            if not missing:
                if set(completed["dataset"]) != {"mimic", "iuhn"}:
                    validation_errors.append("completed file must contain one mimic row and one iuhn row")
                for name in required[1:]:
                    if completed[name].str.strip().eq("").any():
                        validation_errors.append(f"blank required values in {name}")
                for value in completed["evidence_file"]:
                    if value.strip() and not Path(value).exists():
                        validation_errors.append(f"evidence file not found: {value}")
            historical_resolved = not validation_errors

        status = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "observed_cohort_composition_verified": True,
            "all_frozen_test_directory_pairs_analyzed": bool(
                directory_audit["all_directory_pair_image_names_in_manifest"].all()
                and directory_audit["all_manifest_image_names_in_directory_pairs"].all()
            ),
            "historical_sampling_provenance_resolved": historical_resolved,
            "historical_sampling_provenance_status": (
                "documented from completed evidence file"
                if historical_resolved else
                "not established by current computational artifacts; complete the provenance template from contemporaneous records"
            ),
            "validation_errors": validation_errors,
            "completed_provenance_file": str(completed_path),
            "outputs": {
                "cohort_composition": str(OUTPUT / "frozen_cohort_composition.csv"),
                "test_directory_audit": str(OUTPUT / "frozen_test_directory_audit.csv"),
                "evidence_inventory": str(OUTPUT / "sampling_evidence_inventory.csv"),
                "provenance_template": str(template_path),
            },
            "interpretation_guardrail": "Observed class balance is not proof of stratified sampling.",
        }
        write_json(OUTPUT / "notebook10_status.json", status)
        print(json.dumps(status, indent=2))
        """),
    ])


def notebook_11():
    save("11_b_vs_c_paired_inference.ipynb", [
        md("""
        # 11 — B versus C paired inference

        This notebook completes Reviewer 1, Minor Comment 4 by directly
        comparing unconditional four-pass revision (B) with pretrained-verifier
        gating (C). It uses the same frozen study cohort, patient/source
        clusters, 10,000-replicate protocol, and primary metrics as Notebook 08.
        It also recomputes Holm adjustment after expanding each within-stratum
        family from four to five strategy contrasts.
        """),
        code(BOOTSTRAP),
        code("""
        import importlib
        from datetime import datetime, timezone
        import numpy as np, pandas as pd
        from rerun_code.common import write_json
        from rerun_code.config import sha256_path
        import rerun_code.post_rerun as post_module
        post_module = importlib.reload(post_module)
        if int(getattr(post_module, "POST_RERUN_API_VERSION", 0)) < 1:
            raise ImportError("Copy the updated rerun_code/post_rerun.py and restart the kernel.")
        from rerun_code.post_rerun import (
            PRIMARY_METRICS, load_statistical_frame, paired_cluster_bootstrap_difference,
            verified_per_study_path,
        )
        from rerun_code.statistics import paired_cluster_permutation, holm_adjust

        OUTPUT = POST_ROOT / "b_vs_c"
        OUTPUT.mkdir(parents=True, exist_ok=True)
        per_study_path, upstream_audit = verified_per_study_path(PATHS)
        frame = load_statistical_frame(per_study_path)
        reps = int(CONFIG["statistics"]["permutation_replicates"])
        boot_reps = int(CONFIG["statistics"]["bootstrap_replicates"])
        confidence = float(CONFIG["statistics"]["confidence_level"])
        seed = int(CONFIG["statistics"]["seed"])
        print("Verified rows:", len(frame), "groups:", frame.groupby(["model_key", "bundle", "source_dataset"]).ngroups)
        """),
        code("""
        parts = []
        grouped = list(frame.groupby(["model_key", "bundle", "source_dataset"], dropna=False))
        for index, (keys, group) in enumerate(grouped, start=1):
            arm_b = group[group["condition"] == "B_unconditional_4pass"]
            arm_c = group[group["condition"] == "C_pretrained_gate"]
            tests = paired_cluster_permutation(
                arm_b, arm_c, id_column="query_record_id", metrics=PRIMARY_METRICS,
                replicates=reps, seed=seed,
            )
            intervals = paired_cluster_bootstrap_difference(
                arm_b, arm_c, metrics=PRIMARY_METRICS, replicates=boot_reps,
                confidence_level=confidence, seed=seed,
            )
            result = tests.merge(intervals, on="metric", validate="one_to_one")
            if not np.allclose(
                result["difference_b_minus_a"], result["difference_c_minus_b"], equal_nan=True
            ):
                raise AssertionError("B-C point estimates disagree between inference engines")
            result["arm_a_condition"] = "B_unconditional_4pass"
            result["arm_b_condition"] = "C_pretrained_gate"
            result["model_key"], result["bundle"], result["source_dataset"] = keys
            parts.append(result)
            print(f"Completed stratum {index}/{len(grouped)}: {keys}")

        bc = pd.concat(parts, ignore_index=True)
        existing_path = PATHS["statistics"] / "paired_cluster_permutation_tests.csv"
        existing = pd.read_csv(existing_path)
        all_five = pd.concat([
            existing.drop(columns=["p_holm_within_strategy_family"], errors="ignore"),
            bc.drop(columns=[
                "difference_c_minus_b", "difference_ci_low", "difference_ci_high",
                "valid_bootstrap_replicates", "n_clusters",
            ]),
        ], ignore_index=True)
        family = ["model_key", "metric", "bundle", "source_dataset"]
        all_five["p_holm_five_contrast_family"] = all_five.groupby(
            family, dropna=False
        )["p_value"].transform(lambda values: holm_adjust(values.tolist()))
        bc = bc.merge(
            all_five[[
                "model_key", "bundle", "source_dataset", "metric", "arm_a_condition",
                "arm_b_condition", "p_holm_five_contrast_family",
            ]],
            on=["model_key", "bundle", "source_dataset", "metric", "arm_a_condition", "arm_b_condition"],
            validate="one_to_one",
        )
        lower_is_better = {"fer_abnormal", "omission"}
        bc["holm_significant_0_05"] = bc["p_holm_five_contrast_family"] < 0.05
        bc["direction_c_vs_b"] = np.where(
            bc["metric"].isin(lower_is_better),
            np.where(bc["difference_c_minus_b"] < 0, "C better", np.where(bc["difference_c_minus_b"] > 0, "C worse", "tie")),
            np.where(bc["difference_c_minus_b"] > 0, "C better", np.where(bc["difference_c_minus_b"] < 0, "C worse", "tie")),
        )
        bc_path = OUTPUT / "b_vs_c_paired_cluster_results.csv"
        all_path = OUTPUT / "within_model_tests_five_contrasts_recomputed_holm.csv"
        bc.to_csv(bc_path, index=False)
        all_five.to_csv(all_path, index=False)
        """),
        code("""
        summary_rows = []
        for metric, subset in bc.groupby("metric", sort=False):
            significant = subset[subset["holm_significant_0_05"]]
            summary_rows.append({
                "metric": metric,
                "n_strata": len(subset),
                "n_holm_significant": len(significant),
                "n_significant_c_better": int((significant["direction_c_vs_b"] == "C better").sum()),
                "n_significant_c_worse": int((significant["direction_c_vs_b"] == "C worse").sum()),
                "median_difference_c_minus_b": subset["difference_c_minus_b"].median(),
                "minimum_difference_c_minus_b": subset["difference_c_minus_b"].min(),
                "maximum_difference_c_minus_b": subset["difference_c_minus_b"].max(),
            })
        summary = pd.DataFrame(summary_rows)
        summary_path = OUTPUT / "b_vs_c_summary.csv"
        summary.to_csv(summary_path, index=False)
        status = {
            "ready": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            **upstream_audit,
            "n_rows": len(frame),
            "n_strata": int(frame.groupby(["model_key", "bundle", "source_dataset"]).ngroups),
            "n_b_vs_c_tests": len(bc),
            "n_holm_significant": int(bc["holm_significant_0_05"].sum()),
            "bootstrap_replicates": boot_reps,
            "permutation_replicates": reps,
            "confidence_level": confidence,
            "seed": seed,
            "multiplicity_family": "five contrasts within model-metric-bundle-source",
            "outputs": {
                str(path): sha256_path(path) for path in (bc_path, all_path, summary_path)
            },
        }
        write_json(OUTPUT / "notebook11_status.json", status)
        display(summary)
        display(bc[bc["holm_significant_0_05"]])
        print(json.dumps(status, indent=2))
        """),
    ])


def notebook_12():
    save("12_joint_rescue_harm_and_qualitative_review.ipynb", [
        md("""
        # 12 — Joint B/C rescue, reverse harm, and blinded qualitative review

        This notebook completes the quantitative portion of Reviewer 2, Major
        Comment 1. A label-event rescue means B and C are both wrong while D is
        correct; harm is the reverse. Strict study rescue/harm and dominant
        error-count transitions are also reported. Patient/source-clustered
        intervals, paired sign-flip tests, and prespecified Holm corrections are
        used. The notebook then creates two independently completed, blinded
        qualitative-review forms. Quantitative completion does not imply that
        the human qualitative review is complete.
        """),
        code(BOOTSTRAP),
        code("""
        import importlib
        from datetime import datetime, timezone
        import numpy as np, pandas as pd
        from rerun_code.common import write_json
        from rerun_code.config import sha256_path
        from rerun_code.report_labeler import LABELS_13
        import rerun_code.post_rerun as post_module
        post_module = importlib.reload(post_module)
        if int(getattr(post_module, "POST_RERUN_API_VERSION", 0)) < 1:
            raise ImportError("Copy the updated rerun_code/post_rerun.py and restart the kernel.")
        from rerun_code.post_rerun import (
            JOINT_COUNT_COLUMNS, bootstrap_cluster_counts, build_blinded_qualitative_materials,
            clustered_sign_flip_p, load_statistical_frame, make_joint_inputs,
            pooled_cluster_counts, qualitative_case_sample, summarize_pooled_joint,
            joint_stratum_statistics, verified_per_study_path,
        )
        from rerun_code.statistics import holm_adjust

        OUTPUT = POST_ROOT / "joint_rescue_qualitative"
        QUALITATIVE = OUTPUT / "qualitative_review"
        OUTPUT.mkdir(parents=True, exist_ok=True)
        QUALITATIVE.mkdir(parents=True, exist_ok=True)
        per_study_path, upstream_audit = verified_per_study_path(PATHS)
        frame = load_statistical_frame(per_study_path)
        reps = int(CONFIG["statistics"]["permutation_replicates"])
        boot_reps = int(CONFIG["statistics"]["bootstrap_replicates"])
        confidence = float(CONFIG["statistics"]["confidence_level"])
        seed = int(CONFIG["statistics"]["seed"])
        """),
        code("""
        joint_rows, pooled_parts, case_rows = [], [], []
        grouped = list(frame.groupby(["model_key", "bundle", "source_dataset"], dropna=False))
        for index, (keys, group) in enumerate(grouped, start=1):
            inputs = make_joint_inputs(group)
            result, row_counts = joint_stratum_statistics(
                inputs, replicates=boot_reps, confidence_level=confidence, seed=seed
            )
            result.update({"model_key": keys[0], "bundle": keys[1], "source_dataset": keys[2]})
            joint_rows.append(result)
            base = inputs["base"][["query_record_id", "patient_key"]].copy()
            base["model_key"], base["bundle"], base["source_dataset"] = keys
            for column_index, name in enumerate(JOINT_COUNT_COLUMNS):
                base[name] = row_counts[:, column_index]
            for label_index, label in enumerate(LABELS_13):
                base[f"rescue__{label}"] = inputs["rescue"][:, label_index].astype(int)
                base[f"harm__{label}"] = inputs["harm"][:, label_index].astype(int)
                base[f"wrongopp__{label}"] = (
                    inputs["wrong_b"][:, label_index] & inputs["wrong_c"][:, label_index]
                ).astype(int)
                base[f"correctopp__{label}"] = (
                    ~inputs["wrong_b"][:, label_index] & ~inputs["wrong_c"][:, label_index]
                ).astype(int)
            pooled_parts.append(base)

            err_b = np.asarray(inputs["err_b"], dtype=int)
            err_c = np.asarray(inputs["err_c"], dtype=int)
            err_d = np.asarray(inputs["err_d"], dtype=int)
            masks = {
                "strict_rescue": (err_b > 0) & (err_c > 0) & (err_d == 0),
                "strict_harm": (err_b == 0) & (err_c == 0) & (err_d > 0),
                "dominant_rescue": err_d < np.minimum(err_b, err_c),
                "dominant_harm": err_d > np.maximum(err_b, err_c),
            }
            selected = np.logical_or.reduce(list(masks.values()))
            for row_index in np.flatnonzero(selected):
                tags = [name for name, mask in masks.items() if mask[row_index]]
                rescued = [
                    label for label_index, label in enumerate(LABELS_13)
                    if bool(inputs["rescue"][row_index, label_index])
                ]
                harmed = [
                    label for label_index, label in enumerate(LABELS_13)
                    if bool(inputs["harm"][row_index, label_index])
                ]
                query_id = str(base.iloc[row_index]["query_record_id"])
                case_rows.append({
                    "case_key": "|".join((str(keys[0]), str(keys[1]), str(keys[2]), query_id)),
                    "model_key": keys[0], "bundle": keys[1], "source_dataset": keys[2],
                    "query_record_id": query_id,
                    "transition_tags": ";".join(tags),
                    "b_label_error_count": int(err_b[row_index]),
                    "c_label_error_count": int(err_c[row_index]),
                    "d_label_error_count": int(err_d[row_index]),
                    "jointly_rescued_labels": ";".join(rescued),
                    "jointly_harmed_labels": ";".join(harmed),
                })
            print(f"Completed stratum {index}/{len(grouped)}: {keys}")

        joint = pd.DataFrame(joint_rows)
        joint["event_net_p_holm_28_strata"] = holm_adjust(joint["event_net_signflip_p"])
        joint["strict_study_net_p_holm_28_strata"] = holm_adjust(joint["strict_study_net_signflip_p"])
        pooled_rows = pd.concat(pooled_parts, ignore_index=True)
        cases = pd.DataFrame(case_rows).drop_duplicates("case_key").reset_index(drop=True)
        """),
        code("""
        pooled_summaries = [summarize_pooled_joint(
            pooled_rows, "all_models_and_strata", replicates=boot_reps,
            confidence_level=confidence, seed=seed,
        )]
        for model_key, subset in pooled_rows.groupby("model_key", sort=True):
            pooled_summaries.append(summarize_pooled_joint(
                subset, model_key, replicates=boot_reps,
                confidence_level=confidence, seed=seed,
            ))
        pooled = pd.DataFrame(pooled_summaries)
        model_mask = pooled["scope"] != "all_models_and_strata"
        pooled.loc[model_mask, "event_net_p_holm_7_models"] = holm_adjust(
            pooled.loc[model_mask, "event_net_signflip_p"]
        )
        pooled.loc[model_mask, "strict_study_net_p_holm_7_models"] = holm_adjust(
            pooled.loc[model_mask, "strict_study_net_signflip_p"]
        )

        label_rows = []
        for label in LABELS_13:
            columns = [f"rescue__{label}", f"harm__{label}", f"wrongopp__{label}", f"correctopp__{label}"]
            values = np.column_stack([
                pooled_rows[columns].to_numpy(dtype=float), np.ones(len(pooled_rows))
            ])
            counts, cluster_ids = pooled_cluster_counts(pooled_rows, values)
            formulas = {
                "rescue_rate_all": (0, 4), "harm_rate_all": (1, 4),
                "rescue_rate_opportunity": (0, 2), "harm_rate_opportunity": (1, 3),
                "net_rate_all": (0, 1, 4),
            }
            estimates = bootstrap_cluster_counts(
                counts, formulas, replicates=boot_reps,
                confidence_level=confidence, seed=seed,
            )
            totals = counts.sum(axis=0)
            row = {
                "label": label, "n_clusters": len(cluster_ids), "n_design_events": int(totals[4]),
                "rescue_n": int(totals[0]), "harm_n": int(totals[1]),
                "joint_wrong_opportunities": int(totals[2]),
                "joint_correct_opportunities": int(totals[3]),
                "net_signflip_p": clustered_sign_flip_p(
                    counts[:, 0] - counts[:, 1], replicates=reps, seed=seed
                ),
            }
            for name, (estimate, low, high, valid) in estimates.items():
                row[name] = estimate; row[f"{name}_ci_low"] = low
                row[f"{name}_ci_high"] = high; row[f"{name}_valid_bootstrap"] = valid
            label_rows.append(row)
        by_label = pd.DataFrame(label_rows)
        by_label["net_p_holm_13_labels"] = holm_adjust(by_label["net_signflip_p"])

        stratum_path = OUTPUT / "joint_b_c_wrong_d_correct_by_stratum.csv"
        pooled_path = OUTPUT / "joint_b_c_wrong_d_correct_pooled_and_by_model.csv"
        label_path = OUTPUT / "joint_b_c_wrong_d_correct_by_label.csv"
        cases_path = OUTPUT / "joint_transition_case_manifest.csv"
        joint.to_csv(stratum_path, index=False)
        pooled.to_csv(pooled_path, index=False)
        by_label.to_csv(label_path, index=False)
        cases.to_csv(cases_path, index=False)
        display(pooled)
        display(by_label)
        """),
        code("""
        sample_size = int(os.environ.get("JAMIA_QUALITATIVE_SAMPLE_SIZE", "60"))
        selected = qualitative_case_sample(cases, sample_size=sample_size, seed=seed)
        selected_path = QUALITATIVE / "qualitative_case_sample.csv"
        selected.to_csv(selected_path, index=False)
        materials = build_blinded_qualitative_materials(
            per_study_path, selected, QUALITATIVE, seed=seed
        )
        completed1 = QUALITATIVE / "qualitative_reviewer1_completed.csv"
        completed2 = QUALITATIVE / "qualitative_reviewer2_completed.csv"
        qualitative_forms_present = completed1.exists() and completed2.exists()
        status = {
            "quantitative_ready": True,
            "qualitative_forms_present": qualitative_forms_present,
            "qualitative_review_complete": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            **upstream_audit,
            "n_study_model_stratum_comparisons": int(pooled_rows["n_records"].sum()),
            "n_joint_case_manifest_rows": len(cases),
            "n_blinded_qualitative_cases": len(selected),
            "bootstrap_replicates": boot_reps,
            "permutation_replicates": reps,
            "confidence_level": confidence,
            "seed": seed,
            "quantitative_outputs": {
                str(path): sha256_path(path)
                for path in (stratum_path, pooled_path, label_path, cases_path, selected_path)
            },
            "qualitative_materials": materials,
            "next_action": (
                "Validate and summarize the two completed review forms in the next cell."
                if qualitative_forms_present else
                "Give the two templates to independent reviewers without the analyst-only crosswalk."
            ),
        }
        write_json(OUTPUT / "notebook12_status.json", status)
        print(json.dumps(status, indent=2))
        """),
        code("""
        # Optional second-stage validation and adjudication. Rerun this cell
        # after both reviewers return independently completed files. If any
        # binary ratings differ, the cell creates an adjudication template and
        # remains pending until a third reviewer completes it.
        from sklearn.metrics import cohen_kappa_score
        completed1 = QUALITATIVE / "qualitative_reviewer1_completed.csv"
        completed2 = QUALITATIVE / "qualitative_reviewer2_completed.csv"
        if not (completed1.exists() and completed2.exists()):
            print("QUALITATIVE REVIEW PENDING")
            print("Reviewer 1 form:", materials["reviewer1_template"])
            print("Reviewer 2 form:", materials["reviewer2_template"])
        else:
            r1 = pd.read_csv(completed1, dtype=str).fillna("")
            r2 = pd.read_csv(completed2, dtype=str).fillna("")
            keys = ["blinded_case_id", "candidate_code"]
            if r1.duplicated(keys).any() or r2.duplicated(keys).any():
                raise AssertionError("Duplicate blinded case/candidate identifiers in a completed form")
            merged = r1.merge(r2, on=keys, suffixes=("_r1", "_r2"), validate="one_to_one")
            if len(merged) != len(r1) or len(merged) != len(r2):
                raise AssertionError("The two reviewers did not assess identical blinded candidates")
            rating_fields = [
                "fabrication_present_0_or_1", "omission_present_0_or_1",
                "clinically_important_error_0_or_1", "overall_acceptable_0_or_1",
            ]
            agreement_rows = []
            disagreement = np.zeros(len(merged), dtype=bool)
            for field in rating_fields:
                a = pd.to_numeric(merged[f"{field}_r1"], errors="coerce")
                b = pd.to_numeric(merged[f"{field}_r2"], errors="coerce")
                if a.isna().any() or b.isna().any() or not a.isin([0, 1]).all() or not b.isin([0, 1]).all():
                    raise ValueError(f"Both reviewers must fill {field} with 0 or 1")
                disagreement |= a.ne(b).to_numpy()
                agreement_rows.append({
                    "rating": field,
                    "n": len(a),
                    "percent_agreement": float(a.eq(b).mean()),
                    "cohen_kappa": float(cohen_kappa_score(a, b)),
                })
            for suffix in ("r1", "r2"):
                reviewer_ids = merged[f"reviewer_id_{suffix}"].str.strip().unique()
                if len(reviewer_ids) != 1 or not reviewer_ids[0]:
                    raise ValueError(f"Reviewer {suffix} must use one nonempty reviewer_id")
            if merged["reviewer_id_r1"].iloc[0] == merged["reviewer_id_r2"].iloc[0]:
                raise ValueError("The two qualitative reviewers must have distinct reviewer IDs")
            agreement = pd.DataFrame(agreement_rows)
            agreement_path = QUALITATIVE / "qualitative_reader_agreement.csv"
            agreement.to_csv(agreement_path, index=False)
            disagreements = merged.loc[disagreement].copy()
            disagreements_path = QUALITATIVE / "qualitative_disagreements.csv"
            disagreements.to_csv(disagreements_path, index=False)
            print("Completed qualitative ratings:", len(merged))
            print("Rows with any binary-rating disagreement:", len(disagreements))
            display(agreement)

            adjudication_template = QUALITATIVE / "qualitative_adjudication_template.csv"
            adjudication_completed = QUALITATIVE / "qualitative_adjudication_completed.csv"
            final = merged[keys].copy()
            for field in rating_fields:
                a = pd.to_numeric(merged[f"{field}_r1"], errors="raise").astype(int)
                b = pd.to_numeric(merged[f"{field}_r2"], errors="raise").astype(int)
                final[f"final_{field}"] = np.where(a.eq(b), a, np.nan)

            if len(disagreements):
                adjudication = disagreements[keys].copy()
                for descriptive in (
                    "source_dataset", "reference_report", "reference_labels", "candidate_report"
                ):
                    adjudication[descriptive] = disagreements[f"{descriptive}_r1"]
                for field in rating_fields:
                    adjudication[f"reviewer1_{field}"] = disagreements[f"{field}_r1"]
                    adjudication[f"reviewer2_{field}"] = disagreements[f"{field}_r2"]
                    adjudication[f"adjudicated_{field}"] = ""
                adjudication["adjudicator_id"] = ""
                adjudication["adjudication_notes"] = ""
                adjudication.to_csv(adjudication_template, index=False)

            qualitative_complete = len(disagreements) == 0
            if len(disagreements) and adjudication_completed.exists():
                adj = pd.read_csv(adjudication_completed, dtype=str).fillna("")
                if adj.duplicated(keys).any():
                    raise AssertionError("Duplicate identifiers in qualitative adjudication")
                if len(adj) != len(disagreements):
                    raise AssertionError("Adjudication must contain every disagreement row exactly once")
                adj = disagreements[keys].merge(adj, on=keys, validate="one_to_one")
                adjudicator_ids = adj["adjudicator_id"].str.strip().unique()
                if len(adjudicator_ids) != 1 or not adjudicator_ids[0]:
                    raise ValueError("Use one nonempty adjudicator_id on every adjudication row")
                if adjudicator_ids[0] in {
                    merged["reviewer_id_r1"].iloc[0], merged["reviewer_id_r2"].iloc[0]
                }:
                    raise ValueError("The adjudicator must differ from both qualitative reviewers")
                for field in rating_fields:
                    column = f"adjudicated_{field}"
                    values = pd.to_numeric(adj[column], errors="coerce")
                    if values.isna().any() or not values.isin([0, 1]).all():
                        raise ValueError(f"Fill {column} with 0 or 1")
                    mapping = dict(zip(zip(adj["blinded_case_id"], adj["candidate_code"]), values.astype(int)))
                    mask = final.set_index(keys).index.isin(mapping)
                    final.loc[mask, f"final_{field}"] = [
                        mapping[key] for key in final.loc[mask, keys].itertuples(index=False, name=None)
                    ]
                qualitative_complete = True

            final_path = QUALITATIVE / "qualitative_final_ratings.csv"
            summary_path = QUALITATIVE / "qualitative_summary_by_condition.csv"
            if qualitative_complete:
                if final.filter(like="final_").isna().any().any():
                    raise AssertionError("Final qualitative ratings still contain unresolved disagreements")
                crosswalk = pd.read_csv(materials["analyst_only_crosswalk"])
                final = final.merge(crosswalk, on=keys, validate="one_to_one")
                final.to_csv(final_path, index=False)
                summary = final.groupby("condition")[[f"final_{field}" for field in rating_fields]].mean().reset_index()
                summary.to_csv(summary_path, index=False)
                display(summary)
                print("QUALITATIVE REVIEW COMPLETE")
            else:
                print("QUALITATIVE REVIEW PENDING — complete:", adjudication_template)

            status_path = OUTPUT / "notebook12_status.json"
            refreshed = json.loads(status_path.read_text(encoding="utf-8"))
            refreshed.update({
                "qualitative_forms_present": True,
                "qualitative_review_complete": qualitative_complete,
                "n_qualitative_rating_rows": len(merged),
                "n_rows_requiring_adjudication": len(disagreements),
                "reader_agreement": str(agreement_path),
                "disagreements": str(disagreements_path),
                "adjudication_template": str(adjudication_template) if len(disagreements) else None,
                "final_ratings": str(final_path) if qualitative_complete else None,
                "summary_by_condition": str(summary_path) if qualitative_complete else None,
            })
            write_json(status_path, refreshed)
        """),
    ])


def notebook_13():
    save("13_biomedclip_faiss_timing.ipynb", [
        md("""
        # 13 — BiomedCLIP query embedding and FAISS retrieval timing

        This notebook addresses Reviewer 2, Minor Comment 2. It measures image
        loading/preprocessing, BiomedCLIP query encoding, raw IndexFlatIP cosine
        search, and the leakage-safe retrieval wrapper separately. It reports
        median, interquartile range, mean, and 95th percentile on the executing
        hardware. The model identifier is taken from the original FAISS build
        configuration. Model loading and warm-up are excluded from per-query
        latency. By default, all 1,034 unique test images are measured three
        times and FAISS uses one CPU thread.
        """),
        code(BOOTSTRAP),
        code("""
        import importlib.metadata, platform, random, time
        from datetime import datetime, timezone
        import numpy as np, pandas as pd
        from PIL import Image
        import torch
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Install open-clip-torch in the Biowulf environment, then restart the kernel."
            ) from exc
        import faiss
        from rerun_code.common import write_json
        from rerun_code.config import sha256_path
        from rerun_code.leakage_safe_retrieval import (
            load_gallery_bundle, load_query_bundle, safe_search,
        )

        OUTPUT = POST_ROOT / "retrieval_timing"
        OUTPUT.mkdir(parents=True, exist_ok=True)
        MODEL_ID = os.environ.get(
            "JAMIA_BIOMEDCLIP_MODEL_ID",
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
        REPEATS = int(os.environ.get("JAMIA_TIMING_REPEATS", "3"))
        MAX_QUERIES = int(os.environ.get("JAMIA_TIMING_MAX_QUERIES", "0"))
        WARMUP = int(os.environ.get("JAMIA_TIMING_WARMUP", "10"))
        FAISS_THREADS = int(os.environ.get("JAMIA_FAISS_THREADS", "1"))
        SEED = int(CONFIG["statistics"]["seed"])
        MIN_EMBEDDING_COSINE = float(os.environ.get("JAMIA_MIN_EMBEDDING_COSINE", "0.99"))
        if REPEATS < 1 or FAISS_THREADS < 1:
            raise ValueError("JAMIA_TIMING_REPEATS and JAMIA_FAISS_THREADS must be positive")
        if not torch.cuda.is_available() and os.environ.get("JAMIA_ALLOW_CPU_TIMING", "0") != "1":
            raise RuntimeError(
                "A CUDA GPU is required for manuscript timing. Set JAMIA_ALLOW_CPU_TIMING=1 only for a smoke test."
            )
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        faiss.omp_set_num_threads(FAISS_THREADS)
        print("Device:", DEVICE, "FAISS threads:", FAISS_THREADS, "repeats:", REPEATS)
        """),
        code("""
        # Load the exact BiomedCLIP encoder recorded by the original FAISS build.
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
        model = model.to(DEVICE).eval()

        combined_vectors, combined_metadata = load_query_bundle(PATHS["bundles"] / "combined" / "queries")
        if len(combined_vectors) != 1034:
            print("Warning: expected 1,034 combined queries; observed", len(combined_vectors))
        unique_queries = []
        seen = set()
        for vector, metadata in zip(combined_vectors, combined_metadata):
            record_id = str(metadata["record_id"])
            if record_id in seen:
                continue
            seen.add(record_id)
            unique_queries.append((record_id, vector, metadata))
        if MAX_QUERIES > 0:
            rng = random.Random(SEED)
            rng.shuffle(unique_queries)
            unique_queries = unique_queries[:MAX_QUERIES]

        def resolve_image(metadata):
            original = Path(str(metadata.get("image_path", "")))
            dataset = str(metadata["dataset"])
            candidates = [
                original,
                Path(CONFIG["datasets"][dataset]["test"]) / original.name,
                Path(CONFIG["datasets"][dataset]["test"]) / str(metadata.get("image_id", "")),
            ]
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    return candidate
            raise FileNotFoundError(f"Cannot resolve image for {metadata['record_id']}: {candidates}")

        def synchronize():
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)

        # Warm-up is not included in timing.
        with torch.inference_mode():
            for _, _, metadata in unique_queries[:min(WARMUP, len(unique_queries))]:
                with Image.open(resolve_image(metadata)) as image:
                    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
                _ = model.encode_image(tensor)
            synchronize()
        print("Warm-up complete:", min(WARMUP, len(unique_queries)), "images")
        """),
        code("""
        embedding_rows = []
        generated_vectors = {}
        for repeat in range(REPEATS):
            order = list(unique_queries)
            random.Random(SEED + repeat).shuffle(order)
            with torch.inference_mode():
                for index, (record_id, stored_vector, metadata) in enumerate(order, start=1):
                    start = time.perf_counter_ns()
                    with Image.open(resolve_image(metadata)) as image:
                        tensor = preprocess(image.convert("RGB")).unsqueeze(0)
                    after_preprocess = time.perf_counter_ns()
                    tensor = tensor.to(DEVICE, non_blocking=False)
                    synchronize()
                    before_encode = time.perf_counter_ns()
                    encoded = model.encode_image(tensor).float()
                    encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    synchronize()
                    after_encode = time.perf_counter_ns()
                    vector = encoded[0].detach().cpu().numpy().astype(np.float32)
                    cosine = float(np.dot(vector, np.asarray(stored_vector, dtype=np.float32)))
                    generated_vectors[record_id] = vector
                    embedding_rows.append({
                        "query_record_id": record_id,
                        "source_dataset": metadata["dataset"],
                        "repeat": repeat + 1,
                        "preprocess_ms": (after_preprocess - start) / 1e6,
                        "host_to_device_and_sync_ms": (before_encode - after_preprocess) / 1e6,
                        "embedding_ms": (after_encode - before_encode) / 1e6,
                        "query_encoding_total_ms": (after_encode - start) / 1e6,
                        "cosine_vs_frozen_query_embedding": cosine,
                    })
                    if index % 100 == 0:
                        print(f"Embedding repeat {repeat + 1}/{REPEATS}: {index}/{len(order)}")
        embedding = pd.DataFrame(embedding_rows)
        minimum_observed = float(embedding["cosine_vs_frozen_query_embedding"].min())
        if minimum_observed < MIN_EMBEDDING_COSINE:
            raise AssertionError(
                f"New BiomedCLIP vectors do not reproduce the frozen query vectors: "
                f"minimum cosine={minimum_observed:.6f} < {MIN_EMBEDDING_COSINE}. "
                "Do not report timing until the original preprocessing/checkpoint is restored."
            )
        embedding_path = OUTPUT / "query_embedding_timing.csv"
        embedding.to_csv(embedding_path, index=False)
        print("Minimum embedding reproducibility cosine:", minimum_observed)
        """),
        code("""
        search_rows = []
        for bundle in ("mimic", "iuhn", "combined"):
            bundle_root = PATHS["bundles"] / bundle
            index, gallery_metadata = load_gallery_bundle(bundle_root / "gallery")
            _, query_metadata = load_query_bundle(bundle_root / "queries")
            if MAX_QUERIES > 0:
                selected_ids = set(generated_vectors)
                query_metadata = [row for row in query_metadata if row["record_id"] in selected_ids]
            for repeat in range(REPEATS):
                order = list(query_metadata)
                random.Random(SEED + 1000 + repeat).shuffle(order)
                for position, metadata in enumerate(order, start=1):
                    record_id = str(metadata["record_id"])
                    vector = generated_vectors[record_id].reshape(1, -1)
                    start_raw = time.perf_counter_ns()
                    raw_scores, raw_indices = index.search(vector, int(CONFIG["retrieval_k"]))
                    end_raw = time.perf_counter_ns()
                    start_safe = time.perf_counter_ns()
                    neighbors = safe_search(
                        index, gallery_metadata, vector[0], metadata,
                        k=int(CONFIG["retrieval_k"]),
                        phash_threshold=int(CONFIG["phash_threshold"]),
                    )
                    end_safe = time.perf_counter_ns()
                    if len(neighbors) != int(CONFIG["retrieval_k"]):
                        raise AssertionError("Leakage-safe retrieval returned the wrong number of neighbors")
                    search_rows.append({
                        "bundle": bundle,
                        "source_dataset": metadata["dataset"],
                        "query_record_id": record_id,
                        "repeat": repeat + 1,
                        "gallery_size": int(index.ntotal),
                        "k": int(CONFIG["retrieval_k"]),
                        "raw_faiss_indexflatip_ms": (end_raw - start_raw) / 1e6,
                        "leakage_safe_search_ms": (end_safe - start_safe) / 1e6,
                        "raw_rank1_cosine": float(raw_scores[0, 0]),
                    })
                    if position % 250 == 0:
                        print(f"Search {bundle} repeat {repeat + 1}/{REPEATS}: {position}/{len(order)}")
        search = pd.DataFrame(search_rows)
        search_path = OUTPUT / "faiss_search_timing.csv"
        search.to_csv(search_path, index=False)
        """),
        code("""
        def summarize(frame, group_columns, value_columns, stage):
            rows = []
            for keys, group in frame.groupby(group_columns, dropna=False):
                keys = keys if isinstance(keys, tuple) else (keys,)
                fixed = dict(zip(group_columns, keys))
                for value_column in value_columns:
                    values = pd.to_numeric(group[value_column], errors="coerce").dropna()
                    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
                    rows.append({
                        **fixed, "stage": stage, "measurement": value_column,
                        "n_measurements": len(values), "median_ms": median,
                        "q1_ms": q1, "q3_ms": q3, "iqr_ms": q3 - q1,
                        "mean_ms": values.mean(), "p95_ms": values.quantile(0.95),
                    })
            return rows

        summary_rows = []
        summary_rows.extend(summarize(
            embedding, ["source_dataset"],
            ["preprocess_ms", "embedding_ms", "query_encoding_total_ms"], "query_encoding",
        ))
        summary_rows.extend(summarize(
            search, ["bundle", "source_dataset"],
            ["raw_faiss_indexflatip_ms", "leakage_safe_search_ms"], "search",
        ))
        end_to_end = search.merge(
            embedding[["query_record_id", "repeat", "query_encoding_total_ms"]],
            on=["query_record_id", "repeat"], validate="many_to_one",
        )
        end_to_end["embedding_plus_safe_retrieval_ms"] = (
            end_to_end["query_encoding_total_ms"] + end_to_end["leakage_safe_search_ms"]
        )
        summary_rows.extend(summarize(
            end_to_end, ["bundle", "source_dataset"],
            ["embedding_plus_safe_retrieval_ms"], "end_to_end_retrieval",
        ))
        summary = pd.DataFrame(summary_rows)
        summary_path = OUTPUT / "retrieval_timing_summary.csv"
        summary.to_csv(summary_path, index=False)

        complete_cohort = MAX_QUERIES <= 0
        status = {
            "ready_for_manuscript": complete_cohort and DEVICE.type == "cuda",
            "run_scope": "full" if complete_cohort else f"smoke_test_{MAX_QUERIES}_queries",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": MODEL_ID,
            "model_loading_and_warmup_excluded": True,
            "device": str(DEVICE),
            "gpu_name": torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU",
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "open_clip_torch_version": importlib.metadata.version("open_clip_torch"),
            "faiss_version": getattr(faiss, "__version__", "unknown"),
            "faiss_threads": FAISS_THREADS,
            "repeats": REPEATS,
            "warmup_queries": min(WARMUP, len(unique_queries)),
            "n_unique_queries": len(unique_queries),
            "minimum_cosine_vs_frozen_query_embedding": float(
                embedding["cosine_vs_frozen_query_embedding"].min()
            ),
            "outputs": {
                str(path): sha256_path(path)
                for path in (embedding_path, search_path, summary_path)
            },
            "reporting_note": (
                "Report query encoding, raw FAISS search, and leakage-safe retrieval separately; "
                "identify hardware, FAISS thread count, repeats, and IQR."
            ),
        }
        write_json(OUTPUT / "notebook13_status.json", status)
        display(summary)
        print(json.dumps(status, indent=2))
        """),
    ])


for builder in (notebook_10, notebook_11, notebook_12, notebook_13):
    builder()

print("Generated four post-rerun notebooks in", ROOT)
