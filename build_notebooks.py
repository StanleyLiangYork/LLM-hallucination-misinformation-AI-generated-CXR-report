"""Generate the serial JAMIA rerun notebooks from readable source cells."""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent / "notebooks"
ROOT.mkdir(parents=True, exist_ok=True)


def md(value: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value: str):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


def save(name: str, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    nbf.write(notebook, ROOT / name)


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
# Never expose the implementation directory as a top-level import location:
# rerun_code/statistics.py would shadow Python's standard-library statistics.
implementation_dir = (RERUN_DIR / "src" / "rerun_code").resolve()
clean_sys_path = []
for entry in sys.path:
    try:
        resolved_entry = Path(entry or ".").resolve()
    except Exception:
        resolved_entry = None
    if resolved_entry != implementation_dir:
        clean_sys_path.append(entry)
sys.path[:] = clean_sys_path
sys.path.insert(0, str(RERUN_DIR / "src"))
from rerun_code.config import load_config, output_paths
RERUN_DIR, CONFIG = load_config(RERUN_DIR)
PATHS = output_paths(CONFIG)
print("Code:", RERUN_DIR)
print("Output:", PATHS["root"])
"""


def notebook_00():
    save("00_preflight_and_freeze_config.ipynb", [
        md("""
        # 00 — Preflight, provenance, and immutable configuration

        Run this notebook first on Biowulf. It records the exact adapter configurations, base-model identities, software/hardware, dataset and pre-encoded bundle availability, and the frozen rerun configuration. A folder label is never accepted as model identity when `adapter_config.json` says otherwise.
        """),
        code(BOOTSTRAP),
        code("""
        from rerun_code.config import environment_manifest, sha256_path
        from rerun_code.common import write_json

        manifest = environment_manifest(RERUN_DIR, CONFIG)
        path_checks = {}
        for dataset, spec in CONFIG["datasets"].items():
            path_checks[dataset] = {key: {"path": value, "exists": Path(value).exists()} for key, value in spec.items() if key in {"train", "test", "preencoded_train", "preencoded_test"}}
        manifest["path_checks"] = path_checks
        manifest["configuration"] = CONFIG
        write_json(PATHS["root"] / "run_manifest.json", manifest)
        print(json.dumps(manifest["adapter_audit"], indent=2))
        """),
        code("""
        missing_required = []
        for dataset in ("mimic", "iuhn"):
            for key in ("train", "test", "preencoded_train", "preencoded_test"):
                value = Path(CONFIG["datasets"][dataset][key])
                if not value.exists(): missing_required.append(str(value))
        for model_key, audit in manifest["adapter_audit"].items():
            if not audit["adapter_config_exists"]:
                missing_required.append(str(Path(audit["legacy_adapter"]) / "adapter_config.json"))
            if not audit["identity_gate_passed"]:
                print("IDENTITY HOLD:", model_key, audit["actual_base_model"] or "missing adapter config")
        if missing_required:
            raise FileNotFoundError("Missing required inputs:\\n" + "\\n".join(missing_required))
        print("PREFLIGHT PASSED. Resolve every IDENTITY HOLD before running that model.")
        """),
    ])


def notebook_01():
    save("01_build_manifests_and_leakage_gate.ipynb", [
        md("""
        # 01 — Rebuild manifests and enforce the leakage gate

        The supplied source split names are treated as candidates. Test cohorts remain fixed. Training/gallery rows are removed for shared patients/groups, studies, image IDs, exact image hashes, exact report hashes, or near-identical perceptual hashes. The combined gallery is cleaned again across datasets.
        """),
        code(BOOTSTRAP),
        code("""
        import pandas as pd
        from rerun_code.common import write_json
        from rerun_code.leakage_safe_data import (build_manifest, remove_query_conflicts_from_gallery, assert_leakage_free, overlap_audit, cohort_summary, describe_removals, write_jsonl, manifest_sha256)

        raw, clean, audits = {}, {}, {}
        for dataset in ("mimic", "iuhn"):
            spec = CONFIG["datasets"][dataset]
            train = build_manifest(spec["train"], dataset, "train", compute_phash=True, strict_pairs=True)
            test = build_manifest(spec["test"], dataset, "test", compute_phash=True, strict_pairs=True)
            gallery, removals = remove_query_conflicts_from_gallery(train, test, phash_threshold=CONFIG["phash_threshold"], remove_assumed_patient_groups=True)
            after = assert_leakage_free(gallery, test, phash_threshold=CONFIG["phash_threshold"])
            raw[(dataset, "train")], raw[(dataset, "test")] = train, test
            clean[(dataset, "gallery")], clean[(dataset, "queries")] = gallery, test
            target = PATHS["manifests"] / dataset
            write_jsonl(train, target / "candidate_train.jsonl"); write_jsonl(test, target / "fixed_queries.jsonl")
            write_jsonl(gallery, target / "clean_gallery.jsonl"); write_jsonl(removals, target / "excluded_gallery_records.jsonl")
            audits[dataset] = {"train": cohort_summary(train), "test": cohort_summary(test), "before": overlap_audit(train, test), "removed": describe_removals(removals), "after": after}
            print(dataset, json.dumps(audits[dataset], indent=2))
        """),
        code("""
        combined_candidate = pd.concat([clean[(d, "gallery")] for d in ("mimic", "iuhn")], ignore_index=True)
        combined_queries = pd.concat([clean[(d, "queries")] for d in ("mimic", "iuhn")], ignore_index=True)
        combined_gallery, combined_removed = remove_query_conflicts_from_gallery(combined_candidate, combined_queries, phash_threshold=CONFIG["phash_threshold"], remove_assumed_patient_groups=True)
        combined_after = assert_leakage_free(combined_gallery, combined_queries, phash_threshold=CONFIG["phash_threshold"])
        target = PATHS["manifests"] / "combined"
        write_jsonl(combined_gallery, target / "clean_gallery.jsonl"); write_jsonl(combined_queries, target / "fixed_queries.jsonl")
        write_jsonl(combined_removed, target / "excluded_gallery_records.jsonl")
        audits["combined"] = {"removed": describe_removals(combined_removed), "after": combined_after}
        checksums = {str(path.relative_to(PATHS["manifests"])): manifest_sha256(path) for path in sorted(PATHS["manifests"].rglob("*.jsonl"))}
        write_json(PATHS["manifests"] / "leakage_audit.json", {"audits": audits, "sha256": checksums})
        print("LEAKAGE GATE PASSED", json.dumps(checksums, indent=2))
        """),
    ])


def notebook_02():
    save("02_rebuild_training_only_faiss_bundles.ipynb", [
        md("""
        # 02 — Rebuild training-only FAISS galleries from pre-encoded vectors

        Pre-encoded train vectors are aligned to the clean gallery by identifiers. Pre-encoded test vectors become query matrices only; test indexes are never searched. Positional matching is disabled. Every saved neighbor is checked again against all leakage rules.
        """),
        code(BOOTSTRAP),
        code("""
        import numpy as np, pandas as pd
        from rerun_code.common import write_json
        from rerun_code.preencoded import load_preencoded, align_manifest_to_preencoded, inventory_preencoded
        from rerun_code.leakage_safe_data import read_jsonl, assert_leakage_free
        from rerun_code.leakage_safe_retrieval import save_gallery_bundle, save_query_bundle, load_gallery_bundle, load_query_bundle, safe_search, assert_neighbor_records, write_metadata

        built, provenance = {}, {}
        for dataset in ("mimic", "iuhn"):
            gallery = read_jsonl(PATHS["manifests"] / dataset / "clean_gallery.jsonl")
            queries = read_jsonl(PATHS["manifests"] / dataset / "fixed_queries.jsonl")
            assert_leakage_free(gallery, queries, phash_threshold=CONFIG["phash_threshold"])
            spec = CONFIG["datasets"][dataset]
            train_vectors, train_meta, train_prov = load_preencoded(spec["preencoded_train"])
            test_vectors, test_meta, test_prov = load_preencoded(spec["preencoded_test"])
            print(dataset, "training image vectors:", train_prov["vector_file"])
            print(dataset, "test/query image vectors:", test_prov["vector_file"])
            gallery_vectors, _, gallery_alignment = align_manifest_to_preencoded(gallery, train_vectors, train_meta, allow_positional=False)
            query_vectors, _, query_alignment = align_manifest_to_preencoded(queries, test_vectors, test_meta, allow_positional=False)
            root = PATHS["bundles"] / dataset
            save_gallery_bundle(gallery_vectors, gallery.to_dict("records"), root / "gallery")
            save_query_bundle(query_vectors, queries.to_dict("records"), root / "queries")
            built[dataset] = (gallery_vectors, gallery, query_vectors, queries)
            provenance[dataset] = {"train": train_prov, "test": test_prov, "gallery_alignment": gallery_alignment, "query_alignment": query_alignment}
        """),
        code("""
        combined_gallery_vectors = np.concatenate([built[d][0] for d in ("mimic", "iuhn")], axis=0)
        combined_gallery = pd.concat([built[d][1] for d in ("mimic", "iuhn")], ignore_index=True)
        combined_query_vectors = np.concatenate([built[d][2] for d in ("mimic", "iuhn")], axis=0)
        combined_queries = pd.concat([built[d][3] for d in ("mimic", "iuhn")], ignore_index=True)
        expected_gallery = read_jsonl(PATHS["manifests"] / "combined" / "clean_gallery.jsonl")
        expected_ids = expected_gallery["record_id"].tolist()
        index_by_id = {value: i for i, value in enumerate(combined_gallery["record_id"].tolist())}
        positions = [index_by_id[value] for value in expected_ids]
        combined_gallery_vectors = combined_gallery_vectors[positions]
        combined_gallery = combined_gallery.iloc[positions].reset_index(drop=True)
        root = PATHS["bundles"] / "combined"
        save_gallery_bundle(combined_gallery_vectors, combined_gallery.to_dict("records"), root / "gallery")
        save_query_bundle(combined_query_vectors, combined_queries.to_dict("records"), root / "queries")
        provenance["combined_legacy_inventory_only"] = {
            "train": inventory_preencoded(CONFIG["datasets"]["combined"]["preencoded_train"]),
            "test": inventory_preencoded(CONFIG["datasets"]["combined"]["preencoded_test"]),
            "note": "Primary combined bundle was assembled from the two identifier-aligned, cleaned component bundles."
        }
        """),
        code("""
        for dataset in ("mimic", "iuhn", "combined"):
            root = PATHS["bundles"] / dataset
            index, gallery_metadata = load_gallery_bundle(root / "gallery")
            query_vectors, query_metadata = load_query_bundle(root / "queries")
            neighbors = []
            for vector, query in zip(query_vectors, query_metadata):
                found = safe_search(index, gallery_metadata, vector, query, k=CONFIG["retrieval_k"], phash_threshold=CONFIG["phash_threshold"])
                assert_neighbor_records(query, found, expected_k=CONFIG["retrieval_k"], phash_threshold=CONFIG["phash_threshold"])
                neighbors.append({"query_record_id": query["record_id"], "neighbors": found})
            write_metadata(neighbors, root / "neighbors.jsonl")
            exact_like = [{"dataset": dataset, "query": row["query_record_id"], "rank1": row["neighbors"][0]["cosine_score"]} for row in neighbors if row["neighbors"][0]["cosine_score"] >= 0.9999]
            write_json(root / "rank1_score_ge_0_9999.json", exact_like)
            if exact_like: print("MANUAL REVIEW REQUIRED:", dataset, len(exact_like), "rank-1 scores >= 0.9999")
        write_json(PATHS["bundles"] / "preencoded_provenance.json", provenance)
        print("BUNDLE GATE PASSED. Review every rank-1 score >= 0.9999 before generation.")
        """),
    ])


def notebook_03():
    save("03_build_patient_disjoint_verifier_data.ipynb", [
        md("""
        # 03 — Correct verifier data provenance and grouped splits

        This replaces the legacy example-level 98%/2% split. Every harmony record must expose a patient or source-record grouping key. Evaluation patients, source IDs, and exact report hashes are removed, then groups are assigned 80%/10%/10% to train/validation/test.
        """), code(BOOTSTRAP),
        code("""
        import pandas as pd
        from rerun_code.common import read_jsonl, write_jsonl, write_json
        from rerun_code.verifier_data import standardize_verifier_records, remove_evaluation_overlap, grouped_stratified_split, assert_group_disjoint, split_summary

        source = Path(CONFIG["verifier_data"]["source_jsonl"])
        if not source.exists():
            raise FileNotFoundError(f"Set verifier_data.source_jsonl to the harmony/verifier JSONL with patient/source provenance: {source}")
        source_records = read_jsonl(source)
        standardized = standardize_verifier_records(source_records)
        print("Parsed verifier targets:", standardized["verdict_source"].value_counts().to_dict())
        print("Grouping provenance:", standardized["grouping_level"].value_counts().to_dict())
        print("Tasks:", standardized["task"].value_counts().to_dict())
        evaluation = pd.concat([pd.DataFrame(read_jsonl(PATHS["manifests"] / d / "fixed_queries.jsonl")) for d in ("mimic", "iuhn")], ignore_index=True)
        clean, excluded = remove_evaluation_overlap(standardized, evaluation)
        split = grouped_stratified_split(clean, CONFIG["verifier_data"]["split_fractions"], seed=CONFIG["verifier_data"]["seed"])
        assert_group_disjoint(split)
        summary = split_summary(split)
        if summary.get("test", {}).get("n_examples", 0) < CONFIG["verifier_data"]["minimum_test_examples"]:
            raise AssertionError(f"Verifier test split is too small: {summary}")
        write_jsonl(PATHS["verifier_data"] / "all_grouped.jsonl", split.to_dict("records"))
        write_jsonl(PATHS["verifier_data"] / "excluded_evaluation_overlap.jsonl", excluded.to_dict("records"))
        for name in ("train", "validation", "test"):
            write_jsonl(PATHS["verifier_data"] / f"{name}.jsonl", split.loc[split["split"] == name].to_dict("records"))
        write_json(PATHS["verifier_data"] / "split_audit.json", {
            "source": str(source), "n_source": len(standardized), "n_excluded": len(excluded),
            "verdict_sources": standardized["verdict_source"].value_counts().to_dict(),
            "grouping_levels": standardized["grouping_level"].value_counts().to_dict(),
            "tasks": standardized["task"].value_counts().to_dict(), "summary": summary,
            "limitation": "The supplied harmony file has source image/report IDs but no patient IDs; splitting is source-disjoint rather than demonstrably patient-disjoint."
        })
        print(json.dumps(summary, indent=2)); print("VERIFIER SPLIT GATE PASSED")
        """),
    ])


def notebook_04():
    save("04_train_and_test_corrected_verifiers.ipynb", [
        md("""
        # 04 — Train and independently test corrected verifier LoRAs

        Run one model per Biowulf job using `JAMIA_MODEL_KEY`. The legacy adapter supplies only its documented LoRA architecture and provenance; its weights are **not** used to initialize the corrected adapter. Each corrected adapter is trained on the grouped train split, selected on validation loss, and scored once on the untouched grouped test split.

        This notebook is restart-safe. Cell 2 skips fine-tuning when it finds a complete corrected adapter whose weight file and training provenance match `model_key` and the canonical training base. Partial output is moved to a recoverable timestamped quarantine directory before a clean run; contradictory completed provenance remains a hard stop. The independent test appends and validates one prediction at a time, resumes only missing verifier records, and skips model loading when both evaluation artifacts are already complete. CPU offload is prohibited during training but permitted and recorded during evaluation; disk offload remains prohibited.

        The four text-model profiles in this revision reproduce their supplied reference notebooks: GPT-OSS-20B, Llama-3-8B, Med-Qwen2-7B, and OpenBioLLM-8B use `AutoModelForCausalLM` with `torch_dtype=bfloat16` and no bitsandbytes configuration. GPT-OSS retains its checkpoint-native MXFP4 weights. The existing `llama_3_1_8b` job key is retained only for script compatibility; its audited checkpoint is `unsloth/llama-3-8b-Instruct` and all outputs identify it as Llama-3-8B, not Llama 3.1.
        """), code(BOOTSTRAP),
        code("""
        import pandas as pd
        from rerun_code.common import read_jsonl, write_json
        from rerun_code.config import sha256_path
        from rerun_code.modeling import adapter_config, completed_corrected_adapter, IncompleteCorrectedAdapterError, quarantine_incomplete_corrected_adapter, release_accelerator_memory, resolve_training_base_model, train_corrected_verifier, load_processor_and_model, assert_model_identity
        from rerun_code.generation import ModelRunner
        from rerun_code.verifier_data import verifier_prompt

        import importlib.metadata as package_metadata
        print("Python:", sys.version.split()[0])
        runtime_packages = {}
        for package in ("torch", "transformers", "kernels", "peft", "accelerate", "bitsandbytes", "safetensors", "huggingface-hub"):
            try: runtime_packages[package] = package_metadata.version(package)
            except package_metadata.PackageNotFoundError: runtime_packages[package] = "NOT INSTALLED"
            print(package, runtime_packages[package])

        model_key = os.environ.get("JAMIA_MODEL_KEY", "qwen2_1_5b")
        if model_key not in CONFIG["models"]: raise KeyError(model_key)
        spec = CONFIG["models"][model_key]
        legacy_config = adapter_config(spec["legacy_adapter"])
        training_base = resolve_training_base_model(spec)
        print("Legacy adapter base (audit only):", legacy_config.get("base_model_name_or_path"))
        print("Fresh corrected-training base:", training_base)
        print("Loader profile:", spec.get("loader_profile", "default"))
        print("Tokenizer source:", spec.get("tokenizer_source", "base"))
        print("Chat-template policy:", spec.get("chat_template_policy", "auto"))
        print("Reference loading notebook:", spec.get("reference_loading_notebook", "not registered"))
        if spec.get("manuscript_identity_note"):
            print("MANUSCRIPT IDENTITY NOTE:", spec["manuscript_identity_note"])
        train = pd.DataFrame(read_jsonl(PATHS["verifier_data"] / "train.jsonl"))
        validation = pd.DataFrame(read_jsonl(PATHS["verifier_data"] / "validation.jsonl"))
        test = pd.DataFrame(read_jsonl(PATHS["verifier_data"] / "test.jsonl"))
        model_out = PATHS["verifiers"] / model_key
        try:
            completed = completed_corrected_adapter(model_out, model_key, training_base, spec=spec)
        except IncompleteCorrectedAdapterError as exc:
            quarantined = quarantine_incomplete_corrected_adapter(exc.output_dir)
            print("INCOMPLETE PRIOR TRAINING OUTPUT QUARANTINED:", quarantined)
            print("The directory was moved, not deleted. Starting a clean run for:", training_base)
            completed = None
        if completed is not None:
            adapter_dir = completed["adapter_dir"]
            actual_base = completed["actual_base"]
            history = completed["history"]
            print("FINE-TUNING SKIPPED: audited corrected adapter already exists.")
            print("Corrected adapter:", adapter_dir)
            print("Corrected adapter weights:", completed["weights_file"])
            if completed["adapter_base_before_normalization"] != actual_base:
                print(
                    "Normalized saved adapter base from",
                    completed["adapter_base_before_normalization"], "to", actual_base,
                )
        else:
            print("No completed corrected adapter found; starting fine-tuning.")
            adapter_dir, actual_base, history = train_corrected_verifier(model_key, spec, train, validation, model_out, verifier_prompt)
            write_json(model_out / "training_provenance.json", {
                "model_key": model_key, "actual_training_base": actual_base,
                "legacy_adapter_base_audit_only": legacy_config.get("base_model_name_or_path"),
                "legacy_adapter_used_for_configuration_only": spec["legacy_adapter"],
                "legacy_adapter_config_sha256": sha256_path(Path(spec["legacy_adapter"]) / "adapter_config.json"),
                "legacy_weights_loaded": False,
                "loader_profile": spec.get("loader_profile", "default"),
                "tokenizer_source": spec.get("tokenizer_source", "base"),
                "chat_template_policy": spec.get("chat_template_policy", "auto"),
                "reference_loading_notebook": spec.get("reference_loading_notebook"),
                "display_name": spec["display_name"],
                "manuscript_identity_note": spec.get("manuscript_identity_note"),
                "runtime_packages": runtime_packages,
                "history": history
            })
            print("Saved corrected adapter:", adapter_dir)

        independent_predictions_path = model_out / "independent_test_predictions.jsonl"
        independent_metrics_path = model_out / "independent_test_metrics.json"
        missing_evaluation_artifacts = [
            str(path) for path in (independent_predictions_path, independent_metrics_path)
            if not path.exists()
        ]
        if missing_evaluation_artifacts:
            print("INDEPENDENT EVALUATION REQUIRED. Missing:", missing_evaluation_artifacts)
        else:
            print("Independent-evaluation artifacts exist and will be audited in the next cell.")
        """),
        code("""
        def audit_saved_independent_predictions(path, test_frame):
            expected = {
                str(row["verifier_record_id"]): {
                    "truth": bool(row["verdict"]),
                    "patient_or_source_group": str(row["patient_or_source_group"]),
                }
                for _, row in test_frame.iterrows()
            }
            saved = read_jsonl(path) if path.exists() else []
            completed_ids = set()
            for line_number, row in enumerate(saved, start=1):
                record_id = str(row.get("verifier_record_id") or "")
                if record_id not in expected:
                    raise RuntimeError(
                        f"Unknown verifier_record_id in {path}:{line_number}: {record_id!r}"
                    )
                if record_id in completed_ids:
                    raise RuntimeError(f"Duplicate verifier_record_id in {path}: {record_id}")
                if bool(row.get("truth")) != expected[record_id]["truth"]:
                    raise RuntimeError(f"Truth mismatch for {record_id} in {path}")
                if str(row.get("patient_or_source_group") or "") != expected[record_id]["patient_or_source_group"]:
                    raise RuntimeError(f"Group mismatch for {record_id} in {path}")
                if row.get("verdict") not in (True, False):
                    raise RuntimeError(
                        f"Saved verifier output for {record_id} is unparseable. "
                        "Move the independent prediction file aside before retrying."
                    )
                completed_ids.add(record_id)
            return completed_ids, saved

        def score_independent_predictions(rows, evaluation_device_map, evaluation_cpu_offload):
            scored = pd.DataFrame(rows)
            if len(scored) != len(test):
                raise AssertionError(
                    f"Independent evaluation has {len(scored)} predictions; expected {len(test)}"
                )
            if scored["verdict"].isna().any():
                raise AssertionError(f"Unparseable verifier outputs: {scored['verdict'].isna().sum()}")
            truth, pred = scored["truth"].astype(bool), scored["verdict"].astype(bool)
            tp = int((truth & pred).sum())
            tn = int((~truth & ~pred).sum())
            fp = int((~truth & pred).sum())
            fn = int((truth & ~pred).sum())
            return {
                "model_key": model_key,
                "actual_base_model": actual_base,
                "n": len(scored),
                "accuracy": (tp + tn) / len(scored),
                "sensitivity": tp / (tp + fn) if tp + fn else None,
                "specificity": tn / (tn + fp) if tn + fp else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "evaluation_cpu_offload": evaluation_cpu_offload,
                "evaluation_device_map": evaluation_device_map,
                "test_split_sha256": sha256_path(PATHS["verifier_data"] / "test.jsonl"),
            }

        completed_ids, saved_predictions = audit_saved_independent_predictions(
            independent_predictions_path, test
        )
        expected_ids = set(test["verifier_record_id"].astype(str))
        pending_ids = expected_ids - completed_ids
        metrics_complete = independent_metrics_path.exists()
        if not pending_ids and metrics_complete:
            existing_metrics = json.loads(independent_metrics_path.read_text(encoding="utf-8"))
            if int(existing_metrics.get("n") or 0) != len(test):
                raise RuntimeError(
                    f"Existing independent metrics report n={existing_metrics.get('n')}; "
                    f"expected {len(test)}. Move {independent_metrics_path} aside and rerun."
                )
            print("INDEPENDENT EVALUATION SKIPPED: predictions and metrics are complete.")
            print(json.dumps(existing_metrics, indent=2))
        else:
            print(
                f"Independent evaluation resume audit: complete={len(completed_ids)}, "
                f"pending={len(pending_ids)}, metrics_complete={metrics_complete}"
            )
            recorded_adapter_base = adapter_config(adapter_dir).get("base_model_name_or_path")
            print("Corrected adapter-recorded base:", recorded_adapter_base)
            print("Reloading audited canonical base:", actual_base)
            release_accelerator_memory()
            processor, tokenizer, model, actual_base = load_processor_and_model(
                spec,
                adapter_path=adapter_dir,
                base_model_override=actual_base,
            )
            assert_model_identity(model_key, spec, actual_base)
            loaded_backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
            evaluation_device_map = {
                str(key): str(value)
                for key, value in (getattr(loaded_backbone, "hf_device_map", {}) or {}).items()
            }
            evaluation_cpu_offload = any(
                value.lower() == "cpu" for value in evaluation_device_map.values()
            )
            print("Evaluation CPU offload:", evaluation_cpu_offload)
            runner = ModelRunner(
                processor, tokenizer, model, spec["architecture"], CONFIG["generation"],
                actual_base=actual_base,
            )
            progress_every = max(1, int(os.environ.get("JAMIA_VERIFIER_PROGRESS_EVERY", "25")))
            generated = 0
            for _, row in test.iterrows():
                record_id = str(row["verifier_record_id"])
                if record_id in completed_ids:
                    continue
                result = runner.verify(
                    str(row["report_text"]),
                    adapter_enabled=True,
                    image_path=str(row.get("image_path", "") or ""),
                )
                if result.get("verdict") not in (True, False):
                    raise RuntimeError(
                        f"Unparseable verifier output for {record_id}: {result.get('raw')!r}. "
                        "This record was not saved."
                    )
                output = {
                    "verifier_record_id": record_id,
                    "patient_or_source_group": str(row["patient_or_source_group"]),
                    "truth": bool(row["verdict"]),
                    **result,
                }
                with independent_predictions_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(output, ensure_ascii=False) + "\\n")
                    handle.flush()
                completed_ids.add(record_id)
                generated += 1
                if generated % progress_every == 0 or not (expected_ids - completed_ids):
                    print(
                        f"Independent evaluation progress: generated={generated}/{len(pending_ids)}, "
                        f"total_complete={len(completed_ids)}/{len(expected_ids)}"
                    )
            if completed_ids != expected_ids:
                raise AssertionError(
                    f"Independent evaluation ended with {len(expected_ids - completed_ids)} missing records"
                )
            final_predictions = read_jsonl(independent_predictions_path)
            summary = score_independent_predictions(
                final_predictions, evaluation_device_map, evaluation_cpu_offload
            )
            write_json(independent_metrics_path, summary)
            print("Saved independent predictions:", independent_predictions_path)
            print("Saved independent metrics:", independent_metrics_path)
            print(json.dumps(summary, indent=2))
        """),
        md("Repeat this notebook for all seven keys: `medgemma_4b`, `phi4_multimodal`, `gpt_oss_20b`, `llama_3_1_8b`, `medqwen2_7b`, `openbiollm_8b`, and `qwen2_1_5b`. The compatibility key `llama_3_1_8b` writes Llama-3-8B identity into its outputs because that is the checkpoint used by the supplied reference notebook and adapter."),
    ])


def notebook_05():
    save("05_run_multimodel_generation.ipynb", [
        md("""
        # 05 — Run the complete four-condition generation experiment

        The editable `MODEL_KEYS` and `BUNDLE_NAMES` lists define the combinations to run. The outer loop loads each model once, the inner loop processes its selected bundles, and GPU memory is released before the next model. Comma-separated `JAMIA_MODEL_KEYS` and `JAMIA_BUNDLES` variables can override the lists without editing the notebook. The legacy singular variables `JAMIA_MODEL_KEY` and `JAMIA_BUNDLE` remain supported for one-model or one-bundle jobs.

        The authentication cell runs before Transformers is imported. It uses a saved Hugging Face credential or `HF_TOKEN` without printing the token. If no credential is available, it opens Hugging Face's secure login flow. For a compute node without outbound access, restart the kernel with `JAMIA_HF_OFFLINE=1`; this skips login and restricts loading to the shared Hugging Face cache. Offline mode works only if notebook 04 or an earlier download has already cached every selected checkpoint.

        Records are appended after every query/condition. Before an existing record is skipped, the notebook validates its model key, bundle, run fingerprint, query ID, condition, generated ID, uniqueness, and nonempty report. Thus, an interrupted multi-combination job can be restarted without duplicating completed work. Cross-model or stale records stop only the affected combination when continue-on-error mode is enabled.

        Notebook 05 never fine-tunes. It requires notebook 04's audited corrected adapter and independent evaluation for each selected model. CPU offload is permitted for inference and fingerprinted in provenance; disk offload is rejected. If the full loop exceeds the Biowulf wall-time or GPU-memory allocation, divide `MODEL_KEYS` into smaller batches and rerun. The record-level resume safeguard preserves completed work.
        """), code(BOOTSTRAP),
        code("""
        # Hugging Face authentication and network mode. This cell must run before
        # importing Transformers or rerun_code.modeling.
        hf_offline = os.environ.get("JAMIA_HF_OFFLINE", "0").strip().lower() in {"1", "true", "yes", "on"}
        hf_login_enabled = os.environ.get("JAMIA_HF_LOGIN", "1").strip().lower() not in {"0", "false", "no", "off"}
        if hf_offline:
            already_imported = [name for name in ("huggingface_hub", "transformers") if name in sys.modules]
            if already_imported:
                raise RuntimeError(
                    "Offline mode must be enabled before Hugging Face libraries are imported. "
                    f"Already imported={already_imported}. Restart the kernel and rerun from the first cell."
                )
            os.environ["HF_HUB_OFFLINE"] = "1"
            print("Hugging Face offline-cache mode enabled; no login or Hub request will be attempted.")
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
            from huggingface_hub import get_token, login

            environment_token = os.environ.get("HF_TOKEN", "").strip()
            cached_token_available = bool(get_token())
            if environment_token:
                login(
                    token=environment_token,
                    add_to_git_credential=False,
                    skip_if_logged_in=False,
                )
                print("Hugging Face authentication completed using HF_TOKEN.")
            elif cached_token_available:
                print("Using the existing cached Hugging Face credential.")
            elif hf_login_enabled:
                print("No Hugging Face credential was found. Starting the secure login flow.")
                login(add_to_git_credential=False, skip_if_logged_in=True)
                print("Hugging Face authentication completed.")
            else:
                print(
                    "Hugging Face login is disabled and no credential was found. "
                    "Only public checkpoints can be downloaded."
                )
            del environment_token
        """),
        code("""
        import hashlib, importlib, traceback
        from datetime import datetime, timezone
        from rerun_code.common import read_jsonl, write_json
        from rerun_code.config import sha256_path
        import rerun_code.modeling as modeling_module
        from rerun_code.modeling import adapter_config, completed_corrected_adapter, load_processor_and_model, release_accelerator_memory, assert_model_identity, resolve_training_base_model
        import rerun_code.generation as generation_module
        generation_module = importlib.reload(generation_module)
        required_generation_api = 5
        actual_generation_api = int(getattr(generation_module, "GENERATION_API_VERSION", 0))
        if actual_generation_api < required_generation_api:
            raise ImportError(
                "Notebook 05 and rerun_code/generation.py are out of sync. "
                f"Required generation API {required_generation_api}, found {actual_generation_api} at "
                f"{Path(generation_module.__file__).resolve()}. Copy the updated rerun_code/generation.py "
                "to the Biowulf re_run folder, then rerun this cell."
            )
        required_resume_repair = 1
        actual_resume_repair = int(
            getattr(generation_module, "GENERATION_RESUME_REPAIR_VERSION", 0)
        )
        if actual_resume_repair < required_resume_repair:
            raise ImportError(
                "Notebook 05 requires the audited failed-record recovery helper. "
                f"Required repair version {required_resume_repair}, found {actual_resume_repair} at "
                f"{Path(generation_module.__file__).resolve()}. Copy the updated "
                "rerun_code/generation.py to the active Biowulf re_run folder, then rerun this cell."
            )
        required_phi4_image_compat = 3
        actual_phi4_image_compat = int(
            getattr(generation_module, "PHI4_IMAGE_COMPAT_VERSION", 0)
        )
        if actual_phi4_image_compat < required_phi4_image_compat:
            raise ImportError(
                "Notebook 05 requires the legacy-NumPy Phi-4/SigLIP2 image compatibility fix. "
                f"Required version {required_phi4_image_compat}, found {actual_phi4_image_compat} at "
                f"{Path(generation_module.__file__).resolve()}. Copy the updated "
                "rerun_code/generation.py to the active Biowulf re_run folder, restart the kernel, "
                "and rerun this cell."
            )
        required_phi4_cache_compat = 1
        actual_phi4_cache_compat = int(
            getattr(generation_module, "PHI4_CACHE_COMPAT_VERSION", 0)
        )
        if actual_phi4_cache_compat < required_phi4_cache_compat:
            raise ImportError(
                "Notebook 05 requires the Phi-4 DynamicCache compatibility fix. "
                f"Required version {required_phi4_cache_compat}, found {actual_phi4_cache_compat} at "
                f"{Path(generation_module.__file__).resolve()}. Copy the updated "
                "rerun_code/generation.py to the active Biowulf re_run folder, restart the kernel, "
                "and rerun this cell."
            )
        required_phi4_verifier_compat = 1
        actual_phi4_verifier_compat = int(
            getattr(generation_module, "PHI4_VERIFIER_COMPAT_VERSION", 0)
        )
        if actual_phi4_verifier_compat < required_phi4_verifier_compat:
            raise ImportError(
                "Notebook 05 requires the Phi-4 BatchEncoding verifier compatibility fix. "
                f"Required version {required_phi4_verifier_compat}, found "
                f"{actual_phi4_verifier_compat} at "
                f"{Path(generation_module.__file__).resolve()}. Copy the updated "
                "rerun_code/generation.py to the active Biowulf re_run folder, restart the kernel, "
                "and rerun this cell."
            )
        required_phi4_empty_recovery = 1
        actual_phi4_empty_recovery = int(
            getattr(generation_module, "PHI4_EMPTY_OUTPUT_RECOVERY_VERSION", 0)
        )
        if actual_phi4_empty_recovery < required_phi4_empty_recovery:
            raise ImportError(
                "Notebook 05 requires the audited Phi-4 empty-output recovery. "
                f"Required version {required_phi4_empty_recovery}, found "
                f"{actual_phi4_empty_recovery} at "
                f"{Path(generation_module.__file__).resolve()}. Copy the updated "
                "rerun_code/generation.py to the active Biowulf re_run folder, restart the kernel, "
                "and rerun this cell."
            )
        ModelRunner = generation_module.ModelRunner
        CONDITIONS = generation_module.CONDITIONS
        run_condition = generation_module.run_condition
        validated_completed_generation_ids = generation_module.validated_completed_generation_ids
        from rerun_code.leakage_safe_retrieval import load_query_bundle

        # Edit either list to run only a subset. The order is preserved.
        MODEL_KEYS = [
            "qwen2_1_5b",
            "medgemma_4b",
            "phi4_multimodal",
            "gpt_oss_20b",
            "llama_3_1_8b",
            "medqwen2_7b",
            "openbiollm_8b",
        ]
        BUNDLE_NAMES = ["mimic", "iuhn", "combined"]

        def selected_values(plural_env, singular_env, defaults):
            if os.environ.get(plural_env):
                values = [value.strip() for value in os.environ[plural_env].split(",") if value.strip()]
            elif os.environ.get(singular_env):
                values = [os.environ[singular_env].strip()]
            else:
                values = list(defaults)
            if not values:
                raise ValueError(f"No values selected for {plural_env}")
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate values selected for {plural_env}: {values}")
            return values

        model_keys = selected_values("JAMIA_MODEL_KEYS", "JAMIA_MODEL_KEY", MODEL_KEYS)
        bundle_names = selected_values("JAMIA_BUNDLES", "JAMIA_BUNDLE", BUNDLE_NAMES)
        n_items = int(os.environ["JAMIA_N_ITEMS"]) if os.environ.get("JAMIA_N_ITEMS") else None
        progress_every = max(1, int(os.environ.get("JAMIA_PROGRESS_EVERY", "10")))
        continue_on_error = os.environ.get("JAMIA_CONTINUE_ON_ERROR", "1").strip().lower() not in {"0", "false", "no"}
        unknown_models = sorted(set(model_keys) - set(CONFIG["models"]))
        unknown_bundles = sorted(set(bundle_names) - {"mimic", "iuhn", "combined"})
        if unknown_models: raise KeyError(f"Unknown model keys: {unknown_models}")
        if unknown_bundles: raise KeyError(f"Unknown bundle names: {unknown_bundles}")
        if n_items is not None and n_items <= 0: raise ValueError("JAMIA_N_ITEMS must be positive")
        print("Selected models:", model_keys)
        print("Selected bundles:", bundle_names)
        print("Smoke-test query limit:", n_items)
        print("Continue after an isolated combination error:", continue_on_error)

        def load_model_context(model_key):
            spec = CONFIG["models"][model_key]
            canonical_base = resolve_training_base_model(spec)
            model_out = PATHS["verifiers"] / model_key
            adapter_audit = completed_corrected_adapter(model_out, model_key, canonical_base, spec=spec)
            if adapter_audit is None:
                raise FileNotFoundError(
                    f"No completed corrected adapter found in {model_out}. Run notebook 04 for {model_key}."
                )
            corrected_adapter = adapter_audit["adapter_dir"]
            adapter_weights_file = adapter_audit["weights_file"]
            corrected_adapter_config = adapter_config(corrected_adapter)
            training_provenance_path = model_out / "training_provenance.json"
            independent_metrics_path = model_out / "independent_test_metrics.json"
            independent_predictions_path = model_out / "independent_test_predictions.jsonl"
            missing = [str(path) for path in (independent_metrics_path, independent_predictions_path) if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    "Notebook 04 independent evaluation is incomplete for "
                    f"{model_key}. Missing={missing}"
                )
            independent_metrics = json.loads(independent_metrics_path.read_text(encoding="utf-8"))
            independent_predictions = read_jsonl(independent_predictions_path)
            if int(independent_metrics.get("n") or 0) <= 0:
                raise AssertionError(f"Invalid independent verifier sample count for {model_key}")
            if int(independent_metrics["n"]) != len(independent_predictions):
                raise AssertionError(
                    f"Independent verifier metrics for {model_key} report n={independent_metrics['n']}, "
                    f"but predictions contain {len(independent_predictions)} records"
                )

            print("\\nLoading model:", model_key, spec["display_name"])
            print("Canonical backbone:", canonical_base)
            print("Corrected adapter:", corrected_adapter)
            release_accelerator_memory()
            processor, tokenizer, model, actual_base = load_processor_and_model(
                spec, adapter_path=corrected_adapter, base_model_override=canonical_base,
            )
            assert_model_identity(model_key, spec, actual_base)
            if not getattr(model, "peft_config", None):
                raise AssertionError(f"Corrected LoRA was not attached for {model_key}")
            loaded_backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
            backbone_load_method = getattr(loaded_backbone, "_jamia_load_method", "transformers_from_pretrained")
            backbone_device_map = {
                str(key): str(value)
                for key, value in (getattr(loaded_backbone, "hf_device_map", {}) or {}).items()
            }
            evaluation_cpu_offload = any(value.lower() == "cpu" for value in backbone_device_map.values())
            runner = ModelRunner(
                processor, tokenizer, model, spec["architecture"], CONFIG["generation"],
                actual_base=actual_base,
            )
            if model_key == "phi4_multimodal":
                print(
                    "Phi-4 DynamicCache compatibility targets:",
                    list(runner.phi4_dynamic_cache_targets),
                )
            return {
                "model_key": model_key, "spec": spec, "canonical_base": canonical_base,
                "actual_base": actual_base, "model_out": model_out,
                "corrected_adapter": corrected_adapter,
                "adapter_weights_file": adapter_weights_file,
                "corrected_adapter_config": corrected_adapter_config,
                "training_provenance_path": training_provenance_path,
                "independent_metrics_path": independent_metrics_path,
                "independent_predictions_path": independent_predictions_path,
                "independent_metrics": independent_metrics,
                "backbone_load_method": backbone_load_method,
                "backbone_device_map": backbone_device_map,
                "evaluation_cpu_offload": evaluation_cpu_offload,
                "runner": runner,
            }

        def run_model_bundle(context, bundle_name):
            model_key, spec = context["model_key"], context["spec"]
            _, all_queries = load_query_bundle(PATHS["bundles"] / bundle_name / "queries")
            queries = all_queries[:n_items] if n_items is not None else all_queries
            neighbors_path = PATHS["bundles"] / bundle_name / "neighbors.jsonl"
            neighbors_by_query = {row["query_record_id"]: row["neighbors"] for row in read_jsonl(neighbors_path)}
            missing_neighbors = sorted({row["record_id"] for row in queries} - set(neighbors_by_query))
            if missing_neighbors:
                raise KeyError(f"Missing neighbors for {model_key}/{bundle_name}: {missing_neighbors[:10]}")
            output_dir = PATHS["generation"] / model_key / bundle_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "results.jsonl"

            run_provenance = {
                "model_key": model_key,
                "bundle": bundle_name,
                "canonical_base_model": context["canonical_base"],
                "actual_base_model": context["actual_base"],
                "architecture": spec["architecture"],
                "model_display_name": spec["display_name"],
                "loader_profile": spec.get("loader_profile", "default"),
                "tokenizer_source": spec.get("tokenizer_source", "base"),
                "chat_template_policy": spec.get("chat_template_policy", "auto"),
                "reference_loading_notebook": spec.get("reference_loading_notebook"),
                "manuscript_identity_note": spec.get("manuscript_identity_note"),
                "corrected_adapter": str(context["corrected_adapter"]),
                "adapter_recorded_base_model": context["corrected_adapter_config"].get("base_model_name_or_path"),
                "adapter_config_sha256": sha256_path(context["corrected_adapter"] / "adapter_config.json"),
                "adapter_weights_file": context["adapter_weights_file"].name,
                "adapter_weights_sha256": sha256_path(context["adapter_weights_file"]),
                "training_provenance_sha256": sha256_path(context["training_provenance_path"]),
                "independent_test_metrics": context["independent_metrics"],
                "independent_test_metrics_sha256": sha256_path(context["independent_metrics_path"]),
                "independent_test_predictions_sha256": sha256_path(context["independent_predictions_path"]),
                "modeling_code_sha256": sha256_path(Path(modeling_module.__file__).resolve()),
                "backbone_load_method": context["backbone_load_method"],
                "backbone_device_map": context["backbone_device_map"],
                "evaluation_cpu_offload": context["evaluation_cpu_offload"],
                "generation_config": CONFIG["generation"],
                "max_revision_passes": CONFIG["max_revision_passes"],
                "neighbors_sha256": sha256_path(neighbors_path),
            }
            if model_key == "phi4_multimodal":
                run_provenance.update({
                    "generation_api_version": actual_generation_api,
                    "generation_code_sha256": sha256_path(Path(generation_module.__file__).resolve()),
                    "phi4_generation_profile": "reference_chat_image_cache_batchencoding_empty_recovery_v6",
                    "phi4_image_compat_version": actual_phi4_image_compat,
                    "phi4_cache_compat_version": actual_phi4_cache_compat,
                    "phi4_verifier_compat_version": actual_phi4_verifier_compat,
                    "phi4_empty_output_recovery_version": actual_phi4_empty_recovery,
                    "phi4_empty_retry_min_new_tokens": (
                        generation_module.PHI4_EMPTY_RETRY_MIN_NEW_TOKENS
                    ),
                    "phi4_empty_retry_max_prompt_chars": (
                        generation_module.PHI4_EMPTY_RETRY_MAX_PROMPT_CHARS
                    ),
                    "phi4_dynamic_cache_targets": list(
                        context["runner"].phi4_dynamic_cache_targets
                    ),
                })
            run_id = hashlib.sha256(json.dumps(run_provenance, sort_keys=True).encode("utf-8")).hexdigest()
            run_provenance["run_id"] = run_id
            provenance_file = output_dir / "run_provenance.json"
            if provenance_file.exists():
                previous = json.loads(provenance_file.read_text(encoding="utf-8"))
                if previous != run_provenance:
                    has_results = output_file.exists() and any(
                        line.strip() for line in output_file.read_text(encoding="utf-8").splitlines()
                    )
                    if has_results:
                        raise RuntimeError(
                            f"Existing output provenance differs for {model_key}/{bundle_name}: {provenance_file}. "
                            "Move that bundle output aside or restore the prior code/adapter."
                        )
                    write_json(provenance_file, run_provenance)
                    print(
                        f"Refreshed provenance for zero-record output {model_key}/{bundle_name}; "
                        "no generated records were mixed."
                    )
            elif output_file.exists() and output_file.stat().st_size:
                raise RuntimeError(f"Existing results lack a run fingerprint: {output_file}")
            else:
                write_json(provenance_file, run_provenance)

            completed, resume_audit = validated_completed_generation_ids(
                output_file, model_key=model_key, bundle_name=bundle_name, run_id=run_id,
                valid_query_ids=[row["record_id"] for row in all_queries], conditions=CONDITIONS,
                repair_failed_records=True,
            )
            requested_ids = {
                f"{model_key}|{bundle_name}|{query['record_id']}|{condition}"
                for query in queries for condition in CONDITIONS
            }
            pending_ids = requested_ids - completed
            resume_audit.update({
                "n_all_bundle_queries": len(all_queries), "n_requested_queries": len(queries),
                "n_requested_records": len(requested_ids),
                "n_already_complete_requested": len(completed & requested_ids),
                "n_pending_requested": len(pending_ids),
            })
            write_json(output_dir / "resume_audit.json", resume_audit)
            print(f"\\n{model_key}/{bundle_name}")
            print(json.dumps(resume_audit, indent=2))

            skipped_existing = 0
            generated_new = 0
            for query in queries:
                query_id = query["record_id"]
                neighbors = neighbors_by_query[query_id]
                for condition in CONDITIONS:
                    generation_id = f"{model_key}|{bundle_name}|{query_id}|{condition}"
                    if generation_id in completed:
                        skipped_existing += 1
                        continue
                    result = run_condition(
                        context["runner"], condition, query, neighbors,
                        max_revision_passes=CONFIG["max_revision_passes"],
                    )
                    if result.get("empty_output") is True or not str(result.get("final_report") or "").strip():
                        raise RuntimeError(
                            f"Generation returned no usable report for {generation_id}; "
                            "the failed record was not saved. "
                            f"Recovery audit={result.get('empty_output_recovery_by_pass')}"
                        )
                    record = {
                        "generation_record_id": generation_id, "query_record_id": query_id,
                        "model_key": model_key, "model_display_name": spec["display_name"],
                        "actual_base_model": context["actual_base"], "run_id": run_id,
                        "backbone_load_method": context["backbone_load_method"],
                        "evaluation_cpu_offload": context["evaluation_cpu_offload"],
                        "corrected_adapter_sha256": run_provenance["adapter_weights_sha256"],
                        "bundle": bundle_name, "source_dataset": query.get("dataset"),
                        "condition": condition, "patient_key": query.get("patient_key") or query_id,
                        "patient_id_reliable": bool(query.get("patient_id_reliable", False)),
                        "reference_report": query.get("report_text", ""),
                        "reference_labels_json": query.get("labels_raw", []),
                        "reference_labels_13_manifest_check": query.get("labels_13", []),
                        "ground_truth_source": "paired_json.labels",
                        "image_path": query.get("image_path", ""),
                        "neighbor_record_ids": [row["record_id"] for row in neighbors],
                        "neighbor_scores": [row["cosine_score"] for row in neighbors],
                        **result,
                    }
                    with output_file.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\\n")
                        handle.flush()
                    completed.add(generation_id)
                    generated_new += 1
                    if generated_new % progress_every == 0 or generated_new == len(pending_ids):
                        print(
                            f"Progress {model_key}/{bundle_name}: generated={generated_new}/"
                            f"{len(pending_ids)}, skipped={skipped_existing}"
                        )
            remaining = requested_ids - completed
            if remaining:
                raise AssertionError(
                    f"{model_key}/{bundle_name} ended with {len(remaining)} missing records. "
                    f"Examples={sorted(remaining)[:10]}"
                )
            summary = {
                **resume_audit, "status": "complete",
                "n_skipped_existing_this_run": skipped_existing,
                "n_generated_this_run": generated_new,
                "n_complete_requested_after_run": len(requested_ids),
                "n_pending_requested_after_run": 0,
            }
            write_json(output_dir / "resume_audit.json", summary)
            print(
                f"Complete {model_key}/{bundle_name}: requested={len(requested_ids)}, "
                f"skipped={skipped_existing}, generated={generated_new}"
            )
            return summary
        """),
        code("""
        orchestration_started = datetime.now(timezone.utc).isoformat()
        combination_summaries = []
        failures = []
        for model_key in model_keys:
            context = None
            try:
                context = load_model_context(model_key)
            except Exception as exc:
                failure = {
                    "model_key": model_key, "bundle": None, "stage": "model_load",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print("MODEL LOAD FAILED:", json.dumps(failure, indent=2))
                release_accelerator_memory()
                if not continue_on_error: raise
                continue
            try:
                for bundle_name in bundle_names:
                    try:
                        combination_summaries.append(run_model_bundle(context, bundle_name))
                    except Exception as exc:
                        failure = {
                            "model_key": model_key, "bundle": bundle_name, "stage": "generation",
                            "error_type": type(exc).__name__, "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                        failures.append(failure)
                        print("COMBINATION FAILED:", json.dumps(failure, indent=2))
                        if not continue_on_error: raise
            finally:
                runner_to_release = context.pop("runner", None)
                if runner_to_release is not None:
                    runner_to_release.model = None
                    runner_to_release.processor = None
                    runner_to_release.tokenizer = None
                del runner_to_release
                context.clear()
                del context
                release_accelerator_memory()
                print("Released model memory for:", model_key)

        orchestration = {
            "started_at_utc": orchestration_started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_models": model_keys, "selected_bundles": bundle_names,
            "n_items_per_bundle": n_items, "continue_on_error": continue_on_error,
            "n_combinations_requested": len(model_keys) * len(bundle_names),
            "n_combinations_complete": len(combination_summaries),
            "n_failures": len(failures), "combinations": combination_summaries,
            "failures": failures,
        }
        orchestration_path = PATHS["generation"] / "notebook05_orchestration_summary.json"
        write_json(orchestration_path, orchestration)
        print("Orchestration summary:", orchestration_path)
        print(json.dumps({key: value for key, value in orchestration.items() if key not in {"combinations", "failures"}}, indent=2))
        if failures:
            failed_names = [f"{row['model_key']}/{row['bundle'] or '*'}" for row in failures]
            failure_details = "\\n\\n".join(
                f"[{row['model_key']}/{row['bundle'] or '*'}] "
                f"{row['stage']} — {row['error_type']}: {row['error']}\\n"
                f"{row['traceback']}"
                for row in failures
            )
            raise RuntimeError(
                f"Generation loop completed with {len(failures)} failed model/bundle entries: {failed_names}. "
                f"Successful records remain saved and will be skipped on restart. See {orchestration_path}.\\n\\n"
                f"Underlying failures:\\n{failure_details}"
            )
        """),
    ])


def notebook_06():
    save("06_chexbert_labeling_and_validation_gate.ipynb", [
        md("""
        # 06 — Corrected report-to-label extraction and validation gate

        This notebook first requires every frozen query × four conditions for all seven models and all three bundles, verifies each record against its run fingerprint, rejects empty outputs, and rereads the original paired JSON to require an explicit `labels` key. Before counting completeness, it backs up and removes only scientifically equivalent duplicate rows; duplicates that differ in reports, prompts, verifier outputs, labels, or provenance remain a hard failure. An existing but empty `labels` value is the all-zero 13-label reference vector.

        It then resolves pinned official CheXbert code and checkpoint assets. Missing configured assets are installed automatically from the Stanford GitHub and StanfordAIMI Hugging Face repositories, the checkpoint checksum is verified, and the complete asset audit is saved. It also checks the Python interpreter used by CheXbert and installs the missing official `statsmodels==0.14.1` dependency when necessary. An isolated compatibility launcher prevents the installed Hugging Face `datasets` package from shadowing CheXbert's local `datasets/unlabeled_dataset.py` and restores the legacy `BertTokenizer.encode_plus` behavior. CheXbert's pre-tokenized token list is explicitly converted to one flat sequence using `cls_token_id` and `sep_token_id`, matching historical BERT formatting without relying on removed tokenizer helper methods. The launcher does not modify the official checkout or the installed Transformers package. Set `JAMIA_CHEXBERT_REPO` and `JAMIA_CHEXBERT_CHECKPOINT` to use existing installations, `JAMIA_CHEXBERT_AUTO_SETUP=0` to disable asset setup, or `JAMIA_CHEXBERT_AUTO_INSTALL_DEPS=0` to disable dependency installation.

        The official command-line labeler preserves all 14 raw CheXbert states (including `No Finding`), applies the frozen binary mapping to the 13 evaluated observations, records script/checkpoint/input/output checksums, and executes frozen clinical sanity cases for negation, uncertainty, resolved findings, and support devices.

        Finally, it freezes a ≥200-report sample stratified by source dataset, model, condition, and reference normal/abnormal status. Two independently shuffled annotation forms expose no model, condition, source label, or generation identifier; a separate crosswalk remains with the analyst. Publication metrics remain blocked until two distinct annotators complete the forms, all disagreements are adjudicated, all 13 labels have evaluable F1, and the prespecified macro-F1 threshold is met.
        """), code(BOOTSTRAP),
        code("""
        import importlib
        import pandas as pd
        from rerun_code.common import read_jsonl, write_json, write_jsonl
        from rerun_code.config import sha256_path
        from rerun_code.generation import CONDITIONS
        import rerun_code.report_labeler as report_labeler_module
        report_labeler_module = importlib.reload(report_labeler_module)
        required_report_labeler_api = 8
        actual_report_labeler_api = int(
            getattr(report_labeler_module, "REPORT_LABELER_API_VERSION", 0)
        )
        if actual_report_labeler_api < required_report_labeler_api:
            raise ImportError(
                "Notebook 06 and rerun_code/report_labeler.py are out of sync. "
                f"Required report-labeler API {required_report_labeler_api}, found "
                f"{actual_report_labeler_api} at "
                f"{Path(report_labeler_module.__file__).resolve()}. Copy the updated "
                "rerun_code/report_labeler.py to the active Biowulf re_run folder, "
                "restart the kernel, and rerun this cell."
            )
        from rerun_code.report_labeler import audit_complete_generation_results, ensure_official_chexbert_assets, run_official_chexbert, map_chexbert_states, clinical_sanity_reports, validate_clinical_sanity_outputs, create_blinded_annotation_materials, create_adjudication_template, validate_two_annotators

        generation, generation_audit = audit_complete_generation_results(
            generation_root=PATHS["generation"],
            bundles_root=PATHS["bundles"],
            model_keys=list(CONFIG["models"]),
            bundle_names=("mimic", "iuhn", "combined"),
            conditions=CONDITIONS,
            repair_equivalent_duplicates=True,
        )
        write_json(PATHS["labels"] / "generation_completeness_audit.json", generation_audit)
        print("COMPLETE GENERATION COHORT VERIFIED:", generation_audit["n_records"], "records")
        print(
            "Equivalent duplicate rows removed:",
            generation_audit["n_equivalent_duplicate_rows_removed"],
        )
        if generation_audit["duplicate_repair_backups"]:
            print(
                "Original JSONL backup files:",
                *generation_audit["duplicate_repair_backups"],
                sep="\\n- ",
            )

        sanity = clinical_sanity_reports()
        label_input = pd.concat([
            generation[["generation_record_id", "final_report"]], sanity
        ], ignore_index=True)
        chexbert_assets = ensure_official_chexbert_assets(
            repo=CONFIG["labeler"]["chexbert_repo"],
            checkpoint=CONFIG["labeler"]["checkpoint"],
        )
        write_json(PATHS["labels"] / "chexbert_asset_audit.json", chexbert_assets)
        print("CheXbert code:", chexbert_assets["repository"])
        print("CheXbert checkpoint:", chexbert_assets["checkpoint"])
        labeler_dir = PATHS["labeler"] / "chexbert_full"
        raw = run_official_chexbert(label_input, repo=chexbert_assets["repository"], checkpoint=chexbert_assets["checkpoint"], output_dir=labeler_dir, text_column="final_report")
        vectors, states = map_chexbert_states(raw, CONFIG["labeler"]["uncertain_policy"])
        n_generation = len(generation)
        generation["prediction_vector"] = vectors[:n_generation]
        generation["chexbert_states_14"] = states[:n_generation]
        generation["unknown_json_labels"] = [[] for _ in range(n_generation)]
        sanity_result = validate_clinical_sanity_outputs(
            states[n_generation:], vectors[n_generation:]
        )
        write_json(PATHS["labeler"] / "clinical_sanity_results.json", sanity_result)
        write_jsonl(PATHS["labels"] / "labeled_generation.jsonl", generation.to_dict("records"))
        write_json(PATHS["labels"] / "label_extraction_audit.json", {
            "generation_completeness_audit": str(PATHS["labels"] / "generation_completeness_audit.json"),
            "chexbert_provenance": str(labeler_dir / "chexbert_run_provenance.json"),
            "clinical_sanity_results": str(PATHS["labeler"] / "clinical_sanity_results.json"),
            "chexbert_asset_audit": str(PATHS["labels"] / "chexbert_asset_audit.json"),
            "uncertain_policy": CONFIG["labeler"]["uncertain_policy"],
            "evaluated_labels": list(CONFIG["labeler"]["uncertain_policy"]),
            "raw_chexbert_no_finding_preserved": True,
            "ground_truth_source": "paired_json.labels",
            "report_labeler_code": str(Path(report_labeler_module.__file__).resolve()),
            "report_labeler_code_sha256": sha256_path(Path(report_labeler_module.__file__).resolve()),
        })
        """),
        code("""
        validation_dir = PATHS["labeler"] / "validation"; validation_dir.mkdir(parents=True, exist_ok=True)
        minimum = int(CONFIG["labeler"]["minimum_manual_validation_reports"])
        annotation_manifest = create_blinded_annotation_materials(
            generation, validation_dir, n=minimum, seed=CONFIG["statistics"]["seed"]
        )
        annotator1 = Path(os.environ.get("JAMIA_ANNOTATOR1_CSV", validation_dir / "human_annotation_annotator1_completed.csv"))
        annotator2 = Path(os.environ.get("JAMIA_ANNOTATOR2_CSV", validation_dir / "human_annotation_annotator2_completed.csv"))
        adjudication_template = validation_dir / "human_adjudication_template.csv"
        adjudicated = Path(os.environ.get("JAMIA_ADJUDICATED_CSV", validation_dir / "human_adjudication_completed.csv"))
        gate = {
            "passed": False,
            "required_n": minimum,
            "required_macro_f1": CONFIG["labeler"]["minimum_macro_f1"],
            "required_annotators": CONFIG["labeler"]["required_annotators"],
            "required_f1_evaluable_labels": CONFIG["labeler"]["required_f1_evaluable_labels"],
            "annotation_manifest": annotation_manifest,
            "annotator1_completed": str(annotator1),
            "annotator2_completed": str(annotator2),
            "adjudication_completed": str(adjudicated),
        }
        if annotator1.exists() and annotator2.exists():
            try:
                adjudication_status = create_adjudication_template(
                    annotator1, annotator2, adjudication_template
                )
                gate["adjudication"] = adjudication_status
                if adjudication_status["n_disagreement_cells"] == 0 or adjudicated.exists():
                    result = validate_two_annotators(
                        generation[["generation_record_id", "prediction_vector"]],
                        annotator1_csv=annotator1,
                        annotator2_csv=annotator2,
                        crosswalk_csv=annotation_manifest["crosswalk"],
                        adjudicated_csv=adjudicated if adjudicated.exists() else None,
                    )
                    gate.update(result)
                    gate["passed"] = (
                        result["n"] >= minimum
                        and result["n_f1_evaluable_labels"] == CONFIG["labeler"]["required_f1_evaluable_labels"]
                        and result["macro_f1"] is not None
                        and result["macro_f1"] >= CONFIG["labeler"]["minimum_macro_f1"]
                    )
                    if not gate["passed"]:
                        gate["pending"] = "The completed forms do not meet the prespecified validation thresholds."
                else:
                    gate["pending"] = f"Complete adjudication template: {adjudication_template}"
            except Exception as exc:
                # Always replace a stale gate file with an explicit failed gate.
                # This makes crosswalk/hash/column problems visible to Notebook 07.
                gate["pending"] = "Annotation validation failed; correct the completed forms and rerun this cell."
                gate["validation_error"] = f"{type(exc).__name__}: {exc}"
        else:
            gate["pending"] = "Two independently completed blinded annotation forms are required."
        instructions_path = validation_dir / "HUMAN_ANNOTATION_NEXT_STEPS.txt"
        instructions_path.write_text(
            "Notebook 06 human-validation checkpoint\\n\\n"
            f"Annotator 1 template: {annotation_manifest['annotator1_template']}\\n"
            f"Annotator 2 template: {annotation_manifest['annotator2_template']}\\n\\n"
            f"Save completed annotator 1 form as: {annotator1}\\n"
            f"Save completed annotator 2 form as: {annotator2}\\n\\n"
            "Each independent annotator must fill every human_* field with 0 or 1 and "
            "use one consistent, nonempty annotator_id. The two IDs must differ. Do not "
            "give either annotator the analyst-only crosswalk.\\n\\n"
            "After both forms are complete, rerun this cell. If disagreements exist, "
            f"complete {adjudication_template} with a third annotator and save the result "
            f"as {adjudicated}.\\n",
            encoding="utf-8",
        )
        gate["instructions_file"] = str(instructions_path)
        write_json(validation_dir / "validation_gate.json", gate)
        print(json.dumps(gate, indent=2))
        if gate["passed"]:
            print("LABELER VALIDATION GATE PASSED. Notebook 07 may now be run.")
        else:
            print("\\nHUMAN ANNOTATION PENDING — THIS IS A PLANNED CHECKPOINT, NOT A CODE ERROR.")
            print("Annotator 1 template:", annotation_manifest["annotator1_template"])
            print("Annotator 2 template:", annotation_manifest["annotator2_template"])
            print("Instructions:", instructions_path)
            print("Notebook 07 remains blocked until validation_gate.json has passed=true.")
        """),
    ])


def notebook_07():
    save("07_compute_per_study_and_aggregate_metrics.ipynb", [
        md("""
        # 07 — Per-study and aggregate clinical/text metrics

        This notebook computes nothing while the human labeler-validation gate is pending. A pending gate ends at a clear planned checkpoint rather than raising a Python exception. A passed gate must satisfy every prespecified threshold and match the exact frozen labeled-generation cohort before analysis begins.

        After those checks, the notebook computes FER, abnormal-case FER, omission, label F1/accuracy, RadGraph F1, CIDEr, BERTScore F1, ROUGE-L, and measured runtime from the corrected records. Missing metric packages produce explicit missing values, never zeros.
        """), code(BOOTSTRAP),
        code("""
        import pandas as pd, numpy as np
        from rerun_code.common import read_jsonl, write_json, write_jsonl
        from rerun_code.config import sha256_path
        from rerun_code.metrics import compute_text_metrics, aggregate_label_metrics
        from rerun_code.report_labeler import LABELS_13, _cohort_fingerprint

        gate_path = PATHS["labeler"] / "validation" / "validation_gate.json"
        instructions_path = PATHS["labeler"] / "validation" / "HUMAN_ANNOTATION_NEXT_STEPS.txt"
        status_path = PATHS["metrics"] / "notebook07_status.json"
        analysis_ready = False
        gate = {}
        if not gate_path.exists():
            pending_reason = "Notebook 06 has not created validation_gate.json."
        else:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            pending_reason = str(gate.get("pending") or "Human validation has not passed.")
            analysis_ready = bool(gate.get("passed"))

        # A stale gate file can retain passed=true while omitting the metrics
        # written by Notebook 06. Treat that state as pending validation rather
        # than allowing Notebook 07 to fail with an opaque KeyError or assertion.
        if analysis_ready:
            required_gate_fields = (
                "required_n", "required_macro_f1", "required_f1_evaluable_labels",
                "n", "n_f1_evaluable_labels", "macro_f1",
            )
            missing_gate_fields = [
                field for field in required_gate_fields
                if field not in gate or gate.get(field) is None
            ]
            if missing_gate_fields:
                analysis_ready = False
                pending_reason = (
                    "validation_gate.json is marked passed=true but is incomplete "
                    f"(missing values: {missing_gate_fields}). "
                    "Rerun Notebook 06 cell 3; do not edit this file manually."
                )
            else:
                try:
                    required_n = int(gate["required_n"])
                    required_macro_f1 = float(gate["required_macro_f1"])
                    required_labels = int(gate["required_f1_evaluable_labels"])
                    gate_errors = []
                    if int(gate["n"]) < required_n:
                        gate_errors.append(f"validated n={gate.get('n')} is below {required_n}")
                    if int(gate["n_f1_evaluable_labels"]) != required_labels:
                        gate_errors.append(
                            f"evaluable labels={gate.get('n_f1_evaluable_labels')} rather than {required_labels}"
                        )
                    if float(gate["macro_f1"]) < required_macro_f1:
                        gate_errors.append(
                            f"macro F1={gate.get('macro_f1')} is below {required_macro_f1}"
                        )
                except (TypeError, ValueError, KeyError) as exc:
                    gate_errors = [f"validation metrics are not numeric ({exc})"]
                if gate_errors:
                    analysis_ready = False
                    pending_reason = (
                        "validation_gate.json is marked passed=true but does not satisfy the "
                        "prespecified gate: " + "; ".join(gate_errors) + ". "
                        "Rerun Notebook 06 cell 3 after correcting the annotation files."
                    )

        write_json(status_path, {
            "ready": False,
            "gate_path": str(gate_path),
            "gate_passed": bool(gate.get("passed")),
            "pending_reason": None if analysis_ready else pending_reason,
        })

        if not analysis_ready:
            print("HUMAN VALIDATION PENDING — NOTEBOOK 07 DID NOT COMPUTE METRICS.")
            print("Reason:", pending_reason)
            if instructions_path.exists():
                print("Instructions:", instructions_path)
            print("Next action: rerun Notebook 06 cell 3 to regenerate validation_gate.json, then rerun Notebook 07.")
        else:
            labeled_path = PATHS["labels"] / "labeled_generation.jsonl"
            if not labeled_path.exists():
                raise FileNotFoundError(f"Notebook 06 labeled cohort is missing: {labeled_path}")
            frame = pd.DataFrame(read_jsonl(labeled_path))
            required_columns = {
                "generation_record_id", "final_report", "reference_report", "reference_vector",
                "prediction_vector", "model_key", "bundle", "source_dataset", "condition",
                "total_generation_seconds", "total_verifier_seconds",
            }
            missing_columns = sorted(required_columns - set(frame.columns))
            if missing_columns:
                raise KeyError(f"Labeled generation cohort is missing columns: {missing_columns}")
            if frame.empty or frame["generation_record_id"].astype(str).duplicated().any():
                raise AssertionError("Labeled generation cohort is empty or has duplicate identifiers")
            if frame["final_report"].fillna("").astype(str).str.strip().eq("").any():
                raise AssertionError("Labeled generation cohort contains an empty report")
            for vector_column in ("reference_vector", "prediction_vector"):
                invalid_vectors = frame[vector_column].map(
                    lambda value: not isinstance(value, list)
                    or len(value) != len(LABELS_13)
                    or any(item not in (0, 1) for item in value)
                )
                if invalid_vectors.any():
                    raise AssertionError(
                        f"{vector_column} contains {int(invalid_vectors.sum())} invalid vectors"
                    )

            frozen_fingerprint = gate.get("annotation_manifest", {}).get(
                "generation_cohort_fingerprint"
            )
            actual_fingerprint = _cohort_fingerprint(frame)
            if not frozen_fingerprint or actual_fingerprint != frozen_fingerprint:
                raise AssertionError(
                    "The labeled generation cohort does not match the cohort frozen for human validation"
                )

            frame["runtime_seconds"] = (
                frame["total_generation_seconds"].astype(float)
                + frame["total_verifier_seconds"].astype(float)
            )
            frame = compute_text_metrics(frame)
            per_study_path = PATHS["metrics"] / "per_study_metrics.jsonl"
            aggregate_path = PATHS["metrics"] / "aggregate_metrics.csv"
            text_metric_audit_path = PATHS["metrics"] / "text_metric_runtime_audit.json"
            write_json(text_metric_audit_path, {
                "radgraph_legacy_encode_plus_compatibility_attempted": True,
                "radgraph_f1_reward_component": "partial_entity_relation_RG_ER",
                "errors": dict(frame.attrs.get("text_metric_errors", {})),
            })
            write_jsonl(per_study_path, frame.to_dict("records"))
            rows = []
            grouping = ["model_key", "bundle", "source_dataset", "condition"]
            for keys, group in frame.groupby(grouping, dropna=False):
                result = aggregate_label_metrics(group)
                rows.append(dict(zip(grouping, keys), **result))
            aggregate = pd.DataFrame(rows)
            aggregate.to_csv(aggregate_path, index=False)
            write_json(status_path, {
                "ready": True,
                "gate_path": str(gate_path),
                "gate_passed": True,
                "generation_cohort_fingerprint": actual_fingerprint,
                "n_records": int(len(frame)),
                "per_study_metrics": str(per_study_path),
                "per_study_metrics_sha256": sha256_path(per_study_path),
                "aggregate_metrics": str(aggregate_path),
                "aggregate_metrics_sha256": sha256_path(aggregate_path),
                "text_metric_runtime_audit": str(text_metric_audit_path),
            })
            print("NOTEBOOK 07 COMPLETE — METRICS ARE LINKED TO THE PASSED VALIDATION COHORT.")
            display(aggregate.sort_values(["bundle", "model_key", "condition"]))
        """),
    ])


def notebook_08():
    save("08_cluster_bootstrap_and_paired_tests.ipynb", [
        md("""
        # 08 — Patient-cluster bootstrap intervals and paired tests

        This notebook computes nothing until Notebook 07 reports `ready: true`. It verifies the upstream file checksum, record count, frozen cohort fingerprint, four-condition coverage, and identical paired query sets before statistical analysis.

        Label-derived outcomes are recomputed in every patient/group bootstrap and permutation replicate from cluster-level event counts. The vectorized engine implements the same patient/source-cluster resampling and swapping protocol without reconstructing a DataFrame for each replicate. Reference vectors and patient/source cluster assignments must agree across paired arms. Raw p-values are adjusted with Holm correction within each prespecified model-specific strategy-comparison family. Between-model comparisons use a separate family within each metric, bundle, source dataset, and condition.
        """), code(BOOTSTRAP),
        code("""
        import importlib, itertools, pandas as pd
        from rerun_code.common import read_jsonl, write_json
        from rerun_code.config import sha256_path
        from rerun_code.generation import CONDITIONS
        from rerun_code.report_labeler import _cohort_fingerprint
        import rerun_code.statistics as statistics_module
        statistics_module = importlib.reload(statistics_module)
        required_statistics_api = 3
        actual_statistics_api = int(getattr(statistics_module, "STATISTICS_API_VERSION", 0))
        if actual_statistics_api < required_statistics_api:
            raise ImportError(
                "Notebook 08 and rerun_code/statistics.py are out of sync. "
                f"Required statistics API {required_statistics_api}, found {actual_statistics_api} at "
                f"{Path(statistics_module.__file__).resolve()}. Copy the updated statistics.py, "
                "restart the kernel, and rerun this cell."
            )
        from rerun_code.statistics import cluster_bootstrap, paired_cluster_permutation, holm_adjust

        upstream_status_path = PATHS["metrics"] / "notebook07_status.json"
        status_path = PATHS["statistics"] / "notebook08_status.json"
        statistics_ready = False
        pending_reason = None
        upstream = {}
        if not upstream_status_path.exists():
            pending_reason = "Notebook 07 has not written notebook07_status.json."
        else:
            upstream = json.loads(upstream_status_path.read_text(encoding="utf-8"))
            if not upstream.get("ready"):
                pending_reason = str(
                    upstream.get("pending_reason") or "Notebook 07 metrics are not ready."
                )

        write_json(status_path, {
            "ready": False,
            "upstream_status": str(upstream_status_path),
            "pending_reason": pending_reason or "Statistical computation is in progress.",
        })

        if pending_reason:
            print("NOTEBOOK 07 OUTPUTS ARE PENDING — NOTEBOOK 08 DID NOT COMPUTE STATISTICS.")
            print("Reason:", pending_reason)
            print("Complete human validation in Notebook 06, then run Notebook 07.")
        else:
            per_study_path = Path(upstream.get("per_study_metrics", ""))
            expected_sha256 = str(upstream.get("per_study_metrics_sha256", ""))
            if not per_study_path.exists():
                raise FileNotFoundError(f"Notebook 07 per-study metrics are missing: {per_study_path}")
            actual_sha256 = sha256_path(per_study_path)
            if not expected_sha256 or actual_sha256 != expected_sha256:
                raise AssertionError(
                    "Notebook 07 per-study metrics checksum differs from notebook07_status.json"
                )
            frame = pd.DataFrame(read_jsonl(per_study_path))
            if frame.empty or len(frame) != int(upstream.get("n_records", -1)):
                raise AssertionError("Notebook 07 per-study record count is missing or inconsistent")
            if frame["generation_record_id"].astype(str).duplicated().any():
                raise AssertionError("Notebook 07 per-study metrics contain duplicate identifiers")
            actual_fingerprint = _cohort_fingerprint(frame)
            if actual_fingerprint != upstream.get("generation_cohort_fingerprint"):
                raise AssertionError(
                    "Notebook 07 per-study metrics do not match its frozen generation cohort"
                )
            # Label-derived outcomes are deliberately recomputed from the two vectors
            # inside every cluster bootstrap and paired permutation replicate. They are
            # aggregate outcomes, not columns in Notebook 07's per-study JSONL.
            vector_derived_metrics = {
                "fer", "fer_abnormal", "omission", "macro_f1", "micro_f1",
                "hamming_accuracy", "normal_abnormal_accuracy",
                "false_positive_events", "predicted_positive_events",
                "false_negative_events", "reference_positive_events",
                *[f"f1_{label.replace(' ', '_')}" for label in CONFIG["labeler"]["uncertain_policy"]],
            }
            direct_primary_metrics = set(CONFIG["statistics"]["primary_metrics"]) - vector_derived_metrics
            required_columns = {
                "query_record_id", "patient_key", "model_key", "bundle", "source_dataset",
                "condition", "reference_vector", "prediction_vector", *direct_primary_metrics,
            }
            missing_columns = sorted(required_columns - set(frame.columns))
            if missing_columns:
                raise KeyError(f"Per-study metrics are missing statistical inputs: {missing_columns}")
            unavailable_direct_metrics = [
                metric for metric in sorted(direct_primary_metrics)
                if not frame[metric].notna().any()
            ]
            if unavailable_direct_metrics:
                raise RuntimeError(
                    "Notebook 07 did not produce usable values for direct primary metrics: "
                    f"{unavailable_direct_metrics}. Inspect text_metric_runtime_audit.json before inference."
                )
            expected_conditions = set(CONDITIONS)
            pairing_errors = []
            for keys, group in frame.groupby(
                ["model_key", "bundle", "source_dataset"], dropna=False
            ):
                observed_conditions = set(group["condition"].astype(str))
                if observed_conditions != expected_conditions:
                    pairing_errors.append(f"{keys}: conditions={sorted(observed_conditions)}")
                    continue
                query_sets = [
                    set(group.loc[group["condition"] == condition, "query_record_id"].astype(str))
                    for condition in CONDITIONS
                ]
                if any(values != query_sets[0] for values in query_sets[1:]):
                    pairing_errors.append(f"{keys}: paired query sets differ across conditions")
            if pairing_errors:
                raise AssertionError(
                    "Notebook 08 pairing audit failed: " + "; ".join(pairing_errors[:20])
                )
            statistics_ready = True
            print("NOTEBOOK 08 INPUT AUDIT PASSED:", len(frame), "paired per-study records")
        """),
        code("""
        if not statistics_ready:
            print("Bootstrap intervals skipped because Notebook 07 outputs are pending.")
        else:
            reps_b = int(CONFIG["statistics"]["bootstrap_replicates"])
            reps_p = int(CONFIG["statistics"]["permutation_replicates"])
            ci_parts = []
            grouping = ["model_key", "bundle", "source_dataset", "condition"]
            grouped = list(frame.groupby(grouping, dropna=False))
            print(f"Vectorized bootstrap: {len(grouped)} groups × {reps_b:,} patient/source-cluster replicates")
            for index, (keys, group) in enumerate(grouped, start=1):
                ci = cluster_bootstrap(
                    group,
                    replicates=reps_b,
                    confidence_level=CONFIG["statistics"]["confidence_level"],
                    seed=CONFIG["statistics"]["seed"],
                )
                for name, value in zip(grouping, keys):
                    ci[name] = value
                ci_parts.append(ci)
                if index == 1 or index % 10 == 0 or index == len(grouped):
                    print(f"  bootstrap group {index}/{len(grouped)} complete")
            intervals = pd.concat(ci_parts, ignore_index=True)
            intervals_path = PATHS["statistics"] / "cluster_bootstrap_ci.csv"
            intervals.to_csv(intervals_path, index=False)
            print("Saved patient/source-cluster bootstrap intervals:", intervals_path)
            display(intervals.head())
        """),
        code("""
        if not statistics_ready:
            print("Paired tests skipped because Notebook 07 outputs are pending.")
        else:
            test_parts = []
            comparisons = [
                ("A_single_pass", "B_unconditional_4pass"),
                ("A_single_pass", "C_pretrained_gate"),
                ("A_single_pass", "D_corrected_lora_gate"),
                ("C_pretrained_gate", "D_corrected_lora_gate"),
            ]
            grouping = ["model_key", "bundle", "source_dataset"]
            grouped = list(frame.groupby(grouping, dropna=False))
            total_within = len(grouped) * len(comparisons)
            print(f"Vectorized within-model tests: {total_within} comparisons × {reps_p:,} paired replicates")
            within_done = 0
            for keys, group in grouped:
                for arm_a, arm_b in comparisons:
                    a = group[group["condition"] == arm_a]
                    b = group[group["condition"] == arm_b]
                    comparison = paired_cluster_permutation(
                        a,
                        b,
                        id_column="query_record_id",
                        metrics=CONFIG["statistics"]["primary_metrics"],
                        replicates=reps_p,
                        seed=CONFIG["statistics"]["seed"],
                    )
                    comparison["arm_a_condition"] = arm_a
                    comparison["arm_b_condition"] = arm_b
                    for name, value in zip(grouping, keys):
                        comparison[name] = value
                    test_parts.append(comparison)
                    within_done += 1
                    if within_done == 1 or within_done % 25 == 0 or within_done == total_within:
                        print(f"  within-model comparison {within_done}/{total_within} complete")
            tests = pd.concat(test_parts, ignore_index=True)
            within_family = ["model_key", "metric", "bundle", "source_dataset"]
            tests["p_holm_within_strategy_family"] = tests.groupby(
                within_family, dropna=False
            )["p_value"].transform(lambda values: holm_adjust(values.tolist()))
            tests_path = PATHS["statistics"] / "paired_cluster_permutation_tests.csv"
            tests.to_csv(tests_path, index=False)

            between_parts = []
            grouping = ["bundle", "source_dataset", "condition"]
            grouped = list(frame.groupby(grouping, dropna=False))
            total_between = sum(len(list(itertools.combinations(sorted(group["model_key"].unique()), 2))) for _, group in grouped)
            print(f"Vectorized between-model tests: {total_between} comparisons × {reps_p:,} paired replicates")
            between_done = 0
            for keys, group in grouped:
                models = sorted(group["model_key"].unique())
                for model_a, model_b in itertools.combinations(models, 2):
                    a = group[group["model_key"] == model_a]
                    b = group[group["model_key"] == model_b]
                    comparison = paired_cluster_permutation(
                        a,
                        b,
                        id_column="query_record_id",
                        metrics=CONFIG["statistics"]["primary_metrics"],
                        replicates=reps_p,
                        seed=CONFIG["statistics"]["seed"],
                    )
                    comparison["model_a"] = model_a
                    comparison["model_b"] = model_b
                    for name, value in zip(grouping, keys):
                        comparison[name] = value
                    between_parts.append(comparison)
                    between_done += 1
                    if between_done == 1 or between_done % 25 == 0 or between_done == total_between:
                        print(f"  between-model comparison {between_done}/{total_between} complete")
            between = pd.concat(between_parts, ignore_index=True)
            between_family = ["metric", "bundle", "source_dataset", "condition"]
            between["p_holm_between_model_family"] = between.groupby(
                between_family, dropna=False
            )["p_value"].transform(lambda values: holm_adjust(values.tolist()))
            between_path = PATHS["statistics"] / "between_model_paired_tests.csv"
            between.to_csv(between_path, index=False)

            write_json(status_path, {
                "ready": True,
                "upstream_status": str(upstream_status_path),
                "upstream_per_study_metrics_sha256": actual_sha256,
                "generation_cohort_fingerprint": actual_fingerprint,
                "n_records": int(len(frame)),
                "bootstrap_replicates": reps_b,
                "permutation_replicates": reps_p,
                "statistics_engine": "vectorized_cluster_sufficient_statistics_v3",
                "confidence_level": CONFIG["statistics"]["confidence_level"],
                "cluster_bootstrap_ci": str(intervals_path),
                "cluster_bootstrap_ci_sha256": sha256_path(intervals_path),
                "paired_cluster_permutation_tests": str(tests_path),
                "paired_cluster_permutation_tests_sha256": sha256_path(tests_path),
                "between_model_paired_tests": str(between_path),
                "between_model_paired_tests_sha256": sha256_path(between_path),
                "within_model_holm_family": within_family,
                "between_model_holm_family": between_family,
            })
            print("NOTEBOOK 08 COMPLETE — CLUSTERED INFERENCE IS LINKED TO NOTEBOOK 07.")
            display(tests.head())
            display(between.head())
        """),
    ])


def notebook_09():
    save("09_error_transitions_ensemble_runtime_exports.ipynb", [
        md("""
        # 09 — Rescue/harm transitions, prespecified ensemble, and manuscript exports

        Case-level rescue/harm is measured against single-pass RAG. The exploratory ensemble uses the prespecified MedGemma-4B, Qwen2-1.5B, and Med-Qwen2-7B per-label majority vote. It has label metrics only—no synthetic report, text score, or RadGraph score.
        """), code(BOOTSTRAP),
        code("""
        import importlib, json
        import numpy as np, pandas as pd
        from rerun_code.common import read_jsonl, write_jsonl
        from rerun_code.config import sha256_path
        from rerun_code.metrics import majority_vote, aggregate_label_metrics
        import rerun_code.statistics as statistics_module
        statistics_module = importlib.reload(statistics_module)
        if int(getattr(statistics_module, "STATISTICS_API_VERSION", 0)) < 3:
            raise ImportError(
                "Notebook 09 requires the vectorized statistics engine (API 3). "
                "Copy rerun_code/statistics.py, restart the kernel, and rerun this cell."
            )
        from rerun_code.statistics import paired_cluster_permutation, holm_adjust

        # Do not export any downstream analysis from stale or incomplete
        # upstream results. Notebook 08 must have analyzed the exact cohort
        # produced by Notebook 07, including the direct F1-RadGraph outcome.
        notebook07_status_path = PATHS["metrics"] / "notebook07_status.json"
        notebook08_status_path = PATHS["statistics"] / "notebook08_status.json"
        if not notebook07_status_path.exists() or not notebook08_status_path.exists():
            raise FileNotFoundError(
                "Run Notebooks 07 and 08 before Notebook 09; their status files are required."
            )
        notebook07 = json.loads(notebook07_status_path.read_text(encoding="utf-8"))
        notebook08 = json.loads(notebook08_status_path.read_text(encoding="utf-8"))
        if not notebook07.get("ready") or not notebook08.get("ready"):
            raise RuntimeError(
                "Notebook 09 requires completed Notebooks 07 and 08. "
                f"Notebook 07 ready={notebook07.get('ready')}; "
                f"Notebook 08 ready={notebook08.get('ready')}."
            )
        per_study_path = Path(notebook07.get("per_study_metrics", ""))
        expected_sha256 = str(notebook07.get("per_study_metrics_sha256", ""))
        if not per_study_path.exists() or not expected_sha256:
            raise FileNotFoundError("Notebook 07 per-study metrics or its checksum is missing.")
        actual_sha256 = sha256_path(per_study_path)
        if actual_sha256 != expected_sha256:
            raise AssertionError("Notebook 07 per-study metrics differ from notebook07_status.json.")
        if notebook08.get("upstream_per_study_metrics_sha256") != actual_sha256:
            raise AssertionError(
                "Notebook 08 statistics are stale relative to Notebook 07 metrics. "
                "Rerun Notebook 08 before Notebook 09."
            )
        text_audit_path = Path(notebook07.get("text_metric_runtime_audit", ""))
        if not text_audit_path.exists():
            raise FileNotFoundError("Notebook 07 text-metric audit is missing.")
        text_audit = json.loads(text_audit_path.read_text(encoding="utf-8"))
        if text_audit.get("errors", {}).get("radgraph_f1"):
            raise RuntimeError(
                "Notebook 07 RadGraph failed; resolve the recorded error before final exports: "
                + str(text_audit["errors"]["radgraph_f1"])
            )
        if text_audit.get("radgraph_f1_reward_component") != "partial_entity_relation_RG_ER":
            raise AssertionError(
                "Notebook 07 did not record the prespecified partial entity/relation "
                "F1-RadGraph component (RG_ER)."
            )

        frame = pd.DataFrame(read_jsonl(per_study_path))
        if "radgraph_f1" not in frame or not frame["radgraph_f1"].notna().any():
            raise RuntimeError("Notebook 07 per-study output has no usable radgraph_f1 values.")
        frame["label_error_count"] = [int((np.asarray(r) != np.asarray(p)).sum()) for r, p in zip(frame["reference_vector"], frame["prediction_vector"])]
        baseline = frame[frame["condition"] == "A_single_pass"][["model_key", "bundle", "source_dataset", "query_record_id", "label_error_count"]].rename(columns={"label_error_count": "baseline_error_count"})
        transitions = frame.merge(baseline, on=["model_key", "bundle", "source_dataset", "query_record_id"], validate="many_to_one")
        transitions["transition"] = np.where(transitions["label_error_count"] < transitions["baseline_error_count"], "rescue", np.where(transitions["label_error_count"] > transitions["baseline_error_count"], "harm", "unchanged"))
        transitions.groupby(["model_key", "bundle", "source_dataset", "condition", "transition"]).size().rename("n").reset_index().to_csv(PATHS["analysis"] / "error_transitions.csv", index=False)
        """),
        code("""
        ensemble_models = CONFIG["ensemble_models"]
        ensemble_records = []
        source = frame[frame["model_key"].isin(ensemble_models)]
        for keys, group in source.groupby(["bundle", "source_dataset", "condition", "query_record_id"]):
            if set(group["model_key"]) != set(ensemble_models): continue
            first = group.iloc[0]
            ensemble_records.append({"bundle": keys[0], "source_dataset": keys[1], "condition": keys[2], "query_record_id": keys[3], "patient_key": first["patient_key"], "reference_vector": first["reference_vector"], "prediction_vector": majority_vote(group.set_index("model_key").loc[ensemble_models, "prediction_vector"].tolist())})
        ensemble = pd.DataFrame(ensemble_records)
        write_jsonl(PATHS["analysis"] / "ensemble_per_study.jsonl", ensemble.to_dict("records"))
        ensemble_rows = []
        for keys, group in ensemble.groupby(["bundle", "source_dataset", "condition"]):
            ensemble_rows.append({"bundle": keys[0], "source_dataset": keys[1], "condition": keys[2], **aggregate_label_metrics(group)})
        ensemble_aggregate = pd.DataFrame(ensemble_rows)
        ensemble_aggregate.to_csv(PATHS["analysis"] / "ensemble_label_metrics.csv", index=False)
        ensemble_tests = []
        label_primary = [metric for metric in CONFIG["statistics"]["primary_metrics"] if metric != "radgraph_f1"]
        for keys, ensemble_group in ensemble.groupby(["bundle", "source_dataset", "condition"]):
            candidate = frame[(frame["bundle"] == keys[0]) & (frame["source_dataset"] == keys[1]) & (frame["condition"] == keys[2]) & frame["model_key"].isin(ensemble_models)]
            point_estimates = {model: aggregate_label_metrics(group)["macro_f1"] for model, group in candidate.groupby("model_key")}
            best_single = max(point_estimates, key=point_estimates.get)
            for model_key, single_group in candidate.groupby("model_key"):
                comparison = paired_cluster_permutation(single_group, ensemble_group, id_column="query_record_id", metrics=label_primary, replicates=int(CONFIG["statistics"]["permutation_replicates"]), seed=CONFIG["statistics"]["seed"])
                comparison["single_model"] = model_key; comparison["is_best_single_by_macro_f1"] = model_key == best_single
                comparison["bundle"], comparison["source_dataset"], comparison["condition"] = keys
                ensemble_tests.append(comparison)
        ensemble_tests = pd.concat(ensemble_tests, ignore_index=True)
        ensemble_tests["p_holm_ensemble_family"] = ensemble_tests.groupby(["metric", "bundle", "source_dataset", "condition"], dropna=False)["p_value"].transform(lambda values: holm_adjust(values.tolist()))
        ensemble_tests.to_csv(PATHS["analysis"] / "ensemble_paired_tests.csv", index=False)
        """),
        code("""
        runtime = frame.groupby(["model_key", "bundle", "source_dataset", "condition"])["runtime_seconds"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
        runtime.to_csv(PATHS["analysis"] / "runtime_summary.csv", index=False)
        required = [PATHS["metrics"] / "aggregate_metrics.csv", PATHS["statistics"] / "cluster_bootstrap_ci.csv", PATHS["statistics"] / "paired_cluster_permutation_tests.csv", PATHS["statistics"] / "between_model_paired_tests.csv", PATHS["analysis"] / "error_transitions.csv", PATHS["analysis"] / "ensemble_label_metrics.csv", PATHS["analysis"] / "ensemble_paired_tests.csv", PATHS["analysis"] / "runtime_summary.csv"]
        missing = [str(path) for path in required if not path.exists()]
        if missing: raise AssertionError("Missing final exports:\\n" + "\\n".join(missing))
        print("RERUN ANALYSIS COMPLETE. Manuscript-ready source tables are in", PATHS["metrics"], PATHS["statistics"], "and", PATHS["analysis"])
        """),
    ])


for builder in (notebook_00, notebook_01, notebook_02, notebook_03, notebook_04, notebook_05, notebook_06, notebook_07, notebook_08, notebook_09):
    builder()

print("Generated 10 notebooks in", ROOT)
