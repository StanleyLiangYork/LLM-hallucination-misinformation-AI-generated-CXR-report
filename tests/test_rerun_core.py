from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rerun_code.metrics import (
    _install_radgraph_legacy_tokenizer_compatibility,
    aggregate_label_metrics,
    majority_vote,
)
from rerun_code.common import add_leakage_code_to_path
from rerun_code.generation import (
    ModelRunner,
    _chat_template_model_inputs,
    _generate_report_with_empty_recovery,
    _install_phi4_dynamic_cache_compatibility,
    _install_phi4_legacy_numpy_patch_helpers,
    _install_phi4_numpy_normalize_compatibility,
    validated_completed_generation_ids,
)
from rerun_code.modeling import (
    _audit_model_device_map,
    _restore_expected_tied_lm_head,
    _install_phi4_siglip2_processor_compatibility,
    adapter_config,
    assert_model_identity,
    completed_corrected_adapter,
    IncompleteCorrectedAdapterError,
    normalize_adapter_base_model,
    parse_verdict,
    quarantine_incomplete_corrected_adapter,
    render_training_prompt,
    resolve_training_base_model,
)
from rerun_code.preencoded import FAISS_PRIORITY, VECTOR_PRIORITY, _select_named, align_manifest_to_preencoded
from rerun_code.report_labeler import (
    LABELS_13,
    audit_complete_generation_results,
    ensure_chexbert_runtime_dependencies,
    ensure_official_chexbert_assets,
    create_adjudication_template,
    create_blinded_annotation_materials,
    ground_truth_vector_from_json_labels,
    map_chexbert_states,
    prepare_chexbert_import_compatibility,
    validate_clinical_sanity_outputs,
    validate_against_human,
    validate_two_annotators,
)
from rerun_code.statistics import STATISTICS_API_VERSION, cluster_bootstrap, holm_adjust, paired_cluster_permutation
from rerun_code.verifier_data import (
    assert_group_disjoint,
    grouped_stratified_split,
    remove_evaluation_overlap,
    standardize_verifier_records,
)


def test_preencoded_alignment_requires_unique_identifier():
    manifest = pd.DataFrame([{"record_id": "a", "image_path": "/x/a.jpg"}])
    metadata = pd.DataFrame([{"path": "/legacy/a.jpg"}, {"path": "/duplicate/a.jpg"}])
    vectors = np.eye(2, dtype=np.float32)
    with pytest.raises(AssertionError, match="uniquely align"):
        align_manifest_to_preencoded(manifest, vectors, metadata)


def test_packaged_leakage_helpers_do_not_expose_statistics_shadow_path():
    rerun_root = Path(__file__).parents[1].resolve()
    implementation = (rerun_root / "rerun_code").resolve()
    previous = list(sys.path)
    try:
        located = add_leakage_code_to_path(rerun_root)
        assert located.resolve() == implementation
        assert all(Path(entry or ".").resolve() != implementation for entry in sys.path)
    finally:
        sys.path[:] = previous


def test_notebook_05_uses_package_qualified_retrieval_import():
    notebook = json.loads(
        (Path(__file__).parents[1] / "05_run_multimodel_generation.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "from rerun_code.leakage_safe_retrieval import load_query_bundle" in source
    assert "add_leakage_code_to_path" not in source


def test_notebook_04_resumes_and_audits_independent_evaluation():
    notebook = json.loads(
        (Path(__file__).parents[1] / "04_train_and_test_corrected_verifiers.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "audit_saved_independent_predictions" in source
    assert "INDEPENDENT EVALUATION SKIPPED" in source
    assert 'JAMIA_VERIFIER_PROGRESS_EVERY' in source
    assert 'independent_predictions_path.open("a"' in source
    assert "This record was not saved" in source
    assert 'test_split_sha256' in source


def test_notebook_05_has_adjustable_model_and_bundle_loops():
    notebook = json.loads(
        (Path(__file__).parents[1] / "05_run_multimodel_generation.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for model_key in (
        "qwen2_1_5b", "medgemma_4b", "phi4_multimodal", "gpt_oss_20b",
        "llama_3_1_8b", "medqwen2_7b", "openbiollm_8b",
    ):
        assert f'"{model_key}"' in source
    assert 'BUNDLE_NAMES = ["mimic", "iuhn", "combined"]' in source
    assert 'for model_key in model_keys:' in source
    assert 'for bundle_name in bundle_names:' in source
    assert source.index('for model_key in model_keys:') < source.index('for bundle_name in bundle_names:')
    assert 'JAMIA_MODEL_KEYS' in source
    assert 'JAMIA_BUNDLES' in source
    assert 'from huggingface_hub import get_token, login' in source
    assert 'JAMIA_HF_OFFLINE' in source
    assert 'HF_HUB_OFFLINE' in source
    assert 'Hugging Face authentication completed using HF_TOKEN.' in source
    assert 'print(environment_token)' not in source
    assert 'GENERATION_API_VERSION' in source
    assert 'required_generation_api = 5' in source
    assert 'GENERATION_RESUME_REPAIR_VERSION' in source
    assert 'required_resume_repair = 1' in source
    assert 'PHI4_IMAGE_COMPAT_VERSION' in source
    assert 'required_phi4_image_compat = 3' in source
    assert 'PHI4_CACHE_COMPAT_VERSION' in source
    assert 'required_phi4_cache_compat = 1' in source
    assert 'PHI4_VERIFIER_COMPAT_VERSION' in source
    assert 'required_phi4_verifier_compat = 1' in source
    assert 'PHI4_EMPTY_OUTPUT_RECOVERY_VERSION' in source
    assert 'required_phi4_empty_recovery = 1' in source
    assert 'reference_chat_image_cache_batchencoding_empty_recovery_v6' in source
    assert 'repair_failed_records=True' in source
    assert 'the failed record was not saved' in source
    assert 'importlib.reload(generation_module)' in source
    assert 'Notebook 05 and rerun_code/generation.py are out of sync' in source
    assert 'release_accelerator_memory()' in source
    assert 'notebook05_orchestration_summary.json' in source


def test_generation_resume_validates_model_bundle_run_and_skips_only_complete_records(tmp_path):
    results = tmp_path / "results.jsonl"
    row = {
        "generation_record_id": "medgemma_4b|mimic|q1|A_single_pass",
        "query_record_id": "q1", "condition": "A_single_pass",
        "model_key": "medgemma_4b", "bundle": "mimic", "run_id": "run-1",
        "final_report": "No acute cardiopulmonary abnormality.", "empty_output": False,
    }
    results.write_text(json.dumps(row) + "\n", encoding="utf-8")
    completed, audit = validated_completed_generation_ids(
        results, model_key="medgemma_4b", bundle_name="mimic", run_id="run-1",
        valid_query_ids=["q1"], conditions=["A_single_pass", "D_corrected_lora_gate"],
    )
    assert completed == {row["generation_record_id"]}
    assert audit["n_existing_records"] == 1
    assert audit["counts_by_condition"]["A_single_pass"] == 1

    contaminated = dict(row, model_key="qwen2_1_5b")
    results.write_text(json.dumps(contaminated) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Cross-model record"):
        validated_completed_generation_ids(
            results, model_key="medgemma_4b", bundle_name="mimic", run_id="run-1",
            valid_query_ids=["q1"], conditions=["A_single_pass"],
        )


def test_generation_resume_backs_up_and_removes_only_failed_rows(tmp_path):
    results = tmp_path / "results.jsonl"
    valid = {
        "generation_record_id": "openbiollm_8b|mimic|q1|A_single_pass",
        "query_record_id": "q1", "condition": "A_single_pass",
        "model_key": "openbiollm_8b", "bundle": "mimic", "run_id": "run-1",
        "final_report": "No focal airspace opacity.", "empty_output": False,
    }
    failed = {
        "generation_record_id": "openbiollm_8b|mimic|q1|B_unconditional_4pass",
        "query_record_id": "q1", "condition": "B_unconditional_4pass",
        "model_key": "openbiollm_8b", "bundle": "mimic", "run_id": "run-1",
        "final_report": "", "empty_output": True,
    }
    results.write_text(
        json.dumps(valid) + "\n" + json.dumps(failed) + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="empty/failed report"):
        validated_completed_generation_ids(
            results, model_key="openbiollm_8b", bundle_name="mimic", run_id="run-1",
            valid_query_ids=["q1"],
            conditions=["A_single_pass", "B_unconditional_4pass"],
        )

    completed, audit = validated_completed_generation_ids(
        results, model_key="openbiollm_8b", bundle_name="mimic", run_id="run-1",
        valid_query_ids=["q1"],
        conditions=["A_single_pass", "B_unconditional_4pass"],
        repair_failed_records=True,
    )
    assert completed == {valid["generation_record_id"]}
    assert audit["n_failed_records_removed"] == 1
    assert audit["failed_generation_ids_removed"] == [failed["generation_record_id"]]
    backup = Path(audit["failed_records_backup"])
    assert backup.exists()
    assert [json.loads(line) for line in results.read_text().splitlines()] == [valid]
    assert len(backup.read_text().splitlines()) == 2


def test_empty_revision_falls_back_to_last_usable_report():
    from rerun_code.generation import _result

    result = _result(
        "B_unconditional_4pass",
        ["initial", "revision"],
        ["Usable initial report", ""],
        [1.0, 1.0],
        [],
        "empty_revision_fallback",
    )
    assert result["final_report"] == "Usable initial report"
    assert result["empty_output"] is False
    assert result["empty_generation_passes"] == [1]


def test_preencoded_alignment_by_basename():
    manifest = pd.DataFrame([
        {"record_id": "a", "image_path": "/x/a.jpg"},
        {"record_id": "b", "image_path": "/x/b.jpg"},
    ])
    metadata = pd.DataFrame([{"path": "/old/b.jpg"}, {"path": "/old/a.jpg"}])
    vectors = np.asarray([[2.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    aligned, _, audit = align_manifest_to_preencoded(manifest, vectors, metadata)
    assert aligned[:, 0].tolist() == [1.0, 2.0]
    assert audit["alignment_mode"] == "identifier"


def test_preencoded_file_selection_prefers_image_assets(tmp_path):
    image_index = tmp_path / "faiss_image.index"
    text_index = tmp_path / "faiss_text.index"
    image_index.touch(); text_index.touch()
    selected_index = _select_named(
        tmp_path, FAISS_PRIORITY, [str(text_index), str(image_index)], "image FAISS index"
    )
    assert selected_index == image_index

    generic = tmp_path / "other_vectors.npy"
    image_cache = tmp_path / "image_embeddings_cache.npy"
    np.save(generic, np.zeros((1, 2), dtype=np.float32))
    np.save(image_cache, np.ones((1, 2), dtype=np.float32))
    selected_array = _select_named(
        tmp_path, VECTOR_PRIORITY, [str(generic), str(image_cache)], "NumPy vector"
    )
    assert selected_array == image_cache


def test_grouped_verifier_split_is_disjoint_and_has_three_splits():
    records = []
    for group in range(90):
        for repeat in range(2):
            records.append({
                "patient_id": f"p{group}", "record_id": f"r{group}-{repeat}",
                "report": f"report {group} {repeat}", "label": (group + repeat) % 2,
            })
    frame = standardize_verifier_records(records)
    split = grouped_stratified_split(frame, seed=7)
    assert set(split["split"]) == {"train", "validation", "test"}
    assert_group_disjoint(split)
    assert split.groupby("patient_or_source_group")["split"].nunique().max() == 1
    observed = split["split"].value_counts(normalize=True).to_dict()
    assert observed["train"] == pytest.approx(0.8, abs=0.03)
    assert observed["validation"] == pytest.approx(0.1, abs=0.03)
    assert observed["test"] == pytest.approx(0.1, abs=0.03)


def test_actual_harmony_schema_uses_ground_truth_answer_and_file_name():
    frame = standardize_verifier_records([
        {
            "id": "CXR1000_stmt_0000",
            "file_name": "1000_IM-0003-1001",
            "image_path": "/data/example/1000_IM-0003-1001.jpg",
            "messages": [
                {"role": "system", "content": "Output TRUE or FALSE."},
                {"role": "user", "content": "EVIDENCE\nNo edema.\n\nSTATEMENT:\nThere is no edema."},
            ],
            "ground_truth": {"type": "exact_string", "answer": "TRUE"},
            "task": "cxr_truthfulness",
            "meta": {"labels": []},
        }
    ])
    row = frame.iloc[0]
    assert row["verifier_record_id"] == "CXR1000_stmt_0000"
    assert row["source_record_id"] == "1000_IM-0003-1001"
    assert row["patient_or_source_group"] == "1000_IM-0003-1001"
    assert row["grouping_level"] == "source_report_or_image"
    assert bool(row["verdict"]) is True
    assert row["verdict_source"] == "ground_truth.answer"
    assert "STATEMENT" in row["report_text"]


def test_verifier_evaluation_overlap_removed_by_report_hash():
    frame = standardize_verifier_records([
        {"patient_id": "p1", "record_id": "r1", "report": "No edema.", "label": 1},
        {"patient_id": "p2", "record_id": "r2", "report": "Mild edema.", "label": 0},
    ])
    evaluation = pd.DataFrame([{"patient_key": "other", "record_id": "z", "study_id": "s", "image_id": "i", "report_sha256": frame.iloc[0]["report_sha256"]}])
    clean, excluded = remove_evaluation_overlap(frame, evaluation)
    assert clean["source_record_id"].tolist() == ["r2"]
    assert excluded["source_record_id"].tolist() == ["r1"]


def test_label_metrics_use_event_denominators():
    reference = [[1, 0] + [0] * 11, [0, 1] + [0] * 11]
    prediction = [[1, 1] + [0] * 11, [0, 0] + [0] * 11]
    metrics = aggregate_label_metrics(pd.DataFrame({"reference_vector": reference, "prediction_vector": prediction}))
    assert metrics["fer"] == pytest.approx(0.5)
    assert metrics["fer_abnormal"] == pytest.approx(0.5)
    assert metrics["omission"] == pytest.approx(0.5)
    assert metrics["micro_f1"] == pytest.approx(0.5)


def test_majority_vote_is_per_label():
    assert majority_vote([[1, 0], [1, 1], [0, 1]]) == [1, 1]
    with pytest.raises(ValueError):
        majority_vote([[1], [0]])


def test_ground_truth_uses_json_labels_key_values():
    empty_vector, empty_unknown = ground_truth_vector_from_json_labels([])
    assert empty_vector == [0] * 13
    assert empty_unknown == []
    none_vector, none_unknown = ground_truth_vector_from_json_labels(None)
    assert none_vector == [0] * 13
    assert none_unknown == []
    vector, unknown = ground_truth_vector_from_json_labels(
        ["Atelectasis", "pleural_effusion", "Support Devices"]
    )
    assert unknown == []
    assert vector[0] == 1
    assert vector[8] == 1
    assert vector[12] == 1
    assert sum(vector) == 3
    with pytest.raises(ValueError, match="outside the frozen 13-label vocabulary"):
        ground_truth_vector_from_json_labels(["not-a-valid-label"])
    with pytest.raises(TypeError, match="list, string, or null"):
        ground_truth_vector_from_json_labels({"atelectasis": True})


def test_chexbert_mapping_preserves_no_finding_and_uncertain_policy():
    row = {label.title(): np.nan for label in LABELS_13}
    row["No Finding"] = 1
    row["Edema"] = -1
    policy = {label: 0 for label in LABELS_13}
    policy["edema"] = 1
    vectors, states = map_chexbert_states(pd.DataFrame([row]), policy)
    assert vectors[0][LABELS_13.index("edema")] == 1
    assert states[0]["edema"] == -1
    assert states[0]["no finding"] == 1


def test_clinical_sanity_gate_covers_required_behaviors():
    states = [{label: None for label in (*LABELS_13, "no finding")} for _ in range(4)]
    states[1]["edema"] = -1
    vectors = [[0] * len(LABELS_13) for _ in range(4)]
    vectors[1][LABELS_13.index("edema")] = 1
    vectors[3][LABELS_13.index("support devices")] = 1
    result = validate_clinical_sanity_outputs(states, vectors)
    assert result["passed"] is True


def test_complete_generation_audit_requires_source_labels_and_all_conditions(tmp_path):
    bundles = tmp_path / "bundles"
    generation_root = tmp_path / "generation"
    source_json = tmp_path / "source.json"
    source_json.write_text(json.dumps({"labels": []}), encoding="utf-8")
    query = {
        "record_id": "mimic:test:00000001", "dataset": "mimic",
        "json_path": str(source_json),
    }
    query_dir = bundles / "mimic" / "queries"
    query_dir.mkdir(parents=True)
    (query_dir / "query_metadata.jsonl").write_text(json.dumps(query) + "\n", encoding="utf-8")
    run_dir = generation_root / "model" / "mimic"
    run_dir.mkdir(parents=True)
    (run_dir / "run_provenance.json").write_text(json.dumps({
        "model_key": "model", "bundle": "mimic", "run_id": "run-1",
    }), encoding="utf-8")
    records = []
    for condition in ("A", "B"):
        generation_id = f"model|mimic|{query['record_id']}|{condition}"
        records.append({
            "generation_record_id": generation_id,
            "query_record_id": query["record_id"], "condition": condition,
            "model_key": "model", "bundle": "mimic", "run_id": "run-1",
            "final_report": "No acute cardiopulmonary process.", "empty_output": False,
            "source_dataset": "mimic", "ground_truth_source": "paired_json.labels",
            "reference_labels_json": [], "reference_labels_13_manifest_check": [],
        })
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    frame, audit = audit_complete_generation_results(
        generation_root=generation_root, bundles_root=bundles,
        model_keys=["model"], bundle_names=["mimic"], conditions=["A", "B"],
    )
    assert audit["passed"] is True
    assert len(frame) == 2
    assert frame["reference_vector"].tolist() == [[0] * 13, [0] * 13]
    assert audit["n_equivalent_duplicate_rows_removed"] == 0

    results_path = run_dir / "results.jsonl"
    original_records = [
        json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()
    ]
    runtime_duplicate = dict(original_records[1], total_generation_seconds=99.0)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(runtime_duplicate) + "\n")
    repaired_frame, repaired_audit = audit_complete_generation_results(
        generation_root=generation_root, bundles_root=bundles,
        model_keys=["model"], bundle_names=["mimic"], conditions=["A", "B"],
        repair_equivalent_duplicates=True,
    )
    assert len(repaired_frame) == 2
    assert repaired_audit["n_equivalent_duplicate_rows_removed"] == 1
    assert len(repaired_audit["duplicate_repair_backups"]) == 1
    assert Path(repaired_audit["duplicate_repair_backups"][0]).exists()
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 2

    conflicting_duplicate = dict(original_records[1], final_report="Conflicting report.")
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(conflicting_duplicate) + "\n")
    with pytest.raises(AssertionError, match="Conflicting duplicate"):
        audit_complete_generation_results(
            generation_root=generation_root, bundles_root=bundles,
            model_keys=["model"], bundle_names=["mimic"], conditions=["A", "B"],
            repair_equivalent_duplicates=True,
        )
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in original_records),
        encoding="utf-8",
    )

    source_json.write_text(json.dumps({"caption": "missing labels"}), encoding="utf-8")
    with pytest.raises(KeyError, match="lacks the required 'labels' key"):
        audit_complete_generation_results(
            generation_root=generation_root, bundles_root=bundles,
            model_keys=["model"], bundle_names=["mimic"], conditions=["A", "B"],
        )


def test_generation_audit_rejects_partial_and_empty_outputs(tmp_path):
    bundles = tmp_path / "bundles"
    generation_root = tmp_path / "generation"
    source_json = tmp_path / "source.json"
    source_json.write_text(json.dumps({"labels": []}), encoding="utf-8")
    query = {"record_id": "q1", "dataset": "mimic", "json_path": str(source_json)}
    query_dir = bundles / "mimic" / "queries"
    query_dir.mkdir(parents=True)
    (query_dir / "query_metadata.jsonl").write_text(json.dumps(query) + "\n", encoding="utf-8")
    run_dir = generation_root / "model" / "mimic"
    run_dir.mkdir(parents=True)
    (run_dir / "run_provenance.json").write_text(json.dumps({
        "model_key": "model", "bundle": "mimic", "run_id": "run-1",
    }), encoding="utf-8")
    base = {
        "generation_record_id": "model|mimic|q1|A", "query_record_id": "q1",
        "condition": "A", "model_key": "model", "bundle": "mimic", "run_id": "run-1",
        "final_report": "No acute process.", "source_dataset": "mimic",
        "ground_truth_source": "paired_json.labels", "reference_labels_json": [],
    }
    (run_dir / "results.jsonl").write_text(json.dumps(base) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Incomplete generation run"):
        audit_complete_generation_results(
            generation_root=generation_root, bundles_root=bundles,
            model_keys=["model"], bundle_names=["mimic"], conditions=["A", "B"],
        )
    empty = dict(base, final_report="   ", empty_output=True)
    (run_dir / "results.jsonl").write_text(json.dumps(empty) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="Empty/failed generated report"):
        audit_complete_generation_results(
            generation_root=generation_root, bundles_root=bundles,
            model_keys=["model"], bundle_names=["mimic"], conditions=["A"],
        )


def test_blinded_two_annotator_workflow_and_metrics(tmp_path):
    rows = []
    for index, abnormal in enumerate((False, True)):
        vector = [int(abnormal)] * len(LABELS_13)
        rows.append({
            "generation_record_id": f"g{index}", "query_record_id": f"q{index}",
            "source_dataset": "mimic", "bundle": "mimic", "model_key": "model",
            "condition": "A", "reference_vector": vector,
            "final_report": f"Report {index}", "prediction_vector": vector,
        })
    reports = pd.DataFrame(rows)
    manifest = create_blinded_annotation_materials(reports, tmp_path, n=2, seed=7)
    form1 = pd.read_csv(manifest["annotator1_template"])
    form2 = pd.read_csv(manifest["annotator2_template"])
    assert "generation_record_id" not in form1
    assert "model_key" not in form1
    assert "condition" not in form1
    crosswalk = pd.read_csv(manifest["crosswalk"])
    truth_by_blind = {
        row.blinded_annotation_id: rows[int(str(row.generation_record_id)[1:])]["reference_vector"]
        for row in crosswalk.itertuples()
    }
    for form, annotator in ((form1, "reader-A"), (form2, "reader-B")):
        form["annotator_id"] = annotator
        for label_index, label in enumerate(LABELS_13):
            form[f"human_{label}"] = form["blinded_annotation_id"].map(
                lambda value: truth_by_blind[value][label_index]
            )
    completed1, completed2 = tmp_path / "a.csv", tmp_path / "b.csv"
    form1.to_csv(completed1, index=False); form2.to_csv(completed2, index=False)
    adjudication = create_adjudication_template(completed1, completed2, tmp_path / "adjudication.csv")
    assert adjudication["n_disagreement_cells"] == 0
    result = validate_two_annotators(
        reports[["generation_record_id", "prediction_vector"]],
        annotator1_csv=completed1, annotator2_csv=completed2,
        crosswalk_csv=manifest["crosswalk"],
    )
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["macro_sensitivity"] == pytest.approx(1.0)
    assert result["macro_specificity"] == pytest.approx(1.0)
    assert result["n_f1_evaluable_labels"] == 13
    assert result["interrater"]["overall_cell_agreement"] == pytest.approx(1.0)

    original_template = Path(manifest["annotator1_template"])
    original_template.write_text(original_template.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen annotation material changed"):
        create_blinded_annotation_materials(reports, tmp_path, n=2, seed=7)


def test_undefined_label_f1_is_not_replaced_with_one(tmp_path):
    human = pd.DataFrame({"generation_record_id": ["g1"]})
    for label in LABELS_13:
        human[f"human_{label}"] = 0
    human_path = tmp_path / "human.csv"
    human.to_csv(human_path, index=False)
    result = validate_against_human(
        pd.DataFrame({"generation_record_id": ["g1"], "prediction_vector": [[0] * 13]}),
        human_path,
    )
    assert result["macro_f1"] is None
    assert result["n_f1_evaluable_labels"] == 0
    assert all(value["f1"] is None for value in result["per_label"].values())


def test_notebook_06_contains_complete_two_reader_gate():
    notebook = json.loads(
        (Path(__file__).parents[1] / "06_chexbert_labeling_and_validation_gate.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "audit_complete_generation_results" in source
    assert "REPORT_LABELER_API_VERSION" in source
    assert "required_report_labeler_api = 8" in source
    assert "importlib.reload(report_labeler_module)" in source
    assert "repair_equivalent_duplicates=True" in source
    assert "ensure_official_chexbert_assets" in source
    assert "chexbert_asset_audit.json" in source
    assert "JAMIA_CHEXBERT_AUTO_INSTALL_DEPS" in source
    assert "Hugging Face `datasets` package" in source
    assert "HUMAN ANNOTATION PENDING" in source
    assert "HUMAN_ANNOTATION_NEXT_STEPS.txt" in source
    assert "Notebook 07 remains blocked" in source
    assert "human_annotation_annotator1_completed.csv" in source
    assert "human_annotation_annotator2_completed.csv" in source
    assert "create_adjudication_template" in source
    assert "raw_chexbert_no_finding_preserved" in source


def test_notebook_07_stops_cleanly_and_links_passed_gate_to_frozen_cohort():
    notebook = json.loads(
        (Path(__file__).parents[1] / "07_compute_per_study_and_aggregate_metrics.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "HUMAN VALIDATION PENDING" in source
    assert "analysis_ready" in source
    assert "notebook07_status.json" in source
    assert "_cohort_fingerprint(frame)" in source
    assert "generation_cohort_fingerprint" in source
    assert "NOTEBOOK 07 COMPLETE" in source
    assert 'if not gate.get("passed"): raise AssertionError' not in source


def test_notebook_08_requires_verified_notebook07_outputs_and_correct_holm_families():
    notebook = json.loads(
        (Path(__file__).parents[1] / "08_cluster_bootstrap_and_paired_tests.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "required_statistics_api = 3" in source
    assert "notebook07_status.json" in source
    assert "NOTEBOOK 07 OUTPUTS ARE PENDING" in source
    assert "per_study_metrics_sha256" in source
    assert "_cohort_fingerprint(frame)" in source
    assert "paired query sets differ across conditions" in source
    assert '["model_key", "metric", "bundle", "source_dataset"]' in source
    assert '["metric", "bundle", "source_dataset", "condition"]' in source
    assert "notebook08_status.json" in source


def test_chexbert_asset_resolver_accepts_existing_verified_assets(tmp_path):
    repository = tmp_path / "CheXbert"
    label_script = repository / "src" / "label.py"
    label_script.parent.mkdir(parents=True)
    label_script.write_text("print('label')\n", encoding="utf-8")
    checkpoint = tmp_path / "chexbert.pth"
    checkpoint.write_bytes(b"test-checkpoint")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    assets = ensure_official_chexbert_assets(
        repo=repository,
        checkpoint=checkpoint,
        auto_setup=False,
        expected_checkpoint_sha256=expected,
    )
    assert assets["repository"] == str(repository.resolve())
    assert assets["label_script"] == str(label_script.resolve())
    assert assets["checkpoint"] == str(checkpoint.resolve())
    assert assets["checkpoint_sha256"] == expected
    assert assets["code_installed_this_run"] is False
    assert assets["checkpoint_downloaded_this_run"] is False

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ensure_official_chexbert_assets(
            repo=repository,
            checkpoint=checkpoint,
            auto_setup=False,
            expected_checkpoint_sha256="0" * 64,
        )


def test_chexbert_runtime_dependency_preflight_accepts_installed_package(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="0.14.1\n", stderr="")

    monkeypatch.setattr("rerun_code.report_labeler.subprocess.run", fake_run)
    audit = ensure_chexbert_runtime_dependencies(
        python_executable="/env/bin/python",
        output_dir=tmp_path,
    )
    assert audit["statsmodels_version"] == "0.14.1"
    assert audit["installed_this_run"] is False
    assert len(calls) == 1
    assert (tmp_path / "chexbert_runtime_audit.json").exists()


def test_chexbert_runtime_dependency_preflight_installs_when_missing(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'statsmodels'",
            )
        if command[1:3] == ["-m", "pip"]:
            return SimpleNamespace(returncode=0, stdout="installed\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="0.14.1\n", stderr="")

    monkeypatch.setattr("rerun_code.report_labeler.subprocess.run", fake_run)
    audit = ensure_chexbert_runtime_dependencies(
        python_executable="/env/bin/python",
        output_dir=tmp_path,
    )
    assert audit["installed_this_run"] is True
    assert audit["statsmodels_version"] == "0.14.1"
    assert calls[1][-1] == "statsmodels==0.14.1"
    assert len(calls) == 3


def test_chexbert_import_compatibility_prefers_official_local_dataset(tmp_path):
    source = tmp_path / "CheXbert" / "src"
    label_script = source / "label.py"
    local_package = source / "datasets"
    local_package.mkdir(parents=True)
    label_script.write_text("# test labeler\n", encoding="utf-8")
    (local_package / "unlabeled_dataset.py").write_text(
        "MARKER = 'official-chexbert-local-loader'\n", encoding="utf-8"
    )
    environment, audit = prepare_chexbert_import_compatibility(
        label_script=label_script,
        output_dir=tmp_path / "output",
    )
    completed = __import__("subprocess").run(
        [
            sys.executable,
            "-c",
            "from datasets.unlabeled_dataset import MARKER; print(MARKER)",
        ],
        cwd=str(source),
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "official-chexbert-local-loader"
    assert audit["official_source_checkout_modified"] is False
    assert audit["bert_tokenizer_encode_plus_compatibility"] is True
    assert Path(audit["compatibility_entrypoint"]).exists()
    assert (tmp_path / "output" / "chexbert_import_compatibility_audit.json").exists()


def test_chexbert_compatibility_entrypoint_restores_encode_plus(tmp_path, monkeypatch):
    source = tmp_path / "CheXbert" / "src"
    local_package = source / "datasets"
    local_package.mkdir(parents=True)
    (local_package / "unlabeled_dataset.py").write_text("# local loader\n", encoding="utf-8")
    label_script = source / "label.py"
    label_script.write_text(
        "from transformers import BertTokenizer\n"
        "print(BertTokenizer().encode_plus(['mild', 'edema'])['input_ids'])\n",
        encoding="utf-8",
    )
    fake_transformers = tmp_path / "fake_packages" / "transformers"
    fake_transformers.mkdir(parents=True)
    (fake_transformers / "__init__.py").write_text(
        "class BertTokenizer:\n"
        "    cls_token_id = 101\n"
        "    sep_token_id = 102\n"
        "    def _encode_plus(self, text, text_pair=None):\n"
        "        return {'input_ids': [['wrong'], ['ragged', 'batch']]}\n"
        "    def convert_tokens_to_ids(self, tokens):\n"
        "        return [len(token) for token in tokens]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(fake_transformers.parent))
    environment, audit = prepare_chexbert_import_compatibility(
        label_script=label_script,
        output_dir=tmp_path / "output",
    )
    completed = __import__("subprocess").run(
        [sys.executable, audit["compatibility_entrypoint"]],
        cwd=str(source),
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[101, 4, 5, 102]"
    assert audit["pretokenized_input_returns_flat_ids"] is True
    assert audit["bert_special_tokens_constructed_from_token_ids"] is True


def test_radgraph_compatibility_restores_removed_encode_plus(monkeypatch):
    class BertTokenizer:
        cls_token_id = 101
        sep_token_id = 102

        def _encode_plus(self, text, text_pair=None, *args, **kwargs):
            return {"input_ids": [text, text_pair], "kwargs": kwargs}

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BertTokenizer=BertTokenizer))
    assert _install_radgraph_legacy_tokenizer_compatibility() is True
    assert BertTokenizer().encode_plus("a", "b", return_attention_mask=False) == {
        "input_ids": ["a", "b"], "kwargs": {"return_attention_mask": False}
    }
    assert BertTokenizer().build_inputs_with_special_tokens([4, 5]) == [101, 4, 5, 102]
    assert BertTokenizer().build_inputs_with_special_tokens([4], [5]) == [101, 4, 102, 5, 102]
    assert _install_radgraph_legacy_tokenizer_compatibility() is False


def test_notebook_08_recomputes_vector_derived_metrics_within_replicates():
    notebook = Path(__file__).parents[1] / "08_cluster_bootstrap_and_paired_tests.ipynb"
    source = notebook.read_text(encoding="utf-8")
    assert "vector_derived_metrics" in source
    assert "direct_primary_metrics" in source
    assert "Notebook 07 did not produce usable values for direct primary metrics" in source
    assert "vectorized_cluster_sufficient_statistics_v3" in source


def test_notebook_07_records_the_prespecified_radgraph_reward_component():
    notebook = Path(__file__).parents[1] / "07_compute_per_study_and_aggregate_metrics.ipynb"
    source = notebook.read_text(encoding="utf-8")
    assert "radgraph_f1_reward_component" in source
    assert "partial_entity_relation_RG_ER" in source


def test_notebook_09_requires_current_radgraph_and_statistics_outputs():
    notebook = Path(__file__).parents[1] / "09_error_transitions_ensemble_runtime_exports.ipynb"
    source = notebook.read_text(encoding="utf-8")
    assert "Notebook 08 statistics are stale relative to Notebook 07 metrics" in source
    assert "Notebook 07 RadGraph failed" in source
    assert "partial_entity_relation_RG_ER" in source
    assert "vectorized statistics engine (API 3)" in source


def test_verdict_parser_is_strict_and_identity_gate_fails():
    assert parse_verdict("TRUE") is True
    assert parse_verdict("answer: false") is False
    assert parse_verdict("uncertain") is None
    with pytest.raises(AssertionError, match="Do not publish"):
        assert_model_identity("llama", {"display_name": "Llama 3.1", "require_adapter_base_contains": "3.1"}, "unsloth/llama-3-8b-Instruct")


def test_corrected_training_uses_canonical_base_not_legacy_base():
    spec = {
        "training_base_model": "Qwen/Qwen2-1.5B-Instruct",
        "fallback_base_model": "legacy/converted-qwen",
        "legacy_adapter": "/not/loaded/for/weights",
    }
    assert resolve_training_base_model(spec) == "Qwen/Qwen2-1.5B-Instruct"


def test_medgemma_registry_uses_full_bf16_training_loader():
    config = json.loads((Path(__file__).parents[1] / "rerun_config.json").read_text(encoding="utf-8"))
    spec = config["models"]["medgemma_4b"]
    assert spec["training_base_model"] == "unsloth/medgemma-4b-it"
    assert spec["fallback_base_model"] == "unsloth/medgemma-4b-it-bnb-4bit"
    assert spec["loader_profile"] == "medgemma_unsloth_bf16_training"


def test_phi4_registry_and_scoped_siglip2_compatibility_alias():
    config = json.loads((Path(__file__).parents[1] / "rerun_config.json").read_text(encoding="utf-8"))
    spec = config["models"]["phi4_multimodal"]
    assert spec["training_base_model"] == "microsoft/Phi-4-reasoning-vision-15B"
    assert spec["loader_profile"] == "phi4_reasoning_vision_reference_bf16"

    module = SimpleNamespace()
    values = {"filter_out_non_signature_kwargs": object(), "ChannelDimension": object()}
    assert _install_phi4_siglip2_processor_compatibility(module, values) == (
        "ChannelDimension", "filter_out_non_signature_kwargs"
    )
    assert module.filter_out_non_signature_kwargs is values["filter_out_non_signature_kwargs"]
    assert module.ChannelDimension is values["ChannelDimension"]
    assert _install_phi4_siglip2_processor_compatibility(module, values) == ()


def test_phi4_verifier_is_text_only_but_other_multimodal_verifiers_keep_images(monkeypatch):
    monkeypatch.setattr(
        "rerun_code.generation._install_phi4_legacy_numpy_patch_helpers", lambda: False
    )
    phi_processor = SimpleNamespace(image_processor=SimpleNamespace(normalize=lambda **kwargs: kwargs["image"]))
    phi = ModelRunner(phi_processor, None, None, "multimodal", {}, actual_base="microsoft/Phi-4-reasoning-vision-15B")
    medgemma = ModelRunner(None, None, None, "multimodal", {}, actual_base="unsloth/medgemma-4b-it")
    text = ModelRunner(None, None, None, "text", {}, actual_base="Qwen/Qwen2-1.5B-Instruct")
    assert phi._verifier_image_path("/x/image.jpg") is None
    assert medgemma._verifier_image_path("/x/image.jpg") == "/x/image.jpg"
    assert text._verifier_image_path("/x/image.jpg") is None


def test_phi4_dynamic_cache_is_adapted_only_on_concrete_phi_model_instance():
    class Phi4ForCausalLMV:
        def get_vision_tower(self):
            return object()

        def prepare_inputs_labels_for_multimodal(self, *args):
            return ("original", *args)

    class GenericWrapper:
        def __init__(self, model):
            self.model = model

        def get_base_model(self):
            return self.model

    phi_model = Phi4ForCausalLMV()
    wrapper = GenericWrapper(phi_model)
    targets = _install_phi4_dynamic_cache_compatibility(wrapper)
    assert targets == (f"{Phi4ForCausalLMV.__module__}.Phi4ForCausalLMV",)
    assert phi_model._jamia_phi4_dynamic_cache_compatibility is True
    assert not hasattr(wrapper, "_jamia_phi4_dynamic_cache_compatibility")
    assert _install_phi4_dynamic_cache_compatibility(wrapper) == targets

    class FakeCache:
        def get_seq_length(self):
            return 5

    import torch
    input_ids = torch.ones((1, 1), dtype=torch.long)
    attention_mask = torch.ones((1, 4), dtype=torch.long)
    result = phi_model.prepare_inputs_labels_for_multimodal(
        input_ids, None, attention_mask, FakeCache(), None, object()
    )
    assert result[0] is input_ids
    assert result[1].tolist() == [[5]]
    assert result[2].shape == (1, 6)
    assert result[4] is None

    legacy_cache = (("key", "value"),)
    legacy_result = phi_model.prepare_inputs_labels_for_multimodal(
        input_ids, None, attention_mask, legacy_cache, None, object()
    )
    assert legacy_result[0] == "original"


def test_phi4_verifier_accepts_batchencoding_and_tensor_chat_template_results():
    import torch

    class FakeBatchEncoding(dict):
        pass

    encoded = FakeBatchEncoding(
        input_ids=torch.tensor([[3, 4]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        token_type_ids=torch.tensor([[0, 0]], dtype=torch.long),
    )
    normalized = _chat_template_model_inputs(encoded, torch.device("cpu"))
    assert set(normalized) == {"input_ids", "attention_mask"}
    assert normalized["input_ids"].tolist() == [[3, 4]]
    assert normalized["attention_mask"].tolist() == [[1, 1]]

    tensor_only = _chat_template_model_inputs(
        torch.tensor([3, 4], dtype=torch.long), torch.device("cpu")
    )
    assert set(tensor_only) == {"input_ids"}
    assert tensor_only["input_ids"].tolist() == [[3, 4]]

    with pytest.raises(KeyError, match="input_ids"):
        _chat_template_model_inputs(
            FakeBatchEncoding(attention_mask=torch.ones((1, 2))), torch.device("cpu")
        )


def test_phi4_empty_report_gets_one_audited_deterministic_recovery():
    class FakeRunner:
        is_phi4 = True
        config = {}

        def __init__(self):
            self.calls = []

        def generate(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return ("", 1.25) if len(self.calls) == 1 else ("FINDINGS: Clear. IMPRESSION: No acute disease.", 2.5)

    runner = FakeRunner()
    report, seconds, audit = _generate_report_with_empty_recovery(
        runner, "x" * 12000 + "\nGENERATED REPORT:", image_path="/x/image.jpg"
    )
    assert report.startswith("FINDINGS:")
    assert seconds == 3.75
    assert audit["recovery_used"] is True
    assert audit["attempt_count"] == 2
    assert audit["attempt_seconds"] == [1.25, 2.5]
    assert audit["initial_output_empty"] is True
    assert audit["retry_output_empty"] is False
    assert audit["retry_min_new_tokens"] == 32
    assert len(audit["retry_prompt"]) <= 8000
    assert "Retrieved context shortened" in audit["retry_prompt"]
    assert runner.calls[1][1]["min_new_tokens"] == 32


def test_non_phi_empty_report_is_not_silently_recovered():
    class FakeRunner:
        is_phi4 = False
        config = {}

        def generate(self, prompt, **kwargs):
            return "", 1.0

    report, seconds, audit = _generate_report_with_empty_recovery(
        FakeRunner(), "prompt", image_path="/x/image.jpg"
    )
    assert report == ""
    assert seconds == 1.0
    assert audit["recovery_used"] is False


def test_phi4_generation_uses_reference_prompt_and_numpy_normalization_path(monkeypatch):
    monkeypatch.setattr(
        "rerun_code.generation._install_phi4_legacy_numpy_patch_helpers", lambda: False
    )
    calls = []

    def numpy_normalize(**kwargs):
        calls.append(kwargs)
        assert kwargs["data_format"] is None
        return kwargs["image"] + 1

    image_processor = SimpleNamespace(normalize=lambda **kwargs: kwargs["image"])
    processor = SimpleNamespace(image_processor=image_processor)
    assert _install_phi4_numpy_normalize_compatibility(processor, numpy_normalize) is True
    image = np.zeros((2, 2, 3), dtype=np.float32)
    normalized = processor.image_processor.normalize(
        image=image, mean=[0, 0, 0], std=[1, 1, 1], input_data_format="channels_last"
    )
    assert normalized.shape == (2, 2, 3)
    assert np.array_equal(normalized, image + 1)
    assert calls[0]["input_data_format"] == "channels_last"
    assert calls[0]["data_format"] is None
    assert _install_phi4_numpy_normalize_compatibility(processor, numpy_normalize) is False

    def legacy_numpy_normalize(*, image, mean, std, data_format):
        assert data_format is None
        return image

    legacy_processor = SimpleNamespace(
        image_processor=SimpleNamespace(normalize=lambda **kwargs: kwargs["image"])
    )
    _install_phi4_numpy_normalize_compatibility(
        legacy_processor, legacy_numpy_normalize
    )
    legacy_result = legacy_processor.image_processor.normalize(
        image=image, mean=[0, 0, 0], std=[1, 1, 1],
        input_data_format="channels_last",
    )
    assert legacy_result.shape == (2, 2, 3)

    torch_style_module = SimpleNamespace(
        convert_image_to_patches=lambda image, patch_size: image.permute(0, 1, 2),
        pad_along_first_dim=lambda tensor, target_length: (tensor, tensor),
    )
    assert _install_phi4_legacy_numpy_patch_helpers(torch_style_module) is True
    patch_image = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)
    patches = torch_style_module.convert_image_to_patches(patch_image, 2)
    assert patches.shape == (4, 12)
    padded, mask = torch_style_module.pad_along_first_dim(patches, 6)
    assert padded.shape == (6, 12)
    assert mask.tolist() == [1, 1, 1, 1, 0, 0]
    assert _install_phi4_legacy_numpy_patch_helpers(torch_style_module) is False

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return "CHAT"

    tokenizer = FakeTokenizer()
    runner = ModelRunner(
        processor, tokenizer, None, "multimodal", {},
        actual_base="microsoft/Phi-4-reasoning-vision-15B",
    )
    rendered = runner._format_prompt("Write the report", "/x/image.jpg")
    assert rendered == "<image>\nCHAT<|dummy_84|>"
    assert tokenizer.messages[0]["role"] == "system"
    assert tokenizer.messages[1] == {"role": "user", "content": "Write the report"}
    assert tokenizer.kwargs["return_dict"] is False

    verifier = runner._format_prompt("Answer TRUE or FALSE", None, verifier=True)
    assert verifier == "CHAT"
    assert tokenizer.messages == [{"role": "user", "content": "Answer TRUE or FALSE"}]


def test_reference_text_model_registry_matches_supplied_notebooks():
    config = json.loads((Path(__file__).parents[1] / "rerun_config.json").read_text(encoding="utf-8"))
    expected = {
        "gpt_oss_20b": ("openai/gpt-oss-20b", "base", "auto", "gpt-oss_20b_12combo.ipynb"),
        "llama_3_1_8b": ("unsloth/llama-3-8b-Instruct", "adapter_then_base", "plain", "llama_eval_15combo.ipynb"),
        "medqwen2_7b": ("Echelon-AI/Med-Qwen2-7B", "base", "auto", "medqwen2_7b_eval_12combo.ipynb"),
        "openbiollm_8b": ("aaditya/OpenBioLLM-Llama3-8B", "adapter_then_base", "plain", "biomed_llama_eval_15combo.ipynb"),
    }
    for key, (base, tokenizer_source, chat_policy, reference_notebook) in expected.items():
        spec = config["models"][key]
        assert spec["training_base_model"] == base
        assert spec["loader_profile"] == "reference_causal_lm_bf16"
        assert spec["tokenizer_source"] == tokenizer_source
        assert spec["chat_template_policy"] == chat_policy
        assert spec["reference_loading_notebook"] == reference_notebook
    assert "not Llama 3.1" in config["models"]["llama_3_1_8b"]["display_name"]


def test_device_map_policy_rejects_training_offload_but_allows_evaluation_cpu():
    cpu_model = SimpleNamespace(hf_device_map={"model.layers.0": 0, "model.layers.1": "cpu"})
    with pytest.raises(RuntimeError, match="unsafe for this Trainer job"):
        _audit_model_device_map(cpu_model, "test-model", for_training=True)
    assert _audit_model_device_map(cpu_model, "test-model", for_training=False) == {
        "model.layers.0": "0",
        "model.layers.1": "cpu",
    }
    disk_model = SimpleNamespace(hf_device_map={"model.layers.0": "disk"})
    with pytest.raises(RuntimeError, match="no audited persistent offload directory"):
        _audit_model_device_map(disk_model, "test-model", for_training=False)


def test_plain_prompt_policy_bypasses_chat_template_for_training_and_generation():
    class FakeTokenizer:
        _jamia_chat_template_policy = "plain"

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("plain policy must not call apply_chat_template")

    tokenizer = FakeTokenizer()
    assert render_training_prompt(tokenizer, "REPORT") == "REPORT"
    runner = ModelRunner(tokenizer, tokenizer, None, "text", {}, actual_base="reference/base")
    assert runner._format_prompt("REPORT", None) == "REPORT"


def test_corrected_adapter_base_is_normalized_from_cache_path(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config_path = adapter / "adapter_config.json"
    cached = "/vf/users/example/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/abc"
    config_path.write_text(json.dumps({"base_model_name_or_path": cached, "r": 8}), encoding="utf-8")
    previous = normalize_adapter_base_model(adapter, "Qwen/Qwen2-1.5B-Instruct")
    assert previous == cached
    repaired = json.loads(config_path.read_text(encoding="utf-8"))
    assert repaired["base_model_name_or_path"] == "Qwen/Qwen2-1.5B-Instruct"
    assert repaired["r"] == 8


def test_completed_corrected_adapter_is_reused_only_with_audited_provenance(tmp_path):
    model_out = tmp_path / "verifiers" / "qwen2_1_5b"
    assert completed_corrected_adapter(
        model_out, "qwen2_1_5b", "Qwen/Qwen2-1.5B-Instruct"
    ) is None

    adapter = model_out / "adapter"
    adapter.mkdir(parents=True)
    cached = "/vf/users/example/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct/snapshots/abc"
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": cached}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"valid-placeholder")
    (model_out / "training_provenance.json").write_text(json.dumps({
        "model_key": "qwen2_1_5b",
        "actual_training_base": "Qwen/Qwen2-1.5B-Instruct",
        "legacy_weights_loaded": False,
        "history": [{"loss": 0.1}],
    }), encoding="utf-8")
    completed = completed_corrected_adapter(
        model_out, "qwen2_1_5b", "Qwen/Qwen2-1.5B-Instruct"
    )
    assert completed is not None
    assert completed["actual_base"] == "Qwen/Qwen2-1.5B-Instruct"
    assert completed["history"] == [{"loss": 0.1}]
    assert adapter_config(adapter)["base_model_name_or_path"] == "Qwen/Qwen2-1.5B-Instruct"


def test_completed_adapter_rejects_changed_loader_protocol(tmp_path):
    model_out = tmp_path / "verifiers" / "gpt_oss_20b"
    adapter = model_out / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "openai/gpt-oss-20b"}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"valid-placeholder")
    (model_out / "training_provenance.json").write_text(json.dumps({
        "model_key": "gpt_oss_20b",
        "actual_training_base": "openai/gpt-oss-20b",
        "legacy_weights_loaded": False,
        "loader_profile": "default",
    }), encoding="utf-8")
    spec = {
        "loader_profile": "reference_causal_lm_bf16",
        "tokenizer_source": "base",
        "chat_template_policy": "auto",
    }
    with pytest.raises(AssertionError, match="do not mix loading protocols"):
        completed_corrected_adapter(
            model_out, "gpt_oss_20b", "openai/gpt-oss-20b", spec=spec
        )


def test_incomplete_corrected_adapter_is_not_silently_retrained(tmp_path):
    model_out = tmp_path / "verifiers" / "qwen2_1_5b"
    (model_out / "checkpoints").mkdir(parents=True)
    with pytest.raises(IncompleteCorrectedAdapterError, match="Incomplete corrected-adapter output"):
        completed_corrected_adapter(
            model_out, "qwen2_1_5b", "Qwen/Qwen2-1.5B-Instruct"
        )


def test_incomplete_corrected_adapter_can_be_quarantined_recoverably(tmp_path):
    model_out = tmp_path / "verifiers" / "medgemma_4b"
    checkpoint = model_out / "checkpoints" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    marker = checkpoint / "trainer_state.json"
    marker.write_text("{}", encoding="utf-8")
    quarantined = quarantine_incomplete_corrected_adapter(model_out)
    assert not model_out.exists()
    assert quarantined.parent == model_out.parent
    assert quarantined.name.startswith("medgemma_4b.incomplete-")
    assert (quarantined / "checkpoints" / "checkpoint-1" / "trainer_state.json").exists()
    assert completed_corrected_adapter(
        model_out, "medgemma_4b", "unsloth/medgemma-4b-it"
    ) is None


def test_manual_qwen_load_restores_only_configured_tied_lm_head():
    class Embeddings:
        def __init__(self, weight):
            self.weight = weight

    class FakeQwen:
        def __init__(self):
            self.inputs = Embeddings(object())
            self.outputs = Embeddings(object())

        def tie_weights(self):
            self.outputs.weight = self.inputs.weight

        def get_input_embeddings(self):
            return self.inputs

        def get_output_embeddings(self):
            return self.outputs

    class Config:
        tie_word_embeddings = True

    model = FakeQwen()
    assert _restore_expected_tied_lm_head(model, Config(), ["lm_head.weight"]) == []
    assert model.outputs.weight is model.inputs.weight

    Config.tie_word_embeddings = False
    assert _restore_expected_tied_lm_head(model, Config(), ["lm_head.weight"]) == ["lm_head.weight"]


def test_paired_permutation_uses_matching_query_records():
    assert STATISTICS_API_VERSION >= 3
    rows = []
    for index in range(12):
        reference = [0] * 13
        reference[index % 13] = 1
        rows.append({"query_record_id": f"q{index}", "patient_key": f"p{index//2}", "reference_vector": reference, "prediction_vector": reference, "radgraph_f1": 0.9})
    arm_a = pd.DataFrame(rows)
    arm_b = arm_a.copy()
    arm_b["prediction_vector"] = [[0] * 13 for _ in range(len(arm_b))]
    arm_b["radgraph_f1"] = 0.4
    result = paired_cluster_permutation(arm_a, arm_b, metrics=("omission", "macro_f1", "radgraph_f1"), replicates=50, seed=3)
    assert set(result["metric"]) == {"omission", "macro_f1", "radgraph_f1"}
    assert result["valid_permutations"].min() == 50


def test_vectorized_cluster_bootstrap_preserves_point_estimates_and_cluster_count():
    rows = []
    for index in range(12):
        reference = [0] * 13
        prediction = [0] * 13
        reference[index % 13] = 1
        prediction[index % 13] = 1 if index % 3 else 0
        rows.append({
            "generation_record_id": f"g{index}",
            "patient_key": f"p{index // 2}",
            "reference_vector": reference,
            "prediction_vector": prediction,
            "radgraph_f1": 0.25 + index / 100,
        })
    frame = pd.DataFrame(rows)
    point = aggregate_label_metrics(frame)
    result = cluster_bootstrap(frame, replicates=31, seed=7, batch_size=8)
    macro = result.loc[result["metric"] == "macro_f1"].iloc[0]
    radial = result.loc[result["metric"] == "radgraph_f1"].iloc[0]
    assert macro["estimate"] == pytest.approx(point["macro_f1"])
    assert radial["estimate"] == pytest.approx(point["radgraph_f1"])
    assert macro["n_clusters"] == 6
    assert macro["n_valid_replicates"] == 31


def test_paired_permutation_rejects_reference_or_cluster_mismatch():
    reference = [1] + [0] * 12
    arm_a = pd.DataFrame([
        {
            "query_record_id": "q1", "patient_key": "p1",
            "reference_vector": reference, "prediction_vector": reference,
        }
    ])
    arm_b = arm_a.copy()
    arm_b.at[0, "reference_vector"] = [0] * 13
    with pytest.raises(AssertionError, match="identical reference vectors"):
        paired_cluster_permutation(arm_a, arm_b, metrics=("macro_f1",), replicates=2)

    arm_b = arm_a.copy()
    arm_b.loc[0, "patient_key"] = "different-patient"
    with pytest.raises(AssertionError, match="cluster assignments"):
        paired_cluster_permutation(arm_a, arm_b, metrics=("macro_f1",), replicates=2)


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
