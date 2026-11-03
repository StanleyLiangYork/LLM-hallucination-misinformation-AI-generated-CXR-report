"""Offline checks; no network, GPU, model weights or clinical records required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rerun_code.modality_ablation import (
    ARMS, CONDITIONS, atomic_json, read_jsonl, digest, freeze_json,
    run_matched_condition, validate_record, freeze_experiment, load_experiment,
    generate_with_empty_recovery, activate_empty_recovery_amendment, LEGACY_CODE_HASHES,
    record_path, read_json,
)
from rerun_code.modality_ablation_statistics import align_pairs, paired_difference_ci, transitions
from rerun_code.statistics import paired_cluster_permutation


class FakeRunner:
    is_phi4 = False
    def __init__(self, verdict=False, empty=False):
        self.calls = []
        self.verifications = []
        self.input_audit = []
        self.verdict = verdict
        self.empty = empty

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return ("" if self.empty else "No pleural effusion."), 1.

    def verify(self, report, **kwargs):
        self.verifications.append(kwargs)
        return dict(verdict=self.verdict, seconds=.1)


def entry():
    return dict(query=dict(image_path="/data/query.jpg", report_text="SECRET REFERENCE",
                           labels_raw=["SECRET LABEL"]),
                neighbors=[dict(report_text="Retrieved report.", cosine_score=.5)])


def frame():
    return pd.DataFrame([
        dict(query_record_id=f"q{i}", generation_record_id=f"g{i}", patient_key=f"patient{i//2}",
             evidence_sha256="same", reference_vector=[(i+j)%2 for j in range(13)],
             prediction_vector=[(i+j)%2 for j in range(13)], radgraph_f1=.5)
        for i in range(6)])


class MatchedInputTests(unittest.TestCase):
    def test_recovery_is_symmetric_and_keeps_prompt(self):
        class EmptyOnce(FakeRunner):
            def generate(self, prompt, **kwargs):
                text, seconds = super().generate(prompt, **kwargs)
                return ("" if len(self.calls) == 1 else text), seconds
        for image in (None, "/data/query.jpg"):
            runner = EmptyOnce()
            text, seconds, audit = generate_with_empty_recovery(runner, "unchanged evidence", image_path=image)
            self.assertTrue(text)
            self.assertEqual(seconds, 2.)
            self.assertTrue(audit["recovery_used"])
            self.assertEqual(runner.calls[0][0], runner.calls[1][0])
            self.assertEqual(runner.calls[1][1]["min_new_tokens"], 32)
            self.assertEqual(runner.calls[1][1]["image_path"], image)

    def test_success_never_retried_and_empty_retry_is_bounded(self):
        runner = FakeRunner()
        _, _, audit = generate_with_empty_recovery(runner, "prompt", image_path=None)
        self.assertFalse(audit["recovery_used"])
        self.assertEqual(len(runner.calls), 1)
        runner = FakeRunner(empty=True)
        with self.assertRaises(RuntimeError):
            generate_with_empty_recovery(runner, "prompt", image_path=None)
        self.assertEqual(len(runner.calls), 2)
        self.assertTrue(runner.empty_recovery_failure["exhausted"])
    def test_identical_prompt_and_image_routing(self):
        first, second = FakeRunner(), FakeRunner()
        run_matched_condition(first, CONDITIONS[0], entry(), ARMS[0])
        run_matched_condition(second, CONDITIONS[0], entry(), ARMS[1])
        self.assertEqual(first.calls[0][0], second.calls[0][0])
        self.assertIsNone(first.calls[0][1]["image_path"])
        self.assertEqual(second.calls[0][1]["image_path"], "/data/query.jpg")
        self.assertNotIn("SECRET", first.calls[0][0])

    def test_revision_pass_count_and_adapter_routing(self):
        r = FakeRunner()
        result = run_matched_condition(r, CONDITIONS[1], entry(), ARMS[0])
        self.assertEqual(len(r.calls), 5)
        self.assertEqual(result["stopping_pass"], 4)
        self.assertTrue(all(not args["adapter_enabled"] for _, args in r.calls))
        r = FakeRunner(verdict=True)
        run_matched_condition(r, CONDITIONS[3], entry(), ARMS[0])
        self.assertEqual(len(r.calls), 1)
        self.assertTrue(r.verifications[0]["adapter_enabled"])
        self.assertIsNone(r.verifications[0]["image_path"])

    def test_invalid_outputs_not_success(self):
        with self.assertRaises(RuntimeError):
            run_matched_condition(FakeRunner(empty=True), CONDITIONS[0], entry(), ARMS[0])
        with self.assertRaises(RuntimeError):
            run_matched_condition(FakeRunner(verdict=None), CONDITIONS[3], entry(), ARMS[0])

    def test_record_integrity_and_frozen_provenance(self):
        row = dict(experiment_id="e", generation_record_id="g", final_report="Report", status="complete")
        row["payload_sha256"] = digest(row)
        validate_record(row, dict(experiment_id="e"), "g")
        row["final_report"] = "Changed"
        with self.assertRaises(AssertionError):
            validate_record(row, dict(experiment_id="e"), "g")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            freeze_json(path, dict(a=1)); freeze_json(path, dict(a=1))
            with self.assertRaises(RuntimeError):
                freeze_json(path, dict(a=2))
            bad = Path(directory) / "truncated.jsonl"
            bad.write_text('{"a":1}\n{"a":')
            with self.assertRaises(ValueError):
                read_jsonl(bad)

    def test_pairing_rejects_missing_duplicate_or_changed_truth(self):
        a = frame()
        for bad in (a.iloc[:-1], pd.concat([a, a.iloc[:1]])):
            with self.assertRaises(AssertionError):
                align_pairs(a, bad)
        b = copy.deepcopy(a)
        b.at[0, "reference_vector"] = [0]*13
        with self.assertRaises(AssertionError):
            align_pairs(a, b)

    def test_identical_arms_zero_difference_and_p_one(self):
        a = frame()
        metrics = ["fer_abnormal", "omission", "macro_f1", "radgraph_f1"]
        ci = paired_difference_ci(a, a, metrics, replicates=40)
        self.assertTrue((ci.difference_ci_low == 0).all())
        self.assertTrue((ci.difference_ci_high == 0).all())
        tests = paired_cluster_permutation(a, a, metrics=metrics, replicates=40)
        self.assertTrue((tests.p_value == 1).all())

    def test_rescue_harm_orientation(self):
        good, bad = frame(), frame()
        bad["prediction_vector"] = [[1-x for x in r] for r in bad.reference_vector]
        changes = transitions(bad, good)
        self.assertEqual(changes.rescued.sum(), 6)
        self.assertEqual(changes.harmed.sum(), 0)
        ci = paired_difference_ci(bad, good, ["macro_f1", "omission"], replicates=30)
        self.assertEqual(ci.set_index("metric").loc["macro_f1", "difference_ci_low"], 1.)
        self.assertEqual(ci.set_index("metric").loc["omission", "difference_ci_high"], -1.)

    def test_preflight_freeze_and_empty_label_source(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "rerun_config.example.json").read_text())
        from rerun_code.config import sha256_path
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            config["output_root"] = str(tmp)
            image = tmp / "query.jpg"
            image.write_bytes(b"synthetic fixture, not a real image")
            label = tmp / "query.json"
            atomic_json(label, dict(labels=[]))
            atomic_json(tmp / "manifests/leakage_audit.json", dict(status="fixture"))
            base = tmp / "bundles/mimic"
            gallery = [dict(record_id=f"train{j}", source_split="train", report_text="Context") for j in range(10)]
            queries = [dict(record_id=f"mimic:test:{i:08d}", dataset="mimic", patient_key=f"p{i}", patient_id_reliable=True,
                            image_path=str(image), image_sha256=sha256_path(image), json_path=str(label),
                            labels_raw=[], report_text="Reference") for i in range(634)]
            neighbors = [dict(query_record_id=q["record_id"], neighbors=[dict(g, cosine_score=.5) for g in gallery]) for q in queries]
            for relative, rows in [("queries/query_metadata.jsonl", queries), ("neighbors.jsonl", neighbors),
                                   ("gallery/gallery_metadata.jsonl", gallery)]:
                p = base / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
            out, plan = freeze_experiment(root, config, run_name="test_smoke", models=["medgemma_4b"],
                                          bundles=["mimic"], conditions=["A_single_pass"], smoke_n=2)
            self.assertEqual(plan["expected_records"], 4)
            _, _, inputs = load_experiment(root, config, "test_smoke", check_dependencies=True)
            self.assertEqual(inputs["mimic"][0]["query"]["reference_vector"], [0]*13)
            # Emulate the recognized original frozen code, with a successful saved result.
            old_plan = copy.deepcopy(plan)
            old_plan["code_sha256"].update(LEGACY_CODE_HASHES)
            old_plan.pop("experiment_id")
            old_plan["experiment_id"] = digest(old_plan)
            atomic_json(out / "plan.json", old_plan)
            row = dict(experiment_id=old_plan["experiment_id"], generation_record_id="saved",
                       status="complete", final_report="No effusion", reports_by_pass=["No effusion"],
                       model_key="medgemma_4b", bundle="mimic")
            row["payload_sha256"] = digest(row)
            dest = record_path(out, "saved")
            atomic_json(dest, row)
            old_bytes, plan_bytes = dest.read_bytes(), (out / "plan.json").read_bytes()
            with self.assertRaises(RuntimeError):
                load_experiment(root, config, "test_smoke")
            activate_empty_recovery_amendment(root, config, "test_smoke")
            activate_empty_recovery_amendment(root, config, "test_smoke")
            load_experiment(root, config, "test_smoke")
            self.assertEqual(old_bytes, dest.read_bytes())
            self.assertEqual(plan_bytes, (out / "plan.json").read_bytes())
            amended = read_json(out / "empty_output_recovery_amendment_v1.json")
            self.assertEqual(amended["retained_counts"], {"medgemma_4b/mimic": 1})
            image.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                load_experiment(root, config, "test_smoke", check_dependencies=True)


if __name__ == "__main__":
    unittest.main()
