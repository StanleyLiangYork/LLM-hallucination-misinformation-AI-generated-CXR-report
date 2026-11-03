"""Isolated matched-backbone direct-image ablation; never edits primary results."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd

from .config import sha256_path
from .generation import CONDITIONS, ModelRunner, retrieval_context
from .leakage_safe_retrieval import assert_neighbor_records
from .report_labeler import LABELS_13, ground_truth_vector_from_json_labels

VERSION = 1
EMPTY_RECOVERY_VERSION = 1
EMPTY_RETRY_MIN_NEW_TOKENS = 32
LEGACY_CODE_HASHES = {
    "modality_ablation.py": "ddbff32701cee2eb905954d0c913075ed287b8e5412f3776bb18ba1a6e735a62",
    "modality_ablation_statistics.py": "3d2cdbeed4f4c066c6cd91f347b0e262d029699527b14dc55e29bb9100f3de5b",
}
ARMS = ("retrieved_text_only", "retrieved_text_plus_image")
MODELS = ("medgemma_4b", "phi4_multimodal")
SYSTEM = ("You are a radiology assistant. Use the supplied evidence to produce a concise "
          "final chest radiograph report. Output only the report, without reasoning or commentary.")


def biowulf_path(value):
    """Resolve the known /vf alias without rewriting manifests or guessing filenames."""
    value = str(value)
    prefix = os.environ.get("JAMIA_OLD_DATA_ROOT", "/vf/users/liangz2/openi").rstrip("/") + "/"
    canonical = Path(os.environ.get("JAMIA_DATA_ROOT", "/data/liangz2/openi")) / value[len(prefix):] if value.startswith(prefix) else Path(value)
    if canonical.exists():
        return canonical
    if Path(value).exists():
        return Path(value)
    raise FileNotFoundError(f"Required Biowulf path not found: {canonical} (recorded path: {value})")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Truncated or invalid JSON at {path}:{number}; restore the source") from exc
    return rows


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Frozen provenance changed: {path}. Use a NEW RUN_NAME; do not mix runs.")
    else:
        atomic_json(path, value)


def unique(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError(f"Duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result


def first_prompt(neighbors):
    return ("Write a concise FINDINGS and IMPRESSION chest radiograph report for the target case. "
            "Use only the supplied evidence. Retrieved reports are examples, not confirmed findings "
            "for this case; do not copy unsupported details. Do not mention retrieval.\n\n"
            + retrieval_context(neighbors) + "\n\nGENERATED REPORT:")


def next_prompt(report, neighbors, pass_number):
    return (f"Revision pass {pass_number}. Recheck the candidate against the supplied evidence. "
            "Correct unsupported findings, contradictions, and important omissions. "
            "Return only the revised report.\n\nCANDIDATE:\n" + report
            + "\n\n" + retrieval_context(neighbors) + "\n\nREVISED REPORT:")


def source_hashes(root):
    names = ("modality_ablation.py", "modality_ablation_statistics.py", "generation.py",
             "modeling.py", "verifier_data.py", "leakage_safe_retrieval.py", "report_labeler.py",
             "metrics.py", "statistics.py")
    return {name: sha256_path(Path(__file__).resolve().parent / name) for name in names}


def freeze_experiment(root, config, *, run_name="matched_input_v1", models=MODELS,
                      bundles=("mimic", "iuhn", "combined"), conditions=CONDITIONS,
                      smoke_n=None, seed=20261029, max_input_tokens=12000):
    """CPU preflight: freeze all inputs, images, labels and current corrected adapters."""
    from .modeling import resolve_training_base_model
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("RUN_NAME must contain only letters, digits, underscores or hyphens")
    if not models or not set(models) <= set(MODELS) or len(set(models)) != len(models):
        raise ValueError("Choose unique supported multimodal models")
    if not bundles or not set(bundles) <= {"mimic", "iuhn", "combined"} or len(set(bundles)) != len(bundles):
        raise ValueError("Choose unique supported bundles")
    if not conditions or not set(conditions) <= set(CONDITIONS) or len(set(conditions)) != len(conditions):
        raise ValueError("Choose unique supported conditions")
    if "A_single_pass" not in conditions:
        raise ValueError("Keep single-pass RAG as the primary modality comparison")
    if smoke_n is not None and (smoke_n < 1 or "smoke" not in run_name.lower()):
        raise ValueError("Smoke runs need a positive limit and 'smoke' in RUN_NAME")
    if config["generation"].get("do_sample"):
        raise ValueError("This protocol freezes greedy decoding: do_sample must be false")
    if config["retrieval_k"] != 10:
        raise ValueError("The current protocol requires the original top ten neighbors")
    results = Path(config["output_root"])
    out = results / "post_rerun" / "modality_ablation" / run_name
    dependencies, frozen, image_hashes = {}, {}, {}
    audit_path = results / "manifests" / "leakage_audit.json"
    dependencies[str(audit_path)] = sha256_path(audit_path)
    for bundle in bundles:
        directory = results / "bundles" / bundle
        paths = [directory / "queries/query_metadata.jsonl", directory / "neighbors.jsonl",
                 directory / "gallery/gallery_metadata.jsonl"]
        for path in paths:
            dependencies[str(path)] = sha256_path(path)
        queries = unique(read_jsonl(paths[0]), "record_id")
        neighbors = unique(read_jsonl(paths[1]), "query_record_id")
        gallery = unique(read_jsonl(paths[2]), "record_id")
        if set(queries) != set(neighbors):
            raise AssertionError(f"Query/neighbor sets differ: {bundle}")
        expected = {"mimic": 634, "iuhn": 400, "combined": 1034}[bundle]
        if len(queries) != expected:
            raise AssertionError(f"Expected {expected} frozen queries in {bundle}, got {len(queries)}")
        selected = sorted(queries)
        if smoke_n:
            # Deterministic pseudo-random technical subset; never a scientific subsample.
            selected = sorted(selected, key=lambda x: digest([seed, x]))[:smoke_n]
        entries = []
        for qid in selected:
            query = dict(queries[qid])
            ns = neighbors[qid]["neighbors"]
            assert_neighbor_records(query, ns, expected_k=10, phash_threshold=config["phash_threshold"])
            if len({n["record_id"] for n in ns}) != 10:
                raise AssertionError(f"Duplicate neighbors for {qid}")
            for n in ns:
                g = gallery.get(n["record_id"])
                if g is None or g.get("source_split") != "train":
                    raise AssertionError(f"Neighbor not in frozen clean training gallery: {qid}")
                if any(g.get(k) != v for k, v in n.items() if k not in {"gallery_position", "cosine_score"}):
                    raise AssertionError(f"Neighbor content differs from clean gallery: {qid}")
            if not query.get("patient_key"):
                raise AssertionError(f"Missing patient/source group for {qid}")
            if query.get("dataset") == "mimic" and not query.get("patient_id_reliable"):
                raise AssertionError(f"MIMIC must retain its reliable patient identifier: {qid}")
            image = biowulf_path(query["image_path"])
            query["image_path"] = str(image)
            if str(image) not in image_hashes:
                image_hashes[str(image)] = sha256_path(image)
            if image_hashes[str(image)] != query.get("image_sha256"):
                raise AssertionError(f"Query image changed: {image}")
            json_path = biowulf_path(query["json_path"])
            query["json_path"] = str(json_path)
            raw = read_json(json_path)
            if "labels" not in raw:
                raise KeyError(f"Required labels key missing: {json_path}")
            vector, _ = ground_truth_vector_from_json_labels(raw["labels"])
            cached, _ = ground_truth_vector_from_json_labels(query["labels_raw"])
            if vector != cached:
                raise AssertionError(f"Source labels differ from frozen query: {qid}")
            dependencies[str(json_path)] = sha256_path(json_path)
            query["reference_vector"] = vector
            entries.append({"query": query, "neighbors": ns,
                            "evidence_sha256": digest(retrieval_context(ns))})
        frozen[bundle] = entries
    dependencies.update(image_hashes)
    specs = {key: config["models"][key] for key in models}
    for key, spec in specs.items():
        if spec["architecture"] != "multimodal":
            raise ValueError(f"Not a multimodal checkpoint: {key}")
        if "D_corrected_lora_gate" in conditions:
            directory = results / "verifiers" / key
            p = read_json(directory / "training_provenance.json")
            if (p.get("model_key") != key or p.get("legacy_weights_loaded") is not False
                    or p.get("actual_training_base") != resolve_training_base_model(spec)):
                raise AssertionError(f"Corrected verifier provenance mismatch: {key}")
            weights = [p for p in (directory / "adapter").glob("adapter_model.*")
                       if p.suffix in {".bin", ".safetensors"}]
            if len(weights) != 1 or weights[0].stat().st_size == 0:
                raise ValueError(f"Missing/ambiguous corrected adapter weights: {key}")
            metrics = read_json(directory / "independent_test_metrics.json")
            predictions = read_jsonl(directory / "independent_test_predictions.jsonl")
            if metrics.get("n", 0) <= 0 or metrics["n"] != len(predictions):
                raise ValueError(f"Incomplete independent verifier evaluation: {key}")
            for path in [directory / "training_provenance.json", directory / "adapter/adapter_config.json",
                         directory / "independent_test_metrics.json",
                         directory / "independent_test_predictions.jsonl", *weights]:
                dependencies[str(path)] = sha256_path(path)
    plan = dict(version=VERSION, run_name=run_name, models=list(models), bundles=list(bundles),
                conditions=list(conditions), arms=list(ARMS), smoke_n=smoke_n, seed=seed,
                max_input_tokens=max_input_tokens, generation=config["generation"],
                max_revision_passes=config["max_revision_passes"], model_specs=specs,
                labeler=config["labeler"], results_root=str(results),
                primary_metrics=["fer_abnormal", "omission", "macro_f1", "radgraph_f1"],
                bootstrap_replicates=10000, permutation_replicates=10000,
                expected_records=sum(map(len, frozen.values())) * len(models) * len(conditions) * 2,
                query_counts={key: len(value) for key, value in frozen.items()},
                code_sha256=source_hashes(root), dependencies_sha256=dependencies,
                frozen_inputs_sha256=digest(frozen),
                verifier_policy="no pixels anywhere in text arm; existing model-specific verifier in image arm",
                prompt_policy="same modality-neutral text in both arms; rerun BOTH arms; no prompt truncation")
    plan["empty_output_policy"] = "one identical-prompt retry with min_new_tokens=32, both models/arms; unresolved remains failure"
    plan["experiment_id"] = digest(plan)
    freeze_json(out / "plan.json", plan)
    freeze_json(out / "frozen_inputs.json", frozen)
    print(f"Frozen {plan['expected_records']:,} final reports at {out}")
    return out, plan


def load_experiment(root, config, run_name, *, check_dependencies=False):
    out = Path(config["output_root"]) / "post_rerun/modality_ablation" / run_name
    plan = read_json(out / "plan.json")
    if digest({k: v for k, v in plan.items() if k != "experiment_id"}) != plan["experiment_id"]:
        raise AssertionError("Plan was edited after freezing")
    frozen = read_json(out / "frozen_inputs.json")
    if digest(frozen) != plan["frozen_inputs_sha256"]:
        raise RuntimeError("Frozen input data changed; restore the original.")
    current = source_hashes(root)
    if current != plan["code_sha256"]:
        amendment_path = out / "empty_output_recovery_amendment_v1.json"
        if not amendment_path.exists():
            raise RuntimeError("Frozen code differs. Run Notebook 16's explicit recovery-amendment cell; do not edit plan.json.")
        amendment = read_json(amendment_path)
        if (amendment.get("experiment_id") != plan["experiment_id"]
                or amendment.get("original_code_sha256") != plan["code_sha256"]
                or amendment.get("amended_code_sha256") != current
                or amendment.get("amendment_sha256") != digest({k: v for k, v in amendment.items() if k != "amendment_sha256"})):
            raise RuntimeError("Recovery amendment does not match these inputs/code; no hash bypass is allowed.")
    if check_dependencies:
        for path, expected in plan["dependencies_sha256"].items():
            if sha256_path(path) != expected:
                raise RuntimeError(f"Frozen dependency changed: {path}")
    return out, plan, frozen


def activate_empty_recovery_amendment(root, config, run_name):
    """Explicit narrow migration. Preserve plan and every saved generation byte."""
    from datetime import datetime, timezone
    out = Path(config["output_root"]) / "post_rerun/modality_ablation" / run_name
    plan = read_json(out / "plan.json")
    current = source_hashes(root)
    target = out / "empty_output_recovery_amendment_v1.json"
    if current == plan["code_sha256"] or target.exists():
        load_experiment(root, config, run_name, check_dependencies=True)
        return target if target.exists() else None
    for name, old in plan["code_sha256"].items():
        if old != current.get(name) and LEGACY_CODE_HASHES.get(name) != old:
            raise RuntimeError(f"Unrecognized prior code for {name}. Stop and inspect; do not rewrite its hash.")
    if set(current) != set(plan["code_sha256"]):
        raise RuntimeError("Source inventory changed beyond the approved amendment")
    if digest({k: v for k, v in plan.items() if k != "experiment_id"}) != plan["experiment_id"]:
        raise AssertionError("Original plan fingerprint is invalid")
    if digest(read_json(out / "frozen_inputs.json")) != plan["frozen_inputs_sha256"]:
        raise AssertionError("Frozen inputs changed")
    for path, expected in plan["dependencies_sha256"].items():
        if sha256_path(path) != expected:
            raise RuntimeError(f"Original dependency changed: {path}")
    inventory, counts = {}, {}
    # No worker may be writing original records while the migration is audited.
    with contextlib.ExitStack() as stack:
        for model in sorted(plan["models"]):
            stack.enter_context(run_lock(out, model))
        for path in sorted((out / "records").glob("*.json")):
            row = read_json(path)
            identifier = row["generation_record_id"]
            validate_record(row, plan, identifier)
            if path != record_path(out, identifier):
                raise AssertionError(f"Record filename mismatch: {path}")
            passes = row.get("reports_by_pass", [])
            if not passes or any(not str(value).strip() for value in passes):
                raise AssertionError(f"Saved record contains an empty pass: {identifier}")
            inventory[path.name] = sha256_path(path)
            key = f"{row['model_key']}/{row['bundle']}"
            counts[key] = counts.get(key, 0) + 1
        payload = dict(experiment_id=plan["experiment_id"], version=EMPTY_RECOVERY_VERSION,
                       recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                       original_code_sha256=plan["code_sha256"], amended_code_sha256=current,
                       retained_record_sha256=inventory, retained_counts=counts,
                       policy="Retry only an empty report once with the exact same prompt/evidence/image and min_new_tokens=32. "
                              "Apply to both models and arms. No verifier retry; unresolved output remains a failure.",
                       reuse_basis="Old completed records had no empty passes; the unchanged first-attempt branch is reusable. "
                                   "This is a post hoc amendment, not the original frozen protocol.")
        payload["amendment_sha256"] = digest(payload)
        freeze_json(target, payload)
    print("Amendment saved; original records untouched:", counts, flush=True)
    return target


def generate_with_empty_recovery(runner, prompt, *, image_path):
    """Same text/evidence and image routing; never introduce a recovery prompt."""
    attempts = []
    for attempt in range(2):
        kwargs = dict(image_path=image_path, adapter_enabled=False)
        if attempt:
            kwargs["min_new_tokens"] = EMPTY_RETRY_MIN_NEW_TOKENS
        text, seconds = runner.generate(prompt, **kwargs)
        attempts.append(dict(attempt=attempt + 1, empty=not bool(text.strip()), seconds=float(seconds),
                             min_new_tokens=kwargs.get("min_new_tokens"), prompt_sha256=digest(prompt),
                             direct_image=bool(image_path)))
        if text.strip():
            return text, sum(a["seconds"] for a in attempts), dict(attempts=attempts, recovery_used=attempt > 0)
    runner.empty_recovery_failure = dict(attempts=attempts, recovery_used=True, exhausted=True)
    raise RuntimeError("Empty report after original attempt and identical-prompt min_new_tokens=32 retry; record NOT complete")


class MatchedInputRunner(ModelRunner):
    """Preserve established loading/compatibility; remove arm-specific verbal cues."""
    def __init__(self, *args, max_input_tokens=12000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_input_tokens = max_input_tokens
        self.input_audit = []

    def _format_prompt(self, prompt, image_path, *, verifier=False):
        if not self.is_phi4:
            return super()._format_prompt(prompt, image_path, verifier=verifier)
        messages = [{"role": "user", "content": prompt}]
        if not verifier:
            messages.insert(0, {"role": "system", "content": SYSTEM})
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not verifier:
            rendered += "<|dummy_84|>"
        if image_path:
            rendered = "<image>\n" + rendered
        return rendered

    def generate(self, prompt, *, image_path=None, adapter_enabled=False,
                 max_new_tokens=None, min_new_tokens=None, verifier=False):
        import torch
        from PIL import Image
        started = time.perf_counter()
        rendered = self._format_prompt(prompt, image_path, verifier=verifier)
        if not image_path and any(marker in rendered for marker in
                                  ("<image>", "<|image_1|>", "<|image|>", "<start_of_image>")):
            raise AssertionError("No-image arm unexpectedly contains an image placeholder")
        with contextlib.ExitStack() as stack:
            stack.enter_context(self._adapter_context(adapter_enabled))
            if image_path:
                image = stack.enter_context(Image.open(image_path)).convert("RGB")
                stack.callback(image.close)
                inputs = self.processor(text=rendered, images=[image] if self.is_phi4 else image,
                                        return_tensors="pt", truncation=False)
            else:
                # Do not substitute a blank image: genuinely no visual tensors/tokens.
                inputs = self.tokenizer(rendered, return_tensors="pt", truncation=False)
            visual = [key for key, value in inputs.items()
                      if ("pixel" in key or "image" in key) and value is not None]
            if bool(visual) != bool(image_path):
                raise AssertionError(f"Image routing mismatch: image={bool(image_path)}, keys={visual}")
            length = int(inputs["input_ids"].shape[-1])
            if length > self.max_input_tokens:
                raise RuntimeError(f"Input has {length} tokens, above frozen limit {self.max_input_tokens}; "
                                   "no silent truncation. Revise protocol under a new RUN_NAME.")
            self.input_audit.append(dict(verifier=verifier, direct_image=bool(image_path),
                                         prompt_sha256=digest(prompt), rendered_sha256=digest(rendered),
                                         input_tokens=length, visual_keys=visual))
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            kwargs = dict(max_new_tokens=int(max_new_tokens or self.config["max_new_tokens"]),
                          do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                          eos_token_id=self.tokenizer.eos_token_id)
            if min_new_tokens is not None:
                if not 0 < int(min_new_tokens) <= kwargs["max_new_tokens"]:
                    raise ValueError("min_new_tokens must be positive and <= max_new_tokens")
                kwargs["min_new_tokens"] = int(min_new_tokens)
            if self.is_phi4:
                kwargs["use_cache"] = True
            with torch.inference_mode():
                output = self.model.generate(**inputs, **kwargs)
            text = self.tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
        return text, time.perf_counter() - started


def run_matched_condition(runner, condition, entry, arm, *, max_revision_passes=4):
    if arm not in ARMS or condition not in CONDITIONS:
        raise ValueError((arm, condition))
    image = entry["query"]["image_path"] if arm == ARMS[1] else None
    neighbors = entry["neighbors"]
    reports, verifications, prompts, seconds, recovery = [], [], [], [], []
    runner.input_audit = []
    runner.empty_recovery_failure = None
    prompt = first_prompt(neighbors)
    stop = "single_pass" if condition == CONDITIONS[0] else "max_passes"
    for number in range(max_revision_passes + 1):
        try:
            text, elapsed, audit = generate_with_empty_recovery(runner, prompt, image_path=image)
        except RuntimeError as exc:
            raise RuntimeError(f"Report generation failed at pass {number}: {exc}") from exc
        recovery.append(dict(pass_number=number, **audit))
        reports.append(text); prompts.append(prompt); seconds.append(float(elapsed))
        if condition == CONDITIONS[0] or number == max_revision_passes:
            break
        if condition in CONDITIONS[2:]:
            verification = runner.verify(text, image_path=image, evidence=retrieval_context(neighbors),
                                         adapter_enabled=condition == CONDITIONS[3])
            if verification.get("verdict") not in (True, False):
                raise RuntimeError("Unparseable verifier verdict; record NOT complete")
            verifications.append(verification)
            if verification["verdict"]:
                stop = "verifier_true"
                break
        prompt = next_prompt(text, neighbors, number + 1)
    return dict(final_report=reports[-1], reports_by_pass=reports, prompts=prompts,
                empty_output_recovery_version=EMPTY_RECOVERY_VERSION, empty_output_recovery_by_pass=recovery,
                verifier_outputs=verifications, generation_seconds_by_pass=seconds,
                stop_reason=stop, stopping_pass=len(reports)-1, input_audit=runner.input_audit,
                verifier_uses_image=bool(verifications and image and not runner.is_phi4),
                runtime_seconds=sum(seconds)+sum(float(v["seconds"]) for v in verifications))


def record_id(model, bundle, query, condition, arm):
    return "|".join((model, bundle, query, condition, arm))


def record_path(out, identifier):
    return Path(out) / "records" / (hashlib.sha256(identifier.encode()).hexdigest() + ".json")


def validate_record(row, plan, identifier):
    if (row.get("experiment_id") != plan["experiment_id"]
            or row.get("generation_record_id") != identifier
            or not str(row.get("final_report") or "").strip()
            or row.get("status") != "complete"):
        raise AssertionError(f"Invalid saved record: {identifier}")
    if row.get("payload_sha256") != digest({k: v for k, v in row.items() if k != "payload_sha256"}):
        raise AssertionError(f"Saved record content changed: {identifier}")


@contextlib.contextmanager
def run_lock(out, model):
    import fcntl
    path = Path(out) / (model + ".lock")
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {model} in this run") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_generation(root, config, run_name, *, selected_models=None, selected_bundles=None):
    from .modeling import load_processor_and_model, resolve_training_base_model, assert_model_identity, release_accelerator_memory
    from transformers import set_seed
    import torch
    out, plan, frozen = load_experiment(root, config, run_name, check_dependencies=True)
    models = list(selected_models or plan["models"])
    bundles = list(selected_bundles or plan["bundles"])
    if not set(models) <= set(plan["models"]) or not set(bundles) <= set(plan["bundles"]):
        raise ValueError("Execution subset must belong to the frozen plan")
    failures = []
    for key in models:
        runner = model = processor = tokenizer = backbone = None
        with run_lock(out, key):
            try:
                requested = [record_id(key, b, e["query"]["record_id"], c, arm)
                             for b in bundles for e in frozen[b] for c in plan["conditions"] for arm in ARMS]
                completed = 0
                for identifier in requested:
                    path = record_path(out, identifier)
                    if path.exists():
                        validate_record(read_json(path), plan, identifier)
                        completed += 1
                print(f"{key}: {completed}/{len(requested)} requested records already saved", flush=True)
                if completed == len(requested):
                    print(f"Skipping completed model {key}; no model load or record writes.", flush=True)
                    continue
                spec = plan["model_specs"][key]
                base = resolve_training_base_model(spec)
                adapter = Path(plan["results_root"]) / "verifiers" / key / "adapter"
                use_adapter = "D_corrected_lora_gate" in plan["conditions"]
                release_accelerator_memory()
                processor, tokenizer, model, actual = load_processor_and_model(
                    spec, adapter_path=adapter if use_adapter else None, base_model_override=base)
                assert_model_identity(key, spec, actual)
                if use_adapter and (not getattr(model, "peft_config", None) or not callable(getattr(model, "disable_adapter", None))):
                    raise AssertionError("Corrected adapter must be attached and disableable for generation")
                model.eval()
                packages = {}
                for package in ("torch", "transformers", "peft", "numpy", "pillow"):
                    packages[package] = importlib.metadata.version(package)
                backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
                runtime = dict(actual_base=actual, versions=packages, python=sys.version,
                               model_revision=getattr(backbone.config, "_commit_hash", None),
                               model_config=backbone.config.to_dict(),
                               tokenizer_class=type(tokenizer).__name__,
                               tokenizer_chat_template=str(getattr(tokenizer, "chat_template", None)),
                               processor_class=type(processor).__name__,
                               dtype=str(next(model.parameters()).dtype),
                               gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
                # Normalize model configuration to JSON-native data.
                runtime = json.loads(json.dumps(runtime, default=str))
                freeze_json(out / f"runtime_{key}.json", runtime)
                runner = MatchedInputRunner(processor, tokenizer, model, "multimodal", plan["generation"],
                                            actual_base=actual, max_input_tokens=plan["max_input_tokens"])
                for bundle in bundles:
                    for index, entry in enumerate(frozen[bundle]):
                        query = entry["query"]
                        for condition in plan["conditions"]:
                            # Counterbalance order, reset the same seed in each paired arm.
                            pair_seed = int(digest([plan["seed"], key, bundle, query["record_id"], condition])[:8], 16)
                            arms = ARMS if pair_seed % 2 else ARMS[::-1]
                            for arm in arms:
                                identifier = record_id(key, bundle, query["record_id"], condition, arm)
                                dest = record_path(out, identifier)
                                if dest.exists():
                                    validate_record(read_json(dest), plan, identifier)
                                    continue
                                try:
                                    set_seed(pair_seed)
                                    result = run_matched_condition(runner, condition, entry, arm,
                                                                  max_revision_passes=plan["max_revision_passes"])
                                    row = dict(generation_record_id=identifier, experiment_id=plan["experiment_id"],
                                               status="complete", model_key=key, bundle=bundle, condition=condition,
                                               arm=arm, query_record_id=query["record_id"], source_dataset=query["dataset"],
                                               patient_key=query["patient_key"],
                                               reference_report=query["report_text"], reference_vector=query["reference_vector"],
                                               evidence_sha256=entry["evidence_sha256"], seed=pair_seed,
                                               runtime_sha256=digest(runtime), **result)
                                    row["payload_sha256"] = digest(row)
                                    atomic_json(dest, row)
                                except Exception as exc:
                                    failed = dict(generation_record_id=identifier, error=f"{type(exc).__name__}: {exc}",
                                                  traceback=traceback.format_exc(), input_audit=runner.input_audit,
                                                  empty_recovery_failure=runner.empty_recovery_failure)
                                    atomic_json(out / "failures" / dest.name, failed)
                                    raise RuntimeError(f"Stopped {key}/{bundle}: {identifier}. Saved diagnostics in failures/; "
                                                       "valid records will be skipped on restart.") from exc
                        if index % 10 == 0:
                            print(key, bundle, index + 1, "/", len(frozen[bundle]), flush=True)
            except Exception as exc:
                failures.append(dict(model_key=key, error=str(exc), traceback=traceback.format_exc()))
                print(traceback.format_exc())
            finally:
                if runner is not None:
                    runner.model = runner.processor = runner.tokenizer = None
                runner = model = processor = tokenizer = backbone = None
                release_accelerator_memory()
    atomic_json(out / "last_execution.json", dict(models=models, bundles=bundles, failures=failures))
    if failures:
        raise RuntimeError("Generation has unresolved failures. Completed records are preserved.\n" +
                           "\n\n".join(f"{f['model_key']}: {f['error']}\n{f['traceback']}" for f in failures))
    return out


def complete_frame(root, config, run_name):
    out, plan, frozen = load_experiment(root, config, run_name)
    rows, missing = [], []
    expected = set()
    for model in plan["models"]:
        runtime_hash = digest(read_json(out / f"runtime_{model}.json"))
        for bundle in plan["bundles"]:
            for entry in frozen[bundle]:
                q = entry["query"]
                for condition in plan["conditions"]:
                    for arm in ARMS:
                        identifier = record_id(model, bundle, q["record_id"], condition, arm)
                        path = record_path(out, identifier)
                        expected.add(path.name)
                        if not path.exists():
                            missing.append(identifier)
                            continue
                        row = read_json(path)
                        validate_record(row, plan, identifier)
                        if (row["reference_vector"] != q["reference_vector"]
                                or row["evidence_sha256"] != entry["evidence_sha256"]
                                or row["runtime_sha256"] != runtime_hash):
                            raise AssertionError(f"Record does not match frozen inputs: {identifier}")
                        rows.append(row)
    unexpected = {p.name for p in (out / "records").glob("*.json")} - expected
    audit = dict(expected=plan["expected_records"], complete=len(rows), missing=missing,
                 unexpected=sorted(unexpected), smoke_only=plan["smoke_n"] is not None)
    atomic_json(out / "completeness.json", audit)
    if missing or unexpected or len(rows) != plan["expected_records"]:
        raise RuntimeError(f"Incomplete paired cohort. Read {out / 'completeness.json'}; do not analyze survivors only.")
    amendment_path = out / "empty_output_recovery_amendment_v1.json"
    if amendment_path.exists():
        for name, expected_hash in read_json(amendment_path)["retained_record_sha256"].items():
            if sha256_path(out / "records" / name) != expected_hash:
                raise AssertionError(f"A retained pre-amendment record changed: {name}")
    recovery_counts = {}
    for row in rows:
        key = "/".join(row[k] for k in ("model_key", "bundle", "condition", "arm"))
        count = recovery_counts.setdefault(key, dict(n_records=0, records_retried=0, passes_retried=0))
        count["n_records"] += 1
        retries = sum(bool(p.get("recovery_used")) for p in row.get("empty_output_recovery_by_pass", []))
        count["records_retried"] += int(retries > 0)
        count["passes_retried"] += retries
    atomic_json(out / "empty_output_recovery_summary.json", dict(
        post_hoc_amendment=amendment_path.exists(), by_stratum=recovery_counts,
        note="Retained old records contain no empty passes. Retry prevalence is descriptive; disclose the post hoc amendment."))
    return out, plan, pd.DataFrame(rows)


def score_reports(root, config, run_name, *, chunk_size=128):
    from .report_labeler import (ensure_official_chexbert_assets, run_official_chexbert,
                                map_chexbert_states, clinical_sanity_reports, validate_clinical_sanity_outputs)
    from .metrics import _install_radgraph_legacy_tokenizer_compatibility
    out, plan, frame = complete_frame(root, config, run_name)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    assets = ensure_official_chexbert_assets(repo=plan["labeler"]["chexbert_repo"],
                                           checkpoint=plan["labeler"]["checkpoint"], auto_setup=False)
    checkpoint = sha256_path(assets["checkpoint"])
    label_script = Path(assets["repository"]) / "src/label.py"
    if not label_script.exists():
        label_script = Path(assets["repository"]) / "label.py"
    versions = {p: importlib.metadata.version(p) for p in ("radgraph", "transformers", "torch")}
    protocol = dict(experiment_id=plan["experiment_id"], checkpoint=checkpoint,
                    label_script_sha256=sha256_path(label_script), chunk_size=chunk_size,
                    uncertain_policy=plan["labeler"]["uncertain_policy"], versions=versions,
                    cohort_sha256=digest(frame["payload_sha256"].tolist()),
                    radgraph="radgraph-xl; reward_level=all; partial entity/relation index 1")
    freeze_json(out / "scoring_protocol.json", protocol)
    _install_radgraph_legacy_tokenizer_compatibility()
    from radgraph import F1RadGraph
    evaluator = None
    scored = []
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start:start + chunk_size].copy()
        directory = out / "scoring_chunks" / f"{start:07d}"
        cache = directory / "scored.json"
        key = digest([protocol, chunk["generation_record_id"].tolist()])
        if cache.exists():
            saved = read_json(cache)
            if saved["key"] != key or len(saved["rows"]) != len(chunk):
                raise AssertionError(f"Scoring cache mismatch: {cache}")
            scored.extend(saved["rows"])
            continue
        sanity = clinical_sanity_reports()
        inputs = pd.concat([chunk[["generation_record_id", "final_report"]], sanity], ignore_index=True)
        raw = run_official_chexbert(inputs, repo=assets["repository"], checkpoint=assets["checkpoint"],
                                   output_dir=directory / "chexbert")
        vectors, states = map_chexbert_states(raw, plan["labeler"]["uncertain_policy"])
        validate_clinical_sanity_outputs(states[-4:], vectors[-4:])
        if evaluator is None:
            evaluator = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        _, rewards, _, _ = evaluator(hyps=chunk["final_report"].tolist(), refs=chunk["reference_report"].tolist())
        if not isinstance(rewards, (tuple, list)) or len(rewards) != 3:
            raise ValueError("Unexpected RadGraph rewards structure")
        rg = np.asarray(rewards[1], dtype=float).reshape(-1)
        if len(rg) != len(chunk) or not np.isfinite(rg).all() or ((rg < 0) | (rg > 1)).any():
            raise ValueError("Missing/invalid per-report RadGraph F1; do not zero-fill")
        # Preserve each original schema: mixed pre/post-amendment DataFrames
        # otherwise insert NaN into absent fields and invalidate payload hashes.
        rows = []
        for offset, identifier in enumerate(chunk["generation_record_id"]):
            row = read_json(record_path(out, identifier))
            validate_record(row, plan, identifier)
            row["prediction_vector"] = vectors[offset]
            row["radgraph_f1"] = float(rg[offset])
            rows.append(row)
        atomic_json(cache, dict(key=key, rows=rows))
        scored.extend(rows)
        print(f"Scored {min(start+chunk_size, len(frame))}/{len(frame)}", flush=True)
    atomic_json(out / "scored_records.json", dict(protocol_sha256=digest(protocol), rows=scored))
    # Snapshot only: this does not certify new reports or overwrite the prior validation gate.
    gate_path = Path(plan["results_root"]) / "labeler/validation/validation_gate.json"
    atomic_json(out / "measurement_provenance.json", dict(
        previous_validation_gate_path=str(gate_path),
        previous_validation_gate_sha256=sha256_path(gate_path) if gate_path.exists() else None,
        previous_validation_gate=read_json(gate_path) if gate_path.exists() else None,
        new_reports_human_validated=False,
        note="Prior extraction validation does not guarantee accuracy on this new report distribution; "
             "reference-label limitations and single-clinician review remain applicable."))
    return pd.DataFrame(scored)
