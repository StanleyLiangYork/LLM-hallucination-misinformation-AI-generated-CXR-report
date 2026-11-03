from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


LABELS_13 = (
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged cardiomediastinum", "fracture", "lung lesion", "lung opacity",
    "pleural effusion", "pleural other", "pneumonia", "pneumothorax", "support devices",
)
CHEXBERT_CONDITIONS_14 = (*LABELS_13, "no finding")
REPORT_LABELER_API_VERSION = 8
CHEXBERT_CODE_URL = "https://github.com/stanfordmlgroup/CheXbert.git"
CHEXBERT_CODE_REVISION = "6d22a96d73f18d0a7cf5b0dbebdac50cf8e4c1aa"
CHEXBERT_CHECKPOINT_REPO_ID = "StanfordAIMI/RRG_scorers"
CHEXBERT_CHECKPOINT_FILENAME = "chexbert.pth"
CHEXBERT_CHECKPOINT_REVISION = "6646433b3ad83a10f6e141db76d0ece44312b236"
CHEXBERT_CHECKPOINT_SHA256 = "6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1"
CHEXBERT_STATSMODELS_VERSION = "0.14.1"

_DUPLICATE_RUNTIME_KEYS = {
    "generation_seconds_by_pass",
    "total_generation_seconds",
    "total_verifier_seconds",
    "seconds",
    "attempt_seconds",
}


def _canonical(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _without_runtime_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime_fields(item)
            for key, item in value.items()
            if str(key) not in _DUPLICATE_RUNTIME_KEYS
        }
    if isinstance(value, list):
        return [_without_runtime_fields(item) for item in value]
    return value


def _repair_equivalent_generation_duplicates(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    repair: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove only scientifically equivalent duplicate generation rows."""
    results_path = Path(path)
    unique: list[dict[str, Any]] = []
    first_by_id: dict[str, tuple[int, dict[str, Any], Any]] = {}
    duplicate_rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(records, start=1):
        row = dict(raw)
        generation_id = str(row.get("generation_record_id") or "")
        scientific = _without_runtime_fields(row)
        if generation_id not in first_by_id:
            first_by_id[generation_id] = (line_number, row, scientific)
            unique.append(row)
            continue
        first_line, _, first_scientific = first_by_id[generation_id]
        if scientific != first_scientific:
            raise AssertionError(
                "Conflicting duplicate generation_record_id in "
                f"{results_path}: {generation_id!r} at rows {first_line} and "
                f"{line_number}. Reports, predictions, prompts, labels, or provenance differ; "
                "inspect the preserved file and choose the valid run explicitly."
            )
        duplicate_rows.append(
            {
                "generation_record_id": generation_id,
                "kept_row": first_line,
                "removed_row": line_number,
                "differences_limited_to_runtime_fields": row != first_by_id[generation_id][1],
            }
        )

    audit: dict[str, Any] = {
        "n_input_rows": len(records),
        "n_unique_rows": len(unique),
        "n_equivalent_duplicate_rows": len(duplicate_rows),
        "duplicate_generation_ids": sorted(
            {row["generation_record_id"] for row in duplicate_rows}
        ),
        "duplicate_rows": duplicate_rows,
        "repair_enabled": bool(repair),
        "backup_path": None,
        "input_sha256": _sha256_path(results_path),
        "output_sha256": _sha256_path(results_path),
    }
    if not duplicate_rows:
        return unique, audit
    if not repair:
        first = duplicate_rows[0]
        raise AssertionError(
            f"Equivalent duplicate generation_record_id in {results_path}: "
            f"{first['generation_record_id']!r}. Enable repair_equivalent_duplicates "
            "to create a backup and retain only the first scientifically equivalent row."
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = results_path.with_name(
        f"{results_path.name}.equivalent-duplicates-{stamp}.bak"
    )
    temporary = results_path.with_name(f".{results_path.name}.deduplicate-{stamp}.tmp")
    shutil.copy2(results_path, backup)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in unique:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        temporary.replace(results_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    audit["backup_path"] = str(backup)
    audit["output_sha256"] = _sha256_path(results_path)
    return unique, audit


def _chexbert_label_script(repository: str | Path) -> Path | None:
    root = Path(repository)
    for candidate in (root / "src" / "label.py", root / "label.py"):
        if candidate.exists():
            return candidate
    return None


def ensure_official_chexbert_assets(
    *,
    repo: str | Path,
    checkpoint: str | Path,
    auto_setup: bool | None = None,
    expected_checkpoint_sha256: str = CHEXBERT_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    """Resolve or install pinned CheXbert code and checkpoint assets."""
    configured_repo = Path(
        os.environ.get("JAMIA_CHEXBERT_REPO", str(repo))
    ).expanduser()
    configured_checkpoint = Path(
        os.environ.get("JAMIA_CHEXBERT_CHECKPOINT", str(checkpoint))
    ).expanduser()
    if auto_setup is None:
        auto_setup = (
            os.environ.get("JAMIA_CHEXBERT_AUTO_SETUP", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )

    label_script = _chexbert_label_script(configured_repo)
    code_installed = False
    clone_stdout = ""
    clone_stderr = ""
    if label_script is None:
        if not auto_setup:
            raise FileNotFoundError(
                f"Official CheXbert label.py not found below {configured_repo}. "
                "Clone https://github.com/stanfordmlgroup/CheXbert.git there or set "
                "JAMIA_CHEXBERT_REPO. JAMIA_CHEXBERT_AUTO_SETUP=0 disabled setup."
            )
        if configured_repo.exists() and any(configured_repo.iterdir()):
            raise FileNotFoundError(
                f"CheXbert destination exists but has no label.py and is not empty: "
                f"{configured_repo}. Set JAMIA_CHEXBERT_REPO to an empty/new destination "
                "or install the official repository manually."
            )
        configured_repo.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "git", "clone", "--no-checkout", CHEXBERT_CODE_URL, str(configured_repo)
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        clone_stdout, clone_stderr = completed.stdout, completed.stderr
        if completed.returncode:
            raise RuntimeError(
                f"Could not clone official CheXbert into {configured_repo}; git exit "
                f"code={completed.returncode}. stderr={completed.stderr[-2000:]}"
            )
        checkout = subprocess.run(
            [
                "git", "-C", str(configured_repo), "checkout", "--detach",
                CHEXBERT_CODE_REVISION,
            ],
            text=True,
            capture_output=True,
        )
        if checkout.returncode:
            raise RuntimeError(
                f"CheXbert cloned but pinned revision {CHEXBERT_CODE_REVISION} could "
                f"not be checked out. stderr={checkout.stderr[-2000:]}"
            )
        label_script = _chexbert_label_script(configured_repo)
        if label_script is None:
            raise FileNotFoundError(
                f"Pinned CheXbert checkout has no label.py below {configured_repo}"
            )
        code_installed = True

    checkpoint_path = configured_checkpoint
    checkpoint_downloaded = False
    if not checkpoint_path.exists():
        if not auto_setup:
            raise FileNotFoundError(
                f"CheXbert checkpoint not found: {checkpoint_path}. Set "
                "JAMIA_CHEXBERT_CHECKPOINT or enable JAMIA_CHEXBERT_AUTO_SETUP."
            )
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = Path(
                hf_hub_download(
                    repo_id=CHEXBERT_CHECKPOINT_REPO_ID,
                    filename=CHEXBERT_CHECKPOINT_FILENAME,
                    revision=CHEXBERT_CHECKPOINT_REVISION,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not download the pinned Stanford AIMI CheXbert checkpoint. "
                "Run on a node with Hugging Face access or set "
                "JAMIA_CHEXBERT_CHECKPOINT to a local copy."
            ) from exc
        checkpoint_downloaded = True

    actual_checkpoint_sha256 = _sha256_path(checkpoint_path)
    if (
        expected_checkpoint_sha256
        and actual_checkpoint_sha256.lower() != expected_checkpoint_sha256.lower()
    ):
        raise RuntimeError(
            f"CheXbert checkpoint checksum mismatch at {checkpoint_path}. "
            f"Expected={expected_checkpoint_sha256}, actual={actual_checkpoint_sha256}."
        )

    git_revision = None
    if (configured_repo / ".git").exists():
        revision = subprocess.run(
            ["git", "-C", str(configured_repo), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
        )
        if revision.returncode == 0:
            git_revision = revision.stdout.strip()
    return {
        "repository": str(configured_repo.resolve()),
        "label_script": str(label_script.resolve()),
        "label_script_sha256": _sha256_path(label_script),
        "code_url": CHEXBERT_CODE_URL,
        "requested_code_revision": CHEXBERT_CODE_REVISION,
        "actual_code_revision": git_revision,
        "code_installed_this_run": code_installed,
        "clone_stdout": clone_stdout,
        "clone_stderr": clone_stderr,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_repo_id": CHEXBERT_CHECKPOINT_REPO_ID,
        "checkpoint_filename": CHEXBERT_CHECKPOINT_FILENAME,
        "checkpoint_revision": CHEXBERT_CHECKPOINT_REVISION,
        "checkpoint_sha256": actual_checkpoint_sha256,
        "checkpoint_downloaded_this_run": checkpoint_downloaded,
        "auto_setup": bool(auto_setup),
    }


def ensure_chexbert_runtime_dependencies(
    *,
    python_executable: str | Path,
    output_dir: str | Path,
    auto_install: bool | None = None,
) -> dict[str, Any]:
    """Verify the labeling interpreter and install CheXbert's missing statsmodels pin."""
    executable = str(python_executable)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if auto_install is None:
        auto_install = (
            os.environ.get("JAMIA_CHEXBERT_AUTO_INSTALL_DEPS", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
    probe_code = (
        "import importlib.metadata as m; import statsmodels; "
        "print(m.version('statsmodels'))"
    )
    probe = subprocess.run(
        [executable, "-c", probe_code], text=True, capture_output=True
    )
    installed_this_run = False
    install_command: list[str] | None = None
    install_stdout = ""
    install_stderr = ""
    if probe.returncode:
        missing_statsmodels = "No module named 'statsmodels'" in (
            (probe.stderr or "") + (probe.stdout or "")
        )
        if not missing_statsmodels:
            raise RuntimeError(
                f"CheXbert runtime dependency check failed under {executable}. "
                f"stderr={probe.stderr[-2000:]}"
            )
        if not auto_install:
            raise ModuleNotFoundError(
                "CheXbert requires statsmodels. Install "
                f"statsmodels=={CHEXBERT_STATSMODELS_VERSION} in {executable}, set "
                "JAMIA_CHEXBERT_PYTHON to an environment that contains it, or enable "
                "JAMIA_CHEXBERT_AUTO_INSTALL_DEPS."
            )
        install_command = [
            executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            f"statsmodels=={CHEXBERT_STATSMODELS_VERSION}",
        ]
        install = subprocess.run(install_command, text=True, capture_output=True)
        install_stdout, install_stderr = install.stdout, install.stderr
        (output / "chexbert_dependency_install_stdout.txt").write_text(
            install_stdout, encoding="utf-8"
        )
        (output / "chexbert_dependency_install_stderr.txt").write_text(
            install_stderr, encoding="utf-8"
        )
        if install.returncode:
            raise RuntimeError(
                f"Could not install statsmodels=={CHEXBERT_STATSMODELS_VERSION} with "
                f"{executable}. Full logs are in {output}. stderr={install_stderr[-2000:]}"
            )
        installed_this_run = True
        probe = subprocess.run(
            [executable, "-c", probe_code], text=True, capture_output=True
        )
        if probe.returncode:
            raise RuntimeError(
                "statsmodels installation completed but its import still fails under "
                f"{executable}. stderr={probe.stderr[-2000:]}"
            )
    audit = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": executable,
        "statsmodels_version": probe.stdout.strip(),
        "required_by_official_chexbert": True,
        "official_requirements_pin": CHEXBERT_STATSMODELS_VERSION,
        "installed_this_run": installed_this_run,
        "auto_install_enabled": bool(auto_install),
        "install_command": install_command,
    }
    (output / "chexbert_runtime_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return audit


def prepare_chexbert_import_compatibility(
    *,
    label_script: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Prepare isolated namespace and tokenizer compatibility for CheXbert."""
    script = Path(label_script).resolve()
    local_datasets = script.parent / "datasets"
    unlabeled_dataset = local_datasets / "unlabeled_dataset.py"
    if not unlabeled_dataset.exists():
        raise FileNotFoundError(
            f"Official CheXbert dataset loader not found: {unlabeled_dataset}"
        )
    output = Path(output_dir)
    compatibility_root = output / "_chexbert_import_compatibility"
    compatibility_package = compatibility_root / "datasets"
    compatibility_package.mkdir(parents=True, exist_ok=True)
    initializer = compatibility_package / "__init__.py"
    initializer_text = (
        '"""Runtime namespace shim for the official CheXbert dataset loader."""\n'
        f"__path__ = [{str(local_datasets.resolve())!r}]\n"
    )
    initializer.write_text(initializer_text, encoding="utf-8")
    entrypoint = compatibility_root / "run_official_chexbert.py"
    entrypoint_text = (
        '"""Generated compatibility launcher for the pinned official CheXbert labeler."""\n'
        "import runpy\n"
        "import sys\n"
        "from transformers import BertTokenizer\n\n"
        "if not hasattr(BertTokenizer, 'encode_plus'):\n"
        "    if not hasattr(BertTokenizer, '_encode_plus'):\n"
        "        raise AttributeError('BertTokenizer provides neither encode_plus nor _encode_plus')\n"
        "    def _chexbert_encode_plus(self, text, text_pair=None, *args, **kwargs):\n"
        "        # CheXbert passes a pre-tokenized list and reads only input_ids.\n"
        "        # Modern _encode_plus treats that list as a batch and returns ragged IDs.\n"
        "        if isinstance(text, (list, tuple)) and (not text or isinstance(text[0], str)):\n"
        "            first_ids = self.convert_tokens_to_ids(list(text))\n"
        "            second_ids = None\n"
        "            if text_pair is not None:\n"
        "                if not isinstance(text_pair, (list, tuple)):\n"
        "                    return self._encode_plus(text, text_pair=text_pair, *args, **kwargs)\n"
        "                second_ids = self.convert_tokens_to_ids(list(text_pair))\n"
        "            input_ids = [self.cls_token_id] + list(first_ids) + [self.sep_token_id]\n"
        "            if second_ids is not None:\n"
        "                input_ids.extend(list(second_ids) + [self.sep_token_id])\n"
        "            return {'input_ids': input_ids}\n"
        "        return self._encode_plus(text, text_pair=text_pair, *args, **kwargs)\n"
        "    BertTokenizer.encode_plus = _chexbert_encode_plus\n\n"
        f"_LABEL_SCRIPT = {str(script)!r}\n"
        f"_SOURCE_DIRECTORY = {str(script.parent)!r}\n"
        "if _SOURCE_DIRECTORY not in sys.path:\n"
        "    sys.path.insert(0, _SOURCE_DIRECTORY)\n"
        "sys.argv[0] = _LABEL_SCRIPT\n"
        "runpy.run_path(_LABEL_SCRIPT, run_name='__main__')\n"
    )
    entrypoint.write_text(entrypoint_text, encoding="utf-8")
    environment = os.environ.copy()
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    prior_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(compatibility_root.resolve()) + (
        os.pathsep + prior_pythonpath if prior_pythonpath else ""
    )
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "CheXbert's src/datasets directory is a namespace package and can be "
            "shadowed by the installed Hugging Face datasets package."
        ),
        "official_dataset_loader": str(unlabeled_dataset.resolve()),
        "official_dataset_loader_sha256": _sha256_path(unlabeled_dataset),
        "compatibility_initializer": str(initializer.resolve()),
        "compatibility_initializer_sha256": _sha256_path(initializer),
        "compatibility_entrypoint": str(entrypoint.resolve()),
        "compatibility_entrypoint_sha256": _sha256_path(entrypoint),
        "bert_tokenizer_encode_plus_compatibility": True,
        "pretokenized_input_returns_flat_ids": True,
        "bert_special_tokens_constructed_from_token_ids": True,
        "tokenizers_parallelism": environment["TOKENIZERS_PARALLELISM"],
        "official_source_checkout_modified": False,
        "pythonpath_prefix": str(compatibility_root.resolve()),
    }
    (output / "chexbert_import_compatibility_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return environment, audit


def ground_truth_vector_from_json_labels(
    labels: Sequence[str] | str | None,
    *,
    strict: bool = True,
) -> tuple[list[int], list[str]]:
    """Build the reference vector only from an existing JSON ``labels`` value.

    Presence of the key is checked by the caller because a value alone cannot
    distinguish an absent key from an explicitly empty/null value. An existing
    empty list, empty string, or null value means all 13 labels are negative.
    """
    if labels is None:
        values: list[str] = []
    elif isinstance(labels, str):
        values = [labels]
    elif isinstance(labels, Sequence) and not isinstance(labels, (bytes, bytearray)):
        values = [str(value) for value in labels]
    else:
        raise TypeError(f"JSON labels must be a list, string, or null; found {type(labels).__name__}")
    canonical = {_canonical(value) for value in values if str(value).strip()}
    known = set(LABELS_13)
    unknown = sorted(canonical - known)
    if strict and unknown:
        raise ValueError(
            "The JSON labels key contains values outside the frozen 13-label vocabulary: "
            + ", ".join(unknown)
        )
    return [int(label in canonical) for label in LABELS_13], unknown


def _source_labels_for_query(query: Mapping[str, Any]) -> tuple[Any, list[int], str]:
    json_path = Path(str(query.get("json_path") or ""))
    if not json_path.exists():
        raise FileNotFoundError(
            f"Query {query.get('record_id')!r} has no readable paired source JSON: {json_path}"
        )
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot parse paired source JSON {json_path}: {exc}") from exc
    if "labels" not in metadata:
        raise KeyError(
            f"Paired source JSON lacks the required 'labels' key: {json_path}. "
            "Only a present but empty labels value means all-negative."
        )
    labels = metadata["labels"]
    vector, _ = ground_truth_vector_from_json_labels(labels, strict=True)
    return labels, vector, str(json_path.resolve())


def audit_complete_generation_results(
    *,
    generation_root: str | Path,
    bundles_root: str | Path,
    model_keys: Sequence[str],
    bundle_names: Sequence[str],
    conditions: Sequence[str],
    repair_equivalent_duplicates: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only a complete, source-verified multi-model generation cohort."""
    generation_root = Path(generation_root)
    bundles_root = Path(bundles_root)
    missing: list[str] = []
    bundle_queries: dict[str, dict[str, dict[str, Any]]] = {}
    bundle_sources: dict[str, dict[str, tuple[Any, list[int], str]]] = {}

    for bundle in bundle_names:
        metadata_path = bundles_root / bundle / "queries" / "query_metadata.jsonl"
        if not metadata_path.exists():
            missing.append(str(metadata_path))
            continue
        queries = _read_jsonl(metadata_path)
        by_id = {str(row.get("record_id") or ""): row for row in queries}
        if "" in by_id or len(by_id) != len(queries):
            raise AssertionError(f"Query metadata has empty or duplicate record_id values: {metadata_path}")
        bundle_queries[bundle] = by_id
        bundle_sources[bundle] = {
            query_id: _source_labels_for_query(query) for query_id, query in by_id.items()
        }

    expected_run_paths: list[tuple[str, str, Path, Path]] = []
    for model_key in model_keys:
        for bundle in bundle_names:
            output_dir = generation_root / model_key / bundle
            results_path = output_dir / "results.jsonl"
            provenance_path = output_dir / "run_provenance.json"
            expected_run_paths.append((model_key, bundle, results_path, provenance_path))
            for path in (results_path, provenance_path):
                if not path.exists():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Notebook 06 requires the complete frozen generation experiment. Missing files:\n"
            + "\n".join(sorted(set(missing)))
        )

    all_records: list[dict[str, Any]] = []
    run_audit: list[dict[str, Any]] = []
    for model_key, bundle, results_path, provenance_path in expected_run_paths:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if str(provenance.get("model_key")) != model_key or str(provenance.get("bundle")) != bundle:
            raise AssertionError(f"Run provenance identity mismatch: {provenance_path}")
        run_id = str(provenance.get("run_id") or "")
        if not run_id:
            raise AssertionError(f"Run provenance has no run_id: {provenance_path}")
        records, duplicate_audit = _repair_equivalent_generation_duplicates(
            results_path,
            _read_jsonl(results_path),
            repair=repair_equivalent_duplicates,
        )
        expected_queries = bundle_queries[bundle]
        expected_pairs = {(query_id, condition) for query_id in expected_queries for condition in conditions}
        actual_pairs: set[tuple[str, str]] = set()

        for raw in records:
            row = dict(raw)
            query_id = str(row.get("query_record_id") or "")
            condition = str(row.get("condition") or "")
            generation_id = str(row.get("generation_record_id") or "")
            expected_generation_id = f"{model_key}|{bundle}|{query_id}|{condition}"
            if generation_id != expected_generation_id:
                raise AssertionError(
                    f"Generation identifier mismatch in {results_path}: {generation_id!r} != "
                    f"{expected_generation_id!r}"
                )
            if str(row.get("model_key")) != model_key or str(row.get("bundle")) != bundle:
                raise AssertionError(f"Record path identity mismatch for {generation_id}")
            if str(row.get("run_id") or "") != run_id:
                raise AssertionError(f"Record run_id differs from {provenance_path}: {generation_id}")
            if (query_id, condition) not in expected_pairs:
                raise AssertionError(f"Unexpected query/condition in {results_path}: {(query_id, condition)}")
            actual_pairs.add((query_id, condition))
            report = str(row.get("final_report") or "").strip()
            if not report or row.get("empty_output") is True:
                raise AssertionError(
                    f"Empty/failed generated report cannot be interpreted as all-negative: {generation_id}"
                )
            source_labels, source_vector, source_json_path = bundle_sources[bundle][query_id]
            if row.get("ground_truth_source") != "paired_json.labels":
                raise AssertionError(f"Invalid ground_truth_source for {generation_id}")
            if "reference_labels_json" not in row:
                raise AssertionError(f"Missing reference_labels_json for {generation_id}")
            recorded_vector, _ = ground_truth_vector_from_json_labels(
                row["reference_labels_json"], strict=True
            )
            if recorded_vector != source_vector:
                raise AssertionError(
                    f"Recorded reference labels differ from the source JSON labels key: {generation_id}"
                )
            if "reference_labels_13_manifest_check" in row:
                manifest_vector, _ = ground_truth_vector_from_json_labels(
                    row["reference_labels_13_manifest_check"], strict=True
                )
                if manifest_vector != source_vector:
                    raise AssertionError(f"Manifest label check differs from source JSON: {generation_id}")
            query = expected_queries[query_id]
            expected_source_dataset = str(query.get("dataset") or "")
            if str(row.get("source_dataset") or "") != expected_source_dataset:
                raise AssertionError(f"source_dataset mismatch for {generation_id}")
            row["final_report"] = report
            row["reference_labels_source_json"] = source_labels
            row["reference_vector"] = source_vector
            row["reference_abnormal"] = bool(sum(source_vector))
            row["source_json_path"] = source_json_path
            row["labels_key_present"] = True
            all_records.append(row)

        missing_pairs = sorted(expected_pairs - actual_pairs)
        extra_pairs = sorted(actual_pairs - expected_pairs)
        if missing_pairs or extra_pairs or len(records) != len(expected_pairs):
            raise AssertionError(
                f"Incomplete generation run {model_key}/{bundle}: expected={len(expected_pairs)}, "
                f"observed={len(records)}, missing_examples={missing_pairs[:10]}, "
                f"extra_examples={extra_pairs[:10]}"
            )
        run_audit.append({
            "model_key": model_key, "bundle": bundle, "run_id": run_id,
            "n_queries": len(expected_queries), "n_records": len(records),
            "results_sha256": _sha256_path(results_path),
            "run_provenance_sha256": _sha256_path(provenance_path),
            "query_metadata_sha256": _sha256_path(
                bundles_root / bundle / "queries" / "query_metadata.jsonl"
            ),
            "duplicate_repair": duplicate_audit,
        })

    frame = pd.DataFrame(all_records)
    if frame.empty or frame["generation_record_id"].duplicated().any():
        raise AssertionError("Complete generation cohort is empty or has duplicate identifiers")
    audit = {
        "passed": True, "n_models": len(model_keys), "n_bundles": len(bundle_names),
        "n_conditions": len(conditions), "n_runs": len(run_audit), "n_records": len(frame),
        "models": list(model_keys), "bundles": list(bundle_names),
        "conditions": list(conditions), "runs": run_audit,
        "n_equivalent_duplicate_rows_removed": int(
            sum(
                run["duplicate_repair"]["n_equivalent_duplicate_rows"]
                for run in run_audit
            )
        ),
        "duplicate_repair_backups": [
            run["duplicate_repair"]["backup_path"]
            for run in run_audit
            if run["duplicate_repair"]["backup_path"]
        ],
    }
    return frame, audit


def run_official_chexbert(
    reports: pd.DataFrame,
    *,
    repo: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    text_column: str = "final_report",
    id_column: str | None = "generation_record_id",
) -> pd.DataFrame:
    repository = Path(repo)
    label_script = _chexbert_label_script(repository)
    if label_script is None:
        raise FileNotFoundError(f"Official CheXbert label.py not found below {repository}")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if text_column not in reports:
        raise KeyError(f"Missing CheXbert text column: {text_column}")
    texts = reports[text_column].fillna("").astype(str).map(str.strip)
    if texts.eq("").any():
        raise ValueError(f"CheXbert input contains {int(texts.eq('').sum())} empty reports")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_csv = output / "reports_for_chexbert.csv"
    pd.DataFrame({"Report Impression": texts}).to_csv(input_csv, index=False)
    ids = (
        reports[id_column].astype(str).tolist()
        if id_column and id_column in reports
        else [f"row-{index:08d}" for index in range(len(reports))]
    )
    pd.DataFrame({
        "chexbert_row": range(len(reports)), "record_id": ids,
        "report_sha256": [_sha256_text(value) for value in texts],
    }).to_csv(output / "reports_for_chexbert_manifest.csv", index=False)
    executable = os.environ.get("JAMIA_CHEXBERT_PYTHON", sys.executable)
    runtime_audit = ensure_chexbert_runtime_dependencies(
        python_executable=executable,
        output_dir=output,
    )
    command_environment, import_compatibility_audit = prepare_chexbert_import_compatibility(
        label_script=label_script,
        output_dir=output,
    )
    compatibility_entrypoint = import_compatibility_audit["compatibility_entrypoint"]
    command = [
        executable,
        compatibility_entrypoint,
        f"-d={input_csv}",
        f"-o={output}",
        f"-c={checkpoint_path}",
    ]
    completed = subprocess.run(
        command,
        cwd=str(label_script.parent),
        env=command_environment,
        text=True,
        capture_output=True,
    )
    (output / "chexbert_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "chexbert_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        stderr_tail = completed.stderr[-4000:].strip()
        raise RuntimeError(
            f"CheXbert failed with exit code {completed.returncode}. Full logs are in {output}."
            + (f"\nCheXbert stderr (last 4000 characters):\n{stderr_tail}" if stderr_tail else "")
        )
    labeled_path = output / "labeled_reports.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(f"CheXbert returned success but did not create {labeled_path}")
    labeled = pd.read_csv(labeled_path)
    if len(labeled) != len(reports):
        raise AssertionError("CheXbert output row count changed")
    returned_text = labeled["Report Impression"].fillna("").astype(str).map(str.strip)
    if returned_text.tolist() != texts.tolist():
        raise AssertionError("CheXbert changed or reordered report rows")
    provenance = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "n_reports": len(reports),
        "python_executable": executable, "command": command,
        "runtime_dependency_audit": runtime_audit,
        "import_compatibility_audit": import_compatibility_audit,
        "label_script": str(label_script.resolve()), "label_script_sha256": _sha256_path(label_script),
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": _sha256_path(checkpoint_path),
        "input_csv_sha256": _sha256_path(input_csv),
        "input_manifest_sha256": _sha256_path(output / "reports_for_chexbert_manifest.csv"),
        "labeled_reports_sha256": _sha256_path(labeled_path),
    }
    (output / "chexbert_run_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return labeled


def map_chexbert_states(
    labeled: pd.DataFrame,
    uncertain_policy: Mapping[str, int],
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    policy = {_canonical(str(key)): int(value) for key, value in uncertain_policy.items()}
    if set(policy) != set(LABELS_13) or any(value not in (0, 1) for value in policy.values()):
        raise ValueError("Uncertain-state policy must define a binary value for every frozen label")
    columns = {_canonical(str(column)): column for column in labeled.columns}
    missing = [label for label in CHEXBERT_CONDITIONS_14 if label not in columns]
    if missing:
        raise AssertionError(
            f"CheXbert output is missing required columns: {missing}; columns={list(labeled.columns)}"
        )
    vectors: list[list[int]] = []
    raw_states: list[dict[str, Any]] = []
    for _, row in labeled.iterrows():
        states: dict[str, Any] = {}
        for label in CHEXBERT_CONDITIONS_14:
            value = row[columns[label]]
            state = None if pd.isna(value) else int(value)
            if state not in (1, 0, -1, None):
                raise ValueError(f"Unexpected CheXbert state {state!r} for {label}")
            states[label] = state
        vector = [
            1 if states[label] == 1 else policy[label] if states[label] == -1 else 0
            for label in LABELS_13
        ]
        vectors.append(vector)
        raw_states.append(states)
    return vectors, raw_states


def clinical_sanity_reports() -> pd.DataFrame:
    return pd.DataFrame([
        {"generation_record_id": "SANITY-negation", "final_report": "No pleural effusion or pneumothorax."},
        {"generation_record_id": "SANITY-uncertainty", "final_report": "Possible mild pulmonary edema."},
        {"generation_record_id": "SANITY-resolved", "final_report": "Previously seen pneumothorax has resolved."},
        {"generation_record_id": "SANITY-device", "final_report": "Right PICC tip projects over the lower superior vena cava."},
    ])


def validate_clinical_sanity_outputs(
    raw_states: Sequence[Mapping[str, Any]], vectors: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(raw_states) != 4 or len(vectors) != 4:
        raise ValueError("Expected exactly four frozen CheXbert sanity cases")
    label_index = {label: index for index, label in enumerate(LABELS_13)}
    checks = [
        ("negation", vectors[0][label_index["pleural effusion"]] == 0 and vectors[0][label_index["pneumothorax"]] == 0),
        ("uncertainty", raw_states[1]["edema"] in (-1, 1) and vectors[1][label_index["edema"]] == 1),
        ("resolved", vectors[2][label_index["pneumothorax"]] == 0),
        ("support_device", vectors[3][label_index["support devices"]] == 1),
    ]
    result = {
        "passed": all(passed for _, passed in checks),
        "checks": {name: bool(passed) for name, passed in checks},
        "raw_states": [dict(value) for value in raw_states],
        "binary_vectors": [list(map(int, value)) for value in vectors],
    }
    if not result["passed"]:
        raise AssertionError(f"CheXbert failed frozen clinical sanity cases: {result['checks']}")
    return result


def _cohort_fingerprint(reports: pd.DataFrame) -> str:
    rows = []
    for _, row in reports.sort_values("generation_record_id").iterrows():
        rows.append({
            "generation_record_id": str(row["generation_record_id"]),
            "report_sha256": _sha256_text(str(row["final_report"])),
            "reference_vector": list(map(int, row["reference_vector"])),
        })
    return _sha256_text(json.dumps(rows, sort_keys=True, separators=(",", ":")))


def _stratified_sample(reports: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    work = reports.drop_duplicates("generation_record_id").copy()
    if len(work) < n:
        raise ValueError(f"Need at least {n} generated reports for validation; found {len(work)}")
    work["reference_abnormal"] = work["reference_vector"].map(lambda value: bool(sum(value)))
    strata = ["source_dataset", "model_key", "condition", "reference_abnormal"]
    missing = [column for column in strata if column not in work]
    if missing:
        raise KeyError(f"Cannot construct the prespecified validation strata; missing={missing}")
    groups = {
        tuple(key if isinstance(key, tuple) else (key,)): group
        for key, group in work.groupby(strata, dropna=False, sort=True)
    }
    if len(groups) > n:
        raise ValueError(f"The {n}-report sample cannot cover all {len(groups)} nonempty strata")
    allocation = {key: 1 for key in groups}
    while sum(allocation.values()) < n:
        eligible = [key for key, group in groups.items() if allocation[key] < len(group)]
        if not eligible:
            break
        key = max(eligible, key=lambda item: (len(groups[item]) / (allocation[item] + 1), str(item)))
        allocation[key] += 1
    parts = [
        groups[key].sample(n=allocation[key], random_state=seed + index)
        for index, key in enumerate(sorted(groups, key=str))
    ]
    sample = pd.concat(parts, ignore_index=True)
    if len(sample) != n:
        raise AssertionError(f"Stratified allocation produced {len(sample)} rather than {n} reports")
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


def create_blinded_annotation_materials(
    reports: pd.DataFrame,
    output_dir: str | Path,
    *,
    n: int = 200,
    seed: int = 20260831,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "annotation_sample_manifest.json"
    cohort_fingerprint = _cohort_fingerprint(reports)
    paths = {
        "annotator1_template": output / "human_annotation_annotator1_template.csv",
        "annotator2_template": output / "human_annotation_annotator2_template.csv",
        "crosswalk": output / "human_annotation_crosswalk.csv",
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generation_cohort_fingerprint") != cohort_fingerprint:
            raise RuntimeError(
                "The generation cohort changed after the human-validation sample was frozen. "
                f"Move {output} aside before creating a new blinded sample."
            )
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Frozen annotation sample is incomplete: {missing}")
        for name, path in paths.items():
            expected_hash = manifest.get(f"{name}_sha256")
            if not expected_hash or _sha256_path(path) != expected_hash:
                raise RuntimeError(
                    f"Frozen annotation material changed after sampling: {path}. "
                    "Restore the original file or move the validation directory aside and refreeze."
                )
        return manifest

    sample = _stratified_sample(reports, n=n, seed=seed)
    rng = np.random.default_rng(seed)
    sample["blinded_annotation_id"] = [f"VAL-{value:04d}" for value in rng.permutation(np.arange(1, n + 1))]
    sample["report_sha256"] = sample["final_report"].astype(str).map(_sha256_text)
    crosswalk_columns = [
        "blinded_annotation_id", "generation_record_id", "query_record_id", "source_dataset",
        "bundle", "model_key", "condition", "reference_abnormal", "report_sha256",
    ]
    sample[crosswalk_columns].to_csv(paths["crosswalk"], index=False)
    form = sample[["blinded_annotation_id", "final_report"]].copy()
    for label in LABELS_13:
        form[f"human_{label}"] = ""
    form["annotator_id"] = ""
    form["notes"] = ""
    form.sample(frac=1, random_state=seed + 1).to_csv(paths["annotator1_template"], index=False)
    form.sample(frac=1, random_state=seed + 2).to_csv(paths["annotator2_template"], index=False)
    counts = (
        sample.groupby(["source_dataset", "model_key", "condition", "reference_abnormal"], dropna=False)
        .size().reset_index(name="n").to_dict("records")
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_cohort_fingerprint": cohort_fingerprint, "sample_size": n, "seed": seed,
        "strata": ["source_dataset", "model_key", "condition", "reference_abnormal"],
        "stratum_counts": counts, **{key: str(path) for key, path in paths.items()},
        **{f"{key}_sha256": _sha256_path(path) for key, path in paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _validated_annotation_frame(path: str | Path, prefix: str = "human_") -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path)
    required = ["blinded_annotation_id", "annotator_id", *[f"{prefix}{label}" for label in LABELS_13]]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Annotation file {path} is missing columns: {missing}")
    if frame["blinded_annotation_id"].astype(str).duplicated().any():
        raise AssertionError(f"Duplicate blinded_annotation_id values in {path}")
    annotator_ids = {value.strip() for value in frame["annotator_id"].fillna("").astype(str) if value.strip()}
    if len(annotator_ids) != 1 or frame["annotator_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"Every row in {path} must contain the same nonempty annotator_id")
    for label in LABELS_13:
        column = f"{prefix}{label}"
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.isin([0, 1])
        if invalid.any():
            examples = frame.loc[invalid, "blinded_annotation_id"].astype(str).head(10).tolist()
            raise ValueError(f"{column} must be completely annotated with 0/1 in {path}; examples={examples}")
        frame[column] = numeric.astype(int)
    return frame, next(iter(annotator_ids))


def create_adjudication_template(
    annotator1_csv: str | Path,
    annotator2_csv: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    first, first_id = _validated_annotation_frame(annotator1_csv)
    second, second_id = _validated_annotation_frame(annotator2_csv)
    if first_id == second_id:
        raise ValueError("The two annotation files must have different annotator_id values")
    if set(first["blinded_annotation_id"].astype(str)) != set(second["blinded_annotation_id"].astype(str)):
        raise AssertionError("The two annotators did not label the same blinded report set")
    first = first.set_index("blinded_annotation_id")
    second = second.set_index("blinded_annotation_id")
    if "final_report" in first and "final_report" in second:
        if not first["final_report"].fillna("").astype(str).sort_index().equals(
            second["final_report"].fillna("").astype(str).sort_index()
        ):
            raise AssertionError("The two annotation forms contain different report text")
    rows = []
    disagreement_cells = 0
    for blinded_id in sorted(first.index.astype(str)):
        row = {"blinded_annotation_id": blinded_id, "final_report": str(first.loc[blinded_id].get("final_report", ""))}
        for label in LABELS_13:
            a = int(first.loc[blinded_id, f"human_{label}"])
            b = int(second.loc[blinded_id, f"human_{label}"])
            row[f"annotator1_{label}"] = a
            row[f"annotator2_{label}"] = b
            row[f"adjudicated_{label}"] = a if a == b else ""
            disagreement_cells += int(a != b)
        row["annotator_id"] = ""
        row["notes"] = ""
        rows.append(row)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return {
        "annotator1_id": first_id, "annotator2_id": second_id,
        "n_reports": len(rows), "n_disagreement_cells": disagreement_cells,
        "template": str(output),
    }


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float | None:
    observed = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else float((observed - expected) / (1 - expected))


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _classification_validation(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    f1s: list[float] = []
    sensitivities: list[float] = []
    specificities: list[float] = []
    for index, label in enumerate(LABELS_13):
        truth, prediction = y_true[:, index], y_pred[:, index]
        tp = int(((truth == 1) & (prediction == 1)).sum())
        tn = int(((truth == 0) & (prediction == 0)).sum())
        fp = int(((truth == 0) & (prediction == 1)).sum())
        fn = int(((truth == 1) & (prediction == 0)).sum())
        sensitivity, specificity = _safe_ratio(tp, tp + fn), _safe_ratio(tn, tn + fp)
        precision, f1 = _safe_ratio(tp, tp + fp), _safe_ratio(2 * tp, 2 * tp + fp + fn)
        per_label[label] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": sensitivity, "specificity": specificity,
            "precision": precision, "f1": f1,
            "human_positive": int((truth == 1).sum()), "human_negative": int((truth == 0).sum()),
        }
        if f1 is not None: f1s.append(f1)
        if sensitivity is not None: sensitivities.append(sensitivity)
        if specificity is not None: specificities.append(specificity)
    return {
        "n": int(len(y_true)), "macro_f1": float(np.mean(f1s)) if f1s else None,
        "macro_sensitivity": float(np.mean(sensitivities)) if sensitivities else None,
        "macro_specificity": float(np.mean(specificities)) if specificities else None,
        "n_f1_evaluable_labels": len(f1s), "per_label": per_label,
    }


def validate_two_annotators(
    predicted: pd.DataFrame,
    *,
    annotator1_csv: str | Path,
    annotator2_csv: str | Path,
    crosswalk_csv: str | Path,
    adjudicated_csv: str | Path | None = None,
) -> dict[str, Any]:
    first, first_id = _validated_annotation_frame(annotator1_csv)
    second, second_id = _validated_annotation_frame(annotator2_csv)
    if first_id == second_id:
        raise ValueError("The two annotation files must have different annotator_id values")
    crosswalk = pd.read_csv(crosswalk_csv)
    required_crosswalk = {"blinded_annotation_id", "generation_record_id"}
    missing_crosswalk = sorted(required_crosswalk - set(crosswalk.columns))
    if missing_crosswalk:
        raise KeyError(f"Blinded crosswalk is missing columns: {missing_crosswalk}")
    if crosswalk["blinded_annotation_id"].astype(str).duplicated().any():
        raise AssertionError("Blinded crosswalk has duplicate blinded_annotation_id values")
    if crosswalk["generation_record_id"].astype(str).duplicated().any():
        raise AssertionError("Blinded crosswalk has duplicate generation_record_id values")
    expected_ids = set(crosswalk["blinded_annotation_id"].astype(str))
    for name, frame in (("annotator1", first), ("annotator2", second)):
        if set(frame["blinded_annotation_id"].astype(str)) != expected_ids:
            raise AssertionError(f"{name} IDs differ from the frozen blinded crosswalk")
        if "final_report" in frame and "report_sha256" in crosswalk:
            text_hashes = frame.assign(
                report_sha256=frame["final_report"].fillna("").astype(str).map(_sha256_text)
            )[["blinded_annotation_id", "report_sha256"]]
            checked = crosswalk[["blinded_annotation_id", "report_sha256"]].merge(
                text_hashes, on="blinded_annotation_id", suffixes=("_expected", "_observed"),
                validate="one_to_one",
            )
            if not checked["report_sha256_expected"].eq(checked["report_sha256_observed"]).all():
                raise AssertionError(f"{name} report text differs from the frozen crosswalk")
    ordered_ids = sorted(expected_ids)
    first = first.set_index("blinded_annotation_id").loc[ordered_ids]
    second = second.set_index("blinded_annotation_id").loc[ordered_ids]
    a = first[[f"human_{label}" for label in LABELS_13]].to_numpy(dtype=int)
    b = second[[f"human_{label}" for label in LABELS_13]].to_numpy(dtype=int)
    disagreement = a != b
    interrater_per_label, kappas = {}, []
    for index, label in enumerate(LABELS_13):
        kappa = _cohen_kappa(a[:, index], b[:, index])
        interrater_per_label[label] = {
            "agreement": float((a[:, index] == b[:, index]).mean()),
            "cohen_kappa": kappa, "n_disagreements": int(disagreement[:, index].sum()),
        }
        if kappa is not None: kappas.append(kappa)

    consensus = a.copy()
    if disagreement.any():
        if not adjudicated_csv or not Path(adjudicated_csv).exists():
            raise FileNotFoundError(f"{int(disagreement.sum())} label disagreements require a completed adjudication file")
        adjudicated, adjudicator_id = _validated_annotation_frame(adjudicated_csv, prefix="adjudicated_")
        if set(adjudicated["blinded_annotation_id"].astype(str)) != expected_ids:
            raise AssertionError("Adjudication IDs differ from the frozen blinded crosswalk")
        adjudicated = adjudicated.set_index("blinded_annotation_id").loc[ordered_ids]
        consensus = adjudicated[[f"adjudicated_{label}" for label in LABELS_13]].to_numpy(dtype=int)
        if ((consensus != a) & ~disagreement).any():
            raise AssertionError("Adjudication changed labels on which both annotators agreed")
        if adjudicator_id in {first_id, second_id}:
            raise ValueError("The adjudicator_id must differ from both annotator IDs")
    else:
        adjudicator_id = None

    crosswalk["blinded_annotation_id"] = crosswalk["blinded_annotation_id"].astype(str)
    consensus_frame = pd.DataFrame(consensus, columns=[f"human_{label}" for label in LABELS_13])
    consensus_frame.insert(0, "blinded_annotation_id", ordered_ids)
    linked = crosswalk[["blinded_annotation_id", "generation_record_id"]].merge(
        consensus_frame, on="blinded_annotation_id", validate="one_to_one"
    ).merge(
        predicted[["generation_record_id", "prediction_vector"]],
        on="generation_record_id", validate="one_to_one",
    )
    if len(linked) != len(expected_ids):
        raise AssertionError("Some adjudicated reports lack automatic predictions")
    y_true = linked[[f"human_{label}" for label in LABELS_13]].to_numpy(dtype=int)
    raw_predictions = linked["prediction_vector"].tolist()
    if any(not isinstance(value, Sequence) or isinstance(value, (str, bytes)) for value in raw_predictions):
        raise ValueError("Every automatic prediction_vector must be a 13-element binary sequence")
    if any(len(value) != len(LABELS_13) for value in raw_predictions):
        raise ValueError("Every automatic prediction_vector must contain exactly 13 labels")
    y_pred = np.asarray(raw_predictions, dtype=int)
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Automatic prediction vectors must contain only 0/1")
    result = _classification_validation(y_true, y_pred)
    result.update({
        "annotator1_id": first_id, "annotator2_id": second_id, "adjudicator_id": adjudicator_id,
        "interrater": {
            "overall_cell_agreement": float((a == b).mean()),
            "macro_cohen_kappa": float(np.mean(kappas)) if kappas else None,
            "n_disagreement_cells": int(disagreement.sum()), "per_label": interrater_per_label,
        },
    })
    return result


def create_annotation_template(
    reports: pd.DataFrame, output_csv: str | Path, *, n: int = 200, seed: int = 20260831,
) -> Path:
    """Compatibility wrapper; new workflows should use two blinded forms."""
    output_csv = Path(output_csv)
    manifest = create_blinded_annotation_materials(reports, output_csv.parent, n=n, seed=seed)
    source = Path(manifest["annotator1_template"])
    if source != output_csv and not output_csv.exists():
        output_csv.write_bytes(source.read_bytes())
    return output_csv


def validate_against_human(
    predicted: pd.DataFrame, human_csv: str | Path, *, id_column: str = "generation_record_id",
) -> dict[str, Any]:
    """Compatibility validator; Notebook 06 uses the two-annotator workflow."""
    human = pd.read_csv(human_csv)
    if id_column not in human:
        raise KeyError(f"Legacy annotation file is missing {id_column}")
    merged = human.merge(predicted[[id_column, "prediction_vector"]], on=id_column, validate="one_to_one")
    if len(merged) != len(human):
        raise AssertionError("Some human-annotated reports lack automatic labels")
    values = []
    for label in LABELS_13:
        numeric = pd.to_numeric(merged[f"human_{label}"], errors="coerce")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError(f"human_{label} must contain only complete binary annotations")
        values.append(numeric.astype(int).to_numpy())
    return _classification_validation(
        np.stack(values, axis=1), np.asarray(merged["prediction_vector"].tolist(), dtype=int)
    )
