from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def find_project_dir(start: str | Path | None = None) -> Path:
    override = os.environ.get("JAMIA_RERUN_DIR")
    candidates = [Path(override)] if override else []
    here = Path(start or Path.cwd()).resolve()
    candidates += [here, *here.parents]
    for candidate in candidates:
        direct = candidate / "rerun_config.json"
        nested = candidate / "jamia" / "re_run" / "rerun_config.json"
        if direct.exists():
            return direct.parent.resolve()
        if nested.exists():
            return nested.parent.resolve()
    raise FileNotFoundError(
        "Set JAMIA_RERUN_DIR to the folder containing rerun_config.json, "
        "or start from that folder."
    )


def load_config(project_dir: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    root = Path(project_dir).resolve() if project_dir else find_project_dir()
    path = root / "rerun_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    override = os.environ.get("JAMIA_OUTPUT_ROOT")
    if override:
        config["output_root"] = override
    return root, config


def output_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["output_root"])
    names = {
        "root": root,
        "manifests": root / "manifests",
        "bundles": root / "bundles",
        "verifier_data": root / "verifier_data",
        "verifiers": root / "verifiers",
        "labeler": root / "labeler",
        "generation": root / "generation",
        "labels": root / "labels",
        "metrics": root / "metrics",
        "statistics": root / "statistics",
        "analysis": root / "analysis",
        "logs": root / "logs",
    }
    for path in names.values():
        path.mkdir(parents=True, exist_ok=True)
    return names


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_version(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=20).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def environment_manifest(project_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    adapter_audit: dict[str, Any] = {}
    for key, spec in config["models"].items():
        adapter = Path(spec["legacy_adapter"])
        adapter_config = adapter / "adapter_config.json"
        payload: dict[str, Any] = {
            "requested_display_name": spec["display_name"],
            "legacy_adapter": str(adapter),
            "exists": adapter.exists(),
            "adapter_config_exists": adapter_config.exists(),
            "fallback_base_model": spec["fallback_base_model"],
            "training_base_model": spec.get("training_base_model", spec["fallback_base_model"]),
        }
        if adapter_config.exists():
            parsed = json.loads(adapter_config.read_text(encoding="utf-8"))
            actual = str(parsed.get("base_model_name_or_path", ""))
            payload.update({
                "adapter_config_sha256": sha256_path(adapter_config),
                "actual_base_model": actual,
                "lora_r": parsed.get("r"),
                "lora_alpha": parsed.get("lora_alpha"),
                "lora_dropout": parsed.get("lora_dropout"),
                "target_modules": parsed.get("target_modules"),
            })
            required = spec.get("require_adapter_base_contains")
            payload["identity_gate_passed"] = not required or required.lower() in actual.lower()
        else:
            payload["actual_base_model"] = ""
            payload["identity_gate_passed"] = False
        adapter_audit[key] = payload
    config_path = project_dir / "rerun_config.json"
    return {
        "schema_version": config["schema_version"],
        "config_sha256": sha256_path(config_path),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _command_version(["git", "-C", str(project_dir), "rev-parse", "HEAD"]),
        "pip_freeze": _command_version([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi": _command_version(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "adapter_audit": adapter_audit,
    }
