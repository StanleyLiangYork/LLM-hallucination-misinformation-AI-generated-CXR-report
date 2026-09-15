from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def add_leakage_code_to_path(project_dir: Path) -> Path:
    """Locate leakage helpers without exposing ``rerun_code`` as top-level modules.

    Adding ``rerun_code/`` itself to ``sys.path`` makes its ``statistics.py``
    shadow Python's standard-library ``statistics`` module. Transformers and
    its optional dependencies can import that standard module during lazy
    initialization, producing a misleading relative-import failure. The
    packaged rerun helpers are therefore exposed through the project root and
    must be imported as ``rerun_code.<module>``.
    """
    package_dir = project_dir / "rerun_code"
    if (package_dir / "leakage_safe_data.py").exists():
        if str(project_dir) not in sys.path:
            sys.path.insert(0, str(project_dir))
        return package_dir

    # Backward-compatible support for a separately copied legacy helper folder.
    # Only that legacy folder is placed on sys.path because it is not a package.
    for path in (
        project_dir.parent / "revision" / "experiment_code",
        project_dir / "experiment_code",
    ):
        if (path / "leakage_safe_data.py").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path
    raise FileNotFoundError(
        "The shared leakage-safe modules were not found below rerun_code."
    )


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    return result


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return pd.DataFrame(read_jsonl(source))
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            for key in ("records", "data", "metadata", "items"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError(f"JSON table must contain a list: {source}")
        return pd.DataFrame(value)
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(source, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported table type: {source}")


def stable_record_key(row: Mapping[str, Any]) -> str:
    for key in ("record_id", "image_sha256", "image_path", "image_id", "uid", "study_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return f"{key}:{value}"
    raise ValueError("Record has no stable identifier")
