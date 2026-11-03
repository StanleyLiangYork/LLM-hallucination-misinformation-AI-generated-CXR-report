from __future__ import annotations

import contextlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Mapping, Sequence

from PIL import Image

from .modeling import parse_verdict
from .verifier_data import verifier_prompt


CONDITIONS = ("A_single_pass", "B_unconditional_4pass", "C_pretrained_gate", "D_corrected_lora_gate")
GENERATION_API_VERSION = 5
GENERATION_RESUME_REPAIR_VERSION = 1
PHI4_IMAGE_COMPAT_VERSION = 3
PHI4_CACHE_COMPAT_VERSION = 1
PHI4_VERIFIER_COMPAT_VERSION = 1
PHI4_EMPTY_OUTPUT_RECOVERY_VERSION = 1
PHI4_EMPTY_RETRY_MIN_NEW_TOKENS = 32
PHI4_EMPTY_RETRY_MAX_PROMPT_CHARS = 8000

PHI4_SYSTEM = (
    "You are a radiology assistant for chest X-ray interpretation. "
    "Use the query image and retrieved reference evidence to generate a concise final chest X-ray report. "
    "Output only the final report text. Do not output JSON, bullets, markdown, or extra commentary."
)


def _install_phi4_legacy_numpy_patch_helpers(siglip2_module=None) -> bool:
    """Restore the NumPy SigLIP2 helpers expected by Phi-4 remote code.

    Current Transformers exposes torch-only helpers under the same names. The
    Microsoft processor surrounding those helpers still prepares HWC NumPy
    arrays and later asks BatchFeature to convert the completed arrays to torch.
    """
    if siglip2_module is None:
        import transformers.models.siglip2.image_processing_siglip2 as siglip2_module
    if int(getattr(siglip2_module, "_jamia_phi4_numpy_patch_helpers_version", 0)) >= 3:
        return False

    import numpy as np

    def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise TypeError(
                "Phi-4 legacy patch conversion requires a three-dimensional NumPy HWC image"
            )
        height, width, channels = image.shape
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"Phi-4 image shape {image.shape} is not divisible by patch_size={patch_size}"
            )
        rows, columns = height // patch_size, width // patch_size
        patches = image.reshape(rows, patch_size, columns, patch_size, channels)
        return patches.transpose(0, 2, 1, 3, 4).reshape(rows * columns, -1)

    def pad_along_first_dim(
        array: np.ndarray, target_length: int, pad_value: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(array, np.ndarray):
            raise TypeError("Phi-4 legacy patch padding requires a NumPy array")
        padding = target_length - array.shape[0]
        if padding < 0:
            raise ValueError(
                f"Phi-4 patch count {array.shape[0]} exceeds target_length={target_length}"
            )
        mask = np.ones((target_length,), dtype=np.int32)
        if padding:
            pads = [(0, padding)] + [(0, 0)] * (array.ndim - 1)
            array = np.pad(array, pads, mode="constant", constant_values=pad_value)
            mask[-padding:] = 0
        return array, mask

    siglip2_module.convert_image_to_patches = convert_image_to_patches
    siglip2_module.pad_along_first_dim = pad_along_first_dim
    siglip2_module._jamia_phi4_numpy_patch_helpers_version = 3
    return True


def _install_phi4_numpy_normalize_compatibility(processor, numpy_normalize=None) -> bool:
    """Make Phi-4's legacy NumPy image path work with newer fast processors.

    Microsoft's remote Phi-4 processor converts each PIL image to a NumPy
    array before calling ``image_processor.normalize``. Some newer
    Transformers releases bind that method to the torchvision fast backend,
    which accepts tensors but rejects NumPy arrays. Patch only this processor
    instance and route only NumPy inputs to Transformers' NumPy implementation.
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise AttributeError("Phi-4 processor has no image_processor")
    if getattr(image_processor, "_jamia_phi4_numpy_normalize_compatibility", False):
        return False

    if numpy_normalize is None:
        from transformers.image_transforms import normalize as numpy_normalize
    original_normalize = image_processor.normalize

    def normalize_compatibility(
        image,
        mean,
        std,
        data_format=None,
        input_data_format=None,
        **kwargs,
    ):
        import numpy as np

        if isinstance(image, np.ndarray):
            # Phi-4's remote preprocessing loop intentionally uses HWC NumPy
            # arrays. Its legacy patch helper consumes HWC before BatchFeature
            # converts the batched output to torch.
            output_data_format = data_format
            try:
                normalized = numpy_normalize(
                    image=image,
                    mean=mean,
                    std=std,
                    data_format=output_data_format,
                    input_data_format=input_data_format,
                )
            except TypeError as exc:
                # Older Transformers releases infer the input layout and do
                # not expose input_data_format on image_transforms.normalize.
                if "input_data_format" not in str(exc):
                    raise
                normalized = numpy_normalize(
                    image=image,
                    mean=mean,
                    std=std,
                    data_format=output_data_format,
                )
            if np.issubdtype(image.dtype, np.floating) and normalized.dtype != image.dtype:
                normalized = normalized.astype(image.dtype, copy=False)
            return normalized
        return original_normalize(
            image=image,
            mean=mean,
            std=std,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    image_processor.normalize = normalize_compatibility
    image_processor._jamia_phi4_numpy_normalize_compatibility = True
    return True


def _install_phi4_dynamic_cache_compatibility(model) -> tuple[str, ...]:
    """Teach Microsoft's Phi-4 multimodal preparation about DynamicCache.

    The remote model indexes ``past_key_values`` as a nested tuple when it
    extends the attention mask. Current Phi-3 creates ``DynamicCache`` and
    exposes its length through ``get_seq_length``. Patch only the concrete
    Phi-4 model instance inside any PEFT wrappers and preserve the original
    method for first-pass image encoding and legacy cache objects.
    """
    if model is None:
        return ()

    candidates: list[Any] = []
    seen: set[int] = set()
    queue: list[tuple[Any, int]] = [(model, 0)]
    while queue:
        candidate, depth = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        candidates.append(candidate)
        if depth >= 3:
            continue
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append((get_base_model(), depth + 1))
            except Exception:
                pass
        for attribute in ("base_model", "model", "module"):
            try:
                nested = getattr(candidate, attribute, None)
            except Exception:
                nested = None
            if nested is not None and nested is not candidate:
                queue.append((nested, depth + 1))

    patched: list[str] = []
    for candidate in candidates:
        cls = type(candidate)
        identity = f"{cls.__module__}.{cls.__name__}"
        phi_identity = identity.lower()
        if "phi4" not in phi_identity and "phi_4" not in phi_identity:
            continue
        original_prepare = getattr(candidate, "prepare_inputs_labels_for_multimodal", None)
        if not callable(original_prepare):
            continue
        if not getattr(candidate, "_jamia_phi4_dynamic_cache_compatibility", False):
            def prepare_inputs_labels_for_multimodal_compatibility(
                self,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                *,
                _original_prepare=original_prepare,
            ):
                get_seq_length = getattr(past_key_values, "get_seq_length", None)
                vision_tower = self.get_vision_tower()
                if (
                    past_key_values is not None
                    and callable(get_seq_length)
                    and vision_tower is not None
                    and images is not None
                    and input_ids.shape[1] == 1
                ):
                    import torch

                    target_shape = int(get_seq_length()) + 1
                    missing = target_shape - attention_mask.shape[1]
                    if missing > 0:
                        attention_mask = torch.cat(
                            (
                                attention_mask,
                                torch.ones(
                                    (attention_mask.shape[0], missing),
                                    dtype=attention_mask.dtype,
                                    device=attention_mask.device,
                                ),
                            ),
                            dim=1,
                        )
                    elif missing < 0:
                        raise RuntimeError(
                            "Phi-4 DynamicCache length is shorter than the existing attention mask: "
                            f"cache_target={target_shape}, attention_length={attention_mask.shape[1]}"
                        )
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                    return (
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        None,
                        labels,
                    )
                return _original_prepare(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                )

            candidate.prepare_inputs_labels_for_multimodal = MethodType(
                prepare_inputs_labels_for_multimodal_compatibility, candidate
            )
            candidate._jamia_phi4_dynamic_cache_compatibility = True
        patched.append(identity)

    if not patched:
        identities = [f"{type(item).__module__}.{type(item).__name__}" for item in candidates]
        raise RuntimeError(
            "Could not locate the concrete Phi-4 generation model inside the loaded wrappers. "
            f"Inspected={identities}"
        )
    return tuple(dict.fromkeys(patched))


def _chat_template_model_inputs(encoded, device) -> dict[str, Any]:
    """Normalize chat-template tensor and BatchEncoding return formats."""
    import torch

    if isinstance(encoded, Mapping):
        model_inputs = {
            key: value
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
    elif torch.is_tensor(encoded):
        model_inputs = {"input_ids": encoded}
    elif hasattr(encoded, "input_ids"):
        model_inputs = {"input_ids": encoded.input_ids}
        attention_mask = getattr(encoded, "attention_mask", None)
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
    else:
        raise TypeError(
            "Unsupported tokenizer.apply_chat_template return type: "
            f"{type(encoded).__module__}.{type(encoded).__name__}"
        )
    if "input_ids" not in model_inputs:
        raise KeyError("Chat-template encoding does not contain input_ids")
    for key, value in tuple(model_inputs.items()):
        if torch.is_tensor(value):
            if value.ndim == 1:
                value = value.unsqueeze(0)
            model_inputs[key] = value.to(device)
    if not torch.is_tensor(model_inputs["input_ids"]):
        raise TypeError(
            "Chat-template input_ids must be a tensor after normalization, found "
            f"{type(model_inputs['input_ids']).__name__}"
        )
    return model_inputs


def validated_completed_generation_ids(
    results_path: str | Path,
    *,
    model_key: str,
    bundle_name: str,
    run_id: str,
    valid_query_ids: Sequence[str],
    conditions: Sequence[str] = CONDITIONS,
    repair_failed_records: bool = False,
) -> tuple[set[str], dict[str, Any]]:
    """Return audited completed IDs from an existing model/bundle result file.

    Structural/provenance contamination remains a hard error. When explicitly
    enabled, rows whose identity is valid but whose final report is empty or
    marked failed are removed after the original JSONL is copied to a timestamped
    backup. Those generation IDs then remain pending and can be regenerated.
    """
    path = Path(results_path)
    query_ids = {str(value) for value in valid_query_ids}
    allowed_conditions = {str(value) for value in conditions}
    if not path.exists():
        return set(), {
            "results_file": str(path), "n_existing_records": 0,
            "model_key": model_key, "bundle": bundle_name, "run_id": run_id,
            "counts_by_condition": {condition: 0 for condition in conditions},
            "n_failed_records_removed": 0,
            "failed_generation_ids_removed": [],
            "failed_records_backup": None,
        }

    completed: set[str] = set()
    counts = {condition: 0 for condition in conditions}
    valid_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            stored_model = str(row.get("model_key") or "")
            stored_bundle = str(row.get("bundle") or "")
            stored_run = str(row.get("run_id") or "")
            query_id = str(row.get("query_record_id") or "")
            condition = str(row.get("condition") or "")
            generation_id = str(row.get("generation_record_id") or "")
            expected_id = f"{model_key}|{bundle_name}|{query_id}|{condition}"
            if stored_model != model_key:
                raise RuntimeError(
                    f"Cross-model record in {path}:{line_number}. Expected model_key={model_key!r}, "
                    f"found {stored_model!r}. Do not mix model outputs."
                )
            if stored_bundle != bundle_name:
                raise RuntimeError(
                    f"Cross-bundle record in {path}:{line_number}. Expected bundle={bundle_name!r}, "
                    f"found {stored_bundle!r}."
                )
            if stored_run != run_id:
                raise RuntimeError(
                    f"Stale run_id in {path}:{line_number}. Existing={stored_run!r}, current={run_id!r}."
                )
            if query_id not in query_ids:
                raise RuntimeError(f"Unknown query_record_id in {path}:{line_number}: {query_id!r}")
            if condition not in allowed_conditions:
                raise RuntimeError(f"Unknown condition in {path}:{line_number}: {condition!r}")
            if generation_id != expected_id:
                raise RuntimeError(
                    f"Generation ID mismatch in {path}:{line_number}. Expected {expected_id!r}, "
                    f"found {generation_id!r}."
                )
            report = str(row.get("final_report") or "").strip()
            if not report or row.get("empty_output") is True:
                failed_rows.append({
                    "line_number": line_number,
                    "generation_record_id": generation_id,
                    "reason": "empty_final_report" if not report else "empty_output_flag",
                })
                continue
            if generation_id in completed:
                raise RuntimeError(f"Duplicate generation_record_id in {path}: {generation_id}")
            completed.add(generation_id)
            counts[condition] += 1
            valid_rows.append(row)

    failed_backup = None
    if failed_rows:
        first = failed_rows[0]
        if not repair_failed_records:
            raise RuntimeError(
                f"Existing record {first['generation_record_id']!r} has an empty/failed report and "
                "cannot be treated as complete. Enable audited failed-record repair or move the "
                "model/bundle output aside before rerunning."
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        failed_backup = path.with_name(f"{path.name}.failed-records-{stamp}.bak")
        temporary = path.with_name(f".{path.name}.repair-{stamp}.tmp")
        shutil.copy2(path, failed_backup)
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for row in valid_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    return completed, {
        "results_file": str(path), "n_existing_records": len(completed),
        "model_key": model_key, "bundle": bundle_name, "run_id": run_id,
        "counts_by_condition": counts,
        "n_failed_records_removed": len(failed_rows),
        "failed_generation_ids_removed": [
            row["generation_record_id"] for row in failed_rows
        ],
        "failed_records_backup": str(failed_backup) if failed_backup else None,
    }


def retrieval_context(neighbors: Sequence[Mapping[str, Any]], max_neighbors: int = 10) -> str:
    blocks = []
    for rank, neighbor in enumerate(neighbors[:max_neighbors], start=1):
        report = str(neighbor.get("report_text") or neighbor.get("report") or neighbor.get("caption") or "").strip()
        blocks.append(f"REFERENCE {rank} (similarity={float(neighbor.get('cosine_score', 0.0)):.6f}):\n{report}")
    return "\n\n".join(blocks)


def initial_prompt(neighbors: Sequence[Mapping[str, Any]]) -> str:
    return (
        "Write a concise FINDINGS and IMPRESSION chest radiograph report for the query image. "
        "Use the retrieved reports only as context; do not copy unsupported details. Do not mention retrieval.\n\n"
        + retrieval_context(neighbors)
        + "\n\nGENERATED REPORT:"
    )


def revision_prompt(report: str, neighbors: Sequence[Mapping[str, Any]], pass_number: int) -> str:
    return (
        f"Revision pass {pass_number}. Recheck the candidate against the query image and retrieved context. "
        "Correct unsupported findings, contradictions, and important omissions. Return only the revised report.\n\n"
        f"CANDIDATE:\n{report}\n\n{retrieval_context(neighbors)}\n\nREVISED REPORT:"
    )


def _phi4_empty_output_recovery_prompt(prompt: str, max_chars: int) -> str:
    """Compact a failed prompt while preserving its beginning and final request."""
    if max_chars < 1000:
        raise ValueError("phi4_empty_retry_max_prompt_chars must be at least 1000")
    instruction = (
        "\n\nThe previous decoding attempt returned no visible text. A report is required. "
        "Do not end the response immediately. Start with FINDINGS and provide at least one "
        "complete FINDINGS sentence followed by one IMPRESSION sentence."
    )
    available = max_chars - len(instruction)
    source = str(prompt)
    if len(source) > available:
        omission = "\n\n[Retrieved context shortened for empty-output recovery]\n\n"
        payload = max(1, available - len(omission))
        head = min(payload // 3, 2500)
        tail = payload - head
        source = source[:head] + omission + source[-tail:]
    return source + instruction


def _generate_report_with_empty_recovery(
    runner,
    prompt: str,
    *,
    image_path: str,
    adapter_enabled: bool = False,
) -> tuple[str, float, dict[str, Any]]:
    """Run one logical report pass and recover one empty Phi-4 decoding."""
    report, seconds = runner.generate(
        prompt, image_path=image_path, adapter_enabled=adapter_enabled
    )
    audit: dict[str, Any] = {
        "recovery_used": False,
        "attempt_count": 1,
        "attempt_seconds": [float(seconds)],
        "initial_output_empty": not bool(str(report).strip()),
    }
    if str(report).strip() or not getattr(runner, "is_phi4", False):
        return report, float(seconds), audit

    maximum = PHI4_EMPTY_RETRY_MAX_PROMPT_CHARS
    minimum_tokens = PHI4_EMPTY_RETRY_MIN_NEW_TOKENS
    recovery_prompt = _phi4_empty_output_recovery_prompt(prompt, maximum)
    recovered, retry_seconds = runner.generate(
        recovery_prompt,
        image_path=image_path,
        adapter_enabled=adapter_enabled,
        min_new_tokens=minimum_tokens,
    )
    audit.update(
        {
            "recovery_used": True,
            "attempt_count": 2,
            "attempt_seconds": [float(seconds), float(retry_seconds)],
            "retry_output_empty": not bool(str(recovered).strip()),
            "retry_min_new_tokens": minimum_tokens,
            "retry_max_prompt_chars": maximum,
            "retry_prompt": recovery_prompt,
        }
    )
    return recovered, float(seconds + retry_seconds), audit


class ModelRunner:
    def __init__(self, processor, tokenizer, model, architecture: str, generation_config: Mapping[str, Any], actual_base: str = ""):
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model
        self.architecture = architecture
        self.config = dict(generation_config)
        self.actual_base = str(actual_base)
        self.is_phi4 = "phi-4-reasoning-vision" in self.actual_base.lower()
        self.phi4_numpy_normalize_compatibility = False
        self.phi4_dynamic_cache_targets: tuple[str, ...] = ()
        if self.is_phi4 and self.architecture == "multimodal":
            _install_phi4_legacy_numpy_patch_helpers()
            self.phi4_numpy_normalize_compatibility = _install_phi4_numpy_normalize_compatibility(
                self.processor
            )
            if self.model is not None:
                self.phi4_dynamic_cache_targets = _install_phi4_dynamic_cache_compatibility(
                    self.model
                )

    def _adapter_context(self, enabled: bool):
        if enabled:
            return contextlib.nullcontext()
        method = getattr(self.model, "disable_adapter", None)
        return method() if callable(method) else contextlib.nullcontext()

    def _verifier_image_path(self, image_path: str | None) -> str | None:
        # The audited Phi-4 verifier notebook and corrected harmony training are
        # text-only. Images remain enabled for Phi-4 report generation passes.
        if self.is_phi4:
            return None
        return image_path if self.architecture == "multimodal" else None

    def _format_prompt(self, prompt: str, image_path: str | None, *, verifier: bool = False) -> str:
        if self.is_phi4:
            messages = [{"role": "user", "content": prompt}]
            if not verifier:
                messages.insert(0, {"role": "system", "content": PHI4_SYSTEM})
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            rendered = str(rendered)
            if not verifier:
                rendered += "<|dummy_84|>"
            if image_path and not any(
                marker in rendered
                for marker in ("<image>", "<|image_1|>", "<|image|>", "<image_1>", "<start_of_image>")
            ):
                rendered = "<image>\n" + rendered
            return rendered

        template_owner = self.processor if self.architecture == "multimodal" else self.tokenizer
        if getattr(self.tokenizer, "_jamia_chat_template_policy", "auto") == "plain":
            rendered = prompt
            method = None
        else:
            method = getattr(template_owner, "apply_chat_template", None)
        if callable(method):
            try:
                rendered = method(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered = prompt
        else:
            rendered = prompt
        if self.architecture == "multimodal" and image_path:
            lower = self.actual_base.lower()
            markers = ("<start_of_image>", "<image>", "<|image_1|>", "<|image|>")
            if not any(marker in rendered for marker in markers):
                rendered = ("<start_of_image>\n" if "gemma" in lower else "<image>\n") + rendered
        return rendered

    def generate(
        self,
        prompt: str,
        *,
        image_path: str | None = None,
        adapter_enabled: bool = False,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
        verifier: bool = False,
    ) -> tuple[str, float]:
        import torch
        started = time.perf_counter()
        image = None
        if self.architecture == "multimodal" and image_path:
            image = Image.open(image_path).convert("RGB")
        prompt = self._format_prompt(prompt, image_path, verifier=verifier)
        with self._adapter_context(adapter_enabled):
            if self.architecture == "multimodal" and image is not None:
                if self.is_phi4:
                    inputs = self.processor(
                        text=prompt,
                        images=[image],
                        return_tensors="pt",
                        truncation=True,
                        max_length=4096,
                    )
                else:
                    try:
                        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                    except TypeError:
                        inputs = self.processor(prompt, image, return_tensors="pt")
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            input_length = inputs["input_ids"].shape[1]
            kwargs = {
                "max_new_tokens": int(max_new_tokens or self.config["max_new_tokens"]),
                "do_sample": bool(self.config.get("do_sample", False)),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if kwargs["do_sample"]:
                kwargs["temperature"] = float(self.config.get("temperature", 1.0))
            if min_new_tokens is not None:
                kwargs["min_new_tokens"] = int(min_new_tokens)
            if self.is_phi4:
                # The Phi-only preparation shim supports DynamicCache while
                # retaining caching for efficient autoregressive decoding.
                kwargs["use_cache"] = True
            with torch.inference_mode():
                output = self.model.generate(**inputs, **kwargs)
            text = self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
        if image is not None:
            image.close()
        return text, time.perf_counter() - started

    def _verify_phi4(self, prompt: str, *, adapter_enabled: bool) -> dict[str, Any]:
        """Match the audited reference notebook's text-only TRUE/FALSE scorer."""
        import torch

        started = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        with self._adapter_context(adapter_enabled):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
            )
            device = next(self.model.parameters()).device
            model_inputs = _chat_template_model_inputs(encoded, device)
            input_ids = model_inputs["input_ids"]
            true_ids = self.tokenizer.encode("TRUE", add_special_tokens=False)
            false_ids = self.tokenizer.encode("FALSE", add_special_tokens=False)
            if len(true_ids) == 1 and len(false_ids) == 1:
                with torch.inference_mode():
                    next_logits = self.model(**model_inputs).logits[:, -1, :]
                scores = torch.stack(
                    [next_logits[0, true_ids[0]], next_logits[0, false_ids[0]]], dim=0
                )
                probabilities = torch.softmax(scores, dim=0)
                p_true = float(probabilities[0].item())
                p_false = float(probabilities[1].item())
                raw = "TRUE" if p_true >= p_false else "FALSE"
            else:
                with torch.inference_mode():
                    output = self.model.generate(
                        **model_inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                raw = self.tokenizer.decode(
                    output[0, input_ids.shape[1]:], skip_special_tokens=True
                ).strip()
                p_true = 1.0 if parse_verdict(raw) is True else 0.0
                p_false = 1.0 if parse_verdict(raw) is False else 0.0
        return {
            "raw": raw,
            "verdict": parse_verdict(raw),
            "p_true": p_true,
            "p_false": p_false,
            "seconds": time.perf_counter() - started,
        }

    def verify(self, report: str, *, adapter_enabled: bool, image_path: str | None = None, evidence: str | None = None) -> dict[str, Any]:
        prompt = verifier_prompt(report, evidence=evidence)
        if self.is_phi4:
            return self._verify_phi4(prompt, adapter_enabled=adapter_enabled)
        raw, seconds = self.generate(
            prompt,
            image_path=self._verifier_image_path(image_path),
            adapter_enabled=adapter_enabled,
            max_new_tokens=int(self.config.get("verifier_max_new_tokens", 8)),
            verifier=True,
        )
        return {"raw": raw, "verdict": parse_verdict(raw), "seconds": seconds}


def run_condition(
    runner: ModelRunner,
    condition: str,
    query: Mapping[str, Any],
    neighbors: Sequence[Mapping[str, Any]],
    *,
    max_revision_passes: int = 4,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    image_path = str(query.get("image_path", "") or "")
    prompts: list[str] = []
    reports: list[str] = []
    timings: list[float] = []
    recovery_audits: list[dict[str, Any]] = []
    verifier_outputs: list[dict[str, Any]] = []
    first_prompt = initial_prompt(neighbors)
    report, seconds, recovery = _generate_report_with_empty_recovery(
        runner, first_prompt, image_path=image_path, adapter_enabled=False
    )
    prompts.append(first_prompt); reports.append(report); timings.append(seconds)
    recovery_audits.append(recovery)
    if not str(report).strip():
        return _result(
            condition, prompts, reports, timings, verifier_outputs,
            "empty_initial_after_recovery", recovery_audits,
        )
    if condition == "A_single_pass":
        return _result(
            condition, prompts, reports, timings, verifier_outputs, "single_pass",
            recovery_audits,
        )
    for pass_number in range(1, max_revision_passes + 1):
        if condition in {"C_pretrained_gate", "D_corrected_lora_gate"}:
            verification = runner.verify(
                report,
                adapter_enabled=condition == "D_corrected_lora_gate",
                image_path=image_path,
                evidence=retrieval_context(neighbors),
            )
            verification["pass"] = pass_number - 1
            verifier_outputs.append(verification)
            if verification["verdict"] is True:
                return _result(
                    condition, prompts, reports, timings, verifier_outputs, "verifier_true",
                    recovery_audits,
                )
        prompt = revision_prompt(report, neighbors, pass_number)
        revised_report, seconds, recovery = _generate_report_with_empty_recovery(
            runner, prompt, image_path=image_path, adapter_enabled=False
        )
        prompts.append(prompt); reports.append(revised_report); timings.append(seconds)
        recovery_audits.append(recovery)
        if not revised_report.strip():
            return _result(
                condition, prompts, reports, timings, verifier_outputs,
                "empty_revision_fallback", recovery_audits,
            )
        report = revised_report
    return _result(
        condition, prompts, reports, timings, verifier_outputs, "max_passes",
        recovery_audits,
    )


def _result(
    condition,
    prompts,
    reports,
    timings,
    verifier_outputs,
    stop_reason,
    recovery_audits=None,
):
    nonempty_reports = [report for report in reports if str(report).strip()]
    return {
        "condition": condition,
        "prompts": prompts,
        "reports_by_pass": reports,
        "final_report": nonempty_reports[-1] if nonempty_reports else "",
        "generation_seconds_by_pass": timings,
        "empty_output_recovery_by_pass": list(recovery_audits or []),
        "verifier_outputs": verifier_outputs,
        "stopping_pass": len(reports) - 1,
        "stop_reason": stop_reason,
        "empty_output": not bool(nonempty_reports),
        "empty_generation_passes": [
            index for index, report in enumerate(reports) if not str(report).strip()
        ],
        "total_generation_seconds": float(sum(timings)),
        "total_verifier_seconds": float(sum(float(item["seconds"]) for item in verifier_outputs)),
    }
