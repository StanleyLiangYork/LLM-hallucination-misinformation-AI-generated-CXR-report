from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def adapter_config(path: str | Path) -> dict[str, Any]:
    source = Path(path) / "adapter_config.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing PEFT adapter configuration: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def resolve_base_model(spec: Mapping[str, Any], adapter_path: str | Path | None = None) -> str:
    if adapter_path and (Path(adapter_path) / "adapter_config.json").exists():
        value = adapter_config(adapter_path).get("base_model_name_or_path")
        if value:
            return str(value)
    legacy = Path(str(spec["legacy_adapter"]))
    if (legacy / "adapter_config.json").exists():
        value = adapter_config(legacy).get("base_model_name_or_path")
        if value:
            return str(value)
    return str(spec["fallback_base_model"])


def resolve_training_base_model(spec: Mapping[str, Any]) -> str:
    """Canonical base used for a fresh corrected adapter, never the legacy adapter base."""
    return str(spec.get("training_base_model") or spec["fallback_base_model"])


def normalize_adapter_base_model(adapter_path: str | Path, canonical_base: str) -> str:
    """Replace a machine-specific PEFT base path with the audited canonical ID.

    Returns the value originally recorded in ``adapter_config.json``.  This is
    intended for newly trained corrected adapters whose base is already known
    from the frozen model registry; it must not be used to relabel an unaudited
    third-party adapter.
    """
    source = Path(adapter_path) / "adapter_config.json"
    raw = adapter_config(adapter_path)
    previous = str(raw.get("base_model_name_or_path") or "")
    raw["base_model_name_or_path"] = str(canonical_base)
    source.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return previous


class IncompleteCorrectedAdapterError(RuntimeError):
    """A model-specific output directory exists but is not a reusable adapter."""

    def __init__(self, output_dir: str | Path, message: str):
        self.output_dir = Path(output_dir)
        super().__init__(message)


def quarantine_incomplete_corrected_adapter(output_dir: str | Path) -> Path:
    """Move one incomplete model output aside without deleting it."""
    source = Path(output_dir)
    if not source.exists():
        raise FileNotFoundError(f"Incomplete corrected-adapter directory no longer exists: {source}")
    if source.is_symlink():
        raise RuntimeError(f"Refusing to quarantine a symlinked model output directory: {source}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = source.with_name(f"{source.name}.incomplete-{stamp}")
    counter = 1
    while candidate.exists():
        candidate = source.with_name(f"{source.name}.incomplete-{stamp}-{counter}")
        counter += 1
    source.rename(candidate)
    return candidate


def completed_corrected_adapter(
    output_dir: str | Path,
    model_key: str,
    canonical_base: str,
    spec: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return audited completed-adapter provenance, or ``None`` for a fresh run.

    A prior run is reusable only when its adapter configuration, exactly one
    nonempty PEFT weight file, and training provenance are all present and agree
    with the requested model key, frozen canonical base, and (when supplied)
    current loader/tokenizer/prompt protocol. Partial output is a hard stop so
    it cannot be mistaken for a successfully fine-tuned verifier.
    """
    output = Path(output_dir)
    adapter_dir = output / "adapter"
    provenance_path = output / "training_provenance.json"
    expected_paths = [adapter_dir / "adapter_config.json", provenance_path]
    weight_candidates = [
        path for path in (
            adapter_dir / "adapter_model.safetensors",
            adapter_dir / "adapter_model.bin",
        )
        if path.exists()
    ]
    relevant_output_exists = output.exists() and any(output.iterdir())
    if not relevant_output_exists:
        return None

    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing or len(weight_candidates) != 1:
        raise IncompleteCorrectedAdapterError(
            output,
            f"Incomplete corrected-adapter output for {model_key!r} in {output}. "
            f"Missing={missing}; weight_files={[str(path) for path in weight_candidates]}. "
            "Quarantine the incomplete model output directory before starting a fresh training run."
        )
    weights = weight_candidates[0]
    if weights.stat().st_size <= 0:
        raise RuntimeError(f"Corrected-adapter weight file is empty: {weights}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if str(provenance.get("model_key") or "") != str(model_key):
        raise AssertionError(
            f"Corrected-adapter provenance model_key={provenance.get('model_key')!r}; "
            f"requested model_key={model_key!r}"
        )
    actual_base = str(provenance.get("actual_training_base") or "")
    if actual_base != str(canonical_base):
        raise AssertionError(
            f"Corrected adapter was trained on {actual_base!r}; requested canonical base is {canonical_base!r}"
        )
    if provenance.get("legacy_weights_loaded") is not False:
        raise AssertionError(
            "Corrected-adapter provenance does not explicitly confirm legacy_weights_loaded=False"
        )
    if spec is not None:
        expected_runtime = {
            "loader_profile": str(spec.get("loader_profile") or "default"),
            "tokenizer_source": str(spec.get("tokenizer_source") or "base"),
            "chat_template_policy": str(spec.get("chat_template_policy") or "auto"),
        }
        for key, expected in expected_runtime.items():
            observed = str(provenance.get(key) or ({
                "loader_profile": "default",
                "tokenizer_source": "base",
                "chat_template_policy": "auto",
            }[key]))
            if observed != expected:
                raise AssertionError(
                    f"Corrected adapter {model_key!r} was trained with {key}={observed!r}, "
                    f"but the current frozen registry requires {expected!r}. Move the completed "
                    "model output aside and rerun notebook 04; do not mix loading protocols."
                )
        recorded_legacy_hash = str(provenance.get("legacy_adapter_config_sha256") or "")
        if recorded_legacy_hash:
            legacy_config_path = Path(str(spec["legacy_adapter"])) / "adapter_config.json"
            current_legacy_hash = hashlib.sha256(legacy_config_path.read_bytes()).hexdigest()
            if recorded_legacy_hash != current_legacy_hash:
                raise AssertionError(
                    f"Legacy LoRA configuration changed after corrected training for {model_key!r}. "
                    "Move the completed model output aside and rerun notebook 04."
                )

    recorded_base = str(adapter_config(adapter_dir).get("base_model_name_or_path") or "")
    if recorded_base != actual_base:
        normalize_adapter_base_model(adapter_dir, actual_base)
    return {
        "adapter_dir": adapter_dir,
        "actual_base": actual_base,
        "history": provenance.get("history", []),
        "training_provenance": provenance,
        "weights_file": weights,
        "adapter_base_before_normalization": recorded_base,
    }


def assert_model_identity(model_key: str, spec: Mapping[str, Any], actual_base: str) -> None:
    required = str(spec.get("require_adapter_base_contains", "") or "").strip()
    if required and required.lower() not in actual_base.lower():
        raise AssertionError(
            f"{model_key} was requested as {spec['display_name']}, but adapter_config.json names "
            f"{actual_base!r}. Do not publish this run under the requested identity. Supply a matching "
            "adapter or change the registered comparator name."
        )


def _quantization_kwargs() -> dict[str, Any]:
    import torch
    if os.environ.get("JAMIA_LOAD_IN_4BIT", "1") != "1" or not torch.cuda.is_available():
        return {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return {"torch_dtype": torch.bfloat16}
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        "device_map": "auto",
    }


def _restore_expected_tied_lm_head(model, config, missing_keys: Iterable[str]) -> list[str]:
    """Resolve Qwen's intentionally omitted tied output-head tensor.

    Official Qwen2 checkpoints set ``tie_word_embeddings=True`` and store only
    the input embedding matrix.  ``load_state_dict(strict=False)`` consequently
    reports ``lm_head.weight`` as missing when the architecture is constructed
    from config before loading the safetensors file.  This is expected only when
    the configuration requests tying and the model can demonstrably restore the
    shared tensor.
    """
    remaining = list(missing_keys)
    if "lm_head.weight" not in remaining or not bool(getattr(config, "tie_word_embeddings", False)):
        return remaining

    model.tie_weights()
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("Qwen config requests tied embeddings, but an embedding module is unavailable")

    input_weight = input_embeddings.weight
    output_weight = output_embeddings.weight
    shares_storage = output_weight is input_weight
    if not shares_storage and hasattr(output_weight, "data_ptr") and hasattr(input_weight, "data_ptr"):
        shares_storage = output_weight.data_ptr() == input_weight.data_ptr()
    if not shares_storage:
        raise RuntimeError("model.tie_weights() did not restore the Qwen lm_head/input-embedding tie")

    return [key for key in remaining if key != "lm_head.weight"]


def _manual_qwen2_safetensors_load(base: str):
    """Bypass Transformers' automatic conversion reporter for official Qwen2-1.5B."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    if base != "Qwen/Qwen2-1.5B-Instruct":
        raise ValueError(f"Manual fallback is restricted to the audited Qwen checkpoint, not {base!r}")
    local = Path(snapshot_download(
        repo_id=base,
        allow_patterns=[
            "config.json", "generation_config.json", "model.safetensors",
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
            "special_tokens_map.json", "chat_template.jinja",
        ],
    ))
    weights = local / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"Official Qwen safetensors file was not downloaded: {weights}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    config = AutoConfig.from_pretrained(local, trust_remote_code=True)
    # Keep PEFT provenance portable.  Loading config from the downloaded snapshot
    # otherwise records a machine-specific Hugging Face cache path in a newly
    # trained adapter's base_model_name_or_path.
    config._name_or_path = base
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, torch_dtype=dtype
    )
    state = load_file(str(weights), device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    del state
    missing = [key for key in incompatible.missing_keys if not key.endswith("rotary_emb.inv_freq")]
    missing = _restore_expected_tied_lm_head(model, config, missing)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Manual Qwen safetensors load did not match the architecture. "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model._jamia_load_method = "manual_official_safetensors"
    print(
        "Loaded Qwen with direct official model.safetensors fallback; automatic conversion was "
        "bypassed and the configured tied lm_head was restored."
    )
    return model


def _load_medgemma_bf16_training_checkpoint(base: str, *, for_training: bool):
    """Load full-weight MedGemma for LoRA training without bitsandbytes."""
    import torch
    import transformers

    expected = "unsloth/medgemma-4b-it"
    if base != expected:
        raise ValueError(f"MedGemma bfloat16 training loader requires {expected!r}, not {base!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("MedGemma corrected training requires a CUDA GPU for the full bfloat16 checkpoint")
    model_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_class is None:
        model_class = getattr(transformers, "AutoModelForCausalLM")
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    model = model_class.from_pretrained(base, **kwargs)
    _audit_model_device_map(model, "MedGemma", for_training=for_training)
    model._jamia_load_method = "medgemma_unsloth_full_bf16_no_bitsandbytes"
    print(
        "Loaded full bfloat16 MedGemma without bitsandbytes for corrected LoRA training: "
        f"{expected}. Legacy/reference 4-bit weights were not loaded."
    )
    return model


def _install_phi4_siglip2_processor_compatibility(
    siglip2_module=None,
    compatibility_values: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Expose the SigLIP2 names expected by Microsoft's Phi-4 remote code.

    The remote code treats utilities imported by an older SigLIP2 module as a
    public namespace.  Newer/refactored installations may retain the utilities
    in their canonical modules without re-exporting them.  Only missing names
    are installed, and the installed-name tuple is returned for provenance.
    """
    if siglip2_module is None:
        import transformers.models.siglip2.image_processing_siglip2 as siglip2_module
    if compatibility_values is None:
        import math
        import numpy as np
        from transformers import Siglip2ImageProcessor
        from transformers.image_processing_utils import BatchFeature
        from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
        from transformers.image_utils import (
            ChannelDimension,
            PILImageResampling,
            infer_channel_dimension_format,
            make_flat_list_of_images,
            to_numpy_array,
            valid_images,
            validate_preprocess_arguments,
        )
        from transformers.utils.generic import filter_out_non_signature_kwargs

        def get_image_size_for_max_num_patches(
            image_height: int,
            image_width: int,
            patch_size: int,
            max_num_patches: int,
            eps: float = 1e-5,
        ) -> tuple[int, int]:
            def scaled(scale: float, size: int) -> int:
                value = math.ceil((size * scale) / patch_size) * patch_size
                return int(max(patch_size, value))

            lower, upper = eps / 10, 100.0
            while upper - lower >= eps:
                scale = (lower + upper) / 2
                height, width = scaled(scale, image_height), scaled(scale, image_width)
                if (height / patch_size) * (width / patch_size) <= max_num_patches:
                    lower = scale
                else:
                    upper = scale
            return scaled(lower, image_height), scaled(lower, image_width)

        def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
            height, width, channels = image.shape
            rows, columns = height // patch_size, width // patch_size
            patches = image.reshape(rows, patch_size, columns, patch_size, channels)
            return patches.transpose(0, 2, 1, 3, 4).reshape(rows * columns, -1)

        def pad_along_first_dim(
            array: np.ndarray, target_length: int, pad_value: int = 0
        ) -> tuple[np.ndarray, np.ndarray]:
            padding = target_length - array.shape[0]
            mask = np.ones((target_length,), dtype=np.int32)
            if padding > 0:
                pads = [(0, padding)] + [(0, 0)] * (array.ndim - 1)
                array = np.pad(array, pads, mode="constant", constant_values=pad_value)
                mask[-padding:] = 0
            return array, mask

        compatibility_values = {
            "BatchFeature": BatchFeature,
            "ChannelDimension": ChannelDimension,
            "PILImageResampling": PILImageResampling,
            "Siglip2ImageProcessor": Siglip2ImageProcessor,
            "convert_image_to_patches": convert_image_to_patches,
            "convert_to_rgb": convert_to_rgb,
            "filter_out_non_signature_kwargs": filter_out_non_signature_kwargs,
            "get_image_size_for_max_num_patches": get_image_size_for_max_num_patches,
            "infer_channel_dimension_format": infer_channel_dimension_format,
            "make_flat_list_of_images": make_flat_list_of_images,
            "pad_along_first_dim": pad_along_first_dim,
            "resize": resize,
            "to_channel_dimension_format": to_channel_dimension_format,
            "to_numpy_array": to_numpy_array,
            "valid_images": valid_images,
            "validate_preprocess_arguments": validate_preprocess_arguments,
        }

    installed = []
    for name, value in compatibility_values.items():
        if not hasattr(siglip2_module, name):
            setattr(siglip2_module, name, value)
            installed.append(name)
    required = set(compatibility_values)
    still_missing = sorted(name for name in required if not hasattr(siglip2_module, name))
    if still_missing:
        raise RuntimeError(f"Phi-4 SigLIP2 compatibility installation failed; missing={still_missing}")
    return tuple(sorted(installed))


def _load_phi4_reasoning_vision_reference_checkpoint(base: str, *, for_training: bool):
    """Load Phi-4 with the custom class and precision used in the reference notebook."""
    import torch
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    expected = "microsoft/Phi-4-reasoning-vision-15B"
    if base != expected:
        raise ValueError(f"Phi-4 reference loader requires {expected!r}, not {base!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phi-4 corrected training requires a CUDA GPU")
    config = AutoConfig.from_pretrained(base, trust_remote_code=True)
    phi_class = get_class_from_dynamic_module(
        "modeling_phi4_visionr.Phi4ForCausalLMV", base
    )
    model = phi_class.from_pretrained(
        base,
        config=config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    _audit_model_device_map(model, "Phi-4", for_training=for_training)
    model._jamia_load_method = "phi4_reasoning_vision_reference_custom_class_bf16"
    print(
        "Loaded Phi-4 with the reference notebook path: AutoConfig + dynamic "
        "Phi4ForCausalLMV + full bfloat16 weights; bitsandbytes was not used."
    )
    return model


REFERENCE_CAUSAL_LM_BASES = {
    "openai/gpt-oss-20b",
    "unsloth/llama-3-8b-Instruct",
    "Echelon-AI/Med-Qwen2-7B",
    "aaditya/OpenBioLLM-Llama3-8B",
}


def _audit_model_device_map(model, model_label: str, *, for_training: bool) -> dict[str, Any]:
    """Enforce phase-specific Accelerate dispatch safeguards.

    Trainer cannot safely update a model split between GPU and CPU/disk. During
    evaluation, CPU offload is supported by Accelerate hooks and matches the
    supplied reference notebooks' unrestricted ``device_map='auto'`` behavior.
    Disk offload remains a hard stop because no persistent offload directory is
    configured by this rerun.
    """
    device_map = getattr(model, "hf_device_map", {}) or {}
    offloaded = {
        str(device).lower() for device in device_map.values()
        if str(device).lower() in {"cpu", "disk"}
    }
    if for_training and offloaded:
        raise RuntimeError(
            f"{model_label} was partly offloaded to CPU/disk, which is unsafe for this Trainer job. "
            "Request a GPU with enough memory for the reference notebook's bfloat16-dtype load. "
            f"hf_device_map={device_map}"
        )
    if not for_training and "disk" in offloaded:
        raise RuntimeError(
            f"{model_label} requires disk offload during evaluation, but this rerun has no audited "
            f"persistent offload directory. hf_device_map={device_map}"
        )
    if not for_training and "cpu" in offloaded:
        print(
            f"EVALUATION CPU OFFLOAD ACTIVE for {model_label}. This is allowed for inference but "
            f"will be slower than an all-GPU run. hf_device_map={device_map}"
        )
    return {str(key): str(value) for key, value in device_map.items()}


def _reject_training_offload(model, model_label: str) -> None:
    """Backward-compatible wrapper used by structural tests and callers."""
    _audit_model_device_map(model, model_label, for_training=True)


def _load_reference_causal_lm_bf16_checkpoint(base: str, *, for_training: bool):
    """Load one audited text backbone exactly as its supplied reference notebook.

    The four reference notebooks all use ``AutoConfig`` for compatibility
    checking followed by ``AutoModelForCausalLM.from_pretrained`` with full
    ``torch_dtype=bfloat16`` and automatic device placement.  They do not pass
    a BitsAndBytesConfig.  Keeping that behavior in a named profile prevents
    the generic 4-bit loader from silently changing the loading path.  GPT-OSS
    retains the checkpoint's native MXFP4 representation; the dtype argument
    applies to the remaining model components.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if base not in REFERENCE_CAUSAL_LM_BASES:
        raise ValueError(
            f"Reference causal-LM loader is not audited for {base!r}; "
            f"allowed={sorted(REFERENCE_CAUSAL_LM_BASES)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Corrected training for {base} requires a CUDA GPU for the reference "
            "notebook's bfloat16-dtype loading path"
        )
    # This call is intentional even though from_pretrained resolves the config
    # again: it mirrors the reference notebooks and fails early on an
    # incompatible Transformers installation.
    AutoConfig.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _audit_model_device_map(model, base, for_training=for_training)
    model._jamia_load_method = "reference_auto_causal_lm_dtype_bf16_no_bitsandbytes"
    print(
        "Loaded the audited reference causal-LM backbone with torch_dtype=bfloat16 "
        f"and no bitsandbytes configuration: {base}"
    )
    return model


def _load_registered_text_tokenizer(
    spec: Mapping[str, Any],
    base: str,
    adapter_path: str | Path | None,
):
    """Apply the tokenizer-source order recorded by the reference notebook."""
    from transformers import AutoTokenizer

    policy = str(spec.get("tokenizer_source") or "base")
    if policy not in {"base", "adapter_then_base"}:
        raise ValueError(f"Unsupported tokenizer_source={policy!r} for {base}")
    if policy == "adapter_then_base":
        candidates: list[Path] = []
        if adapter_path:
            candidates.append(Path(adapter_path))
        legacy = Path(str(spec["legacy_adapter"]))
        if legacy not in candidates:
            candidates.append(legacy)
        for candidate in candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(candidate),
                    use_fast=True,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                print("Loaded tokenizer from adapter directory:", candidate)
                break
            except Exception as exc:
                print(
                    f"Tokenizer unavailable in {candidate}; falling back to the next audited source "
                    f"({type(exc).__name__})."
                )
        else:
            tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
            print("Loaded tokenizer from base model:", base)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        print("Loaded tokenizer from base model:", base)

    prompt_policy = str(spec.get("chat_template_policy") or "auto")
    if prompt_policy not in {"auto", "plain"}:
        raise ValueError(f"Unsupported chat_template_policy={prompt_policy!r} for {base}")
    tokenizer._jamia_chat_template_policy = prompt_policy
    return tokenizer


def release_accelerator_memory() -> None:
    """Release unreachable model allocations before a same-kernel reload."""
    import gc

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except ImportError:
        pass


def load_processor_and_model(
    spec: Mapping[str, Any],
    *,
    adapter_path: str | Path | None = None,
    trainable_adapter: bool = False,
    base_model_override: str | None = None,
    for_training: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    base = str(base_model_override) if base_model_override else resolve_base_model(spec, adapter_path)
    architecture = str(spec["architecture"])
    common = dict(trust_remote_code=True, **_quantization_kwargs())
    if architecture == "multimodal":
        loader_profile = str(spec.get("loader_profile") or "")
        if loader_profile == "phi4_reasoning_vision_reference_bf16":
            installed_names = _install_phi4_siglip2_processor_compatibility()
            if installed_names:
                print(
                    "Installed scoped Phi-4/SigLIP2 processor compatibility names: "
                    + ", ".join(installed_names)
                )
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        model = None
        errors = []
        if loader_profile == "medgemma_unsloth_bf16_training":
            try:
                model = _load_medgemma_bf16_training_checkpoint(base, for_training=for_training)
            except Exception as exc:
                errors.append(f"MedGemma-bfloat16-training-loader: {type(exc).__name__}: {exc}")
            if model is None:
                raise RuntimeError(
                    "Could not load the required non-bitsandbytes MedGemma training checkpoint:\n"
                    + "\n".join(errors)
                )
        if loader_profile == "phi4_reasoning_vision_reference_bf16":
            try:
                model = _load_phi4_reasoning_vision_reference_checkpoint(base, for_training=for_training)
            except Exception as exc:
                errors.append(f"Phi4-reference-bfloat16-loader: {type(exc).__name__}: {exc}")
            if model is None:
                raise RuntimeError(
                    "Could not load the required non-bitsandbytes Phi-4 training checkpoint:\n"
                    + "\n".join(errors)
                )
        if model is None and "phi-4" in base.lower() and "vision" in base.lower():
            try:
                from transformers import AutoConfig
                from transformers.dynamic_module_utils import get_class_from_dynamic_module
                config = AutoConfig.from_pretrained(base, trust_remote_code=True)
                phi_class = get_class_from_dynamic_module(
                    "modeling_phi4_visionr.Phi4ForCausalLMV", base
                )
                model = phi_class.from_pretrained(base, config=config, **common)
            except Exception as exc:
                errors.append(f"Phi4ForCausalLMV: {type(exc).__name__}: {exc}")
        for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
            if model is not None:
                break
            try:
                import transformers
                cls = getattr(transformers, class_name)
                model = cls.from_pretrained(base, **common)
                break
            except Exception as exc:
                errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
        if model is None:
            raise RuntimeError("Could not load multimodal model:\n" + "\n".join(errors))
        tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    else:
        tokenizer = _load_registered_text_tokenizer(spec, base, adapter_path)
        processor = tokenizer
        errors = []
        model = None
        loader_profile = str(spec.get("loader_profile") or "")
        if loader_profile == "reference_causal_lm_bf16":
            try:
                model = _load_reference_causal_lm_bf16_checkpoint(base, for_training=for_training)
            except Exception as exc:
                errors.append(f"reference-causal-lm-bfloat16: {type(exc).__name__}: {exc}")
            if model is None:
                raise RuntimeError(
                    f"Could not load the required reference text checkpoint {base!r}:\n"
                    + "\n".join(errors)
                )
        for safe_only in (True, None):
            if model is not None:
                break
            try:
                extra = {"use_safetensors": True} if safe_only else {}
                model = AutoModelForCausalLM.from_pretrained(base, **common, **extra)
                break
            except Exception as exc:
                mode = "safetensors-only" if safe_only else "default"
                errors.append(f"{mode}: {type(exc).__name__}: {exc}")
        if model is None and base == "Qwen/Qwen2-1.5B-Instruct":
            try:
                model = _manual_qwen2_safetensors_load(base)
            except Exception as exc:
                errors.append(f"manual-official-safetensors: {type(exc).__name__}: {exc}")
        if model is None:
            raise RuntimeError(
                f"Could not load canonical text base {base!r}. Attempts:\n" + "\n".join(errors)
            )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=trainable_adapter)
    model.eval()
    return processor, tokenizer, model, base


def lora_config_from_legacy(legacy_adapter: str | Path):
    from peft import LoraConfig
    raw = adapter_config(legacy_adapter)
    allowed = {
        "r", "lora_alpha", "lora_dropout", "target_modules", "bias", "task_type",
        "modules_to_save", "layers_to_transform", "layers_pattern", "fan_in_fan_out",
    }
    values = {key: raw[key] for key in allowed if key in raw and raw[key] is not None}
    values["inference_mode"] = False
    values.setdefault("task_type", "CAUSAL_LM")
    return LoraConfig(**values)


def make_causal_lm_features(tokenizer, prompt: str, answer: str, max_length: int) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    answer_ids = tokenizer(" " + answer.strip(), add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    available = max(1, max_length - len(answer_ids) - len(eos))
    prompt_ids = prompt_ids[-available:]
    input_ids = (prompt_ids + answer_ids + eos)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids + eos)[:len(input_ids)]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def render_training_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "_jamia_chat_template_policy", "auto") == "plain":
        return prompt
    method = getattr(tokenizer, "apply_chat_template", None)
    if callable(method):
        try:
            return method(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return prompt


class VerifierDataset:
    def __init__(self, frame, tokenizer, prompt_function, max_length: int = 1024):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.prompt_function = prompt_function
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return make_causal_lm_features(
            self.tokenizer,
            render_training_prompt(
                self.tokenizer, self.prompt_function(str(row["report_text"]))
            ),
            "TRUE" if bool(row["verdict"]) else "FALSE",
            self.max_length,
        )


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch
        max_len = max(len(item["input_ids"]) for item in features)
        pad = self.tokenizer.pad_token_id
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            delta = max_len - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [pad] * delta)
            result["attention_mask"].append(item["attention_mask"] + [0] * delta)
            result["labels"].append(item["labels"] + [-100] * delta)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def train_corrected_verifier(
    model_key: str,
    spec: Mapping[str, Any],
    train_frame,
    validation_frame,
    output_dir: str | Path,
    prompt_function,
    *,
    epochs: float = 3.0,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    max_length: int = 1024,
):
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    training_base = resolve_training_base_model(spec)
    processor, tokenizer, model, actual_base = load_processor_and_model(
        spec, base_model_override=training_base, for_training=True
    )
    assert_model_identity(model_key, spec, actual_base)
    is_kbit = bool(getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False))
    if is_kbit:
        model = prepare_model_for_kbit_training(model)
    else:
        print("Full-precision/bfloat16 backbone detected; skipping bitsandbytes k-bit preparation.")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = get_peft_model(model, lora_config_from_legacy(spec["legacy_adapter"]))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    train_dataset = VerifierDataset(train_frame, tokenizer, prompt_function, max_length=max_length)
    validation_dataset = VerifierDataset(validation_frame, tokenizer, prompt_function, max_length=max_length)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        bf16=True,
        report_to="none",
        seed=20260831,
        data_seed=20260831,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train()
    adapter_dir = output / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    previous_saved_base = normalize_adapter_base_model(adapter_dir, actual_base)
    if previous_saved_base != actual_base:
        print(
            "Normalized corrected adapter base-model provenance from "
            f"{previous_saved_base!r} to {actual_base!r}."
        )
    history = list(trainer.state.log_history)
    # A notebook commonly evaluates immediately after training in the same
    # Python kernel. Explicitly drop Trainer, optimizer, datasets, and the first
    # backbone so the canonical evaluation reload can reclaim GPU memory.
    del trainer, model, train_dataset, validation_dataset, processor, tokenizer
    release_accelerator_memory()
    return adapter_dir, actual_base, history


_TRUE = re.compile(r"\btrue\b", re.I)
_FALSE = re.compile(r"\bfalse\b", re.I)


def parse_verdict(text: str) -> bool | None:
    true = _TRUE.search(text)
    false = _FALSE.search(text)
    if true and not false:
        return True
    if false and not true:
        return False
    if true and false:
        return true.start() < false.start()
    return None
