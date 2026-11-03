"""Build Biowulf notebooks 15–18 from reviewable source cells."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parent / "notebooks"
ROOT.mkdir(exist_ok=True)


def md(text):
    return dict(cell_type="markdown", metadata={}, source=textwrap.dedent(text).strip().splitlines(True))


def code(text):
    source = textwrap.dedent(text).strip()
    compile(source, "notebook_cell", "exec")
    return dict(cell_type="code", execution_count=None, metadata={}, outputs=[], source=source.splitlines(True))


BOOTSTRAP = 'from pathlib import Path\nimport os, sys, json\n\ncandidate = Path(os.environ.get("JAMIA_RERUN_DIR", Path.cwd())).resolve()\nif not (candidate / "rerun_config.json").is_file() and (candidate.parent / "rerun_config.json").is_file():\n    candidate = candidate.parent\nRERUN_DIR = candidate\nif not (RERUN_DIR / "rerun_config.json").is_file():\n    raise FileNotFoundError("Copy rerun_config.example.json to rerun_config.json, configure paths, and set JAMIA_RERUN_DIR to the repository root")\nimplementation = (RERUN_DIR / "src" / "rerun_code").resolve()\nsys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != implementation]\nsys.path.insert(0, str(RERUN_DIR / "src"))\nfrom rerun_code.config import load_config\n_, CONFIG = load_config(RERUN_DIR)\nRESULTS_ROOT = Path(os.environ.get("JAMIA_OUTPUT_ROOT", CONFIG["output_root"]))\nCONFIG["output_root"] = str(RESULTS_ROOT)\nRUN_NAME = os.environ.get("JAMIA_ABLATION_RUN", "matched_input_v1")\nABLATION_ROOT = RESULTS_ROOT / "post_rerun" / "modality_ablation" / RUN_NAME\nprint("Code:", RERUN_DIR)\nprint("Existing inputs:", RESULTS_ROOT)\nprint("NEW output:", ABLATION_ROOT)\n'


def save(name, cells):
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    (ROOT / name).write_text(json.dumps(notebook, indent=1)+"\n", encoding="utf-8")


def build():
    save("15_freeze_matched_input_ablation.ipynb", [
        md('''
        # 15 — Freeze a matched-backbone direct-image ablation on Biowulf

        Run on a CPU allocation. No generation, training, or old-result edits.
        This addresses the unequal-input concern without treating model families as exchangeable.
        Same backbone, same retrieved reports, same neutral instructions, with/without direct query pixels.
        Single-pass is primary. B/C/D are secondary whole-pipeline ablations; MedGemma verifier pixels
        also change, while Phi's verifier remains text-only. Retrieval is image-conditioned in BOTH arms.

        Start with `RUN_NAME = "matched_input_smoke_v1"` in every notebook; then freeze a separate
        `matched_input_v1` production run. Never promote smoke outputs to manuscript results.
        All original Notebooks 00–14 and results remain unchanged.
        '''), code(BOOTSTRAP), code('''
        from rerun_code.modality_ablation import freeze_experiment, CONDITIONS

        # Adjust BEFORE freezing. Changing these later requires a new RUN_NAME.
        MODEL_KEYS = ["medgemma_4b", "phi4_multimodal"]
        BUNDLE_NAMES = ["mimic", "iuhn", "combined"]
        STRATEGIES = list(CONDITIONS)  # use ["A_single_pass"] for the minimal primary experiment
        SMOKE_N = 2 if "smoke" in RUN_NAME.lower() else None

        # Known image/JSON prefixes in saved manifests are resolved from /vf/users/liangz2/openi
        # to /data/liangz2/openi when available, then checked against the ORIGINAL content hashes.
        for relative in ["manifests/leakage_audit.json", "bundles", "verifiers"]:
            path = RESULTS_ROOT / relative
            if not path.exists():
                raise FileNotFoundError(f"Required earlier-run input missing on Biowulf: {path}")
        output, PLAN = freeze_experiment(
            RERUN_DIR, CONFIG, run_name=RUN_NAME, models=MODEL_KEYS, bundles=BUNDLE_NAMES,
            conditions=STRATEGIES, smoke_n=SMOKE_N, seed=20261029,
        )
        print(json.dumps({key: PLAN[key] for key in ["experiment_id", "models", "bundles",
                         "conditions", "query_counts", "expected_records", "smoke_n"]}, indent=2))
        print("Upload/run Notebook 16 next; no new fine-tuning is required.")
        ''')])
    save("16_generate_matched_input_ablation.ipynb", [
        md('''
        # 16 — Generate both input arms, with resumable per-record outputs

        Use a Biowulf GPU allocation compatible with Notebook 05. Start a fresh kernel.
        Both arms are regenerated; old image-arm outputs are NOT substituted because the prompt is new.
        A failed record is never marked complete. Restart this notebook to skip verified saved records.
        Do not run two workers for the same model and RUN_NAME concurrently.
        '''), code(BOOTSTRAP), code('''
        # Optional Hugging Face authentication. Never paste or save a token in this notebook.
        LOGIN_TO_HUGGINGFACE = False
        if LOGIN_TO_HUGGINGFACE:
            from huggingface_hub import login
            from getpass import getpass
            token = os.environ.get("HF_TOKEN") or getpass("Hugging Face token (hidden): ")
            login(token=token, add_to_git_credential=False)
            del token

        from rerun_code.modality_ablation import (
            load_experiment, run_generation, activate_empty_recovery_amendment,
        )
        # Explicitly records the approved recovery amendment for an existing v1 run.
        # Does NOT change plan.json or rewrite any completed MedGemma/Phi records.
        # Stop other workers on this RUN_NAME before this one-time migration.
        APPLY_EMPTY_RECOVERY_AMENDMENT = True
        if APPLY_EMPTY_RECOVERY_AMENDMENT:
            activate_empty_recovery_amendment(RERUN_DIR, CONFIG, RUN_NAME)
        output, PLAN, _ = load_experiment(RERUN_DIR, CONFIG, RUN_NAME)
        print("Frozen final-report count:", PLAN["expected_records"])
        # Execution subsets only; they do not reduce the completeness requirement in Notebook 17.
        SELECTED_MODELS = PLAN["models"]  # e.g. ["medgemma_4b"]
        SELECTED_BUNDLES = PLAN["bundles"]
        RUN_GENERATION = False  # set True after inspecting the frozen plan and obtaining a GPU
        '''), code('''
        if not RUN_GENERATION:
            print("Not started. Set RUN_GENERATION=True in the preceding cell, then run both cells.")
        else:
            run_generation(RERUN_DIR, CONFIG, RUN_NAME,
                           selected_models=SELECTED_MODELS, selected_bundles=SELECTED_BUNDLES)
            print("Requested execution subset finished. Run other frozen models/bundles before Notebook 17.")
        ''')])
    save("17_score_matched_input_ablation.ipynb", [
        md('''
        # 17 — Completeness audit, CheXbert labels, and RadGraph F1

        Run in a fresh GPU kernel after generation to free model memory. This uses the existing
        CheXbert installation/checkpoint and frozen mapping, with negation/uncertainty sanity checks.
        Scoring is chunked and resumable. Missing RadGraph scores are errors, never zeros.
        No new human-validation claim is made, and the old validation gate is not modified.
        '''), code(BOOTSTRAP), code('''
        from rerun_code.modality_ablation import complete_frame, score_reports
        output, PLAN, generation = complete_frame(RERUN_DIR, CONFIG, RUN_NAME)
        print("Complete paired generation records:", len(generation))
        # Keep this fixed when resuming the same run.
        CHUNK_SIZE = 128
        scored = score_reports(RERUN_DIR, CONFIG, RUN_NAME, chunk_size=CHUNK_SIZE)
        print("Scored records:", len(scored))
        print("Saved:", output / "scored_records.json")
        ''')])
    save("18_analyze_matched_input_ablation.ipynb", [
        md('''
        # 18 — Paired inference and supplementary Table S20 exports

        CPU stage. Freeze thread counts before numerical imports; a small CPU allocation suffices.
        Primary: image minus no-image in single-pass RAG, per backbone and source/gallery stratum.
        Secondary: the same input contrast within the three revision/gated strategies.
        Patient/source-group bootstrap CIs and paired permutations use 10,000 draws, with Holm
        correction across selected model/stratum contrasts per outcome and separate primary/secondary families.
        Positive differences mean worse FER/omission, better F1. Do not pool duplicate studies across galleries.

        Outputs are new supplementary tables and a manuscript-insertion draft, NOT edits to the current
        Word files. Smoke runs get a separate technical summary with no publication prose.
        '''), code('''
        import os
        for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
            os.environ[name] = "1"
        '''), code(BOOTSTRAP), code('''
        from rerun_code.modality_ablation_statistics import analyze
        destination, contrasts = analyze(RERUN_DIR, CONFIG, RUN_NAME)
        print("Analysis saved to:", destination)
        display(contrasts[["model_key", "bundle", "source_dataset", "condition", "metric",
                           "arm_a", "arm_b", "difference_b_minus_a", "difference_ci_low",
                           "difference_ci_high", "p_holm", "positive_difference_means"]])
        print("Download the entire run folder for the manuscript update:")
        print(ABLATION_ROOT)
        ''')])


if __name__ == "__main__":
    build()
