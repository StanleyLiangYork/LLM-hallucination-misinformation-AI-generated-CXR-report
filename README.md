# Retrieval grounding over model scale for hallucination-controlled radiology report generation: a leakage-audited, architecture-dependent evaluation

**Zhaohui Liang, Niccolo Marini, Sivaramakrishnan Rajaraman, Zhiyun Xue, Sameer Antani**

National Library of Medicine, NIH, Bethesda, MD, USA

## Abstract

**Objectives:** Generative models can draft radiology reports but may hallucinate unsupported findings (fabrication) or omit clinically relevant ones (omission). We tested whether retrieval-augmented generation (RAG), verifier gating, and iterative self-correction reduce these failures; whether grounding matters more than scale; and whether apparent gains survive train–test leakage control.

**Materials and Methods:** Seven models generated reports for 1,034 chest radiograph studies. Strategies were single-pass RAG, unconditional revision, pretrained-verifier gating, and corrected low-rank adaptation (LoRA) verifier gating. Retrieval galleries excluded evaluation patients and duplicate images or reports. An independent annotation verified CheXbert-derived report labels, and adjudication resolved disagreements. We evaluated outcomes with cluster resampling, paired tests, and study-level rescue–harm counts.

**Results:** The analysis comprised 57,904 generated reports. No residual train–test overlap remained after gallery cleaning. Label validation reached macro F1 0.860. Fabrication and omission estimates increased after cleaning, which suggests that leakage can exaggerate grounding benefits. Performance depended more on architecture than on model size: larger models were not consistently better, and the configurations that best limited fabrication and omission differed. Unconditional revision produced more study-level harms than rescues. Corrected-verifier gating preserved single-pass performance but changed only 0.28% of study comparisons.

**Discussion:** Retrieval grounding, revision, and entailment verification offered complementary, architecture-dependent tools for controlling fabrication and omission. Benefit–harm analysis showed where each strategy helped or harmed.

**Conclusion:** Retrieval grounding with entailment verification supports privacy-preserving radiology natural language processing on local hardware. However, its benefit is architecture-dependent, so grounding and verification must be validated per architecture rather than assumed to transfer.

## Repository overview

This repository contains the code used for a leakage-controlled rerun of retrieval-augmented chest radiograph report generation. The workflow evaluates seven locally deployable language-only and multimodal models under four strategies:

1. single-pass retrieval-augmented generation (RAG);
2. four unconditional revision passes;
3. revision gated by a pretrained verifier; and
4. revision gated by a corrected LoRA verifier trained on source-disjoint data.

The code constructs fixed evaluation cohorts, removes detectable training-gallery overlap, generates reports, labels reports with CheXbert, validates label extraction against blinded human annotation, and computes clustered confidence intervals, paired tests, error transitions, and runtime benchmarks.

## Repository layout

- `notebooks/` - ordered execution notebooks. Outputs have been cleared before release.
- `src/rerun_code/` - shared Python modules used by the notebooks.
- `tests/` - structural tests for the rerun utilities.
- `algorithms/` - LaTeX sources for the manuscript algorithms.
- `rerun_config.example.json` - configuration template with placeholder paths.
- `requirements-biowulf.txt` - starting environment for the Biowulf workflow.
- `build_notebooks.py` and `build_post_rerun_notebooks.py` - notebook-generation utilities.

Generated reports, metrics, statistical outputs, human-annotation forms, model adapters, checkpoints, images, dataset records, and CSV result files are intentionally excluded.

## Data and model access

The code expects public IUHN-CXR/Open-i and MIMIC-CXR-JPG image-report pairs, pre-encoded BiomedCLIP/FAISS assets, and model checkpoints or adapters that the user is authorized to access. MIMIC-CXR-JPG is controlled-access through PhysioNet. This repository does not distribute any patient data, generated report output, human annotation, trained adapter, model weight, or Hugging Face credential.

Before use, copy `rerun_config.example.json` to `rerun_config.json`, set local paths, and review every model and data-access requirement. The example configuration deliberately contains no institution-specific paths.

## Recommended execution order

Run the notebooks from the repository root after setting the code and output locations:

```bash
export JAMIA_RERUN_DIR=/path/to/re_run
export JAMIA_OUTPUT_ROOT=/path/to/re_run/results
```

1. `00_preflight_and_freeze_config.ipynb`
2. `01_build_manifests_and_leakage_gate.ipynb`
3. `02_rebuild_training_only_faiss_bundles.ipynb`
4. `03_build_patient_disjoint_verifier_data.ipynb`
5. `04_train_and_test_corrected_verifiers.ipynb`
6. `05_run_multimodel_generation.ipynb`
7. `06_chexbert_labeling_and_validation_gate.ipynb`
8. `07_compute_per_study_and_aggregate_metrics.ipynb`
9. `08_cluster_bootstrap_and_paired_tests.ipynb`
10. `09_error_transitions_ensemble_runtime_exports.ipynb`
11. `10_sampling_provenance_audit.ipynb`
12. `11_b_vs_c_paired_inference.ipynb`
13. `12_joint_rescue_harm_and_qualitative_review.ipynb`
14. `13_biomedclip_faiss_timing.ipynb`

The workflow fails closed when it detects incomplete cohorts, overlap between evaluation and retrieval records, incompatible model provenance, or incomplete generated-report records. Notebook 06 also pauses for independent blinded annotation and adjudication before subsequent clinical label metrics are produced.

## Reproducibility notes

- Reference vectors are read from the `labels` key of each paired dataset record. An empty list means that all 13 finding labels are negative.
- Retrieval galleries are constructed from training records only and are checked against evaluation records using identifiers, exact file fingerprints, and near-image comparison.
- The corrected verifier split is source-grouped after exclusion of evaluation overlap. It should not be interpreted as external clinical validation.
- Bootstrap confidence intervals and paired permutation tests resample MIMIC patients or available IUHN source groups.
- The notebooks require substantial GPU memory and access to model files. They were designed for a managed HPC environment and are not expected to run end-to-end on a standard laptop.

## Tests

With the repository root on the Python path, run:

```bash
PYTHONPATH=src pytest -q tests
```

## External software

The workflow invokes the official implementations or model assets for CheXbert, RadGraph, Transformers, PEFT, FAISS, and BiomedCLIP-compatible encoders. Review the corresponding upstream licenses and access terms before use.

## License

The code in this repository is released under the [MIT License](LICENSE). Third-party models, checkpoints, datasets, and external tools retain their own terms.


## Additional reference review and matched-input notebooks

- `14_copy_sampled_reference_images.ipynb` exports an existing 200-case reference-review image sample. Supply `JAMIA_REFERENCE_SAMPLE` as a private JSON array with `blinded_review_id`, `image_path`, and `image_sha256`. The sample itself is not distributed.
- `15_freeze_matched_input_ablation.ipynb` freezes the query cohort, retrieved evidence, models, adapters, source fingerprints, and two direct-image input arms.
- `16_generate_matched_input_ablation.ipynb` runs paired generation with per-record checkpoints and validated restart. It requires a working allocated GPU.
- `17_score_matched_input_ablation.ipynb` scores complete reports in resumable chunks with CheXbert and RadGraph.
- `18_analyze_matched_input_ablation.ipynb` computes paired cluster inference, corrected comparisons, and rescue/harm summaries.

Set `JAMIA_RERUN_DIR` to this repository root (not `notebooks/`) and configure `rerun_config.json`. Helpers reside in `src/rerun_code/`. Set `JAMIA_OUTPUT_ROOT` to the existing results root from notebooks 00–13. Notebook 15 requires the earlier fixed retrieval bundles, leakage audit, corrected adapters, and evaluation provenance. The new experiment writes under `results/post_rerun/modality_ablation/<RUN_NAME>/`; it does not replace the original generation cohort. Optional `JAMIA_DATA_ROOT` and `JAMIA_OLD_DATA_ROOT` relocate the known data-root prefix while retaining content verification.

Both modality arms retain image-conditioned retrieval. The comparison estimates direct image access within MedGemma and Phi-4, not the effect of all image information or a general architecture advantage. Single-pass comparisons are primary within this exploratory experiment; revision and gated comparisons are secondary.

Notebook 16 includes a disclosed post hoc empty-output recovery rule: one retry of an empty report pass with unchanged evidence and prompt and `min_new_tokens=32`. It preserves successful first attempts and audits recovery by arm. Unequal recovery limits interpretation of Phi-4 effects. Existing Biowulf frozen plans include source fingerprints: do not overwrite them to force a resume after changing code or repository layout. Continue historical runs with their original matching code, or freeze a separate new run. The versioned legacy amendment only accepts recognized fingerprints.

`build_modality_ablation_notebooks.py` regenerates notebooks 15–18. Offline tests are in `tests/test_modality_ablation.py`; they require no patient data or model downloads. Full GPU execution still requires the configured datasets, checkpoints, and compatible runtime.
