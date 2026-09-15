# Leakage controlled retrieval grounded CXR report generation

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
