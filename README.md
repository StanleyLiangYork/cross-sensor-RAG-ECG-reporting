# Cross Sensor Multimodal RAG Integrating ECG and CXR Evidence for ECG Report Generation

Zhaohui Liang, Niccolo Marini, Sivaramakrishnan Rajaraman, Zhiyun Xue, and Sameer Antani  
Division of Intramural Research, National Library of Medicine, National Institutes of Health, Bethesda, MD, USA

This repository packages the ECG–CXR research pipeline corresponding to `draft03.docx`. It contains the selected Project 01 (`notebook2`) notebooks and Python builder scripts. The layout follows [ECG_CXR_alignment](https://github.com/StanleyLiangYork/ECG_CXR_alignment).

The study tests three questions: whether same-admission ECG–CXR pairing is learnable, whether the bridge transfers to external semantic compatibility, and whether retrieved context improves ECG report generation. Continuous alignment is the reference method; vector-quantized bridges are architectural ablations.

## Repository layout

```text
.
├── notebooks/
│   ├── 00_environment_and_configuration.ipynb
│   ├── ...
│   ├── 20_evaluate_rag_report_generation.ipynb
│   ├── _build_external_ablation_notebooks.py
│   ├── _build_ecg_focused_rag_notebooks.py
│   └── _build_rag_report_generation_notebooks.py
├── config/
│   ├── environment.example.sh
│   └── source_manifest.json
├── tests/
│   └── test_notebook_release.py
├── requirements-biowulf.txt
├── .gitignore
├── LICENSE
└── README.md
```

The 21 numbered notebooks and three Python scripts are unchanged copies of the selected local sources. `config/source_manifest.json` records their SHA-256 hashes, the manuscript hash, and the reference repository commit. It establishes which files were packaged, not which files produced a historical experiment run.

## Scope and reproducibility status

The package matches the manuscript's experimental design. The research notebooks have no saved execution outputs. Dataset records, model checkpoints, generated reports, and numerical result files are not included. Packaging and static checks do not establish reproduction of the manuscript's numerical results; that requires the original run artifacts or a complete rerun with the specified data and environment.

The following files and branches are deliberately outside this release:

- `20_evaluate_rag_report_generation2.ipynb`: alternative extraction rules that differ from the manuscript's named evaluation version.
- `finetune_medgemma15_4b_multitask_lora_rank32_5fold.ipynb`: separate fine-tuning work; the manuscript uses frozen generators.
- Project 02 (`experiment3`): follow-up audits and matched-ECG-context experiments.
- `experiment3/trimodal`: additional laboratory-data experiments outside the two-modality paper.

## Data and model requirements

Obtain authorized access to Symile-MIMIC, MIMIC-IV-ECG, and the MIMIC-CXR-derived image/JSON collection used for the study. Data and model access terms apply separately from the code license.

| Input | Expected structure |
|---|---|
| Symile-MIMIC | `train.csv`, `val.csv`, `val_retrieval.csv`, `test.csv`, and corresponding arrays under `data_npy/` |
| MIMIC-IV-ECG | `record_list.csv`, `machine_measurements.csv`, and WFDB recordings under `files/` |
| External CXR | Recursively discoverable same-stem images and JSON files in training/test directories; reports in `Caption` and CheXpert labels in `labels` |

The CXR loader expects a prepared image/JSON derivative, not an unmodified MIMIC-CXR download. Use the same authorized derivative to reproduce the manuscript's cohort. Notebook 11 handles blank reports and assigns subjects appearing in both source folders to the test partition.

Models include HuBERT-ECG (`Edoardo-BS/hubert-ecg-base`), RAD-DINO (`microsoft/rad-dino`), BiomedCLIP text representations, MedGemma (`google/medgemma-1.5-4b-it`), and Qwen (`Qwen/Qwen3.5-4B`). Model access, downloads, and GPU resources must be available before execution.

## Environment setup

Use a GPU environment suitable for the foundation encoders, bridge training, and multimodal generation. The supplied dependency file and environment template are retained from the reference repository. The requirements contain version ranges and are not a complete historical environment lock. Validate them on the target cluster and save the resolved package versions with each final run.

```bash
python3 -m venv /path/to/venvs/ecg-cxr
source /path/to/venvs/ecg-cxr/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-biowulf.txt
python -m ipykernel install --user --name ecg-cxr --display-name "ECG CXR alignment"
cp config/environment.example.sh config/environment.sh
# Edit all placeholder paths before continuing.
source config/environment.sh
jupyter lab
```

Install a PyTorch build compatible with the target GPU driver. Select the installed kernel in Jupyter. Authenticate for gated models using the platform's credential mechanism.

Set `SYMILE_ROOT`, `MIMIC_ECG_ROOT`, `MIMIC_CXR_TRAIN_ROOT`, `MIMIC_CXR_TEST_ROOT`, `ECG_CXR_BRIDGE_ARTIFACT_ROOT`, and `HF_HOME` before starting Jupyter. The copied notebooks retain their original Biowulf defaults; override these for your environment. Keep artifacts and model caches on storage with adequate capacity. The `MIMIC_CXR_EXPECTED_TRAIN_IMAGES=43700` setting describes the study derivative; changing it changes the cohort check and does not reproduce the original data automatically.

## Execution order and manuscript mapping

Run the notebooks in numeric order, with notebook 08 treated as optional documentation. Keep the configured artifact root consistent across stages.

| Notebooks | Purpose | Manuscript coverage |
|---|---|---|
| 00–02 | Configure paths, verify pairing and patient splits, build manifests | Sections 3.1–3.2 |
| 03–04 | Encode ECG and CXR with frozen foundation models | Section 3.3 |
| 05–07 | Define, train, and evaluate the initial bridge | Bridge foundation for Sections 3.4–3.5 |
| 08 | External adapter contract; optional, superseded operationally by 09–16 | Background interface documentation |
| 09–11 | Configure ablations and prepare external ECG/CXR manifests | Sections 3.2 and 3.5 |
| 12 | Train the five bridge variants across three seeds | Sections 3.4–3.5; Table 2 |
| 13–14 | Encode external sensors, text, and assertion-aware concepts | Sections 3.3 and 3.5 |
| 15–16 | Train semantic adapters and evaluate detailed retrieval | Sections 3.5 and 4.2–4.4; retrieval figures |
| 17–18 | Select bridges by validation and construct the fixed RAG cohort/contexts | Section 3.6 |
| 19 | Generate reports with both frozen language models | Section 3.6 |
| 20 | Evaluate diagnostic content, text similarity, context quality, and paired effects | Section 3.7; Table 3 and generation figures |

Notebooks 17–20 consume the bridge, manifest, feature, and concept outputs from earlier notebooks. They do not require semantic-adapter training in notebook 15 or evaluation in notebook 16. However, 15–16 are required for the manuscript's semantic-transfer results. Notebooks 05 and 12 write reusable model modules under the configured artifact root's `code/` directory; these are generated during the workflow.

### Main experimental settings

- Five bridge variants: continuous, shared VQ, shared-specific VQ, no code matching, and shuffled pairs; bridge seeds 43, 44, and 45.
- Symile retrieval uses predefined groups of ten candidates. External semantic evaluation uses pools of 10, 100, and 1,000.
- Generation uses 250 fixed external test ECG queries and two models.
- Eleven conditions: zero-shot; continuous, VQ, and combined retrieval with ECG-only, CXR-only, or ECG+CXR context; and random-CXR context.
- Single-modality conditions use four examples; mixed conditions use two ECG and two CXR examples. Context is capped at 6,000 characters, with at most 384 generated tokens.
- Complete generation comprises 5,500 reports. Evaluation defaults to 500 paired query-level bootstrap replicates.
- The authoritative evaluation is `20_evaluate_rag_report_generation.ipynb`, using `ecg_generation_assertion_rules_v3`.

Environment overrides can restrict runs for development. Use a separate artifact root for smoke tests and preserve the final configuration, seeds, model revisions, and package versions for definitive runs.

## Python builder scripts

The notebooks are ready to execute; rebuilding them is not required.

- `_build_external_ablation_notebooks.py` generates notebooks 09–16. `BUILD_ONLY_NOTEBOOK` can select one named notebook.
- `_build_ecg_focused_rag_notebooks.py` generates notebooks 17–20.
- `_build_rag_report_generation_notebooks.py` is a historical command-line entry point that delegates to the ECG-focused builder when executed directly.

These scripts remain beside the notebooks because they write into their own directory. Running them overwrites the corresponding notebook files. Do not import the builders as runtime libraries: they contain top-level notebook-writing code. After intentional regeneration, inspect the diff, rerun the release checks, and update the source manifest if the release contents change.

## Outputs and verification

Results are written outside the repository under `ECG_CXR_BRIDGE_ARTIFACT_ROOT`. Key locations include:

- `reports/ablation/`: detailed and summarized Symile retrieval, external paired retrieval, semantic compatibility, and codebook diagnostics.
- `rag_report_generation_ecg/contexts/`: the fixed query/context records.
- `rag_report_generation_ecg/generations/`: resumable per-model generation records.
- `rag_report_generation_ecg/reports/`: diagnostic-content summaries, per-label metrics, bootstrap intervals, paired comparisons, and context-quality summaries.

Retain the run configuration, manifests and cohort signatures, selected checkpoint identities, model revisions, evaluation-rule version, aggregate tables, and complete generation counts to establish numerical provenance. Patient-level artifacts remain subject to the relevant data-use requirements.

Run the inherited release checks from the repository root:

```bash
python -m pytest tests/test_notebook_release.py -q
```

These checks verify the notebook inventory, cleared outputs, and configuration contract. They do not execute model training or verify clinical results.

## Interpretation boundaries

Same-admission pairing establishes concurrence, not diagnostic equivalence. Bridge training uses ECG and CXR features without laboratory inputs or text supervision. External repositories lack certified cross-repository identity linkage, so external concept compatibility is not same-patient retrieval.

The paper's four-ECG versus two-ECG-plus-two-CXR contrast substitutes context sources; it does not isolate CXR addition while preserving all ECG evidence. A shared character ceiling does not guarantee equal token exposure. Project 02's fixed-ECG-anchor contrasts are follow-up experiments and are not included here.

Evaluation uses machine interpretations and automated assertion extraction. Confidence gating, blinded clinician adjudication, and prospective clinical validation are not completed components of this pipeline. No claim of clinical deployment readiness follows from these experiments.

## License and attribution

The reference repository's MIT license and copyright notice are retained in `LICENSE`. Dataset and foundation-model licenses remain separate. When citing the study, use the manuscript title and authors above and add the final journal citation or DOI when available; this package does not assign publication metadata.
