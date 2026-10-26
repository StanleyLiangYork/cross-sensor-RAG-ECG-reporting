#!/usr/bin/env python3
"""Generate notebooks 09-16 for the external ECG/CXR CodeBind ablation study."""

import json
import os
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
ONLY_NOTEBOOK = os.environ.get("BUILD_ONLY_NOTEBOOK", "").strip()


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).lstrip().splitlines(True)}


def code(source):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": dedent(source).lstrip().splitlines(True),
    }


def write(name, cells):
    if ONLY_NOTEBOOK and name != ONLY_NOTEBOOK:
        return
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = ROOT / name
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    print(path)


write("09_external_ablation_design_and_configuration.ipynb", [
    md("""
    # 09 - External ECG/CXR ablation design and configuration

    This experiment separates two questions:

    1. **Encounter bridge:** which Symile objective best transfers matched ECG-CXR evidence?
    2. **Semantic re-alignment:** can independent ECG-diagnosis and CXR-report pairs be mapped into the frozen bridge?

    Same-admission Symile pairs are hard positives for provenance, not proof of diagnostic equivalence. External ECG and CXR collections are not assumed to contain paired instances. Exact cross-sensor retrieval is evaluated only when shared `hadm_id` or `subject_id` values are available; otherwise cross-sensor evaluation uses explicitly labeled semantic compatibility and is reported separately.

    ### Bridge ablations

    | Variant | Continuous alignment | Shared VQ | Specific VQ | Code matching |
    |---|---:|---:|---:|---:|
    | `clip_continuous` | yes | no | no | no |
    | `shared_vq` | yes | yes | no | yes |
    | `shared_specific_vq` | yes | yes | yes | yes |
    | `shared_specific_no_code_match` | yes | yes | yes | no |
    | `shuffled_pairs_control` | yes | yes | yes | yes, but ECG-CXR training pairs are shuffled |

    The current project is a CodeBind-inspired implementation: it tests shared/specific compositional VQ, but does not claim reproduction of every component in the original paper (for example, its reconstruction decoder and EMA codebook maintenance).

    ### External semantic ablations

    - `frozen_zero_shot`: no external adaptation; sensor codes only.
    - `shared_raw_text`: one text adapter shared by ECG machine diagnoses and CXR reports.
    - `separate_raw_text`: independent ECG-text and CXR-text adapters.
    - `shared_concepts`: one shared adapter trained on normalized concept strings; rule-derived by default and replaceable by audited LLM output.

    Use at least three seeds for final comparisons. Model selection uses Symile validation data or the external validation split, never external test results.
    """),
    code("""
    from pathlib import Path
    import importlib.util
    import json
    import os

    ARTIFACT_ROOT = Path(os.environ.get(
        "ECG_CXR_BRIDGE_ARTIFACT_ROOT",
        "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts",
    ))
    config_path = ARTIFACT_ROOT / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run notebook 00 or 01 first; missing {config_path}")
    config = json.loads(config_path.read_text())

    additions = {
        "external_ecg_root": os.environ.get(
            "MIMIC_ECG_ROOT", "/data/liangz2/ecg_analysis/ecg_files/mimic-iv-ecg/1.0"),
        "external_cxr_root": os.environ.get(
            "MIMIC_CXR_TRAIN_ROOT",
            os.environ.get("MIMIC_CXR_JSON_ROOT", "/data/liangz2/openi/mimic_train")),
        "external_cxr_test_root": os.environ.get(
            "MIMIC_CXR_TEST_ROOT", "/data/liangz2/openi/mimic_test"),
        "text_model_id": os.environ.get(
            "CLINICAL_TEXT_MODEL_ID",
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
        "ablation_variants": [
            "clip_continuous", "shared_vq", "shared_specific_vq",
            "shared_specific_no_code_match", "shuffled_pairs_control",
        ],
        "ablation_seeds": [43, 44, 45],
        "external_splits": {"train": 80, "val": 10, "test": 10},
    }
    config.update(additions)
    for name in ["ablation", "external/ecg", "external/cxr", "external/text", "external/adapters", "reports/ablation"]:
        (ARTIFACT_ROOT / name).mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    print(json.dumps(additions, indent=2))
    """),
    code("""
    required = ["numpy", "pandas", "PIL", "torch", "tqdm", "transformers", "wfdb", "scipy", "open_clip"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing packages:", missing)
        print("Install in the Biowulf environment, for example: pip install wfdb scipy open_clip_torch")
    assert Path(config["external_ecg_root"]).exists(), config["external_ecg_root"]
    assert Path(config["external_cxr_root"]).exists(), config["external_cxr_root"]
    assert Path(config["external_cxr_test_root"]).exists(), config["external_cxr_test_root"]
    print("Configuration ready. Missing optional packages:", missing)
    """),
])


write("10_prepare_external_ecg_mimic.ipynb", [
    md("""
    # 10 - Prepare external MIMIC-IV-ECG waveform-diagnosis pairs

    This notebook uses only `record_list.csv` and `machine_measurements.csv` from MIMIC-IV-ECG. It concatenates `report_0` through `report_17` into the machine diagnosis, resolves each WFDB record stem, removes blank reports, and makes deterministic patient-disjoint splits.
    """),
    code("""
    from pathlib import Path
    import hashlib
    import json
    import os
    import re

    import pandas as pd

    ARTIFACT_ROOT = Path(os.environ.get("ECG_CXR_BRIDGE_ARTIFACT_ROOT", "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    config = json.loads((ARTIFACT_ROOT / "config.json").read_text())
    ECG_ROOT = Path(config["external_ecg_root"])
    OUT = ARTIFACT_ROOT / "external" / "ecg"
    OUT.mkdir(parents=True, exist_ok=True)
    RECORD_LIST = ECG_ROOT / "record_list.csv"
    MACHINE = ECG_ROOT / "machine_measurements.csv"
    for path in [RECORD_LIST, MACHINE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    print({"ecg_root": str(ECG_ROOT), "output": str(OUT)})
    """),
    code("""
    record_list = pd.read_csv(RECORD_LIST, low_memory=False)
    machine = pd.read_csv(MACHINE, low_memory=False)
    required_record = {"subject_id", "study_id", "file_name"}
    if missing := required_record - set(record_list.columns):
        raise ValueError(f"record_list.csv missing columns: {sorted(missing)}")
    report_cols = sorted(
        [c for c in machine.columns if c.startswith("report_")],
        key=lambda value: int(value.split("_")[-1]),
    )
    if not report_cols:
        raise ValueError("machine_measurements.csv has no report_0...report_N columns")

    def join_report(row):
        parts = []
        for col in report_cols:
            value = "" if pd.isna(row[col]) else str(row[col]).strip()
            if value and value.lower() != "nan" and value not in parts:
                parts.append(value)
        return " | ".join(parts)

    machine = machine[["study_id", *report_cols]].copy()
    machine["machine_diagnosis"] = machine.apply(join_report, axis=1)
    machine = machine.loc[machine["machine_diagnosis"].str.len().gt(0), ["study_id", "machine_diagnosis"]]
    machine = machine.drop_duplicates("study_id", keep="first")
    linked = record_list.merge(machine, on="study_id", how="inner", validate="one_to_one")
    print({"record_rows": len(record_list), "diagnosis_rows": len(machine), "linked_rows": len(linked)})
    """),
    code("""
    RELEASE_PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"

    # record_list.csv may contain an absolute path, files/p.../s.../study_id,
    # p.../s.../study_id, or a path carrying the old release-directory prefix.
    # Resolve the existing .hea file instead of assuming one extraction layout.
    SEARCH_BASES = []
    for candidate in [
        ECG_ROOT,
        ECG_ROOT / RELEASE_PREFIX,
        ECG_ROOT.parent,
        ECG_ROOT.parent / RELEASE_PREFIX,
        ECG_ROOT.parent.parent,
        ECG_ROOT.parent.parent / RELEASE_PREFIX,
    ]:
        candidate = candidate.resolve()
        if candidate not in SEARCH_BASES:
            SEARCH_BASES.append(candidate)

    def without_wfdb_suffix(path):
        return path.with_suffix("") if path.suffix.lower() in {".hea", ".dat"} else path

    def identifier_string(value):
        # CSV readers may represent an integer identifier as 40689238.0.
        try:
            return str(int(value))
        except (TypeError, ValueError, OverflowError):
            return str(value).strip()

    def waveform_candidates(row):
        subject_id = identifier_string(row["subject_id"])
        study_id = identifier_string(row["study_id"])
        raw_value = identifier_string(row["file_name"])
        raw_path = without_wfdb_suffix(Path(raw_value))
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)

        # Canonical MIMIC-IV-ECG layout, e.g.:
        # files/p1000/p10000032/s40689238/40689238
        patient_folder = f"p{subject_id}"
        patient_group = f"p{subject_id[:4]}"
        study_folder = f"s{study_id}"
        record_basename = raw_path.name
        canonical_relative = Path("files") / patient_group / patient_folder / study_folder / record_basename

        parts = raw_path.parts
        if "files" in parts:
            # Keep only the stable release-relative suffix: files/p.../s.../study_id.
            raw_relative = Path(*parts[parts.index("files"):])
        else:
            cleaned = [part for part in parts if part not in {RELEASE_PREFIX, "/"}]
            raw_relative = Path(*cleaned) if cleaned else raw_path
            if raw_relative.parts and raw_relative.parts[0].startswith("p"):
                raw_relative = Path("files") / raw_relative

        for base in SEARCH_BASES:
            # Try the canonical identifier-derived path first. The raw path is
            # retained as a fallback for releases whose file_name is already relative.
            candidates.append(without_wfdb_suffix(base / canonical_relative))
            candidates.append(without_wfdb_suffix(base / raw_relative))

        unique = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return unique

    def waveform_stem(row):
        candidates = waveform_candidates(row)
        for candidate in candidates:
            if candidate.with_suffix(".hea").is_file():
                return candidate
        # Return the preferred normalized location for a useful diagnostic below.
        return candidates[0]

    def stable_split(subject_id):
        bucket = int(hashlib.sha1(str(int(subject_id)).encode()).hexdigest()[:8], 16) % 100
        return "train" if bucket < 80 else ("val" if bucket < 90 else "test")

    linked["waveform_path"] = linked.apply(waveform_stem, axis=1).map(str)
    linked["split"] = linked["subject_id"].map(stable_split)
    linked["sample_id"] = linked["study_id"].map(lambda value: f"ecg_{int(value)}")
    linked["waveform_exists"] = linked["waveform_path"].map(lambda value: Path(value).with_suffix(".hea").is_file())
    missing_waveforms = int((~linked["waveform_exists"]).sum())
    print("Resolved WFDB headers:", int(linked["waveform_exists"].sum()))
    print("Missing WFDB headers:", missing_waveforms)

    if linked["waveform_exists"].sum() == 0:
        examples = linked[["subject_id", "study_id", "file_name", "waveform_path"]].head(10).to_dict("records")
        searched = [str(path) for path in SEARCH_BASES]
        raise FileNotFoundError(
            "No WFDB .hea files were resolved. This caused the earlier split assertion. "
            f"ECG_ROOT={ECG_ROOT}; searched_bases={searched}; examples={examples}"
        )

    manifest = linked.loc[linked["waveform_exists"], [
        "sample_id", "subject_id", "study_id", "waveform_path", "machine_diagnosis", "split"
    ]].copy()
    if not manifest["sample_id"].is_unique:
        duplicates = manifest.loc[manifest["sample_id"].duplicated(False), "sample_id"].head(20).tolist()
        raise ValueError(f"Duplicate ECG sample_id values: {duplicates}")
    split_counts = manifest.groupby("subject_id", dropna=False)["split"].nunique()
    leaking_subjects = split_counts[split_counts > 1]
    if not leaking_subjects.empty:
        raise AssertionError(f"Subjects assigned to multiple splits: {leaking_subjects.head(20).to_dict()}")
    print(manifest.groupby("split").agg(rows=("sample_id", "size"), subjects=("subject_id", "nunique")))
    """),
    code("""
    output = OUT / "manifest.jsonl"
    with output.open("w") as stream:
        for row in manifest.to_dict("records"):
            stream.write(json.dumps(row) + "\\n")
    summary = {
        "rows": len(manifest), "subjects": int(manifest.subject_id.nunique()),
        "splits": manifest.split.value_counts().to_dict(),
        "source_record_list": str(RECORD_LIST), "source_machine_measurements": str(MACHINE),
        "missing_waveforms_removed": int((~linked["waveform_exists"]).sum()),
    }
    (OUT / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Saved:", output)
    """),
])


write("11_prepare_external_cxr_mimic.ipynb", [
    md("""
    # 11 - Prepare external CXR image-report pairs

    The CXR release has two explicit partitions:

    - training: `/data/liangz2/openi/mimic_train`
    - test: `/data/liangz2/openi/mimic_test`

    The provided test partition remains untouched. The provided training partition is divided patient-wise into training and validation subsets. If a patient occurs in both source folders, test takes precedence and all records for that patient are assigned to the test split. JSON field lookup is case-insensitive; the primary report field is `Caption`, with `FINDINGS` and `IMPRESSION` as fallbacks.
    """),
    code("""
    from pathlib import Path
    import hashlib
    import json
    import os
    import re

    import pandas as pd
    from tqdm.auto import tqdm

    ARTIFACT_ROOT = Path(os.environ.get("ECG_CXR_BRIDGE_ARTIFACT_ROOT", "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    config = json.loads((ARTIFACT_ROOT / "config.json").read_text())
    CXR_TRAIN_ROOT = Path(os.environ.get("MIMIC_CXR_TRAIN_ROOT", config.get("external_cxr_root", "/data/liangz2/openi/mimic_train")))
    CXR_TEST_ROOT = Path(os.environ.get("MIMIC_CXR_TEST_ROOT", config.get("external_cxr_test_root", "/data/liangz2/openi/mimic_test")))
    OUT = ARTIFACT_ROOT / "external" / "cxr"
    OUT.mkdir(parents=True, exist_ok=True)
    ROOTS = {"provided_train": CXR_TRAIN_ROOT, "provided_test": CXR_TEST_ROOT}
    for name, root in ROOTS.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Missing {name} CXR directory: {root}")

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def discover_pairs(root):
        images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        json_files = sorted(path for path in root.rglob("*") if path.suffix.lower() == ".json")
        json_by_stem = {}
        duplicate_stems = []
        for path in json_files:
            key = path.relative_to(root).with_suffix("").as_posix().lower()
            if key in json_by_stem:
                duplicate_stems.append(key)
            else:
                json_by_stem[key] = path
        pairs = []
        for image_path in images:
            key = image_path.relative_to(root).with_suffix("").as_posix().lower()
            if key in json_by_stem:
                pairs.append((image_path, json_by_stem[key]))
        return pairs, {"images": len(images), "json_files": len(json_files),
                       "pairs": len(pairs), "unpaired_images": len(images) - len(pairs),
                       "duplicate_json_stems": duplicate_stems}

    discovered = {}
    discovery_stats = {}
    for partition, root in ROOTS.items():
        discovered[partition], discovery_stats[partition] = discover_pairs(root)
        if not discovered[partition]:
            raise RuntimeError(f"No same-stem image/JSON pairs under {root}")
    print(json.dumps({name: {**stats, "root": str(ROOTS[name])}
                      for name, stats in discovery_stats.items()}, indent=2))
    """),
    code("""
    def casefold_map(meta):
        return {str(key).casefold(): value for key, value in meta.items()}

    def first(meta, keys):
        folded = casefold_map(meta)
        for key in keys:
            if key.casefold() in folded:
                value = folded[key.casefold()]
                if value is not None and not (isinstance(value, str) and not value.strip()):
                    return value
        return None

    def text_value(value):
        if value is None:
            return ""
        if isinstance(value, str):
            return " ".join(value.split())
        if isinstance(value, (list, tuple)):
            return " ".join(part for part in (text_value(item) for item in value) if part)
        if isinstance(value, dict):
            folded = casefold_map(value)
            ordered = [folded.get(key) for key in ["findings", "impression", "text", "caption"] if key in folded]
            return " ".join(part for part in (text_value(item) for item in ordered) if part)
        return str(value).strip()

    def report_text_and_source(meta):
        folded = casefold_map(meta)
        for key in ["caption", "report_text", "report", "text"]:
            if key in folded and text_value(folded[key]):
                return text_value(folded[key]), key
        parts = [text_value(folded.get(key)) for key in ["findings", "impression"]]
        text = " ".join(part for part in parts if part)
        return text, "findings+impression" if text else ""

    def integer_or_none(value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        match = re.search(r"(\d+)$", str(value).strip())
        return int(match.group(1)) if match else None

    def normalized_labels(value):
        if value is None:
            return []
        if isinstance(value, dict):
            return sorted(str(key) for key, present in value.items()
                          if present not in (None, False, 0, 0.0, "", "0", "false", "False"))
        if isinstance(value, str):
            return [part.strip() for part in re.split(r"[,;|]", value) if part.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]

    rows, bad_json = [], []
    blank_reports = {name: 0 for name in ROOTS}
    for partition, pairs in discovered.items():
        root = ROOTS[partition]
        for image_path, json_path in tqdm(pairs, desc=partition):
            try:
                meta = json.loads(json_path.read_text())
                if not isinstance(meta, dict):
                    raise TypeError(f"Expected a JSON object, got {type(meta).__name__}")
            except Exception as exc:
                bad_json.append({"partition": partition, "path": str(json_path), "error": repr(exc)})
                continue
            text, report_source = report_text_and_source(meta)
            if not text:
                blank_reports[partition] += 1
                continue
            relative = image_path.relative_to(root).with_suffix("").as_posix()
            study_id = integer_or_none(first(meta, ["study_id", "dicom_study_id"]))
            rows.append({
                "sample_id": "cxr_" + relative.replace("/", "__"),
                "image_path": str(image_path), "json_path": str(json_path),
                "report_text": text, "report_source": report_source,
                "labels": normalized_labels(first(meta, ["labels", "chexpert_labels", "positive_labels"])),
                "normal": first(meta, ["normal"]), "view_position": first(meta, ["view_position"]),
                "subject_id": integer_or_none(first(meta, ["subject_id", "patient_id", "patientId"])),
                "hadm_id": integer_or_none(first(meta, ["hadm_id", "admission_id", "encounter_id"])),
                "study_id": study_id, "source_partition": partition,
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"All JSON reports were blank or invalid; blank={blank_reports}, bad_json={bad_json[:10]}")
    if not frame.sample_id.is_unique:
        duplicates = frame.loc[frame.sample_id.duplicated(False), ["sample_id", "source_partition", "json_path"]]
        raise ValueError(f"Duplicate CXR sample IDs across partitions: {duplicates.head(20).to_dict('records')}")
    print({"usable_pairs": len(frame), "bad_json": len(bad_json), "blank_reports": blank_reports,
           "report_sources": frame.report_source.value_counts().to_dict(),
           "with_subject_id": int(frame.subject_id.notna().sum()),
           "with_study_id": int(frame.study_id.notna().sum())})
    display(frame[["sample_id", "subject_id", "study_id", "labels", "report_source", "source_partition"]].head())
    """),
    code("""
    def stable_bucket(value):
        return int(hashlib.sha1(str(value).encode()).hexdigest()[:8], 16) % 100

    VAL_PERCENT = int(os.environ.get("CXR_VALIDATION_PERCENT", "10"))
    if not 1 <= VAL_PERCENT <= 50:
        raise ValueError("CXR_VALIDATION_PERCENT must be between 1 and 50")

    frame["split"] = "test"
    provided_train = frame.source_partition.eq("provided_train")
    split_keys = []
    for row in frame.loc[provided_train].itertuples():
        if pd.notna(row.subject_id):
            split_keys.append(str(int(row.subject_id)))
        elif pd.notna(row.hadm_id):
            split_keys.append(f"admission:{int(row.hadm_id)}")
        else:
            split_keys.append(f"sample:{row.sample_id}")
    buckets = pd.Series(split_keys, index=frame.index[provided_train]).map(stable_bucket)
    frame.loc[provided_train, "split"] = buckets.map(
        lambda value: "val" if value >= 100 - VAL_PERCENT else "train")

    # Enforce patient-disjoint model splits. The source folders themselves can overlap;
    # in that case, the supplied test partition takes precedence for the whole patient.
    train_subjects = set(frame.loc[frame.source_partition.eq("provided_train"), "subject_id"].dropna().astype("int64"))
    test_subjects = set(frame.loc[frame.source_partition.eq("provided_test"), "subject_id"].dropna().astype("int64"))
    source_overlap = train_subjects & test_subjects
    overlap_train_rows = provided_train & frame.subject_id.isin(source_overlap)
    reassigned_overlap_rows = int(overlap_train_rows.sum())
    frame.loc[overlap_train_rows, "split"] = "test"
    known_subjects = frame.loc[frame.subject_id.notna()]
    split_counts = known_subjects.groupby("subject_id")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if not leaking.empty:
        raise AssertionError(f"CXR subjects assigned to multiple splits: {leaking.head(20).to_dict()}")

    split_basis = (
        "provided test partition with patient-level test precedence for source-folder overlaps; "
        "patient-hash train/validation split for remaining provided-training patients")
    print(frame.groupby("split").agg(rows=("sample_id", "size"), subjects=("subject_id", "nunique")))
    print({"validation_percent_of_provided_train": VAL_PERCENT,
           "source_overlap_subjects": len(source_overlap),
           "provided_train_rows_reassigned_to_test": reassigned_overlap_rows,
           "source_overlap_examples": [int(value) for value in sorted(source_overlap)[:20]],
           "split_basis": split_basis})
    """),
    code("""
    def json_safe(value):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [json_safe(item) for item in value]
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass
        return value

    output = OUT / "manifest.jsonl"
    with output.open("w") as stream:
        for row in frame.to_dict("records"):
            stream.write(json.dumps(json_safe(row)) + "\\n")
    summary = {
        "rows": len(frame),
        "splits": {str(key): int(value) for key, value in frame.split.value_counts().items()},
        "source_partitions": {str(key): int(value) for key, value in frame.source_partition.value_counts().items()},
        "split_basis": split_basis, "validation_percent": VAL_PERCENT,
        "source_overlap_subjects": len(source_overlap),
        "provided_train_rows_reassigned_to_test": reassigned_overlap_rows,
        "source_overlap_examples": [int(value) for value in sorted(source_overlap)[:20]],
        "with_subject_id": int(frame.subject_id.notna().sum()),
        "with_hadm_id": int(frame.hadm_id.notna().sum()),
        "with_study_id": int(frame.study_id.notna().sum()),
        "report_sources": {str(key): int(value) for key, value in frame.report_source.value_counts().items()},
        "blank_reports": blank_reports,
        "discovery": discovery_stats,
        "bad_json": bad_json[:20],
    }
    (OUT / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Saved:", output)
    """),
])


write("12_train_symile_bridge_ablations.ipynb", [
    md("""
    # 12 - Train Symile bridge ablations

    All variants use identical HuBERT-ECG and RAD-DINO features, patient-disjoint Symile splits, optimizer settings, and seeds. The shuffled-pair run is a negative control. By default all configured variants and three seeds run; environment variables can restrict a smoke test.

    Shared-specific variants reconstruct normalized foundation features from their quantized shared and specific components. This prevents the specific branch from becoming an unused orthogonal residual. Checkpoints are selected by mean bidirectional Recall@1 on the ordinary Symile validation split; the test split is evaluated only after training.

    This notebook uses only the Symile manifests and foundation features produced by notebooks 02-04. It does not consume the external ECG or CXR manifests from notebooks 10-11; those are first used by notebook 13.

    - `ABLATION_VARIANTS=clip_continuous,shared_vq`
    - `ABLATION_SEEDS=43`
    - `ABLATION_EPOCHS=2`
    """),
    code("""
    from pathlib import Path
    import importlib
    import json
    import os
    import random
    import sys
    import time

    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader, Dataset

    ARTIFACT_ROOT = Path(os.environ.get("ECG_CXR_BRIDGE_ARTIFACT_ROOT", "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    FEATURE_ROOT, CODE_ROOT = ARTIFACT_ROOT / "features", ARTIFACT_ROOT / "code"
    CODE_ROOT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ARTIFACT_ROOT / "config.json").read_text())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({"device": str(DEVICE), "artifact_root": str(ARTIFACT_ROOT)})
    """),
    code("""
    module_source = r'''
    from dataclasses import dataclass
    import torch
    from torch import nn
    from torch.nn import functional as F

    MODALITIES = ("ecg", "cxr")

    class CompositionalCodebook(nn.Module):
        def __init__(self, size, code_dim, temperature=0.1):
            super().__init__()
            self.code_dim, self.temperature = int(code_dim), float(temperature)
            self.codes = nn.Parameter(torch.randn(size, code_dim))
            nn.init.normal_(self.codes, std=code_dim ** -0.5)
        def forward(self, features):
            chunks = features.reshape(features.shape[0], -1, self.code_dim)
            logits = torch.einsum("bnd,kd->bnk", F.normalize(chunks, dim=-1), F.normalize(self.codes, dim=-1))
            indices = logits.argmax(-1)
            hard = F.embedding(indices, self.codes)
            commitment = F.mse_loss(chunks, hard.detach()) + F.mse_loss(hard, chunks.detach())
            quantized = chunks + (hard - chunks).detach()
            # Retain the compositional-chunk axis [batch, chunks, codes].
            # Averaging chunks before code matching destroys positional code information.
            probabilities = F.softmax(logits / self.temperature, dim=-1)
            return quantized.flatten(1), indices, probabilities, commitment

    @dataclass
    class Output:
        common: torch.Tensor
        continuous_common: torch.Tensor
        common_indices: object = None
        common_probabilities: object = None
        specific: object = None
        continuous_specific: object = None
        specific_indices: object = None
        commitment: object = None
        reconstruction: object = None

    class AblationBridge(nn.Module):
        def __init__(self, input_dims, variant, common_dim=256, specific_dim=256, code_dim=8,
                     common_codebook_size=512, specific_codebook_size=256,
                     code_temperature=0.1, dropout=0.1):
            super().__init__()
            allowed = {"clip_continuous", "shared_vq", "shared_specific_vq",
                       "shared_specific_no_code_match", "shuffled_pairs_control"}
            if variant not in allowed:
                raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(allowed)}")
            if common_dim % code_dim or (variant in {"shared_specific_vq", "shared_specific_no_code_match", "shuffled_pairs_control"} and specific_dim % code_dim):
                raise ValueError("common_dim and specific_dim must be divisible by code_dim")
            self.variant, self.common_dim = variant, int(common_dim)
            self.uses_vq = variant != "clip_continuous"
            self.uses_specific = variant in {"shared_specific_vq", "shared_specific_no_code_match", "shuffled_pairs_control"}
            output_dim = common_dim + (specific_dim if self.uses_specific else 0)
            self.projectors = nn.ModuleDict({name: nn.Sequential(
                nn.LayerNorm(input_dims[name]), nn.Dropout(dropout), nn.Linear(input_dims[name], output_dim)
            ) for name in MODALITIES})
            self.common_codebook = CompositionalCodebook(common_codebook_size, code_dim, code_temperature) if self.uses_vq else None
            self.specific_codebooks = nn.ModuleDict({name: CompositionalCodebook(
                specific_codebook_size, code_dim, code_temperature) for name in MODALITIES}) if self.uses_specific else None
            self.decoders = nn.ModuleDict({name: nn.Sequential(
                nn.LayerNorm(common_dim + specific_dim), nn.Linear(common_dim + specific_dim, input_dims[name])
            ) for name in MODALITIES}) if self.uses_specific else None

        def encode_modality(self, name, features):
            projected = self.projectors[name](features)
            continuous_common = projected[:, :self.common_dim]
            commitments = []
            if self.uses_vq:
                common, common_indices, probabilities, commitment = self.common_codebook(continuous_common)
                commitments.append(commitment)
            else:
                common, common_indices, probabilities = continuous_common, None, None
                commitment = continuous_common.sum() * 0.0
            specific = continuous_specific = specific_indices = reconstruction = None
            if self.uses_specific:
                continuous_specific = projected[:, self.common_dim:]
                specific_raw, specific_indices, _, specific_commitment = self.specific_codebooks[name](continuous_specific)
                commitments.append(specific_commitment)
                reconstruction = self.decoders[name](torch.cat([common, specific_raw], dim=-1))
                specific = F.normalize(specific_raw, dim=-1)
            if commitments:
                # Average within modality so full CodeBind is not penalized twice
                # merely because it has both shared and specific quantizers.
                commitment = torch.stack(commitments).mean()
            return Output(F.normalize(common, dim=-1), continuous_common, common_indices,
                          probabilities, specific, continuous_specific, specific_indices,
                          commitment, reconstruction)

        def forward(self, batch):
            return {name: self.encode_modality(name, batch[name]) for name in MODALITIES}

    def symmetric_info_nce(left, right, subject_id, temperature):
        logits = left @ right.T / temperature
        same = subject_id[:, None].eq(subject_id[None, :])
        offdiag = ~torch.eye(len(subject_id), dtype=torch.bool, device=left.device)
        logits = logits.masked_fill(same & offdiag, -torch.inf)
        target = torch.arange(len(left), device=left.device)
        return .5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))

    def js(left, right):
        eps = torch.finfo(left.dtype).eps
        middle = .5 * (left + right)
        return .5 * ((left * ((left + eps).log() - (middle + eps).log())).sum(-1)
                     + (right * ((right + eps).log() - (middle + eps).log())).sum(-1)).mean()

    def orthogonality(shared, specific):
        if len(shared) < 2:
            return shared.sum() * 0.0
        shared = (shared - shared.mean(0)) / shared.std(0, unbiased=False).clamp_min(1e-4)
        specific = (specific - specific.mean(0)) / specific.std(0, unbiased=False).clamp_min(1e-4)
        cross_correlation = shared.T @ specific / len(shared)
        return cross_correlation.square().mean()

    def loss(model, outputs, batch, weights):
        zero = outputs["ecg"].common.sum() * 0.0
        alignment = symmetric_info_nce(outputs["ecg"].common, outputs["cxr"].common,
                                        batch["subject_id"], weights["temperature"])
        relation = js(outputs["ecg"].common_probabilities, outputs["cxr"].common_probabilities) if model.uses_vq else zero
        commitment = torch.stack([outputs[m].commitment for m in MODALITIES]).mean() if model.uses_vq else zero
        if model.uses_specific:
            orth = torch.stack([orthogonality(outputs[m].continuous_common,
                                              outputs[m].continuous_specific) for m in MODALITIES]).mean()
            reconstruction = torch.stack([F.mse_loss(
                outputs[m].reconstruction,
                F.layer_norm(batch[m], (batch[m].shape[-1],))) for m in MODALITIES]).mean()
        else:
            orth = zero
            reconstruction = zero
        if model.uses_vq:
            probs = torch.stack([outputs[m].common_probabilities.mean(0) for m in MODALITIES])
            usage = (probs * (probs + torch.finfo(probs.dtype).eps).log()).sum(-1).mean()
        else:
            usage = zero
        relation_weight = 0.0 if model.variant == "shared_specific_no_code_match" else weights["code_relation_weight"]
        total = (alignment + relation_weight * relation
                 + weights["commitment_weight"] * commitment
                 + weights["orthogonality_weight"] * orth
                 + weights["reconstruction_weight"] * reconstruction
                 + weights["code_usage_weight"] * usage)
        return total, {"total": total, "alignment": alignment, "code_relation": relation,
                       "commitment": commitment, "orthogonality": orth,
                       "reconstruction": reconstruction, "code_usage": usage}
    '''
    module_path = CODE_ROOT / "ecg_cxr_ablation.py"
    module_path.write_text(module_source)
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    import ecg_cxr_ablation
    importlib.reload(ecg_cxr_ablation)
    print("Wrote and loaded:", module_path)
    """),
    code("""
    class PairDataset(Dataset):
        def __init__(self, split, shuffle_cxr=False, seed=43):
            paths = {"ecg": FEATURE_ROOT / split / "ecg_hubert.npy",
                     "cxr": FEATURE_ROOT / split / "cxr_raddino.npy",
                     "manifest": ARTIFACT_ROOT / "manifests" / f"{split}.jsonl"}
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError("Run notebooks 02-04 first; missing:\\n" + "\\n".join(missing))
            self.ecg = np.load(paths["ecg"], mmap_mode="r")
            self.cxr = np.load(paths["cxr"], mmap_mode="r")
            manifest = [json.loads(line) for line in paths["manifest"].read_text().splitlines() if line.strip()]
            if not manifest:
                raise ValueError(f"Empty manifest: {paths['manifest']}")
            if self.ecg.ndim != 2 or self.cxr.ndim != 2:
                raise ValueError(f"Expected 2-D feature arrays, got ECG {self.ecg.shape}, CXR {self.cxr.shape}")
            if len(self.ecg) != len(manifest) or len(self.cxr) != len(manifest):
                raise ValueError(f"{split} row mismatch: ECG={len(self.ecg)}, CXR={len(self.cxr)}, manifest={len(manifest)}")
            self.manifest = manifest
            self.subject = np.asarray([row["subject_id"] for row in manifest], dtype="int64")
            self.cxr_index = np.arange(len(manifest))
            if shuffle_cxr:
                rng = np.random.default_rng(seed)
                largest_patient_count = int(np.unique(self.subject, return_counts=True)[1].max())
                if largest_patient_count > len(self.subject) // 2:
                    raise RuntimeError("A cross-patient derangement is impossible because one patient owns more than half the rows")
                for _ in range(1000):
                    candidate = rng.permutation(len(manifest))
                    if np.all(self.subject != self.subject[candidate]):
                        self.cxr_index = candidate
                        break
                else:
                    raise RuntimeError("Could not construct a deranged cross-patient CXR permutation")
        def __len__(self): return len(self.subject)
        def __getitem__(self, index):
            return {"ecg": torch.tensor(np.array(self.ecg[index], dtype="float32")),
                    "cxr": torch.tensor(np.array(self.cxr[self.cxr_index[index]], dtype="float32")),
                    "subject_id": torch.tensor(self.subject[index], dtype=torch.long)}

    DEFAULT_VARIANTS = ["clip_continuous", "shared_vq", "shared_specific_vq",
                        "shared_specific_no_code_match", "shuffled_pairs_control"]
    VARIANTS = [value.strip() for value in os.environ.get(
        "ABLATION_VARIANTS", ",".join(config.get("ablation_variants", DEFAULT_VARIANTS))).split(",") if value.strip()]
    unknown_variants = sorted(set(VARIANTS) - set(DEFAULT_VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown ABLATION_VARIANTS: {unknown_variants}")
    default_seeds = config.get("ablation_seeds", [43, 44, 45])
    SEEDS = [int(value) for value in os.environ.get("ABLATION_SEEDS", ",".join(map(str, default_seeds))).split(",")]
    EPOCHS = int(os.environ.get("ABLATION_EPOCHS", "50"))
    BATCH_SIZE = int(os.environ.get("ABLATION_BATCH_SIZE", "256"))
    NUM_WORKERS = int(os.environ.get("ABLATION_NUM_WORKERS", "4"))
    PATIENCE = int(os.environ.get("ABLATION_PATIENCE", "8"))
    LEARNING_RATE = float(os.environ.get("ABLATION_LR", "3e-4"))
    WEIGHT_DECAY = float(os.environ.get("ABLATION_WEIGHT_DECAY", "1e-4"))
    if not VARIANTS:
        raise ValueError("ABLATION_VARIANTS must contain at least one variant")
    if not SEEDS:
        raise ValueError("ABLATION_SEEDS must contain at least one integer seed")
    if EPOCHS < 1 or PATIENCE < 1:
        raise ValueError("ABLATION_EPOCHS and ABLATION_PATIENCE must both be at least 1")
    if NUM_WORKERS < 0:
        raise ValueError("ABLATION_NUM_WORKERS cannot be negative")
    if BATCH_SIZE < 2:
        raise ValueError("ABLATION_BATCH_SIZE must be at least 2 for InfoNCE")
    HPARAMS = {"common_dim": 256, "specific_dim": 256, "code_dim": 8,
               "common_codebook_size": 512, "specific_codebook_size": 256,
               "code_temperature": .1, "dropout": .1}
    WEIGHTS = {"temperature": .07, "code_relation_weight": .1, "commitment_weight": .25,
               "orthogonality_weight": .05, "reconstruction_weight": .10,
               "code_usage_weight": .01}
    # Validate every input before expensive multi-variant training begins.
    preflight = {split: PairDataset(split) for split in ["train", "val", "val_retrieval", "test"]}
    preflight_train, preflight_val = preflight["train"], preflight["val"]
    subject_overlap = set(preflight_train.subject.tolist()) & set(preflight_val.subject.tolist())
    if subject_overlap:
        raise AssertionError(f"Patient leakage between Symile train and val: {list(subject_overlap)[:20]}")

    retrieval_groups = {}
    for split in ["val_retrieval", "test"]:
        groups = {}
        for row_index, row in enumerate(preflight[split].manifest):
            missing_fields = [key for key in ["label_hadm_id", "label"] if key not in row]
            if missing_fields:
                raise KeyError(f"{split} manifest row {row_index} is missing {missing_fields}")
            groups.setdefault(int(row["label_hadm_id"]), []).append(int(row["label"]))
        invalid = {group_id: {"rows": len(labels), "positives": sum(label == 1 for label in labels)}
                   for group_id, labels in groups.items()
                   if len(labels) != 10 or sum(label == 1 for label in labels) != 1}
        if invalid:
            raise ValueError(f"Invalid 10-candidate groups in {split}: {dict(list(invalid.items())[:20])}")
        retrieval_groups[split] = len(groups)

    for name, dataset in preflight.items():
        sample_index = np.linspace(0, len(dataset) - 1, min(1024, len(dataset)), dtype=int)
        if not np.isfinite(np.asarray(dataset.ecg[sample_index])).all() or not np.isfinite(np.asarray(dataset.cxr[sample_index])).all():
            raise ValueError(f"Non-finite foundation features detected in {name}")
    print({"variants": VARIANTS, "seeds": SEEDS, "epochs": EPOCHS,
           "train_rows": len(preflight_train), "val_rows": len(preflight_val),
           "retrieval_groups": retrieval_groups,
           "patient_overlap": len(subject_overlap), "loss_weights": WEIGHTS})
    """),
    code("""
    def seed_all(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    def run_epoch(model, loader, optimizer=None):
        training = optimizer is not None
        model.train(training)
        rows = []
        with torch.set_grad_enabled(training):
            for batch in loader:
                batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
                if training: optimizer.zero_grad(set_to_none=True)
                total, components = ecg_cxr_ablation.loss(model, model(batch), batch, WEIGHTS)
                if training:
                    total.backward(); clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                rows.append({key: float(value.detach().cpu()) for key, value in components.items()})
        if not rows:
            raise RuntimeError("DataLoader produced no batches; reduce ABLATION_BATCH_SIZE")
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    @torch.inference_mode()
    def validation_retrieval(model, loader):
        model.eval(); collected = {"ecg": [], "cxr": [], "subject_id": []}
        for batch in loader:
            encoded = model({m: batch[m].to(DEVICE, non_blocking=True) for m in ["ecg", "cxr"]})
            for modality in ["ecg", "cxr"]:
                collected[modality].append(encoded[modality].common.cpu())
            collected["subject_id"].append(batch["subject_id"])
        values = {key: torch.cat(parts) for key, parts in collected.items()}
        similarity = F.normalize(values["ecg"].float(), dim=-1) @ F.normalize(values["cxr"].float(), dim=-1).T
        same_patient = values["subject_id"][:, None].eq(values["subject_id"][None, :])
        off_diagonal = ~torch.eye(len(similarity), dtype=torch.bool)
        similarity = similarity.masked_fill(same_patient & off_diagonal, -torch.inf)
        diagonal = similarity.diag()
        ecg_ranks = 1 + (similarity > diagonal[:, None]).sum(1)
        cxr_ranks = 1 + (similarity.T > diagonal[:, None]).sum(1)
        return {"ecg_to_cxr_r1": float((ecg_ranks == 1).float().mean()),
                "cxr_to_ecg_r1": float((cxr_ranks == 1).float().mean()),
                "mean_r1": float(.5 * ((ecg_ranks == 1).float().mean() + (cxr_ranks == 1).float().mean()))}

    experiment_index = []
    for variant in VARIANTS:
        for seed in SEEDS:
            seed_all(seed)
            shuffled = variant == "shuffled_pairs_control"
            train_ds = PairDataset("train", shuffle_cxr=shuffled, seed=seed)
            val_ds = PairDataset("val", shuffle_cxr=False, seed=seed)
            loader_generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                      drop_last=len(train_ds) >= BATCH_SIZE,
                                      num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
                                      persistent_workers=NUM_WORKERS > 0, generator=loader_generator)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
                                    persistent_workers=NUM_WORKERS > 0)
            input_dims = {"ecg": int(train_ds.ecg.shape[1]), "cxr": int(train_ds.cxr.shape[1])}
            model_variant = "shared_specific_vq" if shuffled else variant
            model = ecg_cxr_ablation.AblationBridge(input_dims, model_variant, **HPARAMS).to(DEVICE)
            if shuffled: model.variant = "shuffled_pairs_control"
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            run_dir = ARTIFACT_ROOT / "ablation" / variant / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            resolved = {"variant": variant, "model_variant": model_variant, "seed": seed,
                        "input_dims": input_dims, "model_hparams": HPARAMS, "loss_weights": WEIGHTS,
                        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))}
            resolved.update({"learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                             "batch_size": BATCH_SIZE, "selection_metric": "validation_mean_recall_at_1"})
            best_r1, best_loss, stale, history = -1.0, float("inf"), 0, []
            for epoch in range(EPOCHS):
                started = time.time()
                train_metrics = run_epoch(model, train_loader, optimizer)
                val_metrics = run_epoch(model, val_loader)
                val_retrieval = validation_retrieval(model, val_loader)
                row = {"epoch": epoch, "seconds": time.time() - started,
                       "train": train_metrics, "validation": val_metrics,
                       "validation_retrieval": val_retrieval}
                history.append(row); print(variant, seed, json.dumps(row))
                checkpoint = {"epoch": epoch, "model": model.state_dict(), "resolved_config": resolved}
                torch.save(checkpoint, run_dir / "last.pt")
                improved = (val_retrieval["mean_r1"] > best_r1
                            or (val_retrieval["mean_r1"] == best_r1 and val_metrics["total"] < best_loss))
                if improved:
                    best_r1, best_loss, stale = val_retrieval["mean_r1"], val_metrics["total"], 0
                    torch.save(checkpoint, run_dir / "best.pt")
                else:
                    stale += 1
                    if stale >= PATIENCE: break
            (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))
            experiment_index.append({"variant": variant, "seed": seed,
                                     "best_validation_retrieval_r1": best_r1,
                                     "best_validation_loss": best_loss,
                                     "checkpoint": str(run_dir / "best.pt")})
    index_path = ARTIFACT_ROOT / "ablation" / "experiment_index.json"
    index_path.write_text(json.dumps(experiment_index, indent=2))
    print("Saved:", index_path)
    """),
    code("""
    @torch.inference_mode()
    def encode_symile(checkpoint_path, split):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        resolved = checkpoint["resolved_config"]
        model = ecg_cxr_ablation.AblationBridge(resolved["input_dims"], resolved["model_variant"], **resolved["model_hparams"]).to(DEVICE).eval()
        model.load_state_dict(checkpoint["model"])
        dataset = PairDataset(split, shuffle_cxr=False)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        output = {"ecg": [], "cxr": []}
        for batch in loader:
            encoded = model({m: batch[m].to(DEVICE) for m in ["ecg", "cxr"]})
            for modality in ["ecg", "cxr"]: output[modality].append(encoded[modality].common.cpu())
        output = {key: torch.cat(value) for key, value in output.items()}
        manifest = [json.loads(line) for line in (ARTIFACT_ROOT / "manifests" / f"{split}.jsonl").read_text().splitlines()]
        output["label_hadm_id"] = torch.tensor([row["label_hadm_id"] for row in manifest])
        output["label"] = torch.tensor([row["label"] for row in manifest])
        return output

    def ten_candidate(encoded, query_name, candidate_name):
        ranks = []
        for hadm_id in torch.unique(encoded["label_hadm_id"]):
            group = torch.where(encoded["label_hadm_id"] == hadm_id)[0]
            positives = group[encoded["label"][group] == 1]
            if len(group) != 10 or len(positives) != 1: raise ValueError(f"Invalid group {int(hadm_id)}")
            positive = positives[0]
            query = F.normalize(encoded[query_name][positive].float(), dim=-1)
            scores = F.normalize(encoded[candidate_name][group].float(), dim=-1) @ query
            positive_position = torch.where(group == positive)[0][0]
            ranks.append(1 + int((scores > scores[positive_position]).sum()))
        ranks = np.asarray(ranks)
        return {"queries": len(ranks), "recall_at_1": float(np.mean(ranks <= 1)),
                "recall_at_5": float(np.mean(ranks <= 5)), "recall_at_10": float(np.mean(ranks <= 10)),
                "mrr": float(np.mean(1 / ranks)), "median_rank": float(np.median(ranks)), "chance_r1": .1}

    def supported_ten_candidate(encoded, query_name, candidate_name):
        unique_candidate_counts = []
        for hadm_id in torch.unique(encoded["label_hadm_id"]):
            group = torch.where(encoded["label_hadm_id"] == hadm_id)[0]
            unique_candidate_counts.append(int(torch.unique(encoded[candidate_name][group], dim=0).shape[0]))
        if min(unique_candidate_counts) <= 1:
            return {"skipped": True,
                    "reason": f"{candidate_name} is repeated within at least one 10-candidate group",
                    "minimum_unique_candidates": min(unique_candidate_counts)}
        result = ten_candidate(encoded, query_name, candidate_name)
        result.update({"skipped": False, "minimum_unique_candidates": min(unique_candidate_counts)})
        return result

    symile_results = []
    for experiment in experiment_index:
        row = {**experiment}
        for split in ["val_retrieval", "test"]:
            encoded = encode_symile(experiment["checkpoint"], split)
            row[split] = {"ecg_to_cxr": supported_ten_candidate(encoded, "ecg", "cxr"),
                          "cxr_to_ecg": supported_ten_candidate(encoded, "cxr", "ecg")}
        symile_results.append(row)
        print(experiment["variant"], experiment["seed"], row["val_retrieval"])
    metrics_path = ARTIFACT_ROOT / "reports" / "ablation" / "symile_bridge_ablation_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(symile_results, indent=2))
    summary = {}
    for variant in sorted({row["variant"] for row in symile_results}):
        subset = [row for row in symile_results if row["variant"] == variant]
        summary[variant] = {"seeds": len(subset)}
        for split in ["val_retrieval", "test"]:
            summary[variant][split] = {}
            for direction in ["ecg_to_cxr", "cxr_to_ecg"]:
                values = [row[split][direction]["recall_at_1"] for row in subset
                          if not row[split][direction].get("skipped", False)]
                if values:
                    summary[variant][split][direction] = {
                        "runs": len(values), "recall_at_1_mean": float(np.mean(values)),
                        "recall_at_1_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    }
                else:
                    summary[variant][split][direction] = {"runs": 0, "skipped": True,
                                                          "reason": "candidate modality is repeated within groups"}
    summary_path = ARTIFACT_ROOT / "reports" / "ablation" / "symile_bridge_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Saved:", metrics_path, summary_path)
    """),
])


write("13_encode_external_modalities_with_shared_codebook.ipynb", [
    md("""
    # 13 - Encode external ECG and CXR data with the Symile bridge

    This notebook first creates frozen HuBERT-ECG and RAD-DINO features, then applies every trained bridge ablation. It writes `continuous_common`, final normalized `common`, and shared code indices where applicable. HuBERT and RAD-DINO are loaded sequentially to limit GPU memory use. Status arrays make failed records explicit and allow a rerun to resume.

    Set `EXTERNAL_MAX_ROWS` to a small positive number for a smoke test. Arrays still retain the complete manifest length, with unprocessed rows marked by status `0`, so a later run with `EXTERNAL_MAX_ROWS=0` safely resumes the remainder. Set `EXTERNAL_RETRY_FAILED=1` to retry rows previously marked `-1`.
    """),
    code("""
    from pathlib import Path
    import gc
    import hashlib
    import importlib
    import json
    import os
    import sys
    from fractions import Fraction

    import numpy as np
    from PIL import Image
    from scipy.signal import resample_poly
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoImageProcessor, AutoModel
    import wfdb

    ARTIFACT_ROOT = Path(os.environ.get("ECG_CXR_BRIDGE_ARTIFACT_ROOT", "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    config = json.loads((ARTIFACT_ROOT / "config.json").read_text())
    CODE_ROOT = ARTIFACT_ROOT / "code"
    module_path = CODE_ROOT / "ecg_cxr_ablation.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Run notebook 12 first; missing {module_path}")
    if str(CODE_ROOT) not in sys.path: sys.path.insert(0, str(CODE_ROOT))
    import ecg_cxr_ablation
    importlib.reload(ecg_cxr_ablation)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
    MAX_ROWS = int(os.environ.get("EXTERNAL_MAX_ROWS", "0"))
    RETRY_FAILED = os.environ.get("EXTERNAL_RETRY_FAILED", "0").strip().lower() in {"1", "true", "yes"}
    if MAX_ROWS < 0:
        raise ValueError("EXTERNAL_MAX_ROWS cannot be negative")

    def read_manifest(modality):
        path = ARTIFACT_ROOT / "external" / modality / "manifest.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Run notebook {'10' if modality == 'ecg' else '11'} first; missing {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty {modality} manifest: {path}")
        required = ["sample_id", "split", "waveform_path" if modality == "ecg" else "image_path"]
        invalid = [(index, [key for key in required if key not in row or row[key] in (None, "")])
                   for index, row in enumerate(rows)]
        invalid = [(index, missing) for index, missing in invalid if missing]
        if invalid:
            raise KeyError(f"{modality} manifest rows missing required fields: {invalid[:20]}")
        sample_ids = [str(row["sample_id"]) for row in rows]
        seen, duplicates = set(), set()
        for value in sample_ids:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        if duplicates:
            raise ValueError(f"Duplicate {modality} sample_id values: {sorted(duplicates)[:20]}")
        return rows

    def manifest_sha256(rows):
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    manifests = {name: read_manifest(name) for name in ["ecg", "cxr"]}
    manifest_hashes = {name: manifest_sha256(rows) for name, rows in manifests.items()}
    print({name: {"rows": len(rows), "manifest_sha256": manifest_hashes[name][:12]}
           for name, rows in manifests.items()}, "device", DEVICE,
          "max_rows", MAX_ROWS, "retry_failed", RETRY_FAILED)
    """),
    code("""
    LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    PREPROCESSING_VERSION = "external_foundation_v2"

    def load_ecg(stem):
        record = wfdb.rdrecord(str(stem), physical=True)
        signal = np.asarray(record.p_signal, dtype="float32")
        names = [str(value).upper().replace(" ", "") for value in record.sig_name]
        if not all(lead in names for lead in LEADS):
            raise ValueError(f"Missing canonical leads: {record.sig_name}")
        signal = signal[:, [names.index(lead) for lead in LEADS]]
        fs = float(record.fs)
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"Invalid ECG sampling frequency: {record.fs}")
        if not np.isclose(fs, 500.0):
            ratio = (Fraction(500, 1) / Fraction(str(fs))).limit_denominator(1000)
            signal = resample_poly(signal, ratio.numerator, ratio.denominator, axis=0).astype("float32")
        if len(signal) < 5000:
            signal = np.pad(signal, ((0, 5000 - len(signal)), (0, 0)))
        signal = signal[:5000]
        mean, std = np.nanmean(signal, 0, keepdims=True), np.nanstd(signal, 0, keepdims=True)
        std[std < 1e-6] = 1
        return np.nan_to_num((signal - mean) / std).T.reshape(-1).astype("float32")

    def load_cxr(path):
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def load_input(modality, row):
        return load_ecg(row["waveform_path"]) if modality == "ecg" else load_cxr(row["image_path"])

    def load_foundation_encoder(modality):
        if modality == "ecg":
            model = AutoModel.from_pretrained(
                config["hubert_model_id"], cache_dir=config["hf_cache"],
                trust_remote_code=True, torch_dtype=DTYPE,
            ).to(DEVICE).eval().requires_grad_(False)
            return {"model": model}
        processor = AutoImageProcessor.from_pretrained(config["cxr_model_id"], cache_dir=config["hf_cache"])
        model = AutoModel.from_pretrained(
            config["cxr_model_id"], cache_dir=config["hf_cache"], torch_dtype=DTYPE,
        ).to(DEVICE).eval().requires_grad_(False)
        return {"model": model, "processor": processor}

    @torch.inference_mode()
    def encode_foundation(modality, inputs, bundle):
        if modality == "ecg":
            values = torch.from_numpy(np.stack(inputs)).to(DEVICE, dtype=DTYPE)
            output = bundle["model"](input_values=values)
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
            pooled = hidden.mean(1) if hidden.ndim == 3 else hidden
        else:
            pixels = bundle["processor"](images=inputs, return_tensors="pt")["pixel_values"]
            output = bundle["model"](pixel_values=pixels.to(DEVICE, dtype=DTYPE))
            pooled = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]
        encoded = pooled.float().cpu().numpy()
        if encoded.ndim != 2 or len(encoded) != len(inputs) or not np.isfinite(encoded).all():
            raise ValueError(f"Invalid {modality} encoder output: shape={encoded.shape}")
        return encoded
    """),
    code("""
    def read_metadata(path):
        try:
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def read_errors(path):
        try:
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def resumable_sensor_features(modality, rows, batch_size, bundle):
        if batch_size < 1:
            raise ValueError(f"{modality} batch size must be at least 1")
        out_dir = ARTIFACT_ROOT / "external" / modality
        feature_path = out_dir / f"{modality}_foundation.npy"
        status_path = out_dir / "encoding_status.npy"
        metadata_path = out_dir / "foundation_metadata.json"
        error_path = out_dir / "foundation_encoding_errors.jsonl"

        probe = None
        probe_errors = []
        for index, row in enumerate(rows):
            try:
                probe = encode_foundation(modality, [load_input(modality, row)], bundle)
                break
            except Exception as exc:
                probe_errors.append({"row_index": index, "sample_id": row["sample_id"],
                                     "stage": "probe", "error": repr(exc)})
        if probe is None:
            raise RuntimeError(f"No encodable {modality} record was found; examples={probe_errors[:10]}")

        shape = (len(rows), probe.shape[1])
        identity = {
            "modality": modality, "rows": len(rows),
            "manifest_sha256": manifest_hashes[modality],
            "encoder_model_id": config["hubert_model_id" if modality == "ecg" else "cxr_model_id"],
            "preprocessing_version": PREPROCESSING_VERSION, "output_dim": int(probe.shape[1]),
        }
        old_metadata = read_metadata(metadata_path)
        cache_matches = all(old_metadata.get(key) == value for key, value in identity.items())
        try:
            cache_matches = (cache_matches and feature_path.is_file() and status_path.is_file()
                             and np.load(feature_path, mmap_mode="r").shape == shape
                             and np.load(status_path, mmap_mode="r").shape == (len(rows),))
        except (OSError, ValueError):
            cache_matches = False
        if not cache_matches:
            features = np.lib.format.open_memmap(feature_path, mode="w+", dtype="float32", shape=shape)
            features[:] = np.nan; features.flush()
            status = np.lib.format.open_memmap(status_path, mode="w+", dtype="int8", shape=(len(rows),))
            status[:] = 0; status.flush()
            metadata_path.write_text(json.dumps({**identity, "complete": False}, indent=2))
        features = np.load(feature_path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")
        previous_errors = read_errors(error_path)

        eligible = np.arange(min(MAX_ROWS, len(rows)) if MAX_ROWS > 0 else len(rows))
        pending = eligible[status[eligible] != 1] if RETRY_FAILED else eligible[status[eligible] == 0]
        errors = []

        def record_error(index, stage, exc):
            errors.append({"row_index": int(index), "sample_id": rows[int(index)]["sample_id"],
                           "stage": stage, "error": repr(exc)})

        for start in tqdm(range(0, len(pending), batch_size), desc=f"encode {modality}"):
            indices = pending[start:start + batch_size]
            inputs, valid = [], []
            for index in indices:
                try:
                    inputs.append(load_input(modality, rows[index])); valid.append(int(index))
                except Exception as exc:
                    status[index] = -1
                    record_error(index, "load", exc)
            if inputs:
                try:
                    encoded = encode_foundation(modality, inputs, bundle)
                    features[valid] = encoded; status[valid] = 1
                except Exception:
                    # Isolate a bad record instead of marking the entire batch failed.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for index, value in zip(valid, inputs):
                        try:
                            features[index] = encode_foundation(modality, [value], bundle)[0]
                            status[index] = 1
                        except Exception as exc:
                            status[index] = -1
                            record_error(index, "encode", exc)
            features.flush(); status.flush()

        error_by_index = {
            int(item["row_index"]): item for item in previous_errors
            if 0 <= int(item.get("row_index", -1)) < len(status)
            and status[int(item["row_index"])] == -1
        }
        for item in errors:
            error_by_index[int(item["row_index"])] = item
        current_errors = [error_by_index[index] for index in sorted(error_by_index)
                          if status[index] == -1]
        error_path.write_text("".join(json.dumps(item) + "\\n" for item in current_errors))
        summary = {**identity, "selected_rows": len(eligible), "pending_processed": len(pending),
                   "valid": int((status == 1).sum()), "failed": int((status == -1).sum()),
                   "unprocessed": int((status == 0).sum()), "errors_this_run": len(errors),
                   "complete": bool(np.all(status == 1))}
        metadata_path.write_text(json.dumps(summary, indent=2))
        print(modality, summary, feature_path)
        return summary

    foundation_summaries = {}
    for modality, batch_env, default_batch in [
        ("ecg", "EXTERNAL_ECG_BATCH_SIZE", 8),
        ("cxr", "EXTERNAL_CXR_BATCH_SIZE", 16),
    ]:
        bundle = load_foundation_encoder(modality)
        try:
            foundation_summaries[modality] = resumable_sensor_features(
                modality, manifests[modality], int(os.environ.get(batch_env, str(default_batch))), bundle)
        finally:
            del bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    """),
    code("""
    experiment_path = ARTIFACT_ROOT / "ablation" / "experiment_index.json"
    if not experiment_path.is_file():
        raise FileNotFoundError(f"Run notebook 12 first; missing {experiment_path}")
    experiment_index = json.loads(experiment_path.read_text())
    if not isinstance(experiment_index, list) or not experiment_index:
        raise ValueError(f"No trained bridge experiments in {experiment_path}")
    selected = os.environ.get("EXTERNAL_BRIDGE_VARIANTS", "")
    if selected:
        requested = {value.strip() for value in selected.split(",") if value.strip()}
        available = {row["variant"] for row in experiment_index}
        unknown = requested - available
        if unknown:
            raise ValueError(f"Unknown EXTERNAL_BRIDGE_VARIANTS {sorted(unknown)}; available={sorted(available)}")
        experiment_index = [row for row in experiment_index if row["variant"] in requested]
    BRIDGE_BATCH_SIZE = int(os.environ.get("EXTERNAL_BRIDGE_BATCH_SIZE", "1024"))
    if BRIDGE_BATCH_SIZE < 1:
        raise ValueError("EXTERNAL_BRIDGE_BATCH_SIZE must be at least 1")

    def load_checkpoint(path):
        try:
            return torch.load(path, map_location=DEVICE, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=DEVICE)

    @torch.inference_mode()
    def apply_bridge(row, modality, batch_size=BRIDGE_BATCH_SIZE):
        checkpoint_path = Path(row["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing bridge checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        resolved = checkpoint["resolved_config"]
        model = ecg_cxr_ablation.AblationBridge(resolved["input_dims"], resolved["model_variant"],
                                                **resolved["model_hparams"]).to(DEVICE).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["model"])
        if row["variant"] == "shuffled_pairs_control": model.variant = row["variant"]
        features = np.load(ARTIFACT_ROOT / "external" / modality / f"{modality}_foundation.npy", mmap_mode="r")
        status = np.load(ARTIFACT_ROOT / "external" / modality / "encoding_status.npy", mmap_mode="r")
        if len(features) != len(manifests[modality]) or len(status) != len(manifests[modality]):
            raise ValueError(f"{modality} foundation arrays do not match the manifest; rerun this notebook")
        expected_dim = int(resolved["input_dims"][modality])
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(f"{modality} foundation dimension {features.shape} does not match bridge input {expected_dim}")
        out_dir = ARTIFACT_ROOT / "external" / modality / "bridge" / row["variant"] / f"seed_{row['seed']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        common = np.lib.format.open_memmap(out_dir / "common.npy", mode="w+", dtype="float32", shape=(len(features), resolved["model_hparams"]["common_dim"]))
        continuous = np.lib.format.open_memmap(out_dir / "continuous_common.npy", mode="w+", dtype="float32", shape=common.shape)
        common[:] = np.nan; continuous[:] = np.nan
        indices_out = None
        if model.uses_vq:
            n_chunks = resolved["model_hparams"]["common_dim"] // resolved["model_hparams"]["code_dim"]
            indices_out = np.lib.format.open_memmap(out_dir / "common_indices.npy", mode="w+", dtype="int64", shape=(len(features), n_chunks))
            indices_out[:] = -1
        valid = np.where(status == 1)[0]
        if len(valid) == 0:
            raise RuntimeError(f"No valid external {modality} foundation features")
        for start in tqdm(range(0, len(valid), batch_size), desc=f"{row['variant']} {row['seed']} {modality}"):
            index = valid[start:start + batch_size]
            output = model.encode_modality(modality, torch.from_numpy(np.array(features[index])).to(DEVICE))
            common_batch = output.common.cpu().numpy()
            continuous_batch = output.continuous_common.cpu().numpy()
            if not np.isfinite(common_batch).all() or not np.isfinite(continuous_batch).all():
                raise ValueError(f"Non-finite bridge output for {row['variant']} seed {row['seed']} {modality}")
            common[index] = common_batch; continuous[index] = continuous_batch
            if indices_out is not None:
                index_batch = output.common_indices.cpu().numpy()
                codebook_size = int(resolved["model_hparams"]["common_codebook_size"])
                if index_batch.min() < 0 or index_batch.max() >= codebook_size:
                    raise ValueError("Shared code index outside the configured codebook")
                indices_out[index] = index_batch
        common.flush(); continuous.flush()
        if indices_out is not None: indices_out.flush()
        metadata = {**row, "rows": len(features), "valid": len(valid),
                    "manifest_sha256": manifest_hashes[modality],
                    "foundation_model_id": config["hubert_model_id" if modality == "ecg" else "cxr_model_id"],
                    "bridge_batch_size": batch_size}
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        del model, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for experiment in experiment_index:
        for modality in ["ecg", "cxr"]:
            apply_bridge(experiment, modality)
    print("External shared-code encoding complete.")
    """),
])


write("14_encode_external_text_and_concepts.ipynb", [
    md("""
    # 14 - Encode external diagnoses, reports, and normalized concepts

    BiomedCLIP supplies a frozen biomedical text representation. Two text views are written:

    - `raw`: the complete machine diagnosis or CXR report.
    - `concept`: a short assertion-aware concept list using `present:`, `absent:`, and `uncertain:` prefixes.

    The default concept extractor is deterministic and auditable. To ablate LLM reasoning, provide `LLM_CONCEPTS_JSONL`, one JSON object per line with `sample_id` and a list of assertion-aware `concepts`. LLM outputs are treated as inferred semantic supervision, never as encounter ground truth.

    Text embeddings are written incrementally. `EXTERNAL_TEXT_MAX_ROWS` (or `EXTERNAL_MAX_ROWS`) limits a smoke run without changing array-to-manifest alignment; use `0` later to resume the complete manifests. Set `EXTERNAL_TEXT_RETRY_FAILED=1` to retry failed text rows.
    """),
    code("""
    from pathlib import Path
    import gc
    import hashlib
    import json
    import os
    import re

    import numpy as np
    import torch
    from tqdm.auto import tqdm

    ARTIFACT_ROOT = Path(os.environ.get("ECG_CXR_BRIDGE_ARTIFACT_ROOT", "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    config = json.loads((ARTIFACT_ROOT / "config.json").read_text())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEXT_MAX_ROWS = int(os.environ.get("EXTERNAL_TEXT_MAX_ROWS", os.environ.get("EXTERNAL_MAX_ROWS", "0")))
    TEXT_BATCH_SIZE = int(os.environ.get("EXTERNAL_TEXT_BATCH_SIZE", "256"))
    RETRY_FAILED = os.environ.get("EXTERNAL_TEXT_RETRY_FAILED", "0").strip().lower() in {"1", "true", "yes"}
    if TEXT_MAX_ROWS < 0 or TEXT_BATCH_SIZE < 1:
        raise ValueError("EXTERNAL_TEXT_MAX_ROWS must be nonnegative and EXTERNAL_TEXT_BATCH_SIZE must be positive")
    for key in ["text_model_id", "hf_cache"]:
        if key not in config:
            raise KeyError(f"Run notebook 09 first; config is missing {key!r}")

    def manifest(modality):
        path = ARTIFACT_ROOT / "external" / modality / "manifest.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Run notebook {'10' if modality == 'ecg' else '11'} first; missing {path}")
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        text_key = "machine_diagnosis" if modality == "ecg" else "report_text"
        required = ["sample_id", "split", text_key]
        invalid = [(index, [key for key in required if key not in row or row[key] in (None, "")])
                   for index, row in enumerate(values)]
        invalid = [(index, missing) for index, missing in invalid if missing]
        if not values:
            raise ValueError(f"Empty {modality} manifest: {path}")
        if invalid:
            raise KeyError(f"{modality} manifest rows missing required fields: {invalid[:20]}")
        seen, duplicates = set(), set()
        for row in values:
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                duplicates.add(sample_id)
            seen.add(sample_id)
        if duplicates:
            raise ValueError(f"Duplicate {modality} sample IDs: {sorted(duplicates)[:20]}")
        return values

    def sequence_sha256(values):
        digest = hashlib.sha256()
        for value in values:
            encoded = str(value).encode()
            digest.update(len(encoded).to_bytes(8, "big")); digest.update(encoded)
        return digest.hexdigest()

    rows = {name: manifest(name) for name in ["ecg", "cxr"]}
    manifest_hashes = {name: sequence_sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for row in values)
        for name, values in rows.items()}
    print({name: {"rows": len(value), "manifest_sha256": manifest_hashes[name][:12]}
           for name, value in rows.items()}, "device", DEVICE,
          "text_max_rows", TEXT_MAX_ROWS, "retry_failed", RETRY_FAILED)
    """),
    code("""
    ECG_CONCEPT_RULE_VERSION = "ecg_assertion_rules_v3"
    CXR_CONCEPT_RULE_VERSION = "cxr_assertion_rules_v2"
    ECG_RULES = {
        # Sinus rhythm alone does not imply that the complete ECG is normal.
        "normal_ecg": [
            r"\\bnormal(?:\\s+12[- ]lead)?\\s+(?:ecg|electrocardiogram)\\b",
            r"\\b(?:ecg|electrocardiogram)\\s+(?:is\\s+)?within normal limits\\b",
        ],
        "atrial_fibrillation": [r"\\batrial fibrillation\\b", r"\\ba[- ]?fib(?:rillation)?\\b"],
        "atrial_flutter": [r"\\batrial flutter\\b"],
        "tachycardia": [r"\\btachycard(?:ia|ic)\\b", r"\\b(?:svt|v[- ]?tach)\\b"],
        "bradycardia": [r"\\bbradycard(?:ia|ic)\\b"],
        "conduction_abnormality": [
            r"\\bbundle branch block\\b", r"\\b(?:rbbb|lbbb|ivcd)\\b",
            r"\\b(?:av|atrioventricular|heart) block\\b",
            r"\\b(?:first|second|third)[ -]degree (?:av|atrioventricular) block\\b",
            r"\\bintraventricular conduction (?:delay|defect)\\b",
        ],
        "ventricular_hypertrophy": [
            r"\\b(?:left|right) ventricular hypertrophy\\b",
            r"\\bbiventricular hypertrophy\\b", r"\\bventricular hypertrophy\\b",
            r"\\b(?:lvh|rvh)\\b",
        ],
        "ischemia": [
            r"\\bischemi\\w*\\b", r"\\bst[- ]t abnormalit\\w*\\b",
            r"\\bst(?:[- ]segment)?\\s+(?:depression|elevation)\\b",
        ],
        "infarction": [
            r"\\bmyocardial infarction\\b", r"\\binfarct(?:ion|ed)?\\b",
            r"\\b(?:stemi|nstemi)\\b",
        ],
        "paced_rhythm": [
            r"\\b(?:atrial|ventricular|dual[- ]chamber)?\\s*paced rhythm\\b",
            r"\\bpacing spikes?\\b", r"\\bpacemaker rhythm\\b",
        ],
        "heart_failure": [r"\\bheart failure\\b", r"\\bcongestive heart failure\\b"],
        "low_voltage": [r"\\blow[- ]voltage\\b"],
        "qt_abnormality": [
            r"\\b(?:prolonged|long|short)\\s+qt(?:c)?(?: interval)?\\b",
            r"\\bqt(?:c)? prolongation\\b",
        ],
    }
    CXR_RULES = {
        "no_finding": [r"no acute cardiopulmonary", r"no finding", r"normal chest"],
        "cardiomegaly": [r"cardiomegal", r"enlarged cardiomediast"], "edema": [r"\\bedema\\b", r"vascular congestion"],
        "pleural_effusion": [r"pleural effusion"], "pneumonia": [r"pneumonia"], "consolidation": [r"consolidat"],
        "atelectasis": [r"atelecta"], "pneumothorax": [r"pneumothorax"], "lung_opacity": [r"lung opacity", r"airspace opacity"],
        "support_device": [r"support device", r"pacemaker", r"central line", r"endotracheal tube"],
    }

    CHEXPERT_LABEL_CONCEPTS = {
        "atelectasis": "atelectasis", "cardiomegaly": "cardiomegaly",
        "consolidation": "consolidation", "edema": "edema",
        "enlarged cardiomediastinum": "enlarged_cardiomediastinum",
        "fracture": "fracture", "lung lesion": "lung_lesion",
        "lung opacity": "lung_opacity", "no finding": "no_finding",
        "pleural effusion": "pleural_effusion", "pleural other": "pleural_other",
        "pneumonia": "pneumonia", "pneumothorax": "pneumothorax",
        "support devices": "support_device", "support device": "support_device",
    }
    NEGATED_BEFORE = re.compile(
        r"(?:\\bno\\b|\\bwithout\\b|\\bnegative for\\b|\\babsence of\\b|\\bfree of\\b|\\bdenies\\b)"
        r"(?:\\W+\\w+){0,8}\\W*$", re.I)
    NEGATED_AFTER = re.compile(r"^\\W*(?:is |are |was |were )?(?:not|absent)(?:\\W|$)", re.I)
    UNCERTAIN_BEFORE = re.compile(
        r"(?:\\bpossible\\b|\\bpossibly\\b|\\bprobable\\b|\\blikely\\b|\\bmay\\b|\\bmight\\b|"
        r"\\bcannot exclude\\b|\\bcannot rule out\\b|\\bconcerning for\\b|\\bsuspicious for\\b|\\bquestion of\\b)"
        r"(?:\\W+\\w+){0,8}\\W*$", re.I)
    UNCERTAIN_AFTER = re.compile(r"^\\W*(?:is |are )?(?:possible|probable|suspected)(?:\\W|$)", re.I)

    def assertion(text, match, concept):
        if concept == "no_finding":
            return "present"
        before = text[max(0, match.start() - 120):match.start()]
        before = re.split(r"[.;\\n]|\\b(?:but|however|although|yet)\\b", before, flags=re.I)[-1]
        after = text[match.end():match.end() + 50]
        if NEGATED_BEFORE.search(before) or NEGATED_AFTER.search(after):
            return "absent"
        if UNCERTAIN_BEFORE.search(before) or UNCERTAIN_AFTER.search(after):
            return "uncertain"
        return "present"

    def rule_concepts(text, rules, labels=()):
        value = " ".join(str(text).split()).lower()
        found = set()
        for concept, patterns in rules.items():
            for pattern in patterns:
                for match in re.finditer(pattern, value):
                    found.add(f"{assertion(value, match, concept)}:{concept}")
        if isinstance(labels, dict):
            labels = [key for key, present in labels.items()
                      if present not in (None, False, 0, 0.0, "", "0", "false", "False")]
        elif isinstance(labels, str):
            labels = [labels]
        for label in labels or ():
            concept = CHEXPERT_LABEL_CONCEPTS.get(str(label).strip().lower())
            if concept:
                found.add(f"present:{concept}")
        return sorted(found)

    ECG_RULE_TESTS = [
        ("Normal 12-lead ECG.", {"present:normal_ecg"}),
        ("Normal sinus rhythm. Inferior infarct, age undetermined.", {"present:infarction"}),
        ("No evidence of atrial fibrillation.", {"absent:atrial_fibrillation"}),
        ("Possible left ventricular hypertrophy.", {"uncertain:ventricular_hypertrophy"}),
        ("No acute ischemic changes.", {"absent:ischemia"}),
        ("Ventricular paced rhythm.", {"present:paced_rhythm"}),
    ]
    for test_text, expected in ECG_RULE_TESTS:
        observed = set(rule_concepts(test_text, ECG_RULES))
        if observed != expected:
            raise AssertionError(
                f"ECG concept-rule self-test failed for {test_text!r}: "
                f"expected={sorted(expected)}, observed={sorted(observed)}")
    print("ECG concept extractor:", ECG_CONCEPT_RULE_VERSION,
          "self-tests:", len(ECG_RULE_TESTS))

    def validate_llm_concepts(value, sample_id):
        if not isinstance(value, list):
            raise TypeError(f"LLM concepts for {sample_id} must be a list")
        concepts = []
        for item in value:
            concept = str(item).strip().lower().replace(" ", "_")
            if not re.fullmatch(r"(?:present|absent|uncertain):[a-z0-9][a-z0-9_/-]*", concept):
                raise ValueError(f"Invalid assertion-aware concept for {sample_id}: {item!r}")
            concepts.append(concept)
        return sorted(set(concepts))

    llm_path = os.environ.get("LLM_CONCEPTS_JSONL", "")
    llm = {}
    if llm_path:
        known_ids = {str(row["sample_id"]) for modality_rows in rows.values() for row in modality_rows}
        for line_number, line in enumerate(Path(llm_path).read_text().splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = str(item["sample_id"])
            if sample_id not in known_ids:
                raise KeyError(f"Unknown LLM sample_id at line {line_number}: {sample_id}")
            if sample_id in llm:
                raise ValueError(f"Duplicate LLM sample_id: {sample_id}")
            llm[sample_id] = validate_llm_concepts(item["concepts"], sample_id)

    text_root = ARTIFACT_ROOT / "external" / "text"
    text_root.mkdir(parents=True, exist_ok=True)
    prompt_path = text_root / "llm_concept_requests.jsonl"
    with prompt_path.open("w") as stream:
        for modality in ["ecg", "cxr"]:
            key = "machine_diagnosis" if modality == "ecg" else "report_text"
            for row in rows[modality]:
                prompt = ("Extract only explicitly stated normalized clinical concepts. Return JSON with sample_id and "
                          "concepts, where each concept is exactly present:<concept>, absent:<concept>, or "
                          "uncertain:<concept>. Preserve negation and uncertainty; do not infer identity, causality, "
                          "or unstated diagnoses. Text: " + row[key])
                stream.write(json.dumps({"sample_id": row["sample_id"], "modality": modality, "prompt": prompt}) + "\\n")

    text_views, concept_records = {}, {}
    for modality in ["ecg", "cxr"]:
        key = "machine_diagnosis" if modality == "ecg" else "report_text"
        concept_rule_version = (ECG_CONCEPT_RULE_VERSION if modality == "ecg"
                                else CXR_CONCEPT_RULE_VERSION)
        raw = [" ".join(str(row[key]).split()) for row in rows[modality]]
        rules = ECG_RULES if modality == "ecg" else CXR_RULES
        concepts = []
        for row in rows[modality]:
            values = llm.get(str(row["sample_id"]))
            if values is None:
                values = rule_concepts(row[key], rules, row.get("labels", []))
            concepts.append(values)
        concept_text = ["; ".join(values) if values else "uncertain:unspecified_clinical_finding" for values in concepts]
        text_views[modality] = {"raw": raw, "concept": concept_text}
        concept_records[modality] = concepts
        out = ARTIFACT_ROOT / "external" / modality
        with (out / "concepts.jsonl").open("w") as stream:
            for row, values, text in zip(rows[modality], concepts, concept_text):
                stream.write(json.dumps({"sample_id": row["sample_id"], "concepts": values, "concept_text": text,
                                         "source": "llm" if str(row["sample_id"]) in llm else "rules",
                                         "concept_rule_version": concept_rule_version}) + "\\n")
        print(modality, {"rows": len(raw), "with_concepts": sum(bool(v) for v in concepts),
                         "assertions": {status: sum(any(value.startswith(status + ":") for value in item)
                                                   for item in concepts)
                                        for status in ["present", "absent", "uncertain"]}})
    print("LLM request file:", prompt_path, "loaded LLM responses:", len(llm))
    """),
    code("""
    from open_clip import create_model_from_pretrained, get_tokenizer

    model_result = create_model_from_pretrained(config["text_model_id"], cache_dir=config["hf_cache"])
    # open_clip releases differ here: some return (model, preprocess), while
    # others return (model, preprocess_train, preprocess_val).
    model = model_result[0] if isinstance(model_result, (tuple, list)) else model_result
    if model is None or not hasattr(model, "encode_text"):
        raise TypeError(f"Unexpected create_model_from_pretrained result: {type(model_result).__name__}")
    tokenizer = get_tokenizer(config["text_model_id"])
    model = model.to(DEVICE).eval().requires_grad_(False)
    TEXT_ENCODING_VERSION = "biomedclip_text_v3_assertion_aware"

    @torch.inference_mode()
    def encode_text_batch(texts):
        tokens = tokenizer(texts)
        if hasattr(tokens, "to"):
            tokens = tokens.to(DEVICE)
        encoded = torch.nn.functional.normalize(model.encode_text(tokens), dim=-1).float().cpu().numpy()
        if encoded.ndim != 2 or len(encoded) != len(texts) or not np.isfinite(encoded).all():
            raise ValueError(f"Invalid text encoder output: {encoded.shape}")
        return encoded

    def read_json(path, default):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def read_jsonl(path):
        try:
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def resumable_text_embeddings(modality, view, texts):
        out = ARTIFACT_ROOT / "external" / modality
        feature_path = out / f"text_{view}_biomedclip.npy"
        status_path = out / f"text_{view}_encoding_status.npy"
        metadata_path = out / f"text_{view}_metadata.json"
        error_path = out / f"text_{view}_encoding_errors.jsonl"
        probe = encode_text_batch([texts[0]])
        shape = (len(texts), probe.shape[1])
        identity = {
            "modality": modality, "view": view, "rows": len(texts),
            "manifest_sha256": manifest_hashes[modality],
            "text_sha256": sequence_sha256(texts), "text_model_id": config["text_model_id"],
            "encoding_version": TEXT_ENCODING_VERSION, "output_dim": int(probe.shape[1]),
        }
        old = read_json(metadata_path, {})
        cache_matches = isinstance(old, dict) and all(old.get(key) == value for key, value in identity.items())
        try:
            cache_matches = (cache_matches and feature_path.is_file() and status_path.is_file()
                             and np.load(feature_path, mmap_mode="r").shape == shape
                             and np.load(status_path, mmap_mode="r").shape == (len(texts),))
        except (OSError, ValueError):
            cache_matches = False
        if not cache_matches:
            features = np.lib.format.open_memmap(feature_path, mode="w+", dtype="float32", shape=shape)
            features[:] = np.nan; features.flush()
            status = np.lib.format.open_memmap(status_path, mode="w+", dtype="int8", shape=(len(texts),))
            status[:] = 0; status.flush()
            metadata_path.write_text(json.dumps({**identity, "complete": False}, indent=2))
        features = np.load(feature_path, mmap_mode="r+")
        status = np.load(status_path, mmap_mode="r+")
        eligible = np.arange(min(TEXT_MAX_ROWS, len(texts)) if TEXT_MAX_ROWS > 0 else len(texts))
        pending = eligible[status[eligible] != 1] if RETRY_FAILED else eligible[status[eligible] == 0]
        previous_errors = read_jsonl(error_path)
        errors = []
        for start in tqdm(range(0, len(pending), TEXT_BATCH_SIZE), desc=f"text {modality} {view}"):
            index = pending[start:start + TEXT_BATCH_SIZE]
            batch = [texts[int(value)] for value in index]
            try:
                features[index] = encode_text_batch(batch); status[index] = 1
            except Exception:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for row_index, text in zip(index, batch):
                    try:
                        features[row_index] = encode_text_batch([text])[0]; status[row_index] = 1
                    except Exception as exc:
                        status[row_index] = -1
                        errors.append({"row_index": int(row_index), "sample_id": rows[modality][int(row_index)]["sample_id"],
                                       "error": repr(exc)})
            features.flush(); status.flush()
        error_by_index = {
            int(item["row_index"]): item for item in previous_errors
            if 0 <= int(item.get("row_index", -1)) < len(status)
            and status[int(item["row_index"])] == -1
        }
        for item in errors:
            error_by_index[int(item["row_index"])] = item
        current_errors = [error_by_index[index] for index in sorted(error_by_index)
                          if status[index] == -1]
        error_path.write_text("".join(json.dumps(item) + "\\n" for item in current_errors))
        summary = {**identity, "selected_rows": len(eligible), "pending_processed": len(pending),
                   "valid": int((status == 1).sum()), "failed": int((status == -1).sum()),
                   "unprocessed": int((status == 0).sum()), "errors_this_run": len(errors),
                   "complete": bool(np.all(status == 1))}
        metadata_path.write_text(json.dumps(summary, indent=2))
        print(summary)

    for modality in ["ecg", "cxr"]:
        for view in ["raw", "concept"]:
            resumable_text_embeddings(modality, view, text_views[modality][view])
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    """),
])


write("15_train_external_semantic_adapters.ipynb", [
    md("""
    # 15 - Train external semantic adapters with a frozen shared codebook

    The Symile bridge and shared codebook remain frozen. A small residual adapter is learned for each external sensor, while either one shared or two separate text projectors map diagnoses/reports into the same space. Training uses external train pairs; validation selects checkpoints.

    Raw shared-text adapters are trained for every bridge variant. The two deeper semantic ablations (`separate_raw_text` and `shared_concepts`) are run for `shared_specific_vq`. Sensor and text status masks are intersected, so partial or failed encodings never enter training. Repeated identical texts/concept lists are treated as multiple positives rather than false negatives.

    Finished runs are validated and skipped automatically. A completion marker records the exact bridge, data signatures, semantic configuration, and training hyperparameters; interrupted or stale runs are retrained. Set `SEMANTIC_FORCE_RETRAIN=1` to deliberately rerun completed configurations. A small smoke run can be selected with `SEMANTIC_BRIDGE_VARIANTS`, `SEMANTIC_VARIANTS`, `SEMANTIC_ADAPTER_SEEDS`, and `SEMANTIC_ADAPTER_EPOCHS`.
    """),
    code("""
    from pathlib import Path
    import hashlib
    import importlib
    import json
    import os
    import random
    import sys

    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset

    ARTIFACT_ROOT = Path(os.environ.get("ECG_CXR_BRIDGE_ARTIFACT_ROOT", "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    CODE_ROOT = ARTIFACT_ROOT / "code"
    module_path = CODE_ROOT / "ecg_cxr_ablation.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Run notebook 12 first; missing {module_path}")
    if str(CODE_ROOT) not in sys.path: sys.path.insert(0, str(CODE_ROOT))
    import ecg_cxr_ablation
    importlib.reload(ecg_cxr_ablation)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_rows(modality):
        path = ARTIFACT_ROOT / "external" / modality / "manifest.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Run notebook {'10' if modality == 'ecg' else '11'} first; missing {path}")
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not values:
            raise ValueError(f"Empty {modality} manifest: {path}")
        required = ["sample_id", "split", "machine_diagnosis" if modality == "ecg" else "report_text"]
        invalid = [(index, [key for key in required if key not in row or row[key] in (None, "")])
                   for index, row in enumerate(values)]
        invalid = [(index, missing) for index, missing in invalid if missing]
        if invalid:
            raise KeyError(f"{modality} manifest rows missing required fields: {invalid[:20]}")
        return values

    def sensor_manifest_sha256(values):
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def text_manifest_sha256(values):
        digest = hashlib.sha256()
        for row in values:
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            digest.update(len(encoded).to_bytes(8, "big")); digest.update(encoded)
        return digest.hexdigest()

    def load_concept_texts(modality):
        path = ARTIFACT_ROOT / "external" / modality / "concepts.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Run notebook 14 first; missing {path}")
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(values) != len(rows[modality]):
            raise ValueError(f"{modality} concept rows ({len(values)}) do not match manifest ({len(rows[modality])})")
        for index, (row, concept) in enumerate(zip(rows[modality], values)):
            if str(row["sample_id"]) != str(concept.get("sample_id")):
                raise ValueError(f"{modality} concept order mismatch at row {index}")
        return [str(value.get("concept_text", "")).strip() for value in values]

    rows = {m: load_rows(m) for m in ["ecg", "cxr"]}
    manifest_hashes = {
        m: {"sensor": sensor_manifest_sha256(rows[m]), "text": text_manifest_sha256(rows[m])}
        for m in ["ecg", "cxr"]}
    concept_texts = {m: load_concept_texts(m) for m in ["ecg", "cxr"]}
    experiment_path = ARTIFACT_ROOT / "ablation" / "experiment_index.json"
    if not experiment_path.is_file():
        raise FileNotFoundError(f"Run notebook 12 first; missing {experiment_path}")
    experiment_index = json.loads(experiment_path.read_text())
    if not isinstance(experiment_index, list) or not experiment_index:
        raise ValueError(f"No bridge experiments in {experiment_path}")

    selected_bridge_variants = {value.strip() for value in os.environ.get(
        "SEMANTIC_BRIDGE_VARIANTS", "").split(",") if value.strip()}
    if selected_bridge_variants:
        available = {item["variant"] for item in experiment_index}
        unknown = selected_bridge_variants - available
        if unknown:
            raise ValueError(f"Unknown SEMANTIC_BRIDGE_VARIANTS {sorted(unknown)}; available={sorted(available)}")
        experiment_index = [item for item in experiment_index if item["variant"] in selected_bridge_variants]
    selected_bridge_seeds = {int(value.strip()) for value in os.environ.get(
        "SEMANTIC_BRIDGE_SEEDS", "").split(",") if value.strip()}
    if selected_bridge_seeds:
        available_seeds = {int(item["seed"]) for item in experiment_index}
        unknown_seeds = selected_bridge_seeds - available_seeds
        if unknown_seeds:
            raise ValueError(f"Unknown SEMANTIC_BRIDGE_SEEDS {sorted(unknown_seeds)}; available={sorted(available_seeds)}")
        experiment_index = [item for item in experiment_index if int(item["seed"]) in selected_bridge_seeds]
    if not experiment_index:
        raise ValueError("No bridge experiments selected")

    # Verify the source manifests are patient-disjoint across training and validation.
    for modality in ["ecg", "cxr"]:
        subject_sets = {split: {int(row["subject_id"]) for row in rows[modality]
                                if row["split"] == split and row.get("subject_id") is not None}
                        for split in ["train", "val"]}
        overlap = subject_sets["train"] & subject_sets["val"]
        if overlap:
            raise AssertionError(f"Patient leakage in external {modality} train/val: {sorted(overlap)[:20]}")
    print({"device": str(DEVICE), "bridge_runs": len(experiment_index),
           "manifest_rows": {m: len(rows[m]) for m in rows}})
    """),
    code("""
    def load_json(path):
        try:
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    class ExternalPairs(Dataset):
        def __init__(self, modality, experiment, split, text_view):
            self.modality, self.rows = modality, rows[modality]
            base = ARTIFACT_ROOT / "external" / modality
            bridge = base / "bridge" / experiment["variant"] / f"seed_{experiment['seed']}"
            paths = {
                "sensor": bridge / "continuous_common.npy",
                "text": base / f"text_{text_view}_biomedclip.npy",
                "sensor_status": base / "encoding_status.npy",
                "text_status": base / f"text_{text_view}_encoding_status.npy",
                "bridge_metadata": bridge / "metadata.json",
                "text_metadata": base / f"text_{text_view}_metadata.json",
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError("Run notebooks 13-14 first; missing:\\n" + "\\n".join(missing))
            bridge_metadata = load_json(paths["bridge_metadata"])
            text_metadata = load_json(paths["text_metadata"])
            if bridge_metadata.get("manifest_sha256") != manifest_hashes[modality]["sensor"]:
                raise ValueError(f"Stale {modality} bridge features for {experiment['variant']} seed {experiment['seed']}; rerun notebook 13")
            if text_metadata.get("manifest_sha256") != manifest_hashes[modality]["text"]:
                raise ValueError(f"Stale {modality} {text_view} text features; rerun notebook 14")
            self.sensor = np.load(paths["sensor"], mmap_mode="r")
            self.text = np.load(paths["text"], mmap_mode="r")
            sensor_status = np.load(paths["sensor_status"], mmap_mode="r")
            text_status = np.load(paths["text_status"], mmap_mode="r")
            expected_rows = len(self.rows)
            lengths = {"sensor": len(self.sensor), "text": len(self.text),
                       "sensor_status": len(sensor_status), "text_status": len(text_status)}
            if any(value != expected_rows for value in lengths.values()):
                raise ValueError(f"{modality} array/manifest row mismatch: manifest={expected_rows}, arrays={lengths}")
            if self.sensor.ndim != 2 or self.text.ndim != 2:
                raise ValueError(f"Expected 2-D {modality} arrays, got sensor={self.sensor.shape}, text={self.text.shape}")
            self.index = np.asarray([
                i for i, row in enumerate(self.rows)
                if row["split"] == split and sensor_status[i] == 1 and text_status[i] == 1
            ], dtype="int64")
            raw_key = "machine_diagnosis" if modality == "ecg" else "report_text"
            group_text = concept_texts[modality] if text_view == "concept" else [
                " ".join(str(row[raw_key]).lower().split()) for row in self.rows]
            self.group = np.asarray([
                int(hashlib.sha1(group_text[i].encode()).hexdigest()[:15], 16) for i in self.index
            ], dtype="int64")
            self.subject = np.asarray([
                int(self.rows[i]["subject_id"]) if self.rows[i].get("subject_id") is not None else -(int(i) + 1)
                for i in self.index], dtype="int64")
            for start in range(0, len(self.index), 4096):
                take = self.index[start:start + 4096]
                if not np.isfinite(np.asarray(self.sensor[take])).all() or not np.isfinite(np.asarray(self.text[take])).all():
                    raise ValueError(f"Non-finite {modality} {split}/{text_view} features at selected rows")
        def __len__(self): return len(self.index)
        def __getitem__(self, position):
            index = self.index[position]
            return {"sensor": torch.tensor(np.array(self.sensor[index], dtype="float32")),
                    "text": torch.tensor(np.array(self.text[index], dtype="float32")),
                    "group": torch.tensor(self.group[position]), "subject": torch.tensor(self.subject[position]),
                    "row_index": torch.tensor(index)}

    class Adapter(nn.Module):
        def __init__(self, bridge, text_dim, shared_text=True):
            super().__init__(); self.bridge = bridge; self.shared_text = shared_text
            for parameter in self.bridge.parameters(): parameter.requires_grad = False
            dim = bridge.common_dim
            self.residual = nn.ModuleDict({m: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim)) for m in ["ecg", "cxr"]})
            for module in self.residual.values(): nn.init.zeros_(module[-1].weight); nn.init.zeros_(module[-1].bias)
            names = ["shared"] if shared_text else ["ecg", "cxr"]
            self.text_projector = nn.ModuleDict({name: nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, dim)) for name in names})
        def quantize(self, values):
            if not self.bridge.uses_vq: return F.normalize(values, dim=-1)
            quantized, _, _, _ = self.bridge.common_codebook(values)
            return F.normalize(quantized, dim=-1)
        def sensor(self, modality, values): return self.quantize(values + self.residual[modality](values))
        def text(self, modality, values):
            key = "shared" if self.shared_text else modality
            return self.quantize(self.text_projector[key](values))

        def train(self, mode=True):
            super().train(mode)
            self.bridge.eval()
            return self

    def multi_positive_direction(logits, positives, ignored):
        logits = logits.masked_fill(ignored & ~positives, -torch.inf)
        positive_logits = logits.masked_fill(~positives, -torch.inf)
        if not positives.any(1).all():
            raise RuntimeError("Every query must have at least one positive")
        return -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()

    def contrastive(left, right, group, subject, temperature=.07):
        logits = left @ right.T / temperature
        same_text = group[:, None].eq(group[None, :])
        same_subject = subject[:, None].eq(subject[None, :]) & subject[:, None].ge(0)
        return .5 * (multi_positive_direction(logits, same_text, same_subject)
                     + multi_positive_direction(logits.T, same_text.T, same_subject.T))
    """),
    code("""
    semantic_variants = []
    for experiment in experiment_index:
        semantic_variants.append((experiment, "shared_raw_text", "raw", True))
        if experiment["variant"] == "shared_specific_vq":
            semantic_variants.extend([(experiment, "separate_raw_text", "raw", False),
                                      (experiment, "shared_concepts", "concept", True)])
    selected_semantic = {value.strip() for value in os.environ.get(
        "SEMANTIC_VARIANTS", "").split(",") if value.strip()}
    available_semantic = {value[1] for value in semantic_variants}
    if selected_semantic:
        unknown = selected_semantic - available_semantic
        if unknown:
            raise ValueError(f"Unknown SEMANTIC_VARIANTS {sorted(unknown)}; available={sorted(available_semantic)}")
        semantic_variants = [value for value in semantic_variants if value[1] in selected_semantic]
    if not semantic_variants:
        raise ValueError("No semantic adapter experiments selected")

    seeds = [int(v.strip()) for v in os.environ.get("SEMANTIC_ADAPTER_SEEDS", "43,44,45").split(",") if v.strip()]
    EPOCHS = int(os.environ.get("SEMANTIC_ADAPTER_EPOCHS", "30"))
    BATCH = int(os.environ.get("SEMANTIC_ADAPTER_BATCH_SIZE", "256"))
    PATIENCE = int(os.environ.get("SEMANTIC_ADAPTER_PATIENCE", "5"))
    NUM_WORKERS = int(os.environ.get("SEMANTIC_ADAPTER_NUM_WORKERS", "0"))
    LEARNING_RATE = float(os.environ.get("SEMANTIC_ADAPTER_LR", "3e-4"))
    WEIGHT_DECAY = float(os.environ.get("SEMANTIC_ADAPTER_WEIGHT_DECAY", "1e-4"))
    GRAD_CLIP = float(os.environ.get("SEMANTIC_ADAPTER_GRAD_CLIP", "1.0"))
    TEMPERATURE = float(os.environ.get("SEMANTIC_ADAPTER_TEMPERATURE", "0.07"))
    FORCE_RETRAIN = os.environ.get("SEMANTIC_FORCE_RETRAIN", "0").strip().lower() in {"1", "true", "yes"}
    if not seeds or EPOCHS < 1 or BATCH < 2 or PATIENCE < 1 or NUM_WORKERS < 0:
        raise ValueError("Seeds must be nonempty; epochs/patience >= 1; batch >= 2; workers >= 0")
    if LEARNING_RATE <= 0 or WEIGHT_DECAY < 0 or GRAD_CLIP <= 0 or TEMPERATURE <= 0:
        raise ValueError("Invalid optimizer, gradient clipping, or temperature setting")

    def load_checkpoint(path, map_location=DEVICE):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    training_config = {
        "epochs": EPOCHS, "batch_size": BATCH, "patience": PATIENCE,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRAD_CLIP, "temperature": TEMPERATURE,
    }

    def run_key(item):
        return (str(item["bridge_variant"]), int(item["bridge_seed"]),
                str(item["semantic_variant"]), int(item["adapter_seed"]))

    def run_signature(experiment, semantic_name, text_view, shared_text, seed):
        return {
            "bridge_variant": str(experiment["variant"]),
            "bridge_seed": int(experiment["seed"]),
            "bridge_checkpoint": str(Path(experiment["checkpoint"]).resolve()),
            "semantic_variant": str(semantic_name), "text_view": str(text_view),
            "shared_text": bool(shared_text), "adapter_seed": int(seed),
            "training_config": training_config,
            "manifest_hashes": manifest_hashes,
        }

    def completed_run(run_dir, signature):
        if FORCE_RETRAIN:
            return None
        best_path = run_dir / "best.pt"
        history_path = run_dir / "history.json"
        completion_path = run_dir / "completed.json"
        if not best_path.is_file() or not history_path.is_file():
            return None
        marker = load_json(completion_path)
        if marker:
            if marker.get("status") != "complete" or marker.get("signature") != signature:
                return None
        try:
            history = json.loads(history_path.read_text())
            if not isinstance(history, list) or not history:
                return None
            validation = [float(row["val"]) for row in history]
            if not np.isfinite(validation).all():
                return None
            state = load_checkpoint(best_path, map_location="cpu")
            experiment = state.get("experiment", {})
            state_config = state.get("training_config", {})
            matches = (
                str(experiment.get("variant")) == signature["bridge_variant"]
                and int(experiment.get("seed", -1)) == signature["bridge_seed"]
                and str(Path(state.get("bridge_checkpoint", "")).resolve()) == signature["bridge_checkpoint"]
                and state.get("semantic_variant") == signature["semantic_variant"]
                and state.get("text_view") == signature["text_view"]
                and bool(state.get("shared_text")) == signature["shared_text"]
                and int(state.get("seed", -1)) == signature["adapter_seed"]
                and all(state_config.get(key) == value for key, value in training_config.items())
                and isinstance(state.get("model"), dict) and bool(state["model"])
            )
            if not matches:
                return None
            best_validation = float(min(validation))
            entry = {
                "bridge_variant": signature["bridge_variant"],
                "bridge_seed": signature["bridge_seed"],
                "semantic_variant": signature["semantic_variant"],
                "adapter_seed": signature["adapter_seed"],
                "best_validation_loss": best_validation,
                "checkpoint": str(best_path), "text_view": signature["text_view"],
                "data_counts": state.get("data_counts", {}),
            }
            # Legacy notebook-15 runs wrote history only after the loop completed.
            # Once validated, give them the same explicit completion marker.
            if not marker:
                completion_path.write_text(json.dumps({
                    "status": "complete", "signature": signature,
                    "epochs_completed": len(history),
                    "best_validation_loss": best_validation,
                    "recovered_legacy_run": True,
                }, indent=2))
            return entry
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print("Existing adapter is incomplete or invalid; retraining:", run_dir, repr(exc))
            return None

    index_path = ARTIFACT_ROOT / "external" / "adapters" / "adapter_index.json"
    try:
        existing_adapter_index = json.loads(index_path.read_text())
        if not isinstance(existing_adapter_index, list):
            existing_adapter_index = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing_adapter_index = []
    adapter_index = []
    selected_run_keys = set()
    for experiment, semantic_name, text_view, shared_text in semantic_variants:
        checkpoint_path = Path(experiment["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing bridge checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path)
        resolved = checkpoint["resolved_config"]
        for seed in seeds:
            run_dir = (ARTIFACT_ROOT / "external" / "adapters" / experiment["variant"]
                       / f"bridge_seed_{experiment['seed']}" / semantic_name / f"seed_{seed}")
            run_dir.mkdir(parents=True, exist_ok=True)
            signature = run_signature(experiment, semantic_name, text_view, shared_text, seed)
            selected_run_keys.add(run_key(signature))
            finished_entry = completed_run(run_dir, signature)
            if finished_entry is not None:
                adapter_index.append(finished_entry)
                print("SKIP completed adapter:", experiment["variant"], experiment["seed"],
                      semantic_name, seed, "best_val=", finished_entry["best_validation_loss"])
                continue

            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            bridge = ecg_cxr_ablation.AblationBridge(resolved["input_dims"], resolved["model_variant"], **resolved["model_hparams"]).to(DEVICE)
            bridge.load_state_dict(checkpoint["model"]); bridge.eval()
            datasets = {split: {m: ExternalPairs(m, experiment, split, text_view) for m in ["ecg", "cxr"]} for split in ["train", "val"]}
            counts = {split: {m: len(datasets[split][m]) for m in ["ecg", "cxr"]} for split in ["train", "val"]}
            too_small = [(split, modality, count) for split, values in counts.items()
                         for modality, count in values.items() if count < 2]
            if too_small:
                raise ValueError(f"Need at least two jointly encoded rows per train/val modality; found {too_small}. Complete notebooks 13-14.")
            insufficient_groups = [(split, modality, int(np.unique(datasets[split][modality].group).size))
                                   for split in ["train", "val"] for modality in ["ecg", "cxr"]
                                   if np.unique(datasets[split][modality].group).size < 2]
            if insufficient_groups:
                raise ValueError(f"Contrastive training/validation requires at least two distinct texts: {insufficient_groups}")
            text_dims = {m: int(datasets["train"][m].text.shape[1]) for m in ["ecg", "cxr"]}
            if len(set(text_dims.values())) != 1:
                raise ValueError(f"ECG/CXR text dimensions differ: {text_dims}")
            text_dim = text_dims["ecg"]
            expected_common_dim = int(resolved["model_hparams"]["common_dim"])
            sensor_dims = {m: int(datasets["train"][m].sensor.shape[1]) for m in ["ecg", "cxr"]}
            if set(sensor_dims.values()) != {expected_common_dim}:
                raise ValueError(f"Continuous shared-feature dimensions do not match bridge common_dim: {sensor_dims}")
            model = Adapter(bridge, text_dim, shared_text).to(DEVICE)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            loader_generator = torch.Generator().manual_seed(seed)
            train_loaders = {m: DataLoader(
                datasets["train"][m], batch_size=BATCH, shuffle=True,
                drop_last=len(datasets["train"][m]) >= BATCH,
                num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
                persistent_workers=NUM_WORKERS > 0, generator=loader_generator,
            ) for m in ["ecg", "cxr"]}
            val_loaders = {m: DataLoader(
                datasets["val"][m], batch_size=BATCH, shuffle=False, drop_last=False,
                num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
                persistent_workers=NUM_WORKERS > 0,
            ) for m in ["ecg", "cxr"]}

            def batch_loss(modality, batch):
                sensor, text, group, subject = [batch[k].to(DEVICE, non_blocking=True)
                                                for k in ["sensor", "text", "group", "subject"]]
                return contrastive(model.sensor(modality, sensor), model.text(modality, text),
                                   group, subject, TEMPERATURE)

            def train_epoch():
                model.train(True); losses = []
                iterators = {m: iter(train_loaders[m]) for m in ["ecg", "cxr"]}
                for _ in range(max(len(train_loaders["ecg"]), len(train_loaders["cxr"]))):
                    terms = []
                    for modality in ["ecg", "cxr"]:
                        try:
                            batch = next(iterators[modality])
                        except StopIteration:
                            iterators[modality] = iter(train_loaders[modality])
                            batch = next(iterators[modality])
                        if len(batch["sensor"]) >= 2:
                            terms.append(batch_loss(modality, batch))
                    if not terms:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.stack(terms).mean(); loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
                    optimizer.step(); losses.append(float(loss.detach().cpu()))
                if not losses:
                    raise RuntimeError("No usable training batches; reduce SEMANTIC_ADAPTER_BATCH_SIZE")
                return float(np.mean(losses))

            @torch.inference_mode()
            def validation_epoch():
                model.eval(); modality_losses = []
                for modality in ["ecg", "cxr"]:
                    weighted, total_rows = 0.0, 0
                    for batch in val_loaders[modality]:
                        if len(batch["sensor"]) < 2:
                            continue
                        loss = float(batch_loss(modality, batch).cpu())
                        weighted += loss * len(batch["sensor"]); total_rows += len(batch["sensor"])
                    if total_rows == 0:
                        raise RuntimeError(f"No usable {modality} validation batches")
                    modality_losses.append(weighted / total_rows)
                return float(np.mean(modality_losses))

            completion_path = run_dir / "completed.json"
            completion_path.write_text(json.dumps({
                "status": "running", "signature": signature,
                "message": "A running marker is never treated as a completed adapter."
            }, indent=2))
            best, stale, history = float("inf"), 0, []
            for ep in range(EPOCHS):
                tr, va = train_epoch(), validation_epoch()
                if not np.isfinite(tr) or not np.isfinite(va):
                    raise FloatingPointError(f"Non-finite adapter loss: train={tr}, val={va}")
                history.append({"epoch": ep, "train": tr, "val": va})
                print(experiment["variant"], experiment["seed"], semantic_name, seed, history[-1])
                state = {"model": model.state_dict(), "bridge_checkpoint": experiment["checkpoint"],
                         "experiment": experiment, "semantic_variant": semantic_name, "text_view": text_view,
                         "shared_text": shared_text, "text_dim": text_dim, "seed": seed, "epoch": ep,
                         "data_counts": counts,
                         "training_config": training_config}
                if va < best: best, stale = va, 0; torch.save(state, run_dir / "best.pt")
                else:
                    stale += 1
                    if stale >= PATIENCE: break
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
            if not (run_dir / "best.pt").is_file():
                raise RuntimeError(f"No adapter checkpoint was saved in {run_dir}")
            completion_reason = "early_stopping" if stale >= PATIENCE else "maximum_epochs"
            completion_path.write_text(json.dumps({
                "status": "complete", "signature": signature,
                "epochs_completed": len(history), "completion_reason": completion_reason,
                "best_validation_loss": best,
            }, indent=2))
            adapter_index.append({
                "bridge_variant": experiment["variant"], "bridge_seed": experiment["seed"],
                "semantic_variant": semantic_name, "adapter_seed": seed,
                "best_validation_loss": best, "checkpoint": str(run_dir / "best.pt"),
                "text_view": text_view, "data_counts": counts})
            del model, bridge
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    preserved = []
    for item in existing_adapter_index:
        try:
            if run_key(item) not in selected_run_keys and Path(item["checkpoint"]).is_file():
                preserved.append(item)
        except (KeyError, TypeError, ValueError):
            continue
    merged = {}
    for item in preserved + adapter_index:
        merged[run_key(item)] = item
    final_adapter_index = [merged[key] for key in sorted(merged)]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(final_adapter_index, indent=2))
    print({"selected_runs": len(selected_run_keys), "completed_or_trained": len(adapter_index),
           "preserved_unselected": len(preserved), "index_entries": len(final_adapter_index)})
    print("Saved:", index_path)
    """),
])


if False:  # Superseded by the assertion-aware, relation-separated evaluation below.
    write("16_evaluate_external_retrieval_ablations.ipynb", [
    md("""
    # 16 - Evaluate external retrieval ablations

    Primary endpoints use held-out external test data and deterministic candidate pools:

    1. ECG waveform → its machine diagnosis, and reverse.
    2. CXR image → its report, and reverse.
    3. ECG ↔ CXR admission/patient retrieval, only if common identifiers exist.
    4. ECG ↔ CXR semantic compatibility retrieval, clearly labeled as concept-derived rather than identity ground truth.

    Results are reported separately for 10-, 100-, and 1000-candidate pools. Duplicated report text is handled both as strict instance retrieval and as text-equivalent multi-positive retrieval.
    """),
    code("""
    from pathlib import Path
    import importlib
    import json
    import os
    import sys

    import numpy as np
    import pandas as pd
    import torch
    from torch.nn import functional as F

    ARTIFACT_ROOT = Path(os.environ.get("ECG_CXR_BRIDGE_ARTIFACT_ROOT", "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    CODE_ROOT = ARTIFACT_ROOT / "code"
    if str(CODE_ROOT) not in sys.path: sys.path.insert(0, str(CODE_ROOT))
    import ecg_cxr_ablation
    importlib.reload(ecg_cxr_ablation)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter_index = json.loads((ARTIFACT_ROOT / "external" / "adapters" / "adapter_index.json").read_text())

    def rows_for(modality):
        path = ARTIFACT_ROOT / "external" / modality / "manifest.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = {m: rows_for(m) for m in ["ecg", "cxr"]}
    concepts = {m: [json.loads(line)["concepts"] for line in (ARTIFACT_ROOT / "external" / m / "concepts.jsonl").read_text().splitlines()] for m in ["ecg", "cxr"]}
    """),
    code("""
    def load_adapter(entry):
        state = torch.load(entry["checkpoint"], map_location=DEVICE, weights_only=False)
        bridge_state = torch.load(state["bridge_checkpoint"], map_location=DEVICE, weights_only=False)
        resolved = bridge_state["resolved_config"]
        bridge = ecg_cxr_ablation.AblationBridge(resolved["input_dims"], resolved["model_variant"], **resolved["model_hparams"]).to(DEVICE)
        bridge.load_state_dict(bridge_state["model"])
        from torch import nn
        class Adapter(nn.Module):
            def __init__(self):
                super().__init__(); self.bridge = bridge; dim = bridge.common_dim
                self.shared_text = state["shared_text"]
                self.residual = nn.ModuleDict({m: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim)) for m in ["ecg", "cxr"]})
                names = ["shared"] if self.shared_text else ["ecg", "cxr"]
                self.text_projector = nn.ModuleDict({name: nn.Sequential(nn.LayerNorm(state["text_dim"]), nn.Linear(state["text_dim"], dim)) for name in names})
            def quantize(self, x):
                if not self.bridge.uses_vq: return F.normalize(x, dim=-1)
                return F.normalize(self.bridge.common_codebook(x)[0], dim=-1)
            def sensor(self, m, x): return self.quantize(x + self.residual[m](x))
            def text(self, m, x): return self.quantize(self.text_projector["shared" if self.shared_text else m](x))
        model = Adapter().to(DEVICE); model.load_state_dict(state["model"]); return model.eval().requires_grad_(False), state

    @torch.inference_mode()
    def encode_test(entry, modality):
        model, state = load_adapter(entry)
        experiment = state["experiment"]
        base = ARTIFACT_ROOT / "external" / modality
        continuous = np.load(base / "bridge" / experiment["variant"] / f"seed_{experiment['seed']}" / "continuous_common.npy", mmap_mode="r")
        text = np.load(base / f"text_{state['text_view']}_biomedclip.npy", mmap_mode="r")
        status = np.load(base / "encoding_status.npy", mmap_mode="r")
        index = np.asarray([i for i, row in enumerate(rows[modality]) if row["split"] == "test" and status[i] == 1])
        sensor_parts, text_parts = [], []
        for start in range(0, len(index), 1024):
            take = index[start:start + 1024]
            sensor_parts.append(model.sensor(modality, torch.from_numpy(np.array(continuous[take])).to(DEVICE)).cpu())
            text_parts.append(model.text(modality, torch.from_numpy(np.array(text[take])).to(DEVICE)).cpu())
        return index, torch.cat(sensor_parts), torch.cat(text_parts)
    """),
    code("""
    def sampled_paired(query, candidate, groups, candidate_count, seed=2026, max_queries=5000, multipositive=False):
        rng = np.random.default_rng(seed); n = len(query)
        selected_queries = rng.choice(n, size=min(n, max_queries), replace=False)
        ranks = []
        for i in selected_queries:
            negatives = np.delete(np.arange(n), i)
            sampled = rng.choice(negatives, size=min(candidate_count - 1, len(negatives)), replace=False)
            pool = np.concatenate([[i], sampled]); scores = candidate[pool] @ query[i]
            if multipositive:
                positive = np.asarray([groups[j] == groups[i] for j in pool])
                rank = 1 + int((scores > scores[positive].max()).sum())
            else:
                rank = 1 + int((scores > scores[0]).sum())
            ranks.append(rank)
        ranks = np.asarray(ranks)
        return {"queries": len(ranks), "candidates": min(candidate_count, n),
                "recall_at_1": float(np.mean(ranks <= 1)), "recall_at_5": float(np.mean(ranks <= 5)),
                "recall_at_10": float(np.mean(ranks <= 10)), "mrr": float(np.mean(1 / ranks)),
                "median_rank": float(np.median(ranks)), "chance_r1": 1 / min(candidate_count, n)}

    def normalized_group(modality, index):
        key = "machine_diagnosis" if modality == "ecg" else "report_text"
        return [" ".join(rows[modality][i][key].lower().split()) for i in index]
    """),
    code("""
    results = []
    for entry in adapter_index:
        encoded = {m: encode_test(entry, m) for m in ["ecg", "cxr"]}
        for modality in ["ecg", "cxr"]:
            index, sensor, text = encoded[modality]
            groups = normalized_group(modality, index)
            for candidates in [10, 100, 1000]:
                for direction, query, target in [("sensor_to_text", sensor, text), ("text_to_sensor", text, sensor)]:
                    strict = sampled_paired(query, target, groups, candidates, multipositive=False)
                    equivalent = sampled_paired(query, target, groups, candidates, multipositive=True)
                    results.append({**entry, "task": f"{modality}_{direction}", "label_policy": "strict_instance", **strict})
                    results.append({**entry, "task": f"{modality}_{direction}", "label_policy": "text_equivalent", **equivalent})
        print("evaluated", entry["bridge_variant"], entry["semantic_variant"], entry["adapter_seed"])
    frame = pd.DataFrame(results)
    output = ARTIFACT_ROOT / "reports" / "ablation" / "external_paired_retrieval.csv"
    frame.to_csv(output, index=False)
    aggregate = frame.groupby(["bridge_variant", "semantic_variant", "task", "label_policy", "candidates"], as_index=False).agg(
        seeds=("recall_at_1", "size"), recall_at_1_mean=("recall_at_1", "mean"),
        recall_at_1_std=("recall_at_1", "std"), recall_at_5_mean=("recall_at_5", "mean"),
        recall_at_10_mean=("recall_at_10", "mean"), mrr_mean=("mrr", "mean"))
    aggregate_path = ARTIFACT_ROOT / "reports" / "ablation" / "external_paired_retrieval_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    display(frame.sort_values(["candidates", "task", "recall_at_1"], ascending=[True, True, False]).head(40))
    print("Saved:", output, aggregate_path)
    """),
    md("""
    ## Cross-sensor identity retrieval

    The following evaluation is deliberately conditional. Admission-level relevance is used when both manifests provide `hadm_id`; otherwise patient-level relevance is used when both provide `subject_id`. If neither identifier overlaps, the evaluation is skipped rather than inventing diagonal ground truth.
    """),
    code("""
    def identifier_map(modality, index, key):
        return [rows[modality][i].get(key) for i in index]

    identity_results = []
    cxr_manifest_summary = json.loads((ARTIFACT_ROOT / "external" / "cxr" / "manifest_summary.json").read_text())
    identity_allowed = cxr_manifest_summary.get("split_basis") == "subject_id"
    if not identity_allowed:
        print("Skipping cross-sensor identity retrieval because the CXR split is not patient-disjoint.")
    for entry in adapter_index:
        if not identity_allowed: break
        ecg_i, ecg_sensor, _ = encode_test(entry, "ecg")
        cxr_i, cxr_sensor, _ = encode_test(entry, "cxr")
        key = None
        for candidate_key in ["hadm_id", "subject_id"]:
            left = set(v for v in identifier_map("ecg", ecg_i, candidate_key) if v is not None)
            right = set(v for v in identifier_map("cxr", cxr_i, candidate_key) if v is not None)
            if left & right: key = candidate_key; break
        if key is None:
            continue
        left_ids, right_ids = identifier_map("ecg", ecg_i, key), identifier_map("cxr", cxr_i, key)
        right_lookup = {}
        for j, value in enumerate(right_ids):
            if value is not None: right_lookup.setdefault(value, []).append(j)
        rng = np.random.default_rng(2026); ranks = []; candidate_count = min(1000, len(right_ids))
        eligible = [i for i, value in enumerate(left_ids) if value in right_lookup]
        if len(eligible) > 5000: eligible = rng.choice(eligible, size=5000, replace=False).tolist()
        all_right = np.arange(len(right_ids)); right_array = np.asarray(right_ids, dtype=object)
        for i in eligible:
            value = left_ids[i]; positive_all = np.asarray(right_lookup[value], dtype=int)
            positive_pick = int(rng.choice(positive_all))
            negative_pool = np.flatnonzero(right_array != value)
            negatives = rng.choice(negative_pool, size=min(candidate_count - 1, len(negative_pool)), replace=False)
            pool = np.concatenate([[positive_pick], negatives])
            scores = cxr_sensor[pool] @ ecg_sensor[i]
            ranks.append(1 + int((scores > scores[0]).sum()))
        identity_results.append({**entry, "identity_key": key, "queries": len(ranks),
                                 "candidates": candidate_count,
                                 "recall_at_1": float(np.mean(np.asarray(ranks) <= 1)) if ranks else None,
                                 "recall_at_10": float(np.mean(np.asarray(ranks) <= 10)) if ranks else None,
                                 "mrr": float(np.mean(1 / np.asarray(ranks))) if ranks else None})
    identity_path = ARTIFACT_ROOT / "reports" / "ablation" / "external_cross_sensor_identity.json"
    identity_path.write_text(json.dumps(identity_results, indent=2))
    print(json.dumps(identity_results[:5], indent=2)); print("Saved:", identity_path)
    """),
    md("""
    ## Cross-sensor semantic compatibility retrieval

    This secondary analysis uses a preregistered, transparent compatibility map. It is not patient-pair ground truth. The default strong relations are ventricular hypertrophy→cardiomegaly, paced rhythm→support device, and heart failure→cardiomegaly/edema/pleural effusion. Audited LLM concepts can add exact shared concept names without changing the evaluation code.
    """),
    code("""
    COMPATIBILITY = {
        "ventricular_hypertrophy": {"cardiomegaly"},
        "paced_rhythm": {"support_device"},
        "heart_failure": {"cardiomegaly", "edema", "pleural_effusion"},
    }

    def ecg_cxr_relevant(ecg_values, cxr_values):
        ecg_set, cxr_set = set(ecg_values), set(cxr_values)
        if ecg_set & cxr_set:
            return True
        return any(bool(targets & cxr_set) for source, targets in COMPATIBILITY.items() if source in ecg_set)

    def semantic_sampled(query, target, query_concepts, target_concepts, direction, candidates=1000, max_queries=5000):
        rng = np.random.default_rng(2027); eligible = []; relevance_cache = {}
        target_lookup = {}
        for j, values in enumerate(target_concepts):
            for concept in values: target_lookup.setdefault(concept, []).append(j)
        for i, qconcept in enumerate(query_concepts):
            if direction == "ecg_to_cxr":
                desired = set(qconcept)
                for concept in qconcept: desired.update(COMPATIBILITY.get(concept, set()))
            else:
                desired = set(qconcept)
                desired.update(source for source, targets in COMPATIBILITY.items() if targets & set(qconcept))
            relevant = sorted({j for concept in desired for j in target_lookup.get(concept, [])})
            if relevant:
                eligible.append(i); relevance_cache[i] = np.asarray(relevant, dtype=int)
        if len(eligible) > max_queries: eligible = rng.choice(eligible, size=max_queries, replace=False).tolist()
        ranks = []; universe = np.arange(len(target_concepts)); candidate_count = min(candidates, len(universe))
        for i in eligible:
            relevant = relevance_cache[i]; positive = int(rng.choice(relevant))
            negative_pool = np.setdiff1d(universe, relevant, assume_unique=False)
            negatives = rng.choice(negative_pool, size=min(candidate_count - 1, len(negative_pool)), replace=False)
            pool = np.concatenate([[positive], negatives]); scores = target[pool] @ query[i]
            ranks.append(1 + int((scores > scores[0]).sum()))
        ranks = np.asarray(ranks)
        return {"queries": len(ranks), "candidates": candidate_count,
                "recall_at_1": float(np.mean(ranks <= 1)) if len(ranks) else None,
                "recall_at_5": float(np.mean(ranks <= 5)) if len(ranks) else None,
                "recall_at_10": float(np.mean(ranks <= 10)) if len(ranks) else None,
                "mrr": float(np.mean(1 / ranks)) if len(ranks) else None,
                "ground_truth": "preregistered semantic compatibility; not patient identity"}

    semantic_results = []
    for entry in adapter_index:
        ecg_i, ecg_sensor, _ = encode_test(entry, "ecg")
        cxr_i, cxr_sensor, _ = encode_test(entry, "cxr")
        ecg_concepts = [concepts["ecg"][i] for i in ecg_i]
        cxr_concepts = [concepts["cxr"][i] for i in cxr_i]
        semantic_results.append({**entry, "direction": "ecg_to_cxr",
                                 **semantic_sampled(ecg_sensor, cxr_sensor, ecg_concepts, cxr_concepts, "ecg_to_cxr")})
        semantic_results.append({**entry, "direction": "cxr_to_ecg",
                                 **semantic_sampled(cxr_sensor, ecg_sensor, cxr_concepts, ecg_concepts, "cxr_to_ecg")})
    semantic_path = ARTIFACT_ROOT / "reports" / "ablation" / "external_cross_sensor_semantic.json"
    semantic_path.write_text(json.dumps(semantic_results, indent=2))
    display(pd.DataFrame(semantic_results).sort_values(["direction", "recall_at_1"], ascending=[True, False]).head(30))
    print("Saved:", semantic_path)
    """),
    md("""
    ## Interpretation rules

    - Use strict paired sensor-text retrieval as the primary external endpoint.
    - Treat text-equivalent retrieval as a secondary endpoint because many machine diagnoses are duplicated.
    - Cross-sensor identity results are valid only for actual overlapping identifiers.
    - Rule- or LLM-derived compatibility is hypothesis-generating and must not be described as patient-pair ground truth.
    - Compare variants by mean and standard deviation across seeds; do not select a method using test performance.
    """),
])


# Replacement for notebook 16.  The earlier definition is retained above only so
# old generated studies remain reproducible; this final write is authoritative.
write("16_evaluate_external_retrieval_ablations.ipynb", [
    md("""
    # 16 - Evaluate external ECG-CXR bridge ablations

    This notebook keeps three claims separate:

    1. **Paired external supervision:** ECG waveform-machine diagnosis and CXR image-report retrieval.
    2. **External ECG-CXR patient identity:** explicitly not evaluable because these separately sourced external datasets do not provide a shared patient/admission linkage.
    3. **Concept-derived semantics:** cross-patient retrieval using explicit, assertion-aware clinical compatibility rules. This is a secondary, hypothesis-generating endpoint and is not a substitute for patient-identity validation.

    All tests use held-out data and deterministic 10-, 100-, and 1000-candidate pools. In addition to Hit/Recall@K and MRR, the notebook reports multi-positive precision, recall, MAP, nDCG, AUROC, similarity margins, tie diagnostics for discrete codes, shuffled-query controls, per-relation semantic results, and codebook utilization/collapse measures.
    """),
    code("""
    from pathlib import Path
    import importlib
    import json
    import os
    import sys

    import numpy as np
    import pandas as pd
    import torch
    from torch.nn import functional as F

    ARTIFACT_ROOT = Path(os.environ.get(
        "ECG_CXR_BRIDGE_ARTIFACT_ROOT",
        "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    REPORT_ROOT = ARTIFACT_ROOT / "reports" / "ablation"
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    CODE_ROOT = ARTIFACT_ROOT / "code"
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    import ecg_cxr_ablation
    importlib.reload(ecg_cxr_ablation)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CANDIDATE_COUNTS = [10, 100, 1000]
    MAX_QUERIES = int(os.environ.get("EXTERNAL_EVAL_MAX_QUERIES", "5000"))
    if MAX_QUERIES < 1:
        raise ValueError("EXTERNAL_EVAL_MAX_QUERIES must be positive")

    adapter_path = ARTIFACT_ROOT / "external" / "adapters" / "adapter_index.json"
    adapter_index = json.loads(adapter_path.read_text())
    if not isinstance(adapter_index, list) or not adapter_index:
        raise ValueError(f"No trained adapters in {adapter_path}; run notebook 15 first")

    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    rows = {m: read_jsonl(ARTIFACT_ROOT / "external" / m / "manifest.jsonl")
            for m in ["ecg", "cxr"]}
    concept_rows = {m: read_jsonl(ARTIFACT_ROOT / "external" / m / "concepts.jsonl")
                    for m in ["ecg", "cxr"]}
    concepts = {}
    for modality in ["ecg", "cxr"]:
        if len(rows[modality]) != len(concept_rows[modality]):
            raise ValueError(f"{modality} manifest/concept length mismatch; rerun notebook 14")
        for index, (row, concept) in enumerate(zip(rows[modality], concept_rows[modality])):
            if str(row["sample_id"]) != str(concept.get("sample_id")):
                raise ValueError(f"{modality} concept order mismatch at row {index}")
        concepts[modality] = [item.get("concepts", []) for item in concept_rows[modality]]

    print({"device": str(DEVICE), "adapter_runs": len(adapter_index),
           "rows": {m: len(v) for m, v in rows.items()},
           "report_root": str(REPORT_ROOT)})
    """),
    code("""
    def torch_load(path):
        try:
            return torch.load(path, map_location=DEVICE, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=DEVICE)

    def load_adapter(entry):
        state = torch_load(entry["checkpoint"])
        bridge_state = torch_load(state["bridge_checkpoint"])
        resolved = bridge_state["resolved_config"]
        bridge = ecg_cxr_ablation.AblationBridge(
            resolved["input_dims"], resolved["model_variant"],
            **resolved["model_hparams"]).to(DEVICE)
        bridge.load_state_dict(bridge_state["model"])
        from torch import nn

        class Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.bridge = bridge
                dim = bridge.common_dim
                self.shared_text = state["shared_text"]
                self.residual = nn.ModuleDict({
                    m: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
                    for m in ["ecg", "cxr"]})
                names = ["shared"] if self.shared_text else ["ecg", "cxr"]
                self.text_projector = nn.ModuleDict({
                    name: nn.Sequential(nn.LayerNorm(state["text_dim"]), nn.Linear(state["text_dim"], dim))
                    for name in names})

            def quantize_with_codes(self, values):
                if not self.bridge.uses_vq:
                    return F.normalize(values, dim=-1), None
                quantized, indices, _, _ = self.bridge.common_codebook(values)
                return F.normalize(quantized, dim=-1), indices

            def sensor(self, modality, values):
                return self.quantize_with_codes(values + self.residual[modality](values))

            def text(self, modality, values):
                key = "shared" if self.shared_text else modality
                return self.quantize_with_codes(self.text_projector[key](values))

        model = Adapter().to(DEVICE)
        model.load_state_dict(state["model"])
        return model.eval().requires_grad_(False), state, resolved

    ENCODED = {}

    @torch.inference_mode()
    def encode_test(entry, modality):
        cache_key = (str(entry["checkpoint"]), modality)
        if cache_key in ENCODED:
            return ENCODED[cache_key]
        model, state, resolved = load_adapter(entry)
        experiment = state["experiment"]
        base = ARTIFACT_ROOT / "external" / modality
        continuous = np.load(
            base / "bridge" / experiment["variant"] / f"seed_{experiment['seed']}" / "continuous_common.npy",
            mmap_mode="r")
        text = np.load(base / f"text_{state['text_view']}_biomedclip.npy", mmap_mode="r")
        sensor_status = np.load(base / "encoding_status.npy", mmap_mode="r")
        text_status = np.load(base / f"text_{state['text_view']}_encoding_status.npy", mmap_mode="r")
        expected = len(rows[modality])
        lengths = [len(continuous), len(text), len(sensor_status), len(text_status)]
        if any(length != expected for length in lengths):
            raise ValueError(f"{modality} evaluation array/manifest mismatch: {lengths} vs {expected}")
        index = np.asarray([
            i for i, row in enumerate(rows[modality])
            if row["split"] == "test" and sensor_status[i] == 1 and text_status[i] == 1
        ], dtype="int64")
        if len(index) < 2:
            raise ValueError(f"Need at least two jointly encoded {modality} test rows; found {len(index)}")

        sensor_parts, text_parts, sensor_code_parts, text_code_parts = [], [], [], []
        for start in range(0, len(index), 1024):
            take = index[start:start + 1024]
            sensor, sensor_codes = model.sensor(
                modality, torch.from_numpy(np.array(continuous[take])).to(DEVICE))
            text_value, text_codes = model.text(
                modality, torch.from_numpy(np.array(text[take])).to(DEVICE))
            sensor_parts.append(sensor.cpu())
            text_parts.append(text_value.cpu())
            if sensor_codes is not None:
                sensor_code_parts.append(sensor_codes.cpu())
                text_code_parts.append(text_codes.cpu())
        result = {
            "index": index,
            "sensor": torch.cat(sensor_parts),
            "text": torch.cat(text_parts),
            "sensor_codes": torch.cat(sensor_code_parts).numpy() if sensor_code_parts else None,
            "text_codes": torch.cat(text_code_parts).numpy() if text_code_parts else None,
            "uses_vq": bool(model.bridge.uses_vq),
            "codebook_size": int(resolved["model_hparams"].get("common_codebook_size", 0)),
        }
        for key in ["sensor", "text"]:
            if not torch.isfinite(result[key]).all():
                raise ValueError(f"Non-finite {modality} {key} test embeddings")
        ENCODED[cache_key] = result
        return result
    """),
    code("""
    def chance_hit(candidate_count, positive_count, k):
        n, p, k = int(candidate_count), int(positive_count), min(int(k), int(candidate_count))
        if p <= 0 or n <= 0:
            return np.nan
        if k > n - p:
            return 1.0
        no_positive = 1.0
        for offset in range(k):
            no_positive *= (n - p - offset) / (n - offset)
        return 1.0 - no_positive

    def binary_ranking_metrics(scores, relevant, rng, ks=(1, 5, 10)):
        scores = np.asarray(scores, dtype="float64")
        relevant = np.asarray(relevant, dtype=bool)
        if scores.ndim != 1 or len(scores) != len(relevant) or not relevant.any():
            raise ValueError("Each candidate pool must contain at least one relevant item")
        # Random secondary keys give deterministic, unbiased tie breaking. This is
        # important for VQ models, where many candidates can have identical scores.
        order = np.lexsort((rng.random(len(scores)), -scores))
        ranked = relevant[order].astype("float64")
        positive_count = int(relevant.sum())
        first_rank = int(np.flatnonzero(ranked)[0]) + 1
        cumulative = np.cumsum(ranked)
        positive_scores, negative_scores = scores[relevant], scores[~relevant]
        best_positive = float(positive_scores.max())
        optimistic_rank = 1 + int((scores > best_positive).sum())
        tied_nonrelevant = int(np.isclose(negative_scores, best_positive, rtol=1e-6, atol=1e-8).sum())
        if len(negative_scores):
            ordered_negative = np.sort(negative_scores)
            wins = np.searchsorted(ordered_negative, positive_scores, side="left")
            ties = (np.searchsorted(ordered_negative, positive_scores, side="right") - wins)
            auc = float(np.mean((wins + 0.5 * ties) / len(negative_scores)))
            margin = best_positive - float(negative_scores.max())
            mean_negative = float(negative_scores.mean())
        else:
            auc, margin, mean_negative = np.nan, np.nan, np.nan
        result = {
            "reciprocal_rank": 1.0 / first_rank,
            "first_relevant_rank": float(first_rank),
            "optimistic_reciprocal_rank": 1.0 / optimistic_rank,
            "best_positive_tied_with_nonrelevant": float(tied_nonrelevant > 0),
            "nonrelevant_ties_at_best_positive": float(tied_nonrelevant),
            "auroc": auc,
            "similarity_margin": margin,
            "mean_positive_similarity": float(positive_scores.mean()),
            "mean_negative_similarity": mean_negative,
            "positive_prevalence": positive_count / len(scores),
        }
        for k in ks:
            take = min(k, len(scores))
            hits = float(cumulative[take - 1])
            precision = hits / take
            recall = hits / positive_count
            discounts = 1.0 / np.log2(np.arange(2, take + 2))
            dcg = float((ranked[:take] * discounts).sum())
            ideal = float(discounts[:min(positive_count, take)].sum())
            precision_curve = cumulative[:take] / np.arange(1, take + 1)
            ap = float((precision_curve * ranked[:take]).sum() / min(positive_count, take))
            result.update({
                f"hit_at_{k}": float(hits > 0),
                f"precision_at_{k}": precision,
                f"recall_at_{k}": recall,
                f"map_at_{k}": ap,
                f"ndcg_at_{k}": dcg / ideal if ideal else np.nan,
                f"chance_hit_at_{k}": chance_hit(len(scores), positive_count, take),
            })
        return result

    def average_query_metrics(records):
        if not records:
            empty = {
                "queries": 0, "first_relevant_rank": None, "median_rank": None,
                "mrr": None, "optimistic_mrr": None,
                "best_positive_tied_with_nonrelevant": None,
                "nonrelevant_ties_at_best_positive": None, "auroc": None,
                "similarity_margin": None, "mean_positive_similarity": None,
                "mean_negative_similarity": None, "positive_prevalence": None,
            }
            for k in [1, 5, 10]:
                for name in ["hit", "precision", "recall", "map", "ndcg", "chance_hit"]:
                    empty[f"{name}_at_{k}"] = None
            return empty
        keys = records[0].keys()
        result = {"queries": len(records)}
        for key in keys:
            values = np.asarray([item[key] for item in records], dtype="float64")
            result[key] = float(np.nanmean(values)) if not np.isnan(values).all() else None
        ranks = np.asarray([item["first_relevant_rank"] for item in records])
        result["median_rank"] = float(np.median(ranks))
        result["mrr"] = result.pop("reciprocal_rank")
        result["optimistic_mrr"] = result.pop("optimistic_reciprocal_rank")
        return result

    def sampled_relevance(query, target, relevance, candidate_count, seed,
                          max_queries=MAX_QUERIES, excluded=None):
        rng = np.random.default_rng(seed)
        eligible = np.asarray(sorted(relevance), dtype="int64")
        if len(eligible) > max_queries:
            eligible = rng.choice(eligible, size=max_queries, replace=False)
        universe = np.arange(len(target), dtype="int64")
        records = []
        for query_index in eligible:
            positive = np.asarray(relevance[int(query_index)], dtype="int64")
            blocked = set() if excluded is None else set(excluded.get(int(query_index), ()))
            positive = np.asarray([value for value in positive if int(value) not in blocked], dtype="int64")
            if not len(positive):
                continue
            anchor = int(rng.choice(positive))
            allowed = np.asarray([value for value in universe
                                  if int(value) != anchor and int(value) not in blocked], dtype="int64")
            sampled = rng.choice(
                allowed, size=min(candidate_count - 1, len(allowed)), replace=False)
            pool = np.concatenate([[anchor], sampled])
            relevant_pool = np.isin(pool, positive)
            scores = (target[pool] @ query[int(query_index)]).numpy()
            records.append(binary_ranking_metrics(scores, relevant_pool, rng))
        result = average_query_metrics(records)
        result.update({"requested_candidates": int(candidate_count),
                       "candidates": min(int(candidate_count), len(target))})
        return result

    def normalized_group(modality, index):
        key = "machine_diagnosis" if modality == "ecg" else "report_text"
        return [" ".join(str(rows[modality][i][key]).lower().split()) for i in index]

    def sampled_paired(query, candidate, groups, candidate_count, multipositive, seed=2026):
        n = len(query)
        if multipositive:
            lookup = {}
            for j, value in enumerate(groups):
                lookup.setdefault(value, []).append(j)
            relevance = {i: np.asarray(lookup[groups[i]], dtype="int64") for i in range(n)}
        else:
            relevance = {i: np.asarray([i], dtype="int64") for i in range(n)}
        return sampled_relevance(query, candidate, relevance, candidate_count, seed)
    """),
    md("""
    ## Primary held-out Symile relational retrieval

    This is the direct test of the codebook's intended ground truth. Within each official 10-candidate Symile group, the labeled ECG-CXR pair is the documented same-patient/encounter association. The metrics quantify retrieval of that association only; they do not assign a shared diagnosis to the two modalities. Unlike the earlier evaluation, discrete-score ties are broken deterministically and are also reported explicitly rather than counted optimistically as successful retrieval.
    """),
    code("""
    @torch.inference_mode()
    def encode_symile_bridge(experiment, split="test"):
        checkpoint = torch_load(experiment["checkpoint"])
        resolved = checkpoint["resolved_config"]
        model = ecg_cxr_ablation.AblationBridge(
            resolved["input_dims"], resolved["model_variant"],
            **resolved["model_hparams"]).to(DEVICE).eval()
        model.load_state_dict(checkpoint["model"])
        paths = {
            "ecg": ARTIFACT_ROOT / "features" / split / "ecg_hubert.npy",
            "cxr": ARTIFACT_ROOT / "features" / split / "cxr_raddino.npy",
            "manifest": ARTIFACT_ROOT / "manifests" / f"{split}.jsonl",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Run notebooks 02-04 before notebook 16; missing:\\n" + "\\n".join(missing))
        arrays = {m: np.load(paths[m], mmap_mode="r") for m in ["ecg", "cxr"]}
        manifest = read_jsonl(paths["manifest"])
        if any(len(arrays[m]) != len(manifest) for m in arrays):
            raise ValueError(f"Symile {split} feature/manifest row mismatch")
        output = {"ecg": [], "cxr": []}
        for start in range(0, len(manifest), 1024):
            stop = min(start + 1024, len(manifest))
            batch = {m: torch.from_numpy(np.array(arrays[m][start:stop])).float().to(DEVICE)
                     for m in ["ecg", "cxr"]}
            encoded = model(batch)
            for modality in output:
                output[modality].append(encoded[modality].common.cpu())
        output = {key: torch.cat(value) for key, value in output.items()}
        return output, manifest

    def symile_group_retrieval(encoded, manifest, query_name, candidate_name, seed=5101):
        groups = {}
        for row_index, row in enumerate(manifest):
            if "label_hadm_id" not in row or "label" not in row:
                raise KeyError("Symile test manifest requires label_hadm_id and label")
            groups.setdefault(int(row["label_hadm_id"]), []).append(row_index)
        invalid = {key: value for key, value in groups.items()
                   if len(value) != 10 or sum(int(manifest[i]["label"]) == 1 for i in value) != 1}
        if invalid:
            raise ValueError(f"Invalid Symile 10-candidate groups: {dict(list(invalid.items())[:10])}")
        rng = np.random.default_rng(seed)
        records = []
        for group_id in sorted(groups):
            pool = np.asarray(groups[group_id], dtype="int64")
            relevant = np.asarray([int(manifest[i]["label"]) == 1 for i in pool], dtype=bool)
            positive_index = int(pool[np.flatnonzero(relevant)[0]])
            scores = (encoded[candidate_name][pool] @ encoded[query_name][positive_index]).numpy()
            records.append(binary_ranking_metrics(scores, relevant, rng))
        result = average_query_metrics(records)
        result.update({"candidates": 10, "chance_hit_at_1_reference": 0.1,
                       "ground_truth": "official Symile same-patient/encounter pair; not semantic equivalence"})
        return result

    bridge_experiments = {}
    for entry in adapter_index:
        state = torch_load(entry["checkpoint"])
        experiment = state["experiment"]
        bridge_experiments[(experiment["variant"], int(experiment["seed"]))] = experiment

    symile_relational_results = []
    for (variant, seed), experiment in sorted(bridge_experiments.items()):
        encoded, manifest = encode_symile_bridge(experiment, "test")
        for direction, query_name, candidate_name in [
            ("ecg_to_cxr", "ecg", "cxr"), ("cxr_to_ecg", "cxr", "ecg")]:
            symile_relational_results.append({
                "bridge_variant": variant, "bridge_seed": seed, "direction": direction,
                **symile_group_retrieval(encoded, manifest, query_name, candidate_name)})
        print("Symile relational", variant, seed)

    symile_relational_frame = pd.DataFrame(symile_relational_results)
    symile_relational_path = REPORT_ROOT / "symile_relational_retrieval_detailed.csv"
    symile_relational_frame.to_csv(symile_relational_path, index=False)
    symile_relational_summary = symile_relational_frame.groupby(
        ["bridge_variant", "direction"], as_index=False).agg(
            seeds=("bridge_seed", "size"), hit_at_1_mean=("hit_at_1", "mean"),
            hit_at_1_std=("hit_at_1", "std"), hit_at_5_mean=("hit_at_5", "mean"),
            mrr_mean=("mrr", "mean"), map_at_10_mean=("map_at_10", "mean"),
            ndcg_at_10_mean=("ndcg_at_10", "mean"), auroc_mean=("auroc", "mean"),
            similarity_margin_mean=("similarity_margin", "mean"),
            tied_query_fraction_mean=("best_positive_tied_with_nonrelevant", "mean"))
    symile_relational_summary_path = REPORT_ROOT / "symile_relational_retrieval_summary.csv"
    symile_relational_summary.to_csv(symile_relational_summary_path, index=False)
    display(symile_relational_summary.sort_values(["direction", "ndcg_at_10_mean"],
                                                  ascending=[True, False]))
    print("Saved:", symile_relational_path, symile_relational_summary_path)
    """),
    code("""
    paired_results = []
    for entry in adapter_index:
        encoded = {m: encode_test(entry, m) for m in ["ecg", "cxr"]}
        for modality in ["ecg", "cxr"]:
            item = encoded[modality]
            groups = normalized_group(modality, item["index"])
            for candidates in CANDIDATE_COUNTS:
                directions = [
                    ("sensor_to_text", item["sensor"], item["text"]),
                    ("text_to_sensor", item["text"], item["sensor"]),
                ]
                for direction, query, target in directions:
                    for label_policy, multipositive in [
                        ("strict_instance", False), ("text_equivalent", True)]:
                        metrics = sampled_paired(
                            query, target, groups, candidates, multipositive=multipositive)
                        paired_results.append({**entry, "task": f"{modality}_{direction}",
                                               "label_policy": label_policy, **metrics})
        print("paired", entry["bridge_variant"], entry["semantic_variant"], entry["adapter_seed"])

    paired_frame = pd.DataFrame(paired_results)
    paired_path = REPORT_ROOT / "external_paired_retrieval.csv"
    paired_frame.to_csv(paired_path, index=False)
    metric_columns = [
        "hit_at_1", "hit_at_5", "hit_at_10", "recall_at_1", "recall_at_5", "recall_at_10",
        "precision_at_5", "precision_at_10", "map_at_10", "ndcg_at_10", "mrr", "auroc",
        "similarity_margin", "best_positive_tied_with_nonrelevant"]
    aggregations = {f"{name}_mean": (name, "mean") for name in metric_columns}
    aggregations.update({f"{name}_std": (name, "std") for name in ["hit_at_1", "mrr", "ndcg_at_10"]})
    aggregations["seeds"] = ("adapter_seed", "size")
    paired_summary = paired_frame.groupby(
        ["bridge_variant", "semantic_variant", "task", "label_policy", "candidates"],
        as_index=False).agg(**aggregations)
    paired_summary_path = REPORT_ROOT / "external_paired_retrieval_summary.csv"
    paired_summary.to_csv(paired_summary_path, index=False)
    display(paired_summary.sort_values(["candidates", "task", "ndcg_at_10_mean"],
                                       ascending=[True, True, False]).head(40))
    print("Saved:", paired_path, paired_summary_path)
    """),
    md("""
    ## External ECG-CXR patient-identity validation: not evaluable

    The external MIMIC-ECG and separately prepared MIMIC-CXR collections used here do not provide a verified shared patient/admission linkage. Consequently, external ECG-CXR patient-identity retrieval is not computed. The structured status files written below prevent an empty result from being mistaken for a failed run or a negative retrieval result.
    """),
    code("""
    # These helpers remain available to the later semantic analysis so it can
    # exclude matching identifiers if a future dataset supplies verified links.
    # Their presence does not make identity validation evaluable in this cohort.
    def identifier_values(modality, index, key):
        return [rows[modality][int(i)].get(key) for i in index]

    def common_identity_key(ecg_item, cxr_item, keys=("hadm_id", "subject_id")):
        for key in keys:
            left = {value for value in identifier_values("ecg", ecg_item["index"], key)
                    if value is not None}
            right = {value for value in identifier_values("cxr", cxr_item["index"], key)
                     if value is not None}
            if left & right:
                return key
        return None

    EXTERNAL_IDENTITY_LINKAGE_VERIFIED = False
    identity_status = {
        "status": "not_evaluable",
        "reason_code": "missing_shared_patient_identity",
        "reason": (
            "The external ECG and CXR datasets do not provide a verified shared patient "
            "or admission linkage; cross-sensor patient-identity retrieval cannot be evaluated."),
        "ecg_dataset": "MIMIC-IV-ECG waveform-machine-diagnosis pairs",
        "cxr_dataset": "separately prepared MIMIC-CXR image-report pairs",
        "metrics_computed": False,
        "interpretation": (
            "Do not interpret this endpoint as failed retrieval, zero performance, or external "
            "validation of the Symile patient-concurrence bridge."),
    }
    identity_path = REPORT_ROOT / "external_cross_sensor_identity.json"
    identity_path.write_text(json.dumps(identity_status, indent=2))
    identity_summary_path = REPORT_ROOT / "external_cross_sensor_identity_summary.csv"
    identity_summary = pd.DataFrame([identity_status])
    identity_summary.to_csv(identity_summary_path, index=False)
    display(identity_summary)
    print("Saved:", identity_path, identity_summary_path)
    """),
    md("""
    ## Cross-patient semantic compatibility retrieval

    This secondary analysis uses only concepts asserted as present. Negated and uncertain findings are excluded from positive labels. Same-admission or same-patient candidates are removed only when a verified cross-dataset identity linkage is available; no such linkage exists for the current external datasets. Unsupported@K is reported explicitly; it is the fraction of the top K without a preregistered compatible concept.
    """),
    code("""
    COMPATIBILITY = {
        "ventricular_hypertrophy": {"cardiomegaly"},
        "paced_rhythm": {"support_device"},
        "heart_failure": {"cardiomegaly", "edema", "pleural_effusion"},
    }

    def assertion_sets(values):
        result = {"present": set(), "absent": set(), "uncertain": set()}
        for value in values:
            text = str(value).strip().lower()
            if ":" not in text:
                continue
            status, concept = text.split(":", 1)
            if status in result and concept:
                result[status].add(concept)
        return result

    assertions = {
        modality: [assertion_sets(concepts[modality][int(i)])
                   for i in encode_test(adapter_index[0], modality)["index"]]
        for modality in ["ecg", "cxr"]
    }

    def identity_exclusions(query_modality, query_item, target_modality, target_item):
        if not EXTERNAL_IDENTITY_LINKAGE_VERIFIED:
            return None, {}
        key = common_identity_key(
            query_item if query_modality == "ecg" else target_item,
            target_item if target_modality == "cxr" else query_item,
            keys=("subject_id", "hadm_id"))
        if key is None:
            return None, {}
        query_ids = identifier_values(query_modality, query_item["index"], key)
        target_ids = identifier_values(target_modality, target_item["index"], key)
        lookup = {}
        for j, value in enumerate(target_ids):
            if value is not None:
                lookup.setdefault(value, []).append(j)
        return key, {i: np.asarray(lookup[value], dtype="int64")
                     for i, value in enumerate(query_ids) if value in lookup}

    def semantic_relevance(query_assertions, target_assertions, direction, exclusions):
        target_lookup = {}
        for j, values in enumerate(target_assertions):
            for concept in values["present"]:
                target_lookup.setdefault(concept, []).append(j)
        inverse = {}
        for source, targets in COMPATIBILITY.items():
            for target in targets:
                inverse.setdefault(target, set()).add(source)
        relevance = {}
        for i, values in enumerate(query_assertions):
            present = set(values["present"])
            desired = set(present)  # exact normalized concepts are semantic matches
            relation_sources = set()
            if direction == "ecg_to_cxr":
                for concept in present:
                    mapped = COMPATIBILITY.get(concept, set())
                    if mapped:
                        relation_sources.add(concept)
                        desired.update(mapped)
            else:
                for concept in present:
                    mapped = inverse.get(concept, set())
                    if mapped:
                        relation_sources.add(concept)
                        desired.update(mapped)
            relevant = {j for concept in desired for j in target_lookup.get(concept, [])}
            relevant.difference_update(set(exclusions.get(i, ())))
            if relevant:
                relevance[i] = np.asarray(sorted(relevant), dtype="int64")
        return relevance

    semantic_results, relation_results = [], []
    for entry in adapter_index:
        ecg_item, cxr_item = encode_test(entry, "ecg"), encode_test(entry, "cxr")
        item_assertions = {
            "ecg": [assertion_sets(concepts["ecg"][int(i)]) for i in ecg_item["index"]],
            "cxr": [assertion_sets(concepts["cxr"][int(i)]) for i in cxr_item["index"]],
        }
        directions = [
            ("ecg_to_cxr", "ecg", ecg_item, "cxr", cxr_item),
            ("cxr_to_ecg", "cxr", cxr_item, "ecg", ecg_item),
        ]
        for direction, query_modality, query_item, target_modality, target_item in directions:
            exclusion_key, exclusions = identity_exclusions(
                query_modality, query_item, target_modality, target_item)
            relevance = semantic_relevance(
                item_assertions[query_modality], item_assertions[target_modality], direction, exclusions)
            permutation = np.random.default_rng(4107).permutation(len(query_item["sensor"]))
            shuffled_query = query_item["sensor"][permutation]
            assertion_counts = {
                status: sum(bool(item[status]) for item in item_assertions[query_modality])
                for status in ["present", "absent", "uncertain"]}
            for candidates in CANDIDATE_COUNTS:
                metrics = sampled_relevance(
                    query_item["sensor"], target_item["sensor"], relevance, candidates, seed=4101,
                    excluded=exclusions)
                control = sampled_relevance(
                    shuffled_query, target_item["sensor"], relevance, candidates, seed=4101,
                    excluded=exclusions)
                semantic_results.append({
                    **entry, "direction": direction,
                    "ground_truth": "preregistered present-concept compatibility; not patient identity",
                    "identity_exclusion_key": exclusion_key,
                    "eligible_query_fraction": len(relevance) / len(query_item["sensor"]),
                    **{f"queries_with_{key}_concepts": value for key, value in assertion_counts.items()},
                    **metrics,
                    "unsupported_rate_at_1": 1.0 - metrics.get("precision_at_1", np.nan),
                    "unsupported_rate_at_5": 1.0 - metrics.get("precision_at_5", np.nan),
                    "unsupported_rate_at_10": 1.0 - metrics.get("precision_at_10", np.nan),
                    "shuffled_hit_at_1": control.get("hit_at_1"),
                    "shuffled_mrr": control.get("mrr"),
                    "shuffled_ndcg_at_10": control.get("ndcg_at_10"),
                })

            # Macro/per-relation analysis at the largest candidate pool.
            for source, targets in COMPATIBILITY.items():
                relation = f"{source}=>{'|'.join(sorted(targets))}"
                if direction == "ecg_to_cxr":
                    query_required, target_required = {source}, set(targets)
                else:
                    query_required, target_required = set(targets), {source}
                target_positions = np.asarray([
                    j for j, values in enumerate(item_assertions[target_modality])
                    if values["present"] & target_required], dtype="int64")
                relation_relevance = {}
                if len(target_positions):
                    for i, values in enumerate(item_assertions[query_modality]):
                        if values["present"] & query_required:
                            allowed = np.asarray([j for j in target_positions
                                                  if j not in set(exclusions.get(i, ()))], dtype="int64")
                            if len(allowed):
                                relation_relevance[i] = allowed
                relation_metrics = sampled_relevance(
                    query_item["sensor"], target_item["sensor"], relation_relevance,
                    max(CANDIDATE_COUNTS), seed=4201, excluded=exclusions)
                relation_results.append({**entry, "direction": direction, "relation": relation,
                                         "eligible_query_fraction": len(relation_relevance) / len(query_item["sensor"]),
                                         **relation_metrics})

    semantic_path = REPORT_ROOT / "external_cross_sensor_semantic.json"
    semantic_path.write_text(json.dumps(semantic_results, indent=2))
    semantic_frame = pd.DataFrame(semantic_results)
    semantic_summary_path = REPORT_ROOT / "external_cross_sensor_semantic_summary.csv"
    semantic_summary = semantic_frame.groupby(
        ["bridge_variant", "semantic_variant", "direction", "candidates"], as_index=False).agg(
            seeds=("adapter_seed", "size"), queries_mean=("queries", "mean"),
            eligible_query_fraction_mean=("eligible_query_fraction", "mean"),
            hit_at_1_mean=("hit_at_1", "mean"), hit_at_1_std=("hit_at_1", "std"),
            hit_at_10_mean=("hit_at_10", "mean"), precision_at_10_mean=("precision_at_10", "mean"),
            recall_at_10_mean=("recall_at_10", "mean"), map_at_10_mean=("map_at_10", "mean"),
            ndcg_at_10_mean=("ndcg_at_10", "mean"), auroc_mean=("auroc", "mean"),
            mrr_mean=("mrr", "mean"), unsupported_rate_at_10_mean=("unsupported_rate_at_10", "mean"),
            shuffled_hit_at_1_mean=("shuffled_hit_at_1", "mean"))
    semantic_summary.to_csv(semantic_summary_path, index=False)
    relation_path = REPORT_ROOT / "external_cross_sensor_semantic_by_relation.csv"
    pd.DataFrame(relation_results).to_csv(relation_path, index=False)
    display(semantic_summary.sort_values(["candidates", "direction", "ndcg_at_10_mean"],
                                         ascending=[True, True, False]).head(40))
    print("Saved:", semantic_path, semantic_summary_path, relation_path)
    """),
    md("""
    ## Shared-codebook utilization and collapse diagnostics

    Retrieval alone can hide codebook collapse. The following diagnostics report code utilization, effective perplexity, dominant-code share, complete code-sequence diversity, sensor-text code agreement, and the Jensen-Shannon divergence between external ECG and CXR sensor-code distributions. Continuous CLIP variants are retained with uses_vq=false and no code statistics.
    """),
    code("""
    def code_usage_metrics(codes, codebook_size):
        codes = np.asarray(codes, dtype="int64")
        perplexities = []
        for position in range(codes.shape[1]):
            count = np.bincount(codes[:, position], minlength=codebook_size).astype("float64")
            probability = count[count > 0] / count.sum()
            perplexities.append(float(np.exp(-(probability * np.log(probability)).sum())))
        total = np.bincount(codes.reshape(-1), minlength=codebook_size).astype("float64")
        probability = total / total.sum()
        unique_sequences = len({tuple(value) for value in codes.tolist()})
        return {
            "rows": len(codes), "chunks": codes.shape[1],
            "used_codes": int((total > 0).sum()),
            "code_utilization": float((total > 0).mean()),
            "dead_code_fraction": float((total == 0).mean()),
            "top_code_share": float(probability.max()),
            "per_chunk_perplexity_mean": float(np.mean(perplexities)),
            "per_chunk_perplexity_min": float(np.min(perplexities)),
            "unique_sequence_fraction": unique_sequences / len(codes),
        }

    def code_distribution_js(left, right, codebook_size):
        values = []
        for position in range(left.shape[1]):
            p = np.bincount(left[:, position], minlength=codebook_size).astype("float64")
            q = np.bincount(right[:, position], minlength=codebook_size).astype("float64")
            p, q = p / p.sum(), q / q.sum()
            middle = 0.5 * (p + q)
            left_term = np.where(p > 0, p * np.log((p + 1e-12) / (middle + 1e-12)), 0).sum()
            right_term = np.where(q > 0, q * np.log((q + 1e-12) / (middle + 1e-12)), 0).sum()
            values.append(0.5 * (left_term + right_term))
        return float(np.mean(values))

    codebook_results = []
    for entry in adapter_index:
        encoded = {m: encode_test(entry, m) for m in ["ecg", "cxr"]}
        if not encoded["ecg"]["uses_vq"]:
            codebook_results.append({**entry, "scope": "continuous_baseline", "uses_vq": False})
            continue
        size = encoded["ecg"]["codebook_size"]
        for modality in ["ecg", "cxr"]:
            item = encoded[modality]
            for representation in ["sensor", "text"]:
                code = item[f"{representation}_codes"]
                codebook_results.append({**entry, "scope": f"{modality}_{representation}",
                                         "uses_vq": True, **code_usage_metrics(code, size)})
            sensor_code, text_code = item["sensor_codes"], item["text_codes"]
            codebook_results.append({
                **entry, "scope": f"{modality}_paired_sensor_text", "uses_vq": True,
                "chunk_code_agreement": float((sensor_code == text_code).mean()),
                "full_sequence_agreement": float(np.all(sensor_code == text_code, axis=1).mean()),
            })
        codebook_results.append({
            **entry, "scope": "ecg_vs_cxr_sensor_distribution", "uses_vq": True,
            "mean_per_chunk_js_divergence": code_distribution_js(
                encoded["ecg"]["sensor_codes"], encoded["cxr"]["sensor_codes"], size),
        })

    codebook_path = REPORT_ROOT / "external_codebook_diagnostics.csv"
    pd.DataFrame(codebook_results).to_csv(codebook_path, index=False)
    display(pd.DataFrame(codebook_results).head(40))
    print("Saved:", codebook_path)
    """),
    md("""
    ## Interpretation rules

    - External ECG-CXR patient-identity validation is not evaluable because no verified shared patient/admission linkage is available.
    - Do not report empty or not-evaluable identity outputs as zero performance, failed retrieval, or external validation.
    - Symile same-patient retrieval remains the evidence-based patient-context endpoint and must not be called semantic or diagnostic agreement.
    - Use paired ECG-diagnosis and CXR-report retrieval to show that external adapters preserve modality-specific meaning.
    - Treat cross-patient concept compatibility, per-relation results, and unsupported@K as secondary semantic-transfer analyses.
    - Never use absent or uncertain concepts as semantic positives; audit any LLM-derived concepts before evaluation.
    - Prefer tie-broken MRR/nDCG to optimistic rank metrics for VQ models, and inspect utilization/perplexity before claiming codebook success.
    - Compare prespecified variants using mean and standard deviation across seeds. Do not select a method using test performance.
    """),
])

print("Generated external ablation notebooks 09-16.")
