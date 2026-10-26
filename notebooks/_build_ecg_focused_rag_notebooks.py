#!/usr/bin/env python3
"""Build notebooks 17-20 for the ECG-focused ECG-CXR bridge RAG study."""

from pathlib import Path
from textwrap import dedent
import json


ROOT = Path(__file__).resolve().parent


def markdown(value):
    return {
        "cell_type": "markdown", "metadata": {},
        "source": (dedent(value).strip() + "\n").splitlines(keepends=True),
    }


def code(value):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": (dedent(value).strip() + "\n").splitlines(keepends=True),
    }


def write(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = ROOT / name
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(path)


write("17_configure_rag_report_generation.ipynb", [
    markdown(r"""
    # 17 - Configure ECG-focused CXR-augmented report generation

    The primary question is whether CXR-derived context retrieved through a bridge learned from
    same-patient Symile observations improves **ECG machine-style diagnosis generation**.

    Retrieval method and context source are separated. This is necessary because a condition that mixes
    ECG and CXR neighbors cannot identify the specific contribution of CXR knowledge.

    | Context scope | Purpose |
    |---|---|
    | `zero_shot` | ECG waveform only |
    | `ecg_only_rag` | active ECG-to-ECG neighbor RAG control (ECG-ReGen-style, not an exact reproduction) |
    | `cxr_only_rag` | isolates CXR augmentation |
    | `ecg_plus_cxr_rag` | tests complementary ECG and CXR context |
    | `random_cxr_rag` | controls for extra medical text and prompt length |

    The ECG-only conditions are the critical same-modality retrieval baseline identified by the IEEE ECG-RAG
    literature: they retrieve external training ECG machine diagnoses for the query ECG. CLIP-style continuous,
    CodeBind shared-codebook, and reciprocal-rank-fused retrieval use the same ECG
    cohort and case budget. Retrieved records are always training records from other patients. Same-patient
    concurrence is bridge-training supervision; an external retrieved CXR is not assumed to belong to the
    ECG query patient or to share its diagnosis.
    """),
    code(r"""
    from pathlib import Path
    import importlib.util
    import json
    import os

    ARTIFACT_ROOT = Path(os.environ.get(
        "ECG_CXR_BRIDGE_ARTIFACT_ROOT",
        "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    BASE_CONFIG_PATH = ARTIFACT_ROOT / "config.json"
    if not BASE_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Run notebooks 00 and 09 first; missing {BASE_CONFIG_PATH}")
    base_config = json.loads(BASE_CONFIG_PATH.read_text())

    RAG_ROOT = ARTIFACT_ROOT / "rag_report_generation_ecg"
    for relative in ["contexts", "query_images/ecg", "generations", "reports", "logs"]:
        (RAG_ROOT / relative).mkdir(parents=True, exist_ok=True)

    MODELS = {
        "medgemma_4b": os.environ.get(
            "RAG_MEDGEMMA_MODEL", "google/medgemma-1.5-4b-it"),
        "qwen35_4b": os.environ.get("RAG_QWEN_MODEL", "Qwen/Qwen3.5-4B"),
    }
    RAG_SEED = int(os.environ.get("RAG_SEED", "5107"))
    MAX_TEST_ECG = int(os.environ.get("RAG_MAX_TEST_ECG", os.environ.get("RAG_MAX_TEST_PER_MODALITY", "250")))
    CONTEXT_CASES = int(os.environ.get("RAG_CONTEXT_CASES", "4"))
    BOTH_ECG_CASES = int(os.environ.get("RAG_BOTH_ECG_CASES", "2"))
    BOTH_CXR_CASES = int(os.environ.get("RAG_BOTH_CXR_CASES", "2"))
    MAX_CONTEXT_CHARACTERS = int(os.environ.get("RAG_MAX_CONTEXT_CHARACTERS", "6000"))
    MAX_NEW_TOKENS = int(os.environ.get("RAG_MAX_NEW_TOKENS", "384"))
    CODEBIND_VARIANT = os.environ.get("RAG_CODEBIND_VARIANT", "shared_specific_vq")
    CXR_CONTEXT_CONTENT = os.environ.get(
        "RAG_CXR_CONTEXT_CONTENT", "labels_and_report").strip().lower()
    if MAX_TEST_ECG < 1 or CONTEXT_CASES < 1 or MAX_CONTEXT_CHARACTERS < 1:
        raise ValueError("ECG cohort, context-case, and character budgets must be positive")
    if BOTH_ECG_CASES < 1 or BOTH_CXR_CASES < 1 or BOTH_ECG_CASES + BOTH_CXR_CASES != CONTEXT_CASES:
        raise ValueError("RAG_BOTH_ECG_CASES + RAG_BOTH_CXR_CASES must equal RAG_CONTEXT_CASES")
    if CXR_CONTEXT_CONTENT not in {"report", "labels", "labels_and_report"}:
        raise ValueError("RAG_CXR_CONTEXT_CONTENT must be report, labels, or labels_and_report")
    """),
    code(r"""
    condition_specs = [{
        "condition": "zero_shot", "retrieval_method": "none", "context_scope": "none",
        "ecg_cases": 0, "cxr_cases": 0,
    }]
    for method in ["clip", "codebind", "combined"]:
        condition_specs.extend([
            {"condition": f"{method}_ecg_only_rag", "retrieval_method": method,
             "context_scope": "ecg_only", "ecg_cases": CONTEXT_CASES, "cxr_cases": 0},
            {"condition": f"{method}_cxr_only_rag", "retrieval_method": method,
             "context_scope": "cxr_only", "ecg_cases": 0, "cxr_cases": CONTEXT_CASES},
            {"condition": f"{method}_ecg_plus_cxr_rag", "retrieval_method": method,
             "context_scope": "ecg_plus_cxr", "ecg_cases": BOTH_ECG_CASES,
             "cxr_cases": BOTH_CXR_CASES},
        ])
    condition_specs.append({
        "condition": "random_cxr_rag", "retrieval_method": "random",
        "context_scope": "cxr_only", "ecg_cases": 0, "cxr_cases": CONTEXT_CASES,
    })
    CONDITIONS = [row["condition"] for row in condition_specs]
    if len(CONDITIONS) != len(set(CONDITIONS)):
        raise AssertionError("Duplicate RAG condition names")
    print(json.dumps(condition_specs, indent=2))
    """),
    code(r"""
    experiment_path = ARTIFACT_ROOT / "ablation" / "experiment_index.json"
    experiments = json.loads(experiment_path.read_text())
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"No bridge experiments in {experiment_path}; run notebook 12")

    def select_by_validation(variant):
        candidates = [row for row in experiments if row.get("variant") == variant]
        candidates = [row for row in candidates if row.get("best_validation_retrieval_r1") is not None]
        if not candidates:
            raise KeyError(f"No validation-scored bridge variant {variant!r}")
        return max(candidates, key=lambda row: (
            float(row["best_validation_retrieval_r1"]), -int(row["seed"])))

    selected = {
        "clip": select_by_validation("clip_continuous"),
        "codebind": select_by_validation(CODEBIND_VARIANT),
    }
    if selected["codebind"]["variant"] in {"clip_continuous", "shuffled_pairs_control"}:
        raise ValueError("RAG_CODEBIND_VARIANT must be a non-control VQ bridge")
    print("Validation-selected bridge runs:", json.dumps(selected, indent=2))
    """),
    code(r"""
    required_files = []
    for modality in ["ecg", "cxr"]:
        required_files.extend([
            ARTIFACT_ROOT / "external" / modality / "manifest.jsonl",
            ARTIFACT_ROOT / "external" / modality / "concepts.jsonl",
            ARTIFACT_ROOT / "external" / modality / "encoding_status.npy",
        ])
        if modality == "cxr":
            required_files.append(ARTIFACT_ROOT / "external" / modality / "manifest_summary.json")
        for experiment in selected.values():
            required_files.append(
                ARTIFACT_ROOT / "external" / modality / "bridge" /
                experiment["variant"] / f"seed_{experiment['seed']}" / "common.npy")
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "ECG-focused RAG prerequisites are missing. Run notebooks 10-14:\n" +
            "\n".join(missing[:40]))

    cxr_summary = json.loads(
        (ARTIFACT_ROOT / "external" / "cxr" / "manifest_summary.json").read_text())
    expected_count = int(os.environ.get(
        "MIMIC_CXR_EXPECTED_TRAIN_IMAGES",
        base_config.get("external_cxr_expected_train_images", 43700)))
    expected_roots = {
        "provided_train": str(Path(os.environ.get(
            "MIMIC_CXR_TRAIN_ROOT", base_config.get("external_cxr_root", "")))),
        "provided_test": str(Path(os.environ.get(
            "MIMIC_CXR_TEST_ROOT", base_config.get("external_cxr_test_root", "")))),
    }
    observed_count = int(cxr_summary.get("discovery", {}).get("provided_train", {}).get("images", -1))
    observed_roots = cxr_summary.get("roots", {})
    if observed_count != expected_count or observed_roots != expected_roots:
        raise RuntimeError(
            "Stale external CXR artifacts. "
            f"Expected {expected_count:,} train images and {expected_roots}; "
            f"found {observed_count:,} and {observed_roots}. Rerun notebooks 09, 11, 13, and 14.")

    rag_config = {
        "artifact_root": str(ARTIFACT_ROOT), "rag_root": str(RAG_ROOT),
        "primary_modality": "ecg", "models": MODELS,
        "conditions": CONDITIONS, "condition_specs": condition_specs,
        "seed": RAG_SEED, "max_test_ecg": MAX_TEST_ECG,
        "context_cases": CONTEXT_CASES, "both_ecg_cases": BOTH_ECG_CASES,
        "both_cxr_cases": BOTH_CXR_CASES,
        "max_context_characters": MAX_CONTEXT_CHARACTERS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "cxr_context_content": CXR_CONTEXT_CONTENT,
        "retrieval_selection": selected,
        "hf_cache": base_config.get("hf_cache"),
        "generation_protocol_version": "ecg_cxr_scope_ablation_v1",
        "primary_comparisons": [
            "ECG-to-ECG neighbor RAG versus zero-shot",
            "retrieved CXR versus zero-shot", "ECG+CXR versus ECG-only",
            "CodeBind CXR versus CLIP CXR", "retrieved CXR versus random CXR",
        ],
        "evaluation_text_metrics": [
            "BLEU-1", "BLEU-4", "METEOR", "ROUGE-1/2/L", "BERTScore-F1", "CIDEr",
        ],
        "ecg_neighbor_baseline_note": (
            "ECG-only RAG is an ECG-ReGen-style same-modality control, not an exact reproduction; "
            "it uses this study's validation-selected bridge embeddings and external ECG machine diagnoses."),
        "leakage_controls": [
            "queries are external ECG split=test",
            "all retrieval candidates have split=train",
            "query sample IDs are forbidden",
            "same subject IDs are excluded when available",
            "query machine diagnosis and labels are never included in prompts",
            "bridge selection uses validation retrieval only",
        ],
    }
    config_path = RAG_ROOT / "rag_config.json"
    config_path.write_text(json.dumps(rag_config, indent=2))
    print("Saved:", config_path)
    """),
    code(r"""
    required_packages = [
        "numpy", "pandas", "torch", "transformers", "accelerate", "PIL", "wfdb",
        "matplotlib", "sklearn", "nltk", "rouge_score", "bert_score", "pycocoevalcap",
    ]
    missing_packages = [name for name in required_packages if importlib.util.find_spec(name) is None]
    print({"missing_packages": missing_packages})
    if missing_packages:
        print("Install missing packages in the Biowulf environment:")
        print("pip install -U 'transformers>=5.14.1' accelerate nltk rouge-score bert-score pycocoevalcap scikit-learn")
    try:
        from importlib.metadata import version as package_version
        print("Transformers:", package_version("transformers"),
              "(the working MedGemma 1.5 reference notebook used 5.14.1)")
    except Exception as exc:
        print("Could not read the Transformers version:", repr(exc))
    print("MedGemma 1.5 requires accepted Hugging Face terms and authentication on Biowulf.")
    """),
])


write("18_build_rag_retrieval_contexts.ipynb", [
    markdown(r"""
    # 18 - Build ECG-only, CXR-only, combined, and random-control contexts

    One fixed, label-coverage-aware external ECG test cohort is used for every condition. CLIP and
    CodeBind each retrieve ECG and CXR training records. The same-modality ECG list is the
    ECG-to-ECG neighbor RAG baseline motivated by ECG-ReGen; it is not an exact reproduction because
    the bridge embeddings and machine diagnoses used here differ. Combined retrieval applies reciprocal-rank
    fusion separately within each source modality. A deterministic random-CXR control uses the same
    CXR case budget.

    Same-modality and cross-modal lists are stored separately before condition construction. This makes
    the CXR augmentation effect identifiable and keeps `ecg_plus_cxr_rag` at the same total case budget
    as the single-source conditions.
    """),
    code(r"""
    from pathlib import Path
    import hashlib
    import json
    import os

    import numpy as np
    import torch
    from tqdm.auto import tqdm

    ARTIFACT_ROOT = Path(os.environ.get(
        "ECG_CXR_BRIDGE_ARTIFACT_ROOT",
        "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    RAG_ROOT = ARTIFACT_ROOT / "rag_report_generation_ecg"
    config = json.loads((RAG_ROOT / "rag_config.json").read_text())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    QUERY_BATCH = int(os.environ.get("RAG_RETRIEVAL_QUERY_BATCH", "32"))
    CANDIDATE_CHUNK = int(os.environ.get("RAG_RETRIEVAL_CANDIDATE_CHUNK", "50000"))
    if QUERY_BATCH < 1 or CANDIDATE_CHUNK < 1:
        raise ValueError("Retrieval batch and chunk sizes must be positive")

    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def file_sha256(path, chunk=1024 * 1024):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                value = stream.read(chunk)
                if not value:
                    break
                digest.update(value)
        return digest.hexdigest()

    manifest_paths = {m: ARTIFACT_ROOT / "external" / m / "manifest.jsonl" for m in ["ecg", "cxr"]}
    manifests = {m: read_jsonl(path) for m, path in manifest_paths.items()}
    concepts = {m: read_jsonl(ARTIFACT_ROOT / "external" / m / "concepts.jsonl") for m in ["ecg", "cxr"]}
    for modality in manifests:
        if len(manifests[modality]) != len(concepts[modality]):
            raise ValueError(f"{modality} manifest/concept length mismatch; rerun notebook 14")
        for index, (row, concept) in enumerate(zip(manifests[modality], concepts[modality])):
            if str(row["sample_id"]) != str(concept.get("sample_id")):
                raise ValueError(f"{modality} concept order mismatch at {index}")
    manifest_hashes = {m: file_sha256(path) for m, path in manifest_paths.items()}
    print({m: {s: sum(row.get("split") == s for row in values) for s in ["train", "val", "test"]}
           for m, values in manifests.items()})
    """),
    code(r"""
    selections = config["retrieval_selection"]
    embeddings, status = {}, {}
    for method, experiment in selections.items():
        embeddings[method] = {}
        for modality in ["ecg", "cxr"]:
            path = (ARTIFACT_ROOT / "external" / modality / "bridge" /
                    experiment["variant"] / f"seed_{experiment['seed']}" / "common.npy")
            array = np.load(path, mmap_mode="r")
            if len(array) != len(manifests[modality]):
                raise ValueError(f"{method}/{modality} embeddings do not match the manifest")
            embeddings[method][modality] = array
    for modality in ["ecg", "cxr"]:
        status[modality] = np.load(
            ARTIFACT_ROOT / "external" / modality / "encoding_status.npy", mmap_mode="r")
        if len(status[modality]) != len(manifests[modality]):
            raise ValueError(f"{modality} status does not match the manifest")

    def finite_row_mask(array, chunk_size=50000):
        mask = np.zeros(len(array), dtype=bool)
        for start in tqdm(range(0, len(array), chunk_size), desc="validate embeddings", leave=False):
            stop = min(start + chunk_size, len(array))
            mask[start:stop] = np.isfinite(np.asarray(array[start:stop])).all(axis=1)
        return mask

    split_masks = {
        modality: {split: np.asarray([row.get("split") == split for row in rows], dtype=bool)
                   for split in ["train", "test"]}
        for modality, rows in manifests.items()
    }
    valid_masks = {
        method: {modality: (np.asarray(status[modality]) == 1) & finite_row_mask(array)
                 for modality, array in values.items()}
        for method, values in embeddings.items()
    }
    joint_valid = {
        modality: np.logical_and.reduce([valid_masks[method][modality] for method in embeddings])
        for modality in ["ecg", "cxr"]}
    """),
    code(r"""
    def stable_digest(value):
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def ecg_positive_labels(index):
        return {str(value).split(":", 1)[1] for value in concepts["ecg"][index].get("concepts", [])
                if str(value).startswith("present:")}

    rng = np.random.default_rng(int(config["seed"]))
    eligible = np.flatnonzero(split_masks["ecg"]["test"] & joint_valid["ecg"]).astype("int64")
    if len(eligible) < 2:
        raise ValueError(f"Need at least two valid external ECG test queries; found {len(eligible)}")
    take = min(int(config["max_test_ecg"]), len(eligible))
    shuffled = eligible[rng.permutation(len(eligible))]
    label_to_indices = {}
    for index in shuffled:
        for label in ecg_positive_labels(int(index)):
            label_to_indices.setdefault(label, []).append(int(index))
    selected_indices, selected_set = [], set()
    for label in sorted(label_to_indices, key=lambda value: (len(label_to_indices[value]), value)):
        candidate = next((index for index in label_to_indices[label] if index not in selected_set), None)
        if candidate is not None and len(selected_indices) < take:
            selected_indices.append(candidate); selected_set.add(candidate)
    for index in shuffled:
        if len(selected_indices) >= take:
            break
        if int(index) not in selected_set:
            selected_indices.append(int(index)); selected_set.add(int(index))
    cohort = np.asarray(sorted(selected_indices), dtype="int64")
    selected_labels = sorted(set().union(*(ecg_positive_labels(int(i)) for i in cohort)))
    cohort_signature = stable_digest({
        "manifest_hashes": manifest_hashes,
        "seed": config["seed"], "selected_ids": [manifests["ecg"][int(i)]["sample_id"] for i in cohort],
        "condition_specs": config["condition_specs"],
        "context_cases": config["context_cases"],
        "retrieval_selection": {key: {"variant": value["variant"], "seed": value["seed"]}
                                for key, value in selections.items()},
        "cxr_context_content": config["cxr_context_content"],
    })
    coverage = {
        "available_positive_labels": sorted(label_to_indices),
        "selected_positive_labels": selected_labels,
        "missing_positive_labels": sorted(set(label_to_indices) - set(selected_labels)),
    }
    print({"queries": len(cohort), "signature": cohort_signature[:12], "label_coverage": coverage})
    """),
    code(r"""
    def numeric_subject(row):
        try:
            return int(row.get("subject_id")) if row.get("subject_id") is not None else -1
        except (TypeError, ValueError):
            return -1

    @torch.inference_mode()
    def chunked_topk(query_values, candidate_values, query_rows, candidate_rows, k):
        k = min(int(k), len(candidate_values))
        if k < 1:
            return [[] for _ in query_rows]
        candidate_subjects = np.asarray([numeric_subject(row) for row in candidate_rows], dtype="int64")
        outputs = []
        for q_start in tqdm(range(0, len(query_values), QUERY_BATCH), desc="query batches", leave=False):
            q_stop = min(q_start + QUERY_BATCH, len(query_values))
            q = torch.from_numpy(np.asarray(query_values[q_start:q_stop], dtype="float32")).to(DEVICE)
            q = torch.nn.functional.normalize(q, dim=-1)
            q_subjects = np.asarray([numeric_subject(row) for row in query_rows[q_start:q_stop]], dtype="int64")
            best_scores = torch.full((len(q), k), -torch.inf, device=DEVICE)
            best_indices = torch.full((len(q), k), -1, dtype=torch.long, device=DEVICE)
            for c_start in range(0, len(candidate_values), CANDIDATE_CHUNK):
                c_stop = min(c_start + CANDIDATE_CHUNK, len(candidate_values))
                c = torch.from_numpy(np.asarray(candidate_values[c_start:c_stop], dtype="float32")).to(DEVICE)
                c = torch.nn.functional.normalize(c, dim=-1)
                scores = q @ c.T
                c_subjects = candidate_subjects[c_start:c_stop]
                same_subject = ((q_subjects[:, None] >= 0) &
                                (q_subjects[:, None] == c_subjects[None, :]))
                if same_subject.any():
                    scores.masked_fill_(torch.from_numpy(same_subject).to(DEVICE), -torch.inf)
                local_k = min(k, scores.shape[1])
                local_scores, local_positions = scores.topk(local_k, dim=1)
                merged_scores = torch.cat([best_scores, local_scores], dim=1)
                merged_indices = torch.cat([best_indices, local_positions + c_start], dim=1)
                best_scores, order = merged_scores.topk(k, dim=1)
                best_indices = torch.gather(merged_indices, 1, order)
            for row_scores, row_indices in zip(best_scores.cpu().numpy(), best_indices.cpu().numpy()):
                if (row_indices < 0).any() or not np.isfinite(row_scores).all():
                    raise RuntimeError("Insufficient candidates after patient leakage masking")
                outputs.append([(int(i), float(score)) for i, score in zip(row_indices, row_scores)])
        return outputs

    def retrieval_text(modality, row):
        key = "machine_diagnosis" if modality == "ecg" else "report_text"
        return " ".join(str(row.get(key, "")).split())

    def normalized_labels(row):
        return sorted({str(value).strip() for value in row.get("labels", []) if str(value).strip()})

    def item(method, scope, rank, score, modality, row):
        return {
            "retrieval_method": method, "scope": scope, "rank": int(rank), "score": float(score),
            "modality": modality, "sample_id": str(row["sample_id"]),
            "subject_id": row.get("subject_id"), "text": retrieval_text(modality, row),
            "labels": normalized_labels(row) if modality == "cxr" else [],
        }
    """),
    code(r"""
    query_rows = [manifests["ecg"][int(i)] for i in cohort]
    contexts_by_method = {method: {} for method in ["clip", "codebind"]}
    diagnostics = {"cohort_signature": cohort_signature, "device": str(DEVICE), "retrievals": {}}
    for method in ["clip", "codebind"]:
        method_records = {str(row["sample_id"]): {"same_modality": [], "cross_modal": []}
                          for row in query_rows}
        for scope, candidate_modality in [("same_modality", "ecg"), ("cross_modal", "cxr")]:
            # Use the same finite training candidate pool for every retriever so a
            # CLIP/CodeBind difference cannot be caused by different eligibility.
            candidate_indices = np.flatnonzero(
                split_masks[candidate_modality]["train"] & joint_valid[candidate_modality]
            ).astype("int64")
            candidate_rows = [manifests[candidate_modality][int(i)] for i in candidate_indices]
            found = chunked_topk(
                embeddings[method]["ecg"][cohort],
                embeddings[method][candidate_modality][candidate_indices],
                query_rows, candidate_rows, int(config["context_cases"]))
            for query_row, values in zip(query_rows, found):
                key = str(query_row["sample_id"])
                for rank, (position, score) in enumerate(values, start=1):
                    source = candidate_rows[position]
                    if source.get("split") != "train":
                        raise AssertionError("Non-training retrieval candidate")
                    if str(source["sample_id"]) == key:
                        raise AssertionError("Query sample leaked into retrieval context")
                    q_subject, c_subject = numeric_subject(query_row), numeric_subject(source)
                    if q_subject >= 0 and q_subject == c_subject:
                        raise AssertionError("Query subject leaked into retrieval context")
                    method_records[key][scope].append(
                        item(method, scope, rank, score, candidate_modality, source))
            diagnostics["retrievals"][f"{method}:{scope}"] = {
                "queries": len(query_rows), "candidate_modality": candidate_modality,
                "training_candidates": len(candidate_indices), "top_k": int(config["context_cases"]),
            }
        contexts_by_method[method] = method_records
        print(method, {scope: len(contexts_by_method[method][str(query_rows[0]['sample_id'])][scope])
                       for scope in ["same_modality", "cross_modal"]})
    """),
    code(r"""
    def reciprocal_rank_fusion(left, right, method="combined", constant=60):
        fused = {}
        for value in left + right:
            key = (value["modality"], value["sample_id"])
            entry = fused.setdefault(key, {**value, "score": 0.0, "retrieval_method": method})
            entry["score"] += 1.0 / (constant + int(value["rank"]))
        ordered = sorted(fused.values(), key=lambda row: (-row["score"], row["sample_id"]))
        for rank, value in enumerate(ordered, start=1):
            value["rank"] = rank
        return ordered

    combined_contexts = {}
    for query_row in query_rows:
        key = str(query_row["sample_id"])
        combined_contexts[key] = {}
        for scope in ["same_modality", "cross_modal"]:
            combined_contexts[key][scope] = reciprocal_rank_fusion(
                contexts_by_method["clip"][key][scope],
                contexts_by_method["codebind"][key][scope])[:int(config["context_cases"])]

    random_candidate_indices = np.flatnonzero(
        split_masks["cxr"]["train"] & joint_valid["cxr"]).astype("int64")
    random_candidate_rows = [manifests["cxr"][int(i)] for i in random_candidate_indices]
    random_contexts = {}
    for query_row in query_rows:
        key = str(query_row["sample_id"])
        q_subject = numeric_subject(query_row)
        eligible_positions = np.asarray([
            position for position, row in enumerate(random_candidate_rows)
            if q_subject < 0 or numeric_subject(row) != q_subject], dtype="int64")
        if len(eligible_positions) < int(config["context_cases"]):
            raise RuntimeError("Insufficient random CXR candidates after patient exclusion")
        local_seed = int(hashlib.sha1(
            f"{config['seed']}:{key}:random_cxr".encode()).hexdigest()[:16], 16)
        local_rng = np.random.default_rng(local_seed)
        positions = local_rng.choice(
            eligible_positions, size=int(config["context_cases"]), replace=False)
        random_contexts[key] = [
            item("random", "cross_modal", rank, 0.0, "cxr", random_candidate_rows[int(position)])
            for rank, position in enumerate(positions, start=1)]
    """),
    code(r"""
    spec_lookup = {row["condition"]: row for row in config["condition_specs"]}

    def selected_context(spec, sample_id):
        method, scope = spec["retrieval_method"], spec["context_scope"]
        if method == "none":
            return []
        if method == "random":
            values = random_contexts[sample_id][:int(spec["cxr_cases"])]
        else:
            source = combined_contexts[sample_id] if method == "combined" else contexts_by_method[method][sample_id]
            values = []
            if scope in {"ecg_only", "ecg_plus_cxr"}:
                values.extend(source["same_modality"][:int(spec["ecg_cases"])])
            if scope in {"cxr_only", "ecg_plus_cxr"}:
                values.extend(source["cross_modal"][:int(spec["cxr_cases"])])
        expected = int(spec["ecg_cases"]) + int(spec["cxr_cases"])
        if len(values) != expected:
            raise AssertionError(f"Context budget mismatch for {spec['condition']}: {len(values)} != {expected}")
        return values

    context_rows = []
    for query_index, query_row in zip(cohort, query_rows):
        sample_id = str(query_row["sample_id"])
        common = {
            "sample_id": sample_id, "modality": "ecg", "query_index": int(query_index),
            "query_path": str(query_row["waveform_path"]), "split": str(query_row["split"]),
            "cohort_signature": cohort_signature,
        }
        for condition in config["conditions"]:
            spec = spec_lookup[condition]
            context_rows.append({
                **common, **spec, "retrieved": selected_context(spec, sample_id),
                "cxr_context_content": config["cxr_context_content"],
            })

    output = RAG_ROOT / "contexts" / "rag_contexts.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    with temporary.open("w") as stream:
        for row in context_rows:
            stream.write(json.dumps(row) + "\n")
    temporary.replace(output)
    diagnostics.update({
        "rows": len(context_rows), "conditions": config["conditions"],
        "queries": len(cohort), "condition_case_budgets": {
            row["condition"]: row["ecg_cases"] + row["cxr_cases"] for row in config["condition_specs"]},
        "cohort_label_coverage": coverage, "leakage_violations": 0,
        "random_cxr_training_candidates": len(random_candidate_rows),
    })
    (RAG_ROOT / "contexts" / "retrieval_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2))
    print("Saved:", output, diagnostics)
    """),
])


write("19_generate_rag_reports.ipynb", [
    markdown(r"""
    # 19 - Generate ECG reports with scope-separated RAG

    MedGemma and Qwen3.5 receive the same rendered ECG query under every condition. Retrieved ECG
    diagnoses form the ECG-to-ECG neighbor baseline; retrieved CXR reports/labels test cross-sensor
    augmentation. Both are non-authoritative examples from other training patients.
    The prompt explicitly forbids transferring a CXR finding into the ECG interpretation without
    waveform support.

    Generation is deterministic and resumable. Existing rows are reused only when sample, condition,
    cohort signature, and generation protocol agree. Use `RAG_MODELS`, `RAG_CONDITIONS`,
    `RAG_RETRY_ERRORS`, and `RAG_GENERATION_MAX_ROWS` for planned partial or smoke runs.
    """),
    code(r"""
    from pathlib import Path
    import gc
    import hashlib
    import json
    import os
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from importlib.metadata import version as package_version

    ARTIFACT_ROOT = Path(os.environ.get(
        "ECG_CXR_BRIDGE_ARTIFACT_ROOT",
        "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    RAG_ROOT = ARTIFACT_ROOT / "rag_report_generation_ecg"
    config = json.loads((RAG_ROOT / "rag_config.json").read_text())
    HF_CACHE = Path(config.get("hf_cache") or "/data/liangz2/ecg_analysis/hf_cache")
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    # Set these before importing Transformers/Hugging Face. In particular, the
    # Xet chunk cache otherwise defaults to ~/.cache and can exhaust Biowulf's
    # small home quota even when from_pretrained receives cache_dir.
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["HF_XET_CACHE"] = str(HF_CACHE / "xet")
    for directory in [HF_CACHE / "hub", HF_CACHE / "xet"]:
        directory.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "This notebook requires a Transformers release with "
            "AutoModelForImageTextToText. In the active Biowulf environment run "
            "python -m pip install -U 'transformers>=5.14.1' accelerate, then "
            "restart the notebook kernel.") from exc
    # Migrate configurations written by the earlier notebook version. The
    # project reference notebook was validated with MedGemma 1.5 and
    # Transformers 5.14.1.
    LEGACY_MEDGEMMA_IDS = {"google/medgemma-4b-it"}
    DEFAULT_MEDGEMMA_ID = "google/medgemma-1.5-4b-it"
    configured_medgemma = config.get("models", {}).get("medgemma_4b")
    if configured_medgemma in LEGACY_MEDGEMMA_IDS:
        print(f"Migrating MedGemma model ID: {configured_medgemma} -> {DEFAULT_MEDGEMMA_ID}")
        config["models"]["medgemma_4b"] = DEFAULT_MEDGEMMA_ID
        (RAG_ROOT / "rag_config.json").write_text(json.dumps(config, indent=2))
    print("Transformers:", package_version("transformers"),
          "(working reference environment: 5.14.1)")
    print({"hf_home": os.environ["HF_HOME"], "hf_hub_cache": os.environ["HF_HUB_CACHE"],
           "hf_xet_cache": os.environ["HF_XET_CACHE"]})
    contexts_path = RAG_ROOT / "contexts" / "rag_contexts.jsonl"
    contexts = [json.loads(line) for line in contexts_path.read_text().splitlines() if line.strip()]
    if not contexts:
        raise ValueError(f"No contexts in {contexts_path}; run notebook 18")
    if {row.get("modality") for row in contexts} != {"ecg"}:
        raise ValueError("ECG-focused generation received a non-ECG query")

    selected_models = {value.strip() for value in os.environ.get(
        "RAG_MODELS", ",".join(config["models"])).split(",") if value.strip()}
    selected_conditions = {value.strip() for value in os.environ.get(
        "RAG_CONDITIONS", ",".join(config["conditions"])).split(",") if value.strip()}
    unknown_models = selected_models - set(config["models"])
    unknown_conditions = selected_conditions - set(config["conditions"])
    if unknown_models or unknown_conditions:
        raise ValueError(f"Unknown models={unknown_models} or conditions={unknown_conditions}")
    RETRY_ERRORS = os.environ.get("RAG_RETRY_ERRORS", "0").lower() in {"1", "true", "yes"}
    MAX_ROWS = int(os.environ.get("RAG_GENERATION_MAX_ROWS", "0"))
    DEFAULT_BATCH_SIZE = int(os.environ.get("RAG_GENERATION_BATCH_SIZE", "4"))
    MODEL_BATCH_SIZES = {
        "medgemma_4b": int(os.environ.get("RAG_MEDGEMMA_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
        "qwen35_4b": int(os.environ.get("RAG_QWEN_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
    }
    if MAX_ROWS < 0:
        raise ValueError("RAG_GENERATION_MAX_ROWS cannot be negative")
    if any(value < 1 for value in MODEL_BATCH_SIZES.values()):
        raise ValueError("All RAG generation batch sizes must be positive")
    print({"models": sorted(selected_models), "conditions": sorted(selected_conditions),
           "context_rows": len(contexts), "retry_errors": RETRY_ERRORS, "max_rows": MAX_ROWS,
           "batch_sizes": MODEL_BATCH_SIZES})
    """),
    code(r"""
    ECG_IMAGE_ROOT = RAG_ROOT / "query_images" / "ecg"
    ECG_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    _WFDB_MODULE = None

    def load_wfdb():
        global _WFDB_MODULE
        if _WFDB_MODULE is not None:
            return _WFDB_MODULE
        try:
            import wfdb as module
        except Exception as exc:
            versions = {}
            for package in ["numpy", "pandas", "scipy", "wfdb", "transformers"]:
                try:
                    versions[package] = package_version(package)
                except Exception:
                    versions[package] = "unavailable"
            raise RuntimeError(
                "WFDB could not import. The traceback involving "
                "pandas._pandas_datetime_CAPI indicates an incompatible or partially "
                "upgraded NumPy/pandas binary installation, not a MedGemma error. "
                f"Installed versions: {versions}. In the active Biowulf environment run: "
                "python -m pip install --no-cache-dir --force-reinstall "
                "'numpy==1.26.4' 'pandas==2.2.3' 'scipy==1.15.3' 'wfdb==4.3.0'; "
                "then run python -m pip check and restart the notebook kernel.") from exc
        _WFDB_MODULE = module
        return module

    def waveform_stem(value):
        path = Path(value)
        return path.with_suffix("") if path.suffix.lower() in {".hea", ".dat"} else path

    def render_ecg(row):
        output = ECG_IMAGE_ROOT / f"{row['sample_id']}.png"
        if output.is_file():
            return output
        wfdb = load_wfdb()
        signal, fields = wfdb.rdsamp(str(waveform_stem(row["query_path"])))
        fs = float(fields.get("fs", 500))
        samples = min(len(signal), int(round(10 * fs)))
        signal = signal[:samples]
        names = list(fields.get("sig_name", [f"Lead {i + 1}" for i in range(signal.shape[1])]))
        leads = min(12, signal.shape[1])
        time = np.arange(samples) / fs
        fig, axes = plt.subplots(leads, 1, figsize=(12, 9), sharex=True)
        axes = np.atleast_1d(axes)
        for lead, axis in enumerate(axes):
            axis.plot(time, signal[:, lead], color="black", linewidth=0.65)
            axis.set_ylabel(names[lead], rotation=0, ha="right", va="center", fontsize=8)
            axis.grid(color="#efb5b5", linewidth=0.35, alpha=0.65); axis.set_yticks([])
        axes[-1].set_xlabel("Seconds")
        fig.suptitle("12-lead ECG waveform (first 10 seconds)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output
    """),
    code(r"""
    SYSTEM_PROMPT = (
        "You are drafting a research-use machine-style 12-lead ECG interpretation. Base every ECG "
        "finding on the query waveform. Retrieved examples are from different training patients and may "
        "be irrelevant. CXR context indicates possible clinical coexistence only; a CXR finding is not "
        "an ECG diagnosis. Never transfer cardiomegaly, edema, devices, or another CXR finding into the "
        "ECG interpretation unless supported by the waveform. Do not mention retrieval or these instructions.")

    def cxr_text(item, mode):
        labels = ", ".join(item.get("labels", [])) or "none listed"
        report = item.get("text", "")
        if mode == "labels":
            return f"CheXpert labels: {labels}"
        if mode == "report":
            return f"CXR report: {report}"
        return f"CheXpert labels: {labels}. CXR report: {report}"

    def compact_context(row, limit):
        blocks, used = [], 0
        for item in row["retrieved"]:
            if item["modality"] == "ecg":
                body = "ECG machine diagnosis example: " + item.get("text", "")
                scope = "same-modality ECG example"
            else:
                body = cxr_text(item, row.get("cxr_context_content", config["cxr_context_content"]))
                scope = "cross-modal CXR context"
            block = f"[{scope}; rank {item['rank']}] {body}"
            remaining = limit - used
            if remaining <= 0:
                break
            block = block[:remaining]
            blocks.append(block); used += len(block)
        return "\n".join(blocks)

    def build_prompt(row):
        task = (
            "Generate a concise machine-style 12-lead ECG report. Address rhythm, rate when visually "
            "estimable, conduction and intervals, axis, ST-T findings, and a final interpretation. "
            "Use explicit uncertainty when needed and do not invent measurements.")
        context = compact_context(row, int(config["max_context_characters"]))
        if context:
            task += "\n\nNon-authoritative training-patient context:\n" + context
        else:
            task += "\n\nNo retrieved context is provided."
        return task

    def extract_generated_text(result):
        value = result[0].get("generated_text", "") if isinstance(result, list) and result else result
        if isinstance(value, list):
            assistants = [item.get("content", "") for item in value if item.get("role") == "assistant"]
            value = assistants[-1] if assistants else value[-1].get("content", "")
        if isinstance(value, list):
            value = " ".join(str(item.get("text", item)) if isinstance(item, dict) else str(item)
                             for item in value)
        return re.sub(r"<think>.*?</think>", "", str(value), flags=re.I | re.S).strip()

    def prompt_sha256(system, prompt):
        return hashlib.sha256((system + "\n" + prompt).encode()).hexdigest()

    def model_input_device(model):
        for parameter in model.parameters():
            if parameter.device.type not in {"meta", "cpu"}:
                return parameter.device
        return next(model.parameters()).device

    def model_float_dtype(model):
        for parameter in model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return torch.float32

    def load_image_text_model(model_id):
        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=config.get("hf_cache"))
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, device_map="auto", low_cpu_mem_usage=True,
            cache_dir=config.get("hf_cache"))
        model.eval().requires_grad_(False)
        processor.tokenizer.padding_side = "left"
        return model, processor

    @torch.inference_mode()
    def generate_reports(model, processor, model_key, image_paths, prompts):
        if len(image_paths) != len(prompts) or not prompts:
            raise ValueError("Image and prompt batches must have the same positive length")
        template_kwargs = {"enable_thinking": False} if model_key == "qwen35_4b" else {}
        prompt_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + prompt},
            ]}]
            prompt_texts.append(processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **template_kwargs))
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image_file:
                images.append(image_file.convert("RGB"))
        inputs = processor(
            text=prompt_texts, images=images, return_tensors="pt", padding=True)
        device = model_input_device(model)
        dtype = model_float_dtype(model)
        moved = {
            key: (value.to(device=device, dtype=dtype) if value.is_floating_point()
                  else value.to(device=device))
            for key, value in inputs.items()
        }
        input_length = moved["input_ids"].shape[-1]
        output_ids = model.generate(
            **moved, do_sample=False, max_new_tokens=int(config["max_new_tokens"]),
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id)
        return [value.strip() for value in processor.batch_decode(
            output_ids[:, input_length:], skip_special_tokens=True)]

    def clear_cuda_after_error():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_with_fallback(model, processor, model_key, image_paths, prompts):
        try:
            return [(text, None) for text in generate_reports(
                model, processor, model_key, image_paths, prompts)]
        except Exception as batch_exc:
            if len(prompts) == 1:
                return [("", batch_exc)]
            print(f"Batch of {len(prompts)} failed; retrying each item separately: {batch_exc!r}")
            clear_cuda_after_error()
            results = []
            for image_path, prompt in zip(image_paths, prompts):
                try:
                    text = generate_reports(
                        model, processor, model_key, [image_path], [prompt])[0]
                    results.append((text, None))
                except Exception as item_exc:
                    clear_cuda_after_error()
                    results.append(("", item_exc))
            return results
    """),
    code(r"""
    generation_root = RAG_ROOT / "generations"
    generation_root.mkdir(parents=True, exist_ok=True)
    eligible = [row for row in contexts if row["condition"] in selected_conditions]
    if MAX_ROWS:
        eligible = eligible[:MAX_ROWS]

    for model_key in sorted(selected_models):
        model_id = config["models"][model_key]
        output_path = generation_root / f"{model_key}.jsonl"
        existing = []
        if output_path.is_file():
            existing = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
        completed = {
            (row.get("model_id"), row["condition"], str(row["sample_id"]), row.get("cohort_signature"),
             row.get("generation_protocol_version")): row
            for row in existing if row.get("status") == "ok" or not RETRY_ERRORS
        }
        pending = [row for row in eligible if (
            model_id, row["condition"], str(row["sample_id"]), row.get("cohort_signature"),
            config["generation_protocol_version"]) not in completed]
        print(model_key, {"existing": len(existing), "pending": len(pending), "model_id": model_id})
        if not pending:
            continue
        try:
            model, processor = load_image_text_model(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {model_id}. The working reference notebook used "
                f"Transformers 5.14.1 with AutoProcessor and AutoModelForImageTextToText. "
                f"In the active Biowulf environment run: python -m pip install -U "
                f"'transformers>=5.14.1' accelerate; restart the kernel; accept the model terms "
                f"and authenticate with Hugging Face. "
                f"Original error: {exc!r}") from exc

        batch_size = MODEL_BATCH_SIZES[model_key]
        with output_path.open("a") as stream, tqdm(total=len(pending), desc=model_key) as progress:
            for start in range(0, len(pending), batch_size):
                batch_rows = pending[start:start + batch_size]
                prepared = []
                for row in batch_rows:
                    prompt = build_prompt(row)
                    record = {
                        "model_key": model_key, "model_id": model_id,
                        "sample_id": str(row["sample_id"]), "modality": "ecg",
                        "condition": row["condition"],
                        "retrieval_method": row["retrieval_method"],
                        "context_scope": row["context_scope"],
                        "cohort_signature": row["cohort_signature"],
                        "generation_protocol_version": config["generation_protocol_version"],
                        "prompt_sha256": prompt_sha256(SYSTEM_PROMPT, prompt),
                        "retrieved_sample_ids": [item["sample_id"] for item in row["retrieved"]],
                        "retrieved_modalities": [item["modality"] for item in row["retrieved"]],
                    }
                    try:
                        prepared.append((record, render_ecg(row), prompt))
                    except Exception as exc:
                        record.update({"status": "error", "generated_report": "", "error": repr(exc)})
                        stream.write(json.dumps(record) + "\n"); stream.flush(); progress.update(1)
                if not prepared:
                    continue
                generated = generate_with_fallback(
                    model, processor, model_key,
                    [value[1] for value in prepared], [value[2] for value in prepared])
                for (record, _, _), (text, error) in zip(prepared, generated):
                    if error is None and text:
                        record.update({"status": "ok", "generated_report": text, "error": None})
                    else:
                        error = error or ValueError("Empty generated ECG report")
                        record.update({"status": "error", "generated_report": "", "error": repr(error)})
                    stream.write(json.dumps(record) + "\n"); stream.flush(); progress.update(1)
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("Generation files:", sorted(str(path) for path in generation_root.glob("*.jsonl")))
    """),
])


write("20_evaluate_rag_report_generation.ipynb", [
    markdown(r"""
    # 20 - Evaluate ECG diagnosis generation and CXR augmentation

    The primary endpoint is ECG clinical-content quality. CXR is evaluated only as a context source.
    Fabrication is the fraction of generated positive ECG concepts absent from the held-out machine
    diagnosis; omission is the fraction of reference concepts not generated. Study-macro and label-micro
    rates, macro/micro F1, Hamming accuracy, per-label results, CIDEr, BERTScore, ROUGE,
    BLEU-1/BLEU-4, and METEOR are reported. BLEU and METEOR are included for comparison with
    prior IEEE ECG report-generation work; structured ECG-label outcomes remain primary.

    Paired comparisons directly test the ECG-to-ECG neighbor baseline versus zero-shot, CXR-only versus
    zero-shot, ECG+CXR versus ECG-only, CodeBind versus CLIP CXR retrieval, and retrieved versus random
    CXR context. The ECG-only condition is ECG-ReGen-style but is not an exact reproduction. Automated
    labels are not a clinical safety determination; blinded cardiologist review remains required.
    """),
    code(r"""
    from pathlib import Path
    import gc
    import json
    import os
    import re

    ARTIFACT_ROOT = Path(os.environ.get(
        "ECG_CXR_BRIDGE_ARTIFACT_ROOT",
        "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    RAG_ROOT = ARTIFACT_ROOT / "rag_report_generation_ecg"
    REPORT_ROOT = RAG_ROOT / "reports"
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    config = json.loads((RAG_ROOT / "rag_config.json").read_text())
    HF_CACHE = Path(config.get("hf_cache") or "/data/liangz2/ecg_analysis/hf_cache")
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["HF_XET_CACHE"] = str(HF_CACHE / "xet")
    os.environ["MPLCONFIGDIR"] = str(RAG_ROOT / "cache" / "matplotlib")
    os.environ["NLTK_DATA"] = str(RAG_ROOT / "cache" / "nltk_data")
    for directory in [HF_CACHE / "hub", HF_CACHE / "xet", Path(os.environ["MPLCONFIGDIR"]),
                      Path(os.environ["NLTK_DATA"])]:
        directory.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    try:
        import seaborn as sns
    except ModuleNotFoundError:
        sns = None
        print("Optional package seaborn is not installed; figures will use the Matplotlib fallback.")
    import nltk
    import torch
    from bert_score import BERTScorer
    from nltk.corpus import wordnet
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import single_meteor_score
    from pycocoevalcap.cider.cider import Cider
    from rouge_score import rouge_scorer
    from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support

    ALLOW_INCOMPLETE = os.environ.get("RAG_ALLOW_INCOMPLETE", "0").lower() in {"1", "true", "yes"}
    BOOTSTRAPS = int(os.environ.get("RAG_EVAL_BOOTSTRAPS", "500"))
    BERT_MODEL = os.environ.get("RAG_BERTSCORE_MODEL", "microsoft/deberta-xlarge-mnli")
    BERT_BATCH = int(os.environ.get("RAG_BERTSCORE_BATCH_SIZE", "32"))
    BERT_MAX_LENGTH = int(os.environ.get("RAG_BERTSCORE_MAX_LENGTH", "512"))
    if BERT_BATCH < 1 or BERT_MAX_LENGTH < 8:
        raise ValueError("BERTScore batch size must be positive and max length must be at least 8")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # METEOR uses WordNet. Keep its small resource under the project artifact root
    # rather than Biowulf's quota-limited home directory.
    try:
        wordnet.ensure_loaded()
    except LookupError:
        downloaded = nltk.download(
            "wordnet", download_dir=os.environ["NLTK_DATA"], quiet=True)
        if not downloaded:
            raise RuntimeError(
                "METEOR requires the NLTK WordNet resource. On a network-enabled node run: "
                f"python -m nltk.downloader -d {os.environ['NLTK_DATA']} wordnet")
        wordnet.ensure_loaded()

    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    manifests = {m: read_jsonl(ARTIFACT_ROOT / "external" / m / "manifest.jsonl") for m in ["ecg", "cxr"]}
    concept_rows = {m: read_jsonl(ARTIFACT_ROOT / "external" / m / "concepts.jsonl") for m in ["ecg", "cxr"]}
    manifest_lookup = {m: {str(row["sample_id"]): row for row in values} for m, values in manifests.items()}
    concept_lookup = {m: {str(row["sample_id"]): row for row in values} for m, values in concept_rows.items()}
    contexts = read_jsonl(RAG_ROOT / "contexts" / "rag_contexts.jsonl")
    context_lookup = {(row["condition"], str(row["sample_id"])): row for row in contexts}
    if len(context_lookup) != len(contexts):
        raise ValueError("Duplicate active context keys")
    """),
    code(r"""
    ECG_EVALUATION_RULE_VERSION = "ecg_generation_assertion_rules_v3"
    ECG_RULES = {
        # Sinus rhythm alone does not imply that the complete ECG is normal.
        "normal_ecg": [
            r"\bnormal(?:\s+12[- ]lead)?\s+(?:ecg|electrocardiogram)\b",
            r"\b(?:ecg|electrocardiogram)\s+(?:is\s+)?within normal limits\b",
        ],
        "atrial_fibrillation": [r"\batrial fibrillation\b", r"\ba[- ]?fib(?:rillation)?\b"],
        "atrial_flutter": [r"\batrial flutter\b"],
        "tachycardia": [r"\btachycard(?:ia|ic)\b", r"\b(?:svt|v[- ]?tach)\b"],
        "bradycardia": [r"\bbradycard(?:ia|ic)\b"],
        "conduction_abnormality": [
            r"\bbundle branch block\b", r"\b(?:rbbb|lbbb|ivcd)\b",
            r"\b(?:av|atrioventricular|heart) block\b",
            r"\b(?:first|second|third)[ -]degree (?:av|atrioventricular) block\b",
            r"\bintraventricular conduction (?:delay|defect)\b",
        ],
        "ventricular_hypertrophy": [
            r"\b(?:left|right) ventricular hypertrophy\b",
            r"\bbiventricular hypertrophy\b", r"\bventricular hypertrophy\b",
            r"\b(?:lvh|rvh)\b",
        ],
        "ischemia": [
            r"\bischemi\w*\b", r"\bst[- ]t abnormalit\w*\b",
            r"\bst(?:[- ]segment)?\s+(?:depression|elevation)\b",
        ],
        "infarction": [
            r"\bmyocardial infarction\b", r"\binfarct(?:ion|ed)?\b",
            r"\b(?:stemi|nstemi)\b",
        ],
        "paced_rhythm": [
            r"\b(?:atrial|ventricular|dual[- ]chamber)?\s*paced rhythm\b",
            r"\bpacing spikes?\b", r"\bpacemaker rhythm\b",
        ],
        "low_voltage": [r"\blow[- ]voltage\b"],
        "qt_abnormality": [
            r"\b(?:prolonged|long|short)\s+qt(?:c)?(?: interval)?\b",
            r"\bqt(?:c)? prolongation\b",
        ],
    }
    CHEXPERT_MAP = {
        "atelectasis": "atelectasis", "cardiomegaly": "cardiomegaly",
        "consolidation": "consolidation", "edema": "edema",
        "enlarged cardiomediastinum": "enlarged_cardiomediastinum", "fracture": "fracture",
        "lung lesion": "lung_lesion", "lung opacity": "lung_opacity", "no finding": "no_finding",
        "pleural effusion": "pleural_effusion", "pleural other": "pleural_other",
        "pneumonia": "pneumonia", "pneumothorax": "pneumothorax",
        "support devices": "support_device", "support device": "support_device",
    }
    ECG_LABELS = sorted(ECG_RULES)
    NEGATED_BEFORE = re.compile(
        r"(?:\bno\b|\bwithout\b|\bnegative for\b|\babsence of\b|\bfree of\b)"
        r"(?:\W+\w+){0,8}\W*$", re.I)
    NEGATED_AFTER = re.compile(r"^\W*(?:is |are |was |were )?(?:not|absent)(?:\W|$)", re.I)
    UNCERTAIN_BEFORE = re.compile(
        r"(?:\bpossible\b|\bpossibly\b|\bprobable\b|\blikely\b|\bmay\b|\bmight\b|"
        r"\bcannot exclude\b|\bcannot rule out\b|\bconcerning for\b|\bsuspicious for\b)"
        r"(?:\W+\w+){0,8}\W*$", re.I)
    UNCERTAIN_AFTER = re.compile(r"^\W*(?:is |are )?(?:possible|probable|suspected)(?:\W|$)", re.I)

    def assertion(text, match):
        before = text[max(0, match.start() - 120):match.start()]
        before = re.split(r"[.;\n]|\b(?:but|however|although|yet)\b", before, flags=re.I)[-1]
        after = text[match.end():match.end() + 50]
        if NEGATED_BEFORE.search(before) or NEGATED_AFTER.search(after):
            return "absent"
        if UNCERTAIN_BEFORE.search(before) or UNCERTAIN_AFTER.search(after):
            return "uncertain"
        return "present"

    def extract_ecg_assertions(text):
        value = " ".join(str(text).split()).lower()
        result = {"present": set(), "absent": set(), "uncertain": set()}
        for concept, patterns in ECG_RULES.items():
            for pattern in patterns:
                for match in re.finditer(pattern, value):
                    result[assertion(value, match)].add(concept)
        return result

    def extract_ecg_present(text):
        present = set(extract_ecg_assertions(text)["present"])
        if "normal_ecg" in present and len(present) > 1:
            present.remove("normal_ecg")
        return present

    ECG_RULE_TESTS = [
        ("Normal 12-lead ECG.", {"normal_ecg"}, set(), set()),
        ("Normal sinus rhythm. Inferior infarct, age undetermined.", {"infarction"}, set(), set()),
        ("No evidence of atrial fibrillation.", set(), {"atrial_fibrillation"}, set()),
        ("Possible left ventricular hypertrophy.", set(), set(), {"ventricular_hypertrophy"}),
        ("No acute ischemic changes.", set(), {"ischemia"}, set()),
        ("Ventricular paced rhythm.", {"paced_rhythm"}, set(), set()),
        ("No ST-segment elevation or depression.", set(), {"ischemia"}, set()),
    ]
    for test_text, expected_present, expected_absent, expected_uncertain in ECG_RULE_TESTS:
        observed = extract_ecg_assertions(test_text)
        expected = {
            "present": expected_present, "absent": expected_absent,
            "uncertain": expected_uncertain,
        }
        if observed != expected:
            raise AssertionError(
                f"ECG evaluation-rule self-test failed for {test_text!r}: "
                f"expected={expected}, observed={observed}")
    print("ECG generated-report extractor:", ECG_EVALUATION_RULE_VERSION,
          "self-tests:", len(ECG_RULE_TESTS))

    def reference_ecg_labels(sample_id):
        values = concept_lookup["ecg"][sample_id].get("concepts", [])
        labels = {value.split(":", 1)[1] for value in values
                  if value.startswith("present:") and value.split(":", 1)[1] in ECG_LABELS}
        if "normal_ecg" in labels and len(labels) > 1:
            labels.remove("normal_ecg")
        return labels

    def reference_cxr_labels(sample_id):
        labels = manifest_lookup["cxr"][sample_id].get("labels", [])
        return {CHEXPERT_MAP[str(value).strip().lower()] for value in labels
                if str(value).strip().lower() in CHEXPERT_MAP}

    CXR_TO_ECG_RELATIONS = {
        "cardiomegaly": {"ventricular_hypertrophy"},
        "enlarged_cardiomediastinum": {"ventricular_hypertrophy"},
        "support_device": {"paced_rhythm"},
    }
    ECG_TO_CXR_RELATIONS = {
        "ventricular_hypertrophy": {"cardiomegaly", "enlarged_cardiomediastinum"},
        "paced_rhythm": {"support_device"},
    }

    def context_implied_ecg(row):
        implied = set()
        for item in row["retrieved"]:
            sample_id = str(item["sample_id"])
            if item["modality"] == "ecg":
                implied.update(extract_ecg_present(manifest_lookup["ecg"][sample_id]["machine_diagnosis"]))
            else:
                for label in reference_cxr_labels(sample_id):
                    implied.update(CXR_TO_ECG_RELATIONS.get(label, set()))
        return implied
    """),
    code(r"""
    generation_rows = []
    for model_key in config["models"]:
        path = RAG_ROOT / "generations" / f"{model_key}.jsonl"
        if path.is_file():
            expected_model_id = config["models"][model_key]
            generation_rows.extend(
                row for row in read_jsonl(path)
                if row.get("model_id") == expected_model_id)
    active_signatures = {row["cohort_signature"] for row in contexts}
    if len(active_signatures) != 1:
        raise ValueError(f"Expected one active cohort signature; found {active_signatures}")
    generation_rows = [row for row in generation_rows
                       if row.get("status") == "ok"
                       and row.get("cohort_signature") in active_signatures
                       and row.get("generation_protocol_version") == config["generation_protocol_version"]]
    if not generation_rows:
        raise ValueError("No successful current-protocol generations; run notebook 19")
    keys = [(row["model_key"], row["condition"], str(row["sample_id"])) for row in generation_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate successful generation keys")

    context_keys = {(row["condition"], str(row["sample_id"])) for row in contexts}
    expected_keys = {(model, condition, sample_id) for model in config["models"]
                     for condition, sample_id in context_keys}
    observed_keys = set(keys)
    missing, unexpected = sorted(expected_keys - observed_keys), sorted(observed_keys - expected_keys)
    if (missing or unexpected) and not ALLOW_INCOMPLETE:
        raise RuntimeError(
            f"Incomplete generation matrix. Missing={missing[:20]}; unexpected={unexpected[:20]}. "
            "Finish notebook 19 or set RAG_ALLOW_INCOMPLETE=1 for a smoke analysis.")
    print({"successful_generations": len(generation_rows), "missing": len(missing),
           "unexpected": len(unexpected), "allow_incomplete": ALLOW_INCOMPLETE})
    """),
    code(r"""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    bleu_smoothing = SmoothingFunction().method1

    def report_tokens(value):
        return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", str(value).lower())

    def lexical_generation_scores(reference, candidate):
        reference_tokens = report_tokens(reference)
        candidate_tokens = report_tokens(candidate)
        if not reference_tokens or not candidate_tokens:
            return {"bleu1": 0.0, "bleu4": 0.0, "meteor": 0.0}
        references = [reference_tokens]
        return {
            "bleu1": float(sentence_bleu(
                references, candidate_tokens, weights=(1.0,),
                smoothing_function=bleu_smoothing)),
            "bleu4": float(sentence_bleu(
                references, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=bleu_smoothing)),
            "meteor": float(single_meteor_score(
                reference_tokens, candidate_tokens, wordnet=wordnet)),
        }

    detailed = []
    for row in generation_rows:
        sample_id = str(row["sample_id"])
        reference_text = manifest_lookup["ecg"][sample_id]["machine_diagnosis"]
        truth = reference_ecg_labels(sample_id)
        predicted_assertions = extract_ecg_assertions(row["generated_report"])
        predicted = set(predicted_assertions["present"])
        if "normal_ecg" in predicted and len(predicted) > 1:
            predicted.remove("normal_ecg")
        assertion_matches = set().union(*predicted_assertions.values())
        contradictory = (
            (predicted_assertions["present"] & predicted_assertions["absent"])
            | (predicted_assertions["present"] & predicted_assertions["uncertain"])
            | (predicted_assertions["absent"] & predicted_assertions["uncertain"])
        )
        context_row = context_lookup[(row["condition"], sample_id)]
        context_concepts = context_implied_ecg(context_row)
        y_true = [int(label in truth) for label in ECG_LABELS]
        y_pred = [int(label in predicted) for label in ECG_LABELS]
        fabricated, omitted = predicted - truth, truth - predicted
        tp, fp, fn = len(truth & predicted), len(fabricated), len(omitted)
        rouge = scorer.score(str(reference_text), row["generated_report"])
        lexical = lexical_generation_scores(reference_text, row["generated_report"])
        detailed.append({
            **row, "reference_text": reference_text,
            "ecg_evaluation_rule_version": ECG_EVALUATION_RULE_VERSION,
            "truth_labels": sorted(truth), "predicted_labels": sorted(predicted),
            "predicted_absent_labels": sorted(predicted_assertions["absent"]),
            "predicted_uncertain_labels": sorted(predicted_assertions["uncertain"]),
            "predicted_contradictory_labels": sorted(contradictory),
            "extractor_matched_any_assertion": bool(assertion_matches),
            "extractor_matched_present_label": bool(predicted),
            "context_implied_ecg_labels": sorted(context_concepts),
            "fabricated_labels": sorted(fabricated), "omitted_labels": sorted(omitted),
            "y_true": y_true, "y_pred": y_pred, "tp": tp, "fp": fp, "fn": fn,
            "fabrication_rate": fp / max(len(predicted), 1),
            "omission_rate": fn / max(len(truth), 1),
            "sample_hamming_accuracy": 1 - np.mean(np.not_equal(y_true, y_pred)),
            "context_supported_prediction_fraction": len(predicted & context_concepts) / max(len(predicted), 1),
            "context_attributable_fabrication_fraction": len(fabricated & context_concepts) / max(len(fabricated), 1),
            "rouge1_f1": rouge["rouge1"].fmeasure,
            "rouge2_f1": rouge["rouge2"].fmeasure,
            "rougeL_f1": rouge["rougeL"].fmeasure,
            **lexical,
        })

    candidates = [row["generated_report"] for row in detailed]
    references = [str(row["reference_text"]) for row in detailed]

    def build_bert_scorer(use_fast_tokenizer):
        scorer = BERTScorer(
            model_type=BERT_MODEL, lang="en", batch_size=BERT_BATCH,
            device=DEVICE, rescale_with_baseline=False,
            use_fast_tokenizer=use_fast_tokenizer)
        # Some model tokenizers advertise a huge sentinel model_max_length.
        # New Rust tokenizers cannot convert that value to their bounded integer
        # type and raise OverflowError before scoring. The underlying DeBERTa
        # model supports at most 512 tokens, so cap both public and initialization
        # metadata before the first encode call.
        scorer._tokenizer.model_max_length = BERT_MAX_LENGTH
        if hasattr(scorer._tokenizer, "init_kwargs"):
            scorer._tokenizer.init_kwargs["model_max_length"] = BERT_MAX_LENGTH
        return scorer

    bert_scorer = build_bert_scorer(use_fast_tokenizer=True)
    try:
        _, _, bert_f1 = bert_scorer.score(
            candidates, references, batch_size=BERT_BATCH, verbose=True)
    except OverflowError:
        print("Fast BERTScore tokenizer overflowed despite the length cap; retrying with the slow tokenizer.")
        del bert_scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bert_scorer = build_bert_scorer(use_fast_tokenizer=False)
        _, _, bert_f1 = bert_scorer.score(
            candidates, references, batch_size=BERT_BATCH, verbose=True)
    for row, value in zip(detailed, bert_f1.cpu().numpy()):
        row["bertscore_f1"] = float(value)
    del bert_scorer, bert_f1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Per-sample ECG metrics computed:", len(detailed))
    """),
    code(r"""
    def cider_values(rows):
        references = {i: [str(row["reference_text"])] for i, row in enumerate(rows)}
        candidates = {i: [str(row["generated_report"])] for i, row in enumerate(rows)}
        corpus, per_sample = Cider().compute_score(references, candidates)
        return float(corpus), np.asarray(per_sample, dtype="float64")

    summary_rows = []
    frame_before_cider = pd.DataFrame(detailed)
    for keys, group in frame_before_cider.groupby(["model_key", "condition"], sort=True):
        records = group.to_dict("records")
        y_true = np.asarray(group.y_true.tolist(), dtype="int8")
        y_pred = np.asarray(group.y_pred.tolist(), dtype="int8")
        cider, per_cider = cider_values(records)
        cider_map = {str(record["sample_id"]): float(value) for record, value in zip(records, per_cider)}
        for record in detailed:
            if (record["model_key"], record["condition"]) == keys:
                record["cider"] = cider_map[str(record["sample_id"])]
        summary_rows.append({
            "model_key": keys[0], "condition": keys[1], "n": len(group),
            "ecg_evaluation_rule_version": ECG_EVALUATION_RULE_VERSION,
            "retrieval_method": group.retrieval_method.iloc[0],
            "context_scope": group.context_scope.iloc[0],
            "fabrication_rate": float(group.fabrication_rate.mean()),
            "fabrication_rate_micro": float(group.fp.sum() / max(group.y_pred.map(sum).sum(), 1)),
            "omission_rate": float(group.omission_rate.mean()),
            "omission_rate_micro": float(group.fn.sum() / max(group.y_true.map(sum).sum(), 1)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "hamming_accuracy": float(1 - hamming_loss(y_true, y_pred)),
            "cider": cider, "bertscore_f1": float(group.bertscore_f1.mean()),
            "rouge1_f1": float(group.rouge1_f1.mean()), "rouge2_f1": float(group.rouge2_f1.mean()),
            "rougeL_f1": float(group.rougeL_f1.mean()),
            "bleu1": float(group.bleu1.mean()), "bleu4": float(group.bleu4.mean()),
            "meteor": float(group.meteor.mean()),
            "context_supported_prediction_fraction": float(group.context_supported_prediction_fraction.mean()),
            "context_attributable_fabrication_fraction": float(group.context_attributable_fabrication_fraction.mean()),
            "truth_positive_labels": int(y_true.sum()), "predicted_positive_labels": int(y_pred.sum()),
            "reports_with_any_assertion_fraction": float(group.extractor_matched_any_assertion.mean()),
            "reports_with_present_label_fraction": float(group.extractor_matched_present_label.mean()),
            "reports_with_contradictory_assertion_fraction": float(
                group.predicted_contradictory_labels.map(bool).mean()),
        })
    summary = pd.DataFrame(summary_rows)
    detailed_frame = pd.DataFrame(detailed)
    detailed_frame.to_csv(REPORT_ROOT / "ecg_rag_metrics_detailed.csv", index=False)
    summary.to_csv(REPORT_ROOT / "ecg_rag_metrics_summary.csv", index=False)
    evaluation_protocol = {
        "primary_endpoint_family": "structured ECG clinical-content metrics",
        "reference_source": "held-out external ECG machine diagnosis",
        "bleu1": "mean smoothed sentence BLEU; NLTK SmoothingFunction.method1; weights=(1.0,)",
        "bleu4": "mean smoothed sentence BLEU; NLTK SmoothingFunction.method1; weights=(0.25,0.25,0.25,0.25)",
        "meteor": "mean NLTK single_meteor_score with WordNet synonym matching",
        "lexical_tokenization": "lowercase alphanumeric tokens retaining internal hyphen/apostrophe",
        "nltk_version": nltk.__version__,
        "ecg_evaluation_rule_version": ECG_EVALUATION_RULE_VERSION,
        "generation_protocol_version": config["generation_protocol_version"],
        "ecg_neighbor_baseline": config.get("ecg_neighbor_baseline_note"),
    }
    (REPORT_ROOT / "ecg_rag_evaluation_protocol.json").write_text(
        json.dumps(evaluation_protocol, indent=2))
    extractor_coverage_columns = [
        "model_key", "condition", "n", "reports_with_any_assertion_fraction",
        "reports_with_present_label_fraction", "reports_with_contradictory_assertion_fraction",
        "truth_positive_labels", "predicted_positive_labels",
    ]
    summary[extractor_coverage_columns].to_csv(
        REPORT_ROOT / "ecg_rag_extractor_coverage.csv", index=False)

    per_label_rows = []
    for keys, group in detailed_frame.groupby(["model_key", "condition"], sort=True):
        y_true = np.asarray(group.y_true.tolist(), dtype="int8")
        y_pred = np.asarray(group.y_pred.tolist(), dtype="int8")
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0)
        for label, p, r, score, count in zip(ECG_LABELS, precision, recall, f1, support):
            per_label_rows.append({
                "model_key": keys[0], "condition": keys[1], "label": label,
                "support": int(count), "precision": float(p), "recall": float(r), "f1": float(score)})
    pd.DataFrame(per_label_rows).to_csv(REPORT_ROOT / "ecg_rag_per_label_metrics.csv", index=False)

    query_ids = sorted({str(row["sample_id"]) for row in contexts})
    prevalence = [{
        "label": label, "n_queries": len(query_ids),
        "positive_count": sum(label in reference_ecg_labels(sample_id) for sample_id in query_ids),
        "prevalence": sum(label in reference_ecg_labels(sample_id) for sample_id in query_ids) / max(len(query_ids), 1),
    } for label in ECG_LABELS]
    pd.DataFrame(prevalence).to_csv(REPORT_ROOT / "ecg_rag_cohort_label_prevalence.csv", index=False)
    display(summary.sort_values(["model_key", "condition"]))
    """),
    code(r"""
    def grouped_metrics(group):
        y_true = np.asarray(group.y_true.tolist(), dtype="int8")
        y_pred = np.asarray(group.y_pred.tolist(), dtype="int8")
        return {
            "fabrication_rate": float(group.fabrication_rate.mean()),
            "fabrication_rate_micro": float(group.fp.sum() / max(group.y_pred.map(sum).sum(), 1)),
            "omission_rate": float(group.omission_rate.mean()),
            "omission_rate_micro": float(group.fn.sum() / max(group.y_true.map(sum).sum(), 1)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "hamming_accuracy": float(1 - hamming_loss(y_true, y_pred)),
            "cider": float(group.cider.mean()), "bertscore_f1": float(group.bertscore_f1.mean()),
            "rouge1_f1": float(group.rouge1_f1.mean()), "rouge2_f1": float(group.rouge2_f1.mean()),
            "rougeL_f1": float(group.rougeL_f1.mean()),
            "bleu1": float(group.bleu1.mean()), "bleu4": float(group.bleu4.mean()),
            "meteor": float(group.meteor.mean()),
            "context_attributable_fabrication_fraction": float(
                group.context_attributable_fabrication_fraction.mean()),
        }

    rng = np.random.default_rng(int(config["seed"]) + 91)
    bootstrap_rows = []
    if BOOTSTRAPS > 0:
        for keys, group in detailed_frame.groupby(["model_key", "condition"], sort=True):
            values = {metric: [] for metric in grouped_metrics(group)}
            for _ in range(BOOTSTRAPS):
                sampled = group.iloc[rng.integers(0, len(group), size=len(group))]
                result = grouped_metrics(sampled)
                for metric, value in result.items():
                    values[metric].append(value)
            for metric, samples in values.items():
                bootstrap_rows.append({
                    "model_key": keys[0], "condition": keys[1], "metric": metric,
                    "bootstrap_samples": BOOTSTRAPS,
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                })
    pd.DataFrame(bootstrap_rows).to_csv(
        REPORT_ROOT / "ecg_rag_metric_confidence_intervals.csv", index=False)
    """),
    code(r"""
    comparison_pairs = []
    for condition in config["conditions"]:
        if condition != "zero_shot":
            comparison_pairs.append((condition, "zero_shot", "rag_vs_zero_shot"))
    for method in ["clip", "codebind", "combined"]:
        comparison_pairs.extend([
            (f"{method}_ecg_only_rag", "zero_shot", "ecg_neighbor_vs_zero_shot"),
            (f"{method}_ecg_plus_cxr_rag", f"{method}_ecg_only_rag", "cxr_added_to_ecg_rag"),
            (f"{method}_cxr_only_rag", "random_cxr_rag", "retrieved_vs_random_cxr"),
        ])
    for scope in ["ecg_only_rag", "cxr_only_rag", "ecg_plus_cxr_rag"]:
        comparison_pairs.extend([
            (f"codebind_{scope}", f"clip_{scope}", "codebind_vs_clip"),
            (f"combined_{scope}", f"clip_{scope}", "combined_vs_clip"),
            (f"combined_{scope}", f"codebind_{scope}", "combined_vs_codebind"),
        ])
    comparison_pairs = list(dict.fromkeys(comparison_pairs))
    lower_is_better = {
        "fabrication_rate", "fabrication_rate_micro", "omission_rate", "omission_rate_micro",
        "context_attributable_fabrication_fraction"}
    paired_rows = []
    for model_key, outer in detailed_frame.groupby("model_key", sort=True):
        by_condition = {condition: frame.set_index("sample_id", drop=False)
                        for condition, frame in outer.groupby("condition")}
        for condition_a, condition_b, family in comparison_pairs:
            if condition_a not in by_condition or condition_b not in by_condition:
                continue
            common_ids = sorted(set(by_condition[condition_a].index) & set(by_condition[condition_b].index))
            if not common_ids:
                continue
            left = by_condition[condition_a].loc[common_ids].reset_index(drop=True)
            right = by_condition[condition_b].loc[common_ids].reset_index(drop=True)
            point_a, point_b = grouped_metrics(left), grouped_metrics(right)
            samples = {metric: [] for metric in point_a}
            if BOOTSTRAPS > 0:
                for _ in range(BOOTSTRAPS):
                    positions = rng.integers(0, len(common_ids), size=len(common_ids))
                    metric_a, metric_b = grouped_metrics(left.iloc[positions]), grouped_metrics(right.iloc[positions])
                    for metric in samples:
                        improvement = (metric_b[metric] - metric_a[metric]
                                       if metric in lower_is_better
                                       else metric_a[metric] - metric_b[metric])
                        samples[metric].append(improvement)
            for metric in point_a:
                improvement = (point_b[metric] - point_a[metric]
                               if metric in lower_is_better else point_a[metric] - point_b[metric])
                values = samples[metric]
                paired_rows.append({
                    "model_key": model_key, "comparison_family": family,
                    "condition_a": condition_a, "condition_b": condition_b,
                    "metric": metric, "n_paired": len(common_ids),
                    "value_a": point_a[metric], "value_b": point_b[metric],
                    "improvement_a_over_b": float(improvement),
                    "ci_low": float(np.quantile(values, 0.025)) if values else np.nan,
                    "ci_high": float(np.quantile(values, 0.975)) if values else np.nan,
                    "bootstrap_probability_improvement": float(np.mean(np.asarray(values) > 0))
                    if values else np.nan,
                })
    paired_columns = [
        "model_key", "comparison_family", "condition_a", "condition_b", "metric",
        "n_paired", "value_a", "value_b", "improvement_a_over_b", "ci_low", "ci_high",
        "bootstrap_probability_improvement"]
    paired = pd.DataFrame(paired_rows, columns=paired_columns)
    paired.to_csv(REPORT_ROOT / "ecg_rag_paired_comparisons.csv", index=False)
    primary_metrics = {"micro_f1", "fabrication_rate", "omission_rate"}
    ecg_neighbor_vs_zero = paired.comparison_family.eq("ecg_neighbor_vs_zero_shot")
    cxr_vs_zero = (
        paired.comparison_family.eq("rag_vs_zero_shot") &
        paired.condition_a.str.endswith("_cxr_only_rag"))
    cxr_added = paired.comparison_family.eq("cxr_added_to_ecg_rag")
    retrieved_vs_random = paired.comparison_family.eq("retrieved_vs_random_cxr")
    codebind_cxr_vs_clip = (
        paired.comparison_family.eq("codebind_vs_clip") &
        paired.condition_a.eq("codebind_cxr_only_rag") &
        paired.condition_b.eq("clip_cxr_only_rag"))
    primary_contrasts = paired[
        paired.metric.isin(primary_metrics) &
        (ecg_neighbor_vs_zero | cxr_vs_zero | cxr_added |
         retrieved_vs_random | codebind_cxr_vs_clip)]
    primary_contrasts.to_csv(REPORT_ROOT / "ecg_rag_primary_contrasts.csv", index=False)
    display(primary_contrasts)
    """),
    code(r"""
    context_quality_rows = []
    for row in contexts:
        sample_id = str(row["sample_id"])
        query_labels = reference_ecg_labels(sample_id)
        expected_cxr = set().union(*(ECG_TO_CXR_RELATIONS.get(label, set()) for label in query_labels))
        same_flags, cross_flags = [], []
        for item in row["retrieved"]:
            source_id = str(item["sample_id"])
            if item["modality"] == "ecg":
                source = extract_ecg_present(manifest_lookup["ecg"][source_id]["machine_diagnosis"])
                same_flags.append(bool(query_labels & source))
            elif expected_cxr:
                cross_flags.append(bool(expected_cxr & reference_cxr_labels(source_id)))
        context_quality_rows.append({
            "condition": row["condition"], "retrieval_method": row["retrieval_method"],
            "context_scope": row["context_scope"], "sample_id": sample_id,
            "same_label_hit_at_k": float(any(same_flags)) if same_flags else np.nan,
            "same_label_overlap_fraction": float(np.mean(same_flags)) if same_flags else np.nan,
            "cxr_relation_hit_at_k": float(any(cross_flags)) if cross_flags else np.nan,
            "cxr_relation_overlap_fraction": float(np.mean(cross_flags)) if cross_flags else np.nan,
            "cxr_relation_evaluable": bool(expected_cxr),
        })
    context_quality = pd.DataFrame(context_quality_rows)
    context_quality.to_csv(REPORT_ROOT / "ecg_rag_context_quality_detailed.csv", index=False)
    context_quality_summary = context_quality.groupby(
        ["condition", "retrieval_method", "context_scope"], as_index=False).agg(
        n_queries=("sample_id", "size"),
        same_label_hit_at_k=("same_label_hit_at_k", "mean"),
        same_label_overlap_fraction=("same_label_overlap_fraction", "mean"),
        cxr_relation_hit_at_k=("cxr_relation_hit_at_k", "mean"),
        cxr_relation_overlap_fraction=("cxr_relation_overlap_fraction", "mean"),
        cxr_relation_evaluable=("cxr_relation_evaluable", "sum"))
    context_quality_summary.to_csv(REPORT_ROOT / "ecg_rag_context_quality_summary.csv", index=False)
    display(context_quality_summary)
    """),
    code(r"""
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    order = config["conditions"]
    model_order = list(config["models"])

    def horizontal_metric_plot(metrics, title, filename, xlim=None):
        fig, axes = plt.subplots(1, len(model_order), figsize=(18, 10), sharey=True)
        axes = np.atleast_1d(axes)
        legend_handles, legend_labels = [], []
        for axis, model_key in zip(axes, model_order):
            model_frame = summary[summary.model_key.eq(model_key)].set_index("condition")
            model_frame = model_frame.reindex(order)
            if sns is not None:
                frame = model_frame.reset_index().melt(
                    id_vars="condition", value_vars=metrics,
                    var_name="metric", value_name="value")
                sns.barplot(
                    data=frame, y="condition", x="value", hue="metric",
                    order=order, ax=axis)
            else:
                positions = np.arange(len(order), dtype="float64")
                height = 0.8 / len(metrics)
                for metric_index, metric in enumerate(metrics):
                    offset = (metric_index - (len(metrics) - 1) / 2) * height
                    axis.barh(
                        positions + offset, model_frame[metric].to_numpy(dtype="float64"),
                        height=height, label=metric)
                axis.set_yticks(positions, order); axis.invert_yaxis()
            handles, labels = axis.get_legend_handles_labels()
            if handles and not legend_handles:
                legend_handles, legend_labels = handles, labels
            panel_legend = axis.get_legend()
            if panel_legend is not None:
                panel_legend.remove()
            axis.set_title(model_key); axis.set_ylabel("")
            if xlim is not None:
                axis.set_xlim(*xlim)
        fig.suptitle(title, y=0.995)
        if legend_handles:
            fig.legend(
                legend_handles, legend_labels,
                loc="upper center", bbox_to_anchor=(0.5, 0.955),
                ncol=len(legend_labels), frameon=False,
                fontsize=11, handlelength=1.6, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(REPORT_ROOT / filename, dpi=180, bbox_inches="tight")
        plt.show()

    horizontal_metric_plot(
        ["macro_f1", "micro_f1", "hamming_accuracy"],
        "ECG generation: structured clinical-content performance",
        "ecg_rag_multilabel_metrics.png", (0, 1))
    horizontal_metric_plot(
        ["fabrication_rate", "omission_rate"],
        "ECG generation: fabrication and omission (lower is better)",
        "ecg_rag_fabrication_omission.png", (0, 1))
    horizontal_metric_plot(
        ["cider", "bertscore_f1", "rougeL_f1"],
        "ECG generation: text similarity", "ecg_rag_text_similarity.png")
    horizontal_metric_plot(
        ["bleu1", "bleu4", "meteor"],
        "ECG generation: IEEE-comparable lexical metrics",
        "ecg_rag_bleu_meteor.png", (0, 1))
    print("Saved ECG-focused evaluation outputs to", REPORT_ROOT)
    """),
    markdown(r"""
    ## Interpretation rules

    - The ECG-to-ECG neighbor conditions are the required active RAG baseline. They are inspired by
      ECG-ReGen but are not an exact reproduction because this study uses bridge embeddings and external
      machine diagnoses.
    - The primary CXR augmentation test is CXR-only RAG versus zero-shot.
    - The active-control test is ECG+CXR RAG versus ECG-only RAG at the same case budget.
    - Retrieved CXR context must outperform random CXR context to demonstrate retrieval specificity.
    - Positive values in paired-comparison `improvement_a_over_b` always mean condition A is better;
      fabrication and omission are direction-reversed internally.
    - Do not claim CodeBind superiority unless its paired ECG generation outcome exceeds CLIP; current
      bridge retrieval results favor the continuous baseline.
    - A text-similarity increase with worse fabrication is not a clinical improvement.
    - BLEU-1, BLEU-4, and METEOR facilitate comparison with prior ECG report-generation studies but do
      not measure clinical correctness; interpret them with the ECG-label and fabrication/omission outcomes.
    - CXR-context-attributable fabrication is a safety diagnostic, not proof of causality.
    - Review `ecg_rag_extractor_coverage.csv` and the parser self-tests before interpreting label metrics; low extraction coverage or contradictory assertions invalidate automatic fabrication/omission conclusions.
    - The ECG reference is a machine interpretation, not an adjudicated cardiologist report.
    - Conduct blinded cardiologist review before making clinical utility or safety claims.
    """),
])


print("Generated ECG-focused RAG notebooks 17-20.")
