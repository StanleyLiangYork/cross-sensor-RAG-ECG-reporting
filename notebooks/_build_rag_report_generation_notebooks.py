#!/usr/bin/env python3
"""Build notebooks 17-20 for the ECG-CXR bridge RAG report experiment."""

from pathlib import Path
from textwrap import dedent
import json


ROOT = Path(__file__).resolve().parent


# Compatibility entry point.  The ECG-focused builder supersedes the earlier
# bidirectional report-generation design.  Keep this filename safe for anyone
# who reruns the historical build command.
if __name__ == "__main__":
    replacement = ROOT / "_build_ecg_focused_rag_notebooks.py"
    namespace = {"__file__": str(replacement), "__name__": "__main__"}
    exec(compile(replacement.read_text(), str(replacement), "exec"), namespace)
    raise SystemExit(0)


def markdown(text):
    return {
        "cell_type": "markdown", "metadata": {},
        "source": (dedent(text).strip() + "\n").splitlines(keepends=True),
    }


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": (dedent(text).strip() + "\n").splitlines(keepends=True),
    }


def write(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    (ROOT / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(ROOT / name)


write("17_configure_rag_report_generation.ipynb", [
    markdown(r"""
    # 17 - Configure the ECG-CXR bridge RAG report experiment

    This extension tests whether retrieval-augmented generation improves **CXR report generation** and
    **ECG machine-style report generation**. The query itself is always supplied to the multimodal model:
    a CXR image for the CXR task and a rendered 12-lead waveform for the ECG task.

    | Condition | Retrieved context |
    |---|---|
    | `zero_shot` | no retrieved examples |
    | `clip_rag` | continuous CLIP-style bridge neighbors |
    | `codebind_rag` | shared-codebook bridge neighbors |
    | `combined_rag` | reciprocal-rank fusion of CLIP and shared-codebook neighbors |

    Each retrieval method supplies both same-modality examples and explicitly labeled cross-modal
    patient-context examples. Cross-modal evidence is described in the prompt as *related, not
    diagnostically equivalent*. Only training-split records can enter the retrieval database. Test text
    and labels are held out from prompting and used only by notebook 20.

    The LLM comparison uses `google/medgemma-4b-it` and `Qwen/Qwen3.5-4B`. MedGemma requires accepting
    its Hugging Face terms before Biowulf can download the weights.
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

    RAG_ROOT = ARTIFACT_ROOT / "rag_report_generation"
    for relative in ["contexts", "query_images/ecg", "generations", "reports", "logs"]:
        (RAG_ROOT / relative).mkdir(parents=True, exist_ok=True)

    MODELS = {
        "medgemma_4b": os.environ.get("RAG_MEDGEMMA_MODEL", "google/medgemma-4b-it"),
        "qwen35_4b": os.environ.get("RAG_QWEN_MODEL", "Qwen/Qwen3.5-4B"),
    }
    CONDITIONS = ["zero_shot", "clip_rag", "codebind_rag", "combined_rag"]
    RAG_SEED = int(os.environ.get("RAG_SEED", "5107"))
    MAX_TEST_PER_MODALITY = int(os.environ.get("RAG_MAX_TEST_PER_MODALITY", "250"))
    TOP_K_SAME = int(os.environ.get("RAG_TOP_K_SAME", "3"))
    TOP_K_CROSS = int(os.environ.get("RAG_TOP_K_CROSS", "2"))
    MAX_CONTEXT_CHARACTERS = int(os.environ.get("RAG_MAX_CONTEXT_CHARACTERS", "6000"))
    MAX_NEW_TOKENS = int(os.environ.get("RAG_MAX_NEW_TOKENS", "384"))
    CODEBIND_VARIANT = os.environ.get("RAG_CODEBIND_VARIANT", "shared_specific_vq")
    if MAX_TEST_PER_MODALITY < 1 or TOP_K_SAME < 0 or TOP_K_CROSS < 0:
        raise ValueError("RAG cohort and top-k values must be nonnegative, with at least one test query")
    if TOP_K_SAME + TOP_K_CROSS < 1:
        raise ValueError("At least one retrieved context is required for RAG conditions")
    """),
    code(r"""
    experiment_path = ARTIFACT_ROOT / "ablation" / "experiment_index.json"
    experiments = json.loads(experiment_path.read_text())
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f"No bridge experiments in {experiment_path}; run notebook 12")

    def select_by_validation(variant):
        candidates = [row for row in experiments if row.get("variant") == variant]
        if not candidates:
            raise KeyError(f"No trained bridge variant {variant!r}")
        required = [row for row in candidates if row.get("best_validation_retrieval_r1") is not None]
        if not required:
            raise KeyError(f"Variant {variant!r} has no validation retrieval score")
        return max(required, key=lambda row: (float(row["best_validation_retrieval_r1"]), -int(row["seed"])))

    selected = {
        "clip": select_by_validation("clip_continuous"),
        "codebind": select_by_validation(CODEBIND_VARIANT),
    }
    if selected["codebind"]["variant"] in {"clip_continuous", "shuffled_pairs_control"}:
        raise ValueError("RAG_CODEBIND_VARIANT must be a non-control vector-quantized bridge")
    print(json.dumps(selected, indent=2))
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
        for method, experiment in selected.items():
            required_files.append(
                ARTIFACT_ROOT / "external" / modality / "bridge" /
                experiment["variant"] / f"seed_{experiment['seed']}" / "common.npy")
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "RAG prerequisites are missing. Rerun notebooks 11, 13, and 14 after changing the CXR dataset:\n" +
            "\n".join(missing[:30]))

    cxr_summary = json.loads(
        (ARTIFACT_ROOT / "external" / "cxr" / "manifest_summary.json").read_text())
    expected_cxr_images = int(base_config.get("external_cxr_expected_train_images", 43700))
    observed_cxr_images = int(cxr_summary.get("discovery", {}).get("provided_train", {}).get("images", -1))
    summary_roots = cxr_summary.get("roots", {})
    expected_roots = {
        "provided_train": str(base_config.get("external_cxr_root", "")),
        "provided_test": str(base_config.get("external_cxr_test_root", "")),
    }
    if observed_cxr_images != expected_cxr_images or summary_roots != expected_roots:
        raise RuntimeError(
            "The external CXR artifacts are stale or came from the former dataset. "
            f"Expected {expected_cxr_images:,} training images and roots {expected_roots}; "
            f"manifest summary reports {observed_cxr_images:,} and {summary_roots}. "
            "Rerun notebooks 09, 11, 13, and 14 on Biowulf before notebook 17.")

    rag_config = {
        "artifact_root": str(ARTIFACT_ROOT), "rag_root": str(RAG_ROOT),
        "models": MODELS, "conditions": CONDITIONS, "seed": RAG_SEED,
        "max_test_per_modality": MAX_TEST_PER_MODALITY,
        "top_k_same": TOP_K_SAME, "top_k_cross": TOP_K_CROSS,
        "max_context_characters": MAX_CONTEXT_CHARACTERS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_protocol_version": "rag_report_v2_nonthinking",
        "retrieval_selection": selected,
        "hf_cache": base_config.get("hf_cache"),
        "ground_truth": {
            "cxr": "top-level CheXpert labels plus held-out Caption report",
            "ecg": "held-out machine diagnosis and deterministic assertion-aware concepts",
        },
        "leakage_controls": [
            "retrieval candidates must have split=train",
            "queries must have split=test",
            "same sample IDs are forbidden",
            "same subject IDs are excluded when available",
            "query reference text and labels are not placed in prompts",
            "bridge seed selection uses validation retrieval only",
        ],
        "metrics": [
            "fabrication_rate", "fabrication_rate_micro", "omission_rate", "omission_rate_micro",
            "macro_f1", "micro_f1",
            "hamming_accuracy", "cider", "bertscore_f1", "rouge1_f1",
            "rouge2_f1", "rougeL_f1",
        ],
    }
    config_path = RAG_ROOT / "rag_config.json"
    config_path.write_text(json.dumps(rag_config, indent=2))
    print("Saved:", config_path)
    """),
    code(r"""
    required_packages = [
        "numpy", "pandas", "torch", "transformers", "accelerate", "PIL", "wfdb",
        "matplotlib", "sklearn", "rouge_score", "bert_score", "pycocoevalcap",
    ]
    missing_packages = [name for name in required_packages if importlib.util.find_spec(name) is None]
    print({"missing_packages": missing_packages})
    if missing_packages:
        print("Install missing evaluation packages in the Biowulf environment, for example:")
        print("pip install -U transformers accelerate rouge-score bert-score pycocoevalcap scikit-learn")
    print("MedGemma access: accept the model terms on Hugging Face and authenticate on Biowulf.")
    """),
])


write("18_build_rag_retrieval_contexts.ipynb", [
    markdown(r"""
    # 18 - Build leakage-safe CLIP and CodeBind RAG contexts

    This notebook creates one fixed test cohort per modality and retrieves training examples in four
    conditions. The CLIP condition uses the selected `clip_continuous` common space. The CodeBind
    condition uses the selected shared-codebook common space. `combined_rag` applies reciprocal-rank
    fusion separately to same-modality and cross-modal results.

    Retrieval never uses test reports or diagnoses. Subject matches are masked when identifiers are
    available. The context records intentionally contain no query reference text or query labels.
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
    RAG_ROOT = ARTIFACT_ROOT / "rag_report_generation"
    config = json.loads((RAG_ROOT / "rag_config.json").read_text())
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    QUERY_BATCH = int(os.environ.get("RAG_RETRIEVAL_QUERY_BATCH", "32"))
    CANDIDATE_CHUNK = int(os.environ.get("RAG_RETRIEVAL_CANDIDATE_CHUNK", "50000"))
    if QUERY_BATCH < 1 or CANDIDATE_CHUNK < 1:
        raise ValueError("Retrieval batch and chunk sizes must be positive")

    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    manifests = {m: read_jsonl(ARTIFACT_ROOT / "external" / m / "manifest.jsonl")
                 for m in ["ecg", "cxr"]}
    concepts = {m: read_jsonl(ARTIFACT_ROOT / "external" / m / "concepts.jsonl")
                for m in ["ecg", "cxr"]}
    for modality in manifests:
        if len(manifests[modality]) != len(concepts[modality]):
            raise ValueError(f"{modality} manifest/concept length mismatch; rerun notebook 14")
        for index, (row, concept) in enumerate(zip(manifests[modality], concepts[modality])):
            if str(row["sample_id"]) != str(concept.get("sample_id")):
                raise ValueError(f"{modality} concept order mismatch at {index}")
    print({m: {s: sum(row.get("split") == s for row in values)
               for s in ["train", "val", "test"]} for m, values in manifests.items()})
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
                raise ValueError(f"{method}/{modality} embedding rows do not match manifest")
            embeddings[method][modality] = array
    for modality in ["ecg", "cxr"]:
        status[modality] = np.load(
            ARTIFACT_ROOT / "external" / modality / "encoding_status.npy", mmap_mode="r")
        if len(status[modality]) != len(manifests[modality]):
            raise ValueError(f"{modality} status length does not match manifest")

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
        method: {
            modality: (np.asarray(status[modality]) == 1) & finite_row_mask(embeddings[method][modality])
            for modality in ["ecg", "cxr"]
        }
        for method in embeddings
    }

    def stable_digest(values):
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    rng = np.random.default_rng(config["seed"])
    cohorts = {}
    cohort_label_coverage = {}

    def sampling_labels(modality, index):
        if modality == "cxr":
            return {str(value).strip().lower() for value in manifests[modality][index].get("labels", [])
                    if str(value).strip()}
        return {str(value).split(":", 1)[1] for value in concepts[modality][index].get("concepts", [])
                if str(value).startswith("present:")}

    def prevalence_aware_sample(modality, eligible, take):
        # Cover every available positive label when cohort size permits, then fill randomly.
        shuffled = eligible[rng.permutation(len(eligible))]
        label_to_indices = {}
        for index in shuffled:
            for label in sampling_labels(modality, int(index)):
                label_to_indices.setdefault(label, []).append(int(index))
        selected, selected_set = [], set()
        for label in sorted(label_to_indices, key=lambda value: (len(label_to_indices[value]), value)):
            candidate = next((index for index in label_to_indices[label] if index not in selected_set), None)
            if candidate is not None and len(selected) < take:
                selected.append(candidate); selected_set.add(candidate)
        for index in shuffled:
            if len(selected) >= take:
                break
            if int(index) not in selected_set:
                selected.append(int(index)); selected_set.add(int(index))
        return np.asarray(sorted(selected), dtype="int64"), sorted(label_to_indices)

    for modality in ["ecg", "cxr"]:
        joint_valid = np.logical_and.reduce([valid_masks[method][modality] for method in embeddings])
        eligible = np.flatnonzero(split_masks[modality]["test"] & joint_valid).astype("int64")
        if len(eligible) < 2:
            raise ValueError(f"Need at least two valid test {modality} queries; found {len(eligible)}")
        take = min(int(config["max_test_per_modality"]), len(eligible))
        cohorts[modality], available_labels = prevalence_aware_sample(modality, eligible, take)
        selected_labels = sorted(set().union(*(sampling_labels(modality, int(i)) for i in cohorts[modality])))
        cohort_label_coverage[modality] = {
            "available_positive_labels": available_labels,
            "selected_positive_labels": selected_labels,
            "missing_positive_labels": sorted(set(available_labels) - set(selected_labels)),
        }
    cohort_signature = stable_digest({
        "seed": config["seed"], "max": config["max_test_per_modality"],
        "top_k_same": config["top_k_same"], "top_k_cross": config["top_k_cross"],
        "retrieval_selection": {key: {"variant": value["variant"], "seed": value["seed"]}
                                for key, value in selections.items()},
        "selected_sample_ids": {
            modality: [str(manifests[modality][int(i)]["sample_id"]) for i in indices]
            for modality, indices in cohorts.items()},
    })
    print({m: len(v) for m, v in cohorts.items()}, "signature", cohort_signature[:12])
    print("Positive-label coverage:", cohort_label_coverage)
    """),
    code(r"""
    def numeric_subject(row):
        value = row.get("subject_id")
        try:
            return int(value) if value is not None else -1
        except (TypeError, ValueError):
            return -1

    @torch.inference_mode()
    def chunked_topk(query_values, candidate_values, query_rows, candidate_rows, k):
        if k < 1:
            return [[] for _ in range(len(query_values))]
        k = min(k, len(candidate_values))
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
                known = q_subjects[:, None] >= 0
                same_subject = known & (q_subjects[:, None] == c_subjects[None, :])
                if same_subject.any():
                    scores.masked_fill_(torch.from_numpy(same_subject).to(DEVICE), -torch.inf)
                local_k = min(k, scores.shape[1])
                local_scores, local_positions = scores.topk(local_k, dim=1)
                local_indices = local_positions + c_start
                merged_scores = torch.cat([best_scores, local_scores], dim=1)
                merged_indices = torch.cat([best_indices, local_indices], dim=1)
                best_scores, order = merged_scores.topk(k, dim=1)
                best_indices = torch.gather(merged_indices, 1, order)
            for row_scores, row_indices in zip(best_scores.cpu().numpy(), best_indices.cpu().numpy()):
                if (row_indices < 0).any() or not np.isfinite(row_scores).all():
                    raise RuntimeError("Insufficient retrieval candidates after leakage masking")
                outputs.append([(int(i), float(s)) for i, s in zip(row_indices, row_scores)])
        return outputs

    def retrieval_text(modality, row):
        key = "machine_diagnosis" if modality == "ecg" else "report_text"
        return " ".join(str(row[key]).split())

    def item(method, scope, rank, score, modality, row):
        return {
            "retrieval_method": method, "scope": scope, "rank": int(rank),
            "score": float(score), "modality": modality,
            "sample_id": str(row["sample_id"]), "subject_id": row.get("subject_id"),
            "text": retrieval_text(modality, row),
        }
    """),
    code(r"""
    contexts_by_method = {m: {method: {} for method in ["clip", "codebind"]}
                          for m in ["ecg", "cxr"]}
    diagnostics = {"cohort_signature": cohort_signature, "device": str(DEVICE), "retrievals": {}}
    for query_modality in ["ecg", "cxr"]:
        target_modality = "cxr" if query_modality == "ecg" else "ecg"
        query_indices = cohorts[query_modality]
        query_rows = [manifests[query_modality][int(i)] for i in query_indices]
        for method in ["clip", "codebind"]:
            method_records = {str(row["sample_id"]): [] for row in query_rows}
            for scope, candidate_modality, requested_k in [
                ("same_modality", query_modality, int(config["top_k_same"])),
                ("cross_modal", target_modality, int(config["top_k_cross"])),
            ]:
                candidate_indices = np.flatnonzero(
                    split_masks[candidate_modality]["train"] & valid_masks[method][candidate_modality]
                ).astype("int64")
                candidate_rows = [manifests[candidate_modality][int(i)] for i in candidate_indices]
                found = chunked_topk(
                    embeddings[method][query_modality][query_indices],
                    embeddings[method][candidate_modality][candidate_indices],
                    query_rows, candidate_rows, requested_k)
                for query_row, result in zip(query_rows, found):
                    key = str(query_row["sample_id"])
                    for rank, (position, score) in enumerate(result, start=1):
                        source_row = candidate_rows[position]
                        if source_row.get("split") != "train":
                            raise AssertionError("Non-training retrieval candidate")
                        if str(source_row["sample_id"]) == key:
                            raise AssertionError("Query sample leaked into retrieval context")
                        q_subject, c_subject = numeric_subject(query_row), numeric_subject(source_row)
                        if q_subject >= 0 and q_subject == c_subject:
                            raise AssertionError("Query subject leaked into retrieval context")
                        method_records[key].append(
                            item(method, scope, rank, score, candidate_modality, source_row))
                diagnostics["retrievals"][f"{query_modality}:{method}:{scope}"] = {
                    "queries": len(query_rows), "candidate_modality": candidate_modality,
                    "training_candidates": len(candidate_indices), "top_k": requested_k,
                }
            contexts_by_method[query_modality][method] = method_records
            print(query_modality, method, {k: len(v) for k, v in list(method_records.items())[:1]})
    """),
    code(r"""
    def reciprocal_rank_fusion(items, scope, k, constant=60):
        fused = {}
        for value in items:
            if value["scope"] != scope:
                continue
            key = (value["modality"], value["sample_id"])
            entry = fused.setdefault(key, {**value, "score": 0.0, "retrieval_method": "combined"})
            entry["score"] += 1.0 / (constant + int(value["rank"]))
        ordered = sorted(fused.values(), key=lambda row: (-row["score"], row["sample_id"]))[:k]
        for rank, value in enumerate(ordered, start=1):
            value["rank"] = rank
        return ordered

    output = RAG_ROOT / "contexts" / "rag_contexts.jsonl"
    context_rows = []
    for modality in ["ecg", "cxr"]:
        for index in cohorts[modality]:
            row = manifests[modality][int(index)]
            sample_id = str(row["sample_id"])
            clip = contexts_by_method[modality]["clip"][sample_id]
            codebind = contexts_by_method[modality]["codebind"][sample_id]
            combined = (
                reciprocal_rank_fusion(clip + codebind, "same_modality", int(config["top_k_same"])) +
                reciprocal_rank_fusion(clip + codebind, "cross_modal", int(config["top_k_cross"])))
            path_key = "waveform_path" if modality == "ecg" else "image_path"
            common = {
                "sample_id": sample_id, "modality": modality, "query_index": int(index),
                "query_path": str(row[path_key]), "split": str(row["split"]),
                "cohort_signature": cohort_signature,
            }
            for condition, retrieved in [
                ("zero_shot", []), ("clip_rag", clip),
                ("codebind_rag", codebind), ("combined_rag", combined),
            ]:
                context_rows.append({**common, "condition": condition, "retrieved": retrieved})
    with output.open("w") as stream:
        for row in context_rows:
            stream.write(json.dumps(row) + "\n")
    diagnostics.update({
        "rows": len(context_rows), "conditions": config["conditions"],
        "queries": {m: len(v) for m, v in cohorts.items()},
        "cohort_label_coverage": cohort_label_coverage,
        "leakage_violations": 0,
    })
    (RAG_ROOT / "contexts" / "retrieval_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2))
    print("Saved:", output, diagnostics)
    """),
])


write("19_generate_rag_reports.ipynb", [
    markdown(r"""
    # 19 - Generate CXR and ECG reports with zero-shot and RAG prompts

    This notebook runs MedGemma and Qwen3.5 one model at a time. It is resumable: completed
    `(model, modality, condition, sample_id)` records are skipped, while failures can be retried with
    `RAG_RETRY_ERRORS=1`.

    Qwen3.5 thinking is disabled through its chat-template argument so its token budget is spent on the
    clinical report and the output format remains comparable to MedGemma. Decoding is deterministic.

    The query image is always included. ECG waveforms are rendered into a standardized 12-lead panel.
    Retrieved examples are from the training split and are framed as non-authoritative context from
    different patients. The prompt never includes the query's held-out machine diagnosis, CXR report,
    or labels.
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
    from PIL import Image
    import torch
    from tqdm.auto import tqdm
    import wfdb
    from transformers import pipeline

    ARTIFACT_ROOT = Path(os.environ.get(
        "ECG_CXR_BRIDGE_ARTIFACT_ROOT",
        "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    RAG_ROOT = ARTIFACT_ROOT / "rag_report_generation"
    config = json.loads((RAG_ROOT / "rag_config.json").read_text())
    contexts_path = RAG_ROOT / "contexts" / "rag_contexts.jsonl"
    contexts = [json.loads(line) for line in contexts_path.read_text().splitlines() if line.strip()]
    if not contexts:
        raise ValueError(f"No RAG contexts in {contexts_path}; run notebook 18")
    selected_models = {value.strip() for value in os.environ.get(
        "RAG_MODELS", ",".join(config["models"])).split(",") if value.strip()}
    selected_conditions = {value.strip() for value in os.environ.get(
        "RAG_CONDITIONS", ",".join(config["conditions"])).split(",") if value.strip()}
    unknown_models = selected_models - set(config["models"])
    unknown_conditions = selected_conditions - set(config["conditions"])
    if unknown_models or unknown_conditions:
        raise ValueError(f"Unknown selections: models={unknown_models}, conditions={unknown_conditions}")
    RETRY_ERRORS = os.environ.get("RAG_RETRY_ERRORS", "0").lower() in {"1", "true", "yes"}
    MAX_ROWS = int(os.environ.get("RAG_GENERATION_MAX_ROWS", "0"))
    if MAX_ROWS < 0:
        raise ValueError("RAG_GENERATION_MAX_ROWS cannot be negative")
    print({"models": sorted(selected_models), "conditions": sorted(selected_conditions),
           "records": len(contexts), "retry_errors": RETRY_ERRORS, "max_rows": MAX_ROWS})
    """),
    code(r"""
    ECG_IMAGE_ROOT = RAG_ROOT / "query_images" / "ecg"
    ECG_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    def waveform_stem(value):
        path = Path(value)
        return path.with_suffix("") if path.suffix.lower() in {".hea", ".dat"} else path

    def render_ecg(row):
        output = ECG_IMAGE_ROOT / f"{row['sample_id']}.png"
        if output.is_file():
            return output
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
            axis.grid(color="#efb5b5", linewidth=0.35, alpha=0.65)
            axis.set_yticks([])
        axes[-1].set_xlabel("Seconds")
        fig.suptitle("12-lead ECG waveform (first 10 seconds)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output

    def query_image(row):
        path = render_ecg(row) if row["modality"] == "ecg" else Path(row["query_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing query image: {path}")
        return path
    """),
    code(r"""
    SYSTEM_PROMPT = (
        "You are drafting a research-use clinical report. Describe only findings supported by the query "
        "image. Retrieved examples are from different patients and may be irrelevant. Never copy a finding "
        "unless it is supported by the query. Cross-modal context indicates possible clinical coexistence only, "
        "not diagnostic equivalence. Do not mention retrieval, examples, or these instructions in the report.")

    def compact_context(items, limit):
        blocks, used = [], 0
        for item in items:
            scope = "same-modality example" if item["scope"] == "same_modality" else "cross-modal related context"
            block = f"[{scope}; rank {item['rank']}; {item['modality'].upper()}] {item['text']}"
            remaining = limit - used
            if remaining <= 0:
                break
            block = block[:remaining]
            blocks.append(block); used += len(block)
        return "\n".join(blocks)

    def build_prompt(row):
        if row["modality"] == "cxr":
            task = (
                "Generate a concise chest radiograph report with exactly two headings: FINDINGS and IMPRESSION. "
                "State uncertainty when appropriate and avoid unsupported diagnoses.")
        else:
            task = (
                "Generate a concise machine-style 12-lead ECG report. Address rhythm, rate when visually "
                "estimable, conduction/interval abnormalities, axis, ST-T findings, and a final interpretation. "
                "Do not infer an ECG diagnosis solely from CXR context.")
        retrieved = compact_context(row["retrieved"], int(config["max_context_characters"]))
        if retrieved:
            task += "\n\nNon-authoritative retrieved context from training patients:\n" + retrieved
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
        value = str(value).strip()
        value = re.sub(r"<think>.*?</think>", "", value, flags=re.I | re.S).strip()
        return value

    def prompt_sha256(system, prompt):
        return hashlib.sha256((system + "\n" + prompt).encode()).hexdigest()
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
            (row["modality"], row["condition"], str(row["sample_id"]),
             row.get("cohort_signature"), row.get("generation_protocol_version")): row
            for row in existing if row.get("status") == "ok" or not RETRY_ERRORS
        }
        pending = [row for row in eligible
                   if (row["modality"], row["condition"], str(row["sample_id"]),
                       row.get("cohort_signature"), config["generation_protocol_version"]) not in completed]
        print(model_key, {"existing": len(existing), "pending": len(pending), "model_id": model_id})
        if not pending:
            continue
        try:
            pipe = pipeline(
                "image-text-to-text", model=model_id, device_map="auto",
                model_kwargs={"cache_dir": config.get("hf_cache"), "torch_dtype": "auto"})
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {model_id}. Update Transformers; for MedGemma accept the Hugging Face terms "
                f"and authenticate. Original error: {exc!r}") from exc

        with output_path.open("a") as stream:
            for row in tqdm(pending, desc=model_key):
                prompt = build_prompt(row)
                record = {
                    "model_key": model_key, "model_id": model_id,
                    "sample_id": str(row["sample_id"]), "modality": row["modality"],
                    "condition": row["condition"], "cohort_signature": row["cohort_signature"],
                    "generation_protocol_version": config["generation_protocol_version"],
                    "prompt_sha256": prompt_sha256(SYSTEM_PROMPT, prompt),
                    "retrieved_sample_ids": [item["sample_id"] for item in row["retrieved"]],
                }
                try:
                    image_path = query_image(row)
                    messages = [
                        {"role": "user", "content": [
                            {"type": "image", "url": str(image_path)},
                            # A single user turn is compatible with both Gemma and Qwen chat templates.
                            {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + prompt},
                        ]},
                    ]
                    template_kwargs = {"enable_thinking": False} if model_key == "qwen35_4b" else {}
                    result = pipe(
                        text=messages, max_new_tokens=int(config["max_new_tokens"]),
                        do_sample=False, return_full_text=False, **template_kwargs)
                    generated = extract_generated_text(result)
                    if not generated:
                        raise ValueError("Empty generated report")
                    record.update({"status": "ok", "generated_report": generated, "error": None})
                except Exception as exc:
                    record.update({"status": "error", "generated_report": "", "error": repr(exc)})
                stream.write(json.dumps(record) + "\n"); stream.flush()
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("Generation files:", sorted(str(path) for path in generation_root.glob("*.jsonl")))
    """),
])


write("20_evaluate_rag_report_generation.ipynb", [
    markdown(r"""
    # 20 - Evaluate RAG report generation

    This notebook compares `zero_shot`, `clip_rag`, `codebind_rag`, and `combined_rag` separately for
    MedGemma and Qwen3.5 on ECG and CXR report generation.

    **Clinical-content metrics** use a fixed assertion-aware label ontology. Fabrication is the fraction
    of predicted positive labels absent from the reference positives; omission is the fraction of
    reference positives not generated. Both study-macro rates and label-micro aggregate rates are saved.
    CXR references are top-level CheXpert labels. ECG references are
    deterministic concepts extracted from the held-out machine diagnosis. Negated and uncertain generated
    findings are never counted as predicted positives.

    **Multi-label metrics:** macro F1, micro F1, and Hamming accuracy.  
    **Text metrics:** CIDEr, BERTScore F1, ROUGE-1/2/L F1.

    These automated metrics do not establish clinical safety. A blinded clinician review remains necessary.
    """),
    code(r"""
    from pathlib import Path
    import json
    import os
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    from bert_score import score as bert_score
    from pycocoevalcap.cider.cider import Cider
    from rouge_score import rouge_scorer
    from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support

    ARTIFACT_ROOT = Path(os.environ.get(
        "ECG_CXR_BRIDGE_ARTIFACT_ROOT",
        "/data/liangz2/ecg_analysis/symile/ecg_cxr_bridge_artifacts"))
    RAG_ROOT = ARTIFACT_ROOT / "rag_report_generation"
    REPORT_ROOT = RAG_ROOT / "reports"
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    config = json.loads((RAG_ROOT / "rag_config.json").read_text())
    ALLOW_INCOMPLETE = os.environ.get("RAG_ALLOW_INCOMPLETE", "0").lower() in {"1", "true", "yes"}
    BOOTSTRAPS = int(os.environ.get("RAG_EVAL_BOOTSTRAPS", "500"))
    BERT_MODEL = os.environ.get("RAG_BERTSCORE_MODEL", "microsoft/deberta-xlarge-mnli")
    BERT_BATCH = int(os.environ.get("RAG_BERTSCORE_BATCH_SIZE", "32"))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    manifests = {m: read_jsonl(ARTIFACT_ROOT / "external" / m / "manifest.jsonl")
                 for m in ["ecg", "cxr"]}
    concept_rows = {m: read_jsonl(ARTIFACT_ROOT / "external" / m / "concepts.jsonl")
                    for m in ["ecg", "cxr"]}
    manifest_lookup = {m: {str(row["sample_id"]): row for row in values}
                       for m, values in manifests.items()}
    concept_lookup = {m: {str(row["sample_id"]): row for row in values}
                      for m, values in concept_rows.items()}
    contexts = read_jsonl(RAG_ROOT / "contexts" / "rag_contexts.jsonl")
    if not contexts:
        raise ValueError("No retrieval contexts; run notebook 18")
    """),
    code(r"""
    ECG_RULES = {
        "normal_ecg": [r"normal ecg", r"normal sinus rhythm"],
        "atrial_fibrillation": [r"atrial fibrillation", r"\ba[- ]?fib\b"],
        "atrial_flutter": [r"atrial flutter"], "tachycardia": [r"tachycard"],
        "bradycardia": [r"bradycard"],
        "conduction_abnormality": [r"bundle branch block", r"av block", r"heart block", r"intraventricular conduction"],
        "ventricular_hypertrophy": [r"ventricular hypertrophy"],
        "ischemia": [r"ischemi", r"st[- ]t abnormal", r"st depression", r"st elevation"],
        "infarction": [r"infarct"], "paced_rhythm": [r"paced rhythm", r"pacemaker"],
        "low_voltage": [r"low voltage"], "qt_abnormality": [r"prolonged qt", r"short qt"],
    }
    CXR_RULES = {
        "atelectasis": [r"atelecta"], "cardiomegaly": [r"cardiomegal"],
        "consolidation": [r"consolidat"], "edema": [r"\bedema\b", r"vascular congestion"],
        "enlarged_cardiomediastinum": [r"enlarged cardiomediast"], "fracture": [r"fracture"],
        "lung_lesion": [r"lung lesion", r"pulmonary nodule", r"lung mass"],
        "lung_opacity": [r"lung opacity", r"airspace opacity", r"pulmonary opacity"],
        "no_finding": [r"no acute cardiopulmonary", r"no finding", r"normal chest"],
        "pleural_effusion": [r"pleural effusion"], "pleural_other": [r"pleural thickening", r"pleural abnormal"],
        "pneumonia": [r"pneumonia"], "pneumothorax": [r"pneumothorax"],
        "support_device": [r"support device", r"pacemaker", r"central line", r"endotracheal tube", r"enteric tube"],
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
    LABELS = {"ecg": sorted(ECG_RULES), "cxr": sorted(set(CHEXPERT_MAP.values()))}
    NEGATED_BEFORE = re.compile(
        r"(?:\bno\b|\bwithout\b|\bnegative for\b|\babsence of\b|\bfree of\b|\bdenies\b)"
        r"(?:\W+\w+){0,8}\W*$", re.I)
    NEGATED_AFTER = re.compile(r"^\W*(?:is |are |was |were )?(?:not|absent)(?:\W|$)", re.I)
    UNCERTAIN_BEFORE = re.compile(
        r"(?:\bpossible\b|\bpossibly\b|\bprobable\b|\blikely\b|\bmay\b|\bmight\b|"
        r"\bcannot exclude\b|\bcannot rule out\b|\bconcerning for\b|\bsuspicious for\b|\bquestion of\b)"
        r"(?:\W+\w+){0,8}\W*$", re.I)
    UNCERTAIN_AFTER = re.compile(r"^\W*(?:is |are )?(?:possible|probable|suspected)(?:\W|$)", re.I)

    def assertion(text, match, concept):
        before = text[max(0, match.start() - 120):match.start()]
        before = re.split(r"[.;\n]|\b(?:but|however|although|yet)\b", before, flags=re.I)[-1]
        after = text[match.end():match.end() + 50]
        if NEGATED_BEFORE.search(before) or NEGATED_AFTER.search(after):
            return "absent"
        if UNCERTAIN_BEFORE.search(before) or UNCERTAIN_AFTER.search(after):
            return "uncertain"
        return "present"

    def extract_present(text, rules):
        value = " ".join(str(text).split()).lower()
        present = set()
        for concept, patterns in rules.items():
            for pattern in patterns:
                for match in re.finditer(pattern, value):
                    if assertion(value, match, concept) == "present":
                        present.add(concept)
        return present

    def reference_labels(modality, sample_id):
        if modality == "cxr":
            labels = manifest_lookup[modality][sample_id].get("labels", [])
            return {CHEXPERT_MAP[str(label).strip().lower()] for label in labels
                    if str(label).strip().lower() in CHEXPERT_MAP}
        values = concept_lookup[modality][sample_id].get("concepts", [])
        return {value.split(":", 1)[1] for value in values
                if value.startswith("present:") and value.split(":", 1)[1] in LABELS["ecg"]}

    def enforce_normal_exclusivity(modality, labels):
        labels = set(labels)
        normal = "normal_ecg" if modality == "ecg" else "no_finding"
        if normal in labels and len(labels) > 1:
            labels.remove(normal)
        return labels
    """),
    code(r"""
    generation_rows = []
    for model_key in config["models"]:
        path = RAG_ROOT / "generations" / f"{model_key}.jsonl"
        if path.is_file():
            generation_rows.extend(read_jsonl(path))
    active_signatures = {row["cohort_signature"] for row in contexts}
    if len(active_signatures) != 1:
        raise ValueError(f"Expected one active cohort signature; found {active_signatures}")
    generation_rows = [
        row for row in generation_rows
        if row.get("status") == "ok"
        and row.get("cohort_signature") in active_signatures
        and row.get("generation_protocol_version") == config["generation_protocol_version"]
    ]
    if not generation_rows:
        raise ValueError("No successful generations; run notebook 19")
    keys = [(row["model_key"], row["modality"], row["condition"], str(row["sample_id"]))
            for row in generation_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate successful generation keys; remove duplicate JSONL records")

    context_keys = {(row["modality"], row["condition"], str(row["sample_id"])) for row in contexts}
    expected_generation_keys = {
        (model, modality, condition, sample_id)
        for model in config["models"]
        for modality, condition, sample_id in context_keys}
    generation_keys = set(keys)
    missing_records = sorted(expected_generation_keys - generation_keys)
    unexpected_records = sorted(generation_keys - expected_generation_keys)
    counts = pd.DataFrame(generation_rows).groupby(
        ["model_key", "modality", "condition"]).size().to_dict()
    cohort_counts = pd.DataFrame(contexts).groupby(["modality", "condition"]).size().to_dict()
    expected_counts = {
        (model, modality, condition): count
        for model in config["models"]
        for (modality, condition), count in cohort_counts.items()}
    unequal = {key: {"observed": counts.get(key, 0), "expected": expected}
               for key, expected in expected_counts.items() if counts.get(key, 0) != expected}
    if (missing_records or unexpected_records or unequal) and not ALLOW_INCOMPLETE:
        raise RuntimeError(
            f"Incomplete generation matrix. Missing records={missing_records[:20]}; "
            f"unexpected records={unexpected_records[:20]}; counts={unequal}. "
            "Finish notebook 19 or set RAG_ALLOW_INCOMPLETE=1 for a smoke analysis.")
    print({"successful_generations": len(generation_rows), "groups": counts,
           "expected_groups": expected_counts, "allow_incomplete": ALLOW_INCOMPLETE})
    """),
    code(r"""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    detailed = []
    for row in generation_rows:
        modality, sample_id = row["modality"], str(row["sample_id"])
        manifest_row = manifest_lookup[modality][sample_id]
        reference_text = manifest_row["machine_diagnosis" if modality == "ecg" else "report_text"]
        truth = enforce_normal_exclusivity(modality, reference_labels(modality, sample_id))
        predicted = enforce_normal_exclusivity(
            modality, extract_present(row["generated_report"], ECG_RULES if modality == "ecg" else CXR_RULES))
        vocabulary = LABELS[modality]
        y_true = [int(label in truth) for label in vocabulary]
        y_pred = [int(label in predicted) for label in vocabulary]
        tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        rouge = scorer.score(str(reference_text), row["generated_report"])
        detailed.append({
            **row, "reference_text": reference_text,
            "truth_labels": sorted(truth), "predicted_labels": sorted(predicted),
            "y_true": y_true, "y_pred": y_pred, "tp": tp, "fp": fp, "fn": fn,
            "fabrication_rate": fp / max(len(predicted), 1),
            "omission_rate": fn / max(len(truth), 1),
            "sample_hamming_accuracy": 1 - np.mean(np.not_equal(y_true, y_pred)),
            "rouge1_f1": rouge["rouge1"].fmeasure,
            "rouge2_f1": rouge["rouge2"].fmeasure,
            "rougeL_f1": rouge["rougeL"].fmeasure,
        })

    candidates = [row["generated_report"] for row in detailed]
    references = [str(row["reference_text"]) for row in detailed]
    _, _, bert_f1 = bert_score(
        candidates, references, model_type=BERT_MODEL, lang="en", batch_size=BERT_BATCH,
        device=DEVICE, verbose=True, rescale_with_baseline=False)
    for row, value in zip(detailed, bert_f1.cpu().numpy()):
        row["bertscore_f1"] = float(value)
    print("Per-sample text and label metrics computed:", len(detailed))
    """),
    code(r"""
    def cider_values(rows):
        references = {i: [str(row["reference_text"])] for i, row in enumerate(rows)}
        candidates = {i: [str(row["generated_report"])] for i, row in enumerate(rows)}
        corpus, per_sample = Cider().compute_score(references, candidates)
        return float(corpus), np.asarray(per_sample, dtype="float64")

    summary_rows = []
    for keys, group in pd.DataFrame(detailed).groupby(["model_key", "modality", "condition"], sort=True):
        records = group.to_dict("records")
        y_true = np.asarray(group.y_true.tolist(), dtype="int8")
        y_pred = np.asarray(group.y_pred.tolist(), dtype="int8")
        cider, per_cider = cider_values(records)
        for record, value in zip(records, per_cider):
            record["cider"] = float(value)
        # Propagate CIDEr values back to the canonical detailed records.
        cider_map = {str(record["sample_id"]): float(value) for record, value in zip(records, per_cider)}
        for record in detailed:
            if (record["model_key"], record["modality"], record["condition"]) == keys:
                record["cider"] = cider_map[str(record["sample_id"])]
        summary_rows.append({
            "model_key": keys[0], "modality": keys[1], "condition": keys[2], "n": len(group),
            "fabrication_rate": float(group.fabrication_rate.mean()),
            "fabrication_rate_micro": float(group.fp.sum() / max(group.y_pred.map(sum).sum(), 1)),
            "omission_rate": float(group.omission_rate.mean()),
            "omission_rate_micro": float(group.fn.sum() / max(group.y_true.map(sum).sum(), 1)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "hamming_accuracy": float(1 - hamming_loss(y_true, y_pred)),
            "cider": cider, "bertscore_f1": float(group.bertscore_f1.mean()),
            "rouge1_f1": float(group.rouge1_f1.mean()),
            "rouge2_f1": float(group.rouge2_f1.mean()),
            "rougeL_f1": float(group.rougeL_f1.mean()),
            "truth_positive_labels": int(y_true.sum()), "predicted_positive_labels": int(y_pred.sum()),
        })
    summary = pd.DataFrame(summary_rows)
    detailed_frame = pd.DataFrame(detailed)
    detailed_frame.to_csv(REPORT_ROOT / "rag_report_metrics_detailed.csv", index=False)
    summary.to_csv(REPORT_ROOT / "rag_report_metrics_summary.csv", index=False)

    per_label_rows = []
    for keys, group in detailed_frame.groupby(["model_key", "modality", "condition"], sort=True):
        y_true = np.asarray(group.y_true.tolist(), dtype="int8")
        y_pred = np.asarray(group.y_pred.tolist(), dtype="int8")
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0)
        for label, p, r, score, count in zip(LABELS[keys[1]], precision, recall, f1, support):
            per_label_rows.append({
                "model_key": keys[0], "modality": keys[1], "condition": keys[2],
                "label": label, "support": int(count), "precision": float(p),
                "recall": float(r), "f1": float(score),
            })
    pd.DataFrame(per_label_rows).to_csv(REPORT_ROOT / "rag_report_per_label_metrics.csv", index=False)

    prevalence_rows = []
    active_queries = {(row["modality"], str(row["sample_id"])) for row in contexts}
    for modality in ["ecg", "cxr"]:
        ids = sorted(sample_id for item_modality, sample_id in active_queries if item_modality == modality)
        for label in LABELS[modality]:
            positives = sum(label in enforce_normal_exclusivity(
                modality, reference_labels(modality, sample_id)) for sample_id in ids)
            prevalence_rows.append({
                "modality": modality, "label": label, "n_queries": len(ids),
                "positive_count": positives, "prevalence": positives / max(len(ids), 1),
            })
    pd.DataFrame(prevalence_rows).to_csv(REPORT_ROOT / "rag_cohort_label_prevalence.csv", index=False)
    display(summary.sort_values(["model_key", "modality", "condition"]))
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
            "cider": float(group.cider.mean()),
            "bertscore_f1": float(group.bertscore_f1.mean()),
            "rouge1_f1": float(group.rouge1_f1.mean()),
            "rouge2_f1": float(group.rouge2_f1.mean()),
            "rougeL_f1": float(group.rougeL_f1.mean()),
        }

    rng = np.random.default_rng(int(config["seed"]) + 91)
    bootstrap_rows = []
    if BOOTSTRAPS > 0:
        for keys, group in detailed_frame.groupby(["model_key", "modality", "condition"], sort=True):
            values = {metric: [] for metric in grouped_metrics(group)}
            for _ in range(BOOTSTRAPS):
                sampled = group.iloc[rng.integers(0, len(group), size=len(group))]
                result = grouped_metrics(sampled)
                for metric, value in result.items():
                    values[metric].append(value)
            for metric, samples in values.items():
                bootstrap_rows.append({
                    "model_key": keys[0], "modality": keys[1], "condition": keys[2],
                    "metric": metric, "bootstrap_samples": BOOTSTRAPS,
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                })
    pd.DataFrame(bootstrap_rows).to_csv(REPORT_ROOT / "rag_report_metric_confidence_intervals.csv", index=False)

    metric_columns = [
        "fabrication_rate", "fabrication_rate_micro", "omission_rate", "omission_rate_micro",
        "macro_f1", "micro_f1", "hamming_accuracy",
        "cider", "bertscore_f1", "rouge1_f1", "rouge2_f1", "rougeL_f1",
    ]
    baseline = summary[summary.condition.eq("zero_shot")].set_index(["model_key", "modality"])
    delta_rows = []
    for row in summary.itertuples():
        base = baseline.loc[(row.model_key, row.modality)]
        delta_rows.append({
            "model_key": row.model_key, "modality": row.modality, "condition": row.condition,
            **{f"delta_{metric}_vs_zero_shot": float(getattr(row, metric) - base[metric])
               for metric in metric_columns},
        })
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(REPORT_ROOT / "rag_report_deltas_vs_zero_shot.csv", index=False)
    display(deltas[deltas.condition.ne("zero_shot")])
    """),
    code(r"""
    # Paired comparisons use the identical sample IDs for both conditions. Positive values always
    # mean that condition_a is better than condition_b (lower fabrication/omission; higher otherwise).
    comparison_pairs = [
        ("clip_rag", "zero_shot"), ("codebind_rag", "zero_shot"),
        ("combined_rag", "zero_shot"), ("codebind_rag", "clip_rag"),
        ("combined_rag", "clip_rag"), ("combined_rag", "codebind_rag"),
    ]
    lower_is_better = {
        "fabrication_rate", "fabrication_rate_micro", "omission_rate", "omission_rate_micro"}
    paired_rows = []
    for (model_key, modality), outer in detailed_frame.groupby(["model_key", "modality"], sort=True):
        by_condition = {condition: frame.sort_values("sample_id").reset_index(drop=True)
                        for condition, frame in outer.groupby("condition")}
        for condition_a, condition_b in comparison_pairs:
            if condition_a not in by_condition or condition_b not in by_condition:
                continue
            left, right = by_condition[condition_a], by_condition[condition_b]
            common_ids = sorted(set(left.sample_id) & set(right.sample_id))
            if not common_ids:
                continue
            left = left.set_index("sample_id").loc[common_ids].reset_index()
            right = right.set_index("sample_id").loc[common_ids].reset_index()
            if not np.array_equal(left.sample_id.to_numpy(), right.sample_id.to_numpy()):
                raise AssertionError("Paired comparison sample order mismatch")
            point_a, point_b = grouped_metrics(left), grouped_metrics(right)
            samples = {metric: [] for metric in point_a}
            if BOOTSTRAPS > 0:
                for _ in range(BOOTSTRAPS):
                    positions = rng.integers(0, len(common_ids), size=len(common_ids))
                    metric_a = grouped_metrics(left.iloc[positions])
                    metric_b = grouped_metrics(right.iloc[positions])
                    for metric in samples:
                        difference = (metric_b[metric] - metric_a[metric]
                                      if metric in lower_is_better
                                      else metric_a[metric] - metric_b[metric])
                        samples[metric].append(difference)
            for metric in point_a:
                improvement = (point_b[metric] - point_a[metric]
                               if metric in lower_is_better
                               else point_a[metric] - point_b[metric])
                values = samples[metric]
                paired_rows.append({
                    "model_key": model_key, "modality": modality,
                    "condition_a": condition_a, "condition_b": condition_b,
                    "metric": metric, "n_paired": len(common_ids),
                    "value_a": point_a[metric], "value_b": point_b[metric],
                    "improvement_a_over_b": float(improvement),
                    "ci_low": float(np.quantile(values, 0.025)) if values else np.nan,
                    "ci_high": float(np.quantile(values, 0.975)) if values else np.nan,
                    "bootstrap_probability_improvement": float(np.mean(np.asarray(values) > 0))
                    if values else np.nan,
                })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(REPORT_ROOT / "rag_report_paired_comparisons.csv", index=False)
    display(paired[(paired.condition_a == "codebind_rag") & (paired.condition_b == "clip_rag")])
    """),
    code(r"""
    # Retrieval-context quality is evaluated only after generation and never exposed to the prompts.
    RELATIONS = {
        ("ecg", "ventricular_hypertrophy"): {"cardiomegaly", "enlarged_cardiomediastinum"},
        ("ecg", "paced_rhythm"): {"support_device"},
        ("cxr", "cardiomegaly"): {"ventricular_hypertrophy"},
        ("cxr", "enlarged_cardiomediastinum"): {"ventricular_hypertrophy"},
        ("cxr", "support_device"): {"paced_rhythm"},
    }

    context_quality = []
    for row in contexts:
        if row["condition"] == "zero_shot":
            continue
        modality, sample_id = row["modality"], str(row["sample_id"])
        query_labels = enforce_normal_exclusivity(modality, reference_labels(modality, sample_id))
        expected_cross = set().union(*(RELATIONS.get((modality, label), set()) for label in query_labels))
        same_flags, cross_flags = [], []
        for item in row["retrieved"]:
            source_labels = enforce_normal_exclusivity(
                item["modality"], reference_labels(item["modality"], str(item["sample_id"])))
            if item["scope"] == "same_modality":
                same_flags.append(bool(query_labels & source_labels))
            elif expected_cross:
                cross_flags.append(bool(expected_cross & source_labels))
        context_quality.append({
            "modality": modality, "condition": row["condition"], "sample_id": sample_id,
            "same_label_hit_at_k": float(any(same_flags)) if same_flags else np.nan,
            "same_label_overlap_fraction": float(np.mean(same_flags)) if same_flags else np.nan,
            "cross_relation_hit_at_k": float(any(cross_flags)) if cross_flags else np.nan,
            "cross_relation_overlap_fraction": float(np.mean(cross_flags)) if cross_flags else np.nan,
            "cross_relation_evaluable": bool(expected_cross),
        })
    context_quality = pd.DataFrame(context_quality)
    context_quality.to_csv(REPORT_ROOT / "rag_retrieval_context_quality_detailed.csv", index=False)
    context_quality_summary = context_quality.groupby(["modality", "condition"], as_index=False).agg(
        n_queries=("sample_id", "size"),
        same_label_hit_at_k=("same_label_hit_at_k", "mean"),
        same_label_overlap_fraction=("same_label_overlap_fraction", "mean"),
        cross_relation_hit_at_k=("cross_relation_hit_at_k", "mean"),
        cross_relation_overlap_fraction=("cross_relation_overlap_fraction", "mean"),
        cross_relation_evaluable=("cross_relation_evaluable", "sum"),
    )
    context_quality_summary.to_csv(REPORT_ROOT / "rag_retrieval_context_quality_summary.csv", index=False)
    display(context_quality_summary)
    """),
    code(r"""
    sns.set_theme(style="whitegrid", context="talk")
    order = config["conditions"]
    palette = ["#7f8c8d", "#4c78a8", "#e68624", "#54a24b"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=False)
    for axis, (model, modality) in zip(axes.flat, [
        (model, modality) for model in config["models"] for modality in ["ecg", "cxr"]]):
        frame = summary[(summary.model_key == model) & (summary.modality == modality)]
        melted = frame.melt(id_vars="condition", value_vars=["macro_f1", "micro_f1", "hamming_accuracy"],
                            var_name="metric", value_name="value")
        sns.barplot(data=melted, x="condition", y="value", hue="metric", order=order, ax=axis)
        axis.set_title(f"{model} | {modality.upper()}"); axis.tick_params(axis="x", rotation=25)
        axis.set_ylim(0, 1); axis.set_xlabel("")
    fig.suptitle("RAG report generation: multi-label classification")
    fig.tight_layout(); fig.savefig(REPORT_ROOT / "rag_multilabel_metrics.png", dpi=180, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    for axis, (model, modality) in zip(axes.flat, [
        (model, modality) for model in config["models"] for modality in ["ecg", "cxr"]]):
        frame = summary[(summary.model_key == model) & (summary.modality == modality)]
        melted = frame.melt(id_vars="condition", value_vars=["fabrication_rate", "omission_rate"],
                            var_name="metric", value_name="value")
        sns.barplot(data=melted, x="condition", y="value", hue="metric", order=order, ax=axis)
        axis.set_title(f"{model} | {modality.upper()} (lower is better)")
        axis.tick_params(axis="x", rotation=25); axis.set_ylim(0, 1); axis.set_xlabel("")
    fig.suptitle("RAG report generation: fabrication and omission")
    fig.tight_layout(); fig.savefig(REPORT_ROOT / "rag_fabrication_omission.png", dpi=180, bbox_inches="tight")
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=False)
    for axis, (model, modality) in zip(axes.flat, [
        (model, modality) for model in config["models"] for modality in ["ecg", "cxr"]]):
        frame = summary[(summary.model_key == model) & (summary.modality == modality)]
        melted = frame.melt(id_vars="condition", value_vars=["cider", "bertscore_f1", "rougeL_f1"],
                            var_name="metric", value_name="value")
        sns.barplot(data=melted, x="condition", y="value", hue="metric", order=order, ax=axis)
        axis.set_title(f"{model} | {modality.upper()}"); axis.tick_params(axis="x", rotation=25)
        axis.set_xlabel("")
    fig.suptitle("RAG report generation: text similarity")
    fig.tight_layout(); fig.savefig(REPORT_ROOT / "rag_text_similarity.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("Saved evaluation outputs to", REPORT_ROOT)
    """),
    markdown(r"""
    ## Interpretation rules

    - Prefer a RAG method only if it improves label F1 and text metrics **without increasing fabrication**.
    - Compare within the same model and modality; do not average ECG and CXR into one headline score.
    - Treat the CLIP-versus-CodeBind comparison as paired because every condition uses the same fixed query cohort.
    - In `rag_report_paired_comparisons.csv`, positive improvement means condition A is better for every
      metric; the direction is reversed internally for fabrication and omission because lower is better.
    - Retrieval-context quality is a mechanistic diagnostic, not a generation score. Cross-modal relation
      hits use a small prespecified relation map and should not be interpreted as diagnostic equivalence.
    - A high text-similarity score with worse fabrication is not a clinical improvement.
    - If `combined_rag` helps, inspect whether benefit comes from complementary neighbors or merely more context.
    - Report label prevalence and confidence intervals, especially when macro F1 is dominated by rare labels.
    - The ECG ground truth is a machine interpretation, not an adjudicated cardiologist report.
    - Automated label extraction can miss synonyms and assertion scope; manually audit a stratified sample before publication.
    """),
])
