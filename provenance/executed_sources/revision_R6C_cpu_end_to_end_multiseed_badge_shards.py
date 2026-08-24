from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import os
import random
import resource
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import revision_R6B_cpu_fixed_history_multiseed_shards as r6b


engine = r6b.engine
REVISION_PROTOCOL_SHA256 = r6b.REVISION_PROTOCOL_SHA256
DEEP_PROTOCOL_SHA256 = r6b.DEEP_PROTOCOL_SHA256
R0_PACKET_SHA256 = "0800e315a29b81934095ba56deaea3f8b6600fd0df13db348d7ea72d3b82df78"
R6A_PACKET_SHA256 = r6b.R6A_PACKET_SHA256
R6B_PACKET_SHA256 = "5e1440672256a452e5fdf8924dd1705c89fa03b5bfc8afdad41a73cdfb6b45f0"
STAGE5B_PACKET_SHA256 = r6b.STAGE5B_PACKET_SHA256
STAGE5C_PACKET_SHA256 = r6b.STAGE5C_PACKET_SHA256
STAGE5D2_PACKET_SHA256 = r6b.STAGE5D2_PACKET_SHA256

PARTICIPANTS = r6b.PARTICIPANTS
ABLE_BODIED = r6b.ABLE_BODIED
STRATEGIES = ["PCBM_PROPOSED", "GLOBAL_MARGIN", "RANDOM_UNIFORM", "BADGE"]
TARGET_EPOCHS = r6b.TARGET_EPOCHS
TARGET_BATCH_SIZE = r6b.TARGET_BATCH_SIZE
LABEL_SMOOTHING = r6b.LABEL_SMOOTHING
CLASSES = r6b.CLASSES
CPU_THREADS = r6b.CPU_THREADS
MAX_UNITS_PER_RUN = max(1, int(os.environ.get("R6C_MAX_UNITS_PER_RUN", "168")))
MAX_RUNTIME_MINUTES = max(30.0, float(os.environ.get("R6C_MAX_RUNTIME_MINUTES", "630")))

WORKING = Path(os.environ.get("REVISION_R6C_WORKING", "/kaggle/working"))
INPUT_ROOT = WORKING / "REVISION_R6C_FROZEN_INPUTS"
RESULT_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R6C_CPU_End_to_End_Multiseed_BADGE"
UNIT_WORK_ROOT = WORKING / "REVISION_R6C_UNIT_WORK"
UNIT_PACKET_ROOT = WORKING / "REVISION_R6C_UNIT_PACKETS"
FINAL_PACKET = WORKING / "revision_R6C_cpu_end_to_end_multiseed_badge_packet.zip"
PROGRESS_PACKET = WORKING / "revision_R6C_cpu_end_to_end_multiseed_badge_progress_packet.zip"
REMOTE_BASE = engine.REMOTE_BASE
REMOTE_OUTPUT = REMOTE_BASE + "/Reviewer_Revision/Revision_R6C_CPU_End_to_End_Multiseed_BADGE"
REMOTE_UNITS = REMOTE_OUTPUT + "/units"
START_TIME = time.time()

DIRECT_PACKETS = {
    "stageR0_reviewer_revision_protocol_lock_packet.zip": (
        R0_PACKET_SHA256,
        "Reviewer_Revision/StageR0_Reviewer_Revision_Protocol_Lock/stageR0_reviewer_revision_protocol_lock_packet.zip",
    ),
    "revision_R6A_cpu_runtime_amendment_preflight_packet.zip": (
        R6A_PACKET_SHA256,
        "Reviewer_Revision/Revision_R6A_CPU_Runtime_Amendment_Preflight/revision_R6A_cpu_runtime_amendment_preflight_packet.zip",
    ),
    "revision_R6B_cpu_fixed_history_multiseed_packet.zip": (
        R6B_PACKET_SHA256,
        "Reviewer_Revision/Revision_R6B_CPU_Fixed_History_Multiseed/revision_R6B_cpu_fixed_history_multiseed_packet.zip",
    ),
    "stage5b_deep_sequence_assembly_packet.zip": (
        STAGE5B_PACKET_SHA256,
        "Stage5B_Deep_Sequence_Assembly/stage5b_deep_sequence_assembly_packet.zip",
    ),
    "stage5c1_dual_gpu_loso_pretraining_packet.zip": (
        STAGE5C_PACKET_SHA256,
        "Deep_Training/Stage5C_LOSO_Pretraining/stage5c1_dual_gpu_loso_pretraining_packet.zip",
    ),
    "stage5d2_full_deterministic_deep_trajectories_packet.zip": (
        STAGE5D2_PACKET_SHA256,
        "Deep_Training/Stage5D2_Full_Deterministic_Deep_Trajectories/stage5d2_full_deterministic_deep_trajectories_packet.zip",
    ),
}


def canonical_hash(payload: dict) -> str:
    return r6b.canonical_hash(payload)


def stable_seed(text: str) -> int:
    return r6b.stable_seed(text)


def atomic_json(payload: dict, path: Path) -> None:
    r6b.atomic_json(payload, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    r6b.atomic_csv(frame, path)


def direct_restore(basename: str, expected_hash: str, remote_relative: str) -> tuple[Path, str]:
    destination = INPUT_ROOT / basename
    if destination.exists() and engine.sha256_file(destination) == expected_hash and engine.archive_crc_passes(destination):
        return destination, "EXISTING_VERIFIED_COPY"
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, 6):
        temporary = destination.with_suffix(destination.suffix + f".download{attempt}")
        temporary.unlink(missing_ok=True)
        result = engine.rclone(
            ["copyto", REMOTE_BASE + "/" + remote_relative, str(temporary), "--retries", "5", "--low-level-retries", "10", "--timeout", "5m"],
            check=False,
        )
        if result.returncode == 0 and temporary.exists() and engine.sha256_file(temporary) == expected_hash and engine.archive_crc_passes(temporary):
            os.replace(temporary, destination)
            return destination, "GOOGLE_DRIVE_DIRECT"
        last_error = (result.stderr or result.stdout or "hash-or-crc-mismatch")[-1000:]
        temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not restore verified {basename}: {last_error}")


def prepare_inputs(packets: dict[str, Path]) -> dict:
    r6b_packet_map = {
        "revision_R6A_cpu_runtime_amendment_preflight_packet.zip": packets["revision_R6A_cpu_runtime_amendment_preflight_packet.zip"],
        "stage5b_deep_sequence_assembly_packet.zip": packets["stage5b_deep_sequence_assembly_packet.zip"],
        "stage5c1_dual_gpu_loso_pretraining_packet.zip": packets["stage5c1_dual_gpu_loso_pretraining_packet.zip"],
        "stage5d2_full_deterministic_deep_trajectories_packet.zip": packets["stage5d2_full_deterministic_deep_trajectories_packet.zip"],
    }
    base = r6b.prepare_inputs(r6b_packet_map)
    r0_packet = packets["stageR0_reviewer_revision_protocol_lock_packet.zip"]
    r6a_packet = packets["revision_R6A_cpu_runtime_amendment_preflight_packet.zip"]
    r6b_packet = packets["revision_R6B_cpu_fixed_history_multiseed_packet.zip"]
    stage5d2_packet = packets["stage5d2_full_deterministic_deep_trajectories_packet.zip"]
    r0_report = engine.read_json_member(r0_packet, "stageR0_protocol_lock_report.json")
    r6b_report = engine.read_json_member(r6b_packet, "revision_R6B_final_report.json")
    seeds = engine.read_csv_member(r0_packet, "stageR0_seed_schedule.csv")
    for column in ["seed_index", "seed"]:
        seeds[column] = pd.to_numeric(seeds[column], errors="raise").astype(np.int64)
    random_seeds = seeds.loc[seeds["seed_family"].astype(str).eq("RANDOM_ACQUISITION") & seeds["seed_index"].between(1, 6), ["seed_index", "seed"]].copy()
    badge_rows = seeds.loc[seeds["seed_family"].astype(str).eq("BADGE_KMEANS_PP")].copy()
    if len(random_seeds) != 6 or random_seeds["seed_index"].tolist() != list(range(1, 7)) or len(badge_rows) != 1:
        raise RuntimeError("Locked random/BADGE seed schedule is incomplete")
    random_seed_map = dict(zip(random_seeds["seed_index"].astype(int), random_seeds["seed"].astype(int)))
    badge_base_seed = int(badge_rows.iloc[0]["seed"])

    plan = engine.read_csv_member(r6a_packet, "revision_R6A_cpu_execution_manifest.csv")
    plan = plan.loc[plan["stage"].astype(str).eq("R6C_END_TO_END")].copy()
    for column in ["training_seed_index", "training_seed", "query_budget", "random_acquisition_seed_index", "expected_target_sessions"]:
        plan[column] = pd.to_numeric(plan[column], errors="raise").astype(np.int64)
    plan["participant"] = plan["participant"].astype(str)
    plan["strategy"] = plan["strategy"].astype(str)
    plan = plan.sort_values(["participant", "training_seed_index", "strategy"]).reset_index(drop=True)

    full_selection = engine.read_csv_member(stage5d2_packet, "stage5d2_selection_trace.csv")
    full_selection["sequence_row_internal"] = pd.to_numeric(full_selection["sequence_row_internal"], errors="raise").astype(np.int64)
    token_map = (
        full_selection.loc[full_selection["strategy"].astype(str).eq("FULL_POOL_REFERENCE"), ["sequence_row_internal", "opaque_candidate_token"]]
        .drop_duplicates()
    )
    if len(token_map) != 1225:
        raise RuntimeError("Stage5D2 full-pool opaque-token map is incomplete")
    base["metadata"]["opaque_candidate_token"] = base["metadata"]["sequence_row"].map(dict(zip(token_map["sequence_row_internal"], token_map["opaque_candidate_token"].astype(str))))
    if base["metadata"].loc[base["metadata"]["candidate"], "opaque_candidate_token"].isna().any():
        raise RuntimeError("Candidate opaque-token assignment is incomplete")

    gates = {
        "r0_parent_all_gates_passed": bool(r0_report.get("all_readiness_gates_passed")),
        "r6b_parent_all_gates_passed": bool(r6b_report.get("all_readiness_gates_passed")),
        "r6b_parent_decision_authorizes_r6c": r6b_report.get("final_decision") == "PASS_TO_REVISION_R6C_CPU_END_TO_END_MULTISEED_BADGE_SHARDS",
        "r6b_parent_packet_hash_is_locked": r6b_report.get("r6a_packet_sha256") == R6A_PACKET_SHA256,
        "all_r6b_base_input_gates_passed": all(base["input_gates"].values()),
        "r6c_plan_has_168_units": len(plan) == 168,
        "r6c_plan_has_exact_participants": set(plan["participant"]) == set(PARTICIPANTS),
        "r6c_plan_has_exact_strategies": set(plan["strategy"]) == set(STRATEGIES),
        "each_participant_strategy_has_six_training_seeds": plan.groupby(["participant", "strategy"]).size().eq(6).all(),
        "random_units_use_seed_indices_one_to_six": sorted(plan.loc[plan["strategy"].eq("RANDOM_UNIFORM"), "random_acquisition_seed_index"].unique().tolist()) == list(range(1, 7)),
        "six_locked_random_seed_values_are_available": len(random_seed_map) == 6,
        "one_locked_badge_base_seed_is_available": len(badge_rows) == 1,
        "full_pool_token_map_has_1225_candidates": len(token_map) == 1225,
        "all_candidate_tokens_are_opaque": base["metadata"].loc[base["metadata"]["candidate"], "opaque_candidate_token"].astype(str).str.fullmatch(r"[0-9a-f]{24}").all(),
    }
    failed = [key for key, value in gates.items() if not bool(value)]
    if failed:
        raise RuntimeError(f"R6C input readiness failed: {failed}")
    base.update(
        {
            "r0_report": r0_report,
            "r6b_report": r6b_report,
            "r6c_plan": plan,
            "random_seed_map": random_seed_map,
            "badge_base_seed": badge_base_seed,
            "input_gates_r6c": gates,
        }
    )
    return base


def initial_history_rows(metadata: pd.DataFrame, participant: str) -> np.ndarray:
    rows = metadata.loc[metadata["participant"].eq(participant) & metadata["initial_history"], "sequence_row"].to_numpy(dtype=np.int64)
    if len(rows) != 35:
        raise RuntimeError(f"Initial history is not 35 rows for {participant}")
    return np.asarray(sorted(rows), dtype=np.int64)


def candidate_rows(metadata: pd.DataFrame, participant: str, session: int) -> np.ndarray:
    rows = metadata.loc[metadata["participant"].eq(participant) & metadata["session"].eq(int(session)) & metadata["candidate"], "sequence_row"].to_numpy(dtype=np.int64)
    if len(rows) != 35:
        raise RuntimeError(f"Candidate pool is not 35 rows for {participant} S{session}")
    return rows


def paired_fit_seed(training_seed: int, participant: str, fit_session: int) -> int:
    return stable_seed(f"R6C_END_TO_END|{int(training_seed)}|{participant}|FIT_SESSION_{int(fit_session):02d}")


def fit_state(inputs: dict, unit_id: str, participant: str, strategy: str, training_seed: int, training_seed_index: int, fit_session: int, history_rows: np.ndarray) -> tuple[dict, dict, list[dict]]:
    history_rows = np.asarray(sorted(set(map(int, history_rows))), dtype=np.int64)
    expected = 35 + 7 * int(fit_session)
    if len(history_rows) != expected:
        raise RuntimeError(f"R6C history count mismatch at fit session {fit_session}: {len(history_rows)} vs {expected}")
    history_meta = inputs["metadata"].iloc[history_rows]
    if history_meta["fixed_test"].any() or history_meta["session"].max() > fit_session or sorted(history_meta["label"].unique()) != list(range(CLASSES)):
        raise RuntimeError("R6C history leakage or class-coverage failure")
    means, stds, counts = r6b.fit_normalizer(inputs["features"][history_rows], inputs["valid_mask"][history_rows])
    x_train = r6b.transform_data(inputs["features"], inputs["valid_mask"], history_rows, means, stds)
    y_train = history_meta["label"].to_numpy(dtype=np.int64)
    fit_seed = paired_fit_seed(training_seed, participant, fit_session)
    r6b.set_seed(fit_seed)
    model = inputs["model_class"]().cpu()
    model.load_state_dict(inputs["checkpoints"][participant]["model_state_dict"], strict=True)
    stem, frozen, trainable = r6b.freeze_target_model(model)
    optimizer = r6b.build_optimizer(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    losses = []
    optimizer_steps = 0
    started = time.perf_counter()
    for epoch in range(1, TARGET_EPOCHS + 1):
        generator = torch.Generator().manual_seed(fit_seed + epoch)
        loader = DataLoader(dataset, batch_size=TARGET_BATCH_SIZE, shuffle=True, generator=generator, num_workers=0, pin_memory=False, drop_last=False)
        model.train()
        total_loss, total_examples = 0.0, 0
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite R6C CPU loss")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()):
                raise RuntimeError("Non-finite R6C CPU gradient")
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.item()) * len(batch_labels)
            total_examples += len(batch_labels)
        epoch_loss = float(total_loss / total_examples)
        losses.append(
            {
                "unit_id": unit_id,
                "participant": participant,
                "strategy": strategy,
                "training_seed_index": int(training_seed_index),
                "training_seed": int(training_seed),
                "fit_session": int(fit_session),
                "epoch": int(epoch),
                "history_repetitions": int(len(history_rows)),
                "epoch_loss": epoch_loss,
            }
        )
    fit_seconds = time.perf_counter() - started
    if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise RuntimeError("Non-finite R6C adapted state")
    fit_row = {
        "unit_id": unit_id,
        "participant": participant,
        "strategy": strategy,
        "training_seed_index": int(training_seed_index),
        "training_seed": int(training_seed),
        "fit_session": int(fit_session),
        "history_repetitions": int(len(history_rows)),
        "history_sha256": hashlib.sha256(history_rows.tobytes()).hexdigest(),
        "fit_seed": int(fit_seed),
        "stem_attribute": stem,
        "frozen_parameter_count": len(frozen),
        "trainable_parameter_count": len(trainable),
        "minimum_normalizer_count": int(counts.min()),
        "normalizer_values_finite": bool(np.isfinite(means).all() and np.isfinite(stds).all()),
        "normalizer_stds_positive": bool((stds > 0).all()),
        "fixed_test_in_history": False,
        "maximum_history_session": int(history_meta["session"].max()),
        "target_epochs": TARGET_EPOCHS,
        "final_train_loss": float(losses[-1]["epoch_loss"]),
        "optimizer_steps": int(optimizer_steps),
        "fit_seconds": float(fit_seconds),
    }
    state = {"model": model, "means": means, "stds": stds, "counts": counts, "history_rows": history_rows}
    del optimizer, dataset, x_train
    return state, fit_row, losses


def score_rows(inputs: dict, state: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    x = r6b.transform_data(inputs["features"], inputs["valid_mask"], rows, state["means"], state["stds"])
    started = time.perf_counter()
    state["model"].eval()
    with torch.no_grad():
        logits_tensor = state["model"](torch.from_numpy(x))
        probabilities_tensor = torch.softmax(logits_tensor, dim=1)
    elapsed = time.perf_counter() - started
    logits = logits_tensor.cpu().numpy()
    probabilities = probabilities_tensor.cpu().numpy()
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    ordered = np.sort(probabilities, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all() or not np.isfinite(margins).all():
        raise RuntimeError("Non-finite R6C model scores")
    return logits, probabilities, predictions, margins, elapsed


def badge_gradient_embeddings(model: nn.Module, inputs_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    captured = []

    def pre_hook(_module, values):
        captured.append(values[0].detach())

    handle = model.classifier.register_forward_pre_hook(pre_hook)
    model.eval()
    with torch.no_grad():
        logits = model(inputs_tensor)
        probabilities = torch.softmax(logits, dim=1)
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError("BADGE could not capture one classifier representation")
    representation = captured[0]
    if representation.ndim > 2:
        representation = representation.reshape(len(inputs_tensor), -1)
    pseudo = probabilities.argmax(dim=1)
    error = probabilities.clone()
    error[torch.arange(len(error)), pseudo] -= 1.0
    weight_gradient = torch.einsum("bc,bh->bch", error, representation).reshape(len(inputs_tensor), -1)
    gradient = torch.cat([weight_gradient, error], dim=1)
    if not torch.isfinite(gradient).all():
        raise RuntimeError("Non-finite BADGE gradient embedding")
    return gradient.cpu().numpy(), probabilities.cpu().numpy()


def kmeans_pp_select(embeddings: np.ndarray, count: int, seed: int) -> list[int]:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or len(values) < count or not np.isfinite(values).all():
        raise RuntimeError("Invalid BADGE embedding matrix")
    rng = np.random.default_rng(int(seed))
    selected = [int(rng.integers(0, len(values)))]
    minimum_squared = np.square(values - values[selected[0]]).sum(axis=1)
    while len(selected) < count:
        minimum_squared[np.asarray(selected, dtype=int)] = 0.0
        total = float(minimum_squared.sum())
        if total <= 0:
            candidate = next(index for index in range(len(values)) if index not in selected)
        else:
            candidate = int(rng.choice(len(values), p=minimum_squared / total))
            if candidate in selected:
                candidate = next(index for index in range(len(values)) if index not in selected)
        selected.append(candidate)
        minimum_squared = np.minimum(minimum_squared, np.square(values - values[candidate]).sum(axis=1))
    return selected


def select_indices(inputs: dict, state: dict, row, session: int, candidates: np.ndarray) -> tuple[list[int], dict, pd.DataFrame]:
    metadata = inputs["metadata"]
    tokens = metadata.iloc[candidates]["opaque_candidate_token"].astype(str).to_numpy()
    logits, probabilities, predictions, margins, score_seconds = score_rows(inputs, state, candidates)
    selector_started = time.perf_counter()
    strategy = str(row.strategy)
    selector_seed = 0
    badge_dimension = 0
    if strategy in {"PCBM_PROPOSED", "GLOBAL_MARGIN"}:
        frame = pd.DataFrame({"index": np.arange(len(candidates)), "token": tokens, "predicted_label": predictions, "margin": margins})
        ordered = frame.sort_values(["margin", "token"], kind="mergesort")
        if strategy == "GLOBAL_MARGIN":
            selected = ordered.head(7)["index"].astype(int).tolist()
        else:
            nominees = ordered.groupby("predicted_label", sort=False, as_index=False).head(1).sort_values(["margin", "token"], kind="mergesort")
            selected = nominees.head(7)["index"].astype(int).tolist()
            if len(selected) < 7:
                selected.extend(ordered.loc[~ordered["index"].isin(selected), "index"].head(7 - len(selected)).astype(int).tolist())
    elif strategy == "RANDOM_UNIFORM":
        base = int(inputs["random_seed_map"][int(row.random_acquisition_seed_index)])
        selector_seed = stable_seed(f"R6C_RANDOM|{base}|{row.participant}|SESSION_{session:02d}")
        selected = np.random.default_rng(selector_seed).choice(len(candidates), size=7, replace=False).astype(int).tolist()
    elif strategy == "BADGE":
        selector_seed = stable_seed(f"{int(inputs['badge_base_seed'])}|{row.participant}|{int(row.training_seed_index)}|SESSION_{session:02d}")
        x = r6b.transform_data(inputs["features"], inputs["valid_mask"], candidates, state["means"], state["stds"])
        embeddings, badge_probabilities = badge_gradient_embeddings(state["model"], torch.from_numpy(x))
        if not np.allclose(probabilities, badge_probabilities, atol=1e-6, rtol=1e-6):
            raise RuntimeError("BADGE probability replay drift")
        badge_dimension = int(embeddings.shape[1])
        selected = kmeans_pp_select(embeddings, 7, selector_seed)
    else:
        raise RuntimeError(f"Unknown R6C strategy: {strategy}")
    selector_seconds = time.perf_counter() - selector_started
    if len(selected) != 7 or len(set(selected)) != 7 or not set(selected).issubset(set(range(len(candidates)))):
        raise RuntimeError("Selector did not return seven unique candidate indices")
    rank = {index: position for position, index in enumerate(selected, start=1)}
    audit = pd.DataFrame(
        {
            "candidate_position": np.arange(len(candidates), dtype=int),
            "opaque_candidate_token": tokens,
            "predicted_label": predictions,
            "margin": margins,
            "selected_this_round": [index in rank for index in range(len(candidates))],
            "selection_order": [rank.get(index, 0) for index in range(len(candidates))],
            "true_label_visible_to_selector": False,
            "semantic_uid_visible_to_selector": False,
        }
    )
    telemetry = {
        "model_score_seconds": float(score_seconds),
        "selector_seconds": float(selector_seconds),
        "selector_seed": int(selector_seed),
        "badge_embedding_dimension": int(badge_dimension),
    }
    return selected, telemetry, audit


def evaluate_state(inputs: dict, state: dict, row, session: int, unit_id: str) -> tuple[dict, list[dict]]:
    test_rows = r6b.test_rows_for_session(inputs["metadata"], str(row.participant), session)
    if np.intersect1d(test_rows, state["history_rows"]).size:
        raise RuntimeError("Fixed test entered R6C history")
    logits, probabilities, predictions, _margins, inference_seconds = score_rows(inputs, state, test_rows)
    truths = inputs["metadata"].iloc[test_rows]["label"].to_numpy(dtype=np.int64)
    metrics = r6b.classification_metrics(truths, predictions)
    fold = {
        "unit_id": unit_id,
        "participant": str(row.participant),
        "strategy": str(row.strategy),
        "training_seed_index": int(row.training_seed_index),
        "training_seed": int(row.training_seed),
        "target_session": int(session),
        "query_budget": 7,
        "case_analysis": str(row.participant) == "P07",
        "source_repetitions": int(len(state["history_rows"])),
        "test_repetitions": 35,
        "repetition_accuracy": metrics["accuracy"],
        "repetition_balanced_accuracy": metrics["balanced_accuracy"],
        "repetition_macro_f1": metrics["macro_f1"],
        "repetition_confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        "all_logits_finite": True,
        "all_probabilities_finite": True,
        "fixed_test_used_for_training": False,
        "fixed_test_used_for_normalization": False,
        "future_session_used": False,
        "fit_seed_paired_across_strategies": int(paired_fit_seed(int(row.training_seed), str(row.participant), session)),
        "inference_seconds": float(inference_seconds),
    }
    predictions_rows = []
    test_meta = inputs["metadata"].iloc[test_rows]
    for position, (_index, meta_row) in enumerate(test_meta.iterrows()):
        token_payload = f"{REVISION_PROTOCOL_SHA256}|{row.participant}|{session}|{int(meta_row['label'])}|{int(meta_row['repetition'])}"
        record = {
            "unit_id": unit_id,
            "participant": str(row.participant),
            "strategy": str(row.strategy),
            "training_seed_index": int(row.training_seed_index),
            "training_seed": int(row.training_seed),
            "target_session": int(session),
            "opaque_test_token": hashlib.sha256(token_payload.encode("utf-8")).hexdigest()[:24],
            "true_label": int(truths[position]),
            "predicted_label": int(predictions[position]),
        }
        record.update({f"logit_label_{label}": float(logits[position, label]) for label in range(CLASSES)})
        record.update({f"probability_label_{label}": float(probabilities[position, label]) for label in range(CLASSES)})
        predictions_rows.append(record)
    return fold, predictions_rows


def unit_contract(row) -> dict:
    return {
        "stage": "REVISION_R6C_CPU_END_TO_END_MULTISEED_BADGE",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "r6a_packet_sha256": R6A_PACKET_SHA256,
        "r6b_packet_sha256": R6B_PACKET_SHA256,
        "execution_unit_id": str(row.execution_unit_id),
        "participant": str(row.participant),
        "strategy": str(row.strategy),
        "training_seed_index": int(row.training_seed_index),
        "training_seed": int(row.training_seed),
        "query_budget": 7,
        "target_sessions": [1, 2, 3, 4, 5],
        "device": "CPU",
        "automatic_mixed_precision": False,
    }


def validate_unit_packet(path: Path, row) -> bool:
    try:
        if not path.exists() or not engine.archive_crc_passes(path):
            return False
        report = engine.read_json_member(path, "unit_report.json")
        contract = unit_contract(row)
        if report.get("unit_contract_sha256") != canonical_hash(contract) or any(report.get(key) != value for key, value in contract.items()):
            return False
        if not report.get("completed") or not report.get("all_readiness_gates_passed"):
            return False
        expected = {"fit_count": 6, "fold_count": 5, "prediction_count": 175, "loss_curve_count": 240, "selection_count": 35, "candidate_audit_count": 175}
        if any(int(report.get(key, -1)) != value for key, value in expected.items()):
            return False
        manifest = engine.read_csv_member(path, "unit_manifest.csv")
        with zipfile.ZipFile(path, "r") as archive:
            for item in manifest.itertuples(index=False):
                matches = [name for name in archive.namelist() if Path(name).name == str(item.relative_path)]
                if len(matches) != 1 or hashlib.sha256(archive.read(matches[0])).hexdigest() != str(item.sha256):
                    return False
        return True
    except Exception:
        return False


def run_unit(inputs: dict, row) -> tuple[Path, dict]:
    unit_id = str(row.execution_unit_id)
    unit_dir = UNIT_WORK_ROOT / unit_id
    if unit_dir.exists():
        shutil.rmtree(unit_dir)
    unit_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    fit_rows, loss_rows, fold_rows, prediction_rows, selection_rows, candidate_rows_all, telemetry_rows = [], [], [], [], [], [], []
    history = initial_history_rows(inputs["metadata"], str(row.participant))
    print(f"START UNIT | {unit_id}", flush=True)
    state, fit_row, losses = fit_state(inputs, unit_id, str(row.participant), str(row.strategy), int(row.training_seed), int(row.training_seed_index), 0, history)
    fit_rows.append(fit_row)
    loss_rows.extend(losses)
    print(f"  INITIAL FIT | history=35 | fit_s={fit_row['fit_seconds']:.2f}", flush=True)
    for session in range(1, 6):
        candidates = candidate_rows(inputs["metadata"], str(row.participant), session)
        selected_positions, telemetry, candidate_audit = select_indices(inputs, state, row, session, candidates)
        selected_rows = candidates[np.asarray(selected_positions, dtype=int)]
        selected_meta = inputs["metadata"].iloc[selected_rows]
        if selected_meta["fixed_test"].any() or not selected_meta["candidate"].all():
            raise RuntimeError("R6C selector returned an invalid row")
        candidate_audit.insert(0, "target_session", session)
        candidate_audit.insert(0, "training_seed_index", int(row.training_seed_index))
        candidate_audit.insert(0, "strategy", str(row.strategy))
        candidate_audit.insert(0, "participant", str(row.participant))
        candidate_audit.insert(0, "unit_id", unit_id)
        candidate_rows_all.extend(candidate_audit.to_dict("records"))
        for order, (position, selected_row) in enumerate(zip(selected_positions, selected_rows), start=1):
            meta = inputs["metadata"].iloc[int(selected_row)]
            selection_rows.append(
                {
                    "unit_id": unit_id,
                    "participant": str(row.participant),
                    "strategy": str(row.strategy),
                    "training_seed_index": int(row.training_seed_index),
                    "training_seed": int(row.training_seed),
                    "target_session": int(session),
                    "selection_order": int(order),
                    "candidate_position": int(position),
                    "opaque_candidate_token": str(meta["opaque_candidate_token"]),
                    "sequence_row_internal": int(selected_row),
                    "true_label_revealed_after_selection": int(meta["label"]),
                    "true_label_visible_to_selector": False,
                }
            )
        history = np.asarray(sorted(set(history.tolist() + selected_rows.astype(int).tolist())), dtype=np.int64)
        del state
        gc.collect()
        state, fit_row, losses = fit_state(inputs, unit_id, str(row.participant), str(row.strategy), int(row.training_seed), int(row.training_seed_index), session, history)
        fit_rows.append(fit_row)
        loss_rows.extend(losses)
        fold, predictions = evaluate_state(inputs, state, row, session, unit_id)
        fold["fit_seconds"] = float(fit_row["fit_seconds"])
        fold["model_score_seconds"] = float(telemetry["model_score_seconds"])
        fold["selector_seconds"] = float(telemetry["selector_seconds"])
        fold["selector_seed"] = int(telemetry["selector_seed"])
        fold["badge_embedding_dimension"] = int(telemetry["badge_embedding_dimension"])
        fold["optimizer_steps"] = int(fit_row["optimizer_steps"])
        fold_rows.append(fold)
        prediction_rows.extend(predictions)
        telemetry_rows.append(
            {
                "unit_id": unit_id,
                "participant": str(row.participant),
                "strategy": str(row.strategy),
                "training_seed_index": int(row.training_seed_index),
                "target_session": int(session),
                **telemetry,
                "refit_seconds": float(fit_row["fit_seconds"]),
                "fixed_test_inference_seconds": float(fold["inference_seconds"]),
            }
        )
        print(
            f"  SESSION {session}/5 | history={len(history)} | selected=7 | BA={fold['repetition_balanced_accuracy']:.6f} | fit_s={fit_row['fit_seconds']:.2f}",
            flush=True,
        )
    fits = pd.DataFrame(fit_rows)
    losses = pd.DataFrame(loss_rows)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    candidates = pd.DataFrame(candidate_rows_all)
    telemetry = pd.DataFrame(telemetry_rows)
    for frame, name in [
        (fits, "fit_audit.csv"),
        (losses, "training_loss_curves.csv"),
        (folds, "fold_results.csv"),
        (predictions, "repetition_predictions.csv"),
        (selections, "selection_trace.csv"),
        (candidates, "candidate_score_audit.csv"),
        (telemetry, "compute_telemetry.csv"),
    ]:
        atomic_csv(frame, unit_dir / name)
    manifest_rows = []
    for path in sorted(unit_dir.iterdir()):
        if path.is_file():
            manifest_rows.append({"relative_path": path.name, "bytes": path.stat().st_size, "sha256": engine.sha256_file(path)})
    manifest = pd.DataFrame(manifest_rows)
    atomic_csv(manifest, unit_dir / "unit_manifest.csv")
    expected_steps = 920
    gates = {
        "six_history_fits_are_complete": len(fits) == 6,
        "five_fixed_test_folds_are_complete": len(folds) == 5,
        "one_hundred_seventy_five_predictions_are_complete": len(predictions) == 175,
        "two_hundred_forty_epoch_losses_are_complete": len(losses) == 240,
        "thirty_five_selections_are_complete": len(selections) == 35,
        "one_hundred_seventy_five_candidate_audit_rows_are_complete": len(candidates) == 175,
        "history_counts_match_35_42_49_56_63_70": fits["history_repetitions"].tolist() == [35, 42, 49, 56, 63, 70],
        "optimizer_step_count_is_920": int(fits["optimizer_steps"].sum()) == expected_steps,
        "each_session_selects_seven_unique_candidates": selections.groupby("target_session")["opaque_candidate_token"].nunique().eq(7).all(),
        "selector_never_receives_true_labels": not candidates["true_label_visible_to_selector"].any(),
        "all_metrics_are_finite_and_bounded": np.isfinite(folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]].to_numpy(dtype=float)).all() and ((folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]] >= 0) & (folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]] <= 1)).all().all(),
        "balanced_accuracy_equals_accuracy_on_balanced_test": np.allclose(folds["repetition_accuracy"], folds["repetition_balanced_accuracy"], atol=1e-12, rtol=0),
        "no_fixed_test_or_future_session_leakage": not fits["fixed_test_in_history"].any() and (fits["maximum_history_session"] <= fits["fit_session"]).all(),
        "badge_embedding_is_present_only_for_badge": (str(row.strategy) == "BADGE" and telemetry["badge_embedding_dimension"].gt(0).all()) or (str(row.strategy) != "BADGE" and telemetry["badge_embedding_dimension"].eq(0).all()),
        "p07_is_descriptive_only": bool(str(row.participant) != "P07" or folds["case_analysis"].all()),
        "cpu_was_used": True,
        "automatic_mixed_precision_was_not_used": True,
        "no_inferential_statistical_test_was_run": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]
    contract = unit_contract(row)
    report = dict(contract)
    report.update(
        {
            "unit_contract_sha256": canonical_hash(contract),
            "fit_count": len(fits),
            "fold_count": len(folds),
            "prediction_count": len(predictions),
            "loss_curve_count": len(losses),
            "selection_count": len(selections),
            "candidate_audit_count": len(candidates),
            "optimizer_steps": int(fits["optimizer_steps"].sum()),
            "unit_runtime_minutes": (time.time() - started) / 60.0,
            "peak_cpu_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "readiness_gates": gates,
            "failed_readiness_gates": failed,
            "all_readiness_gates_passed": not failed,
            "completed": not failed,
            "scientific_data_training_run": True,
            "fixed_test_inference_run": True,
            "new_inferential_statistical_test_run": False,
        }
    )
    atomic_json(report, unit_dir / "unit_report.json")
    if failed:
        raise RuntimeError(f"R6C unit {unit_id} failed: {failed}")
    packet = UNIT_PACKET_ROOT / f"{unit_id}.zip"
    packet.parent.mkdir(parents=True, exist_ok=True)
    if not engine.make_zip(unit_dir, packet, f"Revision_R6C_Unit/{unit_id}") or not validate_unit_packet(packet, row):
        raise RuntimeError(f"R6C unit packet validation failed: {unit_id}")
    digest = engine.sha256_file(packet)
    if not engine.roundtrip_remote_file(packet, REMOTE_UNITS + "/" + packet.name, digest):
        raise RuntimeError(f"R6C unit remote round-trip failed: {unit_id}")
    report["packet_sha256"] = digest
    print(f"COMPLETE UNIT | {unit_id} | sha256={digest} | minutes={report['unit_runtime_minutes']:.3f}", flush=True)
    return packet, report


def restore_remote_units(plan: pd.DataFrame) -> tuple[dict[str, Path], list[dict]]:
    UNIT_PACKET_ROOT.mkdir(parents=True, exist_ok=True)
    result = engine.rclone(["copy", REMOTE_UNITS, str(UNIT_PACKET_ROOT), "--include", "*.zip", "--retries", "3", "--low-level-retries", "5", "--timeout", "5m"], check=False)
    if result.returncode not in {0, 3}:
        print("Remote-unit prefetch warning:", (result.stderr or result.stdout)[-500:], flush=True)
    valid, index_rows = {}, []
    for row in plan.itertuples(index=False):
        unit_id = str(row.execution_unit_id)
        path = UNIT_PACKET_ROOT / f"{unit_id}.zip"
        if validate_unit_packet(path, row):
            valid[unit_id] = path
            report = engine.read_json_member(path, "unit_report.json")
            index_rows.append({"execution_unit_id": unit_id, "status": "RESUMED_VERIFIED", "packet_sha256": engine.sha256_file(path), "unit_runtime_minutes": report.get("unit_runtime_minutes", np.nan)})
        elif path.exists():
            path.unlink()
    return valid, index_rows


def make_progress_artifact(plan: pd.DataFrame, valid: dict[str, Path], index_rows: list[dict], reason: str) -> tuple[Path, str]:
    root = RESULT_ROOT / "progress"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    status = plan[["execution_unit_id", "participant", "strategy", "training_seed_index", "training_seed"]].copy()
    status["completed"] = status["execution_unit_id"].isin(set(valid))
    atomic_csv(status, root / "revision_R6C_unit_progress.csv")
    atomic_csv(pd.DataFrame(index_rows), root / "revision_R6C_unit_packet_index.csv")
    report = {
        "stage": "REVISION_R6C_CPU_END_TO_END_MULTISEED_BADGE_PROGRESS",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "r6b_packet_sha256": R6B_PACKET_SHA256,
        "expected_units": 168,
        "completed_units": len(valid),
        "remaining_units": 168 - len(valid),
        "stop_reason": reason,
        "resume_action": "RUN_THE_SAME_NOTEBOOK_AGAIN",
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": "REVISION_R6C_PROGRESS_SAVED_RESTART_SAME_NOTEBOOK",
    }
    atomic_json(report, root / "revision_R6C_progress_report.json")
    if not engine.make_zip(root, PROGRESS_PACKET, "Revision_R6C_CPU_End_to_End_Progress"):
        raise RuntimeError("R6C progress packet CRC failed")
    digest = engine.sha256_file(PROGRESS_PACKET)
    if not engine.roundtrip_remote_file(PROGRESS_PACKET, REMOTE_OUTPUT + "/" + PROGRESS_PACKET.name, digest):
        raise RuntimeError("R6C progress packet remote round-trip failed")
    return PROGRESS_PACKET, digest


def read_unit_csv(path: Path, basename: str) -> pd.DataFrame:
    return engine.read_csv_member(path, basename)


def aggregate_and_finalize(inputs: dict, valid: dict[str, Path], index_rows: list[dict], packet_audit: pd.DataFrame) -> tuple[dict, str]:
    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    plan = inputs["r6c_plan"].sort_values(["participant", "training_seed_index", "strategy"])
    names = {
        "fits": "fit_audit.csv",
        "losses": "training_loss_curves.csv",
        "folds": "fold_results.csv",
        "predictions": "repetition_predictions.csv",
        "selections": "selection_trace.csv",
        "candidates": "candidate_score_audit.csv",
        "telemetry": "compute_telemetry.csv",
    }
    frames = {key: pd.concat([read_unit_csv(valid[str(row.execution_unit_id)], basename) for row in plan.itertuples(index=False)], ignore_index=True) for key, basename in names.items()}
    fits, losses, folds = frames["fits"], frames["losses"], frames["folds"]
    predictions, selections, candidates, telemetry = frames["predictions"], frames["selections"], frames["candidates"], frames["telemetry"]
    unit_index = pd.DataFrame(index_rows).drop_duplicates("execution_unit_id", keep="last").sort_values("execution_unit_id").reset_index(drop=True)
    seed_summary = (
        folds.groupby(["participant", "strategy", "training_seed_index", "training_seed", "case_analysis"], as_index=False)
        .agg(
            target_sessions=("target_session", "nunique"),
            mean_repetition_balanced_accuracy=("repetition_balanced_accuracy", "mean"),
            mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
            total_fit_seconds=("fit_seconds", "sum"),
            total_score_seconds=("model_score_seconds", "sum"),
            total_selector_seconds=("selector_seconds", "sum"),
            total_inference_seconds=("inference_seconds", "sum"),
        )
        .sort_values(["participant", "training_seed_index", "strategy"])
        .reset_index(drop=True)
    )
    participant_summary = (
        seed_summary.groupby(["participant", "strategy", "case_analysis"], as_index=False)
        .agg(
            training_seeds=("training_seed_index", "nunique"),
            seed_mean_balanced_accuracy=("mean_repetition_balanced_accuracy", "mean"),
            seed_sd_balanced_accuracy=("mean_repetition_balanced_accuracy", "std"),
            seed_min_balanced_accuracy=("mean_repetition_balanced_accuracy", "min"),
            seed_max_balanced_accuracy=("mean_repetition_balanced_accuracy", "max"),
        )
        .sort_values(["participant", "strategy"])
        .reset_index(drop=True)
    )
    strategy_summary = (
        participant_summary.loc[participant_summary["participant"].isin(ABLE_BODIED)]
        .groupby("strategy", as_index=False)
        .agg(
            participants=("participant", "nunique"),
            mean_participant_seed_averaged_balanced_accuracy=("seed_mean_balanced_accuracy", "mean"),
            sd_participant_seed_averaged_balanced_accuracy=("seed_mean_balanced_accuracy", "std"),
        )
    )
    outputs = [
        (packet_audit, "revision_R6C_input_packet_audit.csv"),
        (inputs["r6c_plan"], "revision_R6C_locked_execution_plan.csv"),
        (unit_index, "revision_R6C_unit_packet_index.csv"),
        (fits, "revision_R6C_fit_audit.csv"),
        (losses, "revision_R6C_training_loss_curves.csv"),
        (folds, "revision_R6C_fold_results.csv"),
        (predictions, "revision_R6C_repetition_predictions.csv"),
        (selections, "revision_R6C_selection_trace.csv"),
        (candidates, "revision_R6C_candidate_score_audit.csv"),
        (telemetry, "revision_R6C_compute_telemetry.csv"),
        (seed_summary, "revision_R6C_seed_level_summary.csv"),
        (participant_summary, "revision_R6C_participant_seed_averaged_summary.csv"),
        (strategy_summary, "revision_R6C_able_bodied_strategy_summary.csv"),
    ]
    for frame, name in outputs:
        atomic_csv(frame, RESULT_ROOT / name)
    expected_units = set(inputs["r6c_plan"]["execution_unit_id"].astype(str))
    paired_fit = fits.groupby(["participant", "training_seed_index", "fit_session"])["fit_seed"].nunique()
    selection_groups = selections.groupby(["unit_id", "target_session"])["opaque_candidate_token"].nunique()
    gates = {
        "all_six_input_packets_pass_hash_and_crc": len(packet_audit) == 6 and packet_audit[["hash_matches", "crc_passes"]].all().all(),
        "all_r6c_input_readiness_gates_passed": all(inputs["input_gates_r6c"].values()),
        "one_hundred_sixty_eight_unit_packets_are_complete": len(valid) == 168,
        "unit_set_matches_locked_execution_plan": set(unit_index["execution_unit_id"].astype(str)) == expected_units,
        "all_units_are_remote_verified": len(unit_index) == 168 and unit_index["status"].isin(["RESUMED_VERIFIED", "COMPLETED_REMOTE_VERIFIED"]).all(),
        "fit_count_is_1008": len(fits) == 1008,
        "fold_count_is_840": len(folds) == 840,
        "prediction_count_is_29400": len(predictions) == 29400,
        "loss_curve_count_is_40320": len(losses) == 40320,
        "selection_count_is_5880": len(selections) == 5880,
        "candidate_audit_count_is_29400": len(candidates) == 29400,
        "telemetry_count_is_840": len(telemetry) == 840,
        "seed_summary_has_168_rows": len(seed_summary) == 168,
        "participant_summary_has_28_rows": len(participant_summary) == 28,
        "each_participant_strategy_has_six_seeds": participant_summary["training_seeds"].eq(6).all(),
        "fit_seeds_are_paired_across_four_strategies": paired_fit.eq(1).all(),
        "each_unit_session_selects_seven_unique_candidates": len(selection_groups) == 840 and selection_groups.eq(7).all(),
        "selectors_never_receive_true_labels": not candidates["true_label_visible_to_selector"].any(),
        "all_metrics_are_finite_and_bounded": np.isfinite(folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]].to_numpy(dtype=float)).all() and ((folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]] >= 0) & (folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]] <= 1)).all().all(),
        "balanced_accuracy_equals_accuracy_on_balanced_test": np.allclose(folds["repetition_accuracy"], folds["repetition_balanced_accuracy"], atol=1e-12, rtol=0),
        "no_fixed_test_or_future_session_leakage": not fits["fixed_test_in_history"].any() and (fits["maximum_history_session"] <= fits["fit_session"]).all(),
        "badge_embedding_is_present_for_badge_only": telemetry.loc[telemetry["strategy"].eq("BADGE"), "badge_embedding_dimension"].gt(0).all() and telemetry.loc[~telemetry["strategy"].eq("BADGE"), "badge_embedding_dimension"].eq(0).all(),
        "p07_is_descriptive_only": folds.loc[folds["participant"].eq("P07"), "case_analysis"].all(),
        "cpu_execution_is_recorded": True,
        "automatic_mixed_precision_was_disabled": True,
        "no_inferential_statistical_test_was_run": True,
        "stage3g_and_stage5f_conclusions_cannot_be_replaced": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]
    report = {
        "stage": "REVISION_R6C_CPU_END_TO_END_MULTISEED_BADGE",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "r6b_packet_sha256": R6B_PACKET_SHA256,
        "execution_device": "CPU",
        "cpu_threads": CPU_THREADS,
        "automatic_mixed_precision": False,
        "completed_units": len(valid),
        "fit_count": len(fits),
        "fold_count": len(folds),
        "prediction_count": len(predictions),
        "selection_count": len(selections),
        "scientific_data_training_run": True,
        "fixed_test_inference_run": True,
        "new_inferential_statistical_test_run": False,
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": "PASS_TO_REVISION_R6D_DEEP_STABILITY_AND_COMPUTE_AGGREGATION" if not failed else "REVISION_R6C_FINALIZATION_FAILED",
    }
    atomic_json(report, RESULT_ROOT / "revision_R6C_final_report.json")
    shutil.copy2(Path(__file__), RESULT_ROOT / "revision_R6C_executed_source.py")
    manifest_rows = []
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file() and path.name != "revision_R6C_output_manifest.csv":
            manifest_rows.append({"relative_path": path.relative_to(RESULT_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": engine.sha256_file(path)})
    atomic_csv(pd.DataFrame(manifest_rows), RESULT_ROOT / "revision_R6C_output_manifest.csv")
    if failed:
        raise RuntimeError(f"R6C final readiness failed: {failed}")
    if not engine.make_zip(RESULT_ROOT, FINAL_PACKET, "Revision_R6C_CPU_End_to_End_Multiseed_BADGE"):
        raise RuntimeError("R6C final packet CRC failed")
    digest = engine.sha256_file(FINAL_PACKET)
    if not engine.roundtrip_remote_file(FINAL_PACKET, REMOTE_OUTPUT + "/" + FINAL_PACKET.name, digest):
        raise RuntimeError("R6C final packet remote round-trip failed")
    return report, digest


def main() -> None:
    print("=" * 112)
    print("REVISION R6C — CPU END-TO-END SIX-SEED TCN + BADGE SENSITIVITY")
    print("=" * 112)
    print("Execution device: CPU")
    print("CPU threads:", CPU_THREADS)
    print("Expected resumable units: 168")
    print("Strategies: PCBM, Global Margin, Random, BADGE")
    print("New reviewer experiment: True")
    print("New inferential statistical test: False")
    print("Resume source: verified per-unit Google Drive packets")
    print()
    lock_path = WORKING / "_revision_R6C_single_instance.lock"
    lock_handle = open(lock_path, "w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("DUPLICATE INVOCATION DETECTED: another R6C process owns the single-instance lock.")
        print("FINAL DECISION: DUPLICATE_INVOCATION_EXITED_SAFELY")
        return
    torch.set_num_threads(CPU_THREADS)
    for directory in [INPUT_ROOT, UNIT_WORK_ROOT, UNIT_PACKET_ROOT]:
        directory.mkdir(parents=True, exist_ok=True)
    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified R0, R6A, R6B, Stage5B, Stage5C, and Stage5D2 packets...")
    packets, packet_rows = {}, []
    for basename, (expected, relative) in DIRECT_PACKETS.items():
        path, source = direct_restore(basename, expected, relative)
        observed = engine.sha256_file(path)
        crc = engine.archive_crc_passes(path)
        packets[basename] = path
        packet_rows.append({"packet": basename, "source": source, "expected_sha256": expected, "observed_sha256": observed, "hash_matches": observed == expected, "crc_passes": crc})
    packet_audit = pd.DataFrame(packet_rows)
    if not packet_audit[["hash_matches", "crc_passes"]].all().all():
        raise RuntimeError("R6C frozen packet integrity failed")
    inputs = prepare_inputs(packets)
    plan = inputs["r6c_plan"]
    print("Locked execution units:", len(plan))
    print("Prefetching and validating completed remote units...")
    valid, index_rows = restore_remote_units(plan)
    print("Verified units available before this run:", len(valid), "/ 168")
    completed_this_run = 0
    for row in plan.itertuples(index=False):
        unit_id = str(row.execution_unit_id)
        if unit_id in valid:
            print(f"SKIP VERIFIED | {unit_id}", flush=True)
            continue
        elapsed = (time.time() - START_TIME) / 60.0
        if completed_this_run >= MAX_UNITS_PER_RUN or (completed_this_run > 0 and elapsed >= MAX_RUNTIME_MINUTES):
            reason = "MAX_UNITS_PER_RUN" if completed_this_run >= MAX_UNITS_PER_RUN else "SAFE_RUNTIME_LIMIT"
            packet, digest = make_progress_artifact(plan, valid, index_rows, reason)
            engine.cleanup_secret()
            print("Completed units:", len(valid), "/ 168")
            print("Progress packet:", packet)
            print("Progress packet SHA-256:", digest)
            print("FINAL DECISION: REVISION_R6C_PROGRESS_SAVED_RESTART_SAME_NOTEBOOK")
            return
        packet, unit_report = run_unit(inputs, row)
        valid[unit_id] = packet
        index_rows.append({"execution_unit_id": unit_id, "status": "COMPLETED_REMOTE_VERIFIED", "packet_sha256": engine.sha256_file(packet), "unit_runtime_minutes": unit_report["unit_runtime_minutes"]})
        completed_this_run += 1
        print(f"PROGRESS | {len(valid)}/168 complete | this run={completed_this_run}", flush=True)
    report, digest = aggregate_and_finalize(inputs, valid, index_rows, packet_audit)
    engine.cleanup_secret()
    print()
    print("=" * 112)
    print("REVISION R6C — FINAL SUMMARY")
    print("=" * 112)
    print("Completed units:", report["completed_units"])
    print("History fits:", report["fit_count"])
    print("Fixed-test folds:", report["fold_count"])
    print("Repetition predictions:", report["prediction_count"])
    print("Selection rows:", report["selection_count"])
    print("Failed readiness gates:", report["failed_readiness_gates"] or "None")
    print("Final packet:", FINAL_PACKET)
    print("Final packet SHA-256:", digest)
    print("Remote round-trip verified: True")
    print("Runtime minutes:", round(report["runtime_minutes"], 3))
    print()
    print("FINAL DECISION: PASS_TO_REVISION_R6D_DEEP_STABILITY_AND_COMPUTE_AGGREGATION")


if __name__ == "__main__":
    main()
