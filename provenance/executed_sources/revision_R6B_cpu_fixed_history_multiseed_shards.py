from __future__ import annotations

import fcntl
import gc
import hashlib
import importlib.util
import io
import json
import os
import platform
import random
import resource
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import revision_R3A_P1_float32_engine_frozen_trajectory_unit_test as engine


REVISION_PROTOCOL_SHA256 = "6807b71de18ca82013cfa4360d760e0daf9a920a1acc0625dcb13bd8f4d07249"
DEEP_PROTOCOL_SHA256 = "abe15812c1a52b0f4e917b5b6ad39b0dfde50e5bb2d58dfcc35b3cacb22e3bd2"
R6A_PACKET_SHA256 = "72045da36669c61bf69b6553caa99007cc6f2d6925ac0cfe9fea61bc01f49880"
STAGE5B_PACKET_SHA256 = "1c0fbc63f6412362f3ae7cd22609ea6a7fcb23236cdf688ad5fe0578ebaab84d"
STAGE5C_PACKET_SHA256 = "85ea2e8a8440369a77d43f00b5d509ea2f2978d2a60ab2f24fb828ce9ca6b9d4"
STAGE5D2_PACKET_SHA256 = "fc8ac364bac0344639a50977d5f8725b1e5b5b2875758e01587de8c083a1f914"

PARTICIPANTS = [f"P{i:02d}" for i in range(1, 8)]
ABLE_BODIED = PARTICIPANTS[:6]
STRATEGIES = ["PCBM_PROPOSED", "GLOBAL_MARGIN"]
CLASSES = 7
CHANNELS = 64
WINDOWS = 37
TARGET_EPOCHS = 40
TARGET_BATCH_SIZE = 16
ENCODER_LEARNING_RATE = 1.0e-4
CLASSIFIER_LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-3
LABEL_SMOOTHING = 0.05
EXPECTED_MODEL_PARAMETERS = 118536
CPU_THREADS = max(1, min(int(os.environ.get("R6_CPU_THREADS", "4")), os.cpu_count() or 1))
MAX_UNITS_PER_RUN = max(1, int(os.environ.get("R6B_MAX_UNITS_PER_RUN", "84")))
MAX_RUNTIME_MINUTES = max(30.0, float(os.environ.get("R6B_MAX_RUNTIME_MINUTES", "630")))

WORKING = Path(os.environ.get("REVISION_R6B_WORKING", "/kaggle/working"))
INPUT_ROOT = WORKING / "REVISION_R6B_FROZEN_INPUTS"
RESULT_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R6B_CPU_Fixed_History_Multiseed"
UNIT_WORK_ROOT = WORKING / "REVISION_R6B_UNIT_WORK"
UNIT_PACKET_ROOT = WORKING / "REVISION_R6B_UNIT_PACKETS"
FINAL_PACKET = WORKING / "revision_R6B_cpu_fixed_history_multiseed_packet.zip"
PROGRESS_PACKET = WORKING / "revision_R6B_cpu_fixed_history_multiseed_progress_packet.zip"
REMOTE_BASE = engine.REMOTE_BASE
REMOTE_OUTPUT = REMOTE_BASE + "/Reviewer_Revision/Revision_R6B_CPU_Fixed_History_Multiseed"
REMOTE_UNITS = REMOTE_OUTPUT + "/units"
START_TIME = time.time()

DIRECT_PACKETS = {
    "revision_R6A_cpu_runtime_amendment_preflight_packet.zip": (
        R6A_PACKET_SHA256,
        "Reviewer_Revision/Revision_R6A_CPU_Runtime_Amendment_Preflight/"
        "revision_R6A_cpu_runtime_amendment_preflight_packet.zip",
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
        "Deep_Training/Stage5D2_Full_Deterministic_Deep_Trajectories/"
        "stage5d2_full_deterministic_deep_trajectories_packet.zip",
    ),
}


def canonical_hash(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_seed(text: str) -> int:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return int(value % 2_000_000_000 + 1)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


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
        if result.returncode == 0 and temporary.exists():
            if engine.sha256_file(temporary) == expected_hash and engine.archive_crc_passes(temporary):
                os.replace(temporary, destination)
                return destination, "GOOGLE_DRIVE_DIRECT"
            last_error = "hash-or-crc-mismatch"
        else:
            last_error = (result.stderr or result.stdout or f"returncode={result.returncode}")[-1000:]
        temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not restore verified {basename}: {last_error}")


def load_model_class(stage5b_packet: Path):
    model_path = INPUT_ROOT / "stage5b_mask_aware_rms_tcn.py"
    engine.extract_member(stage5b_packet, model_path.name, model_path)
    spec = importlib.util.spec_from_file_location("stage5b_mask_aware_rms_tcn", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MaskAwareRMSTCN, model_path


def extract_checkpoint(packet: Path, participant: str) -> Path:
    destination = INPUT_ROOT / "pretrained" / f"{participant}_best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if name.endswith(f"/{participant}/best.pt")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one best checkpoint for {participant}; found {len(matches)}")
        destination.write_bytes(archive.read(matches[0]))
    return destination


def prepare_inputs(packets: dict[str, Path]) -> dict:
    r6a_packet = packets["revision_R6A_cpu_runtime_amendment_preflight_packet.zip"]
    stage5b_packet = packets["stage5b_deep_sequence_assembly_packet.zip"]
    stage5c_packet = packets["stage5c1_dual_gpu_loso_pretraining_packet.zip"]
    stage5d2_packet = packets["stage5d2_full_deterministic_deep_trajectories_packet.zip"]

    r6a_report = engine.read_json_member(r6a_packet, "revision_R6A_final_report.json")
    amendment = engine.read_json_member(r6a_packet, "revision_R6A_cpu_runtime_amendment.json")
    stage5d2_report = engine.read_json_member(stage5d2_packet, "stage5d2_full_deterministic_report.json")
    execution_plan = engine.read_csv_member(r6a_packet, "revision_R6A_cpu_execution_manifest.csv")
    execution_plan = execution_plan.loc[execution_plan["stage"].astype(str).eq("R6B_FIXED_HISTORY")].copy()
    for column in ["training_seed_index", "training_seed", "query_budget", "expected_target_sessions"]:
        execution_plan[column] = pd.to_numeric(execution_plan[column], errors="raise").astype(np.int64)
    execution_plan["participant"] = execution_plan["participant"].astype(str)
    execution_plan["strategy"] = execution_plan["strategy"].astype(str)
    execution_plan = execution_plan.sort_values(["participant", "training_seed_index", "strategy"]).reset_index(drop=True)

    for basename in [
        "stage5b_rms_repetition_sequences.npy",
        "stage5b_main_valid_repetition_sequences.npy",
        "stage5b_repetition_metadata.csv",
    ]:
        engine.extract_member(stage5b_packet, basename, INPUT_ROOT / basename)
    metadata = pd.read_csv(INPUT_ROOT / "stage5b_repetition_metadata.csv")
    metadata["participant"] = metadata["participant"].astype(str)
    for column in ["session", "label", "repetition", "sequence_row"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(np.int64)
    metadata = metadata.sort_values("sequence_row").reset_index(drop=True)
    if len(metadata) != 2940 or metadata["sequence_row"].tolist() != list(range(2940)):
        raise RuntimeError("Stage5B metadata is not the locked 2,940-row sequence universe")
    metadata["initial_history"] = metadata["session"].eq(0) & metadata["repetition"].le(5)
    metadata["candidate"] = metadata["session"].between(1, 5) & metadata["repetition"].le(5)
    metadata["fixed_test"] = metadata["session"].between(1, 5) & metadata["repetition"].gt(5)

    selections = engine.read_csv_member(stage5d2_packet, "stage5d2_selection_trace.csv")
    for column in ["query_budget", "target_session", "round_index", "selection_order_in_round", "sequence_row_internal"]:
        selections[column] = pd.to_numeric(selections[column], errors="raise").astype(np.int64)
    selections["participant"] = selections["participant"].astype(str)
    selections["strategy"] = selections["strategy"].astype(str)
    selections = selections.loc[
        selections["query_budget"].eq(7)
        & selections["strategy"].isin(STRATEGIES)
        & selections["participant"].isin(PARTICIPANTS)
    ].copy()
    selections = selections.sort_values(
        ["participant", "strategy", "target_session", "round_index", "selection_order_in_round"]
    ).reset_index(drop=True)
    selected_meta = selections.merge(
        metadata[["sequence_row", "participant", "session", "candidate", "fixed_test"]],
        left_on="sequence_row_internal",
        right_on="sequence_row",
        suffixes=("", "_metadata"),
        how="left",
        validate="many_to_one",
    )
    selection_groups = selections.groupby(["participant", "strategy", "target_session"]).size()
    if (
        len(selections) != 490
        or len(selection_groups) != 70
        or not selection_groups.eq(7).all()
        or not selected_meta["candidate"].all()
        or selected_meta["fixed_test"].any()
        or not selected_meta["participant"].eq(selected_meta["participant_metadata"]).all()
        or not selected_meta["target_session"].eq(selected_meta["session"]).all()
    ):
        raise RuntimeError("Frozen Stage5D2 K07 history contract failed")

    features = np.load(INPUT_ROOT / "stage5b_rms_repetition_sequences.npy", mmap_mode="r", allow_pickle=False)
    valid_mask = np.load(INPUT_ROOT / "stage5b_main_valid_repetition_sequences.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (2940, 37, 64) or valid_mask.shape != features.shape:
        raise RuntimeError("Stage5B feature-array contract failed")
    model_class, model_path = load_model_class(stage5b_packet)
    model_parameter_count = int(sum(parameter.numel() for parameter in model_class().parameters()))

    primary_seed_map = (
        execution_plan.loc[execution_plan["training_seed_index"].eq(0), ["participant", "training_seed"]]
        .drop_duplicates()
        .set_index("participant")["training_seed"]
        .to_dict()
    )
    checkpoints = {}
    checkpoint_rows = []
    for participant in PARTICIPANTS:
        path = extract_checkpoint(stage5c_packet, participant)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = model_class()
        strict = True
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except Exception:
            strict = False
        finite = all(torch.isfinite(value).all().item() for value in checkpoint["model_state_dict"].values())
        identity = bool(
            checkpoint.get("protocol_sha256") == DEEP_PROTOCOL_SHA256
            and checkpoint.get("target_participant") == participant
            and int(checkpoint.get("training_seed")) == int(primary_seed_map[participant])
            and checkpoint.get("target_data_used") is False
        )
        checkpoint_rows.append(
            {"participant": participant, "checkpoint_path": str(path), "identity_valid": identity, "state_finite": finite, "strict_model_load": strict}
        )
        checkpoints[participant] = checkpoint
    checkpoint_audit = pd.DataFrame(checkpoint_rows)

    amendment_hash_valid = canonical_hash({key: value for key, value in amendment.items() if key != "amendment_sha256"}) == amendment.get("amendment_sha256")
    gates = {
        "r6a_parent_all_gates_passed": bool(r6a_report.get("all_readiness_gates_passed")),
        "r6a_parent_decision_authorizes_r6b": r6a_report.get("final_decision") == "PASS_TO_REVISION_R6B_CPU_FIXED_HISTORY_MULTISEED_SHARDS",
        "r6a_cpu_amendment_hash_verifies": amendment_hash_valid,
        "stage5d2_parent_all_gates_passed": bool(stage5d2_report.get("all_readiness_gates_passed")),
        "deep_protocol_hash_is_preserved": stage5d2_report.get("deep_protocol_sha256") == DEEP_PROTOCOL_SHA256,
        "r6b_execution_plan_has_84_units": len(execution_plan) == 84,
        "execution_plan_has_exact_participants": set(execution_plan["participant"]) == set(PARTICIPANTS),
        "execution_plan_has_exact_strategies": set(execution_plan["strategy"]) == set(STRATEGIES),
        "each_participant_strategy_has_six_seeds": execution_plan.groupby(["participant", "strategy"]).size().eq(6).all(),
        "all_units_are_k07_five_session": execution_plan["query_budget"].eq(7).all() and execution_plan["expected_target_sessions"].eq(5).all(),
        "frozen_selection_rows_are_490": len(selections) == 490,
        "each_frozen_selection_group_has_seven_rows": len(selection_groups) == 70 and selection_groups.eq(7).all(),
        "all_frozen_selections_are_candidates": bool(selected_meta["candidate"].all()),
        "no_frozen_selection_is_fixed_test": not bool(selected_meta["fixed_test"].any()),
        "feature_shape_is_2940_by_37_by_64": features.shape == (2940, 37, 64),
        "valid_mask_shape_matches_features": valid_mask.shape == features.shape,
        "model_parameter_count_is_118536": model_parameter_count == EXPECTED_MODEL_PARAMETERS,
        "seven_checkpoint_identities_are_valid": len(checkpoint_audit) == 7 and checkpoint_audit["identity_valid"].all(),
        "all_checkpoint_states_are_finite": checkpoint_audit["state_finite"].all(),
        "all_checkpoints_strictly_load": checkpoint_audit["strict_model_load"].all(),
    }
    failed = [key for key, value in gates.items() if not bool(value)]
    if failed:
        raise RuntimeError(f"R6B input readiness failed: {failed}")
    return {
        "r6a_report": r6a_report,
        "amendment": amendment,
        "stage5d2_report": stage5d2_report,
        "execution_plan": execution_plan,
        "metadata": metadata,
        "selections": selections,
        "features": features,
        "valid_mask": valid_mask,
        "model_class": model_class,
        "model_path": model_path,
        "checkpoints": checkpoints,
        "checkpoint_audit": checkpoint_audit,
        "input_gates": gates,
        "model_parameter_count": model_parameter_count,
    }


def fit_normalizer(raw: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logged = np.log1p(np.asarray(raw, dtype=np.float64))
    valid = np.asarray(valid, dtype=bool)
    means = np.zeros(CHANNELS, dtype=np.float64)
    stds = np.zeros(CHANNELS, dtype=np.float64)
    counts = np.zeros(CHANNELS, dtype=np.int64)
    for channel in range(CHANNELS):
        values = logged[:, :, channel][valid[:, :, channel]]
        counts[channel] = len(values)
        if len(values) == 0:
            raise RuntimeError(f"No valid history values for channel {channel}")
        means[channel] = values.mean()
        stds[channel] = values.std(ddof=0)
        if not np.isfinite(stds[channel]) or stds[channel] <= 0:
            raise RuntimeError(f"Invalid history standard deviation for channel {channel}")
    return means, stds, counts


def transform_data(features, valid_mask, rows, means, stds) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    raw = np.asarray(features[rows], dtype=np.float64)
    valid = np.asarray(valid_mask[rows], dtype=bool)
    normalized = (np.log1p(raw) - means[None, None, :]) / stds[None, None, :]
    normalized[~valid] = 0.0
    combined = np.concatenate([normalized.astype(np.float32), valid.astype(np.float32)], axis=2)
    combined = np.transpose(combined, (0, 2, 1)).copy()
    if combined.shape[1:] != (128, WINDOWS) or not np.isfinite(combined).all():
        raise RuntimeError("Invalid transformed TCN input")
    return combined


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def freeze_target_model(model: nn.Module) -> tuple[str, list[str], list[str]]:
    for parameter in model.parameters():
        parameter.requires_grad = True
    stem = next((name for name in ["stem", "input_projection", "input_stem"] if hasattr(model, name) and isinstance(getattr(model, name), nn.Module)), None)
    if stem is None or not hasattr(model, "blocks") or len(model.blocks) != 4:
        raise RuntimeError("TCN does not expose the locked stem plus four blocks")
    for parameter in getattr(model, stem).parameters():
        parameter.requires_grad = False
    for block in model.blocks[:2]:
        for parameter in block.parameters():
            parameter.requires_grad = False
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not frozen or not trainable:
        raise RuntimeError("Invalid target-adaptation freeze state")
    return stem, frozen, trainable


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    classifier = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and name.startswith("classifier.")]
    encoder = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and not name.startswith("classifier.")]
    if not classifier or not encoder:
        raise RuntimeError("Target optimizer groups are incomplete")
    return torch.optim.AdamW(
        [{"params": encoder, "lr": ENCODER_LEARNING_RATE}, {"params": classifier, "lr": CLASSIFIER_LEARNING_RATE}],
        weight_decay=WEIGHT_DECAY,
    )


def classification_metrics(truths: np.ndarray, predictions: np.ndarray) -> dict:
    matrix = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    for truth, prediction in zip(truths, predictions):
        matrix[int(truth), int(prediction)] += 1
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    recall = np.divide(np.diag(matrix), row_sums, out=np.zeros(CLASSES, dtype=float), where=row_sums > 0)
    precision = np.divide(np.diag(matrix), col_sums, out=np.zeros(CLASSES, dtype=float), where=col_sums > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(CLASSES, dtype=float), where=(precision + recall) > 0)
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
    }


def history_rows_for_session(metadata: pd.DataFrame, selections: pd.DataFrame, participant: str, strategy: str, session: int) -> np.ndarray:
    initial = metadata.loc[metadata["participant"].eq(participant) & metadata["initial_history"], "sequence_row"].to_numpy(dtype=np.int64)
    selected = selections.loc[
        selections["participant"].eq(participant)
        & selections["strategy"].eq(strategy)
        & selections["target_session"].le(int(session)),
        "sequence_row_internal",
    ].to_numpy(dtype=np.int64)
    rows = np.asarray(sorted(set(initial.tolist() + selected.tolist())), dtype=np.int64)
    expected = 35 + 7 * int(session)
    if len(initial) != 35 or len(rows) != expected:
        raise RuntimeError(f"History schedule mismatch for {participant} {strategy} S{session}: {len(rows)} vs {expected}")
    history_meta = metadata.iloc[rows]
    if history_meta["fixed_test"].any() or history_meta["session"].max() > session or sorted(history_meta["label"].unique()) != list(range(CLASSES)):
        raise RuntimeError("History leakage or class-coverage failure")
    return rows


def test_rows_for_session(metadata: pd.DataFrame, participant: str, session: int) -> np.ndarray:
    rows = metadata.loc[metadata["participant"].eq(participant) & metadata["session"].eq(int(session)) & metadata["fixed_test"], "sequence_row"].to_numpy(dtype=np.int64)
    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 fixed-test rows for {participant} S{session}")
    return rows


def paired_fit_seed(training_seed: int, participant: str, session: int) -> int:
    return stable_seed(f"R6B_FIXED_HISTORY|{int(training_seed)}|{participant}|SESSION_{int(session):02d}")


def fit_and_evaluate_session(inputs: dict, participant: str, strategy: str, training_seed: int, training_seed_index: int, session: int) -> tuple[dict, list[dict], list[dict], list[dict]]:
    metadata = inputs["metadata"]
    history_rows = history_rows_for_session(metadata, inputs["selections"], participant, strategy, session)
    test_rows = test_rows_for_session(metadata, participant, session)
    if np.intersect1d(history_rows, test_rows).size:
        raise RuntimeError("Fixed-test row entered training history")
    means, stds, counts = fit_normalizer(inputs["features"][history_rows], inputs["valid_mask"][history_rows])
    x_train = transform_data(inputs["features"], inputs["valid_mask"], history_rows, means, stds)
    y_train = metadata.iloc[history_rows]["label"].to_numpy(dtype=np.int64)
    fit_seed = paired_fit_seed(training_seed, participant, session)
    set_seed(fit_seed)

    model = inputs["model_class"]().cpu()
    model.load_state_dict(inputs["checkpoints"][participant]["model_state_dict"], strict=True)
    stem, frozen, trainable = freeze_target_model(model)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loss_rows = []
    optimizer_steps = 0
    fit_started = time.perf_counter()
    for epoch in range(1, TARGET_EPOCHS + 1):
        generator = torch.Generator().manual_seed(fit_seed + epoch)
        loader = DataLoader(dataset, batch_size=TARGET_BATCH_SIZE, shuffle=True, generator=generator, num_workers=0, pin_memory=False, drop_last=False)
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite CPU target-adaptation loss")
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()):
                raise RuntimeError("Non-finite CPU target-adaptation gradient")
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.item()) * len(batch_labels)
            total_examples += len(batch_labels)
        epoch_loss = float(total_loss / total_examples)
        loss_rows.append(
            {
                "participant": participant,
                "strategy": strategy,
                "training_seed_index": int(training_seed_index),
                "training_seed": int(training_seed),
                "target_session": int(session),
                "epoch": int(epoch),
                "history_repetitions": int(len(history_rows)),
                "epoch_loss": epoch_loss,
            }
        )
    fit_seconds = time.perf_counter() - fit_started
    if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise RuntimeError("Non-finite adapted model state")

    x_test = transform_data(inputs["features"], inputs["valid_mask"], test_rows, means, stds)
    inference_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        logits_tensor = model(torch.from_numpy(x_test))
        probabilities_tensor = torch.softmax(logits_tensor, dim=1)
    inference_seconds = time.perf_counter() - inference_started
    logits = logits_tensor.cpu().numpy()
    probabilities = probabilities_tensor.cpu().numpy()
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    truths = metadata.iloc[test_rows]["label"].to_numpy(dtype=np.int64)
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite fixed-test prediction")
    metrics = classification_metrics(truths, predictions)
    fold_row = {
        "unit_id": f"R6B_{participant}_TS{training_seed_index:02d}_{strategy}",
        "participant": participant,
        "strategy": strategy,
        "training_seed_index": int(training_seed_index),
        "training_seed": int(training_seed),
        "target_session": int(session),
        "query_budget": 7,
        "case_analysis": participant == "P07",
        "source_repetitions": int(len(history_rows)),
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
        "fit_seed_paired_across_strategies": int(fit_seed),
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
        "optimizer_steps": int(optimizer_steps),
    }
    fit_row = {
        "unit_id": fold_row["unit_id"],
        "participant": participant,
        "strategy": strategy,
        "training_seed_index": int(training_seed_index),
        "training_seed": int(training_seed),
        "target_session": int(session),
        "history_repetitions": int(len(history_rows)),
        "history_sha256": hashlib.sha256(np.asarray(history_rows, dtype=np.int64).tobytes()).hexdigest(),
        "fit_seed": int(fit_seed),
        "stem_attribute": stem,
        "frozen_parameter_count": len(frozen),
        "trainable_parameter_count": len(trainable),
        "minimum_normalizer_count": int(counts.min()),
        "normalizer_values_finite": bool(np.isfinite(means).all() and np.isfinite(stds).all()),
        "normalizer_stds_positive": bool((stds > 0).all()),
        "fixed_test_in_history": False,
        "maximum_history_session": int(metadata.iloc[history_rows]["session"].max()),
        "target_epochs": TARGET_EPOCHS,
        "final_train_loss": float(loss_rows[-1]["epoch_loss"]),
        "optimizer_steps": int(optimizer_steps),
        "fit_seconds": float(fit_seconds),
    }
    prediction_rows = []
    test_meta = metadata.iloc[test_rows]
    for position, (row_index, meta_row) in enumerate(test_meta.iterrows()):
        token_payload = f"{REVISION_PROTOCOL_SHA256}|{participant}|{session}|{int(meta_row['label'])}|{int(meta_row['repetition'])}"
        row = {
            "unit_id": fold_row["unit_id"],
            "participant": participant,
            "strategy": strategy,
            "training_seed_index": int(training_seed_index),
            "training_seed": int(training_seed),
            "target_session": int(session),
            "opaque_test_token": hashlib.sha256(token_payload.encode("utf-8")).hexdigest()[:24],
            "true_label": int(truths[position]),
            "predicted_label": int(predictions[position]),
        }
        row.update({f"logit_label_{label}": float(logits[position, label]) for label in range(CLASSES)})
        row.update({f"probability_label_{label}": float(probabilities[position, label]) for label in range(CLASSES)})
        prediction_rows.append(row)
    del model, optimizer, dataset, x_train, x_test, logits_tensor, probabilities_tensor
    gc.collect()
    return fit_row, loss_rows, fold_row, prediction_rows


def data_manifest(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in {"unit_report.json", "unit_manifest.csv"}:
            rows.append({"relative_path": path.name, "bytes": path.stat().st_size, "sha256": engine.sha256_file(path)})
    return pd.DataFrame(rows)


def unit_contract(row) -> dict:
    return {
        "stage": "REVISION_R6B_CPU_FIXED_HISTORY_MULTISEED",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "r6a_packet_sha256": R6A_PACKET_SHA256,
        "stage5d2_packet_sha256": STAGE5D2_PACKET_SHA256,
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
        if report.get("unit_contract_sha256") != canonical_hash(contract):
            return False
        if any(report.get(key) != value for key, value in contract.items()):
            return False
        if not report.get("all_readiness_gates_passed") or not report.get("completed"):
            return False
        if report.get("fold_count") != 5 or report.get("prediction_count") != 175 or report.get("fit_count") != 5 or report.get("loss_curve_count") != 200:
            return False
        manifest = engine.read_csv_member(path, "unit_manifest.csv")
        with zipfile.ZipFile(path, "r") as archive:
            for manifest_row in manifest.itertuples(index=False):
                matches = [name for name in archive.namelist() if Path(name).name == str(manifest_row.relative_path)]
                if len(matches) != 1 or sha256_bytes(archive.read(matches[0])) != str(manifest_row.sha256):
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
    fit_rows, loss_rows, fold_rows, prediction_rows = [], [], [], []
    print(f"START UNIT | {unit_id}", flush=True)
    for session in range(1, 6):
        fit_row, session_losses, fold_row, session_predictions = fit_and_evaluate_session(
            inputs,
            str(row.participant),
            str(row.strategy),
            int(row.training_seed),
            int(row.training_seed_index),
            session,
        )
        fit_rows.append(fit_row)
        loss_rows.extend(session_losses)
        fold_rows.append(fold_row)
        prediction_rows.extend(session_predictions)
        print(
            f"  SESSION {session}/5 | history={fit_row['history_repetitions']} | "
            f"BA={fold_row['repetition_balanced_accuracy']:.6f} | fit_s={fit_row['fit_seconds']:.2f}",
            flush=True,
        )
    fits = pd.DataFrame(fit_rows)
    losses = pd.DataFrame(loss_rows)
    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    atomic_csv(fits, unit_dir / "fit_audit.csv")
    atomic_csv(losses, unit_dir / "training_loss_curves.csv")
    atomic_csv(folds, unit_dir / "fold_results.csv")
    atomic_csv(predictions, unit_dir / "repetition_predictions.csv")
    manifest = data_manifest(unit_dir)
    atomic_csv(manifest, unit_dir / "unit_manifest.csv")
    contract = unit_contract(row)
    expected_steps = sum(int(np.ceil((35 + 7 * session) / TARGET_BATCH_SIZE)) * TARGET_EPOCHS for session in range(1, 6))
    gates = {
        "five_target_session_fits_are_complete": len(fits) == 5,
        "five_fixed_test_folds_are_complete": len(folds) == 5,
        "one_hundred_seventy_five_predictions_are_complete": len(predictions) == 175,
        "two_hundred_epoch_losses_are_complete": len(losses) == 200,
        "history_counts_match_42_49_56_63_70": fits["history_repetitions"].tolist() == [42, 49, 56, 63, 70],
        "optimizer_step_count_matches_locked_cpu_schedule": int(fits["optimizer_steps"].sum()) == expected_steps == 800,
        "all_metrics_are_finite": np.isfinite(folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]].to_numpy(dtype=float)).all(),
        "all_metrics_are_between_zero_and_one": ((folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]] >= 0) & (folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]] <= 1)).all().all(),
        "balanced_accuracy_equals_accuracy_on_balanced_test": np.allclose(folds["repetition_accuracy"], folds["repetition_balanced_accuracy"], atol=1e-12, rtol=0),
        "all_logits_and_probabilities_are_finite": folds["all_logits_finite"].all() and folds["all_probabilities_finite"].all(),
        "all_normalizers_are_finite_positive": fits["normalizer_values_finite"].all() and fits["normalizer_stds_positive"].all(),
        "no_fixed_test_enters_history": not fits["fixed_test_in_history"].any(),
        "no_future_session_enters_history": (fits["maximum_history_session"] <= fits["target_session"]).all(),
        "p07_is_descriptive_only": bool(str(row.participant) != "P07" or folds["case_analysis"].all()),
        "cpu_was_used": True,
        "automatic_mixed_precision_was_not_used": True,
        "no_inferential_statistical_test_was_run": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]
    report = dict(contract)
    report.update(
        {
            "unit_contract_sha256": canonical_hash(contract),
            "fit_count": len(fits),
            "fold_count": len(folds),
            "prediction_count": len(predictions),
            "loss_curve_count": len(losses),
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
        raise RuntimeError(f"Unit {unit_id} failed: {failed}")
    packet = UNIT_PACKET_ROOT / f"{unit_id}.zip"
    packet.parent.mkdir(parents=True, exist_ok=True)
    if not engine.make_zip(unit_dir, packet, f"Revision_R6B_Unit/{unit_id}") or not validate_unit_packet(packet, row):
        raise RuntimeError(f"Unit packet validation failed: {unit_id}")
    digest = engine.sha256_file(packet)
    remote_verified = engine.roundtrip_remote_file(packet, REMOTE_UNITS + "/" + packet.name, digest)
    if not remote_verified:
        raise RuntimeError(f"Unit remote round-trip failed: {unit_id}")
    report["packet_sha256"] = digest
    report["remote_roundtrip_verified"] = True
    print(f"COMPLETE UNIT | {unit_id} | sha256={digest} | minutes={report['unit_runtime_minutes']:.3f}", flush=True)
    return packet, report


def restore_remote_units(plan: pd.DataFrame) -> tuple[dict[str, Path], list[dict]]:
    UNIT_PACKET_ROOT.mkdir(parents=True, exist_ok=True)
    result = engine.rclone(
        ["copy", REMOTE_UNITS, str(UNIT_PACKET_ROOT), "--include", "*.zip", "--retries", "3", "--low-level-retries", "5", "--timeout", "5m"],
        check=False,
    )
    if result.returncode not in {0, 3}:
        print("Remote-unit prefetch warning:", (result.stderr or result.stdout)[-500:], flush=True)
    valid = {}
    index_rows = []
    row_map = {str(row.execution_unit_id): row for row in plan.itertuples(index=False)}
    for unit_id, row in row_map.items():
        path = UNIT_PACKET_ROOT / f"{unit_id}.zip"
        if validate_unit_packet(path, row):
            valid[unit_id] = path
            report = engine.read_json_member(path, "unit_report.json")
            index_rows.append(
                {"execution_unit_id": unit_id, "status": "RESUMED_VERIFIED", "packet_sha256": engine.sha256_file(path), "unit_runtime_minutes": report.get("unit_runtime_minutes", np.nan)}
            )
        elif path.exists():
            path.unlink()
    return valid, index_rows


def read_unit_csv(packet: Path, basename: str) -> pd.DataFrame:
    return engine.read_csv_member(packet, basename)


def make_progress_artifact(plan: pd.DataFrame, valid: dict[str, Path], index_rows: list[dict], reason: str) -> tuple[Path, str]:
    progress_root = RESULT_ROOT / "progress"
    if progress_root.exists():
        shutil.rmtree(progress_root)
    progress_root.mkdir(parents=True, exist_ok=True)
    completed = set(valid)
    status = plan[["execution_unit_id", "participant", "strategy", "training_seed_index", "training_seed"]].copy()
    status["completed"] = status["execution_unit_id"].isin(completed)
    atomic_csv(status, progress_root / "revision_R6B_unit_progress.csv")
    atomic_csv(pd.DataFrame(index_rows), progress_root / "revision_R6B_unit_packet_index.csv")
    report = {
        "stage": "REVISION_R6B_CPU_FIXED_HISTORY_MULTISEED_PROGRESS",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "r6a_packet_sha256": R6A_PACKET_SHA256,
        "expected_units": 84,
        "completed_units": len(valid),
        "remaining_units": 84 - len(valid),
        "stop_reason": reason,
        "resume_action": "RUN_THE_SAME_NOTEBOOK_AGAIN",
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": "REVISION_R6B_PROGRESS_SAVED_RESTART_SAME_NOTEBOOK",
    }
    atomic_json(report, progress_root / "revision_R6B_progress_report.json")
    if not engine.make_zip(progress_root, PROGRESS_PACKET, "Revision_R6B_CPU_Fixed_History_Progress"):
        raise RuntimeError("R6B progress packet CRC failed")
    digest = engine.sha256_file(PROGRESS_PACKET)
    if not engine.roundtrip_remote_file(PROGRESS_PACKET, REMOTE_OUTPUT + "/" + PROGRESS_PACKET.name, digest):
        raise RuntimeError("R6B progress packet remote round-trip failed")
    return PROGRESS_PACKET, digest


def aggregate_and_finalize(inputs: dict, valid: dict[str, Path], index_rows: list[dict], packet_audit: pd.DataFrame) -> tuple[dict, str]:
    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ordered = inputs["execution_plan"].sort_values(["participant", "training_seed_index", "strategy"])
    fits = pd.concat([read_unit_csv(valid[str(row.execution_unit_id)], "fit_audit.csv") for row in ordered.itertuples(index=False)], ignore_index=True)
    losses = pd.concat([read_unit_csv(valid[str(row.execution_unit_id)], "training_loss_curves.csv") for row in ordered.itertuples(index=False)], ignore_index=True)
    folds = pd.concat([read_unit_csv(valid[str(row.execution_unit_id)], "fold_results.csv") for row in ordered.itertuples(index=False)], ignore_index=True)
    predictions = pd.concat([read_unit_csv(valid[str(row.execution_unit_id)], "repetition_predictions.csv") for row in ordered.itertuples(index=False)], ignore_index=True)
    unit_index = pd.DataFrame(index_rows).drop_duplicates("execution_unit_id", keep="last").sort_values("execution_unit_id").reset_index(drop=True)
    seed_summary = (
        folds.groupby(["participant", "strategy", "training_seed_index", "training_seed", "case_analysis"], as_index=False)
        .agg(
            target_sessions=("target_session", "nunique"),
            mean_repetition_balanced_accuracy=("repetition_balanced_accuracy", "mean"),
            mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
            total_fit_seconds=("fit_seconds", "sum"),
            total_inference_seconds=("inference_seconds", "sum"),
            total_optimizer_steps=("optimizer_steps", "sum"),
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
    for frame, name in [
        (packet_audit, "revision_R6B_input_packet_audit.csv"),
        (inputs["checkpoint_audit"], "revision_R6B_checkpoint_audit.csv"),
        (inputs["execution_plan"], "revision_R6B_locked_execution_plan.csv"),
        (unit_index, "revision_R6B_unit_packet_index.csv"),
        (fits, "revision_R6B_fit_audit.csv"),
        (losses, "revision_R6B_training_loss_curves.csv"),
        (folds, "revision_R6B_fold_results.csv"),
        (predictions, "revision_R6B_repetition_predictions.csv"),
        (seed_summary, "revision_R6B_seed_level_summary.csv"),
        (participant_summary, "revision_R6B_participant_seed_averaged_summary.csv"),
        (strategy_summary, "revision_R6B_able_bodied_strategy_summary.csv"),
    ]:
        atomic_csv(frame, RESULT_ROOT / name)
    expected_units = set(inputs["execution_plan"]["execution_unit_id"].astype(str))
    observed_units = set(unit_index["execution_unit_id"].astype(str))
    paired_seed_counts = folds.groupby(["participant", "training_seed_index", "target_session"])["fit_seed_paired_across_strategies"].nunique()
    gates = {
        "all_four_input_packets_pass_hash_and_crc": len(packet_audit) == 4 and packet_audit[["hash_matches", "crc_passes"]].all().all(),
        "all_r6b_input_readiness_gates_passed": all(inputs["input_gates"].values()),
        "eighty_four_unit_packets_are_complete": len(valid) == 84,
        "unit_set_matches_locked_execution_plan": observed_units == expected_units,
        "all_units_are_remote_verified": len(unit_index) == 84 and unit_index["status"].isin(["RESUMED_VERIFIED", "COMPLETED_REMOTE_VERIFIED"]).all(),
        "fit_count_is_420": len(fits) == 420,
        "fold_count_is_420": len(folds) == 420,
        "prediction_count_is_14700": len(predictions) == 14700,
        "loss_curve_count_is_16800": len(losses) == 16800,
        "seed_summary_has_84_rows": len(seed_summary) == 84,
        "participant_summary_has_14_rows": len(participant_summary) == 14,
        "each_participant_strategy_has_six_seeds": participant_summary["training_seeds"].eq(6).all(),
        "paired_fit_seed_is_identical_across_strategies": paired_seed_counts.eq(1).all(),
        "all_metrics_are_finite_and_bounded": np.isfinite(folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]].to_numpy(dtype=float)).all() and ((folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]] >= 0) & (folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]] <= 1)).all().all(),
        "balanced_accuracy_equals_accuracy_on_balanced_test": np.allclose(folds["repetition_accuracy"], folds["repetition_balanced_accuracy"], atol=1e-12, rtol=0),
        "no_fixed_test_or_future_session_leakage": not folds["fixed_test_used_for_training"].any() and not folds["fixed_test_used_for_normalization"].any() and not folds["future_session_used"].any(),
        "history_counts_match_locked_schedule": folds.apply(lambda row: int(row["source_repetitions"]) == 35 + 7 * int(row["target_session"]), axis=1).all(),
        "p07_is_descriptive_only": folds.loc[folds["participant"].eq("P07"), "case_analysis"].all(),
        "cpu_execution_is_recorded": True,
        "automatic_mixed_precision_was_disabled": True,
        "no_inferential_statistical_test_was_run": True,
        "stage3g_and_stage5f_conclusions_cannot_be_replaced": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]
    report = {
        "stage": "REVISION_R6B_CPU_FIXED_HISTORY_MULTISEED",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "r6a_packet_sha256": R6A_PACKET_SHA256,
        "cpu_amendment_sha256": inputs["amendment"]["amendment_sha256"],
        "execution_device": "CPU",
        "cpu_threads": CPU_THREADS,
        "automatic_mixed_precision": False,
        "completed_units": len(valid),
        "fit_count": len(fits),
        "fold_count": len(folds),
        "prediction_count": len(predictions),
        "loss_curve_count": len(losses),
        "scientific_data_training_run": True,
        "fixed_test_inference_run": True,
        "new_inferential_statistical_test_run": False,
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": "PASS_TO_REVISION_R6C_CPU_END_TO_END_MULTISEED_BADGE_SHARDS" if not failed else "REVISION_R6B_FINALIZATION_FAILED",
    }
    atomic_json(report, RESULT_ROOT / "revision_R6B_final_report.json")
    shutil.copy2(Path(__file__), RESULT_ROOT / "revision_R6B_executed_source.py")
    manifest_rows = []
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file() and path.name != "revision_R6B_output_manifest.csv":
            manifest_rows.append({"relative_path": path.relative_to(RESULT_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": engine.sha256_file(path)})
    atomic_csv(pd.DataFrame(manifest_rows), RESULT_ROOT / "revision_R6B_output_manifest.csv")
    if failed:
        raise RuntimeError(f"R6B final readiness failed: {failed}")
    if not engine.make_zip(RESULT_ROOT, FINAL_PACKET, "Revision_R6B_CPU_Fixed_History_Multiseed"):
        raise RuntimeError("R6B final packet CRC failed")
    digest = engine.sha256_file(FINAL_PACKET)
    if not engine.roundtrip_remote_file(FINAL_PACKET, REMOTE_OUTPUT + "/" + FINAL_PACKET.name, digest):
        raise RuntimeError("R6B final packet remote round-trip failed")
    return report, digest


def main() -> None:
    print("=" * 108)
    print("REVISION R6B — CPU FIXED-HISTORY SIX-SEED TCN STABILITY")
    print("=" * 108)
    print("Execution device: CPU")
    print("CPU threads:", CPU_THREADS)
    print("Expected resumable units: 84")
    print("Target epochs per session fit: 40")
    print("Fixed-test inference: True")
    print("New inferential statistical test: False")
    print("Resume source: verified per-unit Google Drive packets")
    print()
    lock_path = WORKING / "_revision_R6B_single_instance.lock"
    lock_handle = open(lock_path, "w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("DUPLICATE INVOCATION DETECTED: another R6B process owns the single-instance lock.")
        print("FINAL DECISION: DUPLICATE_INVOCATION_EXITED_SAFELY")
        return

    torch.set_num_threads(CPU_THREADS)
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    UNIT_WORK_ROOT.mkdir(parents=True, exist_ok=True)
    UNIT_PACKET_ROOT.mkdir(parents=True, exist_ok=True)
    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified R6A, Stage5B, Stage5C, and Stage5D2 packets...")
    packets, packet_rows = {}, []
    for basename, (expected_hash, remote_relative) in DIRECT_PACKETS.items():
        path, source = direct_restore(basename, expected_hash, remote_relative)
        observed = engine.sha256_file(path)
        crc = engine.archive_crc_passes(path)
        packets[basename] = path
        packet_rows.append(
            {"packet": basename, "source": source, "expected_sha256": expected_hash, "observed_sha256": observed, "hash_matches": observed == expected_hash, "crc_passes": crc}
        )
    packet_audit = pd.DataFrame(packet_rows)
    if not packet_audit[["hash_matches", "crc_passes"]].all().all():
        raise RuntimeError("R6B frozen packet integrity failed")
    inputs = prepare_inputs(packets)
    plan = inputs["execution_plan"]
    print("Locked execution units:", len(plan))
    print("Prefetching and validating any completed remote units...")
    valid, index_rows = restore_remote_units(plan)
    print("Verified units available before this run:", len(valid), "/ 84")
    completed_this_run = 0
    for row in plan.itertuples(index=False):
        unit_id = str(row.execution_unit_id)
        if unit_id in valid:
            print(f"SKIP VERIFIED | {unit_id}", flush=True)
            continue
        elapsed_minutes = (time.time() - START_TIME) / 60.0
        if completed_this_run >= MAX_UNITS_PER_RUN or (completed_this_run > 0 and elapsed_minutes >= MAX_RUNTIME_MINUTES):
            reason = "MAX_UNITS_PER_RUN" if completed_this_run >= MAX_UNITS_PER_RUN else "SAFE_RUNTIME_LIMIT"
            packet, digest = make_progress_artifact(plan, valid, index_rows, reason)
            engine.cleanup_secret()
            print()
            print("R6B progress saved safely.")
            print("Completed units:", len(valid), "/ 84")
            print("Progress packet:", packet)
            print("Progress packet SHA-256:", digest)
            print("FINAL DECISION: REVISION_R6B_PROGRESS_SAVED_RESTART_SAME_NOTEBOOK")
            return
        packet, unit_report = run_unit(inputs, row)
        valid[unit_id] = packet
        index_rows.append(
            {"execution_unit_id": unit_id, "status": "COMPLETED_REMOTE_VERIFIED", "packet_sha256": engine.sha256_file(packet), "unit_runtime_minutes": unit_report["unit_runtime_minutes"]}
        )
        completed_this_run += 1
        print(f"PROGRESS | {len(valid)}/84 complete | this run={completed_this_run}", flush=True)

    report, digest = aggregate_and_finalize(inputs, valid, index_rows, packet_audit)
    engine.cleanup_secret()
    print()
    print("=" * 108)
    print("REVISION R6B — FINAL SUMMARY")
    print("=" * 108)
    print("Completed units:", report["completed_units"])
    print("Session fits:", report["fit_count"])
    print("Fixed-test folds:", report["fold_count"])
    print("Repetition predictions:", report["prediction_count"])
    print("Failed readiness gates:", report["failed_readiness_gates"] or "None")
    print("Final packet:", FINAL_PACKET)
    print("Final packet SHA-256:", digest)
    print("Remote round-trip verified: True")
    print("Runtime minutes:", round(report["runtime_minutes"], 3))
    print()
    print("FINAL DECISION: PASS_TO_REVISION_R6C_CPU_END_TO_END_MULTISEED_BADGE_SHARDS")


if __name__ == "__main__":
    main()
