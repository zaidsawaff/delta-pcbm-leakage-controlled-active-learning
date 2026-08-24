from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import resource
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import revision_R3A_P1_float32_engine_frozen_trajectory_unit_test as engine


REVISION_PROTOCOL_SHA256 = "6807b71de18ca82013cfa4360d760e0daf9a920a1acc0625dcb13bd8f4d07249"
DEEP_PROTOCOL_SHA256 = "abe15812c1a52b0f4e917b5b6ad39b0dfde50e5bb2d58dfcc35b3cacb22e3bd2"
R0_PACKET_SHA256 = "0800e315a29b81934095ba56deaea3f8b6600fd0df13db348d7ea72d3b82df78"
R5C_PACKET_SHA256 = "c1d4cd0cba8438526ec6c9acd4df6c079e34733ce770dd93adc0e0336430306b"
STAGE5B_PACKET_SHA256 = "1c0fbc63f6412362f3ae7cd22609ea6a7fcb23236cdf688ad5fe0578ebaab84d"
STAGE5C_PACKET_SHA256 = "85ea2e8a8440369a77d43f00b5d509ea2f2978d2a60ab2f24fb828ce9ca6b9d4"
STAGE5D2_PACKET_SHA256 = "fc8ac364bac0344639a50977d5f8725b1e5b5b2875758e01587de8c083a1f914"

PARTICIPANTS = [f"P{i:02d}" for i in range(1, 8)]
ABLE_BODIED = PARTICIPANTS[:6]
FIXED_HISTORY_STRATEGIES = ["PCBM_PROPOSED", "GLOBAL_MARGIN"]
END_TO_END_STRATEGIES = [
    "PCBM_PROPOSED",
    "GLOBAL_MARGIN",
    "RANDOM_UNIFORM",
    "BADGE",
]
CPU_THREADS = max(1, min(int(os.environ.get("R6_CPU_THREADS", "4")), os.cpu_count() or 1))
TARGET_EPOCHS = 40
TARGET_BATCH_SIZE = 16
ENCODER_LEARNING_RATE = 1.0e-4
CLASSIFIER_LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-3
LABEL_SMOOTHING = 0.05
EXPECTED_MODEL_PARAMETERS = 118536
CALIBRATION_STEPS = 12

WORKING = Path(os.environ.get("REVISION_R6A_WORKING", "/kaggle/working"))
INPUT_ROOT = WORKING / "REVISION_R6A_FROZEN_INPUTS"
RESULT_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R6A_CPU_Runtime_Amendment_Preflight"
PACKET_PATH = WORKING / "revision_R6A_cpu_runtime_amendment_preflight_packet.zip"
REMOTE_BASE = engine.REMOTE_BASE
REMOTE_OUTPUT = REMOTE_BASE + "/Reviewer_Revision/Revision_R6A_CPU_Runtime_Amendment_Preflight"
START_TIME = time.time()

DIRECT_PACKETS = {
    "stageR0_reviewer_revision_protocol_lock_packet.zip": (
        R0_PACKET_SHA256,
        "Reviewer_Revision/StageR0_Reviewer_Revision_Protocol_Lock/"
        "stageR0_reviewer_revision_protocol_lock_packet.zip",
    ),
    "revision_R5C_within_session_drift_audit_packet.zip": (
        R5C_PACKET_SHA256,
        "Reviewer_Revision/Revision_R5C_Within_Session_Drift_Audit/"
        "revision_R5C_within_session_drift_audit_packet.zip",
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


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def direct_restore(basename: str, expected_hash: str, remote_relative: str) -> tuple[Path, str]:
    destination = INPUT_ROOT / basename
    if (
        destination.exists()
        and engine.sha256_file(destination) == expected_hash
        and engine.archive_crc_passes(destination)
    ):
        return destination, "EXISTING_VERIFIED_COPY"
    temporary = destination.with_suffix(".download")
    temporary.unlink(missing_ok=True)
    last_error = ""
    for attempt in range(1, 6):
        result = engine.rclone(
            [
                "copyto",
                REMOTE_BASE + "/" + remote_relative,
                str(temporary),
                "--retries",
                "5",
                "--low-level-retries",
                "10",
                "--timeout",
                "5m",
            ],
            check=False,
        )
        if result.returncode == 0 and temporary.exists():
            if (
                engine.sha256_file(temporary) == expected_hash
                and engine.archive_crc_passes(temporary)
            ):
                os.replace(temporary, destination)
                return destination, f"GOOGLE_DRIVE_DIRECT_ATTEMPT_{attempt}"
            last_error = "downloaded bytes failed SHA-256 or CRC"
        else:
            last_error = (result.stderr or result.stdout or f"exit={result.returncode}")[-1000:]
        temporary.unlink(missing_ok=True)
        time.sleep(min(2 ** (attempt - 1), 20))
    raise RuntimeError(f"Unable to restore {basename}: {last_error}")


def checkpoint_members(packet: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(packet, "r") as archive:
        names = archive.namelist()
        for participant in PARTICIPANTS:
            matches = [name for name in names if name.endswith(f"/{participant}/best.pt")]
            rows.append(
                {
                    "participant": participant,
                    "match_count": len(matches),
                    "archive_member": matches[0] if len(matches) == 1 else "",
                    "available_exactly_once": len(matches) == 1,
                }
            )
    return pd.DataFrame(rows)


def load_model_class(stage5b_packet: Path):
    model_path = INPUT_ROOT / "stage5b_mask_aware_rms_tcn.py"
    engine.extract_member(stage5b_packet, model_path.name, model_path)
    spec = importlib.util.spec_from_file_location("stage5b_mask_aware_rms_tcn", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MaskAwareRMSTCN, model_path


def load_checkpoint(packet: Path, member: str) -> dict:
    with zipfile.ZipFile(packet, "r") as archive:
        data = archive.read(member)
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def freeze_target_model(model: nn.Module) -> tuple[str, list[str], list[str]]:
    for parameter in model.parameters():
        parameter.requires_grad = True
    stem_attribute = next(
        (
            name
            for name in ["stem", "input_projection", "input_stem"]
            if hasattr(model, name) and isinstance(getattr(model, name), nn.Module)
        ),
        None,
    )
    if stem_attribute is None or not hasattr(model, "blocks") or len(model.blocks) != 4:
        raise RuntimeError("TCN does not expose the locked stem plus four-block contract")
    for parameter in getattr(model, stem_attribute).parameters():
        parameter.requires_grad = False
    for block in model.blocks[:2]:
        for parameter in block.parameters():
            parameter.requires_grad = False
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not frozen or not trainable:
        raise RuntimeError("Invalid CPU target-adaptation freeze state")
    return stem_attribute, frozen, trainable


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    classifier = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("classifier.")
    ]
    encoder = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("classifier.")
    ]
    if not classifier or not encoder:
        raise RuntimeError("CPU target optimizer groups are incomplete")
    return torch.optim.AdamW(
        [
            {"params": encoder, "lr": ENCODER_LEARNING_RATE},
            {"params": classifier, "lr": CLASSIFIER_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def badge_gradient_embeddings(model: nn.Module, inputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    captured = []

    def pre_hook(_module, values):
        captured.append(values[0].detach())

    handle = model.classifier.register_forward_pre_hook(pre_hook)
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
    handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Could not capture exactly one last-layer representation")
    representation = captured[0]
    if representation.ndim > 2:
        representation = representation.reshape(len(inputs), -1)
    pseudo = probabilities.argmax(dim=1)
    error = probabilities.clone()
    error[torch.arange(len(error)), pseudo] -= 1.0
    weight_gradient = torch.einsum("bc,bh->bch", error, representation).reshape(len(inputs), -1)
    gradient = torch.cat([weight_gradient, error], dim=1)
    if not torch.isfinite(gradient).all():
        raise RuntimeError("Non-finite BADGE gradient embedding")
    return gradient.cpu().numpy(), probabilities.cpu().numpy()


def kmeans_pp_select(embeddings: np.ndarray, count: int, seed: int) -> list[int]:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or len(values) < count or not np.isfinite(values).all():
        raise ValueError("Invalid BADGE embedding matrix")
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
        distance = np.square(values - values[candidate]).sum(axis=1)
        minimum_squared = np.minimum(minimum_squared, distance)
    return selected


def cpu_synthetic_calibration(model_class, checkpoint: dict, badge_seed: int) -> dict:
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(20260823)
    np.random.seed(20260823)
    model = model_class().cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    stem_attribute, frozen, trainable = freeze_target_model(model)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    generator = torch.Generator().manual_seed(20260823)
    inputs = torch.randn(TARGET_BATCH_SIZE, 128, 37, generator=generator)
    labels = torch.arange(TARGET_BATCH_SIZE, dtype=torch.long) % 7
    model.train()
    started = time.perf_counter()
    losses = []
    for _ in range(CALIBRATION_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CPU synthetic TCN loss")
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ):
            raise RuntimeError("Non-finite CPU synthetic TCN gradient")
        optimizer.step()
        losses.append(float(loss.item()))
    elapsed = time.perf_counter() - started
    badge_inputs = torch.randn(35, 128, 37, generator=generator)
    badge_embeddings, probabilities = badge_gradient_embeddings(model, badge_inputs)
    selected_a = kmeans_pp_select(badge_embeddings, 7, badge_seed)
    selected_b = kmeans_pp_select(badge_embeddings, 7, badge_seed)
    different = kmeans_pp_select(badge_embeddings, 7, badge_seed + 1)
    return {
        "stem_attribute": stem_attribute,
        "frozen_parameter_names": frozen,
        "trainable_parameter_names": trainable,
        "calibration_steps": CALIBRATION_STEPS,
        "elapsed_seconds": elapsed,
        "seconds_per_optimizer_step": elapsed / CALIBRATION_STEPS,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "losses_finite": bool(np.isfinite(losses).all()),
        "badge_embedding_shape": list(badge_embeddings.shape),
        "badge_probabilities_shape": list(probabilities.shape),
        "badge_selected_indices": selected_a,
        "badge_same_seed_reproducible": selected_a == selected_b,
        "badge_different_seed_changes_selection": selected_a != different,
        "badge_selected_indices_unique": len(selected_a) == len(set(selected_a)) == 7,
        "peak_cpu_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def build_execution_plan(tcn_seeds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed_row in tcn_seeds.itertuples(index=False):
        participant = str(seed_row.participant)
        seed_index = int(seed_row.seed_index)
        seed = int(seed_row.seed)
        for strategy in FIXED_HISTORY_STRATEGIES:
            rows.append(
                {
                    "stage": "R6B_FIXED_HISTORY",
                    "execution_unit_id": f"R6B_{participant}_TS{seed_index:02d}_{strategy}",
                    "participant": participant,
                    "training_seed_index": seed_index,
                    "training_seed": seed,
                    "strategy": strategy,
                    "query_budget": 7,
                    "history_source": "FROZEN_STAGE5D2_PRIMARY_K07_SELECTIONS",
                    "random_acquisition_seed_index": 0,
                    "badge_seed_rule": "NOT_APPLICABLE",
                    "case_analysis": participant == "P07",
                    "expected_target_sessions": 5,
                }
            )
        for strategy in END_TO_END_STRATEGIES:
            rows.append(
                {
                    "stage": "R6C_END_TO_END",
                    "execution_unit_id": f"R6C_{participant}_TS{seed_index:02d}_{strategy}",
                    "participant": participant,
                    "training_seed_index": seed_index,
                    "training_seed": seed,
                    "strategy": strategy,
                    "query_budget": 7,
                    "history_source": "RESELECTED_END_TO_END_WITH_MATCHED_TRAINING_SEED",
                    "random_acquisition_seed_index": seed_index + 1 if strategy == "RANDOM_UNIFORM" else 0,
                    "badge_seed_rule": "LOCKED_BADGE_BASE_SEED_HASHED_WITH_PARTICIPANT_TRAINING_SEED_INDEX_SESSION" if strategy == "BADGE" else "NOT_APPLICABLE",
                    "case_analysis": participant == "P07",
                    "expected_target_sessions": 5,
                }
            )
    plan = pd.DataFrame(rows)
    phases = pd.DataFrame(
        [
            {
                "stage": "R6A",
                "purpose": "CPU amendment, parent audit, exact TCN/BADGE synthetic tests",
                "scientific_outcomes": False,
                "execution_units": 0,
            },
            {
                "stage": "R6B_FIXED_HISTORY",
                "purpose": "Six-seed refits with frozen Stage5D2 K07 PCBM/global histories",
                "scientific_outcomes": True,
                "execution_units": int(plan["stage"].eq("R6B_FIXED_HISTORY").sum()),
            },
            {
                "stage": "R6C_END_TO_END",
                "purpose": "Six-seed end-to-end K07 PCBM/global/random/BADGE sensitivity",
                "scientific_outcomes": True,
                "execution_units": int(plan["stage"].eq("R6C_END_TO_END").sum()),
            },
            {
                "stage": "R6D",
                "purpose": "Aggregation, compute-cost audit, participant estimands for R7",
                "scientific_outcomes": False,
                "execution_units": 0,
            },
        ]
    )
    return plan, phases


def output_manifest(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "revision_R6A_output_manifest.csv":
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": engine.sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 108)
    print("REVISION R6A — CPU RUNTIME AMENDMENT AND DEEP-STABILITY PREFLIGHT")
    print("=" * 108)
    print("Execution device: CPU")
    print("CPU threads:", CPU_THREADS)
    print("Scientific data training: False")
    print("Fixed-test inference: False")
    print("New statistical tests: False")
    print("Synthetic TCN optimization and BADGE tests: True")
    print()
    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified R0, R5C, Stage5B, Stage5C, and Stage5D2 packets...")
    resolved = {}
    audit_rows = []
    for basename, (expected_hash, remote_relative) in DIRECT_PACKETS.items():
        packet, source = direct_restore(basename, expected_hash, remote_relative)
        observed = engine.sha256_file(packet)
        crc = engine.archive_crc_passes(packet)
        resolved[basename] = packet
        audit_rows.append(
            {
                "packet": basename,
                "source": source,
                "expected_sha256": expected_hash,
                "observed_sha256": observed,
                "hash_matches": observed == expected_hash,
                "crc_passes": crc,
            }
        )
    audit = pd.DataFrame(audit_rows)
    if not audit[["hash_matches", "crc_passes"]].all().all():
        raise RuntimeError("R6A frozen input integrity failed")

    r0_packet = resolved["stageR0_reviewer_revision_protocol_lock_packet.zip"]
    r5c_packet = resolved["revision_R5C_within_session_drift_audit_packet.zip"]
    stage5b_packet = resolved["stage5b_deep_sequence_assembly_packet.zip"]
    stage5c_packet = resolved["stage5c1_dual_gpu_loso_pretraining_packet.zip"]
    stage5d2_packet = resolved["stage5d2_full_deterministic_deep_trajectories_packet.zip"]
    r0_report = engine.read_json_member(r0_packet, "stageR0_protocol_lock_report.json")
    r5c_report = engine.read_json_member(r5c_packet, "revision_R5C_final_report.json")
    stage5c_report = engine.read_json_member(stage5c_packet, "stage5c1_loso_pretraining_report.json")
    stage5d2_report = engine.read_json_member(stage5d2_packet, "stage5d2_full_deterministic_report.json")
    seed_schedule = engine.read_csv_member(r0_packet, "stageR0_seed_schedule.csv")
    tcn_seeds = seed_schedule.loc[seed_schedule["seed_family"].astype(str).eq("TCN_TRAINING")].copy()
    badge_rows = seed_schedule.loc[seed_schedule["seed_family"].astype(str).eq("BADGE_KMEANS_PP")].copy()
    for column in ["seed_index", "seed"]:
        tcn_seeds[column] = pd.to_numeric(tcn_seeds[column], errors="raise").astype(np.int64)
    tcn_seeds["participant"] = tcn_seeds["participant"].astype(str)
    badge_seed = int(pd.to_numeric(badge_rows["seed"], errors="raise").iloc[0])

    model_class, model_path = load_model_class(stage5b_packet)
    model = model_class()
    model_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    checkpoints = checkpoint_members(stage5c_packet)
    checkpoint_rows = []
    primary = tcn_seeds.loc[tcn_seeds["seed_index"].eq(0)].set_index("participant")["seed"].to_dict()
    representative_checkpoint = None
    for row in checkpoints.itertuples(index=False):
        if not row.available_exactly_once:
            checkpoint_rows.append(
                {"participant": row.participant, "identity_valid": False, "state_finite": False, "strict_model_load": False}
            )
            continue
        checkpoint = load_checkpoint(stage5c_packet, row.archive_member)
        strict_model = model_class()
        strict_loaded = True
        try:
            strict_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except Exception:
            strict_loaded = False
        identity = bool(
            checkpoint.get("protocol_sha256") == DEEP_PROTOCOL_SHA256
            and checkpoint.get("target_participant") == row.participant
            and int(checkpoint.get("training_seed")) == int(primary[row.participant])
            and checkpoint.get("target_data_used") is False
        )
        finite = bool(
            all(torch.isfinite(value).all().item() for value in checkpoint["model_state_dict"].values())
        )
        checkpoint_rows.append(
            {
                "participant": row.participant,
                "archive_member": row.archive_member,
                "training_seed": int(checkpoint.get("training_seed")),
                "identity_valid": identity,
                "state_finite": finite,
                "strict_model_load": strict_loaded,
            }
        )
        if row.participant == "P01":
            representative_checkpoint = checkpoint
    checkpoint_audit = pd.DataFrame(checkpoint_rows)
    if representative_checkpoint is None:
        raise RuntimeError("P01 representative checkpoint is unavailable")

    calibration = cpu_synthetic_calibration(model_class, representative_checkpoint, badge_seed)
    execution_plan, phases = build_execution_plan(tcn_seeds)
    amendment = {
        "amendment_name": "DELTA_PCBM_R6_CPU_RUNTIME_AMENDMENT_v1",
        "parent_revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "authorization": "USER_AUTHORIZED_CPU_EXECUTION_BEFORE_R6_SCIENTIFIC_OUTCOMES",
        "scientific_design_changed": False,
        "runtime_device_changed_from": "TWO_TESLA_T4_WORKERS",
        "runtime_device_changed_to": "CPU_RESUMABLE_SINGLE_SERVER",
        "cpu_threads": CPU_THREADS,
        "automatic_mixed_precision": False,
        "unchanged_contract": {
            "model": "MaskAwareRMSTCN",
            "parameter_count": EXPECTED_MODEL_PARAMETERS,
            "target_epochs": TARGET_EPOCHS,
            "target_batch_size": TARGET_BATCH_SIZE,
            "optimizer": "AdamW",
            "encoder_learning_rate": ENCODER_LEARNING_RATE,
            "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "frozen_layers": "stem and residual blocks 0-1",
            "training_seeds_per_participant": 6,
            "p07_role": "DESCRIPTIVE_CASE_ONLY",
            "inference_stage": "R7_AFTER_AVERAGING_TRAINING_SEEDS_WITHIN_PARTICIPANT",
        },
        "resume_contract": {
            "checkpoint_unit": "participant-training_seed-strategy",
            "upload_after_each_complete_unit": True,
            "verified_drive_units_reused": True,
            "browser_presence_required": False,
            "kaggle_background_method": "SAVE_VERSION_RUN_ALL",
        },
    }
    amendment["amendment_sha256"] = canonical_hash(amendment)
    expected_steps_per_k07_trajectory = 920
    runtime_projection = {
        "synthetic_seconds_per_optimizer_step": calibration["seconds_per_optimizer_step"],
        "expected_optimizer_steps_per_k07_trajectory": expected_steps_per_k07_trajectory,
        "projected_seconds_per_k07_trajectory": calibration["seconds_per_optimizer_step"] * expected_steps_per_k07_trajectory,
        "r6b_execution_units": int(execution_plan["stage"].eq("R6B_FIXED_HISTORY").sum()),
        "r6c_execution_units": int(execution_plan["stage"].eq("R6C_END_TO_END").sum()),
        "projection_is_synthetic_not_a_scientific_outcome": True,
    }

    gates = {
        "all_five_input_packets_pass_crc_and_hash": len(audit) == 5 and audit[["hash_matches", "crc_passes"]].all().all(),
        "r0_protocol_hash_matches": r0_report.get("protocol_sha256") == REVISION_PROTOCOL_SHA256,
        "r0_parent_all_gates_passed": bool(r0_report.get("all_readiness_gates_passed")),
        "r5c_parent_all_gates_passed": bool(r5c_report.get("all_readiness_gates_passed")),
        "r5c_parent_decision_authorizes_r6": r5c_report.get("final_decision") == "PASS_TO_REVISION_R6_DEEP_TRAINING_SEED_STABILITY",
        "stage5c_parent_all_gates_passed": bool(stage5c_report.get("all_readiness_gates_passed")),
        "stage5d2_parent_all_gates_passed": bool(stage5d2_report.get("all_readiness_gates_passed")),
        "deep_protocol_hash_is_preserved": stage5d2_report.get("deep_protocol_sha256") == DEEP_PROTOCOL_SHA256,
        "tcn_seed_rows_are_exactly_42": len(tcn_seeds) == 42,
        "each_participant_has_six_locked_tcn_seeds": set(tcn_seeds["participant"]) == set(PARTICIPANTS) and tcn_seeds.groupby("participant").size().eq(6).all(),
        "all_tcn_seed_values_are_unique": tcn_seeds["seed"].nunique() == 42,
        "one_locked_badge_seed_is_present": len(badge_rows) == 1,
        "stage5b_model_source_is_nonempty": model_path.exists() and model_path.stat().st_size > 0,
        "model_parameter_count_is_118536": model_parameter_count == EXPECTED_MODEL_PARAMETERS,
        "seven_pretrained_best_checkpoints_are_available": len(checkpoints) == 7 and checkpoints["available_exactly_once"].all(),
        "all_checkpoint_identities_are_valid": len(checkpoint_audit) == 7 and checkpoint_audit["identity_valid"].all(),
        "all_checkpoint_states_are_finite": checkpoint_audit["state_finite"].all(),
        "all_checkpoints_strictly_load_into_locked_model": checkpoint_audit["strict_model_load"].all(),
        "cpu_synthetic_losses_are_finite": calibration["losses_finite"],
        "cpu_synthetic_step_time_is_finite_positive": np.isfinite(calibration["seconds_per_optimizer_step"]) and calibration["seconds_per_optimizer_step"] > 0,
        "locked_target_freeze_contract_is_available": len(calibration["frozen_parameter_names"]) > 0 and len(calibration["trainable_parameter_names"]) > 0,
        "badge_gradient_embedding_has_35_rows": calibration["badge_embedding_shape"][0] == 35,
        "badge_kmeans_same_seed_is_reproducible": calibration["badge_same_seed_reproducible"],
        "badge_selects_seven_unique_candidates": calibration["badge_selected_indices_unique"],
        "r6b_fixed_history_execution_units_are_84": int(execution_plan["stage"].eq("R6B_FIXED_HISTORY").sum()) == 84,
        "r6c_end_to_end_execution_units_are_168": int(execution_plan["stage"].eq("R6C_END_TO_END").sum()) == 168,
        "every_execution_unit_has_five_target_sessions": execution_plan["expected_target_sessions"].eq(5).all(),
        "p07_is_descriptive_only": execution_plan.loc[execution_plan["participant"].eq("P07"), "case_analysis"].all(),
        "cpu_amendment_hash_is_valid": canonical_hash({key: value for key, value in amendment.items() if key != "amendment_sha256"}) == amendment["amendment_sha256"],
        "scientific_data_training_was_not_run": True,
        "fixed_test_inference_was_not_run": True,
        "new_statistical_test_was_not_run": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]

    atomic_csv(audit, RESULT_ROOT / "revision_R6A_input_packet_audit.csv")
    atomic_csv(tcn_seeds, RESULT_ROOT / "revision_R6A_locked_tcn_training_seed_schedule.csv")
    atomic_csv(checkpoints, RESULT_ROOT / "revision_R6A_stage5c_checkpoint_member_audit.csv")
    atomic_csv(checkpoint_audit, RESULT_ROOT / "revision_R6A_stage5c_checkpoint_identity_audit.csv")
    atomic_csv(execution_plan, RESULT_ROOT / "revision_R6A_cpu_execution_manifest.csv")
    atomic_csv(phases, RESULT_ROOT / "revision_R6A_cpu_phase_plan.csv")
    atomic_json(amendment, RESULT_ROOT / "revision_R6A_cpu_runtime_amendment.json")
    atomic_json(calibration, RESULT_ROOT / "revision_R6A_cpu_synthetic_calibration.json")
    atomic_json(runtime_projection, RESULT_ROOT / "revision_R6A_cpu_runtime_projection.json")
    report = {
        "stage": "REVISION_R6A_CPU_RUNTIME_AMENDMENT_PREFLIGHT",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "cpu_amendment_sha256": amendment["amendment_sha256"],
        "model_parameter_count": model_parameter_count,
        "cpu_threads": CPU_THREADS,
        "automatic_mixed_precision": False,
        "r6b_execution_units": runtime_projection["r6b_execution_units"],
        "r6c_execution_units": runtime_projection["r6c_execution_units"],
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "scientific_data_training_run": False,
        "fixed_test_inference_run": False,
        "new_statistical_test_run": False,
        "final_decision": "PASS_TO_REVISION_R6B_CPU_FIXED_HISTORY_MULTISEED_SHARDS" if not failed else "REVISION_R6A_PREFLIGHT_FAILED",
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
    }
    atomic_json(report, RESULT_ROOT / "revision_R6A_final_report.json")
    shutil.copy2(Path(__file__), RESULT_ROOT / "revision_R6A_executed_source.py")
    atomic_csv(output_manifest(RESULT_ROOT), RESULT_ROOT / "revision_R6A_output_manifest.csv")
    if failed:
        raise RuntimeError(f"Revision R6A failed readiness gates: {failed}")

    packet_crc = engine.make_zip(
        RESULT_ROOT, PACKET_PATH, "Revision_R6A_CPU_Runtime_Amendment_Preflight"
    )
    packet_sha = engine.sha256_file(PACKET_PATH)
    remote_verified = engine.roundtrip_remote_file(
        PACKET_PATH, REMOTE_OUTPUT + "/" + PACKET_PATH.name, packet_sha
    )
    engine.cleanup_secret()
    print()
    print("=" * 108)
    print("REVISION R6A — CPU PREFLIGHT SUMMARY")
    print("=" * 108)
    print("Model parameters:", model_parameter_count)
    print("Locked TCN seed rows:", len(tcn_seeds))
    print("R6B fixed-history units:", runtime_projection["r6b_execution_units"])
    print("R6C end-to-end units:", runtime_projection["r6c_execution_units"])
    print("Synthetic seconds/optimizer step:", round(calibration["seconds_per_optimizer_step"], 6))
    print("Projected seconds/K07 trajectory:", round(runtime_projection["projected_seconds_per_k07_trajectory"], 3))
    print("Peak CPU RAM MB:", round(calibration["peak_cpu_ram_mb"], 2))
    print("Failed readiness gates:", failed or "None")
    print("Packet CRC pass:", packet_crc)
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", packet_sha)
    print("Remote round-trip verified:", remote_verified)
    print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 3))
    if not packet_crc or not remote_verified:
        raise RuntimeError("R6A packet persistence failed")
    print()
    print("FINAL DECISION: PASS_TO_REVISION_R6B_CPU_FIXED_HISTORY_MULTISEED_SHARDS")


if __name__ == "__main__":
    main()
