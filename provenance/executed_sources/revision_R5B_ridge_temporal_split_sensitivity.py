from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import revision_R3A_P1_float32_engine_frozen_trajectory_unit_test as engine


REVISION_PROTOCOL_SHA256 = "6807b71de18ca82013cfa4360d760e0daf9a920a1acc0625dcb13bd8f4d07249"
R5A_PACKET_SHA256 = "c41e8e387b79328040e918621497e386a260fa17a9e802dd94f218f42f9ec11e"
R3A_P1_PACKET_SHA256 = "e5051aaf116af4888c632e27cd7008a7d4848b5308b6af4366a760b30a58435a"
STAGE5B_PACKET_SHA256 = "1c0fbc63f6412362f3ae7cd22609ea6a7fcb23236cdf688ad5fe0578ebaab84d"
OPAQUE_IDENTIFIER_NAMESPACE = "PCBM_OPAQUE_IDENTIFIER_v1_1"

PARTICIPANTS = [f"P{i:02d}" for i in range(1, 8)]
ABLE_BODIED = PARTICIPANTS[:6]
TARGET_SESSIONS = [1, 2, 3, 4, 5]
LABELS = list(range(7))
SPLITS = [
    "FIRST_HALF_ORIGINAL",
    "SECOND_HALF_REVERSED",
    "ODD_CANDIDATE_EVEN_TEST",
    "EVEN_CANDIDATE_ODD_TEST",
]
DETERMINISTIC_STRATEGIES = {
    "NO_ADAPTATION_REFERENCE",
    "PCBM_PROPOSED",
    "GLOBAL_MARGIN",
}
ACTIVE_STRATEGIES = {"PCBM_PROPOSED", "GLOBAL_MARGIN", "RANDOM_UNIFORM"}

EXPECTED_SHARDS = 56
EXPECTED_TRAJECTORIES = 924
EXPECTED_FOLDS = 4620
EXPECTED_PREDICTIONS = 161700
EXPECTED_SELECTIONS = 31360
EXPECTED_SELECTOR_CALLS = 4480
EXPECTED_CANDIDATE_AUDITS = 156800
EXPECTED_NORMALIZERS = 4620
EXPECTED_RECALLS = 32340
EXPECTED_CONFUSIONS = 226380
EXPECTED_COVERAGE = 4480

WORKING = Path(os.environ.get("REVISION_R5B_WORKING", "/kaggle/working"))
INPUT_ROOT = WORKING / "REVISION_R5B_FROZEN_INPUTS"
TEMP_ROOT = WORKING / "REVISION_R5B_TEMP"
PROGRESS_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R5B_Ridge_Temporal_Split_Progress"
FINAL_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R5B_Ridge_Temporal_Split_Final"
PROGRESS_PACKET = WORKING / "revision_R5B_ridge_temporal_split_progress_packet.zip"
FINAL_PACKET = WORKING / "revision_R5B_ridge_temporal_split_sensitivity_packet.zip"
REMOTE_BASE = engine.REMOTE_BASE
REMOTE_OUTPUT = REMOTE_BASE + "/Reviewer_Revision/Revision_R5B_Ridge_Temporal_Split_Sensitivity"
REMOTE_SHARDS = REMOTE_OUTPUT + "/shards"
CPU_WORKERS = max(1, min(int(os.environ.get("R5B_CPU_WORKERS", "4")), os.cpu_count() or 1))
START_TIME = time.time()

for directory in [INPUT_ROOT, TEMP_ROOT, PROGRESS_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

DIRECT_PACKETS = {
    "revision_R5A_temporal_split_drift_unit_test_packet.zip": (
        R5A_PACKET_SHA256,
        "Reviewer_Revision/Revision_R5A_Temporal_Split_Drift_Unit_Tests/"
        "revision_R5A_temporal_split_drift_unit_test_packet.zip",
    ),
    "revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip": (
        R3A_P1_PACKET_SHA256,
        "Reviewer_Revision/Revision_R3A_P1_Float32_Engine_Frozen_Trajectory_Unit_Test/"
        "revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip",
    ),
    "stage5b_deep_sequence_assembly_packet.zip": (
        STAGE5B_PACKET_SHA256,
        "Stage5B_Deep_Sequence_Assembly/stage5b_deep_sequence_assembly_packet.zip",
    ),
}

SHARD_PATTERN = re.compile(r"^(?P<shard_id>R5B_[A-Za-z0-9_]+)__(?P<sha256>[0-9a-f]{64})\.zip$")
SHARD_TABLES = {
    "selections": "revision_R5B_shard_selection_trace.csv",
    "candidate_audits": "revision_R5B_shard_candidate_audit.csv",
    "selector_calls": "revision_R5B_shard_selector_call_audit.csv",
    "normalizers": "revision_R5B_shard_normalizer_audit.csv",
    "folds": "revision_R5B_shard_fold_metrics.csv",
    "predictions": "revision_R5B_shard_repetition_predictions.csv",
    "recalls": "revision_R5B_shard_per_class_recall.csv",
    "confusions": "revision_R5B_shard_confusion_matrices_long.csv",
    "coverage": "revision_R5B_shard_selected_class_distribution.csv",
    "telemetry": "revision_R5B_shard_compute_telemetry.csv",
}
WORKER_CONTEXT: dict = {}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def direct_restore(basename: str, expected_hash: str, remote_relative: str) -> tuple[Path, str]:
    destination = INPUT_ROOT / basename
    if destination.exists() and engine.sha256_file(destination) == expected_hash and engine.archive_crc_passes(destination):
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
            if engine.sha256_file(temporary) == expected_hash and engine.archive_crc_passes(temporary):
                os.replace(temporary, destination)
                return destination, f"GOOGLE_DRIVE_DIRECT_ATTEMPT_{attempt}"
            last_error = "downloaded bytes failed frozen SHA-256 or CRC"
        else:
            last_error = (result.stderr or result.stdout or f"exit={result.returncode}")[-1000:]
        temporary.unlink(missing_ok=True)
        time.sleep(min(2 ** (attempt - 1), 20))
    raise RuntimeError(f"Unable to restore {basename}: {last_error}")


def restore_scoped_packet(basename: str, expected_hash: str, scope: str) -> tuple[Path, str]:
    destination = INPUT_ROOT / basename
    if destination.exists() and engine.sha256_file(destination) == expected_hash and engine.archive_crc_passes(destination):
        return destination, "EXISTING_VERIFIED_COPY"
    listing = engine.rclone(["lsf", REMOTE_BASE + "/" + scope, "--recursive", "--files-only"], check=False)
    candidates = []
    if listing.returncode == 0:
        for relative in listing.stdout.splitlines():
            if Path(relative).name in {basename, basename + ".bin"}:
                candidates.append(scope.rstrip("/") + "/" + relative)
    direct_candidates = [
        scope.rstrip("/") + "/" + basename,
        scope.rstrip("/") + "/" + basename + ".bin",
    ]
    ordered = list(dict.fromkeys(direct_candidates + sorted(candidates)))
    for remote_relative in ordered:
        temporary = destination.with_suffix(".download")
        temporary.unlink(missing_ok=True)
        result = engine.rclone(
            ["copyto", REMOTE_BASE + "/" + remote_relative, str(temporary), "--retries", "5", "--timeout", "5m"],
            check=False,
        )
        if result.returncode == 0 and temporary.exists():
            if engine.sha256_file(temporary) == expected_hash and engine.archive_crc_passes(temporary):
                os.replace(temporary, destination)
                return destination, "GOOGLE_DRIVE_SCOPED_" + remote_relative
        temporary.unlink(missing_ok=True)
    raise RuntimeError(f"No verified {basename} found under scoped Drive directory {scope}")


def regenerate_and_anchor_identifiers(
    metadata: pd.DataFrame,
    frozen_selections: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the locked Stage 3A identifiers and anchor the candidate subset."""
    aligned = metadata.copy()
    aligned["repetition_number"] = aligned["repetition"]
    aligned["repetition_uid"] = aligned.apply(
        lambda row: (
            f"{row.participant}_S{int(row.session):02d}_"
            f"L{int(row.label)}_R{int(row.repetition):02d}"
        ),
        axis=1,
    )
    aligned["opaque_candidate_token"] = aligned["repetition_uid"].map(
        lambda uid: hashlib.sha256(
            f"{OPAQUE_IDENTIFIER_NAMESPACE}|{uid}".encode("utf-8")
        ).hexdigest()[:24]
    )
    if (
        len(aligned) != 2940
        or aligned["repetition_uid"].duplicated().any()
        or aligned["opaque_candidate_token"].duplicated().any()
        or not aligned["opaque_candidate_token"].str.fullmatch(r"[0-9a-f]{24}").all()
    ):
        raise RuntimeError("Regenerated Stage 3A identifier universe failed size/uniqueness/format gates")

    full_pool = frozen_selections.loc[
        frozen_selections["strategy"].astype(str).eq("FULL_POOL_REFERENCE")
        & pd.to_numeric(frozen_selections["query_budget"], errors="raise").eq(35)
    ].copy()
    full_pool["sequence_row"] = pd.to_numeric(full_pool["sequence_row"], errors="raise").astype(int)
    full_pool["opaque_candidate_token"] = full_pool["opaque_candidate_token"].astype(str)
    full_pool = full_pool[["sequence_row", "opaque_candidate_token"]].drop_duplicates()

    expected_candidate_rows = set(
        aligned.loc[
            aligned["session"].isin(TARGET_SESSIONS)
            & aligned["repetition"].isin([1, 2, 3, 4, 5]),
            "sequence_row",
        ].astype(int)
    )
    if (
        len(full_pool) != 1225
        or full_pool["sequence_row"].duplicated().any()
        or set(full_pool["sequence_row"]) != expected_candidate_rows
    ):
        raise RuntimeError("Frozen full-pool anchor is not the exact 1,225-row original candidate universe")

    identifier_anchor = full_pool.merge(
        aligned[["sequence_row", "repetition_uid", "opaque_candidate_token"]],
        on="sequence_row",
        how="left",
        suffixes=("_frozen", "_regenerated"),
        indicator=True,
        validate="one_to_one",
    )
    identifier_anchor["token_matches"] = (
        identifier_anchor["opaque_candidate_token_frozen"]
        == identifier_anchor["opaque_candidate_token_regenerated"]
    )
    if (
        len(identifier_anchor) != 1225
        or not identifier_anchor["_merge"].eq("both").all()
        or not identifier_anchor["token_matches"].all()
    ):
        raise RuntimeError("Regenerated Stage 3A identifiers do not match all 1,225 frozen full-pool tokens")
    return aligned, identifier_anchor


def prepare_inputs() -> dict:
    audit_rows = []
    resolved = {}
    for basename, (expected, remote_relative) in DIRECT_PACKETS.items():
        packet, source = direct_restore(basename, expected, remote_relative)
        resolved[basename] = packet
        audit_rows.append(
            {
                "packet": basename,
                "expected_sha256": expected,
                "observed_sha256": engine.sha256_file(packet),
                "hash_matches": engine.sha256_file(packet) == expected,
                "crc_passes": engine.archive_crc_passes(packet),
                "source": source,
            }
        )
    r5a_packet = resolved["revision_R5A_temporal_split_drift_unit_test_packet.zip"]
    r3a_packet = resolved["revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip"]
    stage5b_packet = resolved["stage5b_deep_sequence_assembly_packet.zip"]
    r5a_report = engine.read_json_member(r5a_packet, "revision_R5A_final_report.json")
    r3a_report = engine.read_json_member(r3a_packet, "revision_R3A_P1_float32_reconstruction_report.json")
    if not r5a_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("R5A parent gates did not pass")
    if r5a_report.get("final_decision") != "PASS_TO_REVISION_R5B_RIDGE_TEMPORAL_SPLIT_SENSITIVITY":
        raise RuntimeError("R5A parent decision does not authorize R5B")
    if not r3a_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("R3A-P1 engine parent gates did not pass")
    embedded_engine_hash = engine.sha256_file(Path(engine.__file__))
    parent_engine_bytes = engine.archive_member(r3a_packet, "revision_R3A_P1_executed_source.py")
    parent_engine_hash = hashlib.sha256(parent_engine_bytes).hexdigest()
    if embedded_engine_hash != parent_engine_hash:
        raise RuntimeError("Embedded R3A-P1 numerical engine differs from the verified parent source")

    r3a_input_audit = engine.read_csv_member(r3a_packet, "revision_R3A_P1_input_packet_audit.csv")
    stage3g_rows = r3a_input_audit.loc[
        r3a_input_audit["packet"].astype(str).eq("stage3g_final_results_freeze_packet.zip")
    ]
    if len(stage3g_rows) != 1:
        raise RuntimeError("R3A-P1 does not expose exactly one Stage 3G hash")
    stage3g_hash = str(stage3g_rows.iloc[0]["sha256"]).lower()
    stage3g_packet, stage3g_source = restore_scoped_packet(
        "stage3g_final_results_freeze_packet.zip", stage3g_hash, "Evidence"
    )
    audit_rows.append(
        {
            "packet": stage3g_packet.name,
            "expected_sha256": stage3g_hash,
            "observed_sha256": engine.sha256_file(stage3g_packet),
            "hash_matches": engine.sha256_file(stage3g_packet) == stage3g_hash,
            "crc_passes": engine.archive_crc_passes(stage3g_packet),
            "source": stage3g_source,
        }
    )
    stage3g_hashes = engine.extract_stage3g_hash_map(stage3g_packet)
    stage3a_basename = "stage3a_v1_1_protocol_amendment_packet.zip"
    if stage3a_basename not in stage3g_hashes:
        raise RuntimeError("Stage 3G does not anchor the Stage 3A v1.1 packet")
    stage3a_hash = stage3g_hashes[stage3a_basename]
    audit = pd.DataFrame(audit_rows)
    if not audit[["hash_matches", "crc_passes"]].all().all():
        raise RuntimeError("R5B frozen input integrity failed")

    for basename in [
        "stage5b_rms_repetition_sequences.npy",
        "stage5b_main_valid_repetition_sequences.npy",
        "stage5b_repetition_metadata.csv",
    ]:
        engine.extract_member(stage5b_packet, basename, INPUT_ROOT / basename)
    metadata = pd.read_csv(INPUT_ROOT / "stage5b_repetition_metadata.csv")
    metadata["participant"] = metadata["participant"].astype(str)
    for column in ["sequence_row", "session", "label", "repetition"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(int)
    metadata = metadata.sort_values("sequence_row", kind="mergesort").reset_index(drop=True)
    if metadata["sequence_row"].tolist() != list(range(2940)):
        raise RuntimeError("Stage 5B sequence-row order drift")

    membership = engine.read_csv_member(r5a_packet, "revision_R5A_temporal_split_membership.csv")
    manifest = engine.read_csv_member(r5a_packet, "revision_R5A_R5B_execution_manifest.csv")
    frozen_folds = engine.read_csv_member(r3a_packet, "revision_R3A_P1_reconstructed_folds.csv")
    frozen_selections = engine.read_csv_member(r3a_packet, "revision_R3A_P1_reconstructed_selection_trace.csv")

    # Recreate the exact Stage 3A identities from the locked identifier
    # contract. Acceptance is conditional on exact agreement with all 1,225
    # immutable FULL_POOL_REFERENCE candidate tokens in the verified parent.
    aligned, identifier_anchor = regenerate_and_anchor_identifiers(metadata, frozen_selections)
    atomic_csv(audit, INPUT_ROOT / "revision_R5B_input_packet_audit.csv")
    atomic_csv(aligned, INPUT_ROOT / "revision_R5B_metadata_protocol_aligned.csv")
    atomic_csv(membership, INPUT_ROOT / "revision_R5B_temporal_split_membership.csv")
    atomic_csv(identifier_anchor, INPUT_ROOT / "revision_R5B_opaque_identifier_anchor.csv")
    return {
        "audit": audit,
        "r5a_report": r5a_report,
        "r3a_report": r3a_report,
        "metadata": aligned,
        "membership": membership,
        "manifest": manifest,
        "frozen_folds": frozen_folds,
        "frozen_selections": frozen_selections,
        "stage3g_hash": stage3g_hash,
        "stage3a_hash": stage3a_hash,
        "opaque_identifier_namespace": OPAQUE_IDENTIFIER_NAMESPACE,
        "opaque_identifier_anchor": identifier_anchor,
        "embedded_engine_hash": embedded_engine_hash,
    }


def prepare_split_metadata(base_metadata: pd.DataFrame, membership: pd.DataFrame, split_id: str) -> pd.DataFrame:
    metadata = base_metadata.copy()
    metadata["protocol_role"] = "UNUSED"
    metadata["eligible_for_query"] = False
    metadata["eligible_for_training"] = False
    metadata["fixed_test_never_query"] = False
    metadata["case_analysis"] = metadata["participant"].eq("P07")
    initial_mask = metadata["session"].eq(0) & metadata["repetition"].isin([1, 2, 3, 4, 5])
    metadata.loc[initial_mask, "protocol_role"] = "INITIAL_LABELED_CALIBRATION"
    metadata.loc[initial_mask, "eligible_for_training"] = True
    subset = membership.loc[membership["split_id"].astype(str).eq(split_id)].copy()
    if len(subset) != 2450 or subset["sequence_row"].duplicated().any():
        raise RuntimeError(f"Temporal membership drift for {split_id}")
    candidate_rows = pd.to_numeric(
        subset.loc[subset["temporal_role"].eq("CURRENT_SESSION_CANDIDATE"), "sequence_row"], errors="raise"
    ).astype(int).to_numpy()
    fixed_rows = pd.to_numeric(
        subset.loc[subset["temporal_role"].eq("FIXED_TEST_NEVER_QUERY"), "sequence_row"], errors="raise"
    ).astype(int).to_numpy()
    if len(candidate_rows) != 1225 or len(fixed_rows) != 1225 or np.intersect1d(candidate_rows, fixed_rows).size:
        raise RuntimeError(f"Candidate/fixed-test role contract failed for {split_id}")
    metadata.loc[candidate_rows, "protocol_role"] = "CURRENT_SESSION_UNLABELED_POOL"
    metadata.loc[candidate_rows, "eligible_for_query"] = True
    metadata.loc[candidate_rows, "eligible_for_training"] = True
    metadata.loc[fixed_rows, "protocol_role"] = "TARGET_FIXED_TEST_NEVER_QUERY"
    metadata.loc[fixed_rows, "fixed_test_never_query"] = True
    tokens = metadata.loc[candidate_rows, "opaque_candidate_token"].astype(str)
    if tokens.isna().any() or not tokens.str.fullmatch(r"[0-9a-f]{24}").all() or tokens.duplicated().any():
        raise RuntimeError(f"Opaque candidate-token contract failed for {split_id}")
    return metadata


def build_shard_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    frame = manifest.copy()
    frame["random_seed_index"] = pd.to_numeric(frame["random_seed_index"], errors="raise").astype(int)
    rows = []
    for split_id in SPLITS:
        for participant in PARTICIPANTS:
            deterministic = frame.loc[
                frame["split_id"].astype(str).eq(split_id)
                & frame["participant"].astype(str).eq(participant)
                & frame["strategy"].astype(str).isin(DETERMINISTIC_STRATEGIES)
            ]
            random_rows = frame.loc[
                frame["split_id"].astype(str).eq(split_id)
                & frame["participant"].astype(str).eq(participant)
                & frame["strategy"].astype(str).eq("RANDOM_UNIFORM")
            ]
            rows.append(
                {
                    "shard_id": f"R5B_{split_id}_{participant}_DETERMINISTIC",
                    "split_id": split_id,
                    "participant": participant,
                    "execution_family": "DETERMINISTIC",
                    "expected_trajectories": len(deterministic),
                    "expected_folds": len(deterministic) * 5,
                }
            )
            rows.append(
                {
                    "shard_id": f"R5B_{split_id}_{participant}_RANDOM30",
                    "split_id": split_id,
                    "participant": participant,
                    "execution_family": "RANDOM30",
                    "expected_trajectories": len(random_rows),
                    "expected_folds": len(random_rows) * 5,
                }
            )
    result = pd.DataFrame(rows)
    if (
        len(result) != EXPECTED_SHARDS
        or result["expected_trajectories"].sum() != EXPECTED_TRAJECTORIES
        or result["expected_folds"].sum() != EXPECTED_FOLDS
        or not result["shard_id"].is_unique
    ):
        raise RuntimeError("R5B shard manifest contract failed")
    return result


def metrics_from_predictions(true: np.ndarray, predicted: np.ndarray) -> dict:
    matrix = np.zeros((7, 7), dtype=np.int64)
    for truth, prediction in zip(true, predicted):
        matrix[int(truth), int(prediction)] += 1
    support = matrix.sum(axis=1)
    if not np.all(support == 5):
        raise RuntimeError(f"Fixed test is not five-per-class balanced: {support.tolist()}")
    recall = np.diag(matrix) / support
    predicted_support = matrix.sum(axis=0)
    precision = np.divide(np.diag(matrix), predicted_support, out=np.zeros(7), where=predicted_support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(7), where=(precision + recall) > 0)
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "recall": recall,
        "matrix": matrix,
    }


def rng_state_sha256(rng: np.random.Generator) -> str:
    payload = json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def random_select(metadata: pd.DataFrame, remaining: list[int], rng: np.random.Generator) -> tuple[list[str], pd.DataFrame, str, str]:
    visible = pd.DataFrame(
        {"opaque_candidate_token": metadata.iloc[np.asarray(remaining, dtype=int)]["opaque_candidate_token"].astype(str).to_numpy()}
    ).sort_values("opaque_candidate_token", kind="mergesort").reset_index(drop=True)
    if visible.columns.tolist() != ["opaque_candidate_token"] or visible["opaque_candidate_token"].duplicated().any():
        raise RuntimeError("Random selector schema or token uniqueness failure")
    before = rng_state_sha256(rng)
    positions = rng.choice(len(visible), size=7, replace=False)
    tokens = visible.iloc[positions]["opaque_candidate_token"].tolist()
    after = rng_state_sha256(rng)
    visible["selected_this_round"] = visible["opaque_candidate_token"].isin(tokens)
    order = {token: index for index, token in enumerate(tokens, start=1)}
    visible["random_draw_order"] = visible["opaque_candidate_token"].map(order).fillna(0).astype(int)
    return tokens, visible, before, after


def class_distribution(labels: np.ndarray) -> tuple[np.ndarray, int, float]:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=7)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum() / np.log(7.0))
    return counts, int(np.sum(counts > 0)), entropy


def run_trajectory(row: SimpleNamespace, features, main_valid, metadata: pd.DataFrame) -> dict[str, list[dict]]:
    output = {key: [] for key in SHARD_TABLES}
    participant = str(row.participant)
    split_id = str(row.split_id)
    strategy = str(row.strategy)
    budget = int(row.query_budget)
    seed_index = int(row.random_seed_index)
    seed = int(row.random_seed)
    trajectory_id = str(row.trajectory_id)
    history = engine.initial_history_rows(metadata, participant).tolist()
    rng = np.random.default_rng(seed) if strategy == "RANDOM_UNIFORM" else None
    common = {
        "trajectory_id": trajectory_id,
        "split_id": split_id,
        "participant": participant,
        "case_analysis": participant == "P07",
        "strategy": strategy,
        "query_budget": budget,
        "random_seed_index": seed_index,
        "random_seed": seed,
        "classifier": "RIDGE_ALPHA_1",
    }
    for session in TARGET_SESSIONS:
        session_started = time.perf_counter()
        remaining = engine.candidate_rows(metadata, participant, session).tolist()
        fixed_rows = engine.fixed_test_rows(metadata, participant, session)
        fixed_set = set(map(int, fixed_rows))
        history_before = list(history)
        selected_rows: list[int] = []
        selection_fit_seconds = 0.0
        candidate_score_seconds = 0.0
        selector_seconds = 0.0
        if strategy in {"PCBM_PROPOSED", "GLOBAL_MARGIN"}:
            started = time.perf_counter()
            selection_state = engine.fit_history_state(features, main_valid, metadata, history)
            selection_fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            _, predicted_pool, margins_pool = engine.score_repetitions(selection_state, features, main_valid, remaining)
            candidate_score_seconds = time.perf_counter() - started
            visible = engine.selector_frame(metadata, remaining, predicted_pool, margins_pool)
            started = time.perf_counter()
            tokens = engine.select_pcbm(visible) if strategy == "PCBM_PROPOSED" else engine.select_global_margin(visible)
            selector_seconds = time.perf_counter() - started
            rows = engine.reveal_rows(metadata, tokens, remaining)
            selected_rows = list(map(int, rows))
            selected_set = set(selected_rows)
            selected_token_set = set(tokens)
            for candidate_row, predicted_label, margin in zip(remaining, predicted_pool, margins_pool):
                output["candidate_audits"].append(
                    {
                        **common,
                        "target_session": session,
                        "query_round": 1,
                        "opaque_candidate_token": str(metadata.iloc[candidate_row]["opaque_candidate_token"]),
                        "predicted_label": int(predicted_label),
                        "margin": float(margin),
                        "selected_this_round": str(metadata.iloc[candidate_row]["opaque_candidate_token"]) in selected_token_set,
                    }
                )
            selector_schema = "opaque_candidate_token|predicted_label|margin"
            rng_before = ""
            rng_after = ""
        elif strategy == "RANDOM_UNIFORM":
            started = time.perf_counter()
            tokens, random_audit, rng_before, rng_after = random_select(metadata, remaining, rng)
            selector_seconds = time.perf_counter() - started
            rows = engine.reveal_rows(metadata, tokens, remaining)
            selected_rows = list(map(int, rows))
            selected_set = set(selected_rows)
            for audit_row in random_audit.to_dict("records"):
                output["candidate_audits"].append(
                    {
                        **common,
                        "target_session": session,
                        "query_round": 1,
                        "opaque_candidate_token": str(audit_row["opaque_candidate_token"]),
                        "predicted_label": np.nan,
                        "margin": np.nan,
                        "selected_this_round": bool(audit_row["selected_this_round"]),
                        "random_draw_order": int(audit_row["random_draw_order"]),
                    }
                )
            selector_schema = "opaque_candidate_token"
        elif strategy == "NO_ADAPTATION_REFERENCE":
            tokens = []
            selected_set = set()
            selector_schema = "NONE"
            rng_before = ""
            rng_after = ""
        else:
            raise RuntimeError(f"Unsupported R5B strategy: {strategy}")

        if strategy in ACTIVE_STRATEGIES:
            if len(selected_rows) != 7 or selected_set.intersection(fixed_set):
                raise RuntimeError("R5B selection-count or fixed-test guard failure")
            selected_meta = metadata.iloc[selected_rows]
            for position, (token, selected_row) in enumerate(zip(tokens, selected_rows), start=1):
                output["selections"].append(
                    {
                        **common,
                        "target_session": session,
                        "query_round": 1,
                        "position_in_round": position,
                        "opaque_candidate_token": str(token),
                        "sequence_row_internal_audit_only": int(selected_row),
                        "true_label_after_reveal": int(metadata.iloc[selected_row]["label"]),
                        "selected_record_is_candidate": int(selected_row) in set(remaining),
                        "selected_record_is_fixed_test": int(selected_row) in fixed_set,
                    }
                )
            history.extend(selected_rows)
            max_before = int(metadata.iloc[np.asarray(history_before, dtype=int)]["session"].max())
            output["selector_calls"].append(
                {
                    **common,
                    "target_session": session,
                    "query_round": 1,
                    "candidate_count": len(remaining),
                    "selector_schema": selector_schema,
                    "selector_forbidden_column_count": 0,
                    "true_label_visible_before_selection": False,
                    "semantic_repetition_uid_visible": False,
                    "fixed_test_used_for_selection": False,
                    "future_session_used": max_before >= session,
                    "history_repetitions_before_query": len(history_before),
                    "rng_state_sha256_before": rng_before,
                    "rng_state_sha256_after": rng_after,
                }
            )
        if len(selected_rows) != budget:
            raise RuntimeError(f"Budget mismatch for {trajectory_id} session {session}")

        started = time.perf_counter()
        final_state = engine.fit_history_state(features, main_valid, metadata, history)
        final_fit_seconds = time.perf_counter() - started
        if np.intersect1d(final_state["history_rows"], fixed_rows).size:
            raise RuntimeError("Fixed test entered final history")
        history_sessions = metadata.iloc[np.asarray(history, dtype=int)]["session"].to_numpy(dtype=int)
        if np.any(history_sessions > session):
            raise RuntimeError("Future session entered final history")
        started = time.perf_counter()
        scores, predicted, margins = engine.score_repetitions(final_state, features, main_valid, fixed_rows)
        test_seconds = time.perf_counter() - started
        truth = metadata.iloc[fixed_rows]["label"].to_numpy(dtype=int)
        metric = metrics_from_predictions(truth, predicted)
        run_id = f"{trajectory_id}__SESSION_{session}"
        output["folds"].append(
            {
                **common,
                "run_id": run_id,
                "target_session": session,
                "history_repetitions": len(history),
                "selected_repetitions_this_session": len(selected_rows),
                "test_repetitions": len(fixed_rows),
                "repetition_accuracy": metric["accuracy"],
                "repetition_balanced_accuracy": metric["balanced_accuracy"],
                "repetition_macro_f1": metric["macro_f1"],
                "balanced_accuracy_equals_accuracy": abs(metric["balanced_accuracy"] - metric["accuracy"]) < 1e-12,
                "fixed_test_entered_history": False,
                "future_session_used": bool(np.any(history_sessions > session)),
            }
        )
        output["normalizers"].append(
            {
                **common,
                "run_id": run_id,
                "target_session": session,
                "history_repetitions": len(history),
                "minimum_mean": float(np.min(final_state["means"])),
                "maximum_mean": float(np.max(final_state["means"])),
                "minimum_std": float(np.min(final_state["stds"])),
                "maximum_std": float(np.max(final_state["stds"])),
                "minimum_valid_count": int(np.min(final_state["counts"])),
                "means_dtype": str(final_state["means"].dtype),
                "stds_dtype": str(final_state["stds"].dtype),
                "training_array_dtype": str(final_state["training_array_dtype"]),
                "model_coefficient_dtype": str(final_state["model_coefficient_dtype"]),
                "numerical_engine_contract": str(final_state["numerical_engine_contract"]),
            }
        )
        test_meta = metadata.iloc[fixed_rows]
        for index, test_row in enumerate(test_meta.itertuples(index=False)):
            record = {
                **common,
                "run_id": run_id,
                "target_session": session,
                "test_position": index + 1,
                "sequence_row_internal_audit_only": int(test_row.sequence_row),
                "repetition_uid_internal_audit_only": str(test_row.repetition_uid),
                "true_label": int(truth[index]),
                "predicted_label": int(predicted[index]),
                "correct": bool(truth[index] == predicted[index]),
                "raw_margin": float(margins[index]),
            }
            for label in LABELS:
                record[f"decision_score_{label}"] = float(scores[index, label])
            output["predictions"].append(record)
        for true_label in LABELS:
            output["recalls"].append(
                {
                    **common,
                    "run_id": run_id,
                    "target_session": session,
                    "class_label": true_label,
                    "class_support": int(metric["matrix"][true_label].sum()),
                    "class_recall": float(metric["recall"][true_label]),
                }
            )
            for predicted_label in LABELS:
                output["confusions"].append(
                    {
                        **common,
                        "run_id": run_id,
                        "target_session": session,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(metric["matrix"][true_label, predicted_label]),
                    }
                )
        if strategy in ACTIVE_STRATEGIES:
            selected_labels = metadata.iloc[np.asarray(selected_rows, dtype=int)]["label"].to_numpy(dtype=int)
            counts, coverage, entropy = class_distribution(selected_labels)
            coverage_record = {
                **common,
                "run_id": run_id,
                "target_session": session,
                "selected_class_coverage": coverage,
                "selected_normalized_class_entropy": entropy,
            }
            for label, count in enumerate(counts):
                coverage_record[f"selected_true_class_{label}_count"] = int(count)
            output["coverage"].append(coverage_record)
        output["telemetry"].append(
            {
                **common,
                "run_id": run_id,
                "target_session": session,
                "selection_fit_seconds": selection_fit_seconds,
                "candidate_score_seconds": candidate_score_seconds,
                "selector_seconds": selector_seconds,
                "final_fit_seconds": final_fit_seconds,
                "fixed_test_inference_seconds": test_seconds,
                "end_to_end_session_seconds": time.perf_counter() - session_started,
            }
        )
    return output


def validate_shard(shard: SimpleNamespace, outputs: dict[str, pd.DataFrame]) -> dict[str, bool]:
    folds = outputs["folds"]
    trajectories = int(shard.expected_trajectories)
    active = trajectories if shard.execution_family == "RANDOM30" else 2
    selections = outputs["selections"]
    calls = outputs["selector_calls"]
    normalizers = outputs["normalizers"]
    metric_columns = ["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]
    gates = {
        "trajectory_count_matches_shard_manifest": folds["trajectory_id"].nunique() == trajectories,
        "fold_count_matches_shard_manifest": len(folds) == trajectories * 5,
        "prediction_count_matches": len(outputs["predictions"]) == trajectories * 5 * 35,
        "selection_count_matches": len(selections) == active * 5 * 7,
        "selector_call_count_matches": len(calls) == active * 5,
        "candidate_audit_count_matches": len(outputs["candidate_audits"]) == active * 5 * 35,
        "normalizer_count_matches": len(normalizers) == trajectories * 5,
        "recall_count_matches": len(outputs["recalls"]) == trajectories * 5 * 7,
        "confusion_count_matches": len(outputs["confusions"]) == trajectories * 5 * 49,
        "coverage_count_matches": len(outputs["coverage"]) == active * 5,
        "telemetry_count_matches": len(outputs["telemetry"]) == trajectories * 5,
        "all_selected_records_are_candidates": bool(selections["selected_record_is_candidate"].all()),
        "no_fixed_test_record_was_selected": bool((~selections["selected_record_is_fixed_test"]).all()),
        "selection_counts_match_budget": bool(
            selections.groupby(["trajectory_id", "target_session"]).size().eq(7).all()
        ),
        "selector_calls_received_no_forbidden_columns": bool(calls["selector_forbidden_column_count"].eq(0).all()),
        "true_labels_were_hidden_before_selection": bool((~calls["true_label_visible_before_selection"]).all()),
        "no_future_session_was_used": bool((~calls["future_session_used"]).all() and (~folds["future_session_used"]).all()),
        "fixed_test_never_entered_history": bool((~folds["fixed_test_entered_history"]).all()),
        "all_metrics_are_finite_and_in_range": bool(
            np.isfinite(folds[metric_columns].to_numpy(float)).all()
            and folds[metric_columns].ge(0).all().all()
            and folds[metric_columns].le(1).all().all()
        ),
        "balanced_accuracy_equals_accuracy": bool(folds["balanced_accuracy_equals_accuracy"].all()),
        "normalizers_are_finite_positive_float32": bool(
            np.isfinite(normalizers[["minimum_mean", "maximum_mean", "minimum_std", "maximum_std"]].to_numpy(float)).all()
            and normalizers["minimum_std"].gt(0).all()
            and normalizers["minimum_valid_count"].gt(0).all()
            and normalizers["means_dtype"].eq("float32").all()
            and normalizers["stds_dtype"].eq("float32").all()
            and normalizers["training_array_dtype"].eq("float32").all()
            and normalizers["model_coefficient_dtype"].eq("float32").all()
        ),
        "p07_is_case_analysis_only": bool(
            folds["case_analysis"].all() if shard.participant == "P07" else (~folds["case_analysis"]).all()
        ),
    }
    if shard.execution_family == "RANDOM30":
        gates["random_seed_indices_are_exactly_one_to_thirty"] = set(folds["random_seed_index"].unique()) == set(range(1, 31))
        gates["random_rng_hashes_are_valid"] = bool(
            calls["rng_state_sha256_before"].str.fullmatch(r"[0-9a-f]{64}").all()
            and calls["rng_state_sha256_after"].str.fullmatch(r"[0-9a-f]{64}").all()
        )
        gates["random_selector_schema_is_exact"] = bool(calls["selector_schema"].eq("opaque_candidate_token").all())
    else:
        gates["deterministic_strategy_set_is_exact"] = set(folds["strategy"].unique()) == DETERMINISTIC_STRATEGIES
        active_calls = calls["strategy"].isin({"PCBM_PROPOSED", "GLOBAL_MARGIN"})
        gates["deterministic_selector_schema_is_exact"] = bool(
            calls.loc[active_calls, "selector_schema"].eq("opaque_candidate_token|predicted_label|margin").all()
        )
    return gates


def initialize_worker(feature_path: str, mask_path: str, metadata_path: str, membership_path: str, manifest_path: str) -> None:
    global WORKER_CONTEXT
    WORKER_CONTEXT = {
        "features": np.load(feature_path, mmap_mode="r", allow_pickle=False),
        "main_valid": np.load(mask_path, mmap_mode="r", allow_pickle=False),
        "metadata": pd.read_csv(metadata_path),
        "membership": pd.read_csv(membership_path),
        "manifest": pd.read_csv(manifest_path),
    }
    for frame_name in ["metadata", "membership", "manifest"]:
        frame = WORKER_CONTEXT[frame_name]
        if "participant" in frame.columns:
            frame["participant"] = frame["participant"].astype(str)
    for column in ["sequence_row", "session", "label", "repetition"]:
        WORKER_CONTEXT["metadata"][column] = pd.to_numeric(WORKER_CONTEXT["metadata"][column], errors="raise").astype(int)


def write_shard_packet(shard: SimpleNamespace, outputs: dict[str, pd.DataFrame], gates: dict[str, bool], runtime: float) -> tuple[Path, str]:
    shard_root = TEMP_ROOT / shard.shard_id
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True)
    for key, basename in SHARD_TABLES.items():
        atomic_csv(outputs[key], shard_root / basename)
    report = {
        "stage": "REVISION_R5B",
        "shard_id": shard.shard_id,
        "split_id": shard.split_id,
        "participant": shard.participant,
        "execution_family": shard.execution_family,
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "r5a_packet_sha256": R5A_PACKET_SHA256,
        "r3a_p1_packet_sha256": R3A_P1_PACKET_SHA256,
        "stage5b_packet_sha256": STAGE5B_PACKET_SHA256,
        "trajectory_count": int(outputs["folds"]["trajectory_id"].nunique()),
        "fold_count": len(outputs["folds"]),
        "readiness_gates": gates,
        "failed_readiness_gates": [key for key, value in gates.items() if not value],
        "all_readiness_gates_passed": all(gates.values()),
        "runtime_seconds": runtime,
        "raw_hdf5_accessed": False,
        "new_statistical_test_run": False,
    }
    atomic_json(report, shard_root / "revision_R5B_shard_report.json")
    manifest_rows = []
    for path in sorted(shard_root.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {"relative_path": path.relative_to(shard_root).as_posix(), "bytes": path.stat().st_size, "sha256": engine.sha256_file(path)}
            )
    atomic_csv(pd.DataFrame(manifest_rows), shard_root / "revision_R5B_shard_manifest.csv")
    packet = TEMP_ROOT / f"{shard.shard_id}.zip"
    if not engine.make_zip(shard_root, packet, shard.shard_id):
        raise RuntimeError(f"Shard CRC failure: {shard.shard_id}")
    return packet, engine.sha256_file(packet)


def execute_shard_worker(payload: dict) -> dict:
    shard = SimpleNamespace(**payload)
    started = time.time()
    with threadpool_limits(limits=1):
        metadata = prepare_split_metadata(WORKER_CONTEXT["metadata"], WORKER_CONTEXT["membership"], shard.split_id)
        manifest = WORKER_CONTEXT["manifest"]
        if shard.execution_family == "DETERMINISTIC":
            rows = manifest.loc[
                manifest["split_id"].astype(str).eq(shard.split_id)
                & manifest["participant"].astype(str).eq(shard.participant)
                & manifest["strategy"].astype(str).isin(DETERMINISTIC_STRATEGIES)
            ]
        else:
            rows = manifest.loc[
                manifest["split_id"].astype(str).eq(shard.split_id)
                & manifest["participant"].astype(str).eq(shard.participant)
                & manifest["strategy"].astype(str).eq("RANDOM_UNIFORM")
            ].sort_values("random_seed_index")
        combined = {key: [] for key in SHARD_TABLES}
        for row in rows.itertuples(index=False):
            trajectory = run_trajectory(row, WORKER_CONTEXT["features"], WORKER_CONTEXT["main_valid"], metadata)
            for key in combined:
                combined[key].extend(trajectory[key])
        outputs = {key: pd.DataFrame(value) for key, value in combined.items()}
        gates = validate_shard(shard, outputs)
        failed = [key for key, value in gates.items() if not value]
        if failed:
            raise RuntimeError(f"Shard {shard.shard_id} failed: {failed}")
        packet, digest = write_shard_packet(shard, outputs, gates, time.time() - started)
        return {
            "shard_id": shard.shard_id,
            "packet": str(packet),
            "sha256": digest,
            "runtime_seconds": time.time() - started,
        }


def discover_completed(expected_ids: set[str]) -> tuple[dict[str, dict], list[dict]]:
    result = engine.rclone(["lsf", REMOTE_SHARDS, "--files-only"], check=False)
    if result.returncode != 0:
        return {}, []
    mapping = {}
    duplicates = []
    for basename in result.stdout.splitlines():
        match = SHARD_PATTERN.fullmatch(Path(basename).name)
        if match is None or match.group("shard_id") not in expected_ids:
            continue
        record = {
            "shard_id": match.group("shard_id"),
            "sha256": match.group("sha256"),
            "remote_basename": Path(basename).name,
            "remote_path": REMOTE_SHARDS + "/" + Path(basename).name,
        }
        if record["shard_id"] in mapping:
            duplicates.append(record)
        else:
            mapping[record["shard_id"]] = record
    return mapping, duplicates


def make_progress_packet(shard_manifest: pd.DataFrame, completed: dict, duplicates: list, decision: str) -> str:
    if PROGRESS_ROOT.exists():
        shutil.rmtree(PROGRESS_ROOT)
    PROGRESS_ROOT.mkdir(parents=True)
    completed_frame = pd.DataFrame(list(completed.values()))
    remaining = shard_manifest.loc[~shard_manifest["shard_id"].isin(set(completed))].copy()
    atomic_csv(shard_manifest, PROGRESS_ROOT / "revision_R5B_expected_shards.csv")
    atomic_csv(completed_frame, PROGRESS_ROOT / "revision_R5B_completed_shards.csv")
    atomic_csv(remaining, PROGRESS_ROOT / "revision_R5B_remaining_shards.csv")
    atomic_csv(pd.DataFrame(duplicates), PROGRESS_ROOT / "revision_R5B_duplicate_records.csv")
    report = {
        "stage": "REVISION_R5B",
        "expected_shards": EXPECTED_SHARDS,
        "completed_shards": len(completed),
        "remaining_shards": len(remaining),
        "completion_fraction": len(completed) / EXPECTED_SHARDS,
        "final_decision": decision,
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "r5a_packet_sha256": R5A_PACKET_SHA256,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
    }
    atomic_json(report, PROGRESS_ROOT / "revision_R5B_progress_report.json")
    if not engine.make_zip(PROGRESS_ROOT, PROGRESS_PACKET, "Revision_R5B_Ridge_Temporal_Split_Progress"):
        raise RuntimeError("R5B progress packet CRC failed")
    digest = engine.sha256_file(PROGRESS_PACKET)
    if not engine.roundtrip_remote_file(PROGRESS_PACKET, REMOTE_OUTPUT + "/" + PROGRESS_PACKET.name, digest):
        raise RuntimeError("R5B progress packet round-trip failed")
    return digest


def read_csv_from_shard(packet: Path, basename: str) -> pd.DataFrame:
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {basename} in {packet.name}")
        return pd.read_csv(io.BytesIO(archive.read(matches[0])))


def bulk_restore(completed: dict) -> dict[str, Path]:
    cache = TEMP_ROOT / "RESTORED_SHARDS"
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    result = engine.rclone(
        ["copy", REMOTE_SHARDS, str(cache), "--include", "*.zip", "--transfers", "16", "--checkers", "32", "--retries", "5", "--timeout", "5m"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "bulk shard restore failed")[-2000:])
    local = {path.name: path for path in cache.glob("*.zip")}
    restored = {}
    for shard_id, record in completed.items():
        path = local.get(record["remote_basename"])
        if path is None or engine.sha256_file(path) != record["sha256"] or not engine.archive_crc_passes(path):
            raise RuntimeError(f"Final shard verification failed: {shard_id}")
        restored[shard_id] = path
    return restored


def aggregate(inputs: dict, shard_manifest: pd.DataFrame, completed: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if FINAL_ROOT.exists():
        shutil.rmtree(FINAL_ROOT)
    FINAL_ROOT.mkdir(parents=True)
    restored = bulk_restore(completed)
    tables = {key: [] for key in SHARD_TABLES}
    shard_audit_rows = []
    for index, shard in enumerate(shard_manifest.itertuples(index=False), start=1):
        packet = restored[shard.shard_id]
        with zipfile.ZipFile(packet, "r") as archive:
            report_names = [name for name in archive.namelist() if Path(name).name == "revision_R5B_shard_report.json"]
            if len(report_names) != 1:
                raise RuntimeError(f"Missing shard report: {shard.shard_id}")
            report = json.loads(archive.read(report_names[0]).decode("utf-8"))
        shard_audit_rows.append(
            {
                "shard_id": shard.shard_id,
                "sha256": completed[shard.shard_id]["sha256"],
                "crc_passes": engine.archive_crc_passes(packet),
                "report_all_gates_passed": bool(report.get("all_readiness_gates_passed")),
                "trajectory_count": int(report.get("trajectory_count", -1)),
                "fold_count": int(report.get("fold_count", -1)),
            }
        )
        for key, basename in SHARD_TABLES.items():
            tables[key].append(read_csv_from_shard(packet, basename))
        if index % 10 == 0 or index == len(shard_manifest):
            print(f"Final shard aggregation: {index}/{len(shard_manifest)}", flush=True)
    aggregate_tables = {key: pd.concat(frames, ignore_index=True) for key, frames in tables.items()}
    for key, frame in aggregate_tables.items():
        atomic_csv(frame, FINAL_ROOT / f"revision_R5B_aggregate_{key}.csv")
    shard_audit = pd.DataFrame(shard_audit_rows)
    atomic_csv(shard_audit, FINAL_ROOT / "revision_R5B_shard_integrity_audit.csv")
    atomic_csv(shard_manifest, FINAL_ROOT / "revision_R5B_shard_manifest.csv")
    atomic_csv(inputs["audit"], FINAL_ROOT / "revision_R5B_input_packet_audit.csv")
    atomic_csv(
        inputs["opaque_identifier_anchor"],
        FINAL_ROOT / "revision_R5B_opaque_identifier_frozen_full_pool_anchor.csv",
    )

    folds = aggregate_tables["folds"]
    trajectory_means = (
        folds.groupby(
            ["trajectory_id", "split_id", "participant", "case_analysis", "strategy", "query_budget", "random_seed_index", "random_seed"],
            as_index=False,
        )[["repetition_balanced_accuracy", "repetition_macro_f1"]]
        .mean()
        .rename(
            columns={
                "repetition_balanced_accuracy": "mean_session_repetition_balanced_accuracy",
                "repetition_macro_f1": "mean_session_repetition_macro_f1",
            }
        )
    )
    participant_summary = (
        trajectory_means.groupby(
            ["split_id", "participant", "case_analysis", "strategy", "query_budget"], as_index=False
        )
        .agg(
            random_seed_replicates=("trajectory_id", "size"),
            mean_repetition_balanced_accuracy=("mean_session_repetition_balanced_accuracy", "mean"),
            std_across_random_seed_means=("mean_session_repetition_balanced_accuracy", "std"),
            mean_repetition_macro_f1=("mean_session_repetition_macro_f1", "mean"),
        )
    )
    participant_summary["std_across_random_seed_means"] = participant_summary["std_across_random_seed_means"].fillna(0.0)
    able = (
        participant_summary.loc[participant_summary["participant"].isin(ABLE_BODIED)]
        .groupby(["split_id", "strategy", "query_budget"], as_index=False)
        .agg(
            participants=("participant", "nunique"),
            minimum_random_seed_replicates=("random_seed_replicates", "min"),
            mean_repetition_balanced_accuracy=("mean_repetition_balanced_accuracy", "mean"),
            std_between_participants=("mean_repetition_balanced_accuracy", "std"),
            mean_repetition_macro_f1=("mean_repetition_macro_f1", "mean"),
        )
    )
    p07 = participant_summary.loc[participant_summary["participant"].eq("P07")].copy()
    pivot = participant_summary.loc[participant_summary["participant"].isin(ABLE_BODIED)].pivot(
        index=["split_id", "participant"], columns="strategy", values="mean_repetition_balanced_accuracy"
    ).reset_index()
    contrasts = pivot[["split_id", "participant"]].copy()
    contrasts["pcbm_minus_random_k07"] = pivot["PCBM_PROPOSED"] - pivot["RANDOM_UNIFORM"]
    contrasts["pcbm_minus_global_k07"] = pivot["PCBM_PROPOSED"] - pivot["GLOBAL_MARGIN"]
    contrasts["scientific_role"] = "PARTICIPANT_LEVEL_ESTIMANDS_ONLY_NO_INFERENCE_UNTIL_R7"
    atomic_csv(trajectory_means, FINAL_ROOT / "revision_R5B_trajectory_session_means.csv")
    atomic_csv(participant_summary, FINAL_ROOT / "revision_R5B_participant_summary.csv")
    atomic_csv(able, FINAL_ROOT / "revision_R5B_able_bodied_descriptive_summary.csv")
    atomic_csv(p07, FINAL_ROOT / "revision_R5B_P07_descriptive_summary.csv")
    atomic_csv(contrasts, FINAL_ROOT / "revision_R5B_participant_level_locked_contrasts.csv")

    original = folds.loc[
        folds["split_id"].eq("FIRST_HALF_ORIGINAL")
        & folds["strategy"].isin(DETERMINISTIC_STRATEGIES)
    ].copy()
    frozen = inputs["frozen_folds"].copy()
    frozen = frozen.loc[
        frozen["strategy"].astype(str).isin(DETERMINISTIC_STRATEGIES)
        & pd.to_numeric(frozen["query_budget"]).isin([0, 7])
    ].copy()
    keys = ["participant", "target_session", "strategy", "query_budget"]
    anchor = original.merge(
        frozen[keys + ["repetition_balanced_accuracy"]], on=keys, how="outer", suffixes=("_r5b", "_frozen"), indicator=True, validate="one_to_one"
    )
    anchor["absolute_metric_difference"] = (
        anchor["repetition_balanced_accuracy_r5b"] - anchor["repetition_balanced_accuracy_frozen"]
    ).abs()
    atomic_csv(anchor, FINAL_ROOT / "revision_R5B_original_split_frozen_fold_anchor.csv")

    observed_selection = aggregate_tables["selections"].loc[
        aggregate_tables["selections"]["split_id"].eq("FIRST_HALF_ORIGINAL")
        & aggregate_tables["selections"]["strategy"].isin({"PCBM_PROPOSED", "GLOBAL_MARGIN"})
    ].copy()
    frozen_selection = inputs["frozen_selections"].loc[
        inputs["frozen_selections"]["strategy"].astype(str).isin({"PCBM_PROPOSED", "GLOBAL_MARGIN"})
        & pd.to_numeric(inputs["frozen_selections"]["query_budget"]).eq(7)
    ].copy()
    selection_audit_rows = []
    for key in sorted(set(map(tuple, observed_selection[["participant", "target_session", "strategy", "query_budget"]].drop_duplicates().to_numpy()))):
        participant, session, strategy, budget = key
        observed_group = observed_selection.loc[
            observed_selection["participant"].eq(participant)
            & observed_selection["target_session"].eq(session)
            & observed_selection["strategy"].eq(strategy)
            & observed_selection["query_budget"].eq(budget)
        ]
        frozen_group = frozen_selection.loc[
            frozen_selection["participant"].eq(participant)
            & frozen_selection["target_session"].eq(session)
            & frozen_selection["strategy"].eq(strategy)
            & frozen_selection["query_budget"].eq(budget)
        ]
        observed_rows = sorted(observed_group["sequence_row_internal_audit_only"].astype(int).tolist())
        frozen_rows = sorted(frozen_group["sequence_row"].astype(int).tolist())
        selection_audit_rows.append(
            {
                "participant": participant,
                "target_session": int(session),
                "strategy": strategy,
                "query_budget": int(budget),
                "observed_count": len(observed_rows),
                "frozen_count": len(frozen_rows),
                "selected_repetition_identity_sets_match": observed_rows == frozen_rows,
            }
        )
    selection_anchor = pd.DataFrame(selection_audit_rows)
    atomic_csv(selection_anchor, FINAL_ROOT / "revision_R5B_original_split_frozen_selection_anchor.csv")

    telemetry_columns = [
        "selection_fit_seconds", "candidate_score_seconds", "selector_seconds", "final_fit_seconds", "fixed_test_inference_seconds", "end_to_end_session_seconds"
    ]
    metrics = folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]]
    normalizers = aggregate_tables["normalizers"]
    gates = {
        "revision_protocol_hash_matches": inputs["r5a_report"].get("revision_protocol_sha256") == REVISION_PROTOCOL_SHA256,
        "r5a_parent_all_gates_passed": bool(inputs["r5a_report"].get("all_readiness_gates_passed")),
        "r3a_p1_parent_all_gates_passed": bool(inputs["r3a_report"].get("all_readiness_gates_passed")),
        "embedded_engine_source_matches_verified_parent": inputs["embedded_engine_hash"] == hashlib.sha256(engine.archive_member(INPUT_ROOT / "revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip", "revision_R3A_P1_executed_source.py")).hexdigest(),
        "all_four_materialized_input_packets_pass_crc_and_hash": bool(
            len(inputs["audit"]) == 4
            and inputs["audit"][["hash_matches", "crc_passes"]].all().all()
        ),
        "stage3a_parent_hash_is_preserved_from_stage3g": bool(
            re.fullmatch(r"[0-9a-f]{64}", inputs["stage3a_hash"])
        ),
        "opaque_identifier_namespace_is_locked_v1_1": inputs["opaque_identifier_namespace"] == OPAQUE_IDENTIFIER_NAMESPACE,
        "all_1225_regenerated_original_candidate_tokens_match_frozen_full_pool": bool(
            len(inputs["opaque_identifier_anchor"]) == 1225
            and inputs["opaque_identifier_anchor"]["_merge"].eq("both").all()
            and inputs["opaque_identifier_anchor"]["token_matches"].all()
        ),
        "shard_count_is_56": len(shard_audit) == EXPECTED_SHARDS,
        "every_shard_passes_crc_and_report_gates": bool(shard_audit[["crc_passes", "report_all_gates_passed"]].all().all()),
        "trajectory_count_is_924": folds["trajectory_id"].nunique() == EXPECTED_TRAJECTORIES,
        "fold_count_is_4620": len(folds) == EXPECTED_FOLDS,
        "prediction_count_is_161700": len(aggregate_tables["predictions"]) == EXPECTED_PREDICTIONS,
        "selection_count_is_31360": len(aggregate_tables["selections"]) == EXPECTED_SELECTIONS,
        "selector_call_count_is_4480": len(aggregate_tables["selector_calls"]) == EXPECTED_SELECTOR_CALLS,
        "candidate_audit_count_is_156800": len(aggregate_tables["candidate_audits"]) == EXPECTED_CANDIDATE_AUDITS,
        "normalizer_count_is_4620": len(normalizers) == EXPECTED_NORMALIZERS,
        "recall_count_is_32340": len(aggregate_tables["recalls"]) == EXPECTED_RECALLS,
        "confusion_count_is_226380": len(aggregate_tables["confusions"]) == EXPECTED_CONFUSIONS,
        "coverage_count_is_4480": len(aggregate_tables["coverage"]) == EXPECTED_COVERAGE,
        "all_four_temporal_splits_are_present": set(folds["split_id"].unique()) == set(SPLITS),
        "all_strategies_are_present": set(folds["strategy"].unique()) == DETERMINISTIC_STRATEGIES | {"RANDOM_UNIFORM"},
        "every_trajectory_has_five_target_sessions": bool(folds.groupby("trajectory_id")["target_session"].nunique().eq(5).all()),
        "all_fixed_tests_have_35_repetitions": bool(folds["test_repetitions"].eq(35).all()),
        "all_selected_records_are_candidates": bool(aggregate_tables["selections"]["selected_record_is_candidate"].all()),
        "no_fixed_test_record_was_selected": bool((~aggregate_tables["selections"]["selected_record_is_fixed_test"]).all()),
        "no_future_session_was_used": bool((~folds["future_session_used"]).all() and (~aggregate_tables["selector_calls"]["future_session_used"]).all()),
        "fixed_test_never_entered_history": bool((~folds["fixed_test_entered_history"]).all()),
        "all_metrics_are_finite_and_in_range": bool(np.isfinite(metrics.to_numpy(float)).all() and metrics.ge(0).all().all() and metrics.le(1).all().all()),
        "balanced_accuracy_equals_accuracy_in_all_folds": bool(folds["balanced_accuracy_equals_accuracy"].all()),
        "normalizers_are_finite_positive_float32": bool(
            np.isfinite(normalizers[["minimum_mean", "maximum_mean", "minimum_std", "maximum_std"]].to_numpy(float)).all()
            and normalizers["minimum_std"].gt(0).all()
            and normalizers["minimum_valid_count"].gt(0).all()
            and normalizers["means_dtype"].eq("float32").all()
            and normalizers["stds_dtype"].eq("float32").all()
            and normalizers["training_array_dtype"].eq("float32").all()
            and normalizers["model_coefficient_dtype"].eq("float32").all()
        ),
        "all_compute_telemetry_is_finite_nonnegative": bool(
            np.isfinite(aggregate_tables["telemetry"][telemetry_columns].to_numpy(float)).all()
            and aggregate_tables["telemetry"][telemetry_columns].ge(0).all().all()
        ),
        "random_cells_have_30_seed_replicates": bool(
            participant_summary.loc[participant_summary["strategy"].eq("RANDOM_UNIFORM"), "random_seed_replicates"].eq(30).all()
        ),
        "deterministic_cells_have_one_trajectory": bool(
            participant_summary.loc[~participant_summary["strategy"].eq("RANDOM_UNIFORM"), "random_seed_replicates"].eq(1).all()
        ),
        "each_able_bodied_summary_cell_has_six_participants": bool(able["participants"].eq(6).all()),
        "p07_is_descriptive_case_only": bool(p07["case_analysis"].all()),
        "original_split_frozen_fold_anchor_has_105_rows": len(anchor) == 105,
        "original_split_frozen_fold_join_is_complete": bool(anchor["_merge"].eq("both").all()),
        "original_split_maximum_fold_metric_difference_below_1e_12": float(anchor["absolute_metric_difference"].max()) < 1e-12,
        "original_split_frozen_selection_anchor_has_70_groups": len(selection_anchor) == 70,
        "original_split_selected_repetition_identities_match_frozen": bool(selection_anchor["selected_repetition_identity_sets_match"].all()),
        "participant_contrasts_are_estimands_only": contrasts["scientific_role"].eq("PARTICIPANT_LEVEL_ESTIMANDS_ONLY_NO_INFERENCE_UNTIL_R7").all(),
        "raw_hdf5_data_was_not_accessed": True,
        "new_statistical_test_was_not_run": True,
        "p07_was_not_used_for_inference": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]
    report = {
        "stage": "REVISION_R5B_RIDGE_TEMPORAL_SPLIT_SENSITIVITY",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "r5a_packet_sha256": R5A_PACKET_SHA256,
        "r3a_p1_packet_sha256": R3A_P1_PACKET_SHA256,
        "stage5b_packet_sha256": STAGE5B_PACKET_SHA256,
        "stage3g_packet_sha256": inputs["stage3g_hash"],
        "stage3a_packet_sha256": inputs["stage3a_hash"],
        "opaque_identifier_namespace": inputs["opaque_identifier_namespace"],
        "opaque_identifier_frozen_anchor_count": len(inputs["opaque_identifier_anchor"]),
        "shards": len(shard_audit),
        "trajectories": int(folds["trajectory_id"].nunique()),
        "folds": len(folds),
        "predictions": len(aggregate_tables["predictions"]),
        "selections": len(aggregate_tables["selections"]),
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "raw_hdf5_accessed": False,
        "new_statistical_test_run": False,
        "final_decision": "PASS_TO_REVISION_R5C_WITHIN_SESSION_DRIFT_AUDIT" if not failed else "REVISION_R5B_FINAL_AUDIT_FAILED",
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
    }
    atomic_json(report, FINAL_ROOT / "revision_R5B_final_report.json")
    shutil.copy2(Path(__file__), FINAL_ROOT / "revision_R5B_executed_source.py")
    manifest_rows = []
    for path in sorted(FINAL_ROOT.rglob("*")):
        if path.is_file() and path.name != "revision_R5B_output_manifest.csv":
            manifest_rows.append(
                {"relative_path": path.relative_to(FINAL_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": engine.sha256_file(path)}
            )
    atomic_csv(pd.DataFrame(manifest_rows), FINAL_ROOT / "revision_R5B_output_manifest.csv")
    return report, able, p07


def main() -> None:
    print("=" * 108)
    print("REVISION R5B — RIDGE TEMPORAL-SPLIT SENSITIVITY")
    print("=" * 108)
    print("Execution device: CPU")
    print("Raw HDF5 accessed: False")
    print("New reviewer experiment: True")
    print("New inferential statistical test: False")
    print("Temporal splits: 4")
    print("Expected shards:", EXPECTED_SHARDS)
    print("Expected trajectories:", EXPECTED_TRAJECTORIES)
    print("Expected folds:", EXPECTED_FOLDS)
    print("CPU workers:", CPU_WORKERS)
    print()
    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified R5A, R3A-P1, Stage 5B, and Stage 3G inputs...")
    print("Stage 3A identifiers: deterministic v1.1 regeneration with 1,225-token frozen anchor")
    inputs = prepare_inputs()
    features = np.load(INPUT_ROOT / "stage5b_rms_repetition_sequences.npy", mmap_mode="r", allow_pickle=False)
    main_valid = np.load(INPUT_ROOT / "stage5b_main_valid_repetition_sequences.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != (2940, 37, 64) or main_valid.shape != features.shape:
        raise RuntimeError("Stage 5B feature/mask shape contract failed")
    manifest = inputs["manifest"].copy()
    for column in ["query_budget", "random_seed_index", "random_seed"]:
        manifest[column] = pd.to_numeric(manifest[column], errors="raise").astype(int)
    shard_manifest = build_shard_manifest(manifest)
    atomic_csv(manifest, INPUT_ROOT / "revision_R5B_execution_manifest.csv")
    atomic_csv(shard_manifest, INPUT_ROOT / "revision_R5B_shard_execution_manifest.csv")
    expected_ids = set(shard_manifest["shard_id"])
    completed, duplicates = discover_completed(expected_ids)
    print(f"Restored completed shard checkpoints: {len(completed)}/{EXPECTED_SHARDS}")
    pending = shard_manifest.loc[~shard_manifest["shard_id"].isin(set(completed))].copy()
    newly_completed = 0
    if len(pending):
        context = mp.get_context("fork")
        initializer_args = (
            str(INPUT_ROOT / "stage5b_rms_repetition_sequences.npy"),
            str(INPUT_ROOT / "stage5b_main_valid_repetition_sequences.npy"),
            str(INPUT_ROOT / "revision_R5B_metadata_protocol_aligned.csv"),
            str(INPUT_ROOT / "revision_R5B_temporal_split_membership.csv"),
            str(INPUT_ROOT / "revision_R5B_execution_manifest.csv"),
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=CPU_WORKERS,
            mp_context=context,
            initializer=initialize_worker,
            initargs=initializer_args,
        ) as executor:
            futures = {
                executor.submit(execute_shard_worker, row._asdict()): row.shard_id
                for row in pending.itertuples(index=False)
            }
            for future in concurrent.futures.as_completed(futures):
                shard_id = futures[future]
                result = future.result()
                packet = Path(result["packet"])
                digest = result["sha256"]
                remote_basename = f"{shard_id}__{digest}.zip"
                remote_path = REMOTE_SHARDS + "/" + remote_basename
                if not engine.roundtrip_remote_file(packet, remote_path, digest):
                    raise RuntimeError(f"Shard remote round-trip failed: {shard_id}")
                completed[shard_id] = {
                    "shard_id": shard_id,
                    "sha256": digest,
                    "remote_basename": remote_basename,
                    "remote_path": remote_path,
                }
                newly_completed += 1
                print(
                    f"SHARD PASS | completed={len(completed):02d}/{EXPECTED_SHARDS} | "
                    f"{shard_id} | runtime={result['runtime_seconds']:.1f}s",
                    flush=True,
                )
    if len(completed) != EXPECTED_SHARDS:
        decision = "PARTIAL_PASS_RESUME_REVISION_R5B_SAME_NOTEBOOK"
        digest = make_progress_packet(shard_manifest, completed, duplicates, decision)
        print("Progress packet SHA-256:", digest)
        print("FINAL DECISION:", decision)
        return
    make_progress_packet(shard_manifest, completed, duplicates, "ALL_SHARDS_COMPLETE_FINAL_AGGREGATION_STARTED")
    print("All R5B shards complete. Restoring and aggregating final evidence...")
    report, able, p07 = aggregate(inputs, shard_manifest, completed)
    if report["failed_readiness_gates"]:
        raise RuntimeError(f"R5B final gates failed: {report['failed_readiness_gates']}")
    packet_crc = engine.make_zip(FINAL_ROOT, FINAL_PACKET, "Revision_R5B_Ridge_Temporal_Split_Sensitivity")
    packet_sha = engine.sha256_file(FINAL_PACKET)
    remote_verified = engine.roundtrip_remote_file(FINAL_PACKET, REMOTE_OUTPUT + "/" + FINAL_PACKET.name, packet_sha)
    engine.cleanup_secret()
    print()
    print("=" * 108)
    print("REVISION R5B — FINAL DESCRIPTIVE SUMMARY")
    print("=" * 108)
    print("Able-bodied summary:")
    print(able.to_string(index=False))
    print()
    print("P07 descriptive summary:")
    print(p07.to_string(index=False))
    print()
    print("Shards:", report["shards"])
    print("Trajectories:", report["trajectories"])
    print("Folds:", report["folds"])
    print("Predictions:", report["predictions"])
    print("Selections:", report["selections"])
    print("Failed readiness gates:", report["failed_readiness_gates"] or "None")
    print("Packet CRC pass:", packet_crc)
    print("Packet:", FINAL_PACKET)
    print("Packet SHA-256:", packet_sha)
    print("Remote round-trip verified:", remote_verified)
    print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 3))
    if not packet_crc or not remote_verified:
        raise RuntimeError("R5B final packet persistence failed")
    print()
    print("FINAL DECISION: PASS_TO_REVISION_R5C_WITHIN_SESSION_DRIFT_AUDIT")


if __name__ == "__main__":
    main()
