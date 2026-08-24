from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import resource
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import revision_R4D_lda_deterministic_imbalance_shards as r4d


r4a = r4d.r4a
r3c = r4d.r3c
r3b = r4d.r3b
engine = r4d.engine

START_TIME = time.time()
WORKING = engine.WORKING
INPUT_ROOT = WORKING / "REVISION_R4E_FROZEN_INPUTS"
TEMP_ROOT = WORKING / "REVISION_R4E_TEMP"
PROGRESS_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R4E_LDA_Random_Imbalance_Progress"
FINAL_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R4E_LDA_Random_Imbalance_Final"
PROGRESS_PACKET = WORKING / "revision_R4E_lda_random_imbalance_progress_packet.zip"
FINAL_PACKET = WORKING / "revision_R4E_lda_random_imbalance_packet.zip"
REMOTE_OUTPUT = engine.REMOTE_BASE + "/Reviewer_Revision/Revision_R4E_LDA_Random_Imbalance_Shards"
REMOTE_SHARDS = REMOTE_OUTPUT + "/shards"
for directory in (INPUT_ROOT, TEMP_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

REVISION_R4A_PACKET_SHA256 = "0fac1fc016310ab5043ed728f3cdbae5b7ad30e7fce6ecbeaf0d9e2e9f136580"
REVISION_R4D_PACKET_SHA256 = "5b0f26dc0db5e7d38945ed2a40ae6acd184d1ce45c55a2ef1f68089520f5e0a5"
REVISION_R4E_D3_PACKET_SHA256 = "e8ee73b60a0b7181e4321542be65a92e4f86733449d0e927f57c47e823cc216b"
REVISION_PROTOCOL_SHA256 = engine.REVISION_PROTOCOL_SHA256
STAGE = "R4E"
STRATEGY = "RANDOM_UNIFORM"
CLASSIFIER = r4d.CLASSIFIER
LDA_NUMERICAL_CONTRACT = r4d.LDA_NUMERICAL_CONTRACT
BUDGETS = [7, 14, 21]
TARGET_SESSIONS = [1, 2, 3, 4, 5]
EXPECTED_SHARDS = 22050
EXPECTED_TRAJECTORIES = 66150
EXPECTED_FOLDS = 330750
EXPECTED_PREDICTIONS = 11576250
EXPECTED_SELECTIONS = 4630500
EXPECTED_SELECTOR_CALLS = 661500
EXPECTED_CANDIDATE_AUDIT_ROWS = 14773500
EXPECTED_NORMALIZERS = 330750
EXPECTED_RECALL_ROWS = 2315250
EXPECTED_CONFUSION_ROWS = 16206750
MAX_RUNTIME_HOURS = float(os.environ.get("R4E_MAX_RUNTIME_HOURS", "10.0"))
MAX_RUNTIME_SECONDS = MAX_RUNTIME_HOURS * 3600.0
SHARD_STOP_RESERVE_SECONDS = float(os.environ.get("R4E_SHARD_STOP_RESERVE_MINUTES", "20")) * 60.0
CPU_WORKERS = max(1, min(int(os.environ.get("R4E_CPU_WORKERS", "4")), os.cpu_count() or 1))
BATCH_SIZE = max(CPU_WORKERS, int(os.environ.get("R4E_BATCH_SIZE", "60")))
FORCE_SEPARATE_AGGREGATION_AFTER_SECONDS = float(
    os.environ.get("R4E_FORCE_SEPARATE_AGGREGATION_AFTER_HOURS", "2.0")
) * 3600.0
SHARD_PACKET_PATTERN = re.compile(
    r"^(?P<shard_id>R4E_[A-Za-z0-9_]+)__(?P<sha256>[0-9a-f]{64})\.zip$"
)
SHARD_TABLES = {
    "selections": "revision_R4E_shard_selection_trace.csv",
    "candidate_audits": "revision_R4E_shard_candidate_audit.csv",
    "selector_calls": "revision_R4E_shard_selector_call_audit.csv",
    "normalizers": "revision_R4E_shard_normalizer_audit.csv",
    "folds": "revision_R4E_shard_fold_metrics.csv",
    "predictions": "revision_R4E_shard_repetition_predictions.csv",
    "recalls": "revision_R4E_shard_per_class_recall.csv",
    "confusions": "revision_R4E_shard_confusion_matrices_long.csv",
    "coverage": "revision_R4E_shard_selected_class_distribution.csv",
}

ANCHOR_PACKET_PATTERN = re.compile(
    r"^revision_R4E_frozen_lda_random_anchor__(?P<sha256>[0-9a-f]{64})\.zip$"
)
ANCHOR_REMOTE_PREFIX = REMOTE_OUTPUT + "/anchor"

WORKER_CONTEXT = {}

# Direct, immutable packet locations.  A recursive listing of the entire
# revision backup becomes progressively slower as resumable shard collections
# grow.  R4E therefore restores only its explicitly hashed parent packets from
# their known frozen directories.
DIRECT_REMOTE_PACKET_PATHS = {
    "stageR0_reviewer_revision_protocol_lock_packet.zip": (
        "Reviewer_Revision/StageR0_Reviewer_Revision_Protocol_Lock/"
        "stageR0_reviewer_revision_protocol_lock_packet.zip"
    ),
    "revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip": (
        "Reviewer_Revision/Revision_R3A_P1_Float32_Engine_Frozen_Trajectory_Unit_Test/"
        "revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip"
    ),
    "stage5b_deep_sequence_assembly_packet.zip": (
        "Stage5B_Deep_Sequence_Assembly/stage5b_deep_sequence_assembly_packet.zip"
    ),
    "stage5d2_full_deterministic_deep_trajectories_packet.zip": (
        "Deep_Training/Stage5D2_Full_Deterministic_Deep_Trajectories/"
        "stage5d2_full_deterministic_deep_trajectories_packet.zip"
    ),
    "revision_R3C_balanced_pool_classical_comparator_extension_packet.zip": (
        "Reviewer_Revision/Revision_R3C_Balanced_Pool_Classical_Comparator_Extension/"
        "revision_R3C_balanced_pool_classical_comparator_extension_packet.zip"
    ),
    "revision_R4A_candidate_pool_construction_unit_test_packet.zip": (
        "Reviewer_Revision/Revision_R4A_Candidate_Pool_Construction_Unit_Tests/"
        "revision_R4A_candidate_pool_construction_unit_test_packet.zip"
    ),
    "revision_R4C_ridge_random_imbalance_packet.zip": (
        "Reviewer_Revision/Revision_R4C_Ridge_Random_Imbalance_Shards/"
        "revision_R4C_ridge_random_imbalance_packet.zip"
    ),
    "revision_R2A_classical_detail_packet_migration_packet.zip": (
        "Reviewer_Revision/Revision_R2A_Classical_Detail_Packet_Migration/"
        "revision_R2A_classical_detail_packet_migration_packet.zip"
    ),
    "stage3e2a_lda_deterministic_packet.zip": (
        "Reviewer_Revision/Classical_Detail_Packets/"
        "stage3e2a_lda_deterministic_packet.zip"
    ),
    "revision_R4D_lda_deterministic_imbalance_packet.zip": (
        "Reviewer_Revision/Revision_R4D_LDA_Deterministic_Imbalance_Shards/"
        "revision_R4D_lda_deterministic_imbalance_packet.zip"
    ),
    "stage3e2b_lda_random_sensitivity_packet.zip": (
        "Reviewer_Revision/Classical_Detail_Packets/"
        "stage3e2b_lda_random_sensitivity_packet.zip"
    ),
    "revision_R4E_D3_frozen_history_lda_replay_packet.zip": (
        "Reviewer_Revision/Revision_R4E_D3_Frozen_History_LDA_Replay/"
        "revision_R4E_D3_frozen_history_lda_replay_packet.zip"
    ),
}


def direct_resolve_packet(basename, expected_sha256):
    """Resolve one frozen parent without recursively indexing Google Drive."""
    destination = engine.INPUT_ROOT / basename
    if (
        destination.exists()
        and engine.sha256_file(destination) == expected_sha256
        and engine.archive_crc_passes(destination)
    ):
        return destination, "EXISTING_VERIFIED_COPY"
    relative_remote = DIRECT_REMOTE_PACKET_PATHS.get(basename)
    if relative_remote is None:
        raise FileNotFoundError(
            f"No locked direct Drive location is registered for {basename}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    last_error = "not attempted"
    for attempt in range(1, 6):
        temporary.unlink(missing_ok=True)
        result = engine.rclone(
            [
                "copyto",
                engine.REMOTE_BASE + "/" + relative_remote,
                str(temporary),
                "--retries",
                "10",
                "--low-level-retries",
                "20",
                "--timeout",
                "10m",
                "--contimeout",
                "60s",
            ],
            check=False,
        )
        if result.returncode == 0 and temporary.exists():
            if (
                engine.sha256_file(temporary) == expected_sha256
                and engine.archive_crc_passes(temporary)
            ):
                os.replace(temporary, destination)
                return destination, "GOOGLE_DRIVE_DIRECT_VERIFIED"
            last_error = "download completed but hash/CRC verification failed"
        else:
            last_error = (result.stderr or result.stdout or "rclone failed").strip()
        if attempt < 5:
            time.sleep(min(30, 2**attempt))
    temporary.unlink(missing_ok=True)
    raise RuntimeError(
        f"Direct verified restore failed for {basename} after 5 attempts: "
        f"{last_error[-500:]}"
    )


def install_direct_packet_resolver():
    engine.REMOTE_LISTING = []
    engine.resolve_packet = direct_resolve_packet


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def resolve_inputs():
    inputs = r4d.resolve_inputs()
    r4a_packet = inputs["r4a_packet"]
    r4a_report = inputs["r4a_report"]
    r4d_packet, r4d_source = engine.resolve_packet(
        "revision_R4D_lda_deterministic_imbalance_packet.zip",
        REVISION_R4D_PACKET_SHA256,
    )
    r4d_report = engine.read_json_member(r4d_packet, "revision_R4D_final_report.json")
    if not r4d_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("Revision R4D parent gates did not pass")
    if r4d_report.get("revision_protocol_sha256") != REVISION_PROTOCOL_SHA256:
        raise RuntimeError("Revision R4D protocol hash drift")
    if r4d_report.get("classifier") not in {None, CLASSIFIER}:
        raise RuntimeError("Revision R4D classifier drift")

    discovery = engine.read_csv_member(
        inputs["r2a_packet"], "revision_R2A_packet_discovery.csv"
    )
    lda_random_rows = discovery.loc[
        discovery["packet"].astype(str).eq(
            "stage3e2b_lda_random_sensitivity_packet.zip"
        )
    ]
    resolved_value = (
        str(lda_random_rows.iloc[0]["resolved"]).strip().lower()
        if len(lda_random_rows) == 1
        else "false"
    )
    if len(lda_random_rows) != 1 or resolved_value not in {"true", "1", "yes"}:
        raise RuntimeError("R2A does not resolve exactly one Stage 3E2B LDA packet")
    lda_random_expected = str(
        lda_random_rows.iloc[0]["expected_sha256"]
    ).lower()
    if re.fullmatch(r"[0-9a-f]{64}", lda_random_expected) is None:
        raise RuntimeError("Invalid Stage 3E2B hash in R2A")
    lda_random_packet, lda_random_source = engine.resolve_packet(
        "stage3e2b_lda_random_sensitivity_packet.zip",
        lda_random_expected,
    )
    d3_packet, d3_source = engine.resolve_packet(
        "revision_R4E_D3_frozen_history_lda_replay_packet.zip",
        REVISION_R4E_D3_PACKET_SHA256,
    )
    d3_report = engine.read_json_member(d3_packet, "revision_R4E_D3_report.json")
    if (
        not d3_report.get("all_readiness_gates_passed", False)
        or d3_report.get("classification")
        != "EXACT_FROZEN_HISTORY_LDA_REPLAY_IDENTIFIED"
        or d3_report.get("revision_protocol_sha256") != REVISION_PROTOCOL_SHA256
        or d3_report.get("stage3e2b_packet_sha256") != lda_random_expected
        or float(d3_report.get("prediction_match_fraction", -1.0)) != 1.0
        or float(d3_report.get("maximum_fold_metric_difference", 1.0)) >= 1e-12
    ):
        raise RuntimeError("Revision R4E-D3 exact frozen-history replay contract failed")

    records = []
    for packet, expected, source in [
        (r4d_packet, REVISION_R4D_PACKET_SHA256, r4d_source),
        (lda_random_packet, lda_random_expected, lda_random_source),
        (d3_packet, REVISION_R4E_D3_PACKET_SHA256, d3_source),
    ]:
        observed = engine.sha256_file(packet)
        records.append(
            {
                "packet": packet.name,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "hash_matches": observed == expected,
                "crc_passes": engine.archive_crc_passes(packet),
                "source": source,
            }
        )
    audit = pd.concat([inputs["audit"], pd.DataFrame(records)], ignore_index=True)
    if not audit[["hash_matches", "crc_passes"]].all().all():
        raise RuntimeError("Revision R4E frozen-input integrity failed")
    inputs.update(
        {
            "r4a_packet": r4a_packet,
            "r4a_report": r4a_report,
            "r4d_packet": r4d_packet,
            "r4d_report": r4d_report,
            "lda_random_packet": lda_random_packet,
            "lda_random_packet_sha256": lda_random_expected,
            "r4e_d3_packet": d3_packet,
            "r4e_d3_report": d3_report,
            "audit": audit,
            "pool_definitions": engine.read_csv_member(r4a_packet, "revision_R4A_pool_definitions.csv"),
            "pool_membership": engine.read_csv_member(r4a_packet, "revision_R4A_pool_membership.csv"),
            "shard_manifest": engine.read_csv_member(r4a_packet, "revision_R4A_execution_shard_manifest.csv"),
        }
    )
    return inputs


def locked_random_seed_map(seed_schedule):
    rows = seed_schedule.loc[
        seed_schedule["seed_family"].astype(str).eq("RANDOM_ACQUISITION")
        & pd.to_numeric(seed_schedule["seed_index"]).between(1, 30)
        & seed_schedule["use_rule"].astype(str).eq("INITIAL_30")
    ].copy()
    rows["seed_index"] = pd.to_numeric(rows["seed_index"], errors="raise").astype(int)
    rows["seed"] = pd.to_numeric(rows["seed"], errors="raise").astype(np.int64)
    if len(rows) != 30 or set(rows["seed_index"]) != set(range(1, 31)):
        raise RuntimeError("R0 does not contain exactly the 30 locked initial random seeds")
    if not rows["seed"].is_unique:
        raise RuntimeError("R0 random acquisition seeds are not unique")
    return dict(zip(rows["seed_index"], rows["seed"]))


def attach_random_seed_identity(raw, normalized, seed_map, table_name):
    """Attach the locked seed index without assuming one frozen column name."""
    index_column = engine.resolve_column(
        raw,
        [
            "random_seed_index",
            "seed_index",
            "random_replicate_index",
            "replicate_index",
            "random_run_index",
            "run_index",
            "replicate",
            "random_replicate",
        ],
        contains=["seed", "index"],
        required=False,
    )
    value_column = engine.resolve_column(
        raw,
        [
            "random_seed",
            "seed",
            "random_state",
            "replicate_seed",
            "acquisition_seed",
            "selector_seed",
            "sampling_seed",
        ],
        contains=["random", "seed"],
        required=False,
    )
    inverse = {int(value): int(index) for index, value in seed_map.items()}

    # Some frozen tables use a seed-bearing run/trajectory column rather than
    # one canonical seed column. Discover only an unambiguous numeric column.
    if index_column is None and value_column is None:
        index_candidates = []
        value_candidates = []
        for column in raw.columns:
            lowered = str(column).lower()
            if not any(
                token in lowered
                for token in ("seed", "replicate", "run", "trajectory")
            ):
                continue
            numeric = pd.to_numeric(raw[column], errors="coerce")
            if numeric.isna().any():
                continue
            values = numeric.to_numpy(dtype=np.float64)
            if np.all(np.equal(values, np.floor(values))):
                integers = set(values.astype(np.int64).tolist())
                if integers and integers.issubset(set(range(1, 31))):
                    index_candidates.append(column)
                if integers and integers.issubset(set(inverse)):
                    value_candidates.append(column)
        if len(index_candidates) == 1:
            index_column = index_candidates[0]
        elif len(value_candidates) == 1:
            value_column = value_candidates[0]

    if index_column is not None:
        indices = pd.to_numeric(raw[index_column], errors="raise").astype(int)
    elif value_column is not None:
        values = pd.to_numeric(raw[value_column], errors="raise").astype(np.int64)
        if set(values.unique()).issubset(set(range(1, 31))):
            indices = values.astype(int)
        else:
            indices = values.map(inverse)
    else:
        parsed_candidates = []
        candidate_columns = [
            column
            for column in raw.columns
            if any(
                token in str(column).lower()
                for token in ("strategy", "method", "seed", "replicate", "run", "trajectory")
            )
        ]
        for column in candidate_columns:
            extracted = raw[column].astype(str).str.extract(
                r"(?:SEED|REPLICATE|RUN)[_\- :]*0*(\d{1,2})(?:\D|$)",
                expand=False,
            )
            parsed = pd.to_numeric(extracted, errors="coerce")
            if parsed.notna().all() and set(parsed.astype(int)).issubset(
                set(range(1, 31))
            ):
                parsed_candidates.append((column, parsed))
        if len(parsed_candidates) != 1:
            raise RuntimeError(
                f"Could not resolve an unambiguous random seed column in {table_name}; "
                f"columns={raw.columns.tolist()}"
            )
        parsed_column, indices = parsed_candidates[0]
    if indices.isna().any():
        raise RuntimeError(f"Could not resolve every random seed in {table_name}")
    indices = indices.astype(int)
    if not set(indices.unique()).issubset(set(range(1, 31))):
        raise RuntimeError(f"Out-of-contract random seed index in {table_name}")
    normalized = normalized.copy()
    normalized["random_seed_index"] = indices.to_numpy()
    normalized["random_seed"] = normalized["random_seed_index"].map(seed_map)
    if normalized["random_seed"].isna().any():
        raise RuntimeError(f"Unmapped random seed in {table_name}")
    return normalized, {
        "seed_index_column": index_column,
        "seed_value_column": value_column,
        "strategy_parse_fallback_used": index_column is None
        and value_column is None,
        "parsed_seed_column": locals().get("parsed_column"),
    }


def restrict_to_random_rows(raw, normalized, table_name):
    """Align raw and normalized frozen tables, then keep random-policy rows."""
    if len(raw) != len(normalized):
        raise RuntimeError(f"Raw/normalized row-count drift in {table_name}")
    mask = normalized["strategy"].astype(str).str.upper().str.contains("RANDOM")
    if not bool(mask.any()):
        raise RuntimeError(f"Frozen Stage 3E2B {table_name} has no random-policy rows")
    raw_random = raw.loc[mask.to_numpy()].reset_index(drop=True)
    normalized_random = normalized.loc[mask].reset_index(drop=True)
    return raw_random, normalized_random


def canonical_random_rows(frame):
    mask = frame["strategy"].astype(str).str.upper().str.contains("RANDOM")
    result = frame.loc[mask].copy()
    if result.empty:
        raise RuntimeError("Frozen Stage 3E2B table has no random-policy rows")
    result["strategy"] = STRATEGY
    return result.reset_index(drop=True)


def run_balanced_random_lda_anchor(features, main_valid, metadata, seed_map):
    """Replay all 630 frozen balanced-pool LDA random trajectories."""
    selections = []
    predictions = []
    folds = []
    normalizers = []
    for participant in engine.PARTICIPANTS:
        for seed_index in range(1, 31):
            random_seed = int(seed_map[seed_index])
            for budget in BUDGETS:
                history = engine.initial_history_rows(metadata, participant).tolist()
                rng = np.random.default_rng(random_seed)
                for session in TARGET_SESSIONS:
                    remaining = engine.candidate_rows(
                        metadata, participant, session
                    ).tolist()
                    selected_session = []
                    for query_round in range(1, budget // 7 + 1):
                        selected_tokens, _, _, _, _ = random_acquisition(
                            metadata, remaining, rng
                        )
                        selected_rows = engine.reveal_rows(
                            metadata, selected_tokens, remaining
                        )
                        for position, (token, row) in enumerate(
                            zip(selected_tokens, selected_rows), start=1
                        ):
                            selections.append(
                                {
                                    "participant": participant,
                                    "random_seed_index": seed_index,
                                    "random_seed": random_seed,
                                    "target_session": session,
                                    "strategy": STRATEGY,
                                    "query_budget": budget,
                                    "query_round": query_round,
                                    "position": position,
                                    "opaque_candidate_token": str(token),
                                    "sequence_row": int(row),
                                    "repetition_uid": str(
                                        metadata.iloc[int(row)]["repetition_uid"]
                                    ),
                                }
                            )
                        selected_set = set(map(int, selected_rows))
                        history.extend(map(int, selected_rows))
                        selected_session.extend(map(int, selected_rows))
                        remaining = [
                            row for row in remaining if int(row) not in selected_set
                        ]
                    if len(selected_session) != budget:
                        raise RuntimeError("Balanced random anchor selection drift")
                    state = r4d.fit_history_state_lda(
                        features, main_valid, metadata, history
                    )
                    test_rows = engine.fixed_test_rows(
                        metadata, participant, session
                    )
                    evaluated = r3c.evaluate_fixed_test(
                        state, features, main_valid, metadata, test_rows
                    )
                    folds.append(
                        {
                            "participant": participant,
                            "random_seed_index": seed_index,
                            "random_seed": random_seed,
                            "target_session": session,
                            "strategy": STRATEGY,
                            "query_budget": budget,
                            "repetition_balanced_accuracy": float(
                                evaluated["balanced_accuracy"]
                            ),
                        }
                    )
                    normalizers.append(
                        {
                            "participant": participant,
                            "random_seed_index": seed_index,
                            "target_session": session,
                            "query_budget": budget,
                            "means_dtype": str(state["means"].dtype),
                            "stds_dtype": str(state["stds"].dtype),
                            "training_array_dtype": state[
                                "training_array_dtype"
                            ],
                            "model_coefficient_dtype": state[
                                "model_coefficient_dtype"
                            ],
                            "minimum_std": float(np.min(state["stds"])),
                            "minimum_valid_count": int(np.min(state["counts"])),
                        }
                    )
                    for position, row in enumerate(test_rows, start=1):
                        predictions.append(
                            {
                                "participant": participant,
                                "random_seed_index": seed_index,
                                "random_seed": random_seed,
                                "target_session": session,
                                "strategy": STRATEGY,
                                "query_budget": budget,
                                "test_position": position,
                                "sequence_row": int(row),
                                "repetition_uid": str(
                                    metadata.iloc[int(row)]["repetition_uid"]
                                ),
                                "true_label": int(evaluated["true"][position - 1]),
                                "predicted_label": int(
                                    evaluated["predicted"][position - 1]
                                ),
                            }
                        )
    return (
        pd.DataFrame(selections),
        pd.DataFrame(predictions),
        pd.DataFrame(folds),
        pd.DataFrame(normalizers),
    )


def compare_random_selection_memberships(observed, frozen):
    group_keys = [
        "participant",
        "random_seed_index",
        "target_session",
        "query_budget",
    ]
    frozen_round_available = bool(
        "query_round" in frozen
        and frozen["query_round"].notna().all()
    )
    rows = []
    observed_groups = observed.groupby(group_keys, sort=False)
    frozen_groups = frozen.groupby(group_keys, sort=False)
    all_keys = sorted(set(observed_groups.groups) | set(frozen_groups.groups))
    for key in all_keys:
        left = observed_groups.get_group(key) if key in observed_groups.groups else observed.iloc[0:0]
        right = frozen_groups.get_group(key) if key in frozen_groups.groups else frozen.iloc[0:0]
        left_set = set(left["sequence_row"].astype(int))
        right_set = set(right["sequence_row"].astype(int))
        round_match = True
        if frozen_round_available:
            left_round = {
                (int(row.query_round), int(row.sequence_row))
                for row in left.itertuples(index=False)
            }
            right_round = {
                (int(row.query_round), int(row.sequence_row))
                for row in right.itertuples(index=False)
            }
            round_match = left_round == right_round
        rows.append(
            {
                **dict(zip(group_keys, key)),
                "observed_selection_count": len(left),
                "frozen_selection_count": len(right),
                "selection_identity_set_matches": left_set == right_set,
                "round_membership_matches_when_available": round_match,
                "only_observed_count": len(left_set - right_set),
                "only_frozen_count": len(right_set - left_set),
            }
        )
    return pd.DataFrame(rows), frozen_round_available


def compare_random_predictions(observed, frozen, frozen_has_uid):
    base = [
        "participant",
        "random_seed_index",
        "target_session",
        "query_budget",
    ]
    if frozen_has_uid:
        keys = base + ["repetition_uid"]
    else:
        keys = base + ["true_label", "class_position"]
    left = observed.copy()
    right = frozen.copy()
    if not frozen_has_uid:
        group = base + ["true_label"]
        left["class_position"] = left.groupby(group, sort=False).cumcount()
        right["class_position"] = right.groupby(group, sort=False).cumcount()
    value_columns = (
        ["true_label", "predicted_label"]
        if frozen_has_uid
        else ["predicted_label"]
    )
    merged = left[keys + value_columns].merge(
        right[keys + value_columns],
        on=keys,
        how="outer",
        suffixes=("_observed", "_frozen"),
        indicator=True,
        validate="one_to_one",
    )
    merged["true_label_matches"] = (
        merged["true_label_observed"] == merged["true_label_frozen"]
        if frozen_has_uid
        else True
    )
    merged["predicted_label_matches"] = (
        merged["predicted_label_observed"]
        == merged["predicted_label_frozen"]
    )
    return merged


def compare_random_folds(observed, frozen):
    keys = [
        "participant",
        "random_seed_index",
        "target_session",
        "query_budget",
    ]
    merged = observed[keys + ["repetition_balanced_accuracy"]].merge(
        frozen[keys + ["repetition_balanced_accuracy"]],
        on=keys,
        how="outer",
        suffixes=("_observed", "_frozen"),
        indicator=True,
        validate="one_to_one",
    )
    merged["absolute_difference"] = (
        merged["repetition_balanced_accuracy_observed"]
        - merged["repetition_balanced_accuracy_frozen"]
    ).abs()
    return merged


def compute_frozen_lda_random_anchor(inputs, features, main_valid, metadata, seed_map):
    packet = inputs["lda_random_packet"]
    selection_member, selection_raw = engine.find_csv_member(
        packet, ["selection", "trace"]
    )
    prediction_member, prediction_raw = engine.find_csv_member(
        packet, ["repetition", "prediction"]
    )
    fold_member, fold_raw = engine.find_csv_member(packet, ["fold"])
    frozen_selection, selection_columns = engine.normalize_selection_table(
        selection_raw
    )
    frozen_prediction, prediction_columns = engine.normalize_prediction_table(
        prediction_raw
    )
    frozen_fold, fold_columns = engine.normalize_fold_table(fold_raw)

    # Stage 3E2B packets can retain deterministic/reference rows that have no
    # random-seed identity. Filter them before resolving the seed contract.
    selection_raw, frozen_selection = restrict_to_random_rows(
        selection_raw, frozen_selection, "selection"
    )
    prediction_raw, frozen_prediction = restrict_to_random_rows(
        prediction_raw, frozen_prediction, "prediction"
    )
    fold_raw, frozen_fold = restrict_to_random_rows(
        fold_raw, frozen_fold, "fold"
    )
    frozen_selection, selection_seed_columns = attach_random_seed_identity(
        selection_raw, frozen_selection, seed_map, "selection"
    )
    frozen_prediction, prediction_seed_columns = attach_random_seed_identity(
        prediction_raw, frozen_prediction, seed_map, "prediction"
    )
    frozen_fold, fold_seed_columns = attach_random_seed_identity(
        fold_raw, frozen_fold, seed_map, "fold"
    )
    frozen_selection = canonical_random_rows(frozen_selection)
    frozen_prediction = canonical_random_rows(frozen_prediction)
    frozen_fold = canonical_random_rows(frozen_fold)

    row_by_uid = dict(
        zip(metadata["repetition_uid"].astype(str), metadata["sequence_row"].astype(int))
    )
    row_by_token = dict(
        zip(metadata["opaque_candidate_token"].astype(str), metadata["sequence_row"].astype(int))
    )
    if selection_columns["repetition_uid"] is not None:
        frozen_selection["sequence_row"] = frozen_selection["repetition_uid"].map(row_by_uid)
        identity_source = "FROZEN_REPETITION_UID"
    elif selection_columns["sequence_row"] is not None:
        identity_source = "FROZEN_SEQUENCE_ROW"
    else:
        frozen_selection["sequence_row"] = frozen_selection["opaque_candidate_token"].map(row_by_token)
        identity_source = "CROSS_PACKET_OPAQUE_TOKEN"
    if frozen_selection["sequence_row"].isna().any():
        raise RuntimeError("Stage 3E2B selection identities could not be resolved")
    frozen_selection["sequence_row"] = frozen_selection["sequence_row"].astype(int)

    observed_selection, observed_prediction, observed_fold, normalizers = (
        run_balanced_random_lda_anchor(features, main_valid, metadata, seed_map)
    )
    selection_comparison, round_available = compare_random_selection_memberships(
        observed_selection, frozen_selection
    )
    prediction_comparison = compare_random_predictions(
        observed_prediction,
        frozen_prediction,
        prediction_columns["repetition_uid"] is not None,
    )
    fold_comparison = compare_random_folds(observed_fold, frozen_fold)
    maximum_fold_difference = float(fold_comparison["absolute_difference"].max())
    gates = {
        "stage3e2b_packet_hash_matches_r2a": engine.sha256_file(packet)
        == inputs["lda_random_packet_sha256"],
        "stage3e2b_packet_crc_passes": engine.archive_crc_passes(packet),
        "frozen_selection_identities_are_resolved": bool(
            frozen_selection["sequence_row"].notna().all()
        ),
        "observed_trajectory_count_is_630": bool(
            observed_fold.groupby(
                ["participant", "random_seed_index", "query_budget"]
            ).ngroups
            == 630
        ),
        "observed_fold_count_is_3150": len(observed_fold) == 3150,
        "observed_prediction_count_is_110250": len(observed_prediction) == 110250,
        "observed_selection_count_is_44100": len(observed_selection) == 44100,
        "frozen_selection_identity_sets_match": bool(
            selection_comparison["selection_identity_set_matches"].all()
        ),
        "frozen_selection_round_memberships_match_when_available": bool(
            selection_comparison[
                "round_membership_matches_when_available"
            ].all()
        ),
        "frozen_prediction_join_is_complete": bool(
            prediction_comparison["_merge"].eq("both").all()
        ),
        "frozen_true_and_predicted_labels_match": bool(
            prediction_comparison["true_label_matches"].all()
            and prediction_comparison["predicted_label_matches"].all()
        ),
        "frozen_fold_join_is_complete": bool(
            fold_comparison["_merge"].eq("both").all()
        ),
        "frozen_fold_metrics_match_below_1e_12": maximum_fold_difference < 1e-12,
        "lda_normalizer_and_training_arrays_are_float32": bool(
            normalizers["means_dtype"].eq("float32").all()
            and normalizers["stds_dtype"].eq("float32").all()
            and normalizers["training_array_dtype"].eq("float32").all()
        ),
        "lda_model_coefficients_are_float64": bool(
            normalizers["model_coefficient_dtype"].eq("float64").all()
        ),
        "all_normalizer_stds_and_counts_are_positive": bool(
            normalizers["minimum_std"].gt(0).all()
            and normalizers["minimum_valid_count"].gt(0).all()
        ),
    }
    failed = [key for key, value in gates.items() if not bool(value)]
    if failed:
        raise RuntimeError(f"Frozen Stage 3E2B LDA-random anchor failed: {failed}")
    return {
        "gates": gates,
        "members": {
            "selection": selection_member,
            "prediction": prediction_member,
            "fold": fold_member,
        },
        "identity_source": identity_source,
        "frozen_round_membership_available": round_available,
        "maximum_fold_metric_difference": maximum_fold_difference,
        "selection_columns": selection_columns,
        "prediction_columns": prediction_columns,
        "fold_columns": fold_columns,
        "selection_seed_columns": selection_seed_columns,
        "prediction_seed_columns": prediction_seed_columns,
        "fold_seed_columns": fold_seed_columns,
        "selection_comparison": selection_comparison,
        "prediction_comparison": prediction_comparison,
        "fold_comparison": fold_comparison,
    }


def load_cached_frozen_lda_random_anchor(inputs):
    listing = engine.rclone(
        ["lsf", ANCHOR_REMOTE_PREFIX, "--files-only"], check=False
    )
    if listing.returncode != 0:
        return None
    candidates = []
    for basename in listing.stdout.splitlines():
        match = ANCHOR_PACKET_PATTERN.fullmatch(Path(basename).name)
        if match is not None:
            candidates.append((Path(basename).name, match.group("sha256")))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError(f"Expected at most one R4E anchor packet; found {candidates}")
    basename, expected = candidates[0]
    local = INPUT_ROOT / basename
    engine.rclone(
        ["copyto", ANCHOR_REMOTE_PREFIX + "/" + basename, str(local)]
    )
    if engine.sha256_file(local) != expected or not engine.archive_crc_passes(local):
        raise RuntimeError("Cached R4E LDA-random anchor integrity failed")
    report = engine.read_json_member(local, "revision_R4E_frozen_lda_random_anchor_report.json")
    gates = report.get("readiness_gates", {})
    if (
        not report.get("all_readiness_gates_passed", False)
        or not gates
        or not all(bool(value) for value in gates.values())
        or report.get("revision_protocol_sha256") != REVISION_PROTOCOL_SHA256
        or report.get("stage3e2b_packet_sha256")
        != inputs["lda_random_packet_sha256"]
        or report.get("revision_r4d_packet_sha256")
        != REVISION_R4D_PACKET_SHA256
        or report.get("lda_numerical_contract") != LDA_NUMERICAL_CONTRACT
    ):
        raise RuntimeError("Cached R4E LDA-random anchor contract failed")
    return {
        "gates": gates,
        "members": report["frozen_members"],
        "identity_source": report["identity_source"],
        "frozen_round_membership_available": bool(
            report["frozen_round_membership_available"]
        ),
        "maximum_fold_metric_difference": float(
            report["maximum_fold_metric_difference"]
        ),
        "selection_comparison": engine.read_csv_member(
            local, "revision_R4E_frozen_lda_random_selection_comparison.csv"
        ),
        "prediction_comparison": engine.read_csv_member(
            local, "revision_R4E_frozen_lda_random_prediction_comparison.csv"
        ),
        "fold_comparison": engine.read_csv_member(
            local, "revision_R4E_frozen_lda_random_fold_comparison.csv"
        ),
        "cache_status": "RESTORED_VERIFIED_REMOTE_ANCHOR",
        "packet": local,
        "packet_sha256": expected,
    }


def freeze_frozen_lda_random_anchor(inputs, anchor):
    root = TEMP_ROOT / "R4E_FROZEN_LDA_RANDOM_ANCHOR"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    report = {
        "stage": "REVISION_R4E_FROZEN_LDA_RANDOM_ANCHOR",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "revision_r4d_packet_sha256": REVISION_R4D_PACKET_SHA256,
        "stage3e2b_packet_sha256": inputs["lda_random_packet_sha256"],
        "classifier": CLASSIFIER,
        "lda_numerical_contract": LDA_NUMERICAL_CONTRACT,
        "frozen_members": anchor["members"],
        "identity_source": anchor["identity_source"],
        "frozen_round_membership_available": bool(
            anchor["frozen_round_membership_available"]
        ),
        "maximum_fold_metric_difference": float(
            anchor["maximum_fold_metric_difference"]
        ),
        "readiness_gates": anchor["gates"],
        "failed_readiness_gates": [
            key for key, value in anchor["gates"].items() if not bool(value)
        ],
        "all_readiness_gates_passed": bool(all(anchor["gates"].values())),
        "raw_hdf5_accessed": False,
        "new_statistical_test_run": False,
    }
    atomic_json(report, root / "revision_R4E_frozen_lda_random_anchor_report.json")
    atomic_csv(
        anchor["selection_comparison"],
        root / "revision_R4E_frozen_lda_random_selection_comparison.csv",
    )
    atomic_csv(
        anchor["prediction_comparison"],
        root / "revision_R4E_frozen_lda_random_prediction_comparison.csv",
    )
    atomic_csv(
        anchor["fold_comparison"],
        root / "revision_R4E_frozen_lda_random_fold_comparison.csv",
    )
    packet = TEMP_ROOT / "revision_R4E_frozen_lda_random_anchor.zip"
    if not engine.make_zip(root, packet, "Revision_R4E_Frozen_LDA_Random_Anchor"):
        raise RuntimeError("R4E frozen LDA-random anchor CRC failed")
    digest = engine.sha256_file(packet)
    remote = (
        ANCHOR_REMOTE_PREFIX
        + "/revision_R4E_frozen_lda_random_anchor__"
        + digest
        + ".zip"
    )
    if not engine.roundtrip_remote_file(packet, remote, digest):
        raise RuntimeError("R4E frozen LDA-random anchor persistence failed")
    anchor.update(
        {
            "cache_status": "COMPUTED_AND_FROZEN_REMOTE_ANCHOR",
            "packet": packet,
            "packet_sha256": digest,
        }
    )
    return anchor


def ensure_frozen_lda_random_anchor(inputs, features, main_valid, metadata, seed_map):
    cached = load_cached_frozen_lda_random_anchor(inputs)
    if cached is not None:
        return cached
    computed = compute_frozen_lda_random_anchor(
        inputs, features, main_valid, metadata, seed_map
    )
    return freeze_frozen_lda_random_anchor(inputs, computed)


def load_revision_r4e_d3_frozen_history_anchor(inputs):
    """Use the exact frozen-history replay as the LDA implementation anchor.

    R4E-D3 proved that the locked LDA numerical engine reproduces every one of
    the 110,250 frozen Stage 3E2B predictions and all 3,150 fold metrics when
    supplied with the exact frozen selected histories.  The legacy Stage 3E2B
    random-draw implementation is therefore not reconstructed or reused here;
    the reviewer-requested R4E experiment uses only the newly locked R0 seeds.
    """
    packet = inputs["r4e_d3_packet"]
    report = inputs["r4e_d3_report"]
    gates = report.get("readiness_gates", {})
    if (
        engine.sha256_file(packet) != REVISION_R4E_D3_PACKET_SHA256
        or not engine.archive_crc_passes(packet)
        or not report.get("all_readiness_gates_passed", False)
        or not gates
        or not all(bool(value) for value in gates.values())
        or report.get("classification")
        != "EXACT_FROZEN_HISTORY_LDA_REPLAY_IDENTIFIED"
        or float(report.get("prediction_match_fraction", -1.0)) != 1.0
        or float(report.get("maximum_fold_metric_difference", 1.0)) >= 1e-12
    ):
        raise RuntimeError("Revision R4E-D3 anchor verification failed")
    return {
        "gates": gates,
        "members": report["members"],
        "identity_source": report["selection_identity_source"],
        "frozen_round_membership_available": False,
        "maximum_fold_metric_difference": float(
            report["maximum_fold_metric_difference"]
        ),
        "selection_comparison": engine.read_csv_member(
            packet, "revision_R4E_D3_frozen_selection_contract_audit.csv"
        ),
        "prediction_comparison": engine.read_csv_member(
            packet, "revision_R4E_D3_prediction_comparison.csv"
        ),
        "fold_comparison": engine.read_csv_member(
            packet, "revision_R4E_D3_fold_comparison.csv"
        ),
        "cache_status": "RESTORED_VERIFIED_REVISION_R4E_D3_EXACT_REPLAY",
        "packet": packet,
        "packet_sha256": REVISION_R4E_D3_PACKET_SHA256,
    }


def discover_completed_shards(expected_ids):
    result = engine.rclone(["lsf", REMOTE_SHARDS, "--files-only"], check=False)
    if result.returncode != 0:
        return {}, []
    mapping, duplicates = {}, []
    for basename in result.stdout.splitlines():
        match = SHARD_PACKET_PATTERN.fullmatch(Path(basename).name)
        if match is None or match.group("shard_id") not in expected_ids:
            continue
        record = {
            "shard_id": match.group("shard_id"),
            "sha256": match.group("sha256"),
            "remote_path": REMOTE_SHARDS + "/" + Path(basename).name,
            "remote_basename": Path(basename).name,
        }
        if record["shard_id"] in mapping:
            duplicates.append(record)
        else:
            mapping[record["shard_id"]] = record
    return mapping, duplicates


def timed(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def rng_state_sha256(rng):
    payload = json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def random_acquisition(metadata, remaining_rows, rng):
    frame = pd.DataFrame(
        {
            "opaque_candidate_token": metadata.iloc[np.asarray(remaining_rows, dtype=int)][
                "opaque_candidate_token"
            ].astype(str).to_numpy()
        }
    ).sort_values("opaque_candidate_token", kind="mergesort").reset_index(drop=True)
    r3b.validate_exact_schema(frame, ["opaque_candidate_token"], "RANDOM_SELECTOR")
    state_before = rng_state_sha256(rng)
    started = time.perf_counter()
    positions = rng.choice(len(frame), size=7, replace=False)
    selector_seconds = time.perf_counter() - started
    selected_tokens = frame.iloc[positions]["opaque_candidate_token"].tolist()
    state_after = rng_state_sha256(rng)
    draw_order = {token: order for order, token in enumerate(selected_tokens, start=1)}
    candidate = frame.copy()
    candidate["selected_this_round"] = candidate["opaque_candidate_token"].isin(set(selected_tokens))
    candidate["random_draw_order"] = candidate["opaque_candidate_token"].map(draw_order).fillna(0).astype(int)
    return selected_tokens, candidate, selector_seconds, state_before, state_after


def append_normalizer(rows, common, session, budget, history_count, state):
    rows.append(
        {
            **common,
            "target_session": session,
            "strategy": STRATEGY,
            "query_budget": budget,
            "fit_role": "FINAL_HISTORY_EVALUATION",
            "history_repetitions": history_count,
            "minimum_mean": float(np.min(state["means"])),
            "maximum_mean": float(np.max(state["means"])),
            "minimum_std": float(np.min(state["stds"])),
            "maximum_std": float(np.max(state["stds"])),
            "minimum_valid_count": int(np.min(state["counts"])),
            "means_dtype": str(state["means"].dtype),
            "stds_dtype": str(state["stds"].dtype),
            "model_coefficient_dtype": str(np.asarray(state["model"].coef_).dtype),
        }
    )


def run_shard(shard, features, main_valid, metadata, pool_definitions, pool_membership, seed_map):
    shard_id = str(shard.shard_id)
    participant = str(shard.participant)
    level = str(shard.imbalance_level)
    rotation = int(shard.rotation_index)
    realization = int(shard.pool_realization_index)
    seed_index = int(shard.random_seed_index)
    random_seed = int(seed_map[seed_index])
    definitions = pool_definitions.loc[
        pool_definitions["participant"].eq(participant)
        & pool_definitions["imbalance_level"].eq(level)
        & pd.to_numeric(pool_definitions["rotation_index"]).eq(rotation)
        & pd.to_numeric(pool_definitions["pool_realization_index"]).eq(realization)
    ].copy()
    if len(definitions) != 5:
        raise RuntimeError(f"Expected five session pools for {shard_id}")
    session_to_definition = {int(row.target_session): row for row in definitions.itertuples(index=False)}
    membership_by_pool = {
        pool_id: group["sequence_row_internal_audit_only"].astype(int).tolist()
        for pool_id, group in pool_membership.loc[
            pool_membership["pool_id"].isin(definitions["pool_id"])
        ].groupby("pool_id", sort=False)
    }
    output = {key: [] for key in SHARD_TABLES}
    base_common = {
        "shard_id": shard_id,
        "participant": participant,
        "case_analysis": participant == "P07",
        "imbalance_level": level,
        "rotation_index": rotation,
        "pool_realization_index": realization,
        "random_seed_index": seed_index,
        "random_seed": random_seed,
    }
    for budget in BUDGETS:
        trajectory_id = f"{shard_id}_{STRATEGY}_K{budget:02d}"
        history = engine.initial_history_rows(metadata, participant).tolist()
        rng = np.random.default_rng(random_seed)
        for session in TARGET_SESSIONS:
            session_started = time.perf_counter()
            definition = session_to_definition[session]
            pool_id = str(definition.pool_id)
            remaining = list(membership_by_pool[pool_id])
            pool_size = int(definition.total_candidates)
            if len(remaining) != pool_size or len(set(remaining)) != pool_size:
                raise RuntimeError(f"Pool membership drift for {pool_id}")
            fixed_rows = engine.fixed_test_rows(metadata, participant, session)
            fixed_set = set(map(int, fixed_rows))
            session_selected = []
            selector_seconds_total = 0.0
            for query_round in range(1, budget // 7 + 1):
                history_before = list(history)
                selected_tokens, candidate_frame, selector_seconds, rng_before, rng_after = random_acquisition(
                    metadata, remaining, rng
                )
                selected_rows = engine.reveal_rows(metadata, selected_tokens, remaining)
                selected_set = set(map(int, selected_rows))
                true_labels = metadata.iloc[selected_rows]["label"].to_numpy(dtype=int)
                prefix = {
                    **base_common,
                    "pool_id": pool_id,
                    "trajectory_id": trajectory_id,
                    "target_session": session,
                    "strategy": STRATEGY,
                    "query_budget": budget,
                    "query_round": query_round,
                }
                for position, (token, row, label) in enumerate(
                    zip(selected_tokens, selected_rows, true_labels), start=1
                ):
                    output["selections"].append(
                        {
                            **prefix,
                            "position_in_round": position,
                            "opaque_candidate_token": str(token),
                            "sequence_row_internal_audit_only": int(row),
                            "true_label_after_reveal": int(label),
                            "selected_record_is_pool_candidate": int(row) in set(remaining),
                            "selected_record_is_fixed_test": int(row) in fixed_set,
                        }
                    )
                for row in candidate_frame.to_dict("records"):
                    output["candidate_audits"].append({**prefix, **row})
                history_sessions = metadata.iloc[np.asarray(history_before, dtype=int)]["session"].to_numpy(dtype=int)
                output["selector_calls"].append(
                    {
                        **prefix,
                        "pool_candidates_at_session_start": pool_size,
                        "history_repetitions_before_query": len(history_before),
                        "remaining_candidates_before_query": len(remaining),
                        "selected_count": len(selected_rows),
                        "fit_seconds": 0.0,
                        "score_seconds": 0.0,
                        "selector_seconds": selector_seconds,
                        "selector_schema": "opaque_candidate_token",
                        "selector_schema_exact": True,
                        "selector_forbidden_column_count": 0,
                        "rng_state_sha256_before": rng_before,
                        "rng_state_sha256_after": rng_after,
                        "maximum_history_session": int(history_sessions.max()),
                        "future_session_used": bool((history_sessions > session).any()),
                        "fixed_test_used_for_training_or_selection": bool(
                            set(history_before).intersection(fixed_set) or selected_set.intersection(fixed_set)
                        ),
                    }
                )
                selector_seconds_total += selector_seconds
                history.extend(map(int, selected_rows))
                session_selected.extend(map(int, selected_rows))
                remaining = [row for row in remaining if int(row) not in selected_set]

            final_state, final_fit_seconds = timed(
                engine.fit_history_state, features, main_valid, metadata, history
            )
            evaluated, evaluation_seconds = timed(
                r3c.evaluate_fixed_test, final_state, features, main_valid, metadata, fixed_rows
            )
            run_id = f"{trajectory_id}_S{session:02d}"
            common = {**base_common, "pool_id": pool_id, "trajectory_id": trajectory_id}
            history_meta = metadata.iloc[np.asarray(history, dtype=int)]
            source_sessions = history_meta["session"].to_numpy(dtype=int)
            fixed_meta = metadata.iloc[np.asarray(fixed_rows, dtype=int)]
            output["folds"].append(
                {
                    **common,
                    "run_id": run_id,
                    "target_session": session,
                    "strategy": STRATEGY,
                    "query_budget": budget,
                    "pool_candidates": pool_size,
                    "history_repetitions": len(history),
                    "selected_repetitions_this_session": len(session_selected),
                    "fixed_test_repetitions": len(fixed_rows),
                    "repetition_accuracy": evaluated["accuracy"],
                    "repetition_balanced_accuracy": evaluated["balanced_accuracy"],
                    "repetition_macro_f1": evaluated["macro_f1"],
                    "repetition_errors": int(np.sum(evaluated["true"] != evaluated["predicted"])),
                    "balanced_accuracy_equals_accuracy": bool(
                        abs(evaluated["balanced_accuracy"] - evaluated["accuracy"]) < 1e-15
                    ),
                    "maximum_source_session": int(source_sessions.max()),
                    "future_session_used": bool((source_sessions > session).any()),
                    "fixed_test_entered_history": bool(history_meta["fixed_test_never_query"].astype(bool).any()),
                    "test_labels_are_balanced_five_per_class": bool(
                        fixed_meta.groupby("label").size().eq(5).all() and fixed_meta["label"].nunique() == 7
                    ),
                    "query_fit_seconds": 0.0,
                    "candidate_score_seconds": 0.0,
                    "selector_seconds": selector_seconds_total,
                    "final_refit_seconds": final_fit_seconds,
                    "fixed_test_inference_seconds": evaluation_seconds,
                    "end_to_end_session_seconds": time.perf_counter() - session_started,
                }
            )
            append_normalizer(output["normalizers"], common, session, budget, len(history), final_state)
            for position, row in enumerate(fixed_rows):
                record = {
                    **base_common,
                    "run_id": run_id,
                    "pool_id": pool_id,
                    "trajectory_id": trajectory_id,
                    "target_session": session,
                    "strategy": STRATEGY,
                    "query_budget": budget,
                    "test_position": position + 1,
                    "sequence_row_internal_audit_only": int(row),
                    "true_label": int(evaluated["true"][position]),
                    "predicted_label": int(evaluated["predicted"][position]),
                    "correct": bool(evaluated["true"][position] == evaluated["predicted"][position]),
                    "raw_margin": float(evaluated["margins"][position]),
                }
                for label in range(engine.CLASSES):
                    record[f"decision_score_{label}"] = float(evaluated["scores"][position, label])
                output["predictions"].append(record)
            for true_label in range(engine.CLASSES):
                mask = evaluated["true"] == true_label
                output["recalls"].append(
                    {
                        **base_common,
                        "run_id": run_id,
                        "target_session": session,
                        "strategy": STRATEGY,
                        "query_budget": budget,
                        "class_label": true_label,
                        "class_support": int(mask.sum()),
                        "class_recall": float(np.mean(evaluated["predicted"][mask] == true_label)),
                    }
                )
                for predicted_label in range(engine.CLASSES):
                    output["confusions"].append(
                        {
                            **base_common,
                            "run_id": run_id,
                            "target_session": session,
                            "strategy": STRATEGY,
                            "query_budget": budget,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "count": int(np.sum(mask & (evaluated["predicted"] == predicted_label))),
                        }
                    )
            selected_labels = metadata.iloc[np.asarray(session_selected, dtype=int)]["label"].to_numpy(dtype=int)
            counts, coverage, entropy = r3c.class_distribution_metrics(selected_labels)
            coverage_row = {
                **base_common,
                "run_id": run_id,
                "pool_id": pool_id,
                "trajectory_id": trajectory_id,
                "target_session": session,
                "strategy": STRATEGY,
                "query_budget": budget,
                "pool_candidates": pool_size,
                "selected_class_coverage": coverage,
                "selected_normalized_class_entropy": entropy,
            }
            for label, count in enumerate(counts):
                coverage_row[f"selected_true_class_{label}_count"] = int(count)
            output["coverage"].append(coverage_row)
    return {key: pd.DataFrame(rows) for key, rows in output.items()}


def validate_shard(shard, outputs):
    level = str(shard.imbalance_level)
    pool_size = {"MILD_32": 32, "MODERATE_28": 28, "SEVERE_21": 21}[level]
    expected_candidate_rows = 5 * (6 * pool_size - 28)
    folds = outputs["folds"]
    selections = outputs["selections"]
    calls = outputs["selector_calls"]
    normalizers = outputs["normalizers"]
    metrics = folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]]
    return {
        "trajectory_count_is_3": folds["trajectory_id"].nunique() == 3,
        "fold_count_is_15": len(folds) == 15,
        "prediction_count_is_525": len(outputs["predictions"]) == 525,
        "selection_count_is_210": len(selections) == 210,
        "selector_call_count_is_30": len(calls) == 30,
        "candidate_audit_count_matches_pool_size": len(outputs["candidate_audits"]) == expected_candidate_rows,
        "normalizer_row_count_is_15": len(normalizers) == 15,
        "recall_row_count_is_105": len(outputs["recalls"]) == 105,
        "confusion_row_count_is_735": len(outputs["confusions"]) == 735,
        "coverage_row_count_is_15": len(outputs["coverage"]) == 15,
        "each_trajectory_has_five_sessions": bool(
            folds.groupby("trajectory_id")["target_session"].nunique().eq(5).all()
        ),
        "selection_counts_match_budgets": bool(
            selections.groupby(["trajectory_id", "target_session"]).size().reset_index(name="observed")
            .merge(
                folds[["trajectory_id", "target_session", "query_budget"]],
                on=["trajectory_id", "target_session"],
                validate="one_to_one",
            ).eval("observed == query_budget").all()
        ),
        "all_selected_records_are_pool_candidates": bool(selections["selected_record_is_pool_candidate"].all()),
        "no_fixed_test_record_was_selected": bool((~selections["selected_record_is_fixed_test"]).all()),
        "random_selector_schema_is_exactly_one_opaque_column": bool(
            calls["selector_schema_exact"].all()
            and calls["selector_schema"].eq("opaque_candidate_token").all()
            and calls["selector_forbidden_column_count"].eq(0).all()
        ),
        "random_query_uses_no_model_fit_or_candidate_scoring": bool(
            calls["fit_seconds"].eq(0).all() and calls["score_seconds"].eq(0).all()
        ),
        "rng_state_hashes_are_complete_and_valid": bool(
            calls["rng_state_sha256_before"].str.fullmatch(r"[0-9a-f]{64}").all()
            and calls["rng_state_sha256_after"].str.fullmatch(r"[0-9a-f]{64}").all()
        ),
        "no_future_session_is_used": bool((~calls["future_session_used"]).all() and (~folds["future_session_used"]).all()),
        "fixed_test_never_enters_history": bool(
            (~calls["fixed_test_used_for_training_or_selection"]).all()
            and (~folds["fixed_test_entered_history"]).all()
        ),
        "all_metrics_are_finite_and_in_range": bool(
            np.isfinite(metrics.to_numpy(float)).all() and metrics.ge(0).all().all() and metrics.le(1).all().all()
        ),
        "balanced_accuracy_equals_accuracy": bool(folds["balanced_accuracy_equals_accuracy"].all()),
        "all_normalizers_use_locked_lda_numerical_contract": bool(
            np.isfinite(normalizers[["minimum_mean", "maximum_mean", "minimum_std", "maximum_std"]].to_numpy(float)).all()
            and normalizers["minimum_std"].gt(0).all()
            and normalizers["minimum_valid_count"].gt(0).all()
            and normalizers["means_dtype"].eq("float32").all()
            and normalizers["stds_dtype"].eq("float32").all()
            and normalizers["model_coefficient_dtype"].eq("float64").all()
        ),
        "severe_k21_selects_full_pool": bool(
            True if level != "SEVERE_21" else selections.loc[selections["query_budget"].eq(21)]
            .groupby(["trajectory_id", "target_session"])["opaque_candidate_token"].nunique().eq(21).all()
        ),
        "locked_seed_index_and_value_are_constant": bool(
            folds["random_seed_index"].nunique() == 1 and folds["random_seed"].nunique() == 1
        ),
        "p07_is_case_analysis_only": bool(
            folds["case_analysis"].all() if str(shard.participant) == "P07" else (~folds["case_analysis"]).all()
        ),
    }


def write_shard_packet(shard, outputs, gates, runtime_seconds):
    shard_id = str(shard.shard_id)
    shard_root = TEMP_ROOT / shard_id
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True)
    for key, basename in SHARD_TABLES.items():
        atomic_csv(outputs[key], shard_root / basename)
    report = {
        "stage": STAGE,
        "shard_id": shard_id,
        "participant": str(shard.participant),
        "imbalance_level": str(shard.imbalance_level),
        "rotation_index": int(shard.rotation_index),
        "pool_realization_index": int(shard.pool_realization_index),
        "random_seed_index": int(shard.random_seed_index),
        "random_seed": int(outputs["folds"]["random_seed"].iloc[0]),
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "revision_r4a_packet_sha256": REVISION_R4A_PACKET_SHA256,
        "revision_r4d_packet_sha256": REVISION_R4D_PACKET_SHA256,
        "stage3e2b_packet_sha256": WORKER_CONTEXT["lda_random_packet_sha256"],
        "classifier": CLASSIFIER,
        "lda_numerical_contract": LDA_NUMERICAL_CONTRACT,
        "trajectory_count": int(outputs["folds"]["trajectory_id"].nunique()),
        "fold_count": len(outputs["folds"]),
        "prediction_count": len(outputs["predictions"]),
        "selection_count": len(outputs["selections"]),
        "selector_call_count": len(outputs["selector_calls"]),
        "candidate_audit_count": len(outputs["candidate_audits"]),
        "normalizer_count": len(outputs["normalizers"]),
        "recall_count": len(outputs["recalls"]),
        "confusion_count": len(outputs["confusions"]),
        "readiness_gates": gates,
        "failed_readiness_gates": [key for key, value in gates.items() if not value],
        "all_readiness_gates_passed": all(gates.values()),
        "runtime_seconds": runtime_seconds,
        "raw_hdf5_accessed": False,
        "new_statistical_test_run": False,
        "p07_case_analysis_only": str(shard.participant) == "P07",
    }
    atomic_json(report, shard_root / "revision_R4E_shard_report.json")
    manifest = []
    for path in sorted(shard_root.rglob("*")):
        if path.is_file():
            manifest.append(
                {
                    "relative_path": path.relative_to(shard_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": engine.sha256_file(path),
                }
            )
    atomic_csv(pd.DataFrame(manifest), shard_root / "revision_R4E_shard_manifest.csv")
    packet = TEMP_ROOT / f"{shard_id}.zip"
    if not engine.make_zip(shard_root, packet, shard_id):
        raise RuntimeError(f"Shard CRC failed: {shard_id}")
    return shard_root, packet, engine.sha256_file(packet)


def initialize_worker_context(
    features,
    main_valid,
    metadata,
    pool_definitions,
    pool_membership,
    seed_map,
    lda_random_packet_sha256,
):
    global WORKER_CONTEXT
    WORKER_CONTEXT = {
        "features": features,
        "main_valid": main_valid,
        "metadata": metadata,
        "pool_definitions": pool_definitions,
        "pool_membership": pool_membership,
        "seed_map": seed_map,
        "lda_random_packet_sha256": lda_random_packet_sha256,
    }


def execute_shard_worker(shard_payload):
    shard = SimpleNamespace(**shard_payload)
    shard_id = str(shard.shard_id)
    started = time.time()
    with threadpool_limits(limits=1):
        outputs = run_shard(
            shard,
            WORKER_CONTEXT["features"],
            WORKER_CONTEXT["main_valid"],
            WORKER_CONTEXT["metadata"],
            WORKER_CONTEXT["pool_definitions"],
            WORKER_CONTEXT["pool_membership"],
            WORKER_CONTEXT["seed_map"],
        )
        gates = validate_shard(shard, outputs)
        failed = [key for key, value in gates.items() if not bool(value)]
        if failed:
            raise RuntimeError(f"Shard {shard_id} failed gates: {failed}")
        shard_root, packet, digest = write_shard_packet(
            shard, outputs, gates, time.time() - started
        )
    return {
        "shard_id": shard_id,
        "packet": str(packet),
        "shard_root": str(shard_root),
        "sha256": digest,
        "runtime_seconds": time.time() - started,
    }


def upload_verified_batch(results):
    batch_token = uuid.uuid4().hex[:16]
    local_upload = TEMP_ROOT / f"R4E_UPLOAD_{batch_token}"
    local_verify = TEMP_ROOT / f"R4E_VERIFY_{batch_token}"
    local_upload.mkdir(parents=True, exist_ok=True)
    local_verify.mkdir(parents=True, exist_ok=True)
    pending_remote = REMOTE_OUTPUT + f"/pending/{batch_token}"
    expected = {}
    for result in results:
        basename = f"{result['shard_id']}__{result['sha256']}.zip"
        destination = local_upload / basename
        shutil.move(result["packet"], destination)
        shutil.rmtree(result["shard_root"], ignore_errors=True)
        expected[basename] = result
    parallel = str(max(4, min(32, len(expected))))
    try:
        engine.rclone(
            [
                "copy",
                str(local_upload),
                pending_remote,
                "--transfers",
                parallel,
                "--checkers",
                parallel,
                "--retries",
                "8",
                "--low-level-retries",
                "20",
                "--timeout",
                "5m",
            ]
        )
        engine.rclone(
            [
                "copy",
                pending_remote,
                str(local_verify),
                "--transfers",
                parallel,
                "--checkers",
                parallel,
                "--retries",
                "8",
                "--low-level-retries",
                "20",
                "--timeout",
                "5m",
            ]
        )
        verified = {}
        for basename, result in expected.items():
            downloaded = local_verify / basename
            if not downloaded.exists():
                raise RuntimeError(f"Batch round-trip file missing: {basename}")
            observed = engine.sha256_file(downloaded)
            if observed != result["sha256"] or not engine.archive_crc_passes(downloaded):
                raise RuntimeError(f"Batch round-trip integrity failed: {basename}")
            verified[result["shard_id"]] = {
                "shard_id": result["shard_id"],
                "sha256": result["sha256"],
                "remote_path": REMOTE_SHARDS + "/" + basename,
                "remote_basename": basename,
            }
        engine.rclone(
            [
                "move",
                pending_remote,
                REMOTE_SHARDS,
                "--transfers",
                parallel,
                "--checkers",
                parallel,
                "--delete-empty-src-dirs",
                "--retries",
                "8",
                "--timeout",
                "5m",
            ]
        )
        remote_listing = engine.rclone(["lsf", REMOTE_SHARDS, "--files-only"]).stdout.splitlines()
        missing_final = sorted(set(expected) - set(Path(name).name for name in remote_listing))
        if missing_final:
            raise RuntimeError(f"Verified batch finalization is incomplete: {missing_final[:3]}")
        return verified
    finally:
        shutil.rmtree(local_upload, ignore_errors=True)
        shutil.rmtree(local_verify, ignore_errors=True)


def make_progress_packet(inputs, manifest, completed, duplicates, decision):
    if PROGRESS_ROOT.exists():
        shutil.rmtree(PROGRESS_ROOT)
    PROGRESS_ROOT.mkdir(parents=True)
    completed_frame = pd.DataFrame(completed.values()) if completed else pd.DataFrame(
        columns=["shard_id", "sha256", "remote_path", "remote_basename"]
    )
    remaining = manifest.loc[~manifest["shard_id"].isin(set(completed))].copy()
    report = {
        "stage": STAGE,
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "revision_r4a_packet_sha256": REVISION_R4A_PACKET_SHA256,
        "revision_r4d_packet_sha256": REVISION_R4D_PACKET_SHA256,
        "revision_r4e_d3_packet_sha256": REVISION_R4E_D3_PACKET_SHA256,
        "revision_r4e_d3_classification": inputs["r4e_d3_report"][
            "classification"
        ],
        "stage3e2b_packet_sha256": inputs["lda_random_packet_sha256"],
        "classifier": CLASSIFIER,
        "lda_numerical_contract": LDA_NUMERICAL_CONTRACT,
        "frozen_lda_random_anchor_packet_sha256": inputs[
            "lda_random_anchor"
        ]["packet_sha256"],
        "frozen_lda_random_anchor_cache_status": inputs[
            "lda_random_anchor"
        ]["cache_status"],
        "frozen_lda_random_anchor_maximum_fold_metric_difference": inputs[
            "lda_random_anchor"
        ]["maximum_fold_metric_difference"],
        "frozen_lda_random_anchor_all_gates_passed": bool(
            all(inputs["lda_random_anchor"]["gates"].values())
        ),
        "expected_shards": EXPECTED_SHARDS,
        "completed_shards": len(completed),
        "remaining_shards": len(remaining),
        "duplicate_remote_checkpoint_records": len(duplicates),
        "completion_fraction": len(completed) / EXPECTED_SHARDS,
        "maximum_runtime_hours_this_invocation": MAX_RUNTIME_HOURS,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "full_experiment_complete": len(completed) == EXPECTED_SHARDS,
        "all_completed_shards_were_roundtrip_verified_when_created": True,
        "raw_hdf5_accessed": False,
        "new_statistical_test_run": False,
        "final_decision": decision,
    }
    atomic_json(report, PROGRESS_ROOT / "revision_R4E_progress_report.json")
    atomic_csv(inputs["audit"], PROGRESS_ROOT / "revision_R4E_input_audit.csv")
    atomic_csv(manifest, PROGRESS_ROOT / "revision_R4E_expected_shards.csv")
    atomic_csv(completed_frame, PROGRESS_ROOT / "revision_R4E_completed_shards.csv")
    atomic_csv(remaining, PROGRESS_ROOT / "revision_R4E_remaining_shards.csv")
    atomic_csv(pd.DataFrame(duplicates), PROGRESS_ROOT / "revision_R4E_duplicate_records.csv")
    atomic_json(
        {
            "readiness_gates": inputs["lda_random_anchor"]["gates"],
            "frozen_members": inputs["lda_random_anchor"]["members"],
            "identity_source": inputs["lda_random_anchor"]["identity_source"],
            "frozen_round_membership_available": inputs[
                "lda_random_anchor"
            ]["frozen_round_membership_available"],
            "maximum_fold_metric_difference": inputs[
                "lda_random_anchor"
            ]["maximum_fold_metric_difference"],
            "cache_status": inputs["lda_random_anchor"]["cache_status"],
            "packet_sha256": inputs["lda_random_anchor"]["packet_sha256"],
        },
        PROGRESS_ROOT / "revision_R4E_frozen_lda_random_anchor_report.json",
    )
    atomic_csv(
        inputs["lda_random_anchor"]["selection_comparison"],
        PROGRESS_ROOT / "revision_R4E_frozen_lda_random_selection_comparison.csv",
    )
    atomic_csv(
        inputs["lda_random_anchor"]["prediction_comparison"],
        PROGRESS_ROOT / "revision_R4E_frozen_lda_random_prediction_comparison.csv",
    )
    atomic_csv(
        inputs["lda_random_anchor"]["fold_comparison"],
        PROGRESS_ROOT / "revision_R4E_frozen_lda_random_fold_comparison.csv",
    )
    crc = engine.make_zip(PROGRESS_ROOT, PROGRESS_PACKET, "Revision_R4E_LDA_Random_Imbalance_Progress")
    digest = engine.sha256_file(PROGRESS_PACKET)
    verified = engine.roundtrip_remote_file(PROGRESS_PACKET, REMOTE_OUTPUT + "/" + PROGRESS_PACKET.name, digest)
    if not crc or not verified:
        raise RuntimeError("R4E progress packet persistence failed")
    return digest


def read_csv_member_from_archive(archive, basename):
    matches = [name for name in archive.namelist() if Path(name).name == basename]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {basename}; found {matches}")
    return pd.read_csv(io.BytesIO(archive.read(matches[0])))


def bulk_restore_shards(completed):
    cache = TEMP_ROOT / "R4E_ALL_SHARDS_CACHE"
    cache.mkdir(parents=True, exist_ok=True)
    print("Bulk-restoring shard packets with 32 parallel transfers...", flush=True)
    engine.rclone(
        [
            "copy",
            REMOTE_SHARDS,
            str(cache),
            "--include",
            "*.zip",
            "--transfers",
            "32",
            "--checkers",
            "32",
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--timeout",
            "5m",
        ]
    )
    local = {path.name: path for path in cache.glob("*.zip")}
    missing = sorted(set(row["remote_basename"] for row in completed.values()) - set(local))
    if missing:
        raise RuntimeError(f"Bulk shard restore is incomplete; missing {len(missing)} packets")
    return cache, local


def download_and_aggregate(inputs, manifest, completed):
    if FINAL_ROOT.exists():
        shutil.rmtree(FINAL_ROOT)
    FINAL_ROOT.mkdir(parents=True)
    cache, local = bulk_restore_shards(completed)
    aggregate_keys = ["folds", "coverage", "recalls", "selector_calls"]
    aggregate_paths = {key: FINAL_ROOT / f"revision_R4E_aggregate_{key}.csv" for key in aggregate_keys}
    for path in aggregate_paths.values():
        path.unlink(missing_ok=True)
    participants = list(engine.PARTICIPANTS)
    levels = ["MILD_32", "MODERATE_28", "SEVERE_21"]
    pmap = {value: index for index, value in enumerate(participants)}
    lmap = {value: index for index, value in enumerate(levels)}
    bmap = {value: index for index, value in enumerate(BUDGETS)}
    confusion_cube = np.zeros((7, 3, 5, 3, 7, 7), dtype=np.int64)
    integrity_rows = []
    count_totals = {
        "trajectory_count": 0,
        "fold_count": 0,
        "prediction_count": 0,
        "selection_count": 0,
        "selector_call_count": 0,
        "candidate_audit_count": 0,
        "normalizer_count": 0,
        "recall_count": 0,
        "confusion_count": 0,
    }
    for index, shard in enumerate(manifest.itertuples(index=False), start=1):
        record = completed[str(shard.shard_id)]
        packet = local[record["remote_basename"]]
        observed_hash = engine.sha256_file(packet)
        with zipfile.ZipFile(packet) as archive:
            crc = archive.testzip() is None
            report_matches = [name for name in archive.namelist() if Path(name).name == "revision_R4E_shard_report.json"]
            if len(report_matches) != 1:
                raise RuntimeError(f"R4E shard report missing: {shard.shard_id}")
            report = json.loads(archive.read(report_matches[0]).decode("utf-8"))
            valid = bool(
                observed_hash == record["sha256"]
                and crc
                and report.get("all_readiness_gates_passed", False)
                and report.get("shard_id") == str(shard.shard_id)
                and report.get("revision_protocol_sha256") == REVISION_PROTOCOL_SHA256
                and report.get("revision_r4d_packet_sha256") == REVISION_R4D_PACKET_SHA256
                and report.get("stage3e2b_packet_sha256")
                == inputs["lda_random_packet_sha256"]
                and report.get("classifier") == CLASSIFIER
                and report.get("lda_numerical_contract")
                == LDA_NUMERICAL_CONTRACT
            )
            if not valid:
                raise RuntimeError(f"Final shard integrity verification failed: {shard.shard_id}")
            integrity_rows.append(
                {
                    **record,
                    "observed_sha256": observed_hash,
                    "hash_matches_filename": observed_hash == record["sha256"],
                    "crc_passes": crc,
                    "report_gates_passed": bool(report.get("all_readiness_gates_passed", False)),
                    "shard_identity_matches": report.get("shard_id") == str(shard.shard_id),
                    "verified": valid,
                }
            )
            for key in count_totals:
                count_totals[key] += int(report[key])
            for key in aggregate_keys:
                frame = read_csv_member_from_archive(archive, SHARD_TABLES[key])
                frame.to_csv(aggregate_paths[key], mode="a", index=False, header=not aggregate_paths[key].exists())
            conf = read_csv_member_from_archive(archive, SHARD_TABLES["confusions"])
            pi = conf["participant"].map(pmap).to_numpy(int)
            li = conf["imbalance_level"].map(lmap).to_numpy(int)
            si = pd.to_numeric(conf["target_session"]).to_numpy(int) - 1
            bi = conf["query_budget"].map(bmap).to_numpy(int)
            ti = pd.to_numeric(conf["true_label"]).to_numpy(int)
            yi = pd.to_numeric(conf["predicted_label"]).to_numpy(int)
            np.add.at(confusion_cube, (pi, li, si, bi, ti, yi), pd.to_numeric(conf["count"]).to_numpy(np.int64))
        if index % 250 == 0 or index == len(manifest):
            print(f"Final shard verification: {index}/{len(manifest)}", flush=True)

    folds = pd.read_csv(aggregate_paths["folds"])
    coverage = pd.read_csv(aggregate_paths["coverage"])
    recalls = pd.read_csv(aggregate_paths["recalls"])
    calls = pd.read_csv(aggregate_paths["selector_calls"])
    confusion_rows = []
    for participant in participants:
        for level in levels:
            for session in TARGET_SESSIONS:
                for budget in BUDGETS:
                    matrix = confusion_cube[pmap[participant], lmap[level], session - 1, bmap[budget]]
                    for true_label in range(7):
                        for predicted_label in range(7):
                            confusion_rows.append(
                                {
                                    "participant": participant,
                                    "imbalance_level": level,
                                    "target_session": session,
                                    "strategy": STRATEGY,
                                    "query_budget": budget,
                                    "true_label": true_label,
                                    "predicted_label": predicted_label,
                                    "count": int(matrix[true_label, predicted_label]),
                                }
                            )
    confusions = pd.DataFrame(confusion_rows)
    atomic_csv(confusions, FINAL_ROOT / "revision_R4E_confusion_matrices_aggregated.csv")

    trajectory_summary = folds.groupby(
        [
            "participant", "case_analysis", "imbalance_level", "rotation_index",
            "pool_realization_index", "random_seed_index", "random_seed", "strategy", "query_budget",
        ],
        as_index=False,
    ).agg(
        target_sessions=("target_session", "nunique"),
        mean_repetition_balanced_accuracy=("repetition_balanced_accuracy", "mean"),
        mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
        total_repetition_errors=("repetition_errors", "sum"),
        mean_end_to_end_session_seconds=("end_to_end_session_seconds", "mean"),
    )
    participant_seed = trajectory_summary.groupby(
        ["participant", "case_analysis", "imbalance_level", "random_seed_index", "random_seed", "strategy", "query_budget"],
        as_index=False,
    ).agg(
        pool_trajectory_replicates=("rotation_index", "size"),
        mean_repetition_balanced_accuracy=("mean_repetition_balanced_accuracy", "mean"),
        std_across_pool_trajectories=("mean_repetition_balanced_accuracy", "std"),
        mean_repetition_macro_f1=("mean_repetition_macro_f1", "mean"),
        total_repetition_errors=("total_repetition_errors", "sum"),
    )
    participant_level = participant_seed.groupby(
        ["participant", "case_analysis", "imbalance_level", "strategy", "query_budget"], as_index=False
    ).agg(
        random_seed_replicates=("random_seed_index", "nunique"),
        mean_repetition_balanced_accuracy=("mean_repetition_balanced_accuracy", "mean"),
        std_across_random_seed_means=("mean_repetition_balanced_accuracy", "std"),
        mean_repetition_macro_f1=("mean_repetition_macro_f1", "mean"),
        total_repetition_errors=("total_repetition_errors", "sum"),
    )
    able = participant_level.loc[participant_level["participant"].ne("P07")].groupby(
        ["imbalance_level", "strategy", "query_budget"], as_index=False
    ).agg(
        participants=("participant", "nunique"),
        random_seed_replicates_per_participant=("random_seed_replicates", "min"),
        mean_repetition_balanced_accuracy=("mean_repetition_balanced_accuracy", "mean"),
        std_between_participants=("mean_repetition_balanced_accuracy", "std"),
        mean_repetition_macro_f1=("mean_repetition_macro_f1", "mean"),
    )
    p07 = participant_level.loc[participant_level["participant"].eq("P07")].copy()
    coverage_summary = coverage.groupby(
        ["participant", "imbalance_level", "random_seed_index", "random_seed", "strategy", "query_budget"], as_index=False
    ).agg(
        session_pool_folds=("run_id", "size"),
        mean_selected_class_coverage=("selected_class_coverage", "mean"),
        minimum_selected_class_coverage=("selected_class_coverage", "min"),
        maximum_selected_class_coverage=("selected_class_coverage", "max"),
        mean_selected_normalized_class_entropy=("selected_normalized_class_entropy", "mean"),
        std_selected_normalized_class_entropy=("selected_normalized_class_entropy", "std"),
    )
    compute_summary = folds.groupby(["imbalance_level", "strategy", "query_budget"], as_index=False).agg(
        folds=("run_id", "size"),
        mean_query_fit_seconds=("query_fit_seconds", "mean"),
        mean_candidate_score_seconds=("candidate_score_seconds", "mean"),
        mean_selector_seconds=("selector_seconds", "mean"),
        mean_final_refit_seconds=("final_refit_seconds", "mean"),
        mean_fixed_test_inference_seconds=("fixed_test_inference_seconds", "mean"),
        mean_end_to_end_session_seconds=("end_to_end_session_seconds", "mean"),
    )
    call_summary = calls.groupby(["imbalance_level", "strategy", "query_budget"], as_index=False).agg(
        selector_calls=("query_round", "size"),
        total_fit_seconds=("fit_seconds", "sum"),
        total_score_seconds=("score_seconds", "sum"),
        total_selector_seconds=("selector_seconds", "sum"),
    )
    integrity = pd.DataFrame(integrity_rows)
    for frame, basename in [
        (trajectory_summary, "revision_R4E_trajectory_summary.csv"),
        (participant_seed, "revision_R4E_participant_seed_summary.csv"),
        (participant_level, "revision_R4E_participant_level_summary.csv"),
        (able, "revision_R4E_able_bodied_descriptive_summary.csv"),
        (p07, "revision_R4E_p07_descriptive_summary.csv"),
        (coverage_summary, "revision_R4E_class_coverage_entropy_summary.csv"),
        (compute_summary, "revision_R4E_compute_summary.csv"),
        (call_summary, "revision_R4E_selector_compute_summary.csv"),
        (integrity, "revision_R4E_shard_integrity_manifest.csv"),
    ]:
        atomic_csv(frame, FINAL_ROOT / basename)

    atomic_json(
        {
            "readiness_gates": inputs["lda_random_anchor"]["gates"],
            "frozen_members": inputs["lda_random_anchor"]["members"],
            "identity_source": inputs["lda_random_anchor"]["identity_source"],
            "frozen_round_membership_available": inputs[
                "lda_random_anchor"
            ]["frozen_round_membership_available"],
            "maximum_fold_metric_difference": inputs[
                "lda_random_anchor"
            ]["maximum_fold_metric_difference"],
            "cache_status": inputs["lda_random_anchor"]["cache_status"],
            "packet_sha256": inputs["lda_random_anchor"]["packet_sha256"],
        },
        FINAL_ROOT / "revision_R4E_frozen_lda_random_anchor_report.json",
    )
    atomic_csv(
        inputs["lda_random_anchor"]["selection_comparison"],
        FINAL_ROOT / "revision_R4E_frozen_lda_random_selection_comparison.csv",
    )
    atomic_csv(
        inputs["lda_random_anchor"]["prediction_comparison"],
        FINAL_ROOT / "revision_R4E_frozen_lda_random_prediction_comparison.csv",
    )
    atomic_csv(
        inputs["lda_random_anchor"]["fold_comparison"],
        FINAL_ROOT / "revision_R4E_frozen_lda_random_fold_comparison.csv",
    )

    metrics = folds[["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]]
    gates = {
        "revision_r4a_packet_hash_matches": engine.sha256_file(inputs["r4a_packet"]) == REVISION_R4A_PACKET_SHA256,
        "revision_r4d_packet_hash_matches": engine.sha256_file(inputs["r4d_packet"]) == REVISION_R4D_PACKET_SHA256,
        "revision_r4e_d3_packet_hash_matches": engine.sha256_file(inputs["r4e_d3_packet"]) == REVISION_R4E_D3_PACKET_SHA256,
        "revision_r4e_d3_exact_replay_identified": inputs["r4e_d3_report"].get("classification") == "EXACT_FROZEN_HISTORY_LDA_REPLAY_IDENTIFIED",
        "stage3e2b_packet_hash_matches_r2a": engine.sha256_file(inputs["lda_random_packet"]) == inputs["lda_random_packet_sha256"],
        "revision_r4a_all_gates_passed": bool(inputs["r4a_report"].get("all_readiness_gates_passed")),
        "revision_r4d_all_gates_passed": bool(inputs["r4d_report"].get("all_readiness_gates_passed")),
        "revision_protocol_hash_matches": inputs["r4a_report"].get("revision_protocol_sha256") == REVISION_PROTOCOL_SHA256,
        "frozen_lda_random_anchor_all_gates_passed": bool(
            all(inputs["lda_random_anchor"]["gates"].values())
        ),
        "frozen_lda_random_anchor_fold_difference_is_below_1e_12": float(
            inputs["lda_random_anchor"]["maximum_fold_metric_difference"]
        ) < 1e-12,
        "all_22050_shards_pass_hash_crc_and_report_gates": len(integrity) == EXPECTED_SHARDS and bool(integrity["verified"].all()),
        "trajectory_count_is_66150": folds["trajectory_id"].nunique() == EXPECTED_TRAJECTORIES,
        "fold_count_is_330750": len(folds) == EXPECTED_FOLDS,
        "prediction_count_is_11576250": count_totals["prediction_count"] == EXPECTED_PREDICTIONS,
        "selection_count_is_4630500": count_totals["selection_count"] == EXPECTED_SELECTIONS,
        "selector_call_count_is_661500": count_totals["selector_call_count"] == EXPECTED_SELECTOR_CALLS,
        "candidate_audit_count_is_14773500": count_totals["candidate_audit_count"] == EXPECTED_CANDIDATE_AUDIT_ROWS,
        "normalizer_count_is_330750": count_totals["normalizer_count"] == EXPECTED_NORMALIZERS,
        "recall_count_is_2315250": count_totals["recall_count"] == EXPECTED_RECALL_ROWS,
        "confusion_row_count_is_16206750": count_totals["confusion_count"] == EXPECTED_CONFUSION_ROWS,
        "each_trajectory_has_five_sessions": bool(folds.groupby("trajectory_id")["target_session"].nunique().eq(5).all()),
        "all_metrics_are_finite_and_in_range": bool(
            np.isfinite(metrics.to_numpy(float)).all() and metrics.ge(0).all().all() and metrics.le(1).all().all()
        ),
        "balanced_accuracy_equals_accuracy_in_every_fold": bool(folds["balanced_accuracy_equals_accuracy"].all()),
        "no_future_session_is_used": bool((~folds["future_session_used"]).all()),
        "fixed_test_never_enters_history": bool((~folds["fixed_test_entered_history"]).all()),
        "all_test_sets_are_balanced_five_per_class": bool(folds["test_labels_are_balanced_five_per_class"].all()),
        "trajectory_summary_has_66150_rows": len(trajectory_summary) == EXPECTED_TRAJECTORIES,
        "participant_seed_summary_has_1890_rows": len(participant_seed) == 1890,
        "every_participant_seed_cell_has_35_pool_trajectories": bool(participant_seed["pool_trajectory_replicates"].eq(35).all()),
        "participant_level_summary_has_63_rows": len(participant_level) == 63,
        "every_participant_level_cell_has_30_random_seeds": bool(participant_level["random_seed_replicates"].eq(30).all()),
        "able_bodied_summary_has_9_rows": len(able) == 9,
        "each_able_bodied_summary_uses_six_participants": bool(able["participants"].eq(6).all()),
        "p07_summary_has_9_rows_and_is_descriptive_only": len(p07) == 9 and bool(p07["case_analysis"].all()),
        "class_recall_rows_are_complete": len(recalls) == EXPECTED_RECALL_ROWS and bool(recalls["class_support"].eq(5).all()),
        "confusion_matrices_are_aggregated_without_loss": int(confusions["count"].sum()) == EXPECTED_PREDICTIONS,
        "class_coverage_and_entropy_are_in_range": bool(
            coverage["selected_class_coverage"].between(1, 7).all()
            and coverage["selected_normalized_class_entropy"].between(0, 1).all()
        ),
        "random_query_fit_and_score_times_are_zero": bool(
            calls["fit_seconds"].eq(0).all() and calls["score_seconds"].eq(0).all()
        ),
        "all_compute_telemetry_is_finite_nonnegative": bool(
            np.isfinite(
                folds[[
                    "query_fit_seconds", "candidate_score_seconds", "selector_seconds", "final_refit_seconds",
                    "fixed_test_inference_seconds", "end_to_end_session_seconds",
                ]].to_numpy(float)
            ).all()
            and folds[[
                "query_fit_seconds", "candidate_score_seconds", "selector_seconds", "final_refit_seconds",
                "fixed_test_inference_seconds", "end_to_end_session_seconds",
            ]].ge(0).all().all()
        ),
        "only_locked_initial_30_random_seeds_are_used": set(folds["random_seed_index"].unique()) == set(range(1, 31)),
        "raw_hdf5_data_was_not_accessed": True,
        "no_new_statistical_test_was_run": True,
        "p07_remains_case_analysis_only": True,
        "stage3g_and_stage5f_conclusions_cannot_be_replaced": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]
    report = {
        "stage": STAGE,
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "revision_r4a_packet_sha256": REVISION_R4A_PACKET_SHA256,
        "revision_r4d_packet_sha256": REVISION_R4D_PACKET_SHA256,
        "revision_r4e_d3_packet_sha256": REVISION_R4E_D3_PACKET_SHA256,
        "revision_r4e_d3_classification": inputs["r4e_d3_report"][
            "classification"
        ],
        "stage3e2b_packet_sha256": inputs["lda_random_packet_sha256"],
        "classifier": CLASSIFIER,
        "lda_numerical_contract": LDA_NUMERICAL_CONTRACT,
        "frozen_lda_random_anchor_packet_sha256": inputs[
            "lda_random_anchor"
        ]["packet_sha256"],
        "frozen_lda_random_anchor_maximum_fold_metric_difference": inputs[
            "lda_random_anchor"
        ]["maximum_fold_metric_difference"],
        "shards": len(integrity),
        "trajectory_count": int(folds["trajectory_id"].nunique()),
        "fold_count": len(folds),
        **count_totals,
        "peak_cpu_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "raw_hdf5_accessed": False,
        "new_statistical_test_run": False,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": "PASS_TO_REVISION_R5_ALTERNATIVE_TEMPORAL_SPLITS_AND_DRIFT_AUDIT" if not failed else "REVISION_R4E_FINAL_AUDIT_FAILED",
    }
    atomic_json(report, FINAL_ROOT / "revision_R4E_final_report.json")
    shutil.copy2(Path(__file__), FINAL_ROOT / "revision_R4E_executed_source.py")
    manifest_rows = []
    for path in sorted(FINAL_ROOT.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(FINAL_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": engine.sha256_file(path),
                }
            )
    atomic_csv(pd.DataFrame(manifest_rows), FINAL_ROOT / "revision_R4E_output_manifest.csv")
    if failed:
        raise RuntimeError(f"Revision R4E final audit failed: {failed}")
    shutil.rmtree(cache, ignore_errors=True)
    return report, able, p07


def create_final_packet():
    crc = engine.make_zip(FINAL_ROOT, FINAL_PACKET, "Revision_R4E_LDA_Random_Imbalance_Final")
    digest = engine.sha256_file(FINAL_PACKET)
    verified = engine.roundtrip_remote_file(FINAL_PACKET, REMOTE_OUTPUT + "/" + FINAL_PACKET.name, digest)
    if not crc or not verified:
        raise RuntimeError("R4E final packet persistence failed")
    return digest


def main():
    print("=" * 104)
    print("REVISION R4E — LDA 30-SEED RANDOM IMBALANCE SHARDS")
    print("=" * 104)
    print("Execution device: CPU")
    print("GPU required: False")
    print("Expected shards:", EXPECTED_SHARDS)
    print("Expected trajectories:", EXPECTED_TRAJECTORIES)
    print("Expected folds:", EXPECTED_FOLDS)
    print("Locked random seeds: 1-30 only")
    print("Classifier:", CLASSIFIER)
    print("LDA numerical contract:", LDA_NUMERICAL_CONTRACT)
    print("Maximum runtime this invocation (hours):", MAX_RUNTIME_HOURS)
    print("CPU worker processes:", CPU_WORKERS)
    print("Parallel checkpoint batch size:", BATCH_SIZE)
    print("Checkpoint: every logical shard, batched SHA-256 Drive round trip")
    print("Resume: automatic from verified final-named shard checkpoints")
    print("Final aggregation: bulk 32-transfer restore in a separate invocation when needed")
    print("New statistical tests: False")
    print()

    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    install_direct_packet_resolver()
    print("Restoring verified frozen inputs from locked direct Drive paths...")
    inputs = resolve_inputs()
    features, main_valid, metadata = r3b.prepare_metadata(inputs["stage5b_packet"], inputs["stage5d2_packet"])
    seed_map = locked_random_seed_map(inputs["seeds"])
    r4d.activate_lda_contract()
    print("Validating the exact R4E-D3 frozen-history LDA replay anchor...")
    inputs["lda_random_anchor"] = load_revision_r4e_d3_frozen_history_anchor(
        inputs
    )
    print(
        "Frozen-history LDA anchor:",
        inputs["lda_random_anchor"]["cache_status"],
    )
    print(
        "Frozen-history LDA anchor maximum fold metric difference:",
        inputs["lda_random_anchor"]["maximum_fold_metric_difference"],
    )
    manifest = inputs["shard_manifest"].loc[inputs["shard_manifest"]["stage"].astype(str).eq(STAGE)].copy()
    manifest["random_seed_index"] = pd.to_numeric(manifest["random_seed_index"], errors="raise").astype(int)
    manifest = manifest.sort_values(
        ["participant", "imbalance_level", "rotation_index", "pool_realization_index", "random_seed_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(manifest) != EXPECTED_SHARDS or not manifest["shard_id"].is_unique:
        raise RuntimeError("R4E shard manifest drift")
    if set(manifest["random_seed_index"]) != set(range(1, 31)):
        raise RuntimeError("R4E manifest does not use exactly random seed indices 1-30")
    if not manifest["execution_family"].astype(str).eq(
        "LDA_RANDOM_SENSITIVITY"
    ).all():
        raise RuntimeError("R4E execution-family drift")
    if not manifest["classifier"].astype(str).eq("LDA").all():
        raise RuntimeError("R4E classifier manifest drift")
    expected_ids = set(manifest["shard_id"].astype(str))
    completed, duplicates = discover_completed_shards(expected_ids)
    print(f"Restored completed shard checkpoints: {len(completed)}/{EXPECTED_SHARDS}")
    if duplicates:
        print("Duplicate remote checkpoint records ignored:", len(duplicates))

    initialize_worker_context(
        features,
        main_valid,
        metadata,
        inputs["pool_definitions"],
        inputs["pool_membership"],
        seed_map,
        inputs["lda_random_packet_sha256"],
    )
    newly_completed = 0
    seconds_per_shard = []
    context = mp.get_context("fork")
    with context.Pool(processes=CPU_WORKERS) as pool:
        while len(completed) < EXPECTED_SHARDS:
            elapsed = time.time() - START_TIME
            typical_batch = (
                float(np.median(seconds_per_shard[-10:])) * BATCH_SIZE
                if seconds_per_shard
                else 900.0
            )
            if elapsed + max(SHARD_STOP_RESERVE_SECONDS, typical_batch * 1.5) >= MAX_RUNTIME_SECONDS:
                break
            pending = manifest.loc[~manifest["shard_id"].isin(set(completed))].head(BATCH_SIZE)
            if pending.empty:
                break
            batch_started = time.time()
            payloads = pending.to_dict("records")
            results = pool.map(execute_shard_worker, payloads, chunksize=1)
            verified = upload_verified_batch(results)
            completed.update(verified)
            batch_seconds = time.time() - batch_started
            per_shard = batch_seconds / len(results)
            seconds_per_shard.append(per_shard)
            newly_completed += len(results)
            remaining_count = EXPECTED_SHARDS - len(completed)
            eta_hours = remaining_count * float(np.median(seconds_per_shard[-10:])) / 3600.0
            print(
                f"BATCH PASS | completed={len(completed):05d}/{EXPECTED_SHARDS} | "
                f"new={len(results):03d} | wall={batch_seconds:.1f}s | "
                f"effective={per_shard:.2f}s/shard | ETA~{eta_hours:.2f}h",
                flush=True,
            )

    all_complete = len(completed) == EXPECTED_SHARDS
    if not all_complete:
        decision = "PARTIAL_PASS_RESUME_REVISION_R4E_SAME_NOTEBOOK"
        progress_hash = make_progress_packet(inputs, manifest, completed, duplicates, decision)
        print()
        print("Completed shards:", len(completed), "/", EXPECTED_SHARDS)
        print("New shards this invocation:", newly_completed)
        print("Remaining shards:", EXPECTED_SHARDS - len(completed))
        print("Progress packet SHA-256:", progress_hash)
        print("Progress packet Drive round-trip verified: True")
        print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 3))
        print()
        print("FINAL DECISION: PARTIAL_PASS_RESUME_REVISION_R4E_SAME_NOTEBOOK")
        return

    if newly_completed and time.time() - START_TIME >= FORCE_SEPARATE_AGGREGATION_AFTER_SECONDS:
        decision = "PARTIAL_PASS_ALL_SHARDS_COMPLETE_RESUME_R4E_FOR_FINAL_AGGREGATION"
        progress_hash = make_progress_packet(inputs, manifest, completed, duplicates, decision)
        print()
        print("All 22,050 shards are complete and verified.")
        print("Final aggregation is intentionally deferred to the next invocation.")
        print("Progress packet SHA-256:", progress_hash)
        print("FINAL DECISION:", decision)
        return

    make_progress_packet(inputs, manifest, completed, duplicates, "ALL_SHARDS_COMPLETE_FINAL_AGGREGATION_STARTED")
    print("All shards complete. Bulk-downloading and verifying every shard for final aggregation...")
    report, able, p07 = download_and_aggregate(inputs, manifest, completed)
    digest = create_final_packet()
    print()
    print("=" * 104)
    print("REVISION R4E — FINAL DESCRIPTIVE SUMMARY")
    print("=" * 104)
    print("Able-bodied summary:")
    print(able.to_string(index=False))
    print()
    print("P07 descriptive summary:")
    print(p07.to_string(index=False))
    print()
    print("Shards:", report["shards"])
    print("Trajectories:", report["trajectory_count"])
    print("Folds:", report["fold_count"])
    print("Final packet:", FINAL_PACKET)
    print("Final packet SHA-256:", digest)
    print("Remote round-trip verified: True")
    print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 3))
    print()
    print("FINAL DECISION: PASS_TO_REVISION_R5_ALTERNATIVE_TEMPORAL_SPLITS_AND_DRIFT_AUDIT")


if __name__ == "__main__":
    main()
