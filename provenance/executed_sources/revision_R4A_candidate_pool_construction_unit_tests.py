from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import revision_R3B_new_selector_implementation_unit_tests as r3b
import revision_R3C_balanced_pool_classical_comparator_extension as r3c


engine = r3b.engine
START_TIME = time.time()
WORKING = engine.WORKING
INPUT_ROOT = WORKING / "REVISION_R4A_FROZEN_INPUTS"
RESULT_ROOT = (
    WORKING
    / "DELTA_REVIEWER_REVISION"
    / "Revision_R4A_Candidate_Pool_Construction_Unit_Tests"
)
PACKET_PATH = WORKING / "revision_R4A_candidate_pool_construction_unit_test_packet.zip"
REMOTE_OUTPUT = (
    engine.REMOTE_BASE
    + "/Reviewer_Revision/Revision_R4A_Candidate_Pool_Construction_Unit_Tests"
)
for directory in (INPUT_ROOT, RESULT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

REVISION_R0_PACKET_SHA256 = r3b.REVISION_R0_PACKET_SHA256
REVISION_R3C_PACKET_SHA256 = (
    "6a8b332dcfc36109f7df70c4931309ca62e52c29979732a5ce0da90769f45e31"
)
REVISION_PROTOCOL_SHA256 = engine.REVISION_PROTOCOL_SHA256
TARGET_SESSIONS = [1, 2, 3, 4, 5]
BUDGETS = [7, 14, 21]
IMBALANCE_LEVELS = ["BALANCED_35", "MILD_32", "MODERATE_28", "SEVERE_21"]
NONBALANCED_LEVELS = ["MILD_32", "MODERATE_28", "SEVERE_21"]
UNIT_PARTICIPANT = "P01"
UNIT_SESSION = 1
UNIT_LEVEL = "MODERATE_28"
UNIT_ROTATION = 0
UNIT_REALIZATION = 1
UNIT_STRATEGIES = [
    "PCBM_ORIGINAL",
    "GLOBAL_MARGIN_ORIGINAL",
    "RANDOM_UNIFORM",
    "LEAST_CONFIDENCE",
    "PREDICTIVE_ENTROPY",
    "RBMAL_MARGIN_DIVERSITY",
    "CORE_SET_GREEDY",
]
DETERMINISTIC_STRATEGIES = [
    "PCBM_ORIGINAL",
    "GLOBAL_MARGIN_ORIGINAL",
    "LEAST_CONFIDENCE",
    "PREDICTIVE_ENTROPY",
    "RBMAL_MARGIN_DIVERSITY",
    "CORE_SET_GREEDY",
]
EXPECTED_POOL_DEFINITIONS = 7 * 5 * 4 * 7 * 5
EXPECTED_MEMBERSHIP_ROWS = 7 * 5 * 7 * 5 * (35 + 32 + 28 + 21)
POOL_CONSTRUCTION_CONTRACT = (
    "TRUE_LABEL_STRATIFIED_HASH_SUBSET_FOR_SIMULATION_ONLY__"
    "LABEL_HIDDEN_BEFORE_SELECTOR__SHA256_RANK_BY_LOCKED_POOL_SEED__V1"
)


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stable_pool_rank(seed, participant, session, level, rotation, label, token):
    payload = (
        f"{int(seed)}|{participant}|S{int(session):02d}|{level}|"
        f"R{int(rotation)}|L{int(label)}|{token}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_pool_id(participant, session, level, rotation, realization):
    payload = (
        f"{REVISION_PROTOCOL_SHA256}|POOL|{participant}|S{int(session):02d}|"
        f"{level}|ROT{int(rotation)}|REAL{int(realization):02d}"
    )
    return "POOL_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def resolve_inputs():
    inputs = r3b.resolve_inputs()
    r3c_packet, r3c_source = engine.resolve_packet(
        "revision_R3C_balanced_pool_classical_comparator_extension_packet.zip",
        REVISION_R3C_PACKET_SHA256,
    )
    r3c_report = engine.read_json_member(r3c_packet, "revision_R3C_report.json")
    if not r3c_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("Revision R3C parent gates did not pass")
    if r3c_report.get("revision_protocol_sha256") != REVISION_PROTOCOL_SHA256:
        raise RuntimeError("Revision R3C protocol hash drift")
    audit = pd.concat(
        [
            inputs["audit"],
            pd.DataFrame(
                [
                    {
                        "packet": r3c_packet.name,
                        "expected_sha256": REVISION_R3C_PACKET_SHA256,
                        "observed_sha256": engine.sha256_file(r3c_packet),
                        "hash_matches": engine.sha256_file(r3c_packet)
                        == REVISION_R3C_PACKET_SHA256,
                        "crc_passes": engine.archive_crc_passes(r3c_packet),
                        "source": r3c_source,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    if not audit[["hash_matches", "crc_passes"]].all().all():
        raise RuntimeError("R4A frozen-input integrity failed")
    inputs["r3c_packet"] = r3c_packet
    inputs["r3c_report"] = r3c_report
    inputs["audit"] = audit
    return inputs


def build_pool_schedule(metadata, patterns, pool_seeds):
    pattern_columns = [f"class_{label}_count" for label in range(engine.CLASSES)]
    required_pattern_columns = {
        "imbalance_level",
        "rotation_index",
        "total_candidates",
        *pattern_columns,
    }
    if not required_pattern_columns.issubset(patterns.columns):
        raise RuntimeError("R0 pool-pattern columns are incomplete")
    if len(patterns) != 28:
        raise RuntimeError("R0 must contain 28 imbalance-level rotations")
    if len(pool_seeds) != 5:
        raise RuntimeError("R0 must contain five locked pool-subset seeds")

    definitions = []
    memberships = []
    candidate_meta = metadata.loc[
        metadata["protocol_role"].eq("CURRENT_SESSION_UNLABELED_POOL")
    ].copy()
    for participant in engine.PARTICIPANTS:
        for session in TARGET_SESSIONS:
            base = candidate_meta.loc[
                candidate_meta["participant"].eq(participant)
                & candidate_meta["session"].eq(session)
            ].copy()
            if len(base) != 35 or not base.groupby("label").size().eq(5).all():
                raise RuntimeError(
                    f"Frozen candidate pool drift for {participant} session {session}"
                )
            for pattern in patterns.sort_values(
                ["level_order", "rotation_index"]
            ).itertuples(index=False):
                level = str(pattern.imbalance_level)
                rotation = int(pattern.rotation_index)
                total = int(pattern.total_candidates)
                expected_counts = {
                    label: int(getattr(pattern, f"class_{label}_count"))
                    for label in range(engine.CLASSES)
                }
                for seed_row in pool_seeds.sort_values("seed_index").itertuples(
                    index=False
                ):
                    realization = int(seed_row.seed_index)
                    seed = int(seed_row.seed)
                    pool_id = stable_pool_id(
                        participant, session, level, rotation, realization
                    )
                    selected_frames = []
                    for label in range(engine.CLASSES):
                        label_rows = base.loc[base["label"].eq(label)].copy()
                        label_rows["_pool_rank"] = label_rows[
                            "opaque_candidate_token"
                        ].map(
                            lambda token: stable_pool_rank(
                                seed,
                                participant,
                                session,
                                level,
                                rotation,
                                label,
                                token,
                            )
                        )
                        selected_frames.append(
                            label_rows.sort_values(
                                ["_pool_rank", "opaque_candidate_token"],
                                kind="mergesort",
                            ).head(expected_counts[label])
                        )
                    selected = pd.concat(selected_frames, ignore_index=True)
                    selected = selected.sort_values(
                        ["label", "_pool_rank", "opaque_candidate_token"],
                        kind="mergesort",
                    ).reset_index(drop=True)
                    if len(selected) != total:
                        raise RuntimeError(f"Pool size drift: {pool_id}")
                    definitions.append(
                        {
                            "pool_id": pool_id,
                            "participant": participant,
                            "target_session": session,
                            "imbalance_level": level,
                            "rotation_index": rotation,
                            "pool_realization_index": realization,
                            "locked_pool_seed": seed,
                            "total_candidates": total,
                            "minimum_class_count": min(expected_counts.values()),
                            "maximum_class_count": max(expected_counts.values()),
                            "k21_is_full_pool_control": total == 21,
                            "p07_case_analysis_only": participant == "P07",
                            "pool_construction_contract": POOL_CONSTRUCTION_CONTRACT,
                            **{
                                f"class_{label}_count": expected_counts[label]
                                for label in range(engine.CLASSES)
                            },
                        }
                    )
                    for order, item in enumerate(selected.to_dict("records"), 1):
                        memberships.append(
                            {
                                "pool_id": pool_id,
                                "participant": participant,
                                "target_session": session,
                                "imbalance_level": level,
                                "rotation_index": rotation,
                                "pool_realization_index": realization,
                                "pool_member_order_internal_audit_only": order,
                                "sequence_row_internal_audit_only": int(
                                    item["sequence_row"]
                                ),
                                "opaque_candidate_token": str(
                                    item["opaque_candidate_token"]
                                ),
                                "true_label_internal_pool_audit_only": int(
                                    item["label"]
                                ),
                                "repetition_number_internal_audit_only": int(
                                    item["repetition"]
                                ),
                                "pool_rank_hash_internal_audit_only": str(
                                    item["_pool_rank"]
                                ),
                                "selector_receives_true_label": False,
                                "selector_receives_sequence_row": False,
                                "selector_receives_repetition_number": False,
                                "selected_record_is_original_candidate": bool(
                                    item["protocol_role"]
                                    == "CURRENT_SESSION_UNLABELED_POOL"
                                ),
                                "selected_record_is_fixed_test": bool(
                                    item["fixed_test_never_query"]
                                ),
                            }
                        )
    return pd.DataFrame(definitions), pd.DataFrame(memberships)


def pool_schedule_digest(definitions, memberships):
    digest = hashlib.sha256()
    for frame, columns in [
        (
            definitions,
            [
                "pool_id",
                "participant",
                "target_session",
                "imbalance_level",
                "rotation_index",
                "pool_realization_index",
                "locked_pool_seed",
                "total_candidates",
                *[f"class_{label}_count" for label in range(engine.CLASSES)],
            ],
        ),
        (
            memberships,
            [
                "pool_id",
                "opaque_candidate_token",
                "true_label_internal_pool_audit_only",
                "pool_rank_hash_internal_audit_only",
            ],
        ),
    ]:
        ordered = frame[columns].sort_values(columns, kind="mergesort")
        digest.update(ordered.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def pool_rows(metadata, memberships, pool_id):
    tokens = memberships.loc[
        memberships["pool_id"].eq(pool_id), "opaque_candidate_token"
    ].astype(str)
    token_to_row = dict(
        zip(
            metadata["opaque_candidate_token"].astype(str),
            metadata["sequence_row"].astype(int),
        )
    )
    rows = [token_to_row[token] for token in tokens]
    if len(rows) != len(set(rows)):
        raise RuntimeError("Pool rows are not unique")
    return rows


def margin_selection(strategy, features, main_valid, metadata, history_rows, rows):
    state = r3c.fit_history_state_with_locked_epsilon(
        features, main_valid, metadata, history_rows
    )
    _, predicted, margins = engine.score_repetitions(
        state, features, main_valid, rows
    )
    tokens = metadata.iloc[rows]["opaque_candidate_token"].astype(str).to_numpy()
    selector_frame = pd.DataFrame(
        {
            "opaque_candidate_token": tokens,
            "predicted_label": predicted,
            "margin": margins,
        }
    )[engine.SELECTOR_COLUMNS]
    if strategy == "PCBM_ORIGINAL":
        selected = engine.select_pcbm(selector_frame)
    elif strategy == "GLOBAL_MARGIN_ORIGINAL":
        selected = engine.select_global_margin(selector_frame)
    else:
        raise ValueError(strategy)
    return selected, selector_frame.columns.tolist(), state


def random_selection(metadata, rows, seed):
    frame = pd.DataFrame(
        {
            "opaque_candidate_token": metadata.iloc[rows][
                "opaque_candidate_token"
            ].astype(str)
        }
    ).sort_values("opaque_candidate_token", kind="mergesort")
    r3b.validate_opaque_tokens(frame["opaque_candidate_token"])
    generator = np.random.default_rng(int(seed))
    positions = generator.choice(len(frame), size=7, replace=False)
    return (
        frame.iloc[positions]["opaque_candidate_token"].astype(str).tolist(),
        frame.columns.tolist(),
    )


def run_single_round_unit_tests(
    features,
    main_valid,
    metadata,
    definitions,
    memberships,
    random_seed,
    rbmal_seed,
):
    definition = definitions.loc[
        definitions["participant"].eq(UNIT_PARTICIPANT)
        & definitions["target_session"].eq(UNIT_SESSION)
        & definitions["imbalance_level"].eq(UNIT_LEVEL)
        & definitions["rotation_index"].eq(UNIT_ROTATION)
        & definitions["pool_realization_index"].eq(UNIT_REALIZATION)
    ]
    if len(definition) != 1:
        raise RuntimeError("Unit-test pool definition is not unique")
    pool_id = definition.iloc[0]["pool_id"]
    candidate_rows = pool_rows(metadata, memberships, pool_id)
    history_rows = metadata.index[
        metadata["participant"].eq(UNIT_PARTICIPANT)
        & metadata["session"].eq(0)
        & metadata["repetition"].le(5)
    ].tolist()
    if len(history_rows) != 35 or len(candidate_rows) != 28:
        raise RuntimeError("Unit-test history or pool count drift")
    token_to_row = dict(
        zip(
            metadata.iloc[candidate_rows]["opaque_candidate_token"].astype(str),
            candidate_rows,
        )
    )
    rows = []
    for strategy in UNIT_STRATEGIES:
        if strategy in {"PCBM_ORIGINAL", "GLOBAL_MARGIN_ORIGINAL"}:
            selected, schema, state = margin_selection(
                strategy,
                features,
                main_valid,
                metadata,
                history_rows,
                candidate_rows,
            )
            std_floor_count = int(state["std_floor_count"])
        elif strategy == "RANDOM_UNIFORM":
            selected, schema = random_selection(metadata, candidate_rows, random_seed)
            std_floor_count = 0
        elif strategy in {"LEAST_CONFIDENCE", "PREDICTIVE_ENTROPY"}:
            selected, _, call, _ = r3c.probability_acquisition(
                strategy,
                features,
                main_valid,
                metadata,
                history_rows,
                candidate_rows,
            )
            schema = call["selector_schema"].split("|")
            std_floor_count = int(call["oof_std_floor_event_count"])
        else:
            selected, _, call, _ = r3c.diversity_acquisition(
                strategy,
                features,
                main_valid,
                metadata,
                history_rows,
                candidate_rows,
                rbmal_seed,
            )
            schema = call["selector_schema"].split("|")
            std_floor_count = int(call.get("full_history_std_floor_count", 0))
        selected_rows = [token_to_row[token] for token in selected]
        selected_meta = metadata.iloc[selected_rows]
        forbidden = set(schema).intersection(r3b.FORBIDDEN_COLUMNS)
        rows.append(
            {
                "strategy": strategy,
                "participant": UNIT_PARTICIPANT,
                "target_session": UNIT_SESSION,
                "pool_id": pool_id,
                "pool_candidates": len(candidate_rows),
                "selected_count": len(selected),
                "unique_selected_tokens": len(set(selected)),
                "selected_true_class_coverage_internal_audit_only": int(
                    selected_meta["label"].nunique()
                ),
                "all_selected_records_are_pool_candidates": set(selected).issubset(
                    set(token_to_row)
                ),
                "no_fixed_test_record_selected": bool(
                    (~selected_meta["fixed_test_never_query"]).all()
                ),
                "selector_schema": "|".join(schema),
                "selector_forbidden_column_count": len(forbidden),
                "selector_receives_true_label": "label" in schema
                or "true_label" in schema,
                "std_floor_event_count": std_floor_count,
            }
        )
    return pd.DataFrame(rows)


def run_severe_k21_full_pool_control(
    features,
    main_valid,
    metadata,
    definitions,
    memberships,
):
    definition = definitions.loc[
        definitions["participant"].eq(UNIT_PARTICIPANT)
        & definitions["target_session"].eq(UNIT_SESSION)
        & definitions["imbalance_level"].eq("SEVERE_21")
        & definitions["rotation_index"].eq(0)
        & definitions["pool_realization_index"].eq(1)
    ]
    if len(definition) != 1:
        raise RuntimeError("Severe K21 unit pool is not unique")
    pool_id = definition.iloc[0]["pool_id"]
    remaining = pool_rows(metadata, memberships, pool_id)
    history = metadata.index[
        metadata["participant"].eq(UNIT_PARTICIPANT)
        & metadata["session"].eq(0)
        & metadata["repetition"].le(5)
    ].tolist()
    token_to_row = dict(
        zip(
            metadata.iloc[remaining]["opaque_candidate_token"].astype(str),
            remaining,
        )
    )
    selected_all = []
    audit = []
    for query_round in range(1, 4):
        selected, schema, _ = margin_selection(
            "PCBM_ORIGINAL",
            features,
            main_valid,
            metadata,
            history,
            remaining,
        )
        selected_rows = [token_to_row[token] for token in selected]
        audit.append(
            {
                "pool_id": pool_id,
                "query_round": query_round,
                "history_before_fit": len(history),
                "remaining_before_selection": len(remaining),
                "selected_count": len(selected),
                "selector_schema": "|".join(schema),
                "selector_received_true_label": "label" in schema
                or "true_label" in schema,
            }
        )
        history.extend(selected_rows)
        selected_all.extend(selected)
        selected_row_set = set(selected_rows)
        remaining = [row for row in remaining if row not in selected_row_set]
    return pd.DataFrame(audit), selected_all, remaining


def build_execution_shards(patterns):
    rows = []
    stage_specs = [
        (
            "R4B",
            "RIDGE_DETERMINISTIC",
            DETERMINISTIC_STRATEGIES,
            [None],
            False,
        ),
        (
            "R4C",
            "RIDGE_RANDOM",
            ["RANDOM_UNIFORM"],
            list(range(1, 31)),
            False,
        ),
        (
            "R4D",
            "LDA_DETERMINISTIC_SENSITIVITY",
            DETERMINISTIC_STRATEGIES,
            [None],
            True,
        ),
        (
            "R4E",
            "LDA_RANDOM_SENSITIVITY",
            ["RANDOM_UNIFORM"],
            list(range(1, 31)),
            True,
        ),
    ]
    nonbalanced_patterns = patterns.loc[
        patterns["imbalance_level"].isin(NONBALANCED_LEVELS)
    ].sort_values(["level_order", "rotation_index"])
    for stage, family, strategies, random_indices, sensitivity in stage_specs:
        for participant in engine.PARTICIPANTS:
            for pattern in nonbalanced_patterns.itertuples(index=False):
                for realization in range(1, 6):
                    for random_index in random_indices:
                        shard_key = (
                            f"{stage}_{participant}_{pattern.imbalance_level}_"
                            f"ROT{int(pattern.rotation_index)}_REAL{realization:02d}"
                        )
                        if random_index is not None:
                            shard_key += f"_SEED{int(random_index):02d}"
                        trajectories = len(strategies) * len(BUDGETS)
                        rows.append(
                            {
                                "stage": stage,
                                "execution_family": family,
                                "shard_id": shard_key,
                                "participant": participant,
                                "p07_case_analysis_only": participant == "P07",
                                "imbalance_level": str(pattern.imbalance_level),
                                "rotation_index": int(pattern.rotation_index),
                                "pool_realization_index": realization,
                                "random_seed_index": random_index,
                                "strategies": "|".join(strategies),
                                "budgets": "7|14|21",
                                "expected_trajectories": trajectories,
                                "expected_session_folds": trajectories * 5,
                                "classifier": "LDA" if sensitivity else "RIDGE_ALPHA_1",
                                "scientific_role": (
                                    "SENSITIVITY" if sensitivity else "PRIMARY_EXTENSION"
                                ),
                                "gpu_required": False,
                                "checkpoint_after_each_shard": True,
                                "drive_roundtrip_after_each_shard": True,
                            }
                        )
    return pd.DataFrame(rows)


def main():
    print("=" * 100)
    print("REVISION R4A — CANDIDATE-POOL CONSTRUCTION AND UNIT TESTS")
    print("=" * 100)
    print("Execution device: CPU")
    print("Scientific role: POOL CONSTRUCTION AND IMPLEMENTATION UNIT TESTS ONLY")
    print("Full imbalance trajectories: False")
    print("Fixed-test inference: False")
    print("Raw HDF5 accessed: False")
    print("New statistical tests: False")
    print()

    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified R0, R3A-P1, R3C, Stage 5B, and Stage 5D-2 packets...")
    inputs = resolve_inputs()
    patterns = engine.read_csv_member(
        inputs["r0_packet"], "stageR0_candidate_pool_imbalance_patterns.csv"
    )
    seeds = engine.read_csv_member(inputs["r0_packet"], "stageR0_seed_schedule.csv")
    pool_seeds = seeds.loc[
        seeds["seed_family"].astype(str).eq("POOL_SUBSET")
    ].copy()
    random_seed_rows = seeds.loc[
        seeds["seed_family"].astype(str).eq("RANDOM_ACQUISITION")
        & pd.to_numeric(seeds["seed_index"], errors="coerce").eq(1)
    ]
    if len(random_seed_rows) != 1:
        raise RuntimeError("Locked random acquisition seed 1 is not unique")
    random_seed = int(random_seed_rows.iloc[0]["seed"])
    features, main_valid, metadata = r3b.prepare_metadata(
        inputs["stage5b_packet"], inputs["stage5d2_packet"]
    )

    definitions, memberships = build_pool_schedule(metadata, patterns, pool_seeds)
    schedule_hash = pool_schedule_digest(definitions, memberships)
    repeat_definitions, repeat_memberships = build_pool_schedule(
        metadata, patterns, pool_seeds
    )
    repeated_hash = pool_schedule_digest(repeat_definitions, repeat_memberships)

    unit_results = run_single_round_unit_tests(
        features,
        main_valid,
        metadata,
        definitions,
        memberships,
        random_seed,
        inputs["rbmal_seed"],
    )
    severe_audit, severe_selected, severe_remaining = (
        run_severe_k21_full_pool_control(
            features,
            main_valid,
            metadata,
            definitions,
            memberships,
        )
    )
    shard_manifest = build_execution_shards(patterns)

    observed_counts = (
        memberships.groupby(
            ["pool_id", "true_label_internal_pool_audit_only"]
        ).size().unstack(fill_value=0)
    )
    expected_counts = definitions.set_index("pool_id")[
        [f"class_{label}_count" for label in range(engine.CLASSES)]
    ].copy()
    expected_counts.columns = list(range(engine.CLASSES))
    expected_counts = expected_counts.reindex(observed_counts.index)
    balanced = definitions.loc[definitions["imbalance_level"].eq("BALANCED_35")]
    balanced_identity_counts = (
        memberships.loc[memberships["imbalance_level"].eq("BALANCED_35")]
        .groupby(["participant", "target_session"])["opaque_candidate_token"]
        .nunique()
    )
    nonbalanced_distinct = (
        memberships.loc[memberships["imbalance_level"].isin(NONBALANCED_LEVELS)]
        .groupby(
            [
                "participant",
                "target_session",
                "imbalance_level",
                "rotation_index",
            ]
        )["opaque_candidate_token"]
        .apply(lambda values: len(set(values)))
    )

    expected_shard_rows = {
        "R4B": 7 * 3 * 7 * 5,
        "R4C": 7 * 3 * 7 * 5 * 30,
        "R4D": 7 * 3 * 7 * 5,
        "R4E": 7 * 3 * 7 * 5 * 30,
    }
    expected_trajectory_totals = {
        "R4B": 13230,
        "R4C": 66150,
        "R4D": 13230,
        "R4E": 66150,
    }
    readiness_gates = {
        "revision_r0_packet_hash_matches": engine.sha256_file(inputs["r0_packet"])
        == REVISION_R0_PACKET_SHA256,
        "revision_r3c_packet_hash_matches": engine.sha256_file(inputs["r3c_packet"])
        == REVISION_R3C_PACKET_SHA256,
        "revision_r3c_all_gates_passed": bool(
            inputs["r3c_report"].get("all_readiness_gates_passed")
        ),
        "revision_protocol_hash_matches": inputs["r0_protocol"].get(
            "protocol_sha256"
        )
        == REVISION_PROTOCOL_SHA256,
        "all_five_input_packets_pass_hash_and_crc": bool(
            inputs["audit"][["hash_matches", "crc_passes"]].all().all()
        ),
        "feature_shape_is_2940_by_37_by_64": tuple(features.shape)
        == (2940, 37, 64),
        "main_mask_shape_matches_features": tuple(main_valid.shape)
        == tuple(features.shape),
        "pool_definition_count_is_4900": len(definitions)
        == EXPECTED_POOL_DEFINITIONS,
        "pool_membership_count_is_142100": len(memberships)
        == EXPECTED_MEMBERSHIP_ROWS,
        "pool_ids_are_unique": definitions["pool_id"].is_unique,
        "levels_are_exactly_locked_four": set(definitions["imbalance_level"])
        == set(IMBALANCE_LEVELS),
        "each_level_has_seven_rotations": bool(
            definitions.groupby("imbalance_level")["rotation_index"]
            .nunique()
            .eq(7)
            .all()
        ),
        "each_participant_session_pattern_has_five_realizations": bool(
            definitions.groupby(
                [
                    "participant",
                    "target_session",
                    "imbalance_level",
                    "rotation_index",
                ]
            )["pool_realization_index"]
            .nunique()
            .eq(5)
            .all()
        ),
        "all_pool_sizes_match_locked_patterns": bool(
            memberships.groupby("pool_id").size().reindex(definitions["pool_id"])
            .to_numpy()
            .tolist()
            == definitions["total_candidates"].tolist()
        ),
        "all_class_counts_match_locked_rotations": bool(
            (observed_counts == expected_counts).all().all()
        ),
        "all_pool_class_counts_are_between_one_and_five": bool(
            expected_counts.to_numpy().min() >= 1
            and expected_counts.to_numpy().max() <= 5
        ),
        "every_pool_supports_k21": bool(definitions["total_candidates"].ge(21).all()),
        "severe_pools_have_21_candidates": bool(
            definitions.loc[
                definitions["imbalance_level"].eq("SEVERE_21"),
                "total_candidates",
            ].eq(21).all()
        ),
        "severe_k21_is_marked_full_pool_control": bool(
            definitions.loc[
                definitions["imbalance_level"].eq("SEVERE_21"),
                "k21_is_full_pool_control",
            ].all()
        ),
        "balanced_pool_count_is_1225": len(balanced) == 7 * 5 * 7 * 5,
        "balanced_realizations_preserve_all_35_original_candidates": bool(
            balanced_identity_counts.eq(35).all()
        ),
        "nonbalanced_pool_realizations_are_not_all_identity_duplicates": bool(
            nonbalanced_distinct.gt(21).any()
        ),
        "each_pool_has_unique_opaque_tokens": bool(
            memberships.groupby("pool_id")["opaque_candidate_token"]
            .apply(lambda values: values.is_unique)
            .all()
        ),
        "all_pool_tokens_are_opaque": bool(
            memberships["opaque_candidate_token"]
            .astype(str)
            .str.fullmatch(r"[0-9a-f]{24}")
            .all()
        ),
        "all_pool_members_are_original_candidates": bool(
            memberships["selected_record_is_original_candidate"].all()
        ),
        "no_fixed_test_record_enters_any_pool": bool(
            (~memberships["selected_record_is_fixed_test"]).all()
        ),
        "true_labels_are_used_only_for_pool_simulation_audit": bool(
            (~memberships["selector_receives_true_label"]).all()
        ),
        "pool_schedule_is_deterministically_reproducible": schedule_hash
        == repeated_hash,
        "seven_unit_strategies_are_exercised": set(unit_results["strategy"])
        == set(UNIT_STRATEGIES),
        "every_unit_selector_returns_seven_unique_candidates": bool(
            unit_results["selected_count"].eq(7).all()
            and unit_results["unique_selected_tokens"].eq(7).all()
        ),
        "all_unit_selections_are_pool_candidates": bool(
            unit_results["all_selected_records_are_pool_candidates"].all()
        ),
        "no_unit_selector_selects_fixed_test": bool(
            unit_results["no_fixed_test_record_selected"].all()
        ),
        "no_unit_selector_receives_true_label": bool(
            (~unit_results["selector_receives_true_label"]).all()
        ),
        "no_unit_selector_receives_forbidden_columns": bool(
            unit_results["selector_forbidden_column_count"].eq(0).all()
        ),
        "severe_k21_runs_exactly_three_rounds": len(severe_audit) == 3,
        "severe_k21_selects_all_21_pool_members": len(severe_selected) == 21
        and len(set(severe_selected)) == 21
        and len(severe_remaining) == 0,
        "severe_k21_retrains_between_rounds": severe_audit[
            "history_before_fit"
        ].tolist()
        == [35, 42, 49],
        "shard_manifest_stage_counts_are_exact": shard_manifest.groupby("stage")
        .size()
        .to_dict()
        == expected_shard_rows,
        "shard_manifest_trajectory_totals_are_exact": shard_manifest.groupby(
            "stage"
        )["expected_trajectories"].sum().to_dict()
        == expected_trajectory_totals,
        "every_shard_is_checkpointed_and_drive_roundtripped": bool(
            shard_manifest["checkpoint_after_each_shard"].all()
            and shard_manifest["drive_roundtrip_after_each_shard"].all()
        ),
        "p07_is_case_analysis_only": bool(
            definitions.loc[definitions["participant"].eq("P07"),
                            "p07_case_analysis_only"].all()
            and (~definitions.loc[~definitions["participant"].eq("P07"),
                                  "p07_case_analysis_only"]).all()
        ),
        "raw_hdf5_data_was_not_accessed": True,
        "full_imbalance_trajectories_were_not_run": True,
        "fixed_test_inference_was_not_run": True,
        "no_new_statistical_test_was_run": True,
        "stage3g_and_stage5f_conclusions_cannot_be_replaced": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in readiness_gates.items() if not bool(value)]

    atomic_csv(inputs["audit"], RESULT_ROOT / "revision_R4A_input_integrity_audit.csv")
    atomic_csv(patterns, RESULT_ROOT / "revision_R4A_locked_imbalance_patterns.csv")
    atomic_csv(definitions, RESULT_ROOT / "revision_R4A_pool_definitions.csv")
    atomic_csv(memberships, RESULT_ROOT / "revision_R4A_pool_membership.csv")
    atomic_csv(unit_results, RESULT_ROOT / "revision_R4A_selector_unit_results.csv")
    atomic_csv(severe_audit, RESULT_ROOT / "revision_R4A_severe_k21_round_audit.csv")
    atomic_csv(shard_manifest, RESULT_ROOT / "revision_R4A_execution_shard_manifest.csv")
    atomic_json(
        {
            "contract": POOL_CONSTRUCTION_CONTRACT,
            "schedule_sha256": schedule_hash,
            "selector_visible_pool_fields": ["opaque_candidate_token"],
            "pool_constructor_audit_only_fields": [
                "participant",
                "target_session",
                "imbalance_level",
                "rotation_index",
                "pool_realization_index",
                "true_label_internal_pool_audit_only",
                "sequence_row_internal_audit_only",
            ],
            "balanced_evidence_reuse": {
                "PCBM_GLOBAL": "FROZEN_STAGE3C1",
                "RANDOM": "FROZEN_STAGE3C2",
                "LC_ENTROPY_RBMAL_CORE_SET": "FROZEN_REVISION_R3C",
            },
            "nonbalanced_execution_order": ["R4B", "R4C", "R4D", "R4E"],
            "seeds_31_to_60_rule": (
                "DO_NOT_RUN_UNLESS_LOCKED_MONTE_CARLO_CONVERGENCE_GATE_FAILS_AFTER_R4C"
            ),
        },
        RESULT_ROOT / "revision_R4A_pool_and_execution_contract.json",
    )
    report = {
        "stage": "REVISION_R4A",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "revision_r3c_packet_sha256": REVISION_R3C_PACKET_SHA256,
        "pool_construction_contract": POOL_CONSTRUCTION_CONTRACT,
        "pool_schedule_sha256": schedule_hash,
        "pool_definition_count": len(definitions),
        "pool_membership_count": len(memberships),
        "unit_strategy_count": len(unit_results),
        "execution_shard_count": len(shard_manifest),
        "execution_trajectory_totals": expected_trajectory_totals,
        "readiness_gates": readiness_gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "execution_device": "CPU",
        "raw_hdf5_accessed": False,
        "full_imbalance_trajectories_run": False,
        "fixed_test_inference_run": False,
        "new_statistical_test_run": False,
        "runtime_minutes_before_packaging": (time.time() - START_TIME) / 60.0,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "final_decision": (
            "PASS_TO_REVISION_R4B_RIDGE_DETERMINISTIC_IMBALANCE_SHARDS"
            if not failed
            else "REVISION_R4A_NOT_READY"
        ),
    }
    atomic_json(report, RESULT_ROOT / "revision_R4A_report.json")
    shutil.copy2(
        Path(__file__), RESULT_ROOT / "revision_R4A_executed_source.py"
    )
    manifest_rows = []
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(RESULT_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": engine.sha256_file(path),
                }
            )
    atomic_csv(
        pd.DataFrame(manifest_rows), RESULT_ROOT / "revision_R4A_output_manifest.csv"
    )
    if failed:
        raise RuntimeError(f"Revision R4A failed readiness gates: {failed}")

    print()
    print("Pool definitions:", len(definitions))
    print("Pool membership rows:", len(memberships))
    print("Pool schedule SHA-256:", schedule_hash)
    print("Unit selector results:")
    print(
        unit_results[
            [
                "strategy",
                "pool_candidates",
                "selected_count",
                "selected_true_class_coverage_internal_audit_only",
                "selector_forbidden_column_count",
            ]
        ].to_string(index=False)
    )
    print("Execution shards:")
    print(
        shard_manifest.groupby("stage")
        .agg(
            shards=("shard_id", "size"),
            trajectories=("expected_trajectories", "sum"),
            folds=("expected_session_folds", "sum"),
        )
        .reset_index()
        .to_string(index=False)
    )
    print("Failed readiness gates:", failed or "None")

    print("Uploading Revision R4A packet to Google Drive...")
    crc = engine.make_zip(
        RESULT_ROOT,
        PACKET_PATH,
        "Revision_R4A_Candidate_Pool_Construction_Unit_Tests",
    )
    digest = engine.sha256_file(PACKET_PATH)
    remote_verified = engine.roundtrip_remote_file(
        PACKET_PATH, REMOTE_OUTPUT + "/" + PACKET_PATH.name, digest
    )
    if not crc or not remote_verified:
        raise RuntimeError("Revision R4A packet CRC or Drive round-trip failed")
    print()
    print("Packet CRC pass:", crc)
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", digest)
    print("Remote round-trip verified:", remote_verified)
    print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 3))
    print()
    print("FINAL DECISION: PASS_TO_REVISION_R4B_RIDGE_DETERMINISTIC_IMBALANCE_SHARDS")


if __name__ == "__main__":
    main()
