from __future__ import annotations

import io
import json
import os
import re
import resource
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import revision_R4A_candidate_pool_construction_unit_tests as r4a


r3c = r4a.r3c
r3b = r4a.r3b
engine = r4a.engine

START_TIME = time.time()
WORKING = engine.WORKING
INPUT_ROOT = WORKING / "REVISION_R4B_FROZEN_INPUTS"
TEMP_ROOT = WORKING / "REVISION_R4B_TEMP"
PROGRESS_ROOT = (
    WORKING
    / "DELTA_REVIEWER_REVISION"
    / "Revision_R4B_Ridge_Deterministic_Imbalance_Progress"
)
FINAL_ROOT = (
    WORKING
    / "DELTA_REVIEWER_REVISION"
    / "Revision_R4B_Ridge_Deterministic_Imbalance_Final"
)
PROGRESS_PACKET = WORKING / "revision_R4B_ridge_deterministic_imbalance_progress_packet.zip"
FINAL_PACKET = WORKING / "revision_R4B_ridge_deterministic_imbalance_packet.zip"
REMOTE_OUTPUT = (
    engine.REMOTE_BASE
    + "/Reviewer_Revision/Revision_R4B_Ridge_Deterministic_Imbalance_Shards"
)
REMOTE_SHARDS = REMOTE_OUTPUT + "/shards"
for directory in (INPUT_ROOT, TEMP_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

REVISION_R4A_PACKET_SHA256 = (
    "0fac1fc016310ab5043ed728f3cdbae5b7ad30e7fce6ecbeaf0d9e2e9f136580"
)
REVISION_PROTOCOL_SHA256 = engine.REVISION_PROTOCOL_SHA256
STAGE = "R4B"
STRATEGIES = list(r4a.DETERMINISTIC_STRATEGIES)
BUDGETS = [7, 14, 21]
TARGET_SESSIONS = [1, 2, 3, 4, 5]
EXPECTED_SHARDS = 735
EXPECTED_TRAJECTORIES = 13230
EXPECTED_FOLDS = 66150
EXPECTED_PREDICTIONS = 2315250
EXPECTED_SELECTIONS = 926100
EXPECTED_SELECTOR_CALLS = 132300
EXPECTED_CANDIDATE_AUDIT_ROWS = 2954700
MAX_RUNTIME_HOURS = float(os.environ.get("R4B_MAX_RUNTIME_HOURS", "10.0"))
MAX_RUNTIME_SECONDS = MAX_RUNTIME_HOURS * 3600.0
FINALIZATION_RESERVE_SECONDS = float(
    os.environ.get("R4B_FINALIZATION_RESERVE_MINUTES", "75")
) * 60.0
SHARD_PACKET_PATTERN = re.compile(
    r"^(?P<shard_id>R4B_[A-Za-z0-9_]+)__(?P<sha256>[0-9a-f]{64})\.zip$"
)

SHARD_TABLES = {
    "selections": "revision_R4B_shard_selection_trace.csv",
    "candidate_audits": "revision_R4B_shard_candidate_score_audit.csv",
    "step_audits": "revision_R4B_shard_sequential_step_audit.csv",
    "selector_calls": "revision_R4B_shard_selector_call_audit.csv",
    "oof_rows": "revision_R4B_shard_ridge_oof_fold_audit.csv",
    "normalizers": "revision_R4B_shard_normalizer_audit.csv",
    "folds": "revision_R4B_shard_fold_metrics.csv",
    "predictions": "revision_R4B_shard_repetition_predictions.csv",
    "recalls": "revision_R4B_shard_per_class_recall.csv",
    "confusions": "revision_R4B_shard_confusion_matrices_long.csv",
    "coverage": "revision_R4B_shard_selected_class_distribution.csv",
}


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


def resolve_inputs():
    parents = r4a.resolve_inputs()
    r4a_packet, r4a_source = engine.resolve_packet(
        "revision_R4A_candidate_pool_construction_unit_test_packet.zip",
        REVISION_R4A_PACKET_SHA256,
    )
    r4a_report = engine.read_json_member(r4a_packet, "revision_R4A_report.json")
    if not r4a_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("Revision R4A parent gates did not pass")
    if r4a_report.get("revision_protocol_sha256") != REVISION_PROTOCOL_SHA256:
        raise RuntimeError("Revision R4A protocol hash drift")
    record = pd.DataFrame(
        [
            {
                "packet": r4a_packet.name,
                "expected_sha256": REVISION_R4A_PACKET_SHA256,
                "observed_sha256": engine.sha256_file(r4a_packet),
                "hash_matches": engine.sha256_file(r4a_packet)
                == REVISION_R4A_PACKET_SHA256,
                "crc_passes": engine.archive_crc_passes(r4a_packet),
                "source": r4a_source,
            }
        ]
    )
    audit = pd.concat([parents["audit"], record], ignore_index=True)
    if not audit[["hash_matches", "crc_passes"]].all().all():
        raise RuntimeError("Revision R4B frozen-input integrity failed")
    parents.update(
        {
            "r4a_packet": r4a_packet,
            "r4a_report": r4a_report,
            "audit": audit,
            "pool_definitions": engine.read_csv_member(
                r4a_packet, "revision_R4A_pool_definitions.csv"
            ),
            "pool_membership": engine.read_csv_member(
                r4a_packet, "revision_R4A_pool_membership.csv"
            ),
            "shard_manifest": engine.read_csv_member(
                r4a_packet, "revision_R4A_execution_shard_manifest.csv"
            ),
        }
    )
    return parents


def discover_completed_shards(expected_ids):
    result = engine.rclone(["lsf", REMOTE_SHARDS, "--files-only"], check=False)
    if result.returncode != 0:
        return {}, []
    mapping = {}
    duplicates = []
    for basename in result.stdout.splitlines():
        match = SHARD_PACKET_PATTERN.fullmatch(Path(basename).name)
        if match is None:
            continue
        shard_id = match.group("shard_id")
        if shard_id not in expected_ids:
            continue
        record = {
            "shard_id": shard_id,
            "sha256": match.group("sha256"),
            "remote_path": REMOTE_SHARDS + "/" + Path(basename).name,
            "remote_basename": Path(basename).name,
        }
        if shard_id in mapping:
            duplicates.append(record)
        else:
            mapping[shard_id] = record
    return mapping, duplicates


def timed(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def margin_acquisition(
    strategy, features, main_valid, metadata, history_rows, remaining_rows
):
    state, fit_seconds = timed(
        engine.fit_history_state,
        features,
        main_valid,
        metadata,
        history_rows,
    )
    scored, score_seconds = timed(
        engine.score_repetitions,
        state,
        features,
        main_valid,
        remaining_rows,
    )
    decision_scores, predicted, margins = scored
    selector_frame = engine.selector_frame(
        metadata, remaining_rows, predicted, margins
    )
    if strategy == "PCBM_ORIGINAL":
        selected_tokens, selector_seconds = timed(
            engine.select_pcbm, selector_frame
        )
    elif strategy == "GLOBAL_MARGIN_ORIGINAL":
        selected_tokens, selector_seconds = timed(
            engine.select_global_margin, selector_frame
        )
    else:
        raise ValueError(strategy)
    selected_set = set(selected_tokens)
    candidate_rows = []
    for index, token in enumerate(
        selector_frame["opaque_candidate_token"].astype(str)
    ):
        row = {
            "opaque_candidate_token": token,
            "raw_predicted_label_internal_audit_only": int(predicted[index]),
            "raw_margin_internal_audit_only": float(margins[index]),
            "acquisition_score": float(-margins[index]),
            "selected_this_round": token in selected_set,
        }
        for label in range(engine.CLASSES):
            row[f"raw_decision_score_{label}"] = float(
                decision_scores[index, label]
            )
        candidate_rows.append(row)
    call = {
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "selector_seconds": selector_seconds,
        "selector_schema": "|".join(selector_frame.columns),
        "selector_schema_exact": selector_frame.columns.tolist()
        == engine.SELECTOR_COLUMNS,
        "selector_forbidden_column_count": len(
            set(selector_frame.columns).intersection(
                engine.FORBIDDEN_SELECTOR_COLUMNS
            )
        ),
        "oof_fold_count": 0,
        "oof_repetition_count": 0,
        "calibrator_max_iterations_used": 0,
        "oof_std_floor_event_count": 0,
        "full_history_std_floor_count": 0,
        "state": state,
        "oof_audit": pd.DataFrame(),
    }
    return selected_tokens, pd.DataFrame(candidate_rows), call, pd.DataFrame()


def acquire(
    strategy,
    features,
    main_valid,
    metadata,
    history_rows,
    remaining_rows,
    rbmal_seed,
):
    if strategy in {"PCBM_ORIGINAL", "GLOBAL_MARGIN_ORIGINAL"}:
        return margin_acquisition(
            strategy,
            features,
            main_valid,
            metadata,
            history_rows,
            remaining_rows,
        )
    if strategy in {"LEAST_CONFIDENCE", "PREDICTIVE_ENTROPY"}:
        return r3c.probability_acquisition(
            strategy,
            features,
            main_valid,
            metadata,
            history_rows,
            remaining_rows,
        )
    if strategy in {"RBMAL_MARGIN_DIVERSITY", "CORE_SET_GREEDY"}:
        return r3c.diversity_acquisition(
            strategy,
            features,
            main_valid,
            metadata,
            history_rows,
            remaining_rows,
            rbmal_seed,
        )
    raise ValueError(strategy)


def append_normalizer(
    rows,
    shard_id,
    pool_id,
    trajectory_id,
    participant,
    level,
    rotation,
    realization,
    session,
    strategy,
    budget,
    fit_role,
    history_count,
    state,
):
    rows.append(
        {
            "shard_id": shard_id,
            "pool_id": pool_id,
            "trajectory_id": trajectory_id,
            "participant": participant,
            "imbalance_level": level,
            "rotation_index": rotation,
            "pool_realization_index": realization,
            "target_session": session,
            "strategy": strategy,
            "query_budget": budget,
            "fit_role": fit_role,
            "history_repetitions": history_count,
            "minimum_mean": float(np.min(state["means"])),
            "maximum_mean": float(np.max(state["means"])),
            "minimum_std": float(np.min(state["stds"])),
            "maximum_std": float(np.max(state["stds"])),
            "minimum_valid_count": int(np.min(state["counts"])),
            "means_dtype": str(state["means"].dtype),
            "stds_dtype": str(state["stds"].dtype),
            "model_coefficient_dtype": str(
                np.asarray(state["model"].coef_).dtype
            ),
        }
    )


def run_shard(shard, features, main_valid, metadata, pool_definitions, pool_membership, rbmal_seed):
    shard_id = str(shard.shard_id)
    participant = str(shard.participant)
    level = str(shard.imbalance_level)
    rotation = int(shard.rotation_index)
    realization = int(shard.pool_realization_index)
    definitions = pool_definitions.loc[
        pool_definitions["participant"].eq(participant)
        & pool_definitions["imbalance_level"].eq(level)
        & pd.to_numeric(pool_definitions["rotation_index"]).eq(rotation)
        & pd.to_numeric(pool_definitions["pool_realization_index"]).eq(realization)
    ].copy()
    if len(definitions) != 5:
        raise RuntimeError(f"Expected five session pools for {shard_id}")
    session_to_definition = {
        int(row.target_session): row for row in definitions.itertuples(index=False)
    }
    membership_by_pool = {
        pool_id: group["sequence_row_internal_audit_only"].astype(int).tolist()
        for pool_id, group in pool_membership.loc[
            pool_membership["pool_id"].isin(definitions["pool_id"])
        ].groupby("pool_id", sort=False)
    }

    output = {key: [] for key in SHARD_TABLES}
    for strategy in STRATEGIES:
        for budget in BUDGETS:
            trajectory_id = (
                f"{shard_id}_{strategy}_K{budget:02d}"
            )
            history = engine.initial_history_rows(metadata, participant).tolist()
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
                fit_seconds_total = 0.0
                score_seconds_total = 0.0
                selector_seconds_total = 0.0
                for query_round in range(1, budget // 7 + 1):
                    history_before = list(history)
                    selected_tokens, candidate_frame, call, step_audit = acquire(
                        strategy,
                        features,
                        main_valid,
                        metadata,
                        history_before,
                        remaining,
                        rbmal_seed,
                    )
                    selected_rows = engine.reveal_rows(
                        metadata, selected_tokens, remaining
                    )
                    selected_set = set(map(int, selected_rows))
                    true_labels = metadata.iloc[selected_rows]["label"].to_numpy(
                        dtype=int
                    )
                    for position, (token, row, label) in enumerate(
                        zip(selected_tokens, selected_rows, true_labels), start=1
                    ):
                        output["selections"].append(
                            {
                                "shard_id": shard_id,
                                "pool_id": pool_id,
                                "trajectory_id": trajectory_id,
                                "participant": participant,
                                "case_analysis": participant == "P07",
                                "imbalance_level": level,
                                "rotation_index": rotation,
                                "pool_realization_index": realization,
                                "target_session": session,
                                "strategy": strategy,
                                "query_budget": budget,
                                "query_round": query_round,
                                "position_in_round": position,
                                "opaque_candidate_token": str(token),
                                "sequence_row_internal_audit_only": int(row),
                                "true_label_after_reveal": int(label),
                                "selected_record_is_pool_candidate": int(row)
                                in set(remaining),
                                "selected_record_is_fixed_test": int(row)
                                in fixed_set,
                            }
                        )
                    candidate_frame = candidate_frame.copy()
                    prefix = {
                        "shard_id": shard_id,
                        "pool_id": pool_id,
                        "trajectory_id": trajectory_id,
                        "participant": participant,
                        "imbalance_level": level,
                        "rotation_index": rotation,
                        "pool_realization_index": realization,
                        "target_session": session,
                        "strategy": strategy,
                        "query_budget": budget,
                        "query_round": query_round,
                    }
                    for row in candidate_frame.to_dict("records"):
                        output["candidate_audits"].append({**prefix, **row})
                    if not step_audit.empty:
                        for row in step_audit.to_dict("records"):
                            output["step_audits"].append(
                                {
                                    **prefix,
                                    "opaque_candidate_token": row[
                                        "opaque_candidate_token"
                                    ],
                                    "selected_this_round": True,
                                    "step_audit_json": json.dumps(
                                        row, sort_keys=True, default=str
                                    ),
                                }
                            )
                    history_sessions = metadata.iloc[
                        np.asarray(history_before, dtype=int)
                    ]["session"].to_numpy(dtype=int)
                    output["selector_calls"].append(
                        {
                            **prefix,
                            "pool_candidates_at_session_start": pool_size,
                            "history_repetitions_before_query": len(history_before),
                            "remaining_candidates_before_query": len(remaining),
                            "selected_count": len(selected_rows),
                            "fit_seconds": call["fit_seconds"],
                            "score_seconds": call["score_seconds"],
                            "selector_seconds": call["selector_seconds"],
                            "selector_schema": call["selector_schema"],
                            "selector_schema_exact": call["selector_schema_exact"],
                            "selector_forbidden_column_count": call[
                                "selector_forbidden_column_count"
                            ],
                            "oof_fold_count": call["oof_fold_count"],
                            "oof_repetition_count": call["oof_repetition_count"],
                            "calibrator_max_iterations_used": call[
                                "calibrator_max_iterations_used"
                            ],
                            "oof_std_floor_event_count": call[
                                "oof_std_floor_event_count"
                            ],
                            "full_history_std_floor_count": call[
                                "full_history_std_floor_count"
                            ],
                            "maximum_history_session": int(history_sessions.max()),
                            "future_session_used": bool(
                                (history_sessions > session).any()
                            ),
                            "fixed_test_used_for_training_or_selection": bool(
                                set(history_before).intersection(fixed_set)
                                or selected_set.intersection(fixed_set)
                            ),
                        }
                    )
                    if not call["oof_audit"].empty:
                        for row in call["oof_audit"].to_dict("records"):
                            output["oof_rows"].append({**prefix, **row})
                    append_normalizer(
                        output["normalizers"],
                        shard_id,
                        pool_id,
                        trajectory_id,
                        participant,
                        level,
                        rotation,
                        realization,
                        session,
                        strategy,
                        budget,
                        f"QUERY_ROUND_{query_round}",
                        len(history_before),
                        call["state"],
                    )
                    fit_seconds_total += call["fit_seconds"]
                    score_seconds_total += call["score_seconds"]
                    selector_seconds_total += call["selector_seconds"]
                    history.extend(map(int, selected_rows))
                    session_selected.extend(map(int, selected_rows))
                    remaining = [
                        row for row in remaining if int(row) not in selected_set
                    ]

                final_state, final_fit_seconds = timed(
                    engine.fit_history_state,
                    features,
                    main_valid,
                    metadata,
                    history,
                )
                evaluated, evaluation_seconds = timed(
                    r3c.evaluate_fixed_test,
                    final_state,
                    features,
                    main_valid,
                    metadata,
                    fixed_rows,
                )
                run_id = f"{trajectory_id}_S{session:02d}"
                history_meta = metadata.iloc[np.asarray(history, dtype=int)]
                source_sessions = history_meta["session"].to_numpy(dtype=int)
                fixed_meta = metadata.iloc[np.asarray(fixed_rows, dtype=int)]
                output["folds"].append(
                    {
                        "run_id": run_id,
                        "shard_id": shard_id,
                        "pool_id": pool_id,
                        "trajectory_id": trajectory_id,
                        "participant": participant,
                        "case_analysis": participant == "P07",
                        "imbalance_level": level,
                        "rotation_index": rotation,
                        "pool_realization_index": realization,
                        "target_session": session,
                        "strategy": strategy,
                        "query_budget": budget,
                        "pool_candidates": pool_size,
                        "history_repetitions": len(history),
                        "selected_repetitions_this_session": len(session_selected),
                        "fixed_test_repetitions": len(fixed_rows),
                        "repetition_accuracy": evaluated["accuracy"],
                        "repetition_balanced_accuracy": evaluated[
                            "balanced_accuracy"
                        ],
                        "repetition_macro_f1": evaluated["macro_f1"],
                        "repetition_errors": int(
                            np.sum(evaluated["true"] != evaluated["predicted"])
                        ),
                        "balanced_accuracy_equals_accuracy": bool(
                            abs(
                                evaluated["balanced_accuracy"]
                                - evaluated["accuracy"]
                            )
                            < 1e-15
                        ),
                        "maximum_source_session": int(source_sessions.max()),
                        "future_session_used": bool(
                            (source_sessions > session).any()
                        ),
                        "fixed_test_entered_history": bool(
                            history_meta["fixed_test_never_query"].astype(bool).any()
                        ),
                        "test_labels_are_balanced_five_per_class": bool(
                            fixed_meta.groupby("label").size().eq(5).all()
                            and fixed_meta["label"].nunique() == 7
                        ),
                        "query_fit_seconds": fit_seconds_total,
                        "candidate_score_seconds": score_seconds_total,
                        "selector_seconds": selector_seconds_total,
                        "final_refit_seconds": final_fit_seconds,
                        "fixed_test_inference_seconds": evaluation_seconds,
                        "end_to_end_session_seconds": time.perf_counter()
                        - session_started,
                    }
                )
                append_normalizer(
                    output["normalizers"],
                    shard_id,
                    pool_id,
                    trajectory_id,
                    participant,
                    level,
                    rotation,
                    realization,
                    session,
                    strategy,
                    budget,
                    "FINAL_HISTORY_EVALUATION",
                    len(history),
                    final_state,
                )
                for position, row in enumerate(fixed_rows):
                    record = {
                        "run_id": run_id,
                        "shard_id": shard_id,
                        "pool_id": pool_id,
                        "participant": participant,
                        "imbalance_level": level,
                        "rotation_index": rotation,
                        "pool_realization_index": realization,
                        "target_session": session,
                        "strategy": strategy,
                        "query_budget": budget,
                        "test_position": position + 1,
                        "sequence_row_internal_audit_only": int(row),
                        "true_label": int(evaluated["true"][position]),
                        "predicted_label": int(evaluated["predicted"][position]),
                        "correct": bool(
                            evaluated["true"][position]
                            == evaluated["predicted"][position]
                        ),
                        "raw_margin": float(evaluated["margins"][position]),
                    }
                    for label in range(engine.CLASSES):
                        record[f"decision_score_{label}"] = float(
                            evaluated["scores"][position, label]
                        )
                    output["predictions"].append(record)
                for true_label in range(engine.CLASSES):
                    mask = evaluated["true"] == true_label
                    output["recalls"].append(
                        {
                            "run_id": run_id,
                            "shard_id": shard_id,
                            "participant": participant,
                            "imbalance_level": level,
                            "rotation_index": rotation,
                            "pool_realization_index": realization,
                            "target_session": session,
                            "strategy": strategy,
                            "query_budget": budget,
                            "class_label": true_label,
                            "class_support": int(mask.sum()),
                            "class_recall": float(
                                np.mean(
                                    evaluated["predicted"][mask] == true_label
                                )
                            ),
                        }
                    )
                    for predicted_label in range(engine.CLASSES):
                        output["confusions"].append(
                            {
                                "run_id": run_id,
                                "shard_id": shard_id,
                                "participant": participant,
                                "imbalance_level": level,
                                "rotation_index": rotation,
                                "pool_realization_index": realization,
                                "target_session": session,
                                "strategy": strategy,
                                "query_budget": budget,
                                "true_label": true_label,
                                "predicted_label": predicted_label,
                                "count": int(
                                    np.sum(
                                        mask
                                        & (
                                            evaluated["predicted"]
                                            == predicted_label
                                        )
                                    )
                                ),
                            }
                        )
                selected_labels = metadata.iloc[
                    np.asarray(session_selected, dtype=int)
                ]["label"].to_numpy(dtype=int)
                counts, coverage, entropy = r3c.class_distribution_metrics(
                    selected_labels
                )
                coverage_row = {
                    "run_id": run_id,
                    "shard_id": shard_id,
                    "pool_id": pool_id,
                    "participant": participant,
                    "case_analysis": participant == "P07",
                    "imbalance_level": level,
                    "rotation_index": rotation,
                    "pool_realization_index": realization,
                    "target_session": session,
                    "strategy": strategy,
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
    expected_candidate_rows = 30 * (6 * pool_size - 28)
    folds = outputs["folds"]
    selections = outputs["selections"]
    calls = outputs["selector_calls"]
    candidate = outputs["candidate_audits"]
    predictions = outputs["predictions"]
    normalizers = outputs["normalizers"]
    coverage = outputs["coverage"]
    gates = {
        "trajectory_count_is_18": folds["trajectory_id"].nunique() == 18,
        "fold_count_is_90": len(folds) == 90,
        "prediction_count_is_3150": len(predictions) == 3150,
        "selection_count_is_1260": len(selections) == 1260,
        "selector_call_count_is_180": len(calls) == 180,
        "candidate_audit_count_matches_pool_size": len(candidate)
        == expected_candidate_rows,
        "normalizer_row_count_is_270": len(normalizers) == 270,
        "recall_row_count_is_630": len(outputs["recalls"]) == 630,
        "confusion_row_count_is_4410": len(outputs["confusions"]) == 4410,
        "coverage_row_count_is_90": len(coverage) == 90,
        "each_trajectory_has_five_sessions": bool(
            folds.groupby("trajectory_id")["target_session"].nunique().eq(5).all()
        ),
        "selection_counts_match_budgets": bool(
            selections.groupby(["trajectory_id", "target_session"])
            .size()
            .reset_index(name="observed")
            .merge(
                folds[["trajectory_id", "target_session", "query_budget"]],
                on=["trajectory_id", "target_session"],
                validate="one_to_one",
            )
            .eval("observed == query_budget")
            .all()
        ),
        "all_selected_records_are_pool_candidates": bool(
            selections["selected_record_is_pool_candidate"].all()
        ),
        "no_fixed_test_record_was_selected": bool(
            (~selections["selected_record_is_fixed_test"]).all()
        ),
        "all_selector_calls_have_exact_schema": bool(
            calls["selector_schema_exact"].all()
        ),
        "no_selector_received_forbidden_columns": bool(
            calls["selector_forbidden_column_count"].eq(0).all()
        ),
        "all_probability_calls_have_five_oof_folds": bool(
            calls.loc[
                calls["strategy"].isin(
                    ["LEAST_CONFIDENCE", "PREDICTIVE_ENTROPY"]
                ),
                "oof_fold_count",
            ].eq(5).all()
        ),
        "all_oof_normalizers_are_history_only": bool(
            outputs["oof_rows"][
                "normalizer_training_rows_are_history_only"
            ].all()
            and outputs["oof_rows"][
                "validation_row_is_not_in_fold_training"
            ].all()
        ),
        "no_future_session_is_used": bool(
            (~calls["future_session_used"]).all()
            and (~folds["future_session_used"]).all()
        ),
        "fixed_test_never_enters_history": bool(
            (~calls["fixed_test_used_for_training_or_selection"]).all()
            and (~folds["fixed_test_entered_history"]).all()
        ),
        "all_metrics_are_finite_and_in_range": bool(
            np.isfinite(
                folds[
                    [
                        "repetition_accuracy",
                        "repetition_balanced_accuracy",
                        "repetition_macro_f1",
                    ]
                ].to_numpy(float)
            ).all()
            and folds[
                [
                    "repetition_accuracy",
                    "repetition_balanced_accuracy",
                    "repetition_macro_f1",
                ]
            ].ge(0).all().all()
            and folds[
                [
                    "repetition_accuracy",
                    "repetition_balanced_accuracy",
                    "repetition_macro_f1",
                ]
            ].le(1).all().all()
        ),
        "balanced_accuracy_equals_accuracy": bool(
            folds["balanced_accuracy_equals_accuracy"].all()
        ),
        "all_normalizers_are_finite_positive_float32": bool(
            np.isfinite(
                normalizers[
                    ["minimum_mean", "maximum_mean", "minimum_std", "maximum_std"]
                ].to_numpy(float)
            ).all()
            and normalizers["minimum_std"].gt(0).all()
            and normalizers["minimum_valid_count"].gt(0).all()
            and normalizers["means_dtype"].eq("float32").all()
            and normalizers["stds_dtype"].eq("float32").all()
            and normalizers["model_coefficient_dtype"].eq("float32").all()
        ),
        "severe_k21_selects_full_pool": bool(
            True
            if level != "SEVERE_21"
            else selections.loc[selections["query_budget"].eq(21)]
            .groupby(["trajectory_id", "target_session"])[
                "opaque_candidate_token"
            ]
            .nunique()
            .eq(21)
            .all()
        ),
        "p07_is_case_analysis_only": bool(
            folds["case_analysis"].all()
            if str(shard.participant) == "P07"
            else (~folds["case_analysis"]).all()
        ),
    }
    return gates


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
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "revision_r4a_packet_sha256": REVISION_R4A_PACKET_SHA256,
        "trajectory_count": int(outputs["folds"]["trajectory_id"].nunique()),
        "fold_count": len(outputs["folds"]),
        "prediction_count": len(outputs["predictions"]),
        "selection_count": len(outputs["selections"]),
        "selector_call_count": len(outputs["selector_calls"]),
        "candidate_audit_count": len(outputs["candidate_audits"]),
        "readiness_gates": gates,
        "failed_readiness_gates": [key for key, value in gates.items() if not value],
        "all_readiness_gates_passed": all(gates.values()),
        "runtime_seconds": runtime_seconds,
        "raw_hdf5_accessed": False,
        "new_statistical_test_run": False,
        "p07_case_analysis_only": str(shard.participant) == "P07",
    }
    atomic_json(report, shard_root / "revision_R4B_shard_report.json")
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
    atomic_csv(
        pd.DataFrame(manifest), shard_root / "revision_R4B_shard_manifest.csv"
    )
    packet = TEMP_ROOT / f"{shard_id}.zip"
    crc = engine.make_zip(shard_root, packet, shard_id)
    if not crc:
        raise RuntimeError(f"Shard CRC failed: {shard_id}")
    digest = engine.sha256_file(packet)
    return shard_root, packet, digest


def upload_verified_shard(shard_id, packet, digest):
    pending = REMOTE_SHARDS + f"/{shard_id}.pending.zip"
    final = REMOTE_SHARDS + f"/{shard_id}__{digest}.zip"
    if not engine.roundtrip_remote_file(packet, pending, digest):
        raise RuntimeError(f"Shard round-trip failed: {shard_id}")
    engine.rclone(["moveto", pending, final, "--retries", "5", "--timeout", "5m"])
    return final


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
    atomic_json(report, PROGRESS_ROOT / "revision_R4B_progress_report.json")
    atomic_csv(inputs["audit"], PROGRESS_ROOT / "revision_R4B_input_audit.csv")
    atomic_csv(manifest, PROGRESS_ROOT / "revision_R4B_expected_shards.csv")
    atomic_csv(completed_frame, PROGRESS_ROOT / "revision_R4B_completed_shards.csv")
    atomic_csv(remaining, PROGRESS_ROOT / "revision_R4B_remaining_shards.csv")
    atomic_csv(pd.DataFrame(duplicates), PROGRESS_ROOT / "revision_R4B_duplicate_records.csv")
    crc = engine.make_zip(
        PROGRESS_ROOT,
        PROGRESS_PACKET,
        "Revision_R4B_Ridge_Deterministic_Imbalance_Progress",
    )
    digest = engine.sha256_file(PROGRESS_PACKET)
    verified = engine.roundtrip_remote_file(
        PROGRESS_PACKET,
        REMOTE_OUTPUT + "/" + PROGRESS_PACKET.name,
        digest,
    )
    if not crc or not verified:
        raise RuntimeError("R4B progress packet persistence failed")
    return digest


def read_csv_from_shard(packet, basename):
    return pd.read_csv(io.BytesIO(engine.archive_member(packet, basename)))


def download_and_aggregate(inputs, manifest, completed):
    if FINAL_ROOT.exists():
        shutil.rmtree(FINAL_ROOT)
    FINAL_ROOT.mkdir(parents=True)
    aggregate_keys = ["folds", "coverage", "recalls", "confusions", "selector_calls"]
    aggregate_paths = {
        key: FINAL_ROOT / f"revision_R4B_aggregate_{key}.csv"
        for key in aggregate_keys
    }
    for path in aggregate_paths.values():
        path.unlink(missing_ok=True)
    integrity_rows = []
    count_totals = {
        "trajectory_count": 0,
        "fold_count": 0,
        "prediction_count": 0,
        "selection_count": 0,
        "selector_call_count": 0,
        "candidate_audit_count": 0,
    }
    temporary_packet = TEMP_ROOT / "aggregate_download.zip"
    for index, shard in enumerate(manifest.itertuples(index=False), start=1):
        record = completed[str(shard.shard_id)]
        engine.rclone(
            [
                "copyto",
                record["remote_path"],
                str(temporary_packet),
                "--retries",
                "5",
                "--timeout",
                "5m",
            ]
        )
        observed_hash = engine.sha256_file(temporary_packet)
        crc = engine.archive_crc_passes(temporary_packet)
        report = engine.read_json_member(
            temporary_packet, "revision_R4B_shard_report.json"
        )
        valid = bool(
            observed_hash == record["sha256"]
            and crc
            and report.get("all_readiness_gates_passed", False)
            and report.get("shard_id") == str(shard.shard_id)
            and report.get("revision_protocol_sha256") == REVISION_PROTOCOL_SHA256
        )
        integrity_rows.append(
            {
                **record,
                "observed_sha256": observed_hash,
                "hash_matches_filename": observed_hash == record["sha256"],
                "crc_passes": crc,
                "report_gates_passed": bool(
                    report.get("all_readiness_gates_passed", False)
                ),
                "shard_identity_matches": report.get("shard_id")
                == str(shard.shard_id),
                "verified": valid,
            }
        )
        if not valid:
            raise RuntimeError(
                f"Final shard integrity verification failed: {shard.shard_id}"
            )
        for key in count_totals:
            count_totals[key] += int(report[key])
        for key in aggregate_keys:
            frame = read_csv_from_shard(temporary_packet, SHARD_TABLES[key])
            if key == "confusions":
                frame = (
                    frame.groupby(
                        [
                            "participant",
                            "imbalance_level",
                            "target_session",
                            "strategy",
                            "query_budget",
                            "true_label",
                            "predicted_label",
                        ],
                        as_index=False,
                    )["count"]
                    .sum()
                )
            frame.to_csv(
                aggregate_paths[key],
                mode="a",
                index=False,
                header=not aggregate_paths[key].exists(),
            )
        temporary_packet.unlink(missing_ok=True)
        if index % 25 == 0 or index == len(manifest):
            print(
                f"Final shard verification: {index}/{len(manifest)}",
                flush=True,
            )

    folds = pd.read_csv(aggregate_paths["folds"])
    coverage = pd.read_csv(aggregate_paths["coverage"])
    recalls = pd.read_csv(aggregate_paths["recalls"])
    calls = pd.read_csv(aggregate_paths["selector_calls"])
    confusion_parts = []
    for part in pd.read_csv(aggregate_paths["confusions"], chunksize=250000):
        confusion_parts.append(part)
    confusions = pd.concat(confusion_parts, ignore_index=True)
    confusions = (
        confusions.groupby(
            [
                "participant",
                "imbalance_level",
                "target_session",
                "strategy",
                "query_budget",
                "true_label",
                "predicted_label",
            ],
            as_index=False,
        )["count"]
        .sum()
    )
    atomic_csv(
        confusions,
        FINAL_ROOT / "revision_R4B_confusion_matrices_aggregated.csv",
    )
    aggregate_paths["confusions"].unlink(missing_ok=True)

    trajectory_summary = (
        folds.groupby(
            [
                "participant",
                "case_analysis",
                "imbalance_level",
                "rotation_index",
                "pool_realization_index",
                "strategy",
                "query_budget",
            ],
            as_index=False,
        )
        .agg(
            target_sessions=("target_session", "nunique"),
            mean_repetition_balanced_accuracy=(
                "repetition_balanced_accuracy",
                "mean",
            ),
            mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
            total_repetition_errors=("repetition_errors", "sum"),
            mean_end_to_end_session_seconds=(
                "end_to_end_session_seconds",
                "mean",
            ),
        )
    )
    participant_level = (
        trajectory_summary.groupby(
            [
                "participant",
                "case_analysis",
                "imbalance_level",
                "strategy",
                "query_budget",
            ],
            as_index=False,
        )
        .agg(
            pool_trajectory_replicates=("rotation_index", "size"),
            mean_repetition_balanced_accuracy=(
                "mean_repetition_balanced_accuracy",
                "mean",
            ),
            std_across_pool_trajectories=(
                "mean_repetition_balanced_accuracy",
                "std",
            ),
            mean_repetition_macro_f1=("mean_repetition_macro_f1", "mean"),
            total_repetition_errors=("total_repetition_errors", "sum"),
        )
    )
    able = (
        participant_level.loc[participant_level["participant"].ne("P07")]
        .groupby(["imbalance_level", "strategy", "query_budget"], as_index=False)
        .agg(
            participants=("participant", "nunique"),
            pool_trajectories_per_participant=(
                "pool_trajectory_replicates",
                "min",
            ),
            mean_repetition_balanced_accuracy=(
                "mean_repetition_balanced_accuracy",
                "mean",
            ),
            std_between_participants=(
                "mean_repetition_balanced_accuracy",
                "std",
            ),
            mean_repetition_macro_f1=("mean_repetition_macro_f1", "mean"),
        )
    )
    p07 = participant_level.loc[
        participant_level["participant"].eq("P07")
    ].copy()
    coverage_summary = (
        coverage.groupby(
            ["participant", "imbalance_level", "strategy", "query_budget"],
            as_index=False,
        )
        .agg(
            session_pool_folds=("run_id", "size"),
            mean_selected_class_coverage=("selected_class_coverage", "mean"),
            minimum_selected_class_coverage=("selected_class_coverage", "min"),
            maximum_selected_class_coverage=("selected_class_coverage", "max"),
            mean_selected_normalized_class_entropy=(
                "selected_normalized_class_entropy",
                "mean",
            ),
            std_selected_normalized_class_entropy=(
                "selected_normalized_class_entropy",
                "std",
            ),
        )
    )
    compute_summary = (
        folds.groupby(["imbalance_level", "strategy", "query_budget"], as_index=False)
        .agg(
            folds=("run_id", "size"),
            mean_query_fit_seconds=("query_fit_seconds", "mean"),
            mean_candidate_score_seconds=("candidate_score_seconds", "mean"),
            mean_selector_seconds=("selector_seconds", "mean"),
            mean_final_refit_seconds=("final_refit_seconds", "mean"),
            mean_fixed_test_inference_seconds=(
                "fixed_test_inference_seconds",
                "mean",
            ),
            mean_end_to_end_session_seconds=(
                "end_to_end_session_seconds",
                "mean",
            ),
        )
    )
    call_summary = (
        calls.groupby(["imbalance_level", "strategy", "query_budget"], as_index=False)
        .agg(
            selector_calls=("query_round", "size"),
            total_fit_seconds=("fit_seconds", "sum"),
            total_score_seconds=("score_seconds", "sum"),
            total_selector_seconds=("selector_seconds", "sum"),
            oof_std_floor_events=("oof_std_floor_event_count", "sum"),
        )
    )
    integrity = pd.DataFrame(integrity_rows)
    tables = [
        (trajectory_summary, "revision_R4B_trajectory_summary.csv"),
        (participant_level, "revision_R4B_participant_level_summary.csv"),
        (able, "revision_R4B_able_bodied_descriptive_summary.csv"),
        (p07, "revision_R4B_p07_descriptive_summary.csv"),
        (coverage_summary, "revision_R4B_class_coverage_entropy_summary.csv"),
        (compute_summary, "revision_R4B_compute_summary.csv"),
        (call_summary, "revision_R4B_selector_compute_summary.csv"),
        (integrity, "revision_R4B_shard_integrity_manifest.csv"),
    ]
    for frame, basename in tables:
        atomic_csv(frame, FINAL_ROOT / basename)

    gates = {
        "revision_r4a_packet_hash_matches": engine.sha256_file(inputs["r4a_packet"])
        == REVISION_R4A_PACKET_SHA256,
        "revision_r4a_all_gates_passed": bool(
            inputs["r4a_report"].get("all_readiness_gates_passed")
        ),
        "revision_protocol_hash_matches": inputs["r4a_report"].get(
            "revision_protocol_sha256"
        )
        == REVISION_PROTOCOL_SHA256,
        "all_735_shards_pass_hash_crc_and_report_gates": len(integrity) == 735
        and bool(integrity["verified"].all()),
        "trajectory_count_is_13230": folds["trajectory_id"].nunique()
        == EXPECTED_TRAJECTORIES,
        "fold_count_is_66150": len(folds) == EXPECTED_FOLDS,
        "prediction_count_is_2315250": count_totals["prediction_count"]
        == EXPECTED_PREDICTIONS,
        "selection_count_is_926100": count_totals["selection_count"]
        == EXPECTED_SELECTIONS,
        "selector_call_count_is_132300": count_totals["selector_call_count"]
        == EXPECTED_SELECTOR_CALLS,
        "candidate_audit_count_is_2954700": count_totals[
            "candidate_audit_count"
        ]
        == EXPECTED_CANDIDATE_AUDIT_ROWS,
        "each_trajectory_has_five_sessions": bool(
            folds.groupby("trajectory_id")["target_session"].nunique().eq(5).all()
        ),
        "all_metrics_are_finite_and_in_range": bool(
            np.isfinite(
                folds[
                    [
                        "repetition_accuracy",
                        "repetition_balanced_accuracy",
                        "repetition_macro_f1",
                    ]
                ].to_numpy(float)
            ).all()
            and folds[
                [
                    "repetition_accuracy",
                    "repetition_balanced_accuracy",
                    "repetition_macro_f1",
                ]
            ].ge(0).all().all()
            and folds[
                [
                    "repetition_accuracy",
                    "repetition_balanced_accuracy",
                    "repetition_macro_f1",
                ]
            ].le(1).all().all()
        ),
        "balanced_accuracy_equals_accuracy_in_every_fold": bool(
            folds["balanced_accuracy_equals_accuracy"].all()
        ),
        "no_future_session_is_used": bool((~folds["future_session_used"]).all()),
        "fixed_test_never_enters_history": bool(
            (~folds["fixed_test_entered_history"]).all()
        ),
        "all_test_sets_are_balanced_five_per_class": bool(
            folds["test_labels_are_balanced_five_per_class"].all()
        ),
        "participant_level_summary_has_378_rows": len(participant_level) == 378,
        "every_participant_level_cell_has_35_pool_trajectories": bool(
            participant_level["pool_trajectory_replicates"].eq(35).all()
        ),
        "able_bodied_summary_has_54_rows": len(able) == 54,
        "each_able_bodied_summary_uses_six_participants": bool(
            able["participants"].eq(6).all()
        ),
        "p07_summary_has_54_rows_and_is_descriptive_only": len(p07) == 54
        and bool(p07["case_analysis"].all()),
        "class_recall_rows_are_complete": len(recalls) == EXPECTED_FOLDS * 7
        and bool(recalls["class_support"].eq(5).all()),
        "confusion_matrices_are_aggregated_without_loss": int(confusions["count"].sum())
        == EXPECTED_PREDICTIONS,
        "class_coverage_and_entropy_are_in_range": bool(
            coverage["selected_class_coverage"].between(1, 7).all()
            and coverage["selected_normalized_class_entropy"].between(0, 1).all()
        ),
        "all_compute_telemetry_is_finite_nonnegative": bool(
            np.isfinite(
                folds[
                    [
                        "query_fit_seconds",
                        "candidate_score_seconds",
                        "selector_seconds",
                        "final_refit_seconds",
                        "fixed_test_inference_seconds",
                        "end_to_end_session_seconds",
                    ]
                ].to_numpy(float)
            ).all()
            and folds[
                [
                    "query_fit_seconds",
                    "candidate_score_seconds",
                    "selector_seconds",
                    "final_refit_seconds",
                    "fixed_test_inference_seconds",
                    "end_to_end_session_seconds",
                ]
            ].ge(0).all().all()
        ),
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
        "shards": len(integrity),
        "trajectory_count": folds["trajectory_id"].nunique(),
        "fold_count": len(folds),
        **count_totals,
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "raw_hdf5_accessed": False,
        "new_statistical_test_run": False,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": (
            "PASS_TO_REVISION_R4C_RIDGE_RANDOM_IMBALANCE_SHARDS"
            if not failed
            else "REVISION_R4B_FINAL_AUDIT_FAILED"
        ),
    }
    atomic_json(report, FINAL_ROOT / "revision_R4B_final_report.json")
    shutil.copy2(Path(__file__), FINAL_ROOT / "revision_R4B_executed_source.py")
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
    atomic_csv(
        pd.DataFrame(manifest_rows), FINAL_ROOT / "revision_R4B_output_manifest.csv"
    )
    if failed:
        raise RuntimeError(f"Revision R4B final audit failed: {failed}")
    return report, able, p07


def create_final_packet():
    crc = engine.make_zip(
        FINAL_ROOT,
        FINAL_PACKET,
        "Revision_R4B_Ridge_Deterministic_Imbalance_Final",
    )
    digest = engine.sha256_file(FINAL_PACKET)
    verified = engine.roundtrip_remote_file(
        FINAL_PACKET, REMOTE_OUTPUT + "/" + FINAL_PACKET.name, digest
    )
    if not crc or not verified:
        raise RuntimeError("R4B final packet persistence failed")
    return digest


def main():
    print("=" * 104)
    print("REVISION R4B — RIDGE DETERMINISTIC IMBALANCE SHARDS")
    print("=" * 104)
    print("Execution device: CPU")
    print("GPU required: False")
    print("Expected shards:", EXPECTED_SHARDS)
    print("Expected trajectories:", EXPECTED_TRAJECTORIES)
    print("Expected folds:", EXPECTED_FOLDS)
    print("Maximum runtime this invocation (hours):", MAX_RUNTIME_HOURS)
    print("Checkpoint: every completed shard, SHA-256 Drive round trip")
    print("Resume: automatic from verified final-named shard checkpoints")
    print("New statistical tests: False")
    print()

    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified frozen inputs...")
    inputs = resolve_inputs()
    features, main_valid, metadata = r3b.prepare_metadata(
        inputs["stage5b_packet"], inputs["stage5d2_packet"]
    )
    manifest = inputs["shard_manifest"].loc[
        inputs["shard_manifest"]["stage"].astype(str).eq(STAGE)
    ].copy()
    manifest = manifest.sort_values(
        [
            "participant",
            "imbalance_level",
            "rotation_index",
            "pool_realization_index",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(manifest) != EXPECTED_SHARDS or not manifest["shard_id"].is_unique:
        raise RuntimeError("R4B shard manifest drift")
    expected_ids = set(manifest["shard_id"].astype(str))
    completed, duplicates = discover_completed_shards(expected_ids)
    print(f"Restored completed shard checkpoints: {len(completed)}/{EXPECTED_SHARDS}")
    if duplicates:
        print("Duplicate remote checkpoint records ignored:", len(duplicates))

    newly_completed = 0
    shard_durations = []
    for shard in manifest.itertuples(index=False):
        shard_id = str(shard.shard_id)
        if shard_id in completed:
            continue
        elapsed = time.time() - START_TIME
        typical = float(np.median(shard_durations[-10:])) if shard_durations else 60.0
        reserve = max(600.0, typical * 2.0)
        if EXPECTED_SHARDS - len(completed) <= 3:
            reserve = max(reserve, FINALIZATION_RESERVE_SECONDS)
        if elapsed + reserve >= MAX_RUNTIME_SECONDS:
            break
        shard_started = time.time()
        outputs = run_shard(
            shard,
            features,
            main_valid,
            metadata,
            inputs["pool_definitions"],
            inputs["pool_membership"],
            inputs["rbmal_seed"],
        )
        gates = validate_shard(shard, outputs)
        failed = [key for key, value in gates.items() if not bool(value)]
        if failed:
            raise RuntimeError(f"Shard {shard_id} failed gates: {failed}")
        shard_root, packet, digest = write_shard_packet(
            shard, outputs, gates, time.time() - shard_started
        )
        remote_path = upload_verified_shard(shard_id, packet, digest)
        completed[shard_id] = {
            "shard_id": shard_id,
            "sha256": digest,
            "remote_path": remote_path,
            "remote_basename": Path(remote_path).name,
        }
        newly_completed += 1
        duration = time.time() - shard_started
        shard_durations.append(duration)
        shutil.rmtree(shard_root, ignore_errors=True)
        packet.unlink(missing_ok=True)
        remaining_count = EXPECTED_SHARDS - len(completed)
        eta_hours = (
            remaining_count * float(np.median(shard_durations[-10:])) / 3600.0
        )
        print(
            f"SHARD PASS {len(completed):03d}/{EXPECTED_SHARDS} | {shard_id} | "
            f"{duration:.1f}s | ETA~{eta_hours:.2f}h",
            flush=True,
        )

    all_complete = len(completed) == EXPECTED_SHARDS
    if not all_complete:
        decision = "PARTIAL_PASS_RESUME_REVISION_R4B_SAME_NOTEBOOK"
        progress_hash = make_progress_packet(
            inputs, manifest, completed, duplicates, decision
        )
        print()
        print("Completed shards:", len(completed), "/", EXPECTED_SHARDS)
        print("New shards this invocation:", newly_completed)
        print("Remaining shards:", EXPECTED_SHARDS - len(completed))
        print("Progress packet SHA-256:", progress_hash)
        print("Progress packet Drive round-trip verified: True")
        print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 3))
        print()
        print("FINAL DECISION: PARTIAL_PASS_RESUME_REVISION_R4B_SAME_NOTEBOOK")
        return

    make_progress_packet(
        inputs,
        manifest,
        completed,
        duplicates,
        "ALL_SHARDS_COMPLETE_FINAL_AGGREGATION_STARTED",
    )
    print("All shards complete. Downloading and verifying every shard for final aggregation...")
    report, able, p07 = download_and_aggregate(
        inputs, manifest, completed
    )
    digest = create_final_packet()
    print()
    print("=" * 104)
    print("REVISION R4B — FINAL DESCRIPTIVE SUMMARY")
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
    print("FINAL DECISION: PASS_TO_REVISION_R4C_RIDGE_RANDOM_IMBALANCE_SHARDS")


if __name__ == "__main__":
    main()
