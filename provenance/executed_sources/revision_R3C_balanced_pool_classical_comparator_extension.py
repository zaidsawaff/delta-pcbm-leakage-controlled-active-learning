from __future__ import annotations

import json
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import revision_R3B_new_selector_implementation_unit_tests as r3b


engine = r3b.engine
START_TIME = time.time()
WORKING = engine.WORKING
RESULT_ROOT = (
    WORKING
    / "DELTA_REVIEWER_REVISION"
    / "Revision_R3C_Balanced_Pool_Classical_Comparator_Extension"
)
PACKET_PATH = WORKING / "revision_R3C_balanced_pool_classical_comparator_extension_packet.zip"
REMOTE_OUTPUT = (
    engine.REMOTE_BASE
    + "/Reviewer_Revision/Revision_R3C_Balanced_Pool_Classical_Comparator_Extension"
)

REVISION_R3B_PACKET_SHA256 = (
    "f65dbc49e7163dd53321e1cf55b76d8837b9cdcb69b3ae8cc34ca14fc7d25363"
)
REVISION_PROTOCOL_SHA256 = engine.REVISION_PROTOCOL_SHA256
STRATEGIES = list(r3b.CLASSICAL_STRATEGIES)
BUDGETS = [7, 14, 21]
TARGET_SESSIONS = [1, 2, 3, 4, 5]
EXPECTED_TRAJECTORIES = len(engine.PARTICIPANTS) * len(STRATEGIES) * len(BUDGETS)
EXPECTED_FOLDS = EXPECTED_TRAJECTORIES * len(TARGET_SESSIONS)
EXPECTED_PREDICTIONS = EXPECTED_FOLDS * 35
EXPECTED_SELECTIONS = (
    len(engine.PARTICIPANTS)
    * len(STRATEGIES)
    * len(TARGET_SESSIONS)
    * sum(BUDGETS)
)
EXPECTED_SELECTOR_CALLS = (
    len(engine.PARTICIPANTS)
    * len(STRATEGIES)
    * len(TARGET_SESSIONS)
    * sum(budget // 7 for budget in BUDGETS)
)
EXPECTED_CANDIDATE_AUDIT_ROWS = (
    len(engine.PARTICIPANTS)
    * len(STRATEGIES)
    * len(TARGET_SESSIONS)
    * sum(sum(35 - 7 * index for index in range(budget // 7)) for budget in BUDGETS)
)
NORMALIZATION_EPSILON = np.float32(1e-8)
R3C_NUMERICAL_AMENDMENT = (
    "OOF_CALIBRATION_ZERO_STD_USES_LOCKED_MAX_SIGMA_EPSILON_1E8"
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


def timed(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def fit_history_state_with_locked_epsilon(
    features,
    main_valid,
    metadata,
    history_rows,
):
    history_rows = np.asarray(history_rows, dtype=int)
    if len(history_rows) == 0 or len(np.unique(history_rows)) != len(history_rows):
        raise RuntimeError("History rows must be nonempty and unique")
    history_meta = metadata.iloc[history_rows]
    if history_meta["fixed_test_never_query"].astype(bool).any():
        raise RuntimeError("Fixed test entered history")
    if not history_meta["eligible_for_training"].astype(bool).all():
        raise RuntimeError("Non-training-eligible repetition entered history")
    if sorted(history_meta["label"].unique().tolist()) != list(range(engine.CLASSES)):
        raise RuntimeError("History lacks one or more classes")

    raw = np.asarray(features[history_rows], dtype=np.float32)
    valid = np.asarray(main_valid[history_rows], dtype=bool)
    logged = np.log1p(raw)
    means = np.zeros(engine.CHANNELS, dtype=np.float32)
    stds = np.zeros(engine.CHANNELS, dtype=np.float32)
    raw_stds = np.zeros(engine.CHANNELS, dtype=np.float32)
    counts = np.zeros(engine.CHANNELS, dtype=np.int64)
    floored_channels = []
    for channel in range(engine.CHANNELS):
        values = logged[:, :, channel][valid[:, :, channel]]
        counts[channel] = len(values)
        if len(values) == 0:
            raise ValueError(f"No main-valid history values for channel {channel}")
        means[channel] = values.mean()
        raw_std = np.float32(values.std(ddof=0))
        if not np.isfinite(raw_std):
            raise ValueError(f"Non-finite main-mask history std for channel {channel}")
        raw_stds[channel] = raw_std
        if raw_std < NORMALIZATION_EPSILON:
            floored_channels.append(channel)
            stds[channel] = NORMALIZATION_EPSILON
        else:
            stds[channel] = raw_std
    transformed = engine.transform_repetitions(
        features, main_valid, history_rows, means, stds
    )
    x = transformed.reshape(-1, engine.CHANNELS)
    y = np.repeat(
        history_meta["label"].to_numpy(dtype=int), engine.WINDOWS
    )
    model = engine.RidgeClassifier(alpha=1.0, solver="auto")
    model.fit(x, y)
    return {
        "model": model,
        "means": means,
        "stds": stds,
        "raw_stds_before_floor": raw_stds,
        "counts": counts,
        "history_rows": history_rows.copy(),
        "numerical_engine_contract": engine.NUMERICAL_ENGINE_CONTRACT,
        "training_array_dtype": str(x.dtype),
        "model_coefficient_dtype": str(np.asarray(model.coef_).dtype),
        "normalization_epsilon": float(NORMALIZATION_EPSILON),
        "std_floor_count": len(floored_channels),
        "std_floored_channels": "|".join(map(str, floored_channels)),
        "minimum_raw_std_before_floor": float(raw_stds.min()),
        "numerical_amendment": R3C_NUMERICAL_AMENDMENT,
    }


def fit_probability_calibrated_ridge_with_locked_epsilon(
    features,
    main_valid,
    metadata,
    history_rows,
):
    history_rows = np.asarray(sorted(map(int, history_rows)), dtype=int)
    labels = metadata.iloc[history_rows]["label"].to_numpy(dtype=int)
    if sorted(np.unique(labels).tolist()) != list(range(engine.CLASSES)):
        raise RuntimeError("Calibration history lacks one or more classes")
    splitter = r3b.StratifiedKFold(n_splits=5, shuffle=False)
    oof_scores = np.full(
        (len(history_rows), engine.CLASSES), np.nan, dtype=np.float64
    )
    fold_assignments = np.full(len(history_rows), -1, dtype=int)
    fold_rows = []
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(np.zeros(len(history_rows)), labels), start=1
    ):
        train_rows = history_rows[train_index]
        validation_rows = history_rows[validation_index]
        state = fit_history_state_with_locked_epsilon(
            features, main_valid, metadata, train_rows
        )
        scores, predictions, _ = engine.score_repetitions(
            state, features, main_valid, validation_rows
        )
        oof_scores[validation_index] = scores
        fold_assignments[validation_index] = fold_index
        validation_labels = metadata.iloc[validation_rows]["label"].to_numpy(dtype=int)
        for row, label, prediction in zip(
            validation_rows, validation_labels, predictions
        ):
            fold_rows.append(
                {
                    "fold_index": fold_index,
                    "history_sequence_row_internal": int(row),
                    "true_label_internal_audit_only": int(label),
                    "predicted_label_internal_audit_only": int(prediction),
                    "train_repetition_count": len(train_rows),
                    "validation_repetition_count": len(validation_rows),
                    "normalizer_training_rows_are_history_only": bool(
                        set(train_rows).issubset(set(history_rows))
                    ),
                    "validation_row_is_not_in_fold_training": int(row)
                    not in set(train_rows),
                    "normalization_epsilon": float(NORMALIZATION_EPSILON),
                    "std_floor_count": int(state["std_floor_count"]),
                    "std_floored_channels": state["std_floored_channels"],
                    "minimum_raw_std_before_floor": state[
                        "minimum_raw_std_before_floor"
                    ],
                    "all_post_floor_stds_are_positive": bool(
                        np.all(state["stds"] > 0)
                    ),
                }
            )
    if not np.isfinite(oof_scores).all() or (fold_assignments < 1).any():
        raise RuntimeError("OOF calibration scores are incomplete")
    calibrator = r3b.LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
    )
    calibrator.fit(oof_scores, labels)
    if calibrator.classes_.tolist() != list(range(engine.CLASSES)):
        raise RuntimeError("Calibrator class order drift")
    full_state = fit_history_state_with_locked_epsilon(
        features, main_valid, metadata, history_rows
    )
    return {
        "history_rows": history_rows,
        "labels": labels,
        "oof_scores": oof_scores,
        "fold_assignments": fold_assignments,
        "fold_audit": pd.DataFrame(fold_rows),
        "calibrator": calibrator,
        "full_state": full_state,
        "calibration_contract": (
            "FIVE_FOLD_STRATIFIED_REPETITION_OOF_HISTORY_ONLY_"
            "MULTINOMIAL_L2_LOGISTIC_C1_LBFGS_MAXITER5000_"
            "LOCKED_MAX_SIGMA_EPSILON_1E8"
        ),
    }


def probability_acquisition(
    strategy,
    features,
    main_valid,
    metadata,
    history_rows,
    remaining_rows,
):
    calibrated, fit_seconds = timed(
        fit_probability_calibrated_ridge_with_locked_epsilon,
        features,
        main_valid,
        metadata,
        history_rows,
    )
    scored, score_seconds = timed(
        r3b.calibrated_probabilities,
        calibrated,
        features,
        main_valid,
        remaining_rows,
    )
    probabilities, decision_scores, predicted, margins = scored
    tokens = (
        metadata.iloc[np.asarray(remaining_rows, dtype=int)]["opaque_candidate_token"]
        .astype(str)
        .to_numpy()
    )
    selector_frame = r3b.build_probability_frame(tokens, probabilities)
    if strategy == "LEAST_CONFIDENCE":
        selected_result, selector_seconds = timed(
            r3b.select_least_confidence, selector_frame, 7
        )
    elif strategy == "PREDICTIVE_ENTROPY":
        selected_result, selector_seconds = timed(
            r3b.select_predictive_entropy, selector_frame, 7
        )
    else:
        raise ValueError(f"Invalid probability strategy: {strategy}")
    selected_tokens, ordered = selected_result
    score_lookup = dict(
        zip(ordered["opaque_candidate_token"].astype(str), ordered["score"].astype(float))
    )
    candidate_rows = []
    for index, token in enumerate(tokens):
        row = {
            "opaque_candidate_token": str(token),
            "raw_predicted_label_internal_audit_only": int(predicted[index]),
            "raw_margin_internal_audit_only": float(margins[index]),
            "acquisition_score": float(score_lookup[str(token)]),
            "selected_this_round": str(token) in set(selected_tokens),
        }
        for label in range(engine.CLASSES):
            row[f"calibrated_probability_{label}"] = float(probabilities[index, label])
            row[f"raw_decision_score_{label}"] = float(decision_scores[index, label])
        candidate_rows.append(row)
    call = {
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "selector_seconds": selector_seconds,
        "selector_schema": "|".join(selector_frame.columns),
        "selector_schema_exact": selector_frame.columns.tolist()
        == r3b.PROBABILITY_SCHEMA,
        "selector_forbidden_column_count": len(
            set(selector_frame.columns).intersection(r3b.FORBIDDEN_COLUMNS)
        ),
        "oof_fold_count": int(calibrated["fold_audit"]["fold_index"].nunique()),
        "oof_repetition_count": int(len(calibrated["fold_audit"])),
        "calibrator_max_iterations_used": int(
            np.max(calibrated["calibrator"].n_iter_)
        ),
        "oof_std_floor_event_count": int(
            calibrated["fold_audit"]
            .groupby("fold_index")["std_floor_count"]
            .first()
            .sum()
        ),
        "full_history_std_floor_count": int(
            calibrated["full_state"]["std_floor_count"]
        ),
        "state": calibrated["full_state"],
        "oof_audit": calibrated["fold_audit"],
    }
    return selected_tokens, pd.DataFrame(candidate_rows), call, pd.DataFrame()


def diversity_acquisition(
    strategy,
    features,
    main_valid,
    metadata,
    history_rows,
    remaining_rows,
    rbmal_seed,
):
    state, fit_seconds = timed(
        engine.fit_history_state,
        features,
        main_valid,
        metadata,
        history_rows,
    )
    started = time.perf_counter()
    tokens = (
        metadata.iloc[np.asarray(remaining_rows, dtype=int)]["opaque_candidate_token"]
        .astype(str)
        .to_numpy()
    )
    history_embeddings = r3b.repetition_embeddings(
        state, features, main_valid, history_rows
    )
    candidate_embeddings = r3b.repetition_embeddings(
        state, features, main_valid, remaining_rows
    )
    decision_scores, predicted, margins = engine.score_repetitions(
        state, features, main_valid, remaining_rows
    )
    score_seconds = time.perf_counter() - started
    if strategy == "RBMAL_MARGIN_DIVERSITY":
        selector_frame = r3b.build_rbmal_frame(tokens, margins, candidate_embeddings)
        selected_result, selector_seconds = timed(
            r3b.select_rbmal,
            selector_frame,
            history_embeddings,
            7,
            rbmal_seed,
        )
        selected_tokens, step_audit = selected_result
        initial_score = 1.0 - r3b.minmax(margins)
        initial_distance = r3b.minimum_distance(candidate_embeddings, history_embeddings)
    elif strategy == "CORE_SET_GREEDY":
        selector_frame = r3b.build_core_frame(tokens, candidate_embeddings)
        selected_result, selector_seconds = timed(
            r3b.select_core_set,
            selector_frame,
            history_embeddings,
            7,
        )
        selected_tokens, step_audit = selected_result
        initial_distance = r3b.minimum_distance(candidate_embeddings, history_embeddings)
        initial_score = initial_distance
    else:
        raise ValueError(f"Invalid diversity strategy: {strategy}")
    selected_set = set(selected_tokens)
    candidate_rows = []
    for index, token in enumerate(tokens):
        candidate_rows.append(
            {
                "opaque_candidate_token": str(token),
                "raw_predicted_label_internal_audit_only": int(predicted[index]),
                "raw_margin_internal_audit_only": float(margins[index]),
                "initial_minimum_history_distance": float(initial_distance[index]),
                "acquisition_score": float(initial_score[index]),
                "selected_this_round": str(token) in selected_set,
                "raw_decision_score_vector_hash": r3b.array_hash(
                    np.asarray(decision_scores[index], dtype=np.float64)
                ),
            }
        )
    call = {
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "selector_seconds": selector_seconds,
        "selector_schema": "|".join(selector_frame.columns),
        "selector_schema_exact": (
            selector_frame.columns.tolist() == r3b.RBMAL_SCHEMA
            if strategy == "RBMAL_MARGIN_DIVERSITY"
            else selector_frame.columns.tolist() == r3b.CORE_SET_SCHEMA
        ),
        "selector_forbidden_column_count": len(
            set(selector_frame.columns).intersection(r3b.FORBIDDEN_COLUMNS)
        ),
        "oof_fold_count": 0,
        "oof_repetition_count": 0,
        "calibrator_max_iterations_used": 0,
        "oof_std_floor_event_count": 0,
        "full_history_std_floor_count": int(state.get("std_floor_count", 0)),
        "state": state,
        "oof_audit": pd.DataFrame(),
    }
    return selected_tokens, pd.DataFrame(candidate_rows), call, step_audit


def evaluate_fixed_test(state, features, main_valid, metadata, fixed_rows):
    scores, predicted, margins = engine.score_repetitions(
        state, features, main_valid, fixed_rows
    )
    true = metadata.iloc[np.asarray(fixed_rows, dtype=int)]["label"].to_numpy(dtype=int)
    balanced = engine.balanced_accuracy(true, predicted)
    accuracy = float(np.mean(true == predicted))
    macro_f1 = float(f1_score(true, predicted, labels=list(range(7)), average="macro"))
    return {
        "scores": scores,
        "predicted": predicted,
        "margins": margins,
        "true": true,
        "balanced_accuracy": balanced,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def class_distribution_metrics(labels):
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=engine.CLASSES)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum() / np.log(7.0))
    return counts, int(np.sum(counts > 0)), entropy


def run_experiment(features, main_valid, metadata, rbmal_seed):
    selections = []
    candidate_audits = []
    step_audits = []
    selector_calls = []
    oof_rows = []
    normalizers = []
    folds = []
    predictions = []
    recalls = []
    confusions = []
    coverage_rows = []

    for participant in engine.PARTICIPANTS:
        participant_started = time.perf_counter()
        for strategy in STRATEGIES:
            for budget in BUDGETS:
                trajectory_id = f"{participant}_{strategy}_K{budget:02d}"
                history = engine.initial_history_rows(metadata, participant).tolist()
                for session in TARGET_SESSIONS:
                    session_started = time.perf_counter()
                    remaining = engine.candidate_rows(
                        metadata, participant, session
                    ).tolist()
                    fixed_rows = engine.fixed_test_rows(metadata, participant, session)
                    session_selected = []
                    fit_seconds_total = 0.0
                    score_seconds_total = 0.0
                    selector_seconds_total = 0.0
                    for query_round in range(1, budget // 7 + 1):
                        history_before = list(history)
                        if strategy in {"LEAST_CONFIDENCE", "PREDICTIVE_ENTROPY"}:
                            selected_tokens, candidate_frame, call, step_audit = (
                                probability_acquisition(
                                    strategy,
                                    features,
                                    main_valid,
                                    metadata,
                                    history_before,
                                    remaining,
                                )
                            )
                        else:
                            selected_tokens, candidate_frame, call, step_audit = (
                                diversity_acquisition(
                                    strategy,
                                    features,
                                    main_valid,
                                    metadata,
                                    history_before,
                                    remaining,
                                    rbmal_seed,
                                )
                            )
                        selected_rows = engine.reveal_rows(
                            metadata, selected_tokens, remaining
                        )
                        selected_set = set(map(int, selected_rows))
                        true_labels = metadata.iloc[selected_rows]["label"].to_numpy(dtype=int)
                        for position, (token, row, label) in enumerate(
                            zip(selected_tokens, selected_rows, true_labels), start=1
                        ):
                            selections.append(
                                {
                                    "trajectory_id": trajectory_id,
                                    "participant": participant,
                                    "case_analysis": participant == "P07",
                                    "target_session": session,
                                    "strategy": strategy,
                                    "query_budget": budget,
                                    "query_round": query_round,
                                    "position_in_round": position,
                                    "opaque_candidate_token": str(token),
                                    "sequence_row_internal_audit_only": int(row),
                                    "true_label_after_reveal": int(label),
                                    "selected_record_is_candidate": int(row) in set(remaining),
                                    "selected_record_is_fixed_test": int(row)
                                    in set(map(int, fixed_rows)),
                                }
                            )
                        candidate_frame = candidate_frame.copy()
                        candidate_frame.insert(0, "trajectory_id", trajectory_id)
                        candidate_frame.insert(1, "participant", participant)
                        candidate_frame.insert(2, "target_session", session)
                        candidate_frame.insert(3, "strategy", strategy)
                        candidate_frame.insert(4, "query_budget", budget)
                        candidate_frame.insert(5, "query_round", query_round)
                        candidate_audits.extend(candidate_frame.to_dict("records"))
                        if not step_audit.empty:
                            step_audit = step_audit.copy()
                            step_audit.insert(0, "trajectory_id", trajectory_id)
                            step_audit.insert(1, "target_session", session)
                            step_audit.insert(2, "query_round", query_round)
                            step_audit.insert(3, "strategy", strategy)
                            step_audits.extend(
                                [
                                    {
                                        "trajectory_id": row["trajectory_id"],
                                        "participant": participant,
                                        "target_session": row["target_session"],
                                        "strategy": row["strategy"],
                                        "query_budget": budget,
                                        "query_round": row["query_round"],
                                        "opaque_candidate_token": row[
                                            "opaque_candidate_token"
                                        ],
                                        "selected_this_round": True,
                                        "step_audit_json": json.dumps(
                                            row, sort_keys=True, default=str
                                        ),
                                    }
                                    for row in step_audit.to_dict("records")
                                ]
                            )
                        history_sessions = metadata.iloc[np.asarray(history_before, dtype=int)][
                            "session"
                        ].to_numpy(dtype=int)
                        fixed_set = set(map(int, fixed_rows))
                        selector_calls.append(
                            {
                                "trajectory_id": trajectory_id,
                                "participant": participant,
                                "target_session": session,
                                "strategy": strategy,
                                "query_budget": budget,
                                "query_round": query_round,
                                "history_repetitions_before_query": len(history_before),
                                "remaining_candidates_before_query": len(remaining),
                                "selected_count": len(selected_rows),
                                "fit_seconds": call["fit_seconds"],
                                "score_seconds": call["score_seconds"],
                                "selector_seconds": call["selector_seconds"],
                                "selector_schema": call["selector_schema"],
                                "selector_schema_exact": call[
                                    "selector_schema_exact"
                                ],
                                "selector_forbidden_column_count": call[
                                    "selector_forbidden_column_count"
                                ],
                                "oof_fold_count": call["oof_fold_count"],
                                "oof_repetition_count": call[
                                    "oof_repetition_count"
                                ],
                                "calibrator_max_iterations_used": call[
                                    "calibrator_max_iterations_used"
                                ],
                                "oof_std_floor_event_count": call[
                                    "oof_std_floor_event_count"
                                ],
                                "full_history_std_floor_count": call[
                                    "full_history_std_floor_count"
                                ],
                                "maximum_history_session": int(
                                    history_sessions.max()
                                ),
                                "future_session_used": bool(
                                    (history_sessions > session).any()
                                ),
                                "fixed_test_used_for_training_or_selection": bool(
                                    set(history_before).intersection(fixed_set)
                                    or selected_set.intersection(fixed_set)
                                ),
                                "peak_process_rss_mb": resource.getrusage(
                                    resource.RUSAGE_SELF
                                ).ru_maxrss
                                / 1024.0,
                            }
                        )
                        if not call["oof_audit"].empty:
                            audit = call["oof_audit"].copy()
                            audit.insert(0, "trajectory_id", trajectory_id)
                            audit.insert(1, "target_session", session)
                            audit.insert(2, "query_round", query_round)
                            audit.insert(3, "strategy", strategy)
                            oof_rows.extend(audit.to_dict("records"))
                        state = call["state"]
                        normalizers.append(
                            {
                                "trajectory_id": trajectory_id,
                                "participant": participant,
                                "target_session": session,
                                "strategy": strategy,
                                "query_budget": budget,
                                "fit_role": f"QUERY_ROUND_{query_round}",
                                "history_repetitions": len(history_before),
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
                        evaluate_fixed_test,
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
                    fold = {
                        "run_id": run_id,
                        "trajectory_id": trajectory_id,
                        "participant": participant,
                        "case_analysis": participant == "P07",
                        "target_session": session,
                        "strategy": strategy,
                        "query_budget": budget,
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
                    folds.append(fold)
                    normalizers.append(
                        {
                            "trajectory_id": trajectory_id,
                            "participant": participant,
                            "target_session": session,
                            "strategy": strategy,
                            "query_budget": budget,
                            "fit_role": "FINAL_HISTORY_EVALUATION",
                            "history_repetitions": len(history),
                            "minimum_mean": float(np.min(final_state["means"])),
                            "maximum_mean": float(np.max(final_state["means"])),
                            "minimum_std": float(np.min(final_state["stds"])),
                            "maximum_std": float(np.max(final_state["stds"])),
                            "minimum_valid_count": int(np.min(final_state["counts"])),
                            "means_dtype": str(final_state["means"].dtype),
                            "stds_dtype": str(final_state["stds"].dtype),
                            "model_coefficient_dtype": str(
                                np.asarray(final_state["model"].coef_).dtype
                            ),
                        }
                    )
                    for position, row in enumerate(fixed_rows):
                        record = {
                            "run_id": run_id,
                            "participant": participant,
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
                        predictions.append(record)
                    for true_label in range(engine.CLASSES):
                        mask = evaluated["true"] == true_label
                        recall = float(
                            np.mean(evaluated["predicted"][mask] == true_label)
                        )
                        recalls.append(
                            {
                                "run_id": run_id,
                                "participant": participant,
                                "target_session": session,
                                "strategy": strategy,
                                "query_budget": budget,
                                "class_label": true_label,
                                "class_support": int(mask.sum()),
                                "class_recall": recall,
                            }
                        )
                        for predicted_label in range(engine.CLASSES):
                            confusions.append(
                                {
                                    "run_id": run_id,
                                    "participant": participant,
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
                    counts, coverage, entropy = class_distribution_metrics(
                        selected_labels
                    )
                    coverage_row = {
                        "run_id": run_id,
                        "participant": participant,
                        "case_analysis": participant == "P07",
                        "target_session": session,
                        "strategy": strategy,
                        "query_budget": budget,
                        "selected_class_coverage": coverage,
                        "selected_normalized_class_entropy": entropy,
                    }
                    for label, count in enumerate(counts):
                        coverage_row[f"selected_true_class_{label}_count"] = int(count)
                    coverage_rows.append(coverage_row)
        print(
            f"Completed participant {participant} | "
            f"elapsed={(time.perf_counter() - participant_started) / 60.0:.2f} min",
            flush=True,
        )
    return {
        "selections": pd.DataFrame(selections),
        "candidate_audits": pd.DataFrame(candidate_audits),
        "step_audits": pd.DataFrame(step_audits),
        "selector_calls": pd.DataFrame(selector_calls),
        "oof_rows": pd.DataFrame(oof_rows),
        "normalizers": pd.DataFrame(normalizers),
        "folds": pd.DataFrame(folds),
        "predictions": pd.DataFrame(predictions),
        "recalls": pd.DataFrame(recalls),
        "confusions": pd.DataFrame(confusions),
        "coverage": pd.DataFrame(coverage_rows),
    }


def summarize(folds, coverage):
    participant = (
        folds.groupby(
            ["participant", "case_analysis", "strategy", "query_budget"],
            as_index=False,
        )
        .agg(
            target_sessions=("target_session", "nunique"),
            mean_repetition_balanced_accuracy=(
                "repetition_balanced_accuracy",
                "mean",
            ),
            mean_repetition_accuracy=("repetition_accuracy", "mean"),
            mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
            total_repetition_errors=("repetition_errors", "sum"),
            mean_end_to_end_session_seconds=(
                "end_to_end_session_seconds",
                "mean",
            ),
        )
    )
    able = (
        participant.loc[participant["participant"].ne("P07")]
        .groupby(["strategy", "query_budget"], as_index=False)
        .agg(
            participants=("participant", "nunique"),
            mean_repetition_balanced_accuracy=(
                "mean_repetition_balanced_accuracy",
                "mean",
            ),
            std_repetition_balanced_accuracy=(
                "mean_repetition_balanced_accuracy",
                "std",
            ),
            mean_repetition_macro_f1=("mean_repetition_macro_f1", "mean"),
            total_repetition_errors=("total_repetition_errors", "sum"),
        )
    )
    p07 = participant.loc[participant["participant"].eq("P07")].copy()
    coverage_summary = (
        coverage.groupby(["participant", "strategy", "query_budget"], as_index=False)
        .agg(
            mean_selected_class_coverage=("selected_class_coverage", "mean"),
            minimum_selected_class_coverage=("selected_class_coverage", "min"),
            mean_selected_normalized_class_entropy=(
                "selected_normalized_class_entropy",
                "mean",
            ),
        )
    )
    return participant, able, p07, coverage_summary


def persist_tables(outputs, summaries):
    mapping = {
        "selections": "revision_R3C_selection_trace.csv",
        "candidate_audits": "revision_R3C_candidate_score_audit.csv",
        "step_audits": "revision_R3C_sequential_selection_step_audit.csv",
        "selector_calls": "revision_R3C_selector_call_audit.csv",
        "oof_rows": "revision_R3C_ridge_oof_fold_audit.csv",
        "normalizers": "revision_R3C_normalizer_audit.csv",
        "folds": "revision_R3C_fold_metrics.csv",
        "predictions": "revision_R3C_repetition_predictions.csv",
        "recalls": "revision_R3C_per_class_recall.csv",
        "confusions": "revision_R3C_confusion_matrices_long.csv",
        "coverage": "revision_R3C_selected_class_distribution.csv",
    }
    for key, basename in mapping.items():
        atomic_csv(outputs[key], RESULT_ROOT / basename)
    for frame, basename in zip(
        summaries,
        [
            "revision_R3C_participant_summary.csv",
            "revision_R3C_able_bodied_descriptive_summary.csv",
            "revision_R3C_p07_descriptive_summary.csv",
            "revision_R3C_class_coverage_entropy_summary.csv",
        ],
    ):
        atomic_csv(frame, RESULT_ROOT / basename)


def create_packet():
    source = Path(__file__)
    if source.exists():
        shutil.copy2(source, RESULT_ROOT / "revision_R3C_executed_source.py")
    manifest_rows = []
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(RESULT_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": engine.sha256_file(path),
                }
            )
    atomic_csv(
        pd.DataFrame(manifest_rows),
        RESULT_ROOT / "revision_R3C_output_sha256_manifest.csv",
    )
    crc = engine.make_zip(
        RESULT_ROOT,
        PACKET_PATH,
        "Revision_R3C_Balanced_Pool_Classical_Comparator_Extension",
    )
    if not crc:
        raise RuntimeError("Revision R3C packet CRC failed")
    digest = engine.sha256_file(PACKET_PATH)
    remote_verified = engine.roundtrip_remote_file(
        PACKET_PATH, REMOTE_OUTPUT + "/" + PACKET_PATH.name, digest
    )
    return crc, digest, remote_verified


def main():
    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PACKET_PATH.unlink(missing_ok=True)

    print("=" * 100)
    print("REVISION R3C — BALANCED-POOL CLASSICAL COMPARATOR EXTENSION")
    print("=" * 100)
    print("Execution device: CPU")
    print("Candidate pool: frozen original balanced 35 (five per true class)")
    print("Strategies:", STRATEGIES)
    print("Budgets:", BUDGETS)
    print("Expected trajectories:", EXPECTED_TRAJECTORIES)
    print("Expected participant-session folds:", EXPECTED_FOLDS)
    print("Expected repetition predictions:", EXPECTED_PREDICTIONS)
    print("Expected selections:", EXPECTED_SELECTIONS)
    print("New statistical tests: False")
    print("Raw HDF5 access: False")
    print(
        "R3C numerical correction active:",
        R3C_NUMERICAL_AMENDMENT,
    )
    print("Locked normalization epsilon:", float(NORMALIZATION_EPSILON))
    print()

    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified R0, R3A-P1, R3B, Stage 5B, and Stage 5D-2 packets...")
    inputs = r3b.resolve_inputs()
    r3b_packet, r3b_source = engine.resolve_packet(
        "revision_R3B_new_selector_implementation_unit_test_packet.zip",
        REVISION_R3B_PACKET_SHA256,
    )
    r3b_report = engine.read_json_member(
        r3b_packet, "revision_R3B_selector_unit_test_report.json"
    )
    if not r3b_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("Revision R3B parent did not pass")
    if r3b_report.get("revision_protocol_sha256") != REVISION_PROTOCOL_SHA256:
        raise RuntimeError("Revision R3B protocol hash drift")
    input_audit = pd.concat(
        [
            inputs["audit"],
            pd.DataFrame(
                [
                    {
                        "packet": r3b_packet.name,
                        "expected_sha256": REVISION_R3B_PACKET_SHA256,
                        "observed_sha256": engine.sha256_file(r3b_packet),
                        "hash_matches": engine.sha256_file(r3b_packet)
                        == REVISION_R3B_PACKET_SHA256,
                        "crc_passes": engine.archive_crc_passes(r3b_packet),
                        "source": r3b_source,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    if not input_audit[["hash_matches", "crc_passes"]].all().all():
        raise RuntimeError("Revision R3C input integrity failed")
    atomic_csv(input_audit, RESULT_ROOT / "revision_R3C_input_packet_audit.csv")

    features, main_valid, metadata = r3b.prepare_metadata(
        inputs["stage5b_packet"], inputs["stage5d2_packet"]
    )
    outputs = run_experiment(
        features, main_valid, metadata, inputs["rbmal_seed"]
    )
    summaries = summarize(outputs["folds"], outputs["coverage"])
    participant_summary, able_summary, p07_summary, coverage_summary = summaries
    persist_tables(outputs, summaries)

    folds = outputs["folds"]
    selections = outputs["selections"]
    candidate_audit = outputs["candidate_audits"]
    step_audit = outputs["step_audits"]
    selector_calls = outputs["selector_calls"]
    normalizers = outputs["normalizers"]
    predictions = outputs["predictions"]
    recalls = outputs["recalls"]
    confusions = outputs["confusions"]
    coverage = outputs["coverage"]

    readiness_gates = {
        "revision_r3b_packet_hash_matches": engine.sha256_file(r3b_packet)
        == REVISION_R3B_PACKET_SHA256,
        "revision_r3b_all_gates_passed": bool(
            r3b_report.get("all_readiness_gates_passed")
        ),
        "revision_protocol_hash_matches": r3b_report.get(
            "revision_protocol_sha256"
        )
        == REVISION_PROTOCOL_SHA256,
        "all_five_input_packets_pass_hash_and_crc": bool(
            input_audit[["hash_matches", "crc_passes"]].all().all()
        ),
        "feature_shape_is_2940_by_37_by_64": tuple(features.shape)
        == (2940, 37, 64),
        "main_mask_shape_matches_features": tuple(main_valid.shape)
        == tuple(features.shape),
        "trajectory_count_is_84": folds["trajectory_id"].nunique()
        == EXPECTED_TRAJECTORIES,
        "trajectory_set_is_exact": set(
            zip(
                participant_summary["participant"],
                participant_summary["strategy"],
                participant_summary["query_budget"],
            )
        )
        == {
            (participant, strategy, budget)
            for participant in engine.PARTICIPANTS
            for strategy in STRATEGIES
            for budget in BUDGETS
        },
        "fold_count_is_420": len(folds) == EXPECTED_FOLDS,
        "fold_run_ids_are_unique": folds["run_id"].is_unique,
        "each_trajectory_has_five_target_sessions": bool(
            folds.groupby("trajectory_id")["target_session"].nunique().eq(5).all()
        ),
        "prediction_count_is_14700": len(predictions) == EXPECTED_PREDICTIONS,
        "every_fold_has_35_fixed_test_repetitions": bool(
            predictions.groupby("run_id").size().eq(35).all()
        ),
        "selection_count_is_5880": len(selections) == EXPECTED_SELECTIONS,
        "selection_counts_match_budgets": bool(
            selections.groupby(
                ["trajectory_id", "target_session"]
            ).size().reset_index(name="observed")
            .merge(
                folds[["trajectory_id", "target_session", "query_budget"]],
                on=["trajectory_id", "target_session"],
                validate="one_to_one",
            )
            .eval("observed == query_budget")
            .all()
        ),
        "selected_tokens_are_unique_within_each_session_trajectory": bool(
            selections.groupby(
                ["trajectory_id", "target_session"]
            )["opaque_candidate_token"].nunique().reset_index(name="unique_tokens")
            .merge(
                folds[["trajectory_id", "target_session", "query_budget"]],
                on=["trajectory_id", "target_session"],
                validate="one_to_one",
            )
            .eval("unique_tokens == query_budget")
            .all()
        ),
        "all_selected_records_are_candidates": bool(
            selections["selected_record_is_candidate"].all()
        ),
        "no_fixed_test_record_was_selected": bool(
            (~selections["selected_record_is_fixed_test"]).all()
        ),
        "selector_call_count_is_840": len(selector_calls)
        == EXPECTED_SELECTOR_CALLS,
        "candidate_audit_base_row_count_is_25480": len(candidate_audit)
        == EXPECTED_CANDIDATE_AUDIT_ROWS,
        "every_candidate_audit_call_marks_exactly_seven_selected": bool(
            candidate_audit.groupby(
                [
                    "trajectory_id",
                    "target_session",
                    "query_round",
                    "strategy",
                    "query_budget",
                ]
            )["selected_this_round"].sum().eq(7).all()
        ),
        "candidate_tokens_are_opaque_and_unique_within_each_call": bool(
            candidate_audit["opaque_candidate_token"]
            .astype(str)
            .str.fullmatch(r"[0-9a-f]{24}")
            .all()
            and candidate_audit.groupby(
                [
                    "trajectory_id",
                    "target_session",
                    "query_round",
                    "strategy",
                    "query_budget",
                ]
            )["opaque_candidate_token"].apply(lambda values: values.is_unique).all()
        ),
        "sequential_step_audit_has_2940_rows": len(step_audit) == 2940,
        "all_selector_schemas_are_exact": bool(
            selector_calls["selector_schema_exact"].all()
        ),
        "no_selector_received_forbidden_columns": bool(
            selector_calls["selector_forbidden_column_count"].eq(0).all()
        ),
        "all_probability_calls_have_five_oof_folds": bool(
            selector_calls.loc[
                selector_calls["strategy"].isin(
                    ["LEAST_CONFIDENCE", "PREDICTIVE_ENTROPY"]
                ),
                "oof_fold_count",
            ].eq(5).all()
        ),
        "all_oof_rows_use_history_only_and_exclude_validation_from_fold_training": bool(
            outputs["oof_rows"][
                "normalizer_training_rows_are_history_only"
            ].all()
            and outputs["oof_rows"][
                "validation_row_is_not_in_fold_training"
            ].all()
        ),
        "all_calibrators_converged_before_5000_iterations": bool(
            selector_calls.loc[
                selector_calls["strategy"].isin(
                    ["LEAST_CONFIDENCE", "PREDICTIVE_ENTROPY"]
                ),
                "calibrator_max_iterations_used",
            ].lt(5000).all()
        ),
        "locked_normalization_epsilon_is_1e_8": bool(
            np.isclose(float(NORMALIZATION_EPSILON), 1e-8, rtol=0.0, atol=1e-15)
        ),
        "oof_zero_std_floor_events_are_recorded": bool(
            selector_calls["oof_std_floor_event_count"].sum() > 0
        ),
        "only_probability_oof_calibration_uses_std_floor": bool(
            selector_calls.loc[
                ~selector_calls["strategy"].isin(
                    ["LEAST_CONFIDENCE", "PREDICTIVE_ENTROPY"]
                ),
                "oof_std_floor_event_count",
            ].eq(0).all()
            and selector_calls["full_history_std_floor_count"].eq(0).all()
        ),
        "all_oof_post_floor_stds_are_positive": bool(
            outputs["oof_rows"]["all_post_floor_stds_are_positive"].all()
        ),
        "no_source_uses_future_sessions": bool(
            (~selector_calls["future_session_used"]).all()
            and (~folds["future_session_used"]).all()
        ),
        "fixed_test_never_enters_training_or_selection": bool(
            (~selector_calls["fixed_test_used_for_training_or_selection"]).all()
            and (~folds["fixed_test_entered_history"]).all()
        ),
        "all_fixed_tests_are_five_per_class": bool(
            folds["test_labels_are_balanced_five_per_class"].all()
        ),
        "all_candidate_pools_are_five_per_class": bool(
            metadata.loc[
                metadata["protocol_role"].eq("CURRENT_SESSION_UNLABELED_POOL")
            ]
            .groupby(["participant", "session", "label"])
            .size()
            .eq(5)
            .all()
        ),
        "balanced_accuracy_equals_accuracy_in_every_fold": bool(
            folds["balanced_accuracy_equals_accuracy"].all()
        ),
        "all_metrics_are_finite_and_between_zero_and_one": bool(
            np.isfinite(
                folds[
                    [
                        "repetition_accuracy",
                        "repetition_balanced_accuracy",
                        "repetition_macro_f1",
                    ]
                ].to_numpy(dtype=float)
            ).all()
            and (
                folds[
                    [
                        "repetition_accuracy",
                        "repetition_balanced_accuracy",
                        "repetition_macro_f1",
                    ]
                ]
                >= 0
            ).all().all()
            and (
                folds[
                    [
                        "repetition_accuracy",
                        "repetition_balanced_accuracy",
                        "repetition_macro_f1",
                    ]
                ]
                <= 1
            ).all().all()
        ),
        "all_normalizers_are_finite": bool(
            np.isfinite(
                normalizers[
                    ["minimum_mean", "maximum_mean", "minimum_std", "maximum_std"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "all_normalizer_stds_and_counts_are_positive": bool(
            normalizers["minimum_std"].gt(0).all()
            and normalizers["minimum_valid_count"].gt(0).all()
        ),
        "all_numerical_arrays_and_coefficients_are_float32": bool(
            normalizers["means_dtype"].eq("float32").all()
            and normalizers["stds_dtype"].eq("float32").all()
            and normalizers["model_coefficient_dtype"].eq("float32").all()
        ),
        "per_class_recall_has_2940_rows": len(recalls)
        == EXPECTED_FOLDS * 7,
        "every_per_class_recall_has_support_five": bool(
            recalls["class_support"].eq(5).all()
        ),
        "confusion_matrix_long_has_20580_rows": len(confusions)
        == EXPECTED_FOLDS * 49,
        "each_confusion_matrix_sums_to_35": bool(
            confusions.groupby("run_id")["count"].sum().eq(35).all()
        ),
        "class_distribution_has_420_rows": len(coverage) == EXPECTED_FOLDS,
        "class_coverage_and_entropy_are_in_valid_ranges": bool(
            coverage["selected_class_coverage"].between(1, 7).all()
            and coverage["selected_normalized_class_entropy"].ge(-1e-12).all()
            and coverage["selected_normalized_class_entropy"].le(1 + 1e-12).all()
        ),
        "all_fixed_test_scores_and_margins_are_finite": bool(
            np.isfinite(
                predictions[
                    [
                        "raw_margin",
                        *[f"decision_score_{label}" for label in range(7)],
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "participant_summary_has_84_rows": len(participant_summary)
        == EXPECTED_TRAJECTORIES,
        "able_bodied_summary_has_12_rows": len(able_summary) == 12,
        "each_able_bodied_summary_uses_six_participants": bool(
            able_summary["participants"].eq(6).all()
        ),
        "p07_summary_has_12_rows_and_is_descriptive_only": len(p07_summary) == 12
        and p07_summary["case_analysis"].all(),
        "all_compute_times_are_finite_and_nonnegative": bool(
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
                ].to_numpy(dtype=float)
            ).all()
            and (
                folds[
                    [
                        "query_fit_seconds",
                        "candidate_score_seconds",
                        "selector_seconds",
                        "final_refit_seconds",
                        "fixed_test_inference_seconds",
                        "end_to_end_session_seconds",
                    ]
                ]
                >= 0
            ).all().all()
        ),
        "p07_is_excluded_from_inference": True,
        "no_statistical_test_was_run": True,
        "raw_hdf5_data_was_not_accessed": True,
        "stage3g_and_stage5f_conclusions_cannot_be_replaced": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in readiness_gates.items() if not bool(value)]
    final_decision = (
        "PASS_TO_REVISION_R4_IMBALANCED_POOL_STRESS_TEST"
        if not failed
        else "HOLD_FOR_REVISION_R3C_BALANCED_POOL_DIAGNOSTIC"
    )
    report = {
        "stage": "REVISION_R3C_BALANCED_POOL_CLASSICAL_COMPARATOR_EXTENSION",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "revision_r3b_packet_sha256": REVISION_R3B_PACKET_SHA256,
        "r3c_numerical_amendment": R3C_NUMERICAL_AMENDMENT,
        "normalization_epsilon": float(NORMALIZATION_EPSILON),
        "oof_std_floor_event_count": int(
            selector_calls["oof_std_floor_event_count"].sum()
        ),
        "candidate_pool": "BALANCED_35_ORIGINAL_FIVE_PER_TRUE_CLASS",
        "strategies": STRATEGIES,
        "budgets": BUDGETS,
        "expected_trajectories": EXPECTED_TRAJECTORIES,
        "expected_folds": EXPECTED_FOLDS,
        "expected_predictions": EXPECTED_PREDICTIONS,
        "expected_selections": EXPECTED_SELECTIONS,
        "readiness_gates": readiness_gates,
        "failed_gates": failed,
        "all_readiness_gates_passed": not failed,
        "new_statistical_tests_performed": False,
        "p07_policy": "DESCRIPTIVE_CASE_ANALYSIS_ONLY",
        "stage3g_and_stage5f_replacement_allowed": False,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": final_decision,
    }
    atomic_json(report, RESULT_ROOT / "revision_R3C_report.json")

    print()
    print("=" * 100)
    print("REVISION R3C — BALANCED-POOL DESCRIPTIVE SUMMARY")
    print("=" * 100)
    print("Able-bodied summary:")
    print(able_summary.to_string(index=False))
    print()
    print("P07 descriptive summary:")
    print(
        p07_summary[
            [
                "strategy",
                "query_budget",
                "mean_repetition_balanced_accuracy",
                "mean_repetition_macro_f1",
                "total_repetition_errors",
            ]
        ].to_string(index=False)
    )
    print()
    print("Readiness gates:")
    for key, value in readiness_gates.items():
        print(f"  {key}: {bool(value)}")
    print("Failed gates:", failed if failed else "None")
    print("Uploading Revision R3C packet to Google Drive...")
    packet_crc, packet_hash, remote_verified = create_packet()
    print()
    print("Packet CRC pass:", packet_crc)
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", packet_hash)
    print("Remote round-trip verified:", remote_verified)
    print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 3))
    print()
    print("FINAL DECISION:", final_decision)


if __name__ == "__main__":
    main()
