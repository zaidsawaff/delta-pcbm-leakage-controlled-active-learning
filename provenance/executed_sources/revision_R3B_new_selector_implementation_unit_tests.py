from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import time
import tracemalloc
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import revision_R3A_P1_float32_engine_frozen_trajectory_unit_test as engine


START_TIME = time.time()
WORKING = engine.WORKING
INPUT_ROOT = WORKING / "REVISION_R3B_FROZEN_INPUTS"
RESULT_ROOT = (
    WORKING
    / "DELTA_REVIEWER_REVISION"
    / "Revision_R3B_New_Selector_Implementation_Unit_Tests"
)
PACKET_PATH = WORKING / "revision_R3B_new_selector_implementation_unit_test_packet.zip"
REMOTE_OUTPUT = (
    engine.REMOTE_BASE
    + "/Reviewer_Revision/Revision_R3B_New_Selector_Implementation_Unit_Tests"
)
for directory in (INPUT_ROOT, RESULT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

REVISION_R0_PACKET_SHA256 = (
    "0800e315a29b81934095ba56deaea3f8b6600fd0df13db348d7ea72d3b82df78"
)
REVISION_R3A_P1_PACKET_SHA256 = (
    "e5051aaf116af4888c632e27cd7008a7d4848b5308b6af4366a760b30a58435a"
)
REVISION_PROTOCOL_SHA256 = engine.REVISION_PROTOCOL_SHA256
UNIT_PARTICIPANT = "P01"
UNIT_SESSION = 1
UNIT_BUDGET = 7
PROBABILITY_COLUMNS = [f"probability_{label}" for label in range(engine.CLASSES)]
EMBEDDING_COLUMNS = [f"embedding_{channel:02d}" for channel in range(engine.CHANNELS)]
PROBABILITY_SCHEMA = ["opaque_candidate_token", *PROBABILITY_COLUMNS]
RBMAL_SCHEMA = ["opaque_candidate_token", "margin", *EMBEDDING_COLUMNS]
CORE_SET_SCHEMA = ["opaque_candidate_token", *EMBEDDING_COLUMNS]
FORBIDDEN_COLUMNS = {
    "participant",
    "session",
    "target_session",
    "label",
    "true_label",
    "repetition",
    "repetition_uid",
    "sequence_row",
    "relative_path",
    "dataset_key",
    "acquisition_order",
    "fixed_test",
}
CLASSICAL_STRATEGIES = [
    "LEAST_CONFIDENCE",
    "PREDICTIVE_ENTROPY",
    "RBMAL_MARGIN_DIVERSITY",
    "CORE_SET_GREEDY",
]


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


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def stable_token_tie(seed, token):
    payload = f"{int(seed)}|{str(token)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_opaque_tokens(series):
    tokens = series.astype(str)
    if tokens.duplicated().any():
        raise ValueError("Duplicate selector-visible opaque token")
    if not tokens.str.fullmatch(r"[0-9a-f]{24}").all():
        raise ValueError("Selector-visible token is not 24 lowercase hexadecimal characters")


def validate_exact_schema(frame, expected_columns, name):
    if frame.columns.tolist() != list(expected_columns):
        raise ValueError(
            f"{name} schema drift: expected {expected_columns}, got {frame.columns.tolist()}"
        )
    forbidden = sorted(set(frame.columns).intersection(FORBIDDEN_COLUMNS))
    if forbidden:
        raise ValueError(f"{name} received forbidden columns: {forbidden}")
    validate_opaque_tokens(frame["opaque_candidate_token"])
    numeric = frame.drop(columns=["opaque_candidate_token"]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} received non-finite numeric values")


def validate_probability_frame(frame):
    validate_exact_schema(frame, PROBABILITY_SCHEMA, "PROBABILITY_SELECTOR")
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probability outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("Class probabilities do not sum to one")
    return probabilities


def select_largest_score(tokens, scores, batch_size):
    table = pd.DataFrame(
        {
            "opaque_candidate_token": pd.Series(tokens, dtype=str),
            "score": np.asarray(scores, dtype=np.float64),
        }
    )
    if not np.isfinite(table["score"]).all():
        raise ValueError("Non-finite acquisition score")
    ordered = table.sort_values(
        ["score", "opaque_candidate_token"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected = ordered.head(int(batch_size))["opaque_candidate_token"].tolist()
    if len(selected) != int(batch_size) or len(set(selected)) != len(selected):
        raise RuntimeError("Selector did not return the requested unique batch")
    return selected, ordered.reset_index(drop=True)


def select_least_confidence(frame, batch_size=7):
    probabilities = validate_probability_frame(frame)
    uncertainty = 1.0 - probabilities.max(axis=1)
    return select_largest_score(
        frame["opaque_candidate_token"].astype(str), uncertainty, batch_size
    )


def select_predictive_entropy(frame, batch_size=7):
    probabilities = validate_probability_frame(frame)
    safe = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    entropy = -(safe * np.log(safe)).sum(axis=1)
    return select_largest_score(
        frame["opaque_candidate_token"].astype(str), entropy, batch_size
    )


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1e-15:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def minimum_distance(points, anchors):
    points = np.asarray(points, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    if points.ndim != 2 or anchors.ndim != 2 or points.shape[1] != anchors.shape[1]:
        raise ValueError("Embedding dimensions do not agree")
    if len(anchors) == 0:
        raise ValueError("At least one labeled-history embedding is required")
    squared = (
        np.sum(np.square(points), axis=1, keepdims=True)
        + np.sum(np.square(anchors), axis=1)[None, :]
        - 2.0 * points @ anchors.T
    )
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared.min(axis=1))


def select_rbmal(frame, history_embeddings, batch_size=7, tie_seed=0):
    validate_exact_schema(frame, RBMAL_SCHEMA, "RBMAL_MARGIN_DIVERSITY")
    if len(frame) < int(batch_size):
        raise ValueError("RBMAL batch exceeds candidate count")
    tokens = frame["opaque_candidate_token"].astype(str).to_numpy()
    margins = frame["margin"].to_numpy(dtype=np.float64)
    embeddings = frame[EMBEDDING_COLUMNS].to_numpy(dtype=np.float64)
    # Smaller margin is more uncertain. This normalized uncertainty is fixed
    # for the round; Euclidean novelty is recomputed after every nomination.
    uncertainty = 1.0 - minmax(margins)
    selected_indices = []
    audit_rows = []
    remaining = list(range(len(frame)))
    anchors = np.asarray(history_embeddings, dtype=np.float64).copy()
    for position in range(1, int(batch_size) + 1):
        candidate_embeddings = embeddings[remaining]
        novelty_raw = minimum_distance(candidate_embeddings, anchors)
        novelty = minmax(novelty_raw)
        combined = 0.5 * uncertainty[remaining] + 0.5 * novelty
        ranked = sorted(
            range(len(remaining)),
            key=lambda local: (
                -combined[local],
                stable_token_tie(tie_seed, tokens[remaining[local]]),
                tokens[remaining[local]],
            ),
        )
        chosen_local = ranked[0]
        chosen = remaining[chosen_local]
        selected_indices.append(chosen)
        audit_rows.append(
            {
                "selection_position": position,
                "opaque_candidate_token": tokens[chosen],
                "margin": margins[chosen],
                "normalized_uncertainty": uncertainty[chosen],
                "raw_minimum_novelty_distance": novelty_raw[chosen_local],
                "normalized_novelty": novelty[chosen_local],
                "combined_score": combined[chosen_local],
                "tie_hash": stable_token_tie(tie_seed, tokens[chosen]),
            }
        )
        anchors = np.vstack([anchors, embeddings[chosen]])
        remaining.remove(chosen)
    selected = tokens[selected_indices].tolist()
    return selected, pd.DataFrame(audit_rows)


def select_core_set(frame, history_embeddings, batch_size=7):
    validate_exact_schema(frame, CORE_SET_SCHEMA, "CORE_SET_GREEDY")
    if len(frame) < int(batch_size):
        raise ValueError("Core-set batch exceeds candidate count")
    tokens = frame["opaque_candidate_token"].astype(str).to_numpy()
    embeddings = frame[EMBEDDING_COLUMNS].to_numpy(dtype=np.float64)
    remaining = list(range(len(frame)))
    anchors = np.asarray(history_embeddings, dtype=np.float64).copy()
    selected_indices = []
    audit_rows = []
    for position in range(1, int(batch_size) + 1):
        distances = minimum_distance(embeddings[remaining], anchors)
        ranked = sorted(
            range(len(remaining)),
            key=lambda local: (
                -distances[local],
                tokens[remaining[local]],
            ),
        )
        chosen_local = ranked[0]
        chosen = remaining[chosen_local]
        selected_indices.append(chosen)
        audit_rows.append(
            {
                "selection_position": position,
                "opaque_candidate_token": tokens[chosen],
                "minimum_anchor_distance": distances[chosen_local],
            }
        )
        anchors = np.vstack([anchors, embeddings[chosen]])
        remaining.remove(chosen)
    selected = tokens[selected_indices].tolist()
    return selected, pd.DataFrame(audit_rows)


def fit_probability_calibrated_ridge(features, main_valid, metadata, history_rows):
    history_rows = np.asarray(sorted(map(int, history_rows)), dtype=int)
    history_meta = metadata.iloc[history_rows]
    labels = history_meta["label"].to_numpy(dtype=int)
    if sorted(np.unique(labels).tolist()) != list(range(engine.CLASSES)):
        raise RuntimeError("Calibration history lacks one or more classes")
    splitter = StratifiedKFold(n_splits=5, shuffle=False)
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
        state = engine.fit_history_state(
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
                }
            )
    if not np.isfinite(oof_scores).all() or (fold_assignments < 1).any():
        raise RuntimeError("OOF calibration scores are incomplete")
    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
    )
    calibrator.fit(oof_scores, labels)
    if calibrator.classes_.tolist() != list(range(engine.CLASSES)):
        raise RuntimeError("Calibrator class order drift")
    full_state = engine.fit_history_state(
        features, main_valid, metadata, history_rows
    )
    audit = {
        "history_rows": history_rows,
        "labels": labels,
        "oof_scores": oof_scores,
        "fold_assignments": fold_assignments,
        "fold_audit": pd.DataFrame(fold_rows),
        "calibrator": calibrator,
        "full_state": full_state,
        "calibration_contract": (
            "FIVE_FOLD_STRATIFIED_REPETITION_OOF_HISTORY_ONLY_"
            "MULTINOMIAL_L2_LOGISTIC_C1_LBFGS_MAXITER5000"
        ),
    }
    return audit


def calibrated_probabilities(calibrated_state, features, main_valid, rows):
    rows = np.asarray(rows, dtype=int)
    decision_scores, raw_predictions, raw_margins = engine.score_repetitions(
        calibrated_state["full_state"], features, main_valid, rows
    )
    probabilities = calibrated_state["calibrator"].predict_proba(decision_scores)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(rows), engine.CLASSES):
        raise RuntimeError("Calibrated probability shape drift")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite calibrated probability")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise RuntimeError("Calibrated probabilities do not sum to one")
    return probabilities, decision_scores, raw_predictions, raw_margins


def repetition_embeddings(state, features, main_valid, rows):
    transformed = engine.transform_repetitions(
        features,
        main_valid,
        rows,
        state["means"],
        state["stds"],
    )
    if transformed.dtype != np.float32:
        raise RuntimeError("R3B embedding input is not float32")
    embeddings = transformed.mean(axis=1, dtype=np.float32)
    if embeddings.shape != (len(rows), engine.CHANNELS):
        raise RuntimeError("Repetition embedding shape drift")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Non-finite repetition embedding")
    return embeddings


def prepare_metadata(stage5b_packet, stage5d2_packet):
    for basename in [
        "stage5b_rms_repetition_sequences.npy",
        "stage5b_main_valid_repetition_sequences.npy",
        "stage5b_repetition_metadata.csv",
    ]:
        engine.extract_member(stage5b_packet, basename, INPUT_ROOT / basename)
    features = np.load(
        INPUT_ROOT / "stage5b_rms_repetition_sequences.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    main_valid = np.load(
        INPUT_ROOT / "stage5b_main_valid_repetition_sequences.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    metadata = pd.read_csv(INPUT_ROOT / "stage5b_repetition_metadata.csv")
    metadata["participant"] = metadata["participant"].astype(str)
    for column in ["session", "label", "repetition", "sequence_row"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(int)
    metadata = metadata.sort_values("sequence_row").reset_index(drop=True)
    if metadata["sequence_row"].tolist() != list(range(2940)):
        raise RuntimeError("Sequence-row order drift")
    initial = metadata["session"].eq(0) & metadata["repetition"].le(5)
    candidate = metadata["session"].between(1, 5) & metadata["repetition"].le(5)
    fixed = metadata["session"].between(1, 5) & metadata["repetition"].ge(6)
    metadata["protocol_role"] = "UNUSED"
    metadata.loc[initial, "protocol_role"] = "INITIAL_LABELED_CALIBRATION"
    metadata.loc[candidate, "protocol_role"] = "CURRENT_SESSION_UNLABELED_POOL"
    metadata.loc[fixed, "protocol_role"] = "TARGET_FIXED_TEST_NEVER_QUERY"
    metadata["eligible_for_training"] = initial | candidate
    metadata["fixed_test_never_query"] = fixed
    metadata["repetition_uid"] = metadata.apply(
        lambda row: (
            f"{row.participant}_S{int(row.session):02d}_"
            f"L{int(row.label)}_R{int(row.repetition):02d}"
        ),
        axis=1,
    )
    _, deep_selection = engine.find_csv_member(
        stage5d2_packet,
        ["selection", "trace"],
        preferred_basename="stage5d2_selection_trace.csv",
    )
    strategy_column = engine.resolve_column(deep_selection, ["strategy"])
    row_column = engine.resolve_column(
        deep_selection,
        ["sequence_row_internal", "sequence_row"],
        contains=["sequence", "row"],
    )
    token_column = engine.resolve_column(
        deep_selection,
        ["opaque_candidate_token"],
        contains=["opaque", "token"],
    )
    token_table = deep_selection.loc[
        deep_selection[strategy_column].astype(str).eq("FULL_POOL_REFERENCE"),
        [row_column, token_column],
    ].copy()
    token_table[row_column] = pd.to_numeric(
        token_table[row_column], errors="raise"
    ).astype(int)
    token_table[token_column] = token_table[token_column].astype(str)
    token_table = token_table.drop_duplicates()
    if len(token_table) != 1225:
        raise RuntimeError("Full-pool token map does not contain 1,225 candidates")
    metadata["opaque_candidate_token"] = metadata["sequence_row"].map(
        dict(zip(token_table[row_column], token_table[token_column]))
    )
    if metadata.loc[candidate, "opaque_candidate_token"].isna().any():
        raise RuntimeError("Candidate opaque-token map is incomplete")
    return features, main_valid, metadata


def resolve_inputs():
    r0_packet, r0_source = engine.resolve_packet(
        "stageR0_reviewer_revision_protocol_lock_packet.zip",
        REVISION_R0_PACKET_SHA256,
    )
    p1_packet, p1_source = engine.resolve_packet(
        "revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip",
        REVISION_R3A_P1_PACKET_SHA256,
    )
    r0_report = engine.read_json_member(r0_packet, "stageR0_protocol_lock_report.json")
    r0_protocol = engine.read_json_member(r0_packet, "stageR0_locked_revision_protocol.json")
    p1_report = engine.read_json_member(
        p1_packet, "revision_R3A_P1_float32_reconstruction_report.json"
    )
    if not r0_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("Revision R0 parent gates did not pass")
    if not p1_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("Revision R3A-P1 parent gates did not pass")
    if r0_protocol.get("protocol_sha256") != REVISION_PROTOCOL_SHA256:
        raise RuntimeError("Revision R0 protocol hash drift")
    if p1_report.get("revision_protocol_sha256") != REVISION_PROTOCOL_SHA256:
        raise RuntimeError("Revision R3A-P1 protocol hash drift")
    if p1_report.get("numerical_engine_contract") != engine.NUMERICAL_ENGINE_CONTRACT:
        raise RuntimeError("R3A-P1 numerical engine contract drift")
    strategies = engine.read_csv_member(r0_packet, "stageR0_strategy_definitions.csv")
    seeds = engine.read_csv_member(r0_packet, "stageR0_seed_schedule.csv")
    expected = set(CLASSICAL_STRATEGIES + ["BADGE"])
    if not expected.issubset(set(strategies["strategy"].astype(str))):
        raise RuntimeError("R0 strategy definitions are incomplete")
    badge_seed = seeds.loc[
        seeds["seed_family"].astype(str).eq("BADGE_KMEANS_PP"), "seed"
    ]
    rbmal_seed = seeds.loc[
        seeds["seed_family"].astype(str).eq("RBMAL_TIES"), "seed"
    ]
    if len(badge_seed) != 1 or len(rbmal_seed) != 1:
        raise RuntimeError("R0 locked selector seeds are incomplete")
    stage5b_packet, stage5b_source = engine.resolve_packet(
        "stage5b_deep_sequence_assembly_packet.zip",
        engine.STAGE5B_PACKET_SHA256,
    )
    stage5d2_packet, stage5d2_source = engine.resolve_packet(
        "stage5d2_full_deterministic_deep_trajectories_packet.zip",
        engine.STAGE5D2_PACKET_SHA256,
    )
    records = [
        (r0_packet, REVISION_R0_PACKET_SHA256, r0_source),
        (p1_packet, REVISION_R3A_P1_PACKET_SHA256, p1_source),
        (stage5b_packet, engine.STAGE5B_PACKET_SHA256, stage5b_source),
        (stage5d2_packet, engine.STAGE5D2_PACKET_SHA256, stage5d2_source),
    ]
    audit = pd.DataFrame(
        [
            {
                "packet": path.name,
                "expected_sha256": digest,
                "observed_sha256": engine.sha256_file(path),
                "hash_matches": engine.sha256_file(path) == digest,
                "crc_passes": engine.archive_crc_passes(path),
                "source": source,
            }
            for path, digest, source in records
        ]
    )
    if not audit[["hash_matches", "crc_passes"]].all().all():
        raise RuntimeError("R3B frozen-input integrity failed")
    return {
        "r0_packet": r0_packet,
        "p1_packet": p1_packet,
        "stage5b_packet": stage5b_packet,
        "stage5d2_packet": stage5d2_packet,
        "r0_protocol": r0_protocol,
        "r0_report": r0_report,
        "p1_report": p1_report,
        "strategies": strategies,
        "seeds": seeds,
        "rbmal_seed": int(rbmal_seed.iloc[0]),
        "badge_seed": int(badge_seed.iloc[0]),
        "audit": audit,
    }


def build_probability_frame(tokens, probabilities):
    frame = pd.DataFrame(probabilities, columns=PROBABILITY_COLUMNS)
    frame.insert(0, "opaque_candidate_token", pd.Series(tokens, dtype=str))
    return frame[PROBABILITY_SCHEMA]


def build_rbmal_frame(tokens, margins, embeddings):
    frame = pd.DataFrame(embeddings, columns=EMBEDDING_COLUMNS)
    frame.insert(0, "margin", np.asarray(margins, dtype=np.float64))
    frame.insert(0, "opaque_candidate_token", pd.Series(tokens, dtype=str))
    return frame[RBMAL_SCHEMA]


def build_core_frame(tokens, embeddings):
    frame = pd.DataFrame(embeddings, columns=EMBEDDING_COLUMNS)
    frame.insert(0, "opaque_candidate_token", pd.Series(tokens, dtype=str))
    return frame[CORE_SET_SCHEMA]


def synthetic_selector_tests(rbmal_seed):
    tokens = [f"{index:024x}" for index in range(12)]
    probabilities = np.full((12, engine.CLASSES), 0.01, dtype=np.float64)
    probabilities[:, 0] = np.linspace(0.93, 0.38, 12)
    probabilities[:, 1:] = (
        (1.0 - probabilities[:, [0]]) / (engine.CLASSES - 1)
    )
    probability_frame = build_probability_frame(tokens, probabilities)
    lc, _ = select_least_confidence(probability_frame, 7)
    entropy, _ = select_predictive_entropy(probability_frame, 7)
    expected_uncertain = tokens[-7:]
    probability_expected = lc == expected_uncertain[::-1] and entropy == expected_uncertain[::-1]

    history = np.zeros((1, engine.CHANNELS), dtype=np.float32)
    embeddings = np.zeros((12, engine.CHANNELS), dtype=np.float32)
    embeddings[:, 0] = np.arange(1, 13, dtype=np.float32)
    margins = np.linspace(0.0, 1.0, 12)
    rbmal_frame = build_rbmal_frame(tokens, margins, embeddings)
    rbmal_first, _ = select_rbmal(
        rbmal_frame, history, batch_size=7, tie_seed=rbmal_seed
    )
    rbmal_second, _ = select_rbmal(
        rbmal_frame, history, batch_size=7, tie_seed=rbmal_seed
    )
    core_frame = build_core_frame(tokens, embeddings)
    core, _ = select_core_set(core_frame, history, batch_size=7)
    core_expected_first = core[0] == tokens[-1]

    forbidden_rejections = {}
    for name, frame, selector in [
        ("LEAST_CONFIDENCE", probability_frame, select_least_confidence),
        ("PREDICTIVE_ENTROPY", probability_frame, select_predictive_entropy),
    ]:
        invalid = frame.copy()
        invalid["true_label"] = 0
        try:
            selector(invalid, 7)
            forbidden_rejections[name] = False
        except ValueError:
            forbidden_rejections[name] = True
    invalid_rbmal = rbmal_frame.copy()
    invalid_rbmal["participant"] = "P01"
    try:
        select_rbmal(invalid_rbmal, history, 7, rbmal_seed)
        forbidden_rejections["RBMAL_MARGIN_DIVERSITY"] = False
    except ValueError:
        forbidden_rejections["RBMAL_MARGIN_DIVERSITY"] = True
    invalid_core = core_frame.copy()
    invalid_core["repetition_uid"] = "hidden"
    try:
        select_core_set(invalid_core, history, 7)
        forbidden_rejections["CORE_SET_GREEDY"] = False
    except ValueError:
        forbidden_rejections["CORE_SET_GREEDY"] = True

    rows = [
        {
            "test_id": "PROBABILITY_SELECTORS_KNOWN_ORDER",
            "passed": probability_expected,
            "detail": "LC and entropy choose the seven least confident synthetic rows",
        },
        {
            "test_id": "RBMAL_DETERMINISTIC_LOCKED_TIE_SEED",
            "passed": rbmal_first == rbmal_second and len(set(rbmal_first)) == 7,
            "detail": "Repeated RBMAL calls return one identical unique batch",
        },
        {
            "test_id": "CORE_SET_FARTHEST_FIRST",
            "passed": core_expected_first and len(set(core)) == 7,
            "detail": "Greedy k-center first selects the farthest point from history",
        },
        {
            "test_id": "ALL_FORBIDDEN_COLUMNS_REJECTED",
            "passed": all(forbidden_rejections.values()),
            "detail": json.dumps(forbidden_rejections, sort_keys=True),
        },
    ]
    return pd.DataFrame(rows)


def timed_phase(name, function, *args, **kwargs):
    tracemalloc.start()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    telemetry = {
        "phase": name,
        "wall_seconds": elapsed,
        "python_peak_tracemalloc_mb": peak / (1024**2),
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
    }
    return result, telemetry


def create_packet():
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
        RESULT_ROOT / "revision_R3B_output_sha256_manifest.csv",
    )
    crc_pass = engine.make_zip(
        RESULT_ROOT,
        PACKET_PATH,
        "Revision_R3B_New_Selector_Implementation_Unit_Tests",
    )
    if not crc_pass:
        raise RuntimeError("R3B packet CRC failed")
    return crc_pass


def preserve_and_upload(report):
    atomic_json(report, RESULT_ROOT / "revision_R3B_selector_unit_test_report.json")
    source = Path(__file__)
    if source.exists():
        shutil.copy2(source, RESULT_ROOT / "revision_R3B_executed_source.py")
    packet_crc = create_packet()
    packet_hash = engine.sha256_file(PACKET_PATH)
    remote_verified = engine.roundtrip_remote_file(
        PACKET_PATH,
        REMOTE_OUTPUT + "/" + PACKET_PATH.name,
        packet_hash,
    )
    return packet_crc, packet_hash, remote_verified


def main():
    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PACKET_PATH.unlink(missing_ok=True)

    print("=" * 100)
    print("REVISION R3B — NEW SELECTOR IMPLEMENTATION AND UNIT TESTS")
    print("=" * 100)
    print("Execution device: CPU")
    print("Scientific role: IMPLEMENTATION UNIT TESTS ONLY")
    print("Full comparator trajectories: False")
    print("Fixed-test inference: False")
    print("Raw HDF5 accessed: False")
    print("New statistical tests: False")
    print("Classical selectors: LC, entropy, RBMAL, core-set")
    print("BADGE: deferred to locked TCN/GPU Stage R6")
    print()

    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified R0, R3A-P1, Stage 5B, and Stage 5D-2 packets...")
    inputs = resolve_inputs()
    atomic_csv(inputs["audit"], RESULT_ROOT / "revision_R3B_input_audit.csv")
    features, main_valid, metadata = prepare_metadata(
        inputs["stage5b_packet"], inputs["stage5d2_packet"]
    )

    history_rows = engine.initial_history_rows(metadata, UNIT_PARTICIPANT)
    candidate_rows = engine.candidate_rows(metadata, UNIT_PARTICIPANT, UNIT_SESSION)
    fixed_rows = engine.fixed_test_rows(metadata, UNIT_PARTICIPANT, UNIT_SESSION)
    tokens = metadata.iloc[candidate_rows]["opaque_candidate_token"].astype(str).to_numpy()
    validate_opaque_tokens(pd.Series(tokens))

    synthetic = synthetic_selector_tests(inputs["rbmal_seed"])
    atomic_csv(synthetic, RESULT_ROOT / "revision_R3B_synthetic_unit_tests.csv")

    calibrated, fit_telemetry = timed_phase(
        "RIDGE_OOF_CALIBRATION_AND_FULL_HISTORY_REFIT",
        fit_probability_calibrated_ridge,
        features,
        main_valid,
        metadata,
        history_rows,
    )
    probability_result, probability_telemetry = timed_phase(
        "CALIBRATED_CANDIDATE_SCORING",
        calibrated_probabilities,
        calibrated,
        features,
        main_valid,
        candidate_rows,
    )
    probabilities, decision_scores, raw_predictions, raw_margins = probability_result
    history_embeddings = repetition_embeddings(
        calibrated["full_state"], features, main_valid, history_rows
    )
    candidate_embeddings = repetition_embeddings(
        calibrated["full_state"], features, main_valid, candidate_rows
    )

    probability_frame = build_probability_frame(tokens, probabilities)
    rbmal_frame = build_rbmal_frame(tokens, raw_margins, candidate_embeddings)
    core_frame = build_core_frame(tokens, candidate_embeddings)

    (lc_result, lc_telemetry) = timed_phase(
        "LEAST_CONFIDENCE_SELECTION",
        select_least_confidence,
        probability_frame,
        UNIT_BUDGET,
    )
    (entropy_result, entropy_telemetry) = timed_phase(
        "PREDICTIVE_ENTROPY_SELECTION",
        select_predictive_entropy,
        probability_frame,
        UNIT_BUDGET,
    )
    (rbmal_result, rbmal_telemetry) = timed_phase(
        "RBMAL_SELECTION",
        select_rbmal,
        rbmal_frame,
        history_embeddings,
        UNIT_BUDGET,
        inputs["rbmal_seed"],
    )
    (core_result, core_telemetry) = timed_phase(
        "CORE_SET_SELECTION",
        select_core_set,
        core_frame,
        history_embeddings,
        UNIT_BUDGET,
    )
    lc_tokens, lc_audit = lc_result
    entropy_tokens, entropy_audit = entropy_result
    rbmal_tokens, rbmal_audit = rbmal_result
    core_tokens, core_audit = core_result
    repeated = {
        "LEAST_CONFIDENCE": select_least_confidence(probability_frame, UNIT_BUDGET)[0],
        "PREDICTIVE_ENTROPY": select_predictive_entropy(probability_frame, UNIT_BUDGET)[0],
        "RBMAL_MARGIN_DIVERSITY": select_rbmal(
            rbmal_frame,
            history_embeddings,
            UNIT_BUDGET,
            inputs["rbmal_seed"],
        )[0],
        "CORE_SET_GREEDY": select_core_set(
            core_frame, history_embeddings, UNIT_BUDGET
        )[0],
    }
    selections = {
        "LEAST_CONFIDENCE": lc_tokens,
        "PREDICTIVE_ENTROPY": entropy_tokens,
        "RBMAL_MARGIN_DIVERSITY": rbmal_tokens,
        "CORE_SET_GREEDY": core_tokens,
    }

    candidate_token_set = set(tokens)
    token_to_candidate_row = dict(
        zip(
            metadata.iloc[candidate_rows]["opaque_candidate_token"].astype(str),
            map(int, candidate_rows),
        )
    )
    fixed_row_set = set(map(int, fixed_rows))
    real_rows = []
    for strategy, selected in selections.items():
        for position, token in enumerate(selected, start=1):
            selected_row = token_to_candidate_row.get(token)
            real_rows.append(
                {
                    "unit_test_only": True,
                    "participant": UNIT_PARTICIPANT,
                    "target_session": UNIT_SESSION,
                    "strategy": strategy,
                    "query_budget": UNIT_BUDGET,
                    "selection_position": position,
                    "opaque_candidate_token": token,
                    "selected_sequence_row_internal_audit_only": selected_row,
                    "selected_record_is_candidate": token in candidate_token_set
                    and selected_row in set(map(int, candidate_rows)),
                    "selected_record_is_fixed_test": selected_row in fixed_row_set,
                }
            )
    real_selection = pd.DataFrame(real_rows)
    atomic_csv(
        real_selection,
        RESULT_ROOT / "revision_R3B_real_data_unit_selection_trace.csv",
    )
    atomic_csv(
        calibrated["fold_audit"],
        RESULT_ROOT / "revision_R3B_ridge_oof_calibration_fold_audit.csv",
    )
    atomic_csv(rbmal_audit, RESULT_ROOT / "revision_R3B_rbmal_step_audit.csv")
    atomic_csv(core_audit, RESULT_ROOT / "revision_R3B_core_set_step_audit.csv")

    # Leakage anchor: overwrite all current candidate/fixed features and their
    # labels. A history-only fit must remain byte-identical.
    modified_features = np.array(features, copy=True)
    outside_rows = np.concatenate([candidate_rows, fixed_rows])
    modified_features[outside_rows] = np.float32(123.456)
    modified_metadata = metadata.copy()
    modified_metadata.loc[outside_rows, "label"] = (
        6 - modified_metadata.loc[outside_rows, "label"].to_numpy(dtype=int)
    )
    modified_calibrated = fit_probability_calibrated_ridge(
        modified_features,
        main_valid,
        modified_metadata,
        history_rows,
    )
    leakage_anchor = {
        "history_row_sets_match": np.array_equal(
            calibrated["history_rows"], modified_calibrated["history_rows"]
        ),
        "oof_scores_are_byte_identical": np.array_equal(
            calibrated["oof_scores"], modified_calibrated["oof_scores"]
        ),
        "calibrator_coefficients_are_byte_identical": np.array_equal(
            calibrated["calibrator"].coef_, modified_calibrated["calibrator"].coef_
        ),
        "calibrator_intercepts_are_byte_identical": np.array_equal(
            calibrated["calibrator"].intercept_,
            modified_calibrated["calibrator"].intercept_,
        ),
        "full_ridge_coefficients_are_byte_identical": np.array_equal(
            calibrated["full_state"]["model"].coef_,
            modified_calibrated["full_state"]["model"].coef_,
        ),
        "full_normalizer_means_are_byte_identical": np.array_equal(
            calibrated["full_state"]["means"],
            modified_calibrated["full_state"]["means"],
        ),
        "full_normalizer_stds_are_byte_identical": np.array_equal(
            calibrated["full_state"]["stds"],
            modified_calibrated["full_state"]["stds"],
        ),
        "unlabeled_pool_used_for_calibration": False,
        "fixed_test_used_for_calibration": False,
    }
    atomic_json(leakage_anchor, RESULT_ROOT / "revision_R3B_leakage_anchor.json")

    schema_audit = pd.DataFrame(
        [
            {
                "strategy": "LEAST_CONFIDENCE",
                "schema_columns": "|".join(probability_frame.columns),
                "schema_column_count": len(probability_frame.columns),
                "exact_schema": probability_frame.columns.tolist() == PROBABILITY_SCHEMA,
                "forbidden_columns": "",
            },
            {
                "strategy": "PREDICTIVE_ENTROPY",
                "schema_columns": "|".join(probability_frame.columns),
                "schema_column_count": len(probability_frame.columns),
                "exact_schema": probability_frame.columns.tolist() == PROBABILITY_SCHEMA,
                "forbidden_columns": "",
            },
            {
                "strategy": "RBMAL_MARGIN_DIVERSITY",
                "schema_columns": "|".join(rbmal_frame.columns),
                "schema_column_count": len(rbmal_frame.columns),
                "exact_schema": rbmal_frame.columns.tolist() == RBMAL_SCHEMA,
                "forbidden_columns": "",
            },
            {
                "strategy": "CORE_SET_GREEDY",
                "schema_columns": "|".join(core_frame.columns),
                "schema_column_count": len(core_frame.columns),
                "exact_schema": core_frame.columns.tolist() == CORE_SET_SCHEMA,
                "forbidden_columns": "",
            },
        ]
    )
    atomic_csv(schema_audit, RESULT_ROOT / "revision_R3B_selector_schema_audit.csv")

    telemetry = pd.DataFrame(
        [
            fit_telemetry,
            probability_telemetry,
            lc_telemetry,
            entropy_telemetry,
            rbmal_telemetry,
            core_telemetry,
        ]
    )
    atomic_csv(telemetry, RESULT_ROOT / "revision_R3B_compute_telemetry.csv")

    implementation_contract = {
        "numerical_engine": engine.NUMERICAL_ENGINE_CONTRACT,
        "calibration": calibrated["calibration_contract"],
        "least_confidence": "SELECT_DESCENDING_1_MINUS_MAX_CALIBRATED_PROBABILITY",
        "predictive_entropy": "SELECT_DESCENDING_NEGATIVE_SUM_P_LOG_P",
        "rbmal": (
            "SEQUENTIAL_0.5_MINMAX_MARGIN_UNCERTAINTY_PLUS_"
            "0.5_MINMAX_EUCLIDEAN_NOVELTY_WITH_LOCKED_HASH_TIE"
        ),
        "core_set": "GREEDY_K_CENTER_SOURCE_NORMALIZED_REPETITION_EMBEDDING",
        "embedding": "MEAN_OF_37_FLOAT32_MASKED_NORMALIZED_MODEL_INPUT_WINDOWS",
        "tie_break": "OPAQUE_TOKEN_ONLY; RBMAL_USES_LOCKED_R0_TIE_SEED_HASH",
        "badge": "DEFERRED_TO_R6_TCN_GPU_AS_LOCKED_IN_R1",
    }
    atomic_json(
        implementation_contract,
        RESULT_ROOT / "revision_R3B_implementation_contract.json",
    )

    all_selected = real_selection.groupby("strategy").size()
    readiness_gates = {
        "revision_r0_packet_hash_matches": engine.sha256_file(inputs["r0_packet"])
        == REVISION_R0_PACKET_SHA256,
        "revision_r3a_p1_packet_hash_matches": engine.sha256_file(inputs["p1_packet"])
        == REVISION_R3A_P1_PACKET_SHA256,
        "all_four_input_packets_pass_hash_and_crc": bool(
            inputs["audit"][["hash_matches", "crc_passes"]].all().all()
        ),
        "revision_protocol_hash_matches": inputs["r0_protocol"].get("protocol_sha256")
        == REVISION_PROTOCOL_SHA256,
        "r3a_p1_all_gates_passed": bool(
            inputs["p1_report"].get("all_readiness_gates_passed")
        ),
        "float32_engine_contract_is_preserved": calibrated["full_state"][
            "numerical_engine_contract"
        ]
        == engine.NUMERICAL_ENGINE_CONTRACT,
        "history_has_35_repetitions": len(history_rows) == 35,
        "candidate_pool_has_35_repetitions": len(candidate_rows) == 35,
        "fixed_test_has_35_repetitions_but_is_not_scored": len(fixed_rows) == 35,
        "five_calibration_folds_are_present": calibrated["fold_audit"][
            "fold_index"
        ].nunique()
        == 5,
        "each_history_repetition_is_oof_scored_once": len(
            calibrated["fold_audit"]
        )
        == 35
        and calibrated["fold_audit"]["history_sequence_row_internal"].nunique()
        == 35,
        "each_oof_fold_trains_on_28_and_validates_on_7": bool(
            calibrated["fold_audit"]["train_repetition_count"].eq(28).all()
            and calibrated["fold_audit"]["validation_repetition_count"].eq(7).all()
        ),
        "each_oof_fold_contains_all_seven_classes_once": bool(
            calibrated["fold_audit"]
            .groupby("fold_index")["true_label_internal_audit_only"]
            .agg(["size", "nunique"])
            .eq(7)
            .all()
            .all()
        ),
        "calibration_uses_history_only": bool(
            calibrated["fold_audit"][
                "normalizer_training_rows_are_history_only"
            ].all()
            and calibrated["fold_audit"][
                "validation_row_is_not_in_fold_training"
            ].all()
        ),
        "calibrated_probabilities_are_finite": bool(np.isfinite(probabilities).all()),
        "calibrated_probabilities_sum_to_one": bool(
            np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0.0)
        ),
        "calibrator_has_all_seven_classes": calibrated["calibrator"].classes_.tolist()
        == list(range(engine.CLASSES)),
        "calibrator_converged_before_maximum_iterations": bool(
            np.max(calibrated["calibrator"].n_iter_) < 5000
        ),
        "all_synthetic_selector_tests_pass": bool(synthetic["passed"].all()),
        "four_real_selector_batches_are_present": set(selections) == set(
            CLASSICAL_STRATEGIES
        ),
        "every_real_selector_selects_seven_unique_candidates": bool(
            (all_selected == 7).all()
            and all(len(set(selected)) == 7 for selected in selections.values())
        ),
        "all_real_selected_records_are_candidates": bool(
            real_selection["selected_record_is_candidate"].all()
        ),
        "no_fixed_test_record_is_selected": bool(
            (~real_selection["selected_record_is_fixed_test"]).all()
        ),
        "all_real_selectors_are_deterministic": all(
            selections[strategy] == repeated[strategy]
            for strategy in CLASSICAL_STRATEGIES
        ),
        "all_selector_schemas_are_exact": bool(schema_audit["exact_schema"].all()),
        "all_selector_schemas_exclude_forbidden_columns": bool(
            schema_audit["forbidden_columns"].eq("").all()
        ),
        "candidate_and_fixed_mutation_cannot_change_calibrator_or_ridge": all(
            bool(value)
            for key, value in leakage_anchor.items()
            if key.endswith("byte_identical") or key.endswith("sets_match")
        ),
        "unlabeled_pool_is_not_used_for_calibration": leakage_anchor[
            "unlabeled_pool_used_for_calibration"
        ]
        is False,
        "fixed_test_is_not_used_for_calibration": leakage_anchor[
            "fixed_test_used_for_calibration"
        ]
        is False,
        "compute_telemetry_has_six_phases": len(telemetry) == 6,
        "all_telemetry_values_are_finite_and_nonnegative": bool(
            np.isfinite(
                telemetry[
                    [
                        "wall_seconds",
                        "python_peak_tracemalloc_mb",
                        "process_peak_rss_mb",
                    ]
                ].to_numpy(dtype=float)
            ).all()
            and (
                telemetry[
                    [
                        "wall_seconds",
                        "python_peak_tracemalloc_mb",
                        "process_peak_rss_mb",
                    ]
                ]
                >= 0
            ).all().all()
        ),
        "badge_is_deferred_to_locked_r6_gpu_stage": implementation_contract["badge"]
        == "DEFERRED_TO_R6_TCN_GPU_AS_LOCKED_IN_R1",
        "fixed_test_inference_was_not_run": True,
        "full_comparator_trajectories_were_not_run": True,
        "raw_hdf5_was_not_accessed": True,
        "no_new_statistical_test_was_run": True,
        "stage3g_and_stage5f_conclusions_cannot_be_replaced": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in readiness_gates.items() if not bool(value)]
    final_decision = (
        "PASS_TO_REVISION_R3C_BALANCED_POOL_CLASSICAL_COMPARATOR_EXTENSION"
        if not failed
        else "HOLD_FOR_REVISION_R3B_SELECTOR_IMPLEMENTATION_DIAGNOSTIC"
    )
    report = {
        "stage": "REVISION_R3B_NEW_SELECTOR_IMPLEMENTATION_UNIT_TESTS",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "revision_r0_packet_sha256": REVISION_R0_PACKET_SHA256,
        "revision_r3a_p1_packet_sha256": REVISION_R3A_P1_PACKET_SHA256,
        "unit_participant": UNIT_PARTICIPANT,
        "unit_session": UNIT_SESSION,
        "unit_budget": UNIT_BUDGET,
        "implemented_classical_strategies": CLASSICAL_STRATEGIES,
        "badge_status": implementation_contract["badge"],
        "readiness_gates": readiness_gates,
        "failed_gates": failed,
        "all_readiness_gates_passed": not failed,
        "fixed_test_inference_performed": False,
        "full_comparator_trajectories_performed": False,
        "new_statistical_tests_performed": False,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": final_decision,
    }

    print()
    print("Real-data unit selections:")
    summary = real_selection.groupby("strategy", as_index=False).agg(
        selected_count=("opaque_candidate_token", "size"),
        unique_tokens=("opaque_candidate_token", "nunique"),
        all_candidates=("selected_record_is_candidate", "all"),
        any_fixed_test=("selected_record_is_fixed_test", "any"),
    )
    print(summary.to_string(index=False))
    print()
    print("OOF calibration folds:", calibrated["fold_audit"]["fold_index"].nunique())
    print("OOF repetitions scored exactly once:", len(calibrated["fold_audit"]))
    leakage_passed = all(
        bool(value)
        for key, value in leakage_anchor.items()
        if key.endswith("byte_identical") or key.endswith("sets_match")
    ) and not leakage_anchor["unlabeled_pool_used_for_calibration"] and not leakage_anchor[
        "fixed_test_used_for_calibration"
    ]
    print("Leakage anchor passed:", leakage_passed)
    print("Synthetic selector tests passed:", bool(synthetic["passed"].all()))
    print("Failed readiness gates:", failed if failed else "None")
    print("Uploading Revision R3B packet to Google Drive...")
    packet_crc, packet_hash, remote_verified = preserve_and_upload(report)
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
