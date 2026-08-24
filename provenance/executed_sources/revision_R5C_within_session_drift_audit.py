from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import revision_R3A_P1_float32_engine_frozen_trajectory_unit_test as engine


REVISION_PROTOCOL_SHA256 = "6807b71de18ca82013cfa4360d760e0daf9a920a1acc0625dcb13bd8f4d07249"
R5A_PACKET_SHA256 = "c41e8e387b79328040e918621497e386a260fa17a9e802dd94f218f42f9ec11e"
R5B_PACKET_SHA256 = "2cc8b8d83afb6483060b57d4c1a1d9f67213a7fb4c09de1226b86d2de640cccf"
STAGE5B_PACKET_SHA256 = "1c0fbc63f6412362f3ae7cd22609ea6a7fcb23236cdf688ad5fe0578ebaab84d"
R3A_ENGINE_SOURCE_SHA256 = "6944d03a771d8b26f22d2cefd3ca7914ffc72b2ba2450d4efee1c9ef198ab2ee"
NORMALIZATION_EPSILON = np.float32(1e-8)

PARTICIPANTS = [f"P{i:02d}" for i in range(1, 8)]
ABLE_BODIED = PARTICIPANTS[:6]
TARGET_SESSIONS = [1, 2, 3, 4, 5]
LABELS = list(range(7))
REPETITIONS = list(range(1, 11))

WORKING = Path(os.environ.get("REVISION_R5C_WORKING", "/kaggle/working"))
INPUT_ROOT = WORKING / "REVISION_R5C_FROZEN_INPUTS"
RESULT_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R5C_Within_Session_Drift_Audit"
PACKET_PATH = WORKING / "revision_R5C_within_session_drift_audit_packet.zip"
REMOTE_BASE = engine.REMOTE_BASE
REMOTE_OUTPUT = REMOTE_BASE + "/Reviewer_Revision/Revision_R5C_Within_Session_Drift_Audit"
START_TIME = time.time()

DIRECT_PACKETS = {
    "revision_R5A_temporal_split_drift_unit_test_packet.zip": (
        R5A_PACKET_SHA256,
        "Reviewer_Revision/Revision_R5A_Temporal_Split_Drift_Unit_Tests/"
        "revision_R5A_temporal_split_drift_unit_test_packet.zip",
    ),
    "revision_R5B_ridge_temporal_split_sensitivity_packet.zip": (
        R5B_PACKET_SHA256,
        "Reviewer_Revision/Revision_R5B_Ridge_Temporal_Split_Sensitivity/"
        "revision_R5B_ridge_temporal_split_sensitivity_packet.zip",
    ),
    "stage5b_deep_sequence_assembly_packet.zip": (
        STAGE5B_PACKET_SHA256,
        "Stage5B_Deep_Sequence_Assembly/stage5b_deep_sequence_assembly_packet.zip",
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


def normalize_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    metadata = frame.copy()
    metadata["participant"] = metadata["participant"].astype(str)
    for column in ["sequence_row", "session", "label", "repetition"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(int)
    metadata = metadata.sort_values("sequence_row", kind="mergesort").reset_index(drop=True)
    if len(metadata) != 2940 or metadata["sequence_row"].tolist() != list(range(2940)):
        raise RuntimeError("Stage 5B metadata row contract failed")
    return metadata


def prepare_inputs() -> dict:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    resolved = {}
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
        raise RuntimeError("R5C frozen input integrity failed")

    r5a_packet = resolved["revision_R5A_temporal_split_drift_unit_test_packet.zip"]
    r5b_packet = resolved["revision_R5B_ridge_temporal_split_sensitivity_packet.zip"]
    stage5b_packet = resolved["stage5b_deep_sequence_assembly_packet.zip"]
    r5a_report = engine.read_json_member(r5a_packet, "revision_R5A_final_report.json")
    r5b_report = engine.read_json_member(r5b_packet, "revision_R5B_final_report.json")
    drift_spec = engine.read_csv_member(r5a_packet, "revision_R5A_drift_implementation_specification.csv")
    predictions = engine.read_csv_member(r5b_packet, "revision_R5B_aggregate_predictions.csv")
    folds = engine.read_csv_member(r5b_packet, "revision_R5B_aggregate_folds.csv")
    for basename in [
        "stage5b_rms_repetition_sequences.npy",
        "stage5b_main_valid_repetition_sequences.npy",
        "stage5b_repetition_metadata.csv",
    ]:
        engine.extract_member(stage5b_packet, basename, INPUT_ROOT / basename)
    metadata = normalize_metadata(pd.read_csv(INPUT_ROOT / "stage5b_repetition_metadata.csv"))
    atomic_csv(audit, INPUT_ROOT / "revision_R5C_input_packet_audit.csv")
    atomic_csv(drift_spec, INPUT_ROOT / "revision_R5C_locked_drift_specification.csv")
    return {
        "audit": audit,
        "r5a_report": r5a_report,
        "r5b_report": r5b_report,
        "drift_spec": drift_spec,
        "predictions": predictions,
        "folds": folds,
        "metadata": metadata,
    }


def extract_no_adaptation_performance(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chosen_splits = {
        "SECOND_HALF_REVERSED": "EARLY",
        "FIRST_HALF_ORIGINAL": "LATE",
    }
    subset = predictions.loc[
        predictions["strategy"].astype(str).eq("NO_ADAPTATION_REFERENCE")
        & predictions["split_id"].astype(str).isin(chosen_splits)
        & pd.to_numeric(predictions["query_budget"], errors="raise").eq(0)
    ].copy()
    subset["sequence_row"] = pd.to_numeric(
        subset["sequence_row_internal_audit_only"], errors="raise"
    ).astype(int)
    subset = subset.merge(
        metadata[["sequence_row", "participant", "session", "label", "repetition"]],
        on="sequence_row",
        how="left",
        suffixes=("_prediction", "_metadata"),
        validate="many_to_one",
        indicator=True,
    )
    if not subset["_merge"].eq("both").all():
        raise RuntimeError("R5B prediction-to-metadata join is incomplete")
    for predicted_column, metadata_column in [
        ("participant_prediction", "participant_metadata"),
        ("target_session", "session"),
        ("true_label", "label"),
    ]:
        if predicted_column in subset and not (
            subset[predicted_column].astype(str) == subset[metadata_column].astype(str)
        ).all():
            raise RuntimeError(f"R5B prediction metadata mismatch: {predicted_column}")
    subset["participant"] = subset["participant_metadata"].astype(str)
    subset["target_session"] = pd.to_numeric(subset["target_session"], errors="raise").astype(int)
    subset["true_label"] = pd.to_numeric(subset["true_label"], errors="raise").astype(int)
    subset["predicted_label"] = pd.to_numeric(subset["predicted_label"], errors="raise").astype(int)
    subset["period"] = subset["split_id"].map(chosen_splits)
    subset["correct_recomputed"] = subset["true_label"].eq(subset["predicted_label"])
    expected_period = np.where(subset["repetition"].le(5), "EARLY", "LATE")
    if len(subset) != 2450 or not np.array_equal(subset["period"].to_numpy(), expected_period):
        raise RuntimeError("Early/late no-adaptation prediction extraction failed")
    keys = ["participant", "target_session", "label", "repetition"]
    if subset.duplicated(keys).any():
        raise RuntimeError("Duplicate no-adaptation repetition prediction")

    period_rows = []
    for key, group in subset.groupby(["participant", "target_session", "period"], sort=True):
        participant, session, period = key
        class_support = group.groupby("true_label").size()
        if len(group) != 35 or not class_support.reindex(LABELS, fill_value=0).eq(5).all():
            raise RuntimeError("Early/late performance cell is not five-per-class balanced")
        accuracy = float(group["correct_recomputed"].mean())
        period_rows.append(
            {
                "participant": participant,
                "target_session": int(session),
                "period": period,
                "repetition_count": len(group),
                "accuracy": accuracy,
                "balanced_accuracy": accuracy,
            }
        )
    period_metrics = pd.DataFrame(period_rows)
    wide = period_metrics.pivot(
        index=["participant", "target_session"], columns="period", values="accuracy"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"EARLY": "early_accuracy", "LATE": "late_accuracy"})
    wide["late_minus_early_accuracy"] = wide["late_accuracy"] - wide["early_accuracy"]
    wide["negative_indicates_worsening"] = True
    wide["case_analysis"] = wide["participant"].eq("P07")

    chosen_folds = folds.loc[
        folds["strategy"].astype(str).eq("NO_ADAPTATION_REFERENCE")
        & folds["split_id"].astype(str).isin(chosen_splits)
        & pd.to_numeric(folds["query_budget"], errors="raise").eq(0)
    ].copy()
    chosen_folds["period"] = chosen_folds["split_id"].map(chosen_splits)
    fold_audit = period_metrics.merge(
        chosen_folds[["participant", "target_session", "period", "repetition_accuracy"]],
        on=["participant", "target_session", "period"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    fold_audit["absolute_accuracy_difference"] = (
        fold_audit["accuracy"] - fold_audit["repetition_accuracy"]
    ).abs()
    return subset, wide, fold_audit


def repetition_embeddings(
    features: np.ndarray,
    main_valid: np.ndarray,
    rows: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, dtype=int)
    raw = np.asarray(features[rows], dtype=np.float32)
    valid = np.asarray(main_valid[rows], dtype=bool)
    safe_stds = np.maximum(np.asarray(stds, dtype=np.float32), NORMALIZATION_EPSILON)
    transformed = (
        np.log1p(raw) - np.asarray(means, dtype=np.float32)[None, None, :]
    ) / safe_stds[None, None, :]
    if not np.isfinite(transformed).all():
        raise RuntimeError("Non-finite normalized RMS value")
    counts = valid.sum(axis=1, dtype=np.int64)
    sums = np.where(valid, transformed, np.float32(0.0)).sum(axis=1, dtype=np.float32)
    embeddings = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=embeddings, where=counts > 0)
    return embeddings, counts


def compute_feature_drift(
    features: np.ndarray,
    main_valid: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    embedding_frames = []
    centroid_rows = []
    distance_rows = []
    normalizer_rows = []
    channel_columns = [f"channel_{channel:02d}" for channel in range(64)]

    for participant in PARTICIPANTS:
        source_mask = (
            metadata["participant"].eq(participant)
            & metadata["session"].eq(0)
            & metadata["repetition"].isin([1, 2, 3, 4, 5])
        )
        source_rows = metadata.index[source_mask].to_numpy(dtype=int)
        target_mask = metadata["participant"].eq(participant) & metadata["session"].isin(TARGET_SESSIONS)
        target_rows = metadata.index[target_mask].to_numpy(dtype=int)
        if len(source_rows) != 35 or len(target_rows) != 350:
            raise RuntimeError(f"Feature-drift row contract failed for {participant}")
        means, stds, counts = engine.fit_normalizer(
            features[source_rows], main_valid[source_rows]
        )
        safe_stds = np.maximum(stds, NORMALIZATION_EPSILON)
        source_embeddings, source_counts = repetition_embeddings(
            features, main_valid, source_rows, means, safe_stds
        )
        target_embeddings, target_counts = repetition_embeddings(
            features, main_valid, target_rows, means, safe_stds
        )
        source_meta = metadata.iloc[source_rows].reset_index(drop=True)
        target_meta = metadata.iloc[target_rows].reset_index(drop=True)

        all_embeddings = np.vstack([source_embeddings, target_embeddings])
        all_counts = np.vstack([source_counts, target_counts])
        all_meta = pd.concat([source_meta, target_meta], ignore_index=True)
        embedding_frame = all_meta[["sequence_row", "participant", "session", "label", "repetition"]].copy()
        embedding_frame["analysis_role"] = np.where(
            embedding_frame["session"].eq(0), "SOURCE_REFERENCE", "TARGET_DRIFT"
        )
        embedding_frame["valid_channel_count"] = (all_counts > 0).sum(axis=1)
        embedding_frame[channel_columns] = all_embeddings
        embedding_frames.append(embedding_frame)

        centroids = {}
        for label in LABELS:
            label_mask = source_meta["label"].eq(label).to_numpy()
            centroid = np.nanmean(source_embeddings[label_mask], axis=0).astype(np.float32)
            if not np.isfinite(centroid).any():
                raise RuntimeError(f"Empty source centroid for {participant} label {label}")
            centroids[label] = centroid
            for channel, value in enumerate(centroid):
                centroid_rows.append(
                    {
                        "participant": participant,
                        "class_label": label,
                        "channel": channel,
                        "source_repetitions": int(label_mask.sum()),
                        "centroid_value": float(value) if np.isfinite(value) else np.nan,
                        "channel_has_reference": bool(np.isfinite(value)),
                    }
                )
        for index, row in target_meta.iterrows():
            embedding = target_embeddings[index]
            centroid = centroids[int(row["label"])]
            joint = np.isfinite(embedding) & np.isfinite(centroid)
            if not joint.any():
                raise RuntimeError("No jointly valid channels for feature distance")
            difference = embedding[joint].astype(np.float64) - centroid[joint].astype(np.float64)
            distance_rows.append(
                {
                    "sequence_row": int(row["sequence_row"]),
                    "participant": participant,
                    "target_session": int(row["session"]),
                    "true_label": int(row["label"]),
                    "repetition_order": int(row["repetition"]),
                    "joint_valid_channels": int(joint.sum()),
                    "class_conditioned_rms_distance": float(np.sqrt(np.mean(np.square(difference)))),
                }
            )
        normalizer_rows.append(
            {
                "participant": participant,
                "history_repetitions": len(source_rows),
                "minimum_mean": float(means.min()),
                "maximum_mean": float(means.max()),
                "minimum_std_before_epsilon": float(stds.min()),
                "minimum_std_after_epsilon": float(safe_stds.min()),
                "epsilon": float(NORMALIZATION_EPSILON),
                "minimum_valid_count": int(counts.min()),
                "means_dtype": str(means.dtype),
                "stds_dtype": str(stds.dtype),
            }
        )

    embeddings = pd.concat(embedding_frames, ignore_index=True)
    centroids = pd.DataFrame(centroid_rows)
    distances = pd.DataFrame(distance_rows)
    normalizers = pd.DataFrame(normalizer_rows)
    order_distance = (
        distances.groupby(["participant", "target_session", "repetition_order"], as_index=False)
        .agg(
            class_count=("true_label", "nunique"),
            mean_class_conditioned_rms_distance=("class_conditioned_rms_distance", "mean"),
            minimum_joint_valid_channels=("joint_valid_channels", "min"),
        )
        .sort_values(["participant", "target_session", "repetition_order"], kind="mergesort")
        .reset_index(drop=True)
    )
    slope_rows = []
    for (participant, session), group in order_distance.groupby(
        ["participant", "target_session"], sort=True
    ):
        group = group.sort_values("repetition_order")
        x = group["repetition_order"].to_numpy(dtype=np.float64)
        y = group["mean_class_conditioned_rms_distance"].to_numpy(dtype=np.float64)
        if len(group) != 10 or not group["class_count"].eq(7).all() or not np.isfinite(y).all():
            raise RuntimeError("Feature-drift order-distance cell is incomplete")
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        residual = y - fitted
        slope_rows.append(
            {
                "participant": participant,
                "target_session": int(session),
                "feature_drift_slope": float(slope),
                "ols_intercept": float(intercept),
                "r_squared": float(1.0 - np.square(residual).sum() / np.square(y - y.mean()).sum())
                if np.square(y - y.mean()).sum() > 0
                else 0.0,
                "positive_indicates_increasing_displacement": True,
                "case_analysis": participant == "P07",
            }
        )
    slopes = pd.DataFrame(slope_rows)
    return embeddings, centroids, distances, order_distance, slopes, normalizers


def build_estimands(
    performance_drift: pd.DataFrame,
    feature_slopes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_participant = (
        performance_drift.groupby("participant", as_index=False)["late_minus_early_accuracy"]
        .mean()
        .rename(columns={"late_minus_early_accuracy": "mean_session_late_minus_early_accuracy"})
    )
    feature_participant = (
        feature_slopes.groupby("participant", as_index=False)["feature_drift_slope"]
        .mean()
        .rename(columns={"feature_drift_slope": "mean_session_feature_drift_slope"})
    )
    estimands = performance_participant.merge(feature_participant, on="participant", validate="one_to_one")
    estimands["case_analysis"] = estimands["participant"].eq("P07")
    estimands["scientific_role"] = np.where(
        estimands["case_analysis"],
        "DESCRIPTIVE_CASE_ONLY",
        "PARTICIPANT_LEVEL_ESTIMANDS_ONLY_NO_INFERENCE_UNTIL_R7",
    )
    able = estimands.loc[estimands["participant"].isin(ABLE_BODIED)]
    descriptive = pd.DataFrame(
        [
            {
                "population": "P01-P06",
                "participants": len(able),
                "mean_late_minus_early_accuracy": float(able["mean_session_late_minus_early_accuracy"].mean()),
                "sd_late_minus_early_accuracy": float(able["mean_session_late_minus_early_accuracy"].std(ddof=1)),
                "mean_feature_drift_slope": float(able["mean_session_feature_drift_slope"].mean()),
                "sd_feature_drift_slope": float(able["mean_session_feature_drift_slope"].std(ddof=1)),
                "inferential_test_run": False,
            }
        ]
    )
    p07 = estimands.loc[estimands["participant"].eq("P07")].copy()
    return estimands, descriptive, p07


def synthetic_drift_unit_tests() -> pd.DataFrame:
    truth = np.tile(np.asarray(LABELS, dtype=int), 5)
    early_pred = truth.copy()
    late_pred = truth.copy()
    late_pred[:7] = (late_pred[:7] + 1) % 7
    performance_delta = float(np.mean(late_pred == truth) - np.mean(early_pred == truth))
    order = np.arange(1, 11, dtype=np.float64)
    distance = 0.25 * order + 0.4
    slope = float(np.polyfit(order, distance, 1)[0])
    audit = pd.DataFrame(
        [
            {"test": "KNOWN_LATE_MINUS_EARLY_ACCURACY", "observed": performance_delta, "expected": -0.2},
            {"test": "KNOWN_POSITIVE_FEATURE_SLOPE", "observed": slope, "expected": 0.25},
        ]
    )
    audit["absolute_difference"] = (audit["observed"] - audit["expected"]).abs()
    audit["passes"] = audit["absolute_difference"].lt(1e-12)
    return audit


def output_manifest(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "revision_R5C_output_manifest.csv":
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": engine.sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 104)
    print("REVISION R5C — WITHIN-SESSION DRIFT AUDIT")
    print("=" * 104)
    print("Execution device: CPU")
    print("Raw HDF5 accessed: False")
    print("Stage 5B feature-array payload read: True")
    print("Model training: False")
    print("New fixed-test inference: False")
    print("New inferential statistical test: False")
    print("Purpose: locked descriptive performance and feature drift estimands for R7")
    print()
    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified R5A, R5B, and Stage 5B packets...")
    inputs = prepare_inputs()
    features = np.load(
        INPUT_ROOT / "stage5b_rms_repetition_sequences.npy", mmap_mode="r", allow_pickle=False
    )
    main_valid = np.load(
        INPUT_ROOT / "stage5b_main_valid_repetition_sequences.npy", mmap_mode="r", allow_pickle=False
    )
    if features.shape != (2940, 37, 64) or main_valid.shape != features.shape:
        raise RuntimeError("Stage 5B feature/mask shape contract failed")

    predictions, performance_drift, performance_fold_audit = extract_no_adaptation_performance(
        inputs["predictions"], inputs["folds"], inputs["metadata"]
    )
    (
        embeddings,
        centroids,
        distances,
        order_distance,
        feature_slopes,
        normalizers,
    ) = compute_feature_drift(features, main_valid, inputs["metadata"])
    estimands, descriptive, p07 = build_estimands(performance_drift, feature_slopes)
    synthetic = synthetic_drift_unit_tests()

    spec = inputs["drift_spec"]
    gates = {
        "revision_protocol_hash_matches": inputs["r5a_report"].get("revision_protocol_sha256") == REVISION_PROTOCOL_SHA256,
        "r5a_parent_all_gates_passed": bool(inputs["r5a_report"].get("all_readiness_gates_passed")),
        "r5b_parent_all_gates_passed": bool(inputs["r5b_report"].get("all_readiness_gates_passed")),
        "r5b_parent_decision_authorizes_r5c": inputs["r5b_report"].get("final_decision") == "PASS_TO_REVISION_R5C_WITHIN_SESSION_DRIFT_AUDIT",
        "all_three_input_packets_pass_crc_and_hash": len(inputs["audit"]) == 3 and inputs["audit"][["hash_matches", "crc_passes"]].all().all(),
        "embedded_r3a_engine_source_hash_is_exact": engine.sha256_file(Path(engine.__file__)) == R3A_ENGINE_SOURCE_SHA256,
        "locked_drift_specification_has_fourteen_components": len(spec) == 14,
        "performance_definition_is_late_minus_early": spec.loc[spec["component"].eq("PERFORMANCE_DRIFT"), "locked_implementation"].str.contains("late accuracy minus early accuracy", case=False).all(),
        "feature_definition_is_order_distance_ols_slope": spec.loc[spec["component"].eq("FEATURE_DRIFT_SLOPE"), "locked_implementation"].str.contains("OLS slope", case=False).all(),
        "normalization_epsilon_is_1e_minus_8": float(NORMALIZATION_EPSILON) == float(np.float32(1e-8)),
        "performance_prediction_rows_are_2450": len(predictions) == 2450,
        "performance_participant_session_rows_are_35": len(performance_drift) == 35,
        "performance_early_and_late_values_are_finite_in_range": np.isfinite(performance_drift[["early_accuracy", "late_accuracy", "late_minus_early_accuracy"]].to_numpy(float)).all() and performance_drift[["early_accuracy", "late_accuracy"]].ge(0).all().all() and performance_drift[["early_accuracy", "late_accuracy"]].le(1).all().all(),
        "performance_recomputation_matches_r5b_fold_metrics": len(performance_fold_audit) == 70 and performance_fold_audit["_merge"].eq("both").all() and performance_fold_audit["absolute_accuracy_difference"].max() < 1e-12,
        "embedding_rows_are_2695": len(embeddings) == 2695,
        "source_centroid_rows_are_3136": len(centroids) == 7 * 7 * 64,
        "feature_distance_rows_are_2450": len(distances) == 2450,
        "order_distance_rows_are_350": len(order_distance) == 350,
        "each_order_distance_averages_seven_classes": order_distance["class_count"].eq(7).all(),
        "feature_slope_rows_are_35": len(feature_slopes) == 35,
        "all_feature_distances_and_slopes_are_finite": np.isfinite(distances["class_conditioned_rms_distance"].to_numpy(float)).all() and np.isfinite(feature_slopes["feature_drift_slope"].to_numpy(float)).all(),
        "all_feature_distances_use_at_least_one_joint_channel": distances["joint_valid_channels"].gt(0).all(),
        "normalizer_rows_are_seven": len(normalizers) == 7,
        "all_normalizers_are_finite_positive_float32": np.isfinite(normalizers[["minimum_mean", "maximum_mean", "minimum_std_after_epsilon"]].to_numpy(float)).all() and normalizers["minimum_std_after_epsilon"].gt(0).all() and normalizers["minimum_valid_count"].gt(0).all() and normalizers["means_dtype"].eq("float32").all() and normalizers["stds_dtype"].eq("float32").all(),
        "participant_estimands_are_exactly_p01_to_p07": len(estimands) == 7 and set(estimands["participant"]) == set(PARTICIPANTS),
        "p01_to_p06_are_estimands_only_no_inference_until_r7": estimands.loc[estimands["participant"].isin(ABLE_BODIED), "scientific_role"].eq("PARTICIPANT_LEVEL_ESTIMANDS_ONLY_NO_INFERENCE_UNTIL_R7").all(),
        "p07_is_descriptive_case_only": len(p07) == 1 and p07["scientific_role"].eq("DESCRIPTIVE_CASE_ONLY").all(),
        "synthetic_drift_unit_tests_pass": synthetic["passes"].all(),
        "raw_hdf5_data_was_not_accessed": True,
        "no_model_was_trained": True,
        "no_new_fixed_test_inference_was_run": True,
        "no_new_inferential_statistical_test_was_run": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]

    atomic_csv(inputs["audit"], RESULT_ROOT / "revision_R5C_input_packet_audit.csv")
    atomic_csv(spec, RESULT_ROOT / "revision_R5C_locked_drift_specification.csv")
    atomic_csv(synthetic, RESULT_ROOT / "revision_R5C_synthetic_drift_unit_tests.csv")
    atomic_csv(predictions, RESULT_ROOT / "revision_R5C_no_adaptation_repetition_predictions.csv")
    atomic_csv(performance_drift, RESULT_ROOT / "revision_R5C_participant_session_performance_drift.csv")
    atomic_csv(performance_fold_audit, RESULT_ROOT / "revision_R5C_performance_fold_anchor_audit.csv")
    atomic_csv(embeddings, RESULT_ROOT / "revision_R5C_repetition_feature_embeddings.csv")
    atomic_csv(centroids, RESULT_ROOT / "revision_R5C_source_class_centroids_long.csv")
    atomic_csv(distances, RESULT_ROOT / "revision_R5C_class_conditioned_feature_distances.csv")
    atomic_csv(order_distance, RESULT_ROOT / "revision_R5C_order_distance.csv")
    atomic_csv(feature_slopes, RESULT_ROOT / "revision_R5C_participant_session_feature_slopes.csv")
    atomic_csv(normalizers, RESULT_ROOT / "revision_R5C_source_only_normalizer_audit.csv")
    atomic_csv(estimands, RESULT_ROOT / "revision_R5C_participant_level_estimands_for_R7.csv")
    atomic_csv(descriptive, RESULT_ROOT / "revision_R5C_able_bodied_descriptive_summary.csv")
    atomic_csv(p07, RESULT_ROOT / "revision_R5C_P07_descriptive_case.csv")
    report = {
        "stage": "REVISION_R5C_WITHIN_SESSION_DRIFT_AUDIT",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "r5a_packet_sha256": R5A_PACKET_SHA256,
        "r5b_packet_sha256": R5B_PACKET_SHA256,
        "stage5b_packet_sha256": STAGE5B_PACKET_SHA256,
        "performance_participant_session_rows": len(performance_drift),
        "feature_distance_rows": len(distances),
        "feature_slope_rows": len(feature_slopes),
        "participant_estimand_rows": len(estimands),
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "raw_hdf5_accessed": False,
        "feature_array_payload_read": True,
        "model_training_run": False,
        "new_fixed_test_inference_run": False,
        "new_inferential_statistical_test_run": False,
        "final_decision": "PASS_TO_REVISION_R6_DEEP_TRAINING_SEED_STABILITY" if not failed else "REVISION_R5C_FINAL_AUDIT_FAILED",
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
    }
    atomic_json(report, RESULT_ROOT / "revision_R5C_final_report.json")
    shutil.copy2(Path(__file__), RESULT_ROOT / "revision_R5C_executed_source.py")
    atomic_csv(output_manifest(RESULT_ROOT), RESULT_ROOT / "revision_R5C_output_manifest.csv")
    if failed:
        raise RuntimeError(f"Revision R5C failed readiness gates: {failed}")

    packet_crc = engine.make_zip(RESULT_ROOT, PACKET_PATH, "Revision_R5C_Within_Session_Drift_Audit")
    packet_sha = engine.sha256_file(PACKET_PATH)
    remote_verified = engine.roundtrip_remote_file(
        PACKET_PATH, REMOTE_OUTPUT + "/" + PACKET_PATH.name, packet_sha
    )
    engine.cleanup_secret()
    print()
    print("=" * 104)
    print("REVISION R5C — FINAL DESCRIPTIVE SUMMARY")
    print("=" * 104)
    print(descriptive.to_string(index=False))
    print()
    print("P07 descriptive case:")
    print(p07.to_string(index=False))
    print()
    print("Performance participant-session estimands:", len(performance_drift))
    print("Feature-distance rows:", len(distances))
    print("Feature-slope participant-session estimands:", len(feature_slopes))
    print("Failed readiness gates:", failed or "None")
    print("Packet CRC pass:", packet_crc)
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", packet_sha)
    print("Remote round-trip verified:", remote_verified)
    print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 3))
    if not packet_crc or not remote_verified:
        raise RuntimeError("R5C final packet persistence failed")
    print()
    print("FINAL DECISION: PASS_TO_REVISION_R6_DEEP_TRAINING_SEED_STABILITY")


if __name__ == "__main__":
    main()
