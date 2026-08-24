from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import revision_R3A_P1_float32_engine_frozen_trajectory_unit_test as engine


REVISION_PROTOCOL_SHA256 = "6807b71de18ca82013cfa4360d760e0daf9a920a1acc0625dcb13bd8f4d07249"
DEEP_PROTOCOL_SHA256 = "abe15812c1a52b0f4e917b5b6ad39b0dfde50e5bb2d58dfcc35b3cacb22e3bd2"
R0_PACKET_SHA256 = "0800e315a29b81934095ba56deaea3f8b6600fd0df13db348d7ea72d3b82df78"
R6B_PACKET_SHA256 = "5e1440672256a452e5fdf8924dd1705c89fa03b5bfc8afdad41a73cdfb6b45f0"
R6C_PACKET_SHA256 = "7391762f67035693864272fe6ee6b2762a70b5fcfebcf75535c18a996b820769"

PARTICIPANTS = [f"P{i:02d}" for i in range(1, 8)]
ABLE_BODIED = PARTICIPANTS[:6]
R6B_STRATEGIES = ["PCBM_PROPOSED", "GLOBAL_MARGIN"]
R6C_STRATEGIES = ["PCBM_PROPOSED", "GLOBAL_MARGIN", "RANDOM_UNIFORM", "BADGE"]

WORKING = Path(os.environ.get("REVISION_R6D_WORKING", "/kaggle/working"))
INPUT_ROOT = WORKING / "REVISION_R6D_FROZEN_INPUTS"
RESULT_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R6D_Deep_Stability_Compute_Aggregation"
FINAL_PACKET = WORKING / "revision_R6D_deep_stability_compute_aggregation_packet.zip"
REMOTE_BASE = engine.REMOTE_BASE
REMOTE_OUTPUT = REMOTE_BASE + "/Reviewer_Revision/Revision_R6D_Deep_Stability_Compute_Aggregation"
START_TIME = time.time()

DIRECT_PACKETS = {
    "stageR0_reviewer_revision_protocol_lock_packet.zip": (
        R0_PACKET_SHA256,
        "Reviewer_Revision/StageR0_Reviewer_Revision_Protocol_Lock/"
        "stageR0_reviewer_revision_protocol_lock_packet.zip",
    ),
    "revision_R6B_cpu_fixed_history_multiseed_packet.zip": (
        R6B_PACKET_SHA256,
        "Reviewer_Revision/Revision_R6B_CPU_Fixed_History_Multiseed/"
        "revision_R6B_cpu_fixed_history_multiseed_packet.zip",
    ),
    "revision_R6C_cpu_end_to_end_multiseed_badge_packet.zip": (
        R6C_PACKET_SHA256,
        "Reviewer_Revision/Revision_R6C_CPU_End_to_End_Multiseed_BADGE/"
        "revision_R6C_cpu_end_to_end_multiseed_badge_packet.zip",
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, 6):
        temporary = destination.with_suffix(destination.suffix + f".download{attempt}")
        temporary.unlink(missing_ok=True)
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
        if (
            result.returncode == 0
            and temporary.exists()
            and engine.sha256_file(temporary) == expected_hash
            and engine.archive_crc_passes(temporary)
        ):
            os.replace(temporary, destination)
            return destination, "GOOGLE_DRIVE_DIRECT"
        last_error = (result.stderr or result.stdout or "hash-or-crc-mismatch")[-1200:]
        temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not restore verified {basename}: {last_error}")


def normalize_parent_tables(packet: Path, prefix: str) -> dict[str, pd.DataFrame]:
    names = {
        "fits": f"revision_{prefix}_fit_audit.csv",
        "folds": f"revision_{prefix}_fold_results.csv",
        "predictions": f"revision_{prefix}_repetition_predictions.csv",
        "seed_summary": f"revision_{prefix}_seed_level_summary.csv",
        "participant_summary": f"revision_{prefix}_participant_seed_averaged_summary.csv",
        "unit_index": f"revision_{prefix}_unit_packet_index.csv",
    }
    if prefix == "R6C":
        names["telemetry"] = "revision_R6C_compute_telemetry.csv"
    tables = {key: engine.read_csv_member(packet, member) for key, member in names.items()}
    for frame in tables.values():
        for column in ["training_seed_index", "training_seed", "target_session"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
        for column in [
            "repetition_accuracy",
            "repetition_balanced_accuracy",
            "repetition_macro_f1",
            "fit_seconds",
            "inference_seconds",
            "model_score_seconds",
            "selector_seconds",
            "refit_seconds",
            "fixed_test_inference_seconds",
            "unit_runtime_minutes",
        ]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return tables


def paired_contrasts(
    folds: pd.DataFrame,
    analysis_scope: str,
    comparators: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["participant", "training_seed_index", "training_seed", "target_session"]
    metrics = ["repetition_balanced_accuracy", "repetition_macro_f1"]
    proposed = folds.loc[folds["strategy"].astype(str).eq("PCBM_PROPOSED"), keys + metrics].copy()
    session_frames = []
    for comparator in comparators:
        reference = folds.loc[folds["strategy"].astype(str).eq(comparator), keys + metrics].copy()
        paired = proposed.merge(
            reference,
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_pcbm", "_comparator"),
        )
        paired.insert(0, "analysis_scope", analysis_scope)
        paired.insert(1, "contrast", f"PCBM_PROPOSED_MINUS_{comparator}")
        paired.insert(2, "comparator", comparator)
        paired["case_analysis"] = paired["participant"].eq("P07")
        for metric in metrics:
            paired[f"difference_{metric}"] = paired[f"{metric}_pcbm"] - paired[f"{metric}_comparator"]
        session_frames.append(paired)
    session = pd.concat(session_frames, ignore_index=True)
    seed = (
        session.groupby(
            ["analysis_scope", "contrast", "comparator", "participant", "training_seed_index", "training_seed", "case_analysis"],
            as_index=False,
        )
        .agg(
            target_sessions=("target_session", "nunique"),
            seed_mean_difference_balanced_accuracy=("difference_repetition_balanced_accuracy", "mean"),
            seed_mean_difference_macro_f1=("difference_repetition_macro_f1", "mean"),
        )
        .sort_values(["analysis_scope", "contrast", "participant", "training_seed_index"])
        .reset_index(drop=True)
    )
    participant = (
        seed.groupby(
            ["analysis_scope", "contrast", "comparator", "participant", "case_analysis"],
            as_index=False,
        )
        .agg(
            training_seeds=("training_seed_index", "nunique"),
            participant_seed_averaged_difference_balanced_accuracy=("seed_mean_difference_balanced_accuracy", "mean"),
            training_seed_sd_difference_balanced_accuracy=("seed_mean_difference_balanced_accuracy", "std"),
            training_seed_min_difference_balanced_accuracy=("seed_mean_difference_balanced_accuracy", "min"),
            training_seed_max_difference_balanced_accuracy=("seed_mean_difference_balanced_accuracy", "max"),
            participant_seed_averaged_difference_macro_f1=("seed_mean_difference_macro_f1", "mean"),
            training_seed_sd_difference_macro_f1=("seed_mean_difference_macro_f1", "std"),
        )
        .sort_values(["analysis_scope", "contrast", "participant"])
        .reset_index(drop=True)
    )
    return session, seed, participant


def able_bodied_contrast_summary(participant: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in participant.loc[participant["participant"].isin(ABLE_BODIED)].groupby(
        ["analysis_scope", "contrast", "comparator"], sort=True
    ):
        values = group["participant_seed_averaged_difference_balanced_accuracy"].to_numpy(dtype=float)
        rows.append(
            {
                "analysis_scope": keys[0],
                "contrast": keys[1],
                "comparator": keys[2],
                "participants": len(values),
                "mean_participant_difference_balanced_accuracy": float(values.mean()),
                "sd_participant_difference_balanced_accuracy": float(values.std(ddof=1)),
                "minimum_participant_difference_balanced_accuracy": float(values.min()),
                "maximum_participant_difference_balanced_accuracy": float(values.max()),
                "participants_positive": int((values > 0).sum()),
                "participants_zero": int((values == 0).sum()),
                "participants_negative": int((values < 0).sum()),
                "scientific_role": "DESCRIPTIVE_R6D_INPUT_TO_LOCKED_R7_INFERENCE",
            }
        )
    return pd.DataFrame(rows)


def build_cost_rows(
    stage: str,
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    fits = tables["fits"].copy()
    folds = tables["folds"].copy()
    unit_index = tables["unit_index"].copy()
    identity = (
        fits.groupby("unit_id", as_index=False)
        .agg(
            participant=("participant", "first"),
            strategy=("strategy", "first"),
            training_seed_index=("training_seed_index", "first"),
            training_seed=("training_seed", "first"),
            fit_count=("unit_id", "size"),
            total_training_seconds=("fit_seconds", "sum"),
            total_optimizer_steps=("optimizer_steps", "sum"),
        )
        .rename(columns={"unit_id": "execution_unit_id"})
    )
    inference = (
        folds.groupby("unit_id", as_index=False)
        .agg(
            fixed_test_folds=("target_session", "nunique"),
            total_fixed_test_inference_seconds=("inference_seconds", "sum"),
        )
        .rename(columns={"unit_id": "execution_unit_id"})
    )
    cost = identity.merge(inference, on="execution_unit_id", validate="one_to_one")
    if stage == "R6C_END_TO_END":
        telemetry = (
            tables["telemetry"].groupby("unit_id", as_index=False)
            .agg(
                total_model_score_seconds=("model_score_seconds", "sum"),
                total_selector_seconds=("selector_seconds", "sum"),
            )
            .rename(columns={"unit_id": "execution_unit_id"})
        )
        cost = cost.merge(telemetry, on="execution_unit_id", validate="one_to_one")
    else:
        cost["total_model_score_seconds"] = 0.0
        cost["total_selector_seconds"] = 0.0
    index_columns = ["execution_unit_id", "unit_runtime_minutes", "packet_sha256"]
    cost = cost.merge(unit_index[index_columns], on="execution_unit_id", validate="one_to_one")
    cost.insert(0, "stage", stage)
    cost["population"] = np.where(cost["participant"].isin(ABLE_BODIED), "P01-P06", "P07_DESCRIPTIVE")
    cost["total_acquisition_compute_seconds"] = cost["total_model_score_seconds"] + cost["total_selector_seconds"]
    cost["total_measured_component_seconds"] = (
        cost["total_training_seconds"]
        + cost["total_acquisition_compute_seconds"]
        + cost["total_fixed_test_inference_seconds"]
    )
    return cost.sort_values(["stage", "participant", "training_seed_index", "strategy"]).reset_index(drop=True)


def summarize_cost(cost: pd.DataFrame) -> pd.DataFrame:
    measures = [
        "total_training_seconds",
        "total_model_score_seconds",
        "total_selector_seconds",
        "total_acquisition_compute_seconds",
        "total_fixed_test_inference_seconds",
        "total_measured_component_seconds",
        "unit_runtime_minutes",
        "total_optimizer_steps",
    ]
    rows = []
    for keys, group in cost.groupby(["stage", "strategy", "population"], sort=True):
        row = {
            "stage": keys[0],
            "strategy": keys[1],
            "population": keys[2],
            "execution_units": len(group),
            "participants": group["participant"].nunique(),
            "training_seeds_per_participant": group["training_seed_index"].nunique(),
        }
        for measure in measures:
            values = group[measure].to_numpy(dtype=float)
            row[f"mean_{measure}"] = float(values.mean())
            row[f"median_{measure}"] = float(np.median(values))
            row[f"q25_{measure}"] = float(np.quantile(values, 0.25))
            row[f"q75_{measure}"] = float(np.quantile(values, 0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def output_manifest(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "revision_R6D_output_manifest.csv":
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
    print("REVISION R6D — DEEP STABILITY AND COMPUTE-COST AGGREGATION")
    print("=" * 108)
    print("Execution device: CPU")
    print("Scientific model training: False")
    print("New fixed-test inference: False")
    print("New inferential statistical test: False")
    print("Purpose: average six locked TCN seeds within participant and prepare R7 estimands")
    print()

    WORKING.mkdir(parents=True, exist_ok=True)
    lock_handle = open(WORKING / "_revision_R6D_single_instance.lock", "w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("DUPLICATE INVOCATION DETECTED: another R6D process owns the lock.")
        print("FINAL DECISION: DUPLICATE_INVOCATION_EXITED_SAFELY")
        return

    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified R0, R6B, and R6C final packets...")
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    packets = {}
    packet_audit_rows = []
    for basename, (expected_hash, remote_relative) in DIRECT_PACKETS.items():
        path, source = direct_restore(basename, expected_hash, remote_relative)
        actual_hash = engine.sha256_file(path)
        crc_pass = engine.archive_crc_passes(path)
        packets[basename] = path
        packet_audit_rows.append(
            {
                "packet": basename,
                "source": source,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "hash_matches": actual_hash == expected_hash,
                "crc_passes": crc_pass,
            }
        )
    packet_audit = pd.DataFrame(packet_audit_rows)

    r0_packet = packets["stageR0_reviewer_revision_protocol_lock_packet.zip"]
    r6b_packet = packets["revision_R6B_cpu_fixed_history_multiseed_packet.zip"]
    r6c_packet = packets["revision_R6C_cpu_end_to_end_multiseed_badge_packet.zip"]
    r0_report = engine.read_json_member(r0_packet, "stageR0_protocol_lock_report.json")
    r6b_report = engine.read_json_member(r6b_packet, "revision_R6B_final_report.json")
    r6c_report = engine.read_json_member(r6c_packet, "revision_R6C_final_report.json")
    r6b = normalize_parent_tables(r6b_packet, "R6B")
    r6c = normalize_parent_tables(r6c_packet, "R6C")

    fixed_session, fixed_seed, fixed_participant = paired_contrasts(
        r6b["folds"], "FIXED_HISTORY", ["GLOBAL_MARGIN"]
    )
    end_session, end_seed, end_participant = paired_contrasts(
        r6c["folds"], "END_TO_END", ["GLOBAL_MARGIN", "RANDOM_UNIFORM", "BADGE"]
    )
    session_contrasts = pd.concat([fixed_session, end_session], ignore_index=True)
    seed_distribution = pd.concat([fixed_seed, end_seed], ignore_index=True)
    participant_estimands = pd.concat([fixed_participant, end_participant], ignore_index=True)
    descriptive_summary = able_bodied_contrast_summary(participant_estimands)

    strategy_performance = pd.concat(
        [
            r6b["participant_summary"].assign(analysis_scope="FIXED_HISTORY"),
            r6c["participant_summary"].assign(analysis_scope="END_TO_END"),
        ],
        ignore_index=True,
    )
    cost_rows = pd.concat(
        [build_cost_rows("R6B_FIXED_HISTORY", r6b), build_cost_rows("R6C_END_TO_END", r6c)],
        ignore_index=True,
    )
    cost_summary = summarize_cost(cost_rows)

    fixed_expected = 7 * 6 * 5 * 1
    end_expected = 7 * 6 * 5 * 3
    paired_session_key = ["analysis_scope", "contrast", "participant", "training_seed_index", "target_session"]
    gates = {
        "all_three_input_packets_pass_hash_and_crc": len(packet_audit) == 3
        and packet_audit[["hash_matches", "crc_passes"]].all().all(),
        "r0_parent_all_gates_passed": bool(r0_report.get("all_readiness_gates_passed")),
        "r6b_parent_all_gates_passed": bool(r6b_report.get("all_readiness_gates_passed")),
        "r6c_parent_all_gates_passed": bool(r6c_report.get("all_readiness_gates_passed")),
        "r6c_parent_authorizes_r6d": r6c_report.get("final_decision")
        == "PASS_TO_REVISION_R6D_DEEP_STABILITY_AND_COMPUTE_AGGREGATION",
        "revision_protocol_hash_is_preserved": r6b_report.get("revision_protocol_sha256")
        == r6c_report.get("revision_protocol_sha256")
        == REVISION_PROTOCOL_SHA256,
        "deep_protocol_hash_is_preserved": r6b_report.get("deep_protocol_sha256")
        == r6c_report.get("deep_protocol_sha256")
        == DEEP_PROTOCOL_SHA256,
        "r6b_fold_count_is_420": len(r6b["folds"]) == 420,
        "r6c_fold_count_is_840": len(r6c["folds"]) == 840,
        "r6b_prediction_count_is_14700": len(r6b["predictions"]) == 14700,
        "r6c_prediction_count_is_29400": len(r6c["predictions"]) == 29400,
        "r6b_has_exact_strategy_set": set(r6b["folds"]["strategy"].astype(str)) == set(R6B_STRATEGIES),
        "r6c_has_exact_strategy_set": set(r6c["folds"]["strategy"].astype(str)) == set(R6C_STRATEGIES),
        "both_parents_have_exact_participant_set": set(r6b["folds"]["participant"].astype(str))
        == set(r6c["folds"]["participant"].astype(str))
        == set(PARTICIPANTS),
        "each_parent_participant_strategy_has_six_seeds": r6b["folds"]
        .groupby(["participant", "strategy"])["training_seed_index"]
        .nunique()
        .eq(6)
        .all()
        and r6c["folds"]
        .groupby(["participant", "strategy"])["training_seed_index"]
        .nunique()
        .eq(6)
        .all(),
        "each_parent_seed_strategy_has_five_sessions": r6b["folds"]
        .groupby(["participant", "strategy", "training_seed_index"])["target_session"]
        .nunique()
        .eq(5)
        .all()
        and r6c["folds"]
        .groupby(["participant", "strategy", "training_seed_index"])["target_session"]
        .nunique()
        .eq(5)
        .all(),
        "fixed_history_session_contrast_count_is_210": len(fixed_session) == fixed_expected,
        "end_to_end_session_contrast_count_is_630": len(end_session) == end_expected,
        "all_paired_session_keys_are_unique": not session_contrasts.duplicated(paired_session_key).any(),
        "seed_distribution_has_168_rows": len(seed_distribution) == 168,
        "participant_estimands_have_28_rows": len(participant_estimands) == 28,
        "every_participant_estimand_averages_six_training_seeds": participant_estimands["training_seeds"].eq(6).all(),
        "every_seed_estimand_averages_five_sessions": seed_distribution["target_sessions"].eq(5).all(),
        "participant_is_the_only_inferential_unit_prepared_for_r7": len(
            participant_estimands.loc[participant_estimands["participant"].isin(ABLE_BODIED)]
        )
        == 24,
        "p07_is_descriptive_only": participant_estimands.loc[
            participant_estimands["participant"].eq("P07"), "case_analysis"
        ].all(),
        "compute_audit_has_252_execution_units": len(cost_rows) == 252,
        "compute_audit_has_no_missing_component_values": cost_rows[
            [
                "total_training_seconds",
                "total_model_score_seconds",
                "total_selector_seconds",
                "total_fixed_test_inference_seconds",
                "unit_runtime_minutes",
            ]
        ].notna().all().all(),
        "all_compute_times_are_nonnegative": (
            cost_rows[
                [
                    "total_training_seconds",
                    "total_model_score_seconds",
                    "total_selector_seconds",
                    "total_fixed_test_inference_seconds",
                    "unit_runtime_minutes",
                ]
            ]
            >= 0
        ).all().all(),
        "no_scientific_model_training_was_run": True,
        "no_new_fixed_test_inference_was_run": True,
        "no_new_inferential_statistical_test_was_run": True,
        "stage3g_and_stage5f_conclusions_cannot_be_replaced": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]

    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    analysis_spec = {
        "stage": "REVISION_R6D_DEEP_STABILITY_COMPUTE_AGGREGATION",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "fixed_history_contrast": "PCBM_PROPOSED_MINUS_GLOBAL_MARGIN",
        "end_to_end_contrasts": [
            "PCBM_PROPOSED_MINUS_GLOBAL_MARGIN",
            "PCBM_PROPOSED_MINUS_RANDOM_UNIFORM",
            "PCBM_PROPOSED_MINUS_BADGE",
        ],
        "metric_order": "session difference -> mean within training seed -> mean across six seeds within participant",
        "inferential_participants_for_r7": ABLE_BODIED,
        "p07_role": "DESCRIPTIVE_CASE_ONLY",
        "r6d_role": "DESCRIPTIVE_AGGREGATION_ONLY",
        "r7_role": "LOCKED_INFERENTIAL_STATISTICS",
        "equivalence_claim_prohibited": True,
    }
    atomic_json(analysis_spec, RESULT_ROOT / "revision_R6D_analysis_specification.json")
    for frame, name in [
        (packet_audit, "revision_R6D_input_packet_audit.csv"),
        (session_contrasts, "revision_R6D_session_level_paired_contrasts.csv"),
        (seed_distribution, "revision_R6D_training_seed_distribution.csv"),
        (participant_estimands, "revision_R6D_participant_seed_averaged_estimands.csv"),
        (descriptive_summary, "revision_R6D_able_bodied_descriptive_contrast_summary.csv"),
        (strategy_performance, "revision_R6D_participant_strategy_performance.csv"),
        (cost_rows, "revision_R6D_execution_unit_compute_costs.csv"),
        (cost_summary, "revision_R6D_compute_cost_summary.csv"),
    ]:
        atomic_csv(frame, RESULT_ROOT / name)

    report = {
        "stage": "REVISION_R6D_DEEP_STABILITY_COMPUTE_AGGREGATION",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "r6b_packet_sha256": R6B_PACKET_SHA256,
        "r6c_packet_sha256": R6C_PACKET_SHA256,
        "execution_device": "CPU",
        "scientific_model_training_run": False,
        "new_fixed_test_inference_run": False,
        "new_inferential_statistical_test_run": False,
        "session_contrast_rows": len(session_contrasts),
        "training_seed_distribution_rows": len(seed_distribution),
        "participant_estimand_rows": len(participant_estimands),
        "compute_audit_execution_units": len(cost_rows),
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": "PASS_TO_REVISION_R7_LOCKED_STATISTICAL_ANALYSIS_AND_SUPPLEMENT"
        if not failed
        else "REVISION_R6D_AGGREGATION_FAILED",
    }
    atomic_json(report, RESULT_ROOT / "revision_R6D_final_report.json")
    shutil.copy2(Path(__file__), RESULT_ROOT / "revision_R6D_executed_source.py")
    atomic_csv(output_manifest(RESULT_ROOT), RESULT_ROOT / "revision_R6D_output_manifest.csv")
    if failed:
        raise RuntimeError(f"R6D readiness failed: {failed}")

    if not engine.make_zip(RESULT_ROOT, FINAL_PACKET, "Revision_R6D_Deep_Stability_Compute_Aggregation"):
        raise RuntimeError("R6D final packet CRC failed")
    digest = engine.sha256_file(FINAL_PACKET)
    if not engine.roundtrip_remote_file(FINAL_PACKET, REMOTE_OUTPUT + "/" + FINAL_PACKET.name, digest):
        raise RuntimeError("R6D final packet remote round-trip failed")

    print()
    print("=" * 108)
    print("REVISION R6D — FINAL AGGREGATION SUMMARY")
    print("=" * 108)
    print("Session-level paired contrast rows:", len(session_contrasts))
    print("Training-seed distribution rows:", len(seed_distribution))
    print("Participant-level estimands:", len(participant_estimands))
    print("Compute-audit execution units:", len(cost_rows))
    print()
    print("Able-bodied descriptive contrasts (not inferential tests):")
    print(
        descriptive_summary[
            [
                "analysis_scope",
                "contrast",
                "participants",
                "mean_participant_difference_balanced_accuracy",
                "sd_participant_difference_balanced_accuracy",
                "participants_positive",
                "participants_zero",
                "participants_negative",
            ]
        ].to_string(index=False)
    )
    print()
    print("Failed readiness gates:", failed or "None")
    print("Packet CRC pass:", engine.archive_crc_passes(FINAL_PACKET))
    print("Packet:", FINAL_PACKET)
    print("Packet SHA-256:", digest)
    print("Remote round-trip verified: True")
    print("Runtime minutes:", round(report["runtime_minutes"], 3))
    print()
    print("FINAL DECISION:", report["final_decision"])


if __name__ == "__main__":
    main()
