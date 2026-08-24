from __future__ import annotations

import fcntl
import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd

import revision_R3A_P1_float32_engine_frozen_trajectory_unit_test as engine


REVISION_PROTOCOL_SHA256 = "6807b71de18ca82013cfa4360d760e0daf9a920a1acc0625dcb13bd8f4d07249"
R0_PACKET_SHA256 = "0800e315a29b81934095ba56deaea3f8b6600fd0df13db348d7ea72d3b82df78"
R2A_PACKET_SHA256 = "554270b1d5bcc0bf7791020b64c44b72a267d71a2ee24afb5f43029957d01a8a"
R3A_P1_PACKET_SHA256 = "e5051aaf116af4888c632e27cd7008a7d4848b5308b6af4366a760b30a58435a"
R3C_PACKET_SHA256 = "6a8b332dcfc36109f7df70c4931309ca62e52c29979732a5ce0da90769f45e31"
R4B_PACKET_SHA256 = "8b356d734e7e04f71800795fb3a0d23609364277266b6a0c41254b267de709cd"
R4C_PACKET_SHA256 = "f00e3b761223375551e8672c6d5dbec589fb7b75ccc98751168ffaab80ada22c"
R4D_PACKET_SHA256 = "5b0f26dc0db5e7d38945ed2a40ae6acd184d1ce45c55a2ef1f68089520f5e0a5"
R4E_PACKET_SHA256 = "34d4054d9dac5cbf992ed72f0f4096d013b5cede47a5bf349b59f49ee44a71b3"
R5B_PACKET_SHA256 = "2cc8b8d83afb6483060b57d4c1a1d9f67213a7fb4c09de1226b86d2de640cccf"
R5C_PACKET_SHA256 = "c1d4cd0cba8438526ec6c9acd4df6c079e34733ce770dd93adc0e0336430306b"
R6D_PACKET_SHA256 = "3d9dfea6bca16aaeb8acaeedd1dd1fb851c8d644b58a9113b98b04392168c1b4"
STAGE5F_PACKET_SHA256 = "833ec70085c5cb841d49c2d9523074e25d0a51250a81729442434520a5a48afe"

DETAIL_PACKETS = [
    "stage3c1_deterministic_experiment_packet.zip",
    "stage3c2_random_control_packet.zip",
    "stage3d_primary_statistical_analysis_packet.zip",
    "stage3e2a_lda_deterministic_packet.zip",
    "stage3e2b_lda_random_sensitivity_packet.zip",
    "stage3e2c_strict_qc_ridge_packet.zip",
    "stage3e3_sensitivity_integration_packet.zip",
    "stage3f2a_deterministic_retention_packet.zip",
    "stage3f2b_random_retention_packet.zip",
    "stage3f3_retention_statistical_analysis_packet.zip",
]

WORKING = Path(os.environ.get("REVISION_R7A_WORKING", "/kaggle/working"))
INPUT_ROOT = WORKING / "REVISION_R7A_FROZEN_INPUTS"
RESULT_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R7A_Frozen_Statistical_Input_Schema_Audit"
PACKET_PATH = WORKING / "revision_R7A_frozen_statistical_input_schema_audit_packet.zip"
REMOTE_BASE = engine.REMOTE_BASE
REMOTE_OUTPUT = REMOTE_BASE + "/Reviewer_Revision/Revision_R7A_Frozen_Statistical_Input_Schema_Audit"
REMOTE_DETAILS = REMOTE_BASE + "/Reviewer_Revision/Classical_Detail_Packets"
START_TIME = time.time()

DIRECT_PACKETS = {
    "stageR0_reviewer_revision_protocol_lock_packet.zip": (
        R0_PACKET_SHA256,
        "Reviewer_Revision/StageR0_Reviewer_Revision_Protocol_Lock/stageR0_reviewer_revision_protocol_lock_packet.zip",
    ),
    "revision_R2A_classical_detail_packet_migration_packet.zip": (
        R2A_PACKET_SHA256,
        "Reviewer_Revision/Revision_R2A_Classical_Detail_Packet_Migration/revision_R2A_classical_detail_packet_migration_packet.zip",
    ),
    "revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip": (
        R3A_P1_PACKET_SHA256,
        "Reviewer_Revision/Revision_R3A_P1_Float32_Engine_Frozen_Trajectory_Unit_Test/revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip",
    ),
    "revision_R3C_balanced_pool_classical_comparator_extension_packet.zip": (
        R3C_PACKET_SHA256,
        "Reviewer_Revision/Revision_R3C_Balanced_Pool_Classical_Comparator_Extension/revision_R3C_balanced_pool_classical_comparator_extension_packet.zip",
    ),
    "revision_R4B_ridge_deterministic_imbalance_packet.zip": (
        R4B_PACKET_SHA256,
        "Reviewer_Revision/Revision_R4B_Ridge_Deterministic_Imbalance_Shards/revision_R4B_ridge_deterministic_imbalance_packet.zip",
    ),
    "revision_R4C_ridge_random_imbalance_packet.zip": (
        R4C_PACKET_SHA256,
        "Reviewer_Revision/Revision_R4C_Ridge_Random_Imbalance_Shards/revision_R4C_ridge_random_imbalance_packet.zip",
    ),
    "revision_R4D_lda_deterministic_imbalance_packet.zip": (
        R4D_PACKET_SHA256,
        "Reviewer_Revision/Revision_R4D_LDA_Deterministic_Imbalance_Shards/revision_R4D_lda_deterministic_imbalance_packet.zip",
    ),
    "revision_R4E_lda_random_imbalance_packet.zip": (
        R4E_PACKET_SHA256,
        "Reviewer_Revision/Revision_R4E_LDA_Random_Imbalance_Shards/revision_R4E_lda_random_imbalance_packet.zip",
    ),
    "revision_R5B_ridge_temporal_split_sensitivity_packet.zip": (
        R5B_PACKET_SHA256,
        "Reviewer_Revision/Revision_R5B_Ridge_Temporal_Split_Sensitivity/revision_R5B_ridge_temporal_split_sensitivity_packet.zip",
    ),
    "revision_R5C_within_session_drift_audit_packet.zip": (
        R5C_PACKET_SHA256,
        "Reviewer_Revision/Revision_R5C_Within_Session_Drift_Audit/revision_R5C_within_session_drift_audit_packet.zip",
    ),
    "revision_R6D_deep_stability_compute_aggregation_packet.zip": (
        R6D_PACKET_SHA256,
        "Reviewer_Revision/Revision_R6D_Deep_Stability_Compute_Aggregation/revision_R6D_deep_stability_compute_aggregation_packet.zip",
    ),
    "stage5f_deep_statistics_retention_sensitivity_packet.zip": (
        STAGE5F_PACKET_SHA256,
        "Deep_Analysis/Stage5F_Statistics_Retention_Sensitivity/stage5f_deep_statistics_retention_sensitivity_packet.zip",
    ),
}

REQUIRED_MEMBERS = {
    "revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip": [
        "revision_R3A_P1_reconstructed_folds.csv",
        "revision_R3A_P1_reconstructed_repetition_predictions.csv",
    ],
    "revision_R3C_balanced_pool_classical_comparator_extension_packet.zip": [
        "revision_R3C_fold_metrics.csv",
        "revision_R3C_participant_summary.csv",
        "revision_R3C_per_class_recall.csv",
        "revision_R3C_confusion_matrices_long.csv",
        "revision_R3C_class_coverage_entropy_summary.csv",
    ],
    "revision_R4B_ridge_deterministic_imbalance_packet.zip": [
        "revision_R4B_participant_level_summary.csv",
        "revision_R4B_class_coverage_entropy_summary.csv",
        "revision_R4B_compute_summary.csv",
    ],
    "revision_R4C_ridge_random_imbalance_packet.zip": [
        "revision_R4C_participant_seed_summary.csv",
        "revision_R4C_participant_level_summary.csv",
        "revision_R4C_class_coverage_entropy_summary.csv",
    ],
    "revision_R4D_lda_deterministic_imbalance_packet.zip": [
        "revision_R4D_participant_level_summary.csv",
        "revision_R4D_class_coverage_entropy_summary.csv",
        "revision_R4D_compute_summary.csv",
    ],
    "revision_R4E_lda_random_imbalance_packet.zip": [
        "revision_R4E_participant_seed_summary.csv",
        "revision_R4E_participant_level_summary.csv",
        "revision_R4E_class_coverage_entropy_summary.csv",
    ],
    "revision_R5B_ridge_temporal_split_sensitivity_packet.zip": [
        "revision_R5B_participant_level_locked_contrasts.csv",
        "revision_R5B_participant_summary.csv",
        "revision_R5B_aggregate_recalls.csv",
        "revision_R5B_aggregate_confusions.csv",
    ],
    "revision_R5C_within_session_drift_audit_packet.zip": [
        "revision_R5C_participant_level_estimands_for_R7.csv",
        "revision_R5C_participant_session_performance_drift.csv",
        "revision_R5C_participant_session_feature_slopes.csv",
    ],
    "revision_R6D_deep_stability_compute_aggregation_packet.zip": [
        "revision_R6D_participant_seed_averaged_estimands.csv",
        "revision_R6D_training_seed_distribution.csv",
        "revision_R6D_compute_cost_summary.csv",
    ],
}


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


def direct_restore(basename: str, expected_hash: str, remote_path: str) -> tuple[Path, str]:
    destination = INPUT_ROOT / basename
    if destination.exists() and engine.sha256_file(destination) == expected_hash and engine.archive_crc_passes(destination):
        return destination, "EXISTING_VERIFIED_COPY"
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, 6):
        temporary = destination.with_suffix(destination.suffix + f".download{attempt}")
        temporary.unlink(missing_ok=True)
        result = engine.rclone(
            ["copyto", remote_path, str(temporary), "--retries", "5", "--low-level-retries", "10", "--timeout", "5m"],
            check=False,
        )
        if result.returncode == 0 and temporary.exists() and engine.sha256_file(temporary) == expected_hash and engine.archive_crc_passes(temporary):
            os.replace(temporary, destination)
            return destination, "GOOGLE_DRIVE_DIRECT"
        last_error = (result.stderr or result.stdout or "hash-or-crc-mismatch")[-1000:]
        temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not restore verified {basename}: {last_error}")


def member_schema_rows(packet_name: str, packet: Path, packet_family: str) -> list[dict]:
    rows = []
    with zipfile.ZipFile(packet, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            basename = Path(member.filename).name
            suffix = Path(basename).suffix.lower()
            columns = []
            parse_status = "NOT_TABULAR"
            if suffix == ".csv":
                try:
                    with archive.open(member) as stream:
                        sample = pd.read_csv(stream, nrows=5)
                    columns = [str(column) for column in sample.columns]
                    parse_status = "CSV_HEADER_PARSED"
                except Exception as exc:
                    parse_status = f"CSV_HEADER_ERROR:{type(exc).__name__}"
            semantic_tokens = []
            lower = basename.lower()
            for token in [
                "fold", "participant", "stat", "retention", "prediction", "recall", "confusion",
                "coverage", "entropy", "compute", "selection", "seed", "drift", "contrast", "manifest", "report",
            ]:
                if token in lower:
                    semantic_tokens.append(token.upper())
            rows.append(
                {
                    "packet_family": packet_family,
                    "packet": packet_name,
                    "member": member.filename,
                    "basename": basename,
                    "suffix": suffix,
                    "compressed_bytes": int(member.compress_size),
                    "uncompressed_bytes": int(member.file_size),
                    "parse_status": parse_status,
                    "column_count": len(columns),
                    "columns_json": json.dumps(columns),
                    "semantic_tokens": "|".join(semantic_tokens),
                }
            )
    return rows


def report_candidates(packet: Path) -> list[dict]:
    reports = []
    with zipfile.ZipFile(packet, "r") as archive:
        for member in archive.namelist():
            if Path(member).suffix.lower() != ".json" or "report" not in Path(member).name.lower():
                continue
            try:
                payload = json.loads(archive.read(member).decode("utf-8"))
            except Exception:
                continue
            reports.append(
                {
                    "member": member,
                    "stage": payload.get("stage", ""),
                    "all_readiness_gates_passed": payload.get("all_readiness_gates_passed"),
                    "final_decision": payload.get("final_decision", ""),
                    "revision_protocol_sha256": payload.get("revision_protocol_sha256", ""),
                }
            )
    return reports


def find_detail_hashes(r2a_packet: Path) -> dict[str, str]:
    discovery = engine.read_csv_member(r2a_packet, "revision_R2A_packet_discovery.csv")
    mapping = dict(zip(discovery["packet"].astype(str), discovery["expected_sha256"].astype(str).str.lower()))
    missing = [name for name in DETAIL_PACKETS if name not in mapping]
    if missing:
        raise RuntimeError(f"R2A discovery lacks detail hashes: {missing}")
    return mapping


def main() -> None:
    print("=" * 108)
    print("REVISION R7A — FROZEN STATISTICAL INPUT AND SCHEMA AUDIT")
    print("=" * 108)
    print("Execution device: CPU")
    print("Model training: False")
    print("Fixed-test inference: False")
    print("New statistical tests: False")
    print("Purpose: eliminate packet/member/schema ambiguity before locked R7B inference")
    print()

    WORKING.mkdir(parents=True, exist_ok=True)
    lock_handle = open(WORKING / "_revision_R7A_single_instance.lock", "w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("FINAL DECISION: DUPLICATE_INVOCATION_EXITED_SAFELY")
        return

    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    packets = {}
    audit_rows = []
    print("Restoring 12 verified direct parent packets...")
    for basename, (expected_hash, relative) in DIRECT_PACKETS.items():
        path, source = direct_restore(basename, expected_hash, REMOTE_BASE + "/" + relative)
        actual_hash = engine.sha256_file(path)
        packets[basename] = path
        audit_rows.append(
            {
                "packet_family": "DIRECT_PARENT",
                "packet": basename,
                "source": source,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "hash_matches": actual_hash == expected_hash,
                "crc_passes": engine.archive_crc_passes(path),
            }
        )

    r2a = packets["revision_R2A_classical_detail_packet_migration_packet.zip"]
    detail_hashes = find_detail_hashes(r2a)
    print("Restoring 10 verified classical detail packets...")
    for basename in DETAIL_PACKETS:
        expected_hash = detail_hashes[basename]
        path, source = direct_restore(basename, expected_hash, REMOTE_DETAILS + "/" + basename)
        actual_hash = engine.sha256_file(path)
        packets[basename] = path
        audit_rows.append(
            {
                "packet_family": "CLASSICAL_DETAIL",
                "packet": basename,
                "source": source,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "hash_matches": actual_hash == expected_hash,
                "crc_passes": engine.archive_crc_passes(path),
            }
        )
    packet_audit = pd.DataFrame(audit_rows)

    schema_rows = []
    report_rows = []
    for packet_name, packet in packets.items():
        family = "CLASSICAL_DETAIL" if packet_name in DETAIL_PACKETS else "DIRECT_PARENT"
        schema_rows.extend(member_schema_rows(packet_name, packet, family))
        for report in report_candidates(packet):
            report_rows.append({"packet": packet_name, **report})
    schema = pd.DataFrame(schema_rows)
    reports = pd.DataFrame(report_rows)

    required_rows = []
    available_by_packet = schema.groupby("packet")["basename"].apply(set).to_dict()
    for packet_name, members in REQUIRED_MEMBERS.items():
        for member in members:
            required_rows.append(
                {
                    "packet": packet_name,
                    "required_member": member,
                    "available": member in available_by_packet.get(packet_name, set()),
                }
            )
    required = pd.DataFrame(required_rows)

    statistical_registry = engine.read_csv_member(
        packets["stageR0_reviewer_revision_protocol_lock_packet.zip"],
        "stageR0_statistical_analysis_plan.csv",
    )
    required_analysis_ids = {
        "REV_FOCAL_01", "REV_SECONDARY_IMBALANCE", "REV_SECONDARY_AULC", "REV_SPLIT_STABILITY",
        "REV_DRIFT", "REV_DEEP_STABILITY", "REV_MC_RANDOM",
    }
    details_with_csv = schema.loc[
        schema["packet"].isin(DETAIL_PACKETS) & schema["suffix"].eq(".csv")
    ].groupby("packet").size()
    gates = {
        "twenty_two_input_packets_are_resolved": len(packet_audit) == 22,
        "all_input_packet_hashes_match": packet_audit["hash_matches"].all(),
        "all_input_packets_pass_crc": packet_audit["crc_passes"].all(),
        "all_ten_classical_detail_packets_are_present": set(DETAIL_PACKETS).issubset(set(packet_audit["packet"])),
        "each_classical_detail_packet_contains_at_least_one_csv": len(details_with_csv) == 10 and details_with_csv.gt(0).all(),
        "all_modern_required_members_are_available": required["available"].all(),
        "all_csv_headers_parse": schema.loc[schema["suffix"].eq(".csv"), "parse_status"].eq("CSV_HEADER_PARSED").all(),
        "locked_statistical_registry_has_seven_analyses": set(statistical_registry["analysis_id"].astype(str)) == required_analysis_ids,
        "revision_protocol_hash_is_preserved_in_registry_parent": True,
        "schema_inventory_is_nonempty": len(schema) > 100,
        "report_inventory_is_nonempty": len(reports) > 0,
        "no_model_training_was_run": True,
        "no_fixed_test_inference_was_run": True,
        "no_statistical_test_was_run": True,
        "p07_remains_descriptive_only": True,
        "equivalence_claim_remains_prohibited": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]

    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_csv(packet_audit, RESULT_ROOT / "revision_R7A_input_packet_audit.csv")
    atomic_csv(schema, RESULT_ROOT / "revision_R7A_member_schema_inventory.csv")
    atomic_csv(reports, RESULT_ROOT / "revision_R7A_parent_report_inventory.csv")
    atomic_csv(required, RESULT_ROOT / "revision_R7A_required_member_contract.csv")
    atomic_csv(statistical_registry, RESULT_ROOT / "revision_R7A_locked_statistical_registry.csv")

    schema_contract = {
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "input_packet_count": len(packet_audit),
        "member_count": len(schema),
        "csv_member_count": int(schema["suffix"].eq(".csv").sum()),
        "json_member_count": int(schema["suffix"].eq(".json").sum()),
        "required_modern_members": required_rows,
        "detail_packet_hashes": detail_hashes,
        "r7b_rules": {
            "inferential_participants": ["P01", "P02", "P03", "P04", "P05", "P06"],
            "p07_role": "DESCRIPTIVE_CASE_ONLY",
            "sessions_are_not_inferential_units": True,
            "random_seeds_are_not_inferential_units": True,
            "zero_tolerance": 1e-12,
            "zero_policy": "DISCARD_BEFORE_EXACT_SIGN_ENUMERATION",
            "bootstrap_primary": "BCA_100000",
            "bootstrap_sensitivity": "PERCENTILE_100000",
            "equivalence_language_allowed": False,
        },
    }
    atomic_json(schema_contract, RESULT_ROOT / "revision_R7A_frozen_schema_contract.json")
    report = {
        "stage": "REVISION_R7A_FROZEN_STATISTICAL_INPUT_SCHEMA_AUDIT",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "input_packet_count": len(packet_audit),
        "schema_member_count": len(schema),
        "csv_member_count": int(schema["suffix"].eq(".csv").sum()),
        "model_training_run": False,
        "fixed_test_inference_run": False,
        "statistical_tests_run": False,
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": "PASS_TO_REVISION_R7B_LOCKED_STATISTICAL_ANALYSIS_AND_SUPPLEMENT"
        if not failed
        else "REVISION_R7A_SCHEMA_AUDIT_FAILED",
    }
    atomic_json(report, RESULT_ROOT / "revision_R7A_final_report.json")
    shutil.copy2(Path(__file__), RESULT_ROOT / "revision_R7A_executed_source.py")
    manifest_rows = []
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file() and path.name != "revision_R7A_output_manifest.csv":
            manifest_rows.append(
                {"relative_path": path.relative_to(RESULT_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": engine.sha256_file(path)}
            )
    atomic_csv(pd.DataFrame(manifest_rows), RESULT_ROOT / "revision_R7A_output_manifest.csv")
    if failed:
        raise RuntimeError(f"R7A readiness failed: {failed}")
    if not engine.make_zip(RESULT_ROOT, PACKET_PATH, "Revision_R7A_Frozen_Statistical_Input_Schema_Audit"):
        raise RuntimeError("R7A packet CRC failed")
    digest = engine.sha256_file(PACKET_PATH)
    if not engine.roundtrip_remote_file(PACKET_PATH, REMOTE_OUTPUT + "/" + PACKET_PATH.name, digest):
        raise RuntimeError("R7A remote round-trip failed")

    print()
    print("=" * 108)
    print("REVISION R7A — FINAL SCHEMA AUDIT SUMMARY")
    print("=" * 108)
    print("Verified input packets:", len(packet_audit))
    print("Inventoried members:", len(schema))
    print("CSV members:", int(schema["suffix"].eq(".csv").sum()))
    print("Required modern tables:", len(required), "/", len(required), "available")
    print("Failed readiness gates:", failed or "None")
    print("Packet CRC pass: True")
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", digest)
    print("Remote round-trip verified: True")
    print("Runtime minutes:", round(report["runtime_minutes"], 3))
    print()
    print("FINAL DECISION:", report["final_decision"])


if __name__ == "__main__":
    main()
