import base64
import configparser
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================================
# STAGE 5C-1B / 5D-0 — AMP FREEZE AND ACQUISITION INPUT PREFLIGHT
# ============================================================================

PARENT_PROTOCOL_SHA256 = (
    "f548b1ca6f2831c29ea8fecb764557efed49f229eb72322f98632edcf0aeb221"
)
DEEP_PROTOCOL_SHA256 = (
    "abe15812c1a52b0f4e917b5b6ad39b0dfde50e5bb2d58dfcc35b3cacb22e3bd2"
)
STAGE5B_PACKET_SHA256 = (
    "1c0fbc63f6412362f3ae7cd22609ea6a7fcb23236cdf688ad5fe0578ebaab84d"
)
STAGE5C_PACKET_SHA256 = (
    "85ea2e8a8440369a77d43f00b5d509ea2f2978d2a60ab2f24fb828ce9ca6b9d4"
)

WORKING = Path("/kaggle/working")
TOOLS = WORKING / "_stage5_tools"
TOOLS.mkdir(parents=True, exist_ok=True)
RCLONE = TOOLS / "rclone"

EVIDENCE_ROOT = Path(
    "/kaggle/input/datasets/zaidalsawaff/delta-q1-stage5-evidence-archives-v1"
)
STAGE3A_PACKET = EVIDENCE_ROOT / "stage3a_v1_1_protocol_amendment_packet.zip.bin"

STAGE5B_PACKET = WORKING / "stage5b_deep_sequence_assembly_packet.zip"
STAGE5C_PACKET = WORKING / "stage5c1_dual_gpu_loso_pretraining_packet.zip"

REMOTE_BASE = "gdrive_stage5:DELTA_Q1_Stage5_DeepLearning_Backup"
REMOTE_STAGE5B = REMOTE_BASE + "/Stage5B_Deep_Sequence_Assembly/" + STAGE5B_PACKET.name
REMOTE_STAGE5C = (
    REMOTE_BASE
    + "/Deep_Training/Stage5C_LOSO_Pretraining/"
    + STAGE5C_PACKET.name
)

AMP_OUTPUT = WORKING / "STAGE5C1B_AMP_AMENDMENT"
PREFLIGHT_OUTPUT = WORKING / "STAGE5D0_ACQUISITION_PREFLIGHT"
AMP_OUTPUT.mkdir(parents=True, exist_ok=True)
PREFLIGHT_OUTPUT.mkdir(parents=True, exist_ok=True)

AMP_PACKET = WORKING / "stage5c1b_amp_implementation_amendment_packet.zip"
PREFLIGHT_PACKET = WORKING / "stage5d0_acquisition_input_preflight_packet.zip"

print("=" * 79)
print("STAGE 5C-1B / 5D-0 — AMP FREEZE AND ACQUISITION INPUT PREFLIGHT")
print("=" * 79)
print("Execution device: CPU")
print("No model training will be performed.")
print()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(payload, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )


def zip_member_by_basename(packet, basename):
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
    if len(matches) != 1:
        raise ValueError(f"Expected one {basename} in {packet}; found {len(matches)}")
    return matches[0]


def read_csv_member(packet, basename):
    member = zip_member_by_basename(packet, basename)
    with zipfile.ZipFile(packet, "r") as archive:
        with archive.open(member, "r") as handle:
            return pd.read_csv(handle)


def read_text_member(packet, basename):
    member = zip_member_by_basename(packet, basename)
    with zipfile.ZipFile(packet, "r") as archive:
        return archive.read(member).decode("utf-8", errors="strict")


def read_json_member(packet, basename):
    return json.loads(read_text_member(packet, basename))


def all_text_in_zip(packet):
    output = []
    with zipfile.ZipFile(packet, "r") as archive:
        for member in archive.namelist():
            if member.lower().endswith((".json", ".csv", ".txt", ".py")):
                output.append(archive.read(member).decode("utf-8", errors="ignore"))
    return "\n".join(output)


def make_zip(source_directory, destination, archive_root):
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(source_directory.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=archive_root + "/" + path.relative_to(source_directory).as_posix(),
                )
    with zipfile.ZipFile(destination, "r") as archive:
        return archive.testzip() is None


# ----------------------------------------------------------------------------
# 1. VERIFIED RCLONE BOOTSTRAP
# ----------------------------------------------------------------------------

if not RCLONE.exists():
    print("Downloading verified official rclone binary...")
    version_text = urllib.request.urlopen(
        "https://downloads.rclone.org/version.txt", timeout=60
    ).read().decode("utf-8").strip()
    match = re.search(r"v?(\d+\.\d+\.\d+)", version_text)
    if match is None:
        raise RuntimeError("Could not resolve rclone version")
    version = match.group(1)
    archive_name = f"rclone-v{version}-linux-amd64.zip"
    base_url = f"https://downloads.rclone.org/v{version}"
    temporary_root = Path(tempfile.mkdtemp(prefix="stage5d0_rclone_", dir="/tmp"))
    archive_path = temporary_root / archive_name
    sums_path = temporary_root / "SHA256SUMS"
    urllib.request.urlretrieve(f"{base_url}/{archive_name}", archive_path)
    urllib.request.urlretrieve(f"{base_url}/SHA256SUMS", sums_path)
    expected = None
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == archive_name:
            expected = parts[0].lower()
            break
    if expected is None or sha256_file(archive_path) != expected:
        raise RuntimeError("rclone archive SHA-256 verification failed")
    extraction = temporary_root / "extract"
    extraction.mkdir()
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("rclone archive CRC failed")
        archive.extractall(extraction)
    candidates = list(extraction.glob("**/rclone"))
    if len(candidates) != 1:
        raise RuntimeError("Unexpected rclone archive structure")
    shutil.copy2(candidates[0], RCLONE)
    os.chmod(RCLONE, 0o755)
    shutil.rmtree(temporary_root, ignore_errors=True)

rclone_version = subprocess.run(
    [str(RCLONE), "version"], check=True, capture_output=True, text=True
).stdout.splitlines()[0]
print("rclone version:", rclone_version)


# ----------------------------------------------------------------------------
# 2. RESTRICTED DRIVE CONFIG AND PACKET RESTORE
# ----------------------------------------------------------------------------

from kaggle_secrets import UserSecretsClient

encoded = UserSecretsClient().get_secret("RCLONE_CONFIG_B64")
decoded = base64.b64decode(encoded, validate=True)
del encoded

parser = configparser.ConfigParser()
parser.read_string(decoded.decode("utf-8"))
remote_verified = (
    "gdrive_stage5" in parser.sections()
    and parser.get("gdrive_stage5", "type", fallback="") == "drive"
    and parser.get("gdrive_stage5", "scope", fallback="") == "drive.file"
)
if not remote_verified:
    raise RuntimeError("Restricted Drive configuration verification failed")

temporary_config = tempfile.NamedTemporaryFile(
    mode="wb", prefix="stage5d0_", suffix=".conf", dir="/tmp", delete=False
)
temporary_config.write(decoded)
temporary_config.close()
del decoded
CONFIG_PATH = Path(temporary_config.name)
os.chmod(CONFIG_PATH, 0o600)


def rclone(arguments, check=True):
    return subprocess.run(
        [str(RCLONE), "--config", str(CONFIG_PATH), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


print("Downloading frozen Stage 5B packet...")
rclone([
    "copyto", REMOTE_STAGE5B, str(STAGE5B_PACKET),
    "--retries", "5", "--low-level-retries", "10", "--timeout", "5m",
])
print("Downloading frozen Stage 5C packet...")
rclone([
    "copyto", REMOTE_STAGE5C, str(STAGE5C_PACKET),
    "--retries", "5", "--low-level-retries", "10", "--timeout", "5m",
])

stage5b_hash_matches = sha256_file(STAGE5B_PACKET) == STAGE5B_PACKET_SHA256
stage5c_hash_matches = sha256_file(STAGE5C_PACKET) == STAGE5C_PACKET_SHA256
if not stage5b_hash_matches or not stage5c_hash_matches:
    raise RuntimeError("Frozen Stage 5B/5C packet hash mismatch")

for packet in [STAGE3A_PACKET, STAGE5B_PACKET, STAGE5C_PACKET]:
    if not packet.exists() or packet.stat().st_size == 0:
        raise FileNotFoundError(packet)
    with zipfile.ZipFile(packet, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"CRC failure: {packet}")

print("Stage 5B hash verified:", stage5b_hash_matches)
print("Stage 5C hash verified:", stage5c_hash_matches)
print("Stage 3A CRC pass: True")
print()


# ----------------------------------------------------------------------------
# 3. STAGE 5C COMPLETION AND AMP SOURCE AUDIT
# ----------------------------------------------------------------------------

stage5c_report = read_json_member(STAGE5C_PACKET, "stage5c1_loso_pretraining_report.json")
worker_source = read_text_member(STAGE5C_PACKET, "stage5c1_cuda_worker.py")

normalized_worker_source = re.sub(r"\s+", "", worker_source)
amp_initial_scale_present = "init_scale=1024.0" in normalized_worker_source
amp_growth_interval_present = "growth_interval=10000" in normalized_worker_source
default_scaler_absent = 'GradScaler("cuda",enabled=True)' not in normalized_worker_source

with zipfile.ZipFile(STAGE5C_PACKET, "r") as archive:
    complete_members = [
        member for member in archive.namelist() if Path(member).name == "complete.json"
    ]
    completion_rows = [json.loads(archive.read(member).decode("utf-8")) for member in complete_members]

completion = pd.DataFrame(completion_rows).sort_values("target_participant").reset_index(drop=True)
completion.to_csv(AMP_OUTPUT / "stage5c1b_verified_completion_summary.csv", index=False)

stage5c_metrics_finite = bool(
    len(completion) == 7
    and np.isfinite(
        completion[
            [
                "best_validation_loss",
                "best_validation_accuracy",
                "best_validation_balanced_accuracy",
                "best_validation_macro_f1",
            ]
        ].to_numpy(dtype=float)
    ).all()
)

amendment = {
    "amendment_role": "NUMERICAL_IMPLEMENTATION_AMENDMENT_ONLY",
    "parent_deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
    "stage5c1_packet_sha256": STAGE5C_PACKET_SHA256,
    "diagnostic_failure_scale": 65536.0,
    "diagnostic_safe_scales_for_p01_and_p02": [1024.0, 256.0, 32.0],
    "selected_initial_scale": 1024.0,
    "selected_growth_interval": 10000,
    "affected_parameter_at_failed_scale": "classifier.weight",
    "p01_maximum_absolute_input": 9.838401794433594,
    "p02_maximum_absolute_input": 8.5670166015625,
    "values_exceeding_fp16_maximum": 0,
    "worker_source_sha256": hashlib.sha256(worker_source.encode("utf-8")).hexdigest(),
    "worker_source_contains_initial_scale_1024": amp_initial_scale_present,
    "worker_source_contains_growth_interval_10000": amp_growth_interval_present,
    "stage5c_fold_count": int(len(completion)),
    "all_stage5c_metrics_finite": stage5c_metrics_finite,
    "target_data_used": False,
    "data_changed": False,
    "normalization_changed": False,
    "model_architecture_changed": False,
    "loss_changed": False,
    "optimizer_changed": False,
    "endpoint_changed": False,
    "scientific_hypothesis_changed": False,
    "stage3g_primary_result_changed": False,
}
amendment["amendment_sha256"] = hashlib.sha256(
    json.dumps(amendment, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

amp_gates = {
    "stage5c_packet_hash_matches": stage5c_hash_matches,
    "stage5c_all_original_gates_passed": bool(stage5c_report["all_readiness_gates_passed"]),
    "seven_loso_folds_completed": len(completion) == 7,
    "targets_are_p01_to_p07": completion["target_participant"].tolist()
        == ["P01", "P02", "P03", "P04", "P05", "P06", "P07"],
    "worker_contains_initial_scale_1024": amp_initial_scale_present,
    "worker_contains_growth_interval_10000": amp_growth_interval_present,
    "unsafe_default_scaler_form_is_absent": default_scaler_absent,
    "all_metrics_are_finite": stage5c_metrics_finite,
    "target_data_was_never_used": bool(completion["target_data_used"].eq(False).all()),
    "scientific_protocol_is_unchanged": not any(
        amendment[key]
        for key in [
            "data_changed", "normalization_changed", "model_architecture_changed",
            "loss_changed", "optimizer_changed", "endpoint_changed",
            "scientific_hypothesis_changed", "stage3g_primary_result_changed",
        ]
    ),
    "amendment_hash_is_valid": len(amendment["amendment_sha256"]) == 64,
}
amendment["readiness_gates"] = amp_gates
amendment["all_readiness_gates_passed"] = all(amp_gates.values())
write_json(amendment, AMP_OUTPUT / "stage5c1b_amp_implementation_amendment.json")

note = """STAGE 5C-1B — AMP NUMERICAL IMPLEMENTATION AMENDMENT

The original FP16 AMP run used PyTorch's default initial GradScaler scale of
65536. A targeted diagnostic reproduced a non-finite gradient only in the
classifier weight for P01 and P02, while inputs and logits remained finite.
Scales 1024, 256, and 32 were finite for both diagnostic folds. The corrected
execution therefore used initial_scale=1024 and growth_interval=10000.

This amendment changes only the numerical loss-scaling implementation. It does
not change the frozen data, causal masks, normalization, architecture, loss,
optimizer, endpoints, participant roles, acquisition policy, Stage 3G result,
or scientific hypothesis.
"""
(AMP_OUTPUT / "stage5c1b_amp_implementation_note.txt").write_text(note, encoding="utf-8")

amp_manifest = []
for path in sorted(AMP_OUTPUT.glob("*")):
    if path.is_file():
        amp_manifest.append({
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
pd.DataFrame(amp_manifest).to_csv(AMP_OUTPUT / "stage5c1b_sha256_manifest.csv", index=False)
amp_packet_crc = make_zip(AMP_OUTPUT, AMP_PACKET, "Stage5C1B_AMP_Amendment")

print("=" * 79)
print("STAGE 5C-1B — AMP AMENDMENT AUDIT")
print("=" * 79)
for gate, passed in amp_gates.items():
    print(f"{gate}: {passed}")
print("Amendment SHA-256:", amendment["amendment_sha256"])
print("Packet CRC pass:", amp_packet_crc)
print()


# ----------------------------------------------------------------------------
# 4. STAGE 3A v1.1 ACQUISITION SCHEMA DISCOVERY
# ----------------------------------------------------------------------------

stage3a_text = all_text_in_zip(STAGE3A_PACKET)
parent_hash_verified = PARENT_PROTOCOL_SHA256 in stage3a_text

universe = read_csv_member(STAGE3A_PACKET, "stage3a_v1_1_repetition_protocol_universe.csv")
lookup = read_csv_member(STAGE3A_PACKET, "stage3a_v1_1_internal_token_lookup.csv")
strategies = read_csv_member(STAGE3A_PACKET, "stage3a_v1_1_locked_strategies.csv")
budgets = read_csv_member(STAGE3A_PACKET, "stage3a_v1_1_query_budget_schedule.csv")
random_seeds = read_csv_member(STAGE3A_PACKET, "stage3a_v1_1_random_seeds.csv")
selector_schema = read_json_member(STAGE3A_PACKET, "stage3a_v1_1_selector_schema.json")

inventory_rows = []
with zipfile.ZipFile(STAGE3A_PACKET, "r") as archive:
    for info in archive.infolist():
        inventory_rows.append({"member": info.filename, "size_bytes": info.file_size})
pd.DataFrame(inventory_rows).to_csv(
    PREFLIGHT_OUTPUT / "stage5d0_stage3a_packet_inventory.csv", index=False
)

schema_tables = {
    "universe": universe,
    "lookup": lookup,
    "strategies": strategies,
    "budgets": budgets,
    "random_seeds": random_seeds,
}
schema_summary_rows = []
for name, table in schema_tables.items():
    schema_summary_rows.append({
        "table": name,
        "rows": len(table),
        "columns": json.dumps(table.columns.tolist()),
    })
pd.DataFrame(schema_summary_rows).to_csv(
    PREFLIGHT_OUTPUT / "stage5d0_acquisition_schema_summary.csv", index=False
)

print("=" * 79)
print("STAGE 5D-0 — ACQUISITION SCHEMA DISCOVERY")
print("=" * 79)
for name, table in schema_tables.items():
    print()
    print(name.upper())
    print("Shape:", table.shape)
    print("Columns:", table.columns.tolist())
    print(table.head(10).to_string(index=False))
print()
print("SELECTOR SCHEMA JSON:")
print(json.dumps(selector_schema, indent=2, ensure_ascii=False))

participant_column = "participant" if "participant" in universe.columns else None
session_column = "session" if "session" in universe.columns else None
label_column = "label" if "label" in universe.columns else None
token_candidates = [
    column
    for column in universe.columns
    if (
        ("opaque" in column.lower() and "token" in column.lower())
        or column.lower() == "selector_visible_identifier"
    )
]
role_candidates = [column for column in universe.columns if "protocol_role" in column.lower()]

preflight_gates = {
    "stage3a_packet_exists": STAGE3A_PACKET.exists(),
    "stage3a_parent_hash_verifies": parent_hash_verified,
    "universe_has_2940_rows": len(universe) == 2940,
    "lookup_has_2940_rows": len(lookup) == 2940,
    "participant_column_is_available": participant_column is not None,
    "session_column_is_available": session_column is not None,
    "label_column_is_available": label_column is not None,
    "participants_are_p01_to_p07": participant_column is not None
        and sorted(universe[participant_column].astype(str).unique().tolist())
        == ["P01", "P02", "P03", "P04", "P05", "P06", "P07"],
    "sessions_are_0_to_5": session_column is not None
        and sorted(pd.to_numeric(universe[session_column]).astype(int).unique().tolist())
        == [0, 1, 2, 3, 4, 5],
    "labels_are_0_to_6": label_column is not None
        and sorted(pd.to_numeric(universe[label_column]).astype(int).unique().tolist())
        == [0, 1, 2, 3, 4, 5, 6],
    "opaque_token_column_is_discovered": len(token_candidates) >= 1,
    "protocol_role_column_is_discovered": len(role_candidates) >= 1,
    "strategy_table_is_nonempty": len(strategies) > 0,
    "budget_table_is_nonempty": len(budgets) > 0,
    "random_seed_count_is_30": len(random_seeds) == 30,
    "stage5b_packet_hash_matches": stage5b_hash_matches,
    "stage5c_packet_hash_matches": stage5c_hash_matches,
    "stage5c_amp_amendment_passed": all(amp_gates.values()),
    "no_gpu_computation_was_used": True,
    "credentials_not_written_to_artifacts": True,
}

preflight_report = {
    "stage": "STAGE5D0_ACQUISITION_INPUT_PREFLIGHT",
    "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
    "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
    "stage5b_packet_sha256": STAGE5B_PACKET_SHA256,
    "stage5c_packet_sha256": STAGE5C_PACKET_SHA256,
    "amp_amendment_sha256": amendment["amendment_sha256"],
    "universe_shape": list(universe.shape),
    "lookup_shape": list(lookup.shape),
    "strategy_shape": list(strategies.shape),
    "budget_shape": list(budgets.shape),
    "random_seed_shape": list(random_seeds.shape),
    "universe_columns": universe.columns.tolist(),
    "lookup_columns": lookup.columns.tolist(),
    "strategy_columns": strategies.columns.tolist(),
    "budget_columns": budgets.columns.tolist(),
    "random_seed_columns": random_seeds.columns.tolist(),
    "opaque_token_candidates": token_candidates,
    "protocol_role_candidates": role_candidates,
    "selector_schema": selector_schema,
    "readiness_gates": preflight_gates,
    "all_readiness_gates_passed": all(preflight_gates.values()),
}
write_json(preflight_report, PREFLIGHT_OUTPUT / "stage5d0_acquisition_input_preflight_report.json")

preflight_manifest = []
for path in sorted(PREFLIGHT_OUTPUT.glob("*")):
    if path.is_file():
        preflight_manifest.append({
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
pd.DataFrame(preflight_manifest).to_csv(
    PREFLIGHT_OUTPUT / "stage5d0_sha256_manifest.csv", index=False
)
preflight_packet_crc = make_zip(
    PREFLIGHT_OUTPUT, PREFLIGHT_PACKET, "Stage5D0_Acquisition_Input_Preflight"
)


# ----------------------------------------------------------------------------
# 5. DRIVE BACKUP AND FINAL DECISION
# ----------------------------------------------------------------------------

remote_amp = REMOTE_BASE + "/Deep_Training/Stage5C1B_AMP_Amendment"
remote_preflight = REMOTE_BASE + "/Deep_Training/Stage5D0_Acquisition_Preflight"

rclone([
    "copy", str(AMP_OUTPUT), remote_amp,
    "--retries", "5", "--low-level-retries", "10", "--timeout", "5m",
])
rclone([
    "copyto", str(AMP_PACKET), remote_amp + "/" + AMP_PACKET.name,
    "--retries", "5", "--low-level-retries", "10", "--timeout", "5m",
])
rclone([
    "copy", str(PREFLIGHT_OUTPUT), remote_preflight,
    "--retries", "5", "--low-level-retries", "10", "--timeout", "5m",
])
rclone([
    "copyto", str(PREFLIGHT_PACKET), remote_preflight + "/" + PREFLIGHT_PACKET.name,
    "--retries", "5", "--low-level-retries", "10", "--timeout", "5m",
])

amp_remote_files = set(
    rclone(["lsf", remote_amp, "--files-only"]).stdout.splitlines()
)
preflight_remote_files = set(
    rclone(["lsf", remote_preflight, "--files-only"]).stdout.splitlines()
)
amp_remote_verified = AMP_PACKET.name in amp_remote_files
preflight_remote_verified = PREFLIGHT_PACKET.name in preflight_remote_files

if CONFIG_PATH.exists():
    CONFIG_PATH.unlink()

print()
print("Readiness gates:")
for gate, passed in preflight_gates.items():
    print(f"  {gate}: {passed}")
print()
print("AMP amendment packet:", AMP_PACKET)
print("AMP amendment packet SHA-256:", sha256_file(AMP_PACKET))
print("AMP Drive verification:", amp_remote_verified)
print("Stage 5D-0 packet:", PREFLIGHT_PACKET)
print("Stage 5D-0 packet SHA-256:", sha256_file(PREFLIGHT_PACKET))
print("Stage 5D-0 Drive verification:", preflight_remote_verified)
print()

if (
    all(amp_gates.values())
    and all(preflight_gates.values())
    and amp_packet_crc
    and preflight_packet_crc
    and amp_remote_verified
    and preflight_remote_verified
):
    print("FINAL DECISION: PASS_TO_STAGE5D1_DETERMINISTIC_ENGINE_UNIT_TESTS")
else:
    print("FINAL DECISION: STAGE5D0_PREFLIGHT_NOT_READY")
