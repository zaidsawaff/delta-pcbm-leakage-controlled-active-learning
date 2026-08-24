from __future__ import annotations

import atexit
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
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REVISION_PROTOCOL_SHA256 = "6807b71de18ca82013cfa4360d760e0daf9a920a1acc0625dcb13bd8f4d07249"
R0_PACKET_SHA256 = "0800e315a29b81934095ba56deaea3f8b6600fd0df13db348d7ea72d3b82df78"
STAGE5B_PACKET_SHA256 = "1c0fbc63f6412362f3ae7cd22609ea6a7fcb23236cdf688ad5fe0578ebaab84d"
R4E_PACKET_SHA256 = "34d4054d9dac5cbf992ed72f0f4096d013b5cede47a5bf349b59f49ee44a71b3"

PARTICIPANTS = [f"P{i:02d}" for i in range(1, 8)]
ABLE_BODIED = PARTICIPANTS[:6]
TARGET_SESSIONS = [1, 2, 3, 4, 5]
LABELS = list(range(7))
REPETITIONS = list(range(1, 11))
EXPECTED_SPLITS = {
    "FIRST_HALF_ORIGINAL": ({1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}),
    "SECOND_HALF_REVERSED": ({6, 7, 8, 9, 10}, {1, 2, 3, 4, 5}),
    "ODD_CANDIDATE_EVEN_TEST": ({1, 3, 5, 7, 9}, {2, 4, 6, 8, 10}),
    "EVEN_CANDIDATE_ODD_TEST": ({2, 4, 6, 8, 10}, {1, 3, 5, 7, 9}),
}
R5B_STRATEGIES = [
    ("NO_ADAPTATION_REFERENCE", 0, False),
    ("PCBM_PROPOSED", 7, False),
    ("GLOBAL_MARGIN", 7, False),
    ("RANDOM_UNIFORM", 7, True),
]

WORKING = Path(os.environ.get("REVISION_R5A_WORKING", "/kaggle/working"))
TOOLS = WORKING / "_stage5_tools"
RCLONE = TOOLS / "rclone"
INPUT_ROOT = WORKING / "REVISION_R5A_FROZEN_INPUTS"
RESULT_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R5A_Temporal_Split_Drift_Unit_Tests"
PACKET_PATH = WORKING / "revision_R5A_temporal_split_drift_unit_test_packet.zip"
REMOTE_BASE = "gdrive_stage5:DELTA_Q1_Stage5_DeepLearning_Backup"
REMOTE_OUTPUT = REMOTE_BASE + "/Reviewer_Revision/Revision_R5A_Temporal_Split_Drift_Unit_Tests"
CONFIG_PATH: Path | None = None
START_TIME = time.time()

DIRECT_PACKETS = {
    "stageR0_reviewer_revision_protocol_lock_packet.zip": (
        R0_PACKET_SHA256,
        "Reviewer_Revision/StageR0_Reviewer_Revision_Protocol_Lock/"
        "stageR0_reviewer_revision_protocol_lock_packet.zip",
    ),
    "stage5b_deep_sequence_assembly_packet.zip": (
        STAGE5B_PACKET_SHA256,
        "Stage5B_Deep_Sequence_Assembly/stage5b_deep_sequence_assembly_packet.zip",
    ),
    "revision_R4E_lda_random_imbalance_packet.zip": (
        R4E_PACKET_SHA256,
        "Reviewer_Revision/Revision_R4E_LDA_Random_Imbalance_Shards/"
        "revision_R4E_lda_random_imbalance_packet.zip",
    ),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_crc_passes(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def member_matches(packet: Path, basename: str) -> list[str]:
    with zipfile.ZipFile(packet, "r") as archive:
        return [name for name in archive.namelist() if Path(name).name == basename]


def member_bytes(packet: Path, basename: str) -> bytes:
    matches = member_matches(packet, basename)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {basename} in {packet.name}; found {matches}")
    with zipfile.ZipFile(packet, "r") as archive:
        return archive.read(matches[0])


def read_csv_member(packet: Path, basename: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(member_bytes(packet, basename)))


def read_json_member(packet: Path, basename: str) -> dict:
    return json.loads(member_bytes(packet, basename).decode("utf-8"))


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def atomic_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, destination)


def make_zip(source: Path, destination: Path, archive_root: str) -> bool:
    destination.unlink(missing_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(archive_root) / path.relative_to(source)).as_posix())
    return archive_crc_passes(destination)


def cleanup_secret() -> None:
    global CONFIG_PATH
    if CONFIG_PATH is not None:
        try:
            CONFIG_PATH.unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(cleanup_secret)


def bootstrap_rclone() -> None:
    TOOLS.mkdir(parents=True, exist_ok=True)
    if RCLONE.exists():
        RCLONE.chmod(0o755)
        return
    print("Downloading verified official rclone binary...", flush=True)
    version_text = urllib.request.urlopen("https://downloads.rclone.org/version.txt", timeout=60).read().decode("utf-8")
    match = re.search(r"v?(\d+\.\d+\.\d+)", version_text)
    if match is None:
        raise RuntimeError("Could not resolve official rclone version")
    version = match.group(1)
    archive_name = f"rclone-v{version}-linux-amd64.zip"
    base_url = f"https://downloads.rclone.org/v{version}"
    temporary_root = Path(tempfile.mkdtemp(prefix="revision_r5a_rclone_", dir="/tmp"))
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
        raise RuntimeError("rclone SHA-256 verification failed")
    with zipfile.ZipFile(archive_path, "r") as archive:
        binaries = [name for name in archive.namelist() if name.endswith("/rclone")]
        if archive.testzip() is not None or len(binaries) != 1:
            raise RuntimeError("rclone archive verification failed")
        with archive.open(binaries[0]) as source, open(RCLONE, "wb") as target:
            shutil.copyfileobj(source, target)
    RCLONE.chmod(0o755)
    shutil.rmtree(temporary_root, ignore_errors=True)


def create_rclone_config() -> None:
    global CONFIG_PATH
    from kaggle_secrets import UserSecretsClient

    encoded = UserSecretsClient().get_secret("RCLONE_CONFIG_B64")
    decoded = base64.b64decode(encoded, validate=True)
    parser = configparser.ConfigParser()
    parser.read_string(decoded.decode("utf-8"))
    if not parser.has_section("gdrive_stage5"):
        raise RuntimeError("gdrive_stage5 remote is missing")
    if parser.get("gdrive_stage5", "type", fallback="") != "drive":
        raise RuntimeError("gdrive_stage5 is not a Drive remote")
    if parser.get("gdrive_stage5", "scope", fallback="") != "drive.file":
        raise RuntimeError("Drive scope is not restricted to drive.file")
    handle = tempfile.NamedTemporaryFile(prefix="revision_r5a_", suffix=".conf", dir="/tmp", delete=False)
    handle.write(decoded)
    handle.flush()
    handle.close()
    os.chmod(handle.name, 0o600)
    CONFIG_PATH = Path(handle.name)


def rclone(arguments: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(RCLONE), "--config", str(CONFIG_PATH)] + list(arguments),
        check=check,
        capture_output=True,
        text=True,
    )


def resolve_direct_packet(basename: str) -> tuple[Path, str]:
    expected_hash, relative_remote = DIRECT_PACKETS[basename]
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = INPUT_ROOT / basename
    if destination.exists() and sha256_file(destination) == expected_hash and archive_crc_passes(destination):
        return destination, "EXISTING_VERIFIED_COPY"
    temporary = destination.with_suffix(".download")
    temporary.unlink(missing_ok=True)
    last_error = ""
    for attempt in range(1, 6):
        result = rclone(
            [
                "copyto",
                REMOTE_BASE + "/" + relative_remote,
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
            if sha256_file(temporary) == expected_hash and archive_crc_passes(temporary):
                os.replace(temporary, destination)
                return destination, f"GOOGLE_DRIVE_DIRECT_ATTEMPT_{attempt}"
            last_error = "downloaded bytes failed frozen SHA-256 or CRC"
        else:
            last_error = (result.stderr or result.stdout or f"exit={result.returncode}")[-1000:]
        temporary.unlink(missing_ok=True)
        time.sleep(min(2 ** (attempt - 1), 20))
    raise RuntimeError(f"Unable to restore {basename} from its locked path: {last_error}")


def roundtrip_remote(local_path: Path, remote_path: str, expected_hash: str) -> bool:
    rclone(["copyto", str(local_path), remote_path, "--retries", "5", "--timeout", "5m"])
    descriptor, name = tempfile.mkstemp(prefix="revision_r5a_roundtrip_", suffix=".zip", dir="/tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        rclone(["copyto", remote_path, str(temporary), "--retries", "5", "--timeout", "5m"])
        return sha256_file(temporary) == expected_hash and archive_crc_passes(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def parse_repetition_set(value: str) -> set[int]:
    return {int(item) for item in str(value).split("|") if str(item).strip()}


def normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"sequence_row", "participant", "session", "label", "repetition"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise RuntimeError(f"Stage 5B metadata missing required columns: {missing}")
    result = metadata.copy()
    result["participant"] = result["participant"].astype(str)
    for column in ["sequence_row", "session", "label", "repetition"]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(int)
    return result.sort_values("sequence_row", kind="mergesort").reset_index(drop=True)


def validate_locked_splits(splits: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[set[int], set[int]]]]:
    expected_columns = {
        "split_id",
        "candidate_repetition_numbers",
        "fixed_test_repetition_numbers",
        "is_frozen_original_split",
    }
    if not expected_columns.issubset(splits.columns):
        raise RuntimeError("R0 temporal split schedule schema is incomplete")
    parsed: dict[str, tuple[set[int], set[int]]] = {}
    audit_rows = []
    for row in splits.itertuples(index=False):
        split_id = str(row.split_id)
        candidate = parse_repetition_set(row.candidate_repetition_numbers)
        test = parse_repetition_set(row.fixed_test_repetition_numbers)
        parsed[split_id] = (candidate, test)
        expected_candidate, expected_test = EXPECTED_SPLITS.get(split_id, (set(), set()))
        audit_rows.append(
            {
                "split_id": split_id,
                "candidate_repetitions": "|".join(map(str, sorted(candidate))),
                "fixed_test_repetitions": "|".join(map(str, sorted(test))),
                "candidate_count": len(candidate),
                "fixed_test_count": len(test),
                "sets_are_disjoint": candidate.isdisjoint(test),
                "union_is_repetitions_1_to_10": candidate | test == set(REPETITIONS),
                "matches_locked_expected_definition": candidate == expected_candidate and test == expected_test,
                "is_frozen_original_split": bool(row.is_frozen_original_split),
            }
        )
    return pd.DataFrame(audit_rows), parsed


def build_membership(metadata: pd.DataFrame, split_map: dict[str, tuple[set[int], set[int]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = metadata.loc[
        metadata["participant"].isin(PARTICIPANTS)
        & metadata["session"].isin(TARGET_SESSIONS)
        & metadata["label"].isin(LABELS)
        & metadata["repetition"].isin(REPETITIONS),
        ["sequence_row", "participant", "session", "label", "repetition"],
    ].copy()
    frames = []
    for split_id, (candidate, test) in split_map.items():
        frame = target.copy()
        frame.insert(0, "split_id", split_id)
        frame["temporal_role"] = np.where(
            frame["repetition"].isin(candidate), "CURRENT_SESSION_CANDIDATE", "FIXED_TEST_NEVER_QUERY"
        )
        frames.append(frame)
    membership = pd.concat(frames, ignore_index=True)
    summary = (
        membership.groupby(["split_id", "participant", "session", "temporal_role"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(columns=None)
    )
    for required in ["CURRENT_SESSION_CANDIDATE", "FIXED_TEST_NEVER_QUERY"]:
        if required not in summary.columns:
            summary[required] = 0
    class_counts = (
        membership.groupby(["split_id", "participant", "session", "temporal_role", "label"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    class_audit = (
        class_counts.groupby(["split_id", "participant", "session", "temporal_role"], sort=True)["count"]
        .agg(class_count="size", minimum_per_class="min", maximum_per_class="max")
        .reset_index()
    )
    class_wide = class_audit.pivot(
        index=["split_id", "participant", "session"], columns="temporal_role"
    ).reset_index()
    class_wide.columns = [
        "_".join(str(part) for part in column if str(part) not in {"", "None"}).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in class_wide.columns
    ]
    summary = summary.merge(
        class_wide,
        on=["split_id", "participant", "session"],
        how="left",
    )
    return membership, summary


def build_causal_schedule(metadata: pd.DataFrame, split_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    session0 = metadata.loc[
        metadata["participant"].isin(PARTICIPANTS)
        & metadata["session"].eq(0)
        & metadata["label"].isin(LABELS)
        & metadata["repetition"].isin({1, 2, 3, 4, 5})
    ].copy()
    initial_counts = session0.groupby("participant").size().to_dict()
    rows = []
    for split_id in split_ids:
        for participant in PARTICIPANTS:
            for target_session in TARGET_SESSIONS:
                allowed_history_sessions = list(range(0, target_session))
                rows.append(
                    {
                        "split_id": split_id,
                        "participant": participant,
                        "target_session": target_session,
                        "allowed_history_sessions": "|".join(map(str, allowed_history_sessions)),
                        "maximum_history_session": max(allowed_history_sessions),
                        "initial_session0_repetitions": int(initial_counts.get(participant, 0)),
                        "prior_target_sessions": target_session - 1,
                        "k07_history_repetitions_before_current_query": 35 + 7 * (target_session - 1),
                        "current_candidate_repetitions": 35,
                        "current_fixed_test_repetitions": 35,
                        "future_session_accessible": False,
                        "current_fixed_test_enters_history": False,
                    }
                )
    return session0, pd.DataFrame(rows)


def build_r5b_manifest(seed_schedule: pd.DataFrame, split_ids: list[str]) -> pd.DataFrame:
    random_rows = seed_schedule.loc[
        seed_schedule["seed_family"].astype(str).eq("RANDOM_ACQUISITION")
        & seed_schedule["use_rule"].astype(str).eq("INITIAL_30")
    ].copy()
    random_rows["seed_index"] = pd.to_numeric(random_rows["seed_index"], errors="raise").astype(int)
    random_rows["seed"] = pd.to_numeric(random_rows["seed"], errors="raise").astype(int)
    random_rows = random_rows.sort_values("seed_index")
    rows = []
    for split_id in split_ids:
        for participant in PARTICIPANTS:
            for strategy, budget, stochastic in R5B_STRATEGIES:
                seeds = random_rows[["seed_index", "seed"]].itertuples(index=False) if stochastic else [(0, 0)]
                for seed_index, seed in seeds:
                    trajectory_id = f"R5B__{split_id}__{participant}__{strategy}__K{budget:02d}__S{int(seed_index):02d}"
                    rows.append(
                        {
                            "trajectory_id": trajectory_id,
                            "split_id": split_id,
                            "participant": participant,
                            "case_analysis_only": participant == "P07",
                            "classifier": "RIDGE_ALPHA_1",
                            "strategy": strategy,
                            "query_budget": budget,
                            "random_seed_index": int(seed_index),
                            "random_seed": int(seed),
                            "target_sessions": "1|2|3|4|5",
                            "expected_session_folds": 5,
                            "future_sessions_accessible": False,
                            "fixed_test_ever_queryable": False,
                            "scientific_role": "R5_LOCKED_TEMPORAL_SPLIT_SENSITIVITY",
                        }
                    )
    return pd.DataFrame(rows)


def build_drift_specification() -> pd.DataFrame:
    rows = [
        ("DRIFT_POPULATION", "P01-P06 inference; P07 descriptive case only", "Preserve the locked independent unit"),
        ("NO_ADAPTATION_HISTORY", "Session 0 repetitions 1-5 per true label only", "No current or future session enters the model"),
        ("NO_ADAPTATION_MODEL", "RIDGE_ALPHA_1 under the frozen float32 classical contract", "Match the primary classical engine"),
        ("PERFORMANCE_UNIT", "One repetition prediction from the arithmetic mean of 37 window decision-score vectors", "Match the frozen repetition endpoint"),
        ("EARLY_PERFORMANCE", "Accuracy over repetitions 1-5, balanced 5 per class", "Accuracy equals balanced accuracy by construction"),
        ("LATE_PERFORMANCE", "Accuracy over repetitions 6-10, balanced 5 per class", "Accuracy equals balanced accuracy by construction"),
        ("PERFORMANCE_DRIFT", "late accuracy minus early accuracy for each participant-session", "Negative values indicate worsening later in the session"),
        ("FEATURE_NORMALIZER", "Session-0 initial history only; log1p then per-channel z-score; epsilon=1e-8", "No target repetition is used for normalization"),
        ("REPETITION_EMBEDDING", "Per-channel mean normalized log1p RMS over valid windows; invalid channels excluded", "Uses Stage 1C-2 main validity mask"),
        ("CLASS_REFERENCE", "True-class centroid from session-0 repetitions 1-5", "True labels are allowed only in post hoc drift analysis, never selection"),
        ("FEATURE_DISTANCE", "Root-mean-square channel distance to the matching source-class centroid over jointly valid channels", "Channel-count normalized and finite"),
        ("ORDER_DISTANCE", "Mean class-conditioned distance at each repetition order 1-10", "Each order contributes seven classes"),
        ("FEATURE_DRIFT_SLOPE", "OLS slope of order-distance against repetition order within participant-session", "Positive values indicate increasing feature displacement"),
        ("INFERENCE_STAGE", "R5 exports participant-level estimands; locked tests and BCa intervals run only in R7", "No inferential test in R5A"),
    ]
    return pd.DataFrame(rows, columns=["component", "locked_implementation", "rationale"])


def synthetic_drift_tests() -> tuple[pd.DataFrame, dict[str, bool]]:
    rows = []
    for label in LABELS:
        for order in REPETITIONS:
            rows.append({"label": label, "repetition_order": order, "distance": 0.25 * order + label * 0.1})
    frame = pd.DataFrame(rows)
    order_distance = frame.groupby("repetition_order", sort=True)["distance"].mean().reset_index()
    observed_slope = float(np.polyfit(order_distance["repetition_order"], order_distance["distance"], 1)[0])
    zero_frame = frame.copy()
    zero_frame["distance"] = zero_frame["label"] * 0.1
    zero_order = zero_frame.groupby("repetition_order", sort=True)["distance"].mean().reset_index()
    zero_slope = float(np.polyfit(zero_order["repetition_order"], zero_order["distance"], 1)[0])
    early_truth = np.tile(np.asarray(LABELS, dtype=int), 5)
    early_pred = early_truth.copy()
    late_truth = np.tile(np.asarray(LABELS, dtype=int), 5)
    late_pred = late_truth.copy()
    late_pred[:7] = (late_pred[:7] + 1) % 7
    early_accuracy = float(np.mean(early_pred == early_truth))
    late_accuracy = float(np.mean(late_pred == late_truth))
    performance_delta = late_accuracy - early_accuracy
    audit = pd.DataFrame(
        [
            {"test": "KNOWN_POSITIVE_FEATURE_SLOPE", "observed": observed_slope, "expected": 0.25},
            {"test": "ZERO_FEATURE_SLOPE", "observed": zero_slope, "expected": 0.0},
            {"test": "KNOWN_LATE_MINUS_EARLY_ACCURACY", "observed": performance_delta, "expected": -0.2},
        ]
    )
    audit["absolute_difference"] = (audit["observed"] - audit["expected"]).abs()
    audit["passes"] = audit["absolute_difference"] < 1e-12
    gates = {
        "synthetic_positive_feature_slope_is_exact": abs(observed_slope - 0.25) < 1e-12,
        "synthetic_zero_feature_slope_is_zero": abs(zero_slope) < 1e-12,
        "synthetic_late_minus_early_accuracy_is_exact": abs(performance_delta + 0.2) < 1e-12,
    }
    return audit, gates


def manifest_for_directory(directory: Path, excluded: set[str] | None = None) -> pd.DataFrame:
    excluded = excluded or set()
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 100)
    print("REVISION R5A — TEMPORAL-SPLIT SCHEDULER AND DRIFT-DEFINITION UNIT TESTS")
    print("=" * 100)
    print("Execution device: CPU")
    print("Scientific role: IMPLEMENTATION UNIT TESTS AND ANALYSIS-DEFINITION LOCK ONLY")
    print("Raw HDF5 accessed: False")
    print("Feature-array payload values read: False")
    print("Model training: False")
    print("Fixed-test inference: False")
    print("New statistical tests: False")
    print()

    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    bootstrap_rclone()
    create_rclone_config()
    print("rclone version:", rclone(["version"]).stdout.splitlines()[0])
    print("Restoring R0, Stage 5B, and final R4E from locked direct Drive paths...")
    resolved = {}
    source_rows = []
    for basename in DIRECT_PACKETS:
        path, source = resolve_direct_packet(basename)
        resolved[basename] = path
        source_rows.append(
            {
                "packet": basename,
                "source": source,
                "sha256": sha256_file(path),
                "crc_passes": archive_crc_passes(path),
            }
        )

    r0_packet = resolved["stageR0_reviewer_revision_protocol_lock_packet.zip"]
    stage5b_packet = resolved["stage5b_deep_sequence_assembly_packet.zip"]
    r4e_packet = resolved["revision_R4E_lda_random_imbalance_packet.zip"]
    r0_report = read_json_member(r0_packet, "stageR0_protocol_lock_report.json")
    r4e_report = read_json_member(r4e_packet, "revision_R4E_final_report.json")
    splits = read_csv_member(r0_packet, "stageR0_temporal_split_schedule.csv")
    seed_schedule = read_csv_member(r0_packet, "stageR0_seed_schedule.csv")
    metadata = normalize_metadata(read_csv_member(stage5b_packet, "stage5b_repetition_metadata.csv"))

    split_audit, split_map = validate_locked_splits(splits)
    membership, membership_summary = build_membership(metadata, split_map)
    session0, causal_schedule = build_causal_schedule(metadata, list(split_map))
    execution_manifest = build_r5b_manifest(seed_schedule, list(split_map))
    drift_spec = build_drift_specification()
    synthetic_audit, synthetic_gates = synthetic_drift_tests()

    target_design = metadata.loc[
        metadata["participant"].isin(PARTICIPANTS)
        & metadata["session"].isin(TARGET_SESSIONS)
        & metadata["label"].isin(LABELS)
        & metadata["repetition"].isin(REPETITIONS)
    ].copy()
    target_key_columns = ["participant", "session", "label", "repetition"]
    membership_key_columns = ["split_id", "participant", "session", "label", "repetition"]
    overlap_check = (
        membership.groupby(membership_key_columns, sort=False)["temporal_role"].nunique().max()
        if len(membership)
        else 0
    )
    role_counts = membership.groupby(["split_id", "temporal_role"]).size().unstack(fill_value=0)
    class_counts = membership.groupby(["split_id", "participant", "session", "temporal_role", "label"]).size()
    random_seed_rows = seed_schedule.loc[
        seed_schedule["seed_family"].astype(str).eq("RANDOM_ACQUISITION")
        & seed_schedule["use_rule"].astype(str).eq("INITIAL_30")
    ]
    strategy_set = set(execution_manifest["strategy"].astype(str))
    original = membership.loc[membership["split_id"].eq("FIRST_HALF_ORIGINAL")]
    original_candidate = set(original.loc[original["temporal_role"].eq("CURRENT_SESSION_CANDIDATE"), "repetition"].unique())
    original_test = set(original.loc[original["temporal_role"].eq("FIXED_TEST_NEVER_QUERY"), "repetition"].unique())

    gates = {
        "revision_protocol_hash_matches_r0": r0_report.get("protocol_sha256") == REVISION_PROTOCOL_SHA256,
        "r0_parent_all_gates_passed": bool(r0_report.get("all_readiness_gates_passed")),
        "r4e_parent_all_gates_passed": bool(r4e_report.get("all_readiness_gates_passed")),
        "r4e_parent_decision_passes_to_r5": r4e_report.get("final_decision") == "PASS_TO_REVISION_R5_ALTERNATIVE_TEMPORAL_SPLITS_AND_DRIFT_AUDIT",
        "all_three_parent_hashes_match": all(row["sha256"] == DIRECT_PACKETS[row["packet"]][0] for row in source_rows),
        "all_three_parent_packets_pass_crc": all(bool(row["crc_passes"]) for row in source_rows),
        "metadata_has_2940_rows": len(metadata) == 2940,
        "metadata_repetition_keys_are_unique": not metadata.duplicated(target_key_columns).any(),
        "participants_are_exactly_p01_to_p07": set(metadata["participant"]) == set(PARTICIPANTS),
        "sessions_are_exactly_zero_to_five": set(metadata["session"]) == set(range(6)),
        "labels_are_exactly_zero_to_six": set(metadata["label"]) == set(LABELS),
        "target_design_has_2450_repetitions": len(target_design) == 2450,
        "four_locked_temporal_splits_are_exact": set(split_map) == set(EXPECTED_SPLITS) and split_audit["matches_locked_expected_definition"].all(),
        "all_split_candidate_and_test_sets_are_disjoint": split_audit["sets_are_disjoint"].all(),
        "all_split_candidate_and_test_sets_cover_one_to_ten": split_audit["union_is_repetitions_1_to_10"].all(),
        "temporal_membership_has_9800_rows": len(membership) == 9800,
        "temporal_membership_keys_are_unique": not membership.duplicated(membership_key_columns).any(),
        "each_membership_key_has_one_role": int(overlap_check) == 1,
        "every_split_has_1225_candidates": (role_counts.get("CURRENT_SESSION_CANDIDATE", pd.Series(dtype=int)) == 1225).all(),
        "every_split_has_1225_fixed_tests": (role_counts.get("FIXED_TEST_NEVER_QUERY", pd.Series(dtype=int)) == 1225).all(),
        "every_participant_session_role_has_five_per_class": len(class_counts) == 4 * 7 * 5 * 2 * 7 and int(class_counts.min()) == 5 and int(class_counts.max()) == 5,
        "original_split_matches_repetitions_1_to_5_vs_6_to_10": original_candidate == {1, 2, 3, 4, 5} and original_test == {6, 7, 8, 9, 10},
        "initial_session0_history_has_245_repetitions": len(session0) == 245,
        "each_participant_initial_history_has_35_repetitions": (session0.groupby("participant").size() == 35).all(),
        "causal_schedule_has_140_rows": len(causal_schedule) == 4 * 7 * 5,
        "no_future_session_is_accessible": not causal_schedule["future_session_accessible"].any() and (causal_schedule["maximum_history_session"] < causal_schedule["target_session"]).all(),
        "fixed_test_never_enters_history": not causal_schedule["current_fixed_test_enters_history"].any(),
        "r5b_manifest_has_924_trajectories": len(execution_manifest) == 924,
        "r5b_manifest_has_4620_expected_folds": int(execution_manifest["expected_session_folds"].sum()) == 4620,
        "r5b_trajectory_ids_are_unique": execution_manifest["trajectory_id"].is_unique,
        "r5b_strategy_set_is_exact": strategy_set == {item[0] for item in R5B_STRATEGIES},
        "random_initial_seed_count_is_30": len(random_seed_rows) == 30 and set(pd.to_numeric(random_seed_rows["seed_index"]).astype(int)) == set(range(1, 31)),
        "p07_is_case_analysis_only": execution_manifest.loc[execution_manifest["participant"].eq("P07"), "case_analysis_only"].all() and not execution_manifest.loc[execution_manifest["participant"].isin(ABLE_BODIED), "case_analysis_only"].any(),
        "drift_implementation_spec_has_fourteen_components": len(drift_spec) == 14,
        "performance_drift_is_late_minus_early": drift_spec.loc[drift_spec["component"].eq("PERFORMANCE_DRIFT"), "locked_implementation"].str.contains("late accuracy minus early accuracy", case=False).all(),
        "feature_drift_is_order_slope": drift_spec.loc[drift_spec["component"].eq("FEATURE_DRIFT_SLOPE"), "locked_implementation"].str.contains("OLS slope", case=False).all(),
        **synthetic_gates,
        "raw_hdf5_data_was_not_accessed": True,
        "feature_array_payload_values_were_not_read": True,
        "no_model_was_trained": True,
        "no_fixed_test_inference_was_run": True,
        "no_new_statistical_test_was_run": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]
    if failed:
        raise RuntimeError(f"Revision R5A failed readiness gates: {failed}")

    source_frame = pd.DataFrame(source_rows)
    atomic_csv(source_frame, RESULT_ROOT / "revision_R5A_input_packet_audit.csv")
    atomic_csv(split_audit, RESULT_ROOT / "revision_R5A_locked_temporal_split_audit.csv")
    atomic_csv(membership, RESULT_ROOT / "revision_R5A_temporal_split_membership.csv")
    atomic_csv(membership_summary, RESULT_ROOT / "revision_R5A_temporal_split_fold_summary.csv")
    atomic_csv(causal_schedule, RESULT_ROOT / "revision_R5A_causal_history_schedule.csv")
    atomic_csv(execution_manifest, RESULT_ROOT / "revision_R5A_R5B_execution_manifest.csv")
    atomic_csv(drift_spec, RESULT_ROOT / "revision_R5A_drift_implementation_specification.csv")
    atomic_csv(synthetic_audit, RESULT_ROOT / "revision_R5A_synthetic_drift_unit_tests.csv")

    source_name = globals().get("__file__")
    if not source_name or not Path(source_name).is_file():
        raise RuntimeError("R5A executed source could not be captured")
    shutil.copy2(Path(source_name), RESULT_ROOT / "revision_R5A_executed_source.py")

    report = {
        "stage": "REVISION_R5A_TEMPORAL_SPLIT_SCHEDULER_AND_DRIFT_DEFINITION_UNIT_TESTS",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "parent_packet_sha256": {key: value[0] for key, value in DIRECT_PACKETS.items()},
        "temporal_splits": len(split_map),
        "membership_rows": len(membership),
        "r5b_expected_trajectories": len(execution_manifest),
        "r5b_expected_folds": int(execution_manifest["expected_session_folds"].sum()),
        "drift_definition_components": len(drift_spec),
        "raw_hdf5_accessed": False,
        "feature_array_payload_values_read": False,
        "model_training_performed": False,
        "fixed_test_inference_performed": False,
        "new_statistical_test_performed": False,
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "final_decision": "PASS_TO_REVISION_R5B_RIDGE_TEMPORAL_SPLIT_SENSITIVITY",
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
    }
    atomic_json(report, RESULT_ROOT / "revision_R5A_final_report.json")
    output_manifest = manifest_for_directory(RESULT_ROOT, {"revision_R5A_output_manifest.csv"})
    atomic_csv(output_manifest, RESULT_ROOT / "revision_R5A_output_manifest.csv")
    packet_crc = make_zip(RESULT_ROOT, PACKET_PATH, "Revision_R5A_Temporal_Split_Drift_Unit_Tests")
    packet_sha = sha256_file(PACKET_PATH)
    remote_verified = roundtrip_remote(PACKET_PATH, REMOTE_OUTPUT + "/" + PACKET_PATH.name, packet_sha)
    cleanup_secret()

    print()
    print("=" * 100)
    print("REVISION R5A — UNIT-TEST SUMMARY")
    print("=" * 100)
    print(split_audit.to_string(index=False))
    print()
    print("Temporal membership rows:", len(membership))
    print("R5B planned trajectories:", len(execution_manifest))
    print("R5B planned folds:", int(execution_manifest["expected_session_folds"].sum()))
    print("Drift definition components:", len(drift_spec))
    print("Failed readiness gates:", failed if failed else "None")
    print("Packet CRC pass:", packet_crc)
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", packet_sha)
    print("Remote round-trip verified:", remote_verified)
    print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 3))
    if not packet_crc or not remote_verified:
        raise RuntimeError("Revision R5A packet persistence failed")
    print()
    print("FINAL DECISION: PASS_TO_REVISION_R5B_RIDGE_TEMPORAL_SPLIT_SENSITIVITY")


if __name__ == "__main__":
    main()
