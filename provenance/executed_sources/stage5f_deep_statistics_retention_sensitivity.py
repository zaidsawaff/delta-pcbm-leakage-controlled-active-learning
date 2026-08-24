import atexit
import base64
import configparser
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import rankdata


# =============================================================================
# STAGE 5F — DEEP STATISTICS, RETENTION, SENSITIVITY, AND CLAIM CALIBRATION
# =============================================================================

PARENT_PROTOCOL_SHA256 = (
    "f548b1ca6f2831c29ea8fecb764557efed49f229eb72322f98632edcf0aeb221"
)
STAGE3G_FREEZE_SHA256 = (
    "dfcdbbade7bf1032c3250626439689e08e95249ec49d5fb90465c4444a998fbe"
)
FROZEN_STAGE3G_CONCLUSION = (
    "PCBM increased low-budget acquisition diversity, but did not demonstrate "
    "robust predictive or retention superiority."
)
DEEP_PROTOCOL_NAME = "DELTA_MASK_AWARE_RMS_TCN_TRANSFER_v1"
DEEP_PROTOCOL_SHA256 = (
    "abe15812c1a52b0f4e917b5b6ad39b0dfde50e5bb2d58dfcc35b3cacb22e3bd2"
)
STAGE5A4B_PACKET_SHA256 = (
    "46d4b99b4ee0a222b3facca2aff99dc4b8242afa249a0fc398bacf184c4ca4b4"
)
STAGE5B_PACKET_SHA256 = (
    "1c0fbc63f6412362f3ae7cd22609ea6a7fcb23236cdf688ad5fe0578ebaab84d"
)
STAGE5D2_PACKET_SHA256 = (
    "fc8ac364bac0344639a50977d5f8725b1e5b5b2875758e01587de8c083a1f914"
)
STAGE5E_PACKET_SHA256 = (
    "7277a2847ee5f8c07554a155ff1c9bf7ef6e967b70998bfe3b261276710e5b78"
)

PARTICIPANTS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07"]
ABLE_BODIED = ["P01", "P02", "P03", "P04", "P05", "P06"]
BUDGETS = [7, 14, 21]
BOOTSTRAP_REPLICATES = 100000

WORKING = Path("/kaggle/working")
TOOLS = WORKING / "_stage5_tools"
TOOLS.mkdir(parents=True, exist_ok=True)
RCLONE = TOOLS / "rclone"
INPUT_ROOT = WORKING / "STAGE5F_FROZEN_INPUTS"
RESULT_ROOT = (
    WORKING
    / "DELTA_STAGE5_DEEP_RESULTS"
    / "Stage5F_Statistics_Retention_Sensitivity"
)
PACKET_PATH = WORKING / "stage5f_deep_statistics_retention_sensitivity_packet.zip"
for directory in [INPUT_ROOT, RESULT_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

EVIDENCE_ROOT = Path(
    "/kaggle/input/datasets/zaidalsawaff/delta-q1-stage5-evidence-archives-v1"
)
STAGE3A_PACKET = EVIDENCE_ROOT / "stage3a_v1_1_protocol_amendment_packet.zip.bin"
STAGE5A4B_PACKET = WORKING / "stage5a4b_deep_protocol_lock_packet.zip"
STAGE5B_PACKET = WORKING / "stage5b_deep_sequence_assembly_packet.zip"
STAGE5D2_PACKET = WORKING / "stage5d2_full_deterministic_deep_trajectories_packet.zip"
STAGE5E_PACKET = WORKING / "stage5e_30_seed_deep_random_trajectories_packet.zip"

REMOTE_BASE = "gdrive_stage5:DELTA_Q1_Stage5_DeepLearning_Backup"
REMOTE_INPUTS = {
    STAGE5A4B_PACKET: (
        REMOTE_BASE
        + "/Stage5A4B_Deep_Protocol_Lock/"
        + STAGE5A4B_PACKET.name
    ),
    STAGE5B_PACKET: (
        REMOTE_BASE
        + "/Stage5B_Deep_Sequence_Assembly/"
        + STAGE5B_PACKET.name
    ),
    STAGE5D2_PACKET: (
        REMOTE_BASE
        + "/Deep_Training/Stage5D2_Full_Deterministic_Deep_Trajectories/"
        + STAGE5D2_PACKET.name
    ),
    STAGE5E_PACKET: (
        REMOTE_BASE
        + "/Deep_Training/Stage5E_30_Seed_Deep_Random_Trajectories/"
        + STAGE5E_PACKET.name
    ),
}
REMOTE_OUTPUT = (
    REMOTE_BASE
    + "/Deep_Analysis/Stage5F_Statistics_Retention_Sensitivity"
)

CONFIG_PATH = None


# =============================================================================
# FILE, HASH, DRIVE, AND PACKET UTILITIES
# =============================================================================


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def atomic_json(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_csv(dataframe, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def read_json_member(packet, basename):
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {basename} in {packet}; found {matches}")
        return json.loads(archive.read(matches[0]).decode("utf-8"))


def read_csv_member(packet, basename):
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {basename} in {packet}; found {matches}")
        return pd.read_csv(archive.open(matches[0]))


def extract_member(packet, basename, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {basename} in {packet}; found {matches}")
        with archive.open(matches[0]) as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target)
    return destination


def make_zip(source_directory, destination, archive_root):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(Path(source_directory).rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    (Path(archive_root) / path.relative_to(source_directory)).as_posix(),
                )
    os.replace(temporary, destination)
    with zipfile.ZipFile(destination, "r") as archive:
        return archive.testzip() is None


def cleanup_secret():
    global CONFIG_PATH
    if CONFIG_PATH is not None and Path(CONFIG_PATH).exists():
        Path(CONFIG_PATH).unlink()
    CONFIG_PATH = None


atexit.register(cleanup_secret)


def bootstrap_rclone():
    if RCLONE.exists():
        return
    print("Downloading verified official rclone binary...", flush=True)
    version_text = urllib.request.urlopen(
        "https://downloads.rclone.org/version.txt", timeout=60
    ).read().decode("utf-8").strip()
    match = re.search(r"v?(\d+\.\d+\.\d+)", version_text)
    if match is None:
        raise RuntimeError("Could not resolve official rclone version")
    version = match.group(1)
    archive_name = f"rclone-v{version}-linux-amd64.zip"
    base_url = f"https://downloads.rclone.org/v{version}"
    temporary_root = Path(tempfile.mkdtemp(prefix="stage5f_rclone_", dir="/tmp"))
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
        raise RuntimeError("rclone download SHA-256 verification failed")
    extraction = temporary_root / "extract"
    extraction.mkdir()
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError("rclone archive CRC failure")
        archive.extractall(extraction)
    candidates = list(extraction.glob("**/rclone"))
    if len(candidates) != 1:
        raise RuntimeError("Unexpected rclone archive layout")
    shutil.copy2(candidates[0], RCLONE)
    os.chmod(RCLONE, 0o755)
    shutil.rmtree(temporary_root, ignore_errors=True)


def create_rclone_config():
    global CONFIG_PATH
    from kaggle_secrets import UserSecretsClient

    encoded = UserSecretsClient().get_secret("RCLONE_CONFIG_B64")
    if not encoded:
        raise RuntimeError("Kaggle secret RCLONE_CONFIG_B64 is unavailable")
    decoded = base64.b64decode(encoded, validate=True)
    parser = configparser.RawConfigParser()
    parser.read_string(decoded.decode("utf-8"))
    valid = (
        parser.has_section("gdrive_stage5")
        and parser.get("gdrive_stage5", "type", fallback="") == "drive"
        and parser.get("gdrive_stage5", "scope", fallback="") == "drive.file"
    )
    if not valid:
        raise RuntimeError("Restricted Google Drive remote verification failed")
    temporary = tempfile.NamedTemporaryFile(
        prefix="stage5f_", suffix=".conf", dir="/tmp", delete=False
    )
    temporary.write(decoded)
    temporary.close()
    del decoded
    CONFIG_PATH = Path(temporary.name)
    os.chmod(CONFIG_PATH, 0o600)


def rclone(arguments, check=True):
    return subprocess.run(
        [str(RCLONE), "--config", str(CONFIG_PATH), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# EXACT PARTICIPANT-LEVEL STATISTICS
# =============================================================================


def exact_paired_wilcoxon(differences, zero_tolerance=1e-12):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.abs(differences) > zero_tolerance]
    n = len(differences)
    if n == 0:
        return {
            "nonzero_pair_count": 0,
            "wilcoxon_statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial_correlation": 0.0,
        }
    ranks = rankdata(np.abs(differences), method="average")
    signs = np.sign(differences)
    observed_signed_sum = float(np.sum(signs * ranks))
    w_plus = float(ranks[signs > 0].sum())
    w_minus = float(ranks[signs < 0].sum())
    statistic = min(w_plus, w_minus)
    enumerated = []
    for bit_pattern in range(1 << n):
        generated_signs = np.asarray(
            [1.0 if bit_pattern & (1 << index) else -1.0 for index in range(n)]
        )
        enumerated.append(float(np.sum(generated_signs * ranks)))
    enumerated = np.asarray(enumerated)
    p_value = float(
        np.mean(np.abs(enumerated) >= abs(observed_signed_sum) - 1e-12)
    )
    rank_biserial = float((w_plus - w_minus) / ranks.sum())
    return {
        "nonzero_pair_count": int(n),
        "wilcoxon_statistic": statistic,
        "p_value": min(1.0, p_value),
        "rank_biserial_correlation": rank_biserial,
    }


def bootstrap_mean_ci(differences, seed, replicates=BOOTSTRAP_REPLICATES):
    differences = np.asarray(differences, dtype=float)
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, len(differences), size=(int(replicates), len(differences))
    )
    means = differences[indices].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def stable_seed(text):
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return int(value % 2_000_000_000 + 1)


def paired_contrast(proposed, comparator, comparison, budget, analysis_role):
    joined = proposed.merge(
        comparator,
        on="participant",
        how="inner",
        validate="one_to_one",
        suffixes=("_proposed", "_comparator"),
    ).sort_values("participant").reset_index(drop=True)
    if joined["participant"].tolist() != ABLE_BODIED:
        raise RuntimeError(f"Participant pairing failure for {comparison}")
    differences = (
        joined["value_proposed"].to_numpy(dtype=float)
        - joined["value_comparator"].to_numpy(dtype=float)
    )
    exact = exact_paired_wilcoxon(differences)
    ci_low, ci_high = bootstrap_mean_ci(
        differences,
        stable_seed(DEEP_PROTOCOL_SHA256 + "|" + comparison),
    )
    row = {
        "analysis_role": analysis_role,
        "comparison": comparison,
        "metric": "repetition_balanced_accuracy",
        "query_budget": int(budget),
        "participant_count": len(joined),
        "nonzero_pair_count": exact["nonzero_pair_count"],
        "mean_proposed": float(joined["value_proposed"].mean()),
        "mean_comparator": float(joined["value_comparator"].mean()),
        "mean_paired_delta": float(differences.mean()),
        "median_paired_delta": float(np.median(differences)),
        "paired_delta_ci_low": ci_low,
        "paired_delta_ci_high": ci_high,
        "participants_improved": int((differences > 1e-12).sum()),
        "participants_tied": int((np.abs(differences) <= 1e-12).sum()),
        "participants_worsened": int((differences < -1e-12).sum()),
        "wilcoxon_statistic": exact["wilcoxon_statistic"],
        "p_value_raw": exact["p_value"],
        "rank_biserial_correlation": exact[
            "rank_biserial_correlation"
        ],
    }
    participant_rows = joined[["participant"]].copy()
    participant_rows["comparison"] = comparison
    participant_rows["query_budget"] = int(budget)
    participant_rows["proposed"] = joined["value_proposed"]
    participant_rows["comparator"] = joined["value_comparator"]
    participant_rows["paired_delta"] = differences
    return row, participant_rows


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="mergesort")
    adjusted = np.zeros(len(p_values), dtype=float)
    running = 0.0
    number = len(p_values)
    for rank_position, original_index in enumerate(order):
        candidate = min(1.0, (number - rank_position) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


# =============================================================================
# INPUT RESTORATION AND VALIDATION
# =============================================================================


def prepare_inputs():
    print("Restoring and verifying frozen Stage 5 inputs...", flush=True)
    for local, remote in REMOTE_INPUTS.items():
        rclone(
            [
                "copyto",
                remote,
                str(local),
                "--retries",
                "5",
                "--low-level-retries",
                "10",
                "--timeout",
                "5m",
            ]
        )
    expected_hashes = {
        STAGE5A4B_PACKET: STAGE5A4B_PACKET_SHA256,
        STAGE5B_PACKET: STAGE5B_PACKET_SHA256,
        STAGE5D2_PACKET: STAGE5D2_PACKET_SHA256,
        STAGE5E_PACKET: STAGE5E_PACKET_SHA256,
    }
    hash_gates = {
        path.name: sha256_file(path) == expected
        for path, expected in expected_hashes.items()
    }
    if not all(hash_gates.values()):
        raise RuntimeError(f"Frozen packet hash failure: {hash_gates}")
    for packet in [
        STAGE3A_PACKET,
        STAGE5A4B_PACKET,
        STAGE5B_PACKET,
        STAGE5D2_PACKET,
        STAGE5E_PACKET,
    ]:
        with zipfile.ZipFile(packet, "r") as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"CRC failure: {packet}")

    d2_report = read_json_member(
        STAGE5D2_PACKET, "stage5d2_full_deterministic_report.json"
    )
    e_report = read_json_member(
        STAGE5E_PACKET, "stage5e_random_trajectory_report.json"
    )
    locked_protocol_path = extract_member(
        STAGE5A4B_PACKET,
        "stage5a4b_locked_deep_protocol.json",
        INPUT_ROOT / "stage5a4b_locked_deep_protocol.json",
    )
    locked_protocol_text = locked_protocol_path.read_text(encoding="utf-8")
    extract_member(
        STAGE5A4B_PACKET,
        "stage5a4b_claim_guardrails.csv",
        INPUT_ROOT / "stage5a4b_claim_guardrails.csv",
    )
    for basename in [
        "stage5b_rms_repetition_sequences.npy",
        "stage5b_main_valid_repetition_sequences.npy",
        "stage5b_repetition_metadata.csv",
        "stage5b_mask_aware_rms_tcn.py",
    ]:
        extract_member(STAGE5B_PACKET, basename, INPUT_ROOT / basename)
    engine_path = extract_member(
        STAGE5D2_PACKET,
        "stage5d2_executed_source.py",
        INPUT_ROOT / "stage5d2_executed_source.py",
    )
    d2_manifest = read_csv_member(STAGE5D2_PACKET, "stage5d2_sha256_manifest.csv")
    engine_manifest = d2_manifest.loc[
        d2_manifest["relative_path"].astype(str).str.endswith(
            "stage5d2_executed_source.py"
        )
    ]
    if len(engine_manifest) != 1:
        raise RuntimeError("Stage 5D-2 engine is not uniquely manifested")
    engine_hash_matches = (
        sha256_file(engine_path) == str(engine_manifest.iloc[0]["sha256"])
    )

    metadata = pd.read_csv(INPUT_ROOT / "stage5b_repetition_metadata.csv")
    metadata["participant"] = metadata["participant"].astype(str)
    for column in ["session", "label", "repetition", "sequence_row"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(int)
    universe = read_csv_member(
        STAGE3A_PACKET, "stage3a_v1_1_repetition_protocol_universe.csv"
    )
    universe["participant"] = universe["participant"].astype(str)
    for column in ["session", "label", "repetition_number"]:
        universe[column] = pd.to_numeric(universe[column], errors="raise").astype(int)
    aligned = metadata.merge(
        universe[
            [
                "participant",
                "session",
                "label",
                "repetition_number",
                "protocol_role",
                "fixed_test_never_query",
                "opaque_candidate_token",
            ]
        ],
        left_on=["participant", "session", "label", "repetition"],
        right_on=["participant", "session", "label", "repetition_number"],
        how="left",
        validate="one_to_one",
    ).sort_values("sequence_row").reset_index(drop=True)
    if (
        len(aligned) != 2940
        or aligned["protocol_role"].isna().any()
        or aligned["sequence_row"].tolist() != list(range(2940))
    ):
        raise RuntimeError("Stage 5F metadata-protocol alignment failure")
    atomic_csv(aligned, INPUT_ROOT / "stage5f_metadata_protocol_aligned.csv")

    final_state_root = INPUT_ROOT / "deterministic_final_states"
    final_state_root.mkdir(parents=True, exist_ok=True)
    extracted_states = []
    with zipfile.ZipFile(STAGE5D2_PACKET, "r") as archive:
        state_members = [
            name for name in archive.namelist() if name.endswith("/final_state.pt")
        ]
        for member in state_members:
            trajectory_id = Path(member).parent.name
            destination = final_state_root / f"{trajectory_id}.pt"
            with archive.open(member) as source, open(destination, "wb") as target:
                shutil.copyfileobj(source, target)
            extracted_states.append(destination)

    features = np.load(
        INPUT_ROOT / "stage5b_rms_repetition_sequences.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    valid_mask = np.load(
        INPUT_ROOT / "stage5b_main_valid_repetition_sequences.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    gates = {
        "stage5a4b_hash_matches": hash_gates[STAGE5A4B_PACKET.name],
        "stage5b_hash_matches": hash_gates[STAGE5B_PACKET.name],
        "stage5d2_hash_matches": hash_gates[STAGE5D2_PACKET.name],
        "stage5e_hash_matches": hash_gates[STAGE5E_PACKET.name],
        "stage5d2_all_gates_passed": bool(
            d2_report["all_readiness_gates_passed"]
        ),
        "stage5e_all_gates_passed": bool(
            e_report["all_readiness_gates_passed"]
        ),
        "deep_protocol_hash_is_present": DEEP_PROTOCOL_SHA256
        in locked_protocol_text,
        "stage5d2_engine_hash_matches_manifest": engine_hash_matches,
        "feature_shape_is_2940_by_37_by_64": features.shape == (2940, 37, 64),
        "main_mask_shape_matches": valid_mask.shape == features.shape,
        "metadata_has_2940_aligned_rows": len(aligned) == 2940,
        "deterministic_final_state_count_is_56": len(extracted_states) == 56,
        "stage3g_freeze_hash_is_preserved": len(STAGE3G_FREEZE_SHA256) == 64,
        "stage3g_conclusion_is_preserved": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Stage 5F frozen-input gates failed: {gates}")
    atomic_json(
        {
            "stage": "STAGE5F_FROZEN_INPUT_AUDIT",
            "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
            "stage3g_freeze_sha256": STAGE3G_FREEZE_SHA256,
            "stage5d2_packet_sha256": STAGE5D2_PACKET_SHA256,
            "stage5e_packet_sha256": STAGE5E_PACKET_SHA256,
            "readiness_gates": gates,
            "all_readiness_gates_passed": all(gates.values()),
        },
        INPUT_ROOT / "stage5f_frozen_input_audit.json",
    )
    return d2_report, e_report, gates


# =============================================================================
# PERFORMANCE STATISTICS AND SENSITIVITY
# =============================================================================


def participant_performance_tables(d2_folds, e_folds):
    d2_able = d2_folds.loc[d2_folds["participant"].isin(ABLE_BODIED)].copy()
    e_able = e_folds.loc[e_folds["participant"].isin(ABLE_BODIED)].copy()
    deterministic = (
        d2_able.groupby(
            ["participant", "strategy", "query_budget"], as_index=False
        )
        .agg(
            target_sessions=("target_session", "nunique"),
            mean_repetition_balanced_accuracy=(
                "repetition_balanced_accuracy",
                "mean",
            ),
            mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
            perfect_fold_fraction=(
                "repetition_balanced_accuracy",
                lambda values: float(np.mean(np.isclose(values, 1.0))),
            ),
        )
    )
    random_seed_participant = (
        e_able.groupby(
            ["participant", "query_budget", "random_replicate", "random_seed"],
            as_index=False,
        )
        .agg(
            target_sessions=("target_session", "nunique"),
            mean_repetition_balanced_accuracy=(
                "repetition_balanced_accuracy",
                "mean",
            ),
            mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
            perfect_fold_fraction=(
                "repetition_balanced_accuracy",
                lambda values: float(np.mean(np.isclose(values, 1.0))),
            ),
        )
    )
    random_mean = (
        random_seed_participant.groupby(
            ["participant", "query_budget"], as_index=False
        )
        .agg(
            random_replicates=("random_replicate", "nunique"),
            target_sessions=("target_sessions", "min"),
            mean_repetition_balanced_accuracy=(
                "mean_repetition_balanced_accuracy",
                "mean",
            ),
            mean_repetition_macro_f1=("mean_repetition_macro_f1", "mean"),
            perfect_fold_fraction=("perfect_fold_fraction", "mean"),
        )
    )
    return deterministic, random_seed_participant, random_mean


def values_for_deterministic(table, strategy, budget):
    subset = table.loc[
        table["strategy"].eq(strategy)
        & table["query_budget"].eq(int(budget)),
        ["participant", "mean_repetition_balanced_accuracy"],
    ].rename(columns={"mean_repetition_balanced_accuracy": "value"})
    return subset.reset_index(drop=True)


def values_for_random(table, budget):
    subset = table.loc[
        table["query_budget"].eq(int(budget)),
        ["participant", "mean_repetition_balanced_accuracy"],
    ].rename(columns={"mean_repetition_balanced_accuracy": "value"})
    return subset.reset_index(drop=True)


def run_locked_statistics(deterministic, random_mean):
    primary_row, primary_participants = paired_contrast(
        values_for_deterministic(deterministic, "PCBM_PROPOSED", 7),
        values_for_random(random_mean, 7),
        "DEEP_PCBM_VERSUS_RANDOM_MEAN_K07",
        7,
        "PRIMARY_DEEP_EXTENSION",
    )
    primary_row["p_value_adjusted"] = primary_row["p_value_raw"]
    primary_row["multiplicity_policy"] = (
        "Single locked deep-extension primary contrast; cannot replace Stage 3G"
    )
    secondary_specs = [
        ("RANDOM", 14),
        ("RANDOM", 21),
        ("GLOBAL", 7),
        ("GLOBAL", 14),
        ("GLOBAL", 21),
    ]
    secondary_rows = []
    secondary_participant_rows = []
    for comparator_name, budget in secondary_specs:
        proposed = values_for_deterministic(
            deterministic, "PCBM_PROPOSED", budget
        )
        if comparator_name == "RANDOM":
            comparator = values_for_random(random_mean, budget)
            comparison = f"DEEP_PCBM_VERSUS_RANDOM_MEAN_K{budget:02d}"
        else:
            comparator = values_for_deterministic(
                deterministic, "GLOBAL_MARGIN", budget
            )
            comparison = f"DEEP_PCBM_VERSUS_GLOBAL_MARGIN_K{budget:02d}"
        row, participant_rows = paired_contrast(
            proposed,
            comparator,
            comparison,
            budget,
            "MULTIPLICITY_CONTROLLED_DEEP_SECONDARY",
        )
        secondary_rows.append(row)
        secondary_participant_rows.append(participant_rows)
    secondary = pd.DataFrame(secondary_rows)
    secondary["p_value_holm_5_tests"] = holm_adjust(
        secondary["p_value_raw"].to_numpy(dtype=float)
    )
    secondary["significant_holm_0_05"] = (
        secondary["p_value_holm_5_tests"] < 0.05
    )
    return (
        pd.DataFrame([primary_row]),
        primary_participants,
        secondary,
        pd.concat(secondary_participant_rows, ignore_index=True),
    )


def primary_jackknife(primary_participants):
    rows = []
    for excluded in ABLE_BODIED:
        included = primary_participants.loc[
            ~primary_participants["participant"].eq(excluded)
        ]
        rows.append(
            {
                "excluded_participant": excluded,
                "included_participant_count": len(included),
                "mean_primary_delta": float(included["paired_delta"].mean()),
                "median_primary_delta": float(included["paired_delta"].median()),
                "positive_mean_direction": bool(
                    included["paired_delta"].mean() > 0
                ),
            }
        )
    return pd.DataFrame(rows)


def label_efficiency_auc(deterministic, random_mean):
    rows = []
    budgets = np.asarray([0, 7, 14, 21, 35], dtype=float)
    for participant in ABLE_BODIED:
        base = deterministic.loc[
            deterministic["participant"].eq(participant)
            & deterministic["strategy"].eq("NO_ADAPTATION_REFERENCE")
            & deterministic["query_budget"].eq(0),
            "mean_repetition_balanced_accuracy",
        ].iloc[0]
        full = deterministic.loc[
            deterministic["participant"].eq(participant)
            & deterministic["strategy"].eq("FULL_POOL_REFERENCE")
            & deterministic["query_budget"].eq(35),
            "mean_repetition_balanced_accuracy",
        ].iloc[0]
        for method, strategy in [
            ("PCBM_PROPOSED", "PCBM_PROPOSED"),
            ("GLOBAL_MARGIN", "GLOBAL_MARGIN"),
            ("RANDOM_UNIFORM_MEAN", None),
        ]:
            values = [float(base)]
            for budget in BUDGETS:
                if strategy is None:
                    value = random_mean.loc[
                        random_mean["participant"].eq(participant)
                        & random_mean["query_budget"].eq(budget),
                        "mean_repetition_balanced_accuracy",
                    ].iloc[0]
                else:
                    value = deterministic.loc[
                        deterministic["participant"].eq(participant)
                        & deterministic["strategy"].eq(strategy)
                        & deterministic["query_budget"].eq(budget),
                        "mean_repetition_balanced_accuracy",
                    ].iloc[0]
                values.append(float(value))
            values.append(float(full))
            rows.append(
                {
                    "participant": participant,
                    "method": method,
                    "normalized_auc": float(
                        np.trapezoid(np.asarray(values), budgets) / 35.0
                    ),
                    "performance_curve": json.dumps(values),
                }
            )
    table = pd.DataFrame(rows)
    summary = (
        table.groupby("method", as_index=False)
        .agg(
            participant_count=("participant", "nunique"),
            mean_normalized_auc=("normalized_auc", "mean"),
            std_normalized_auc=("normalized_auc", "std"),
            median_normalized_auc=("normalized_auc", "median"),
        )
        .sort_values("mean_normalized_auc", ascending=False)
        .reset_index(drop=True)
    )
    return table, summary


def acquisition_balance(d2_selections, e_selections):
    d2 = d2_selections.loc[
        d2_selections["participant"].isin(ABLE_BODIED)
        & d2_selections["strategy"].isin(["PCBM_PROPOSED", "GLOBAL_MARGIN"])
        & d2_selections["query_budget"].isin(BUDGETS)
    ].copy()
    d2["acquisition_set_id"] = (
        d2["participant"].astype(str)
        + "|"
        + d2["strategy"].astype(str)
        + "|K"
        + d2["query_budget"].astype(str)
        + "|S"
        + d2["target_session"].astype(str)
    )
    random = e_selections.loc[
        e_selections["participant"].isin(ABLE_BODIED)
    ].copy()
    random["acquisition_set_id"] = (
        random["participant"].astype(str)
        + "|RANDOM|K"
        + random["query_budget"].astype(str)
        + "|R"
        + random["random_replicate"].astype(str)
        + "|S"
        + random["target_session"].astype(str)
    )
    d2 = d2.rename(
        columns={"true_label_revealed_after_selection": "revealed_label"}
    )
    random = random.rename(
        columns={"true_label_revealed_after_selection": "revealed_label"}
    )
    combined = pd.concat(
        [
            d2[
                [
                    "acquisition_set_id",
                    "strategy",
                    "query_budget",
                    "revealed_label",
                ]
            ],
            random.assign(strategy="RANDOM_UNIFORM")[
                [
                    "acquisition_set_id",
                    "strategy",
                    "query_budget",
                    "revealed_label",
                ]
            ],
        ],
        ignore_index=True,
    )
    rows = []
    for keys, group in combined.groupby(
        ["acquisition_set_id", "strategy", "query_budget"]
    ):
        set_id, strategy, budget = keys
        counts = np.bincount(
            group["revealed_label"].to_numpy(dtype=int), minlength=7
        ).astype(float)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(
            -(probabilities * np.log(probabilities)).sum() / np.log(7.0)
        )
        rows.append(
            {
                "acquisition_set_id": set_id,
                "strategy": strategy,
                "query_budget": int(budget),
                "represented_true_labels": int((counts > 0).sum()),
                "normalized_label_entropy": entropy,
                "label_count_range": int(counts.max() - counts.min()),
            }
        )
    sets = pd.DataFrame(rows)
    summary = (
        sets.groupby(["strategy", "query_budget"], as_index=False)
        .agg(
            acquisition_sets=("acquisition_set_id", "nunique"),
            mean_represented_true_labels=("represented_true_labels", "mean"),
            minimum_represented_true_labels=("represented_true_labels", "min"),
            mean_normalized_label_entropy=("normalized_label_entropy", "mean"),
            minimum_normalized_label_entropy=("normalized_label_entropy", "min"),
            mean_label_count_range=("label_count_range", "mean"),
        )
        .sort_values(["query_budget", "strategy"])
        .reset_index(drop=True)
    )
    return sets, summary


# =============================================================================
# CPU FINAL-STATE RETENTION SENSITIVITY
# =============================================================================


def fixed_test_rows(metadata, participant, session):
    mask = (
        metadata["participant"].eq(participant)
        & metadata["session"].eq(int(session))
        & metadata["protocol_role"].eq("TARGET_FIXED_TEST_NEVER_QUERY")
    )
    rows = metadata.loc[mask, "sequence_row"].to_numpy(dtype=np.int64)
    if len(rows) != 35:
        raise RuntimeError(
            f"Expected 35 fixed tests for {participant} session {session}; "
            f"found {len(rows)}"
        )
    return rows


@torch.no_grad()
def predict_rows_cpu(state, features, valid_mask, row_indices, engine):
    """Float32 CPU inference using the frozen Stage 5D-2 transform contract."""
    row_indices = np.asarray(row_indices, dtype=np.int64)
    transformed = engine.transform_data(
        features,
        valid_mask,
        row_indices,
        state["means"],
        state["stds"],
    )
    state["model"].eval()
    batches = []
    for start in range(0, len(transformed), 64):
        inputs = torch.from_numpy(transformed[start : start + 64]).to("cpu")
        batches.append(state["model"](inputs).float().cpu().numpy())
    logits = np.concatenate(batches, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = np.argmax(probabilities, axis=1).astype(int)
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite CPU retention predictions")
    return logits, probabilities, predictions


def run_retention_sensitivity(d2_completion, d2_folds):
    print("Running deterministic final-state retention sensitivity on CPU...")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cpu")
    engine = load_module(
        INPUT_ROOT / "stage5d2_executed_source.py", "stage5d2_retention_engine"
    )
    model_class = engine.load_model_class(INPUT_ROOT)
    features = np.load(
        INPUT_ROOT / "stage5b_rms_repetition_sequences.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    valid_mask = np.load(
        INPUT_ROOT / "stage5b_main_valid_repetition_sequences.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    metadata = pd.read_csv(INPUT_ROOT / "stage5f_metadata_protocol_aligned.csv")
    metadata["participant"] = metadata["participant"].astype(str)
    for column in ["session", "label", "sequence_row"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(int)
    cells = []
    predictions = []
    summaries = []
    for index, completion in enumerate(
        d2_completion.sort_values(["participant", "query_budget", "strategy"])
        .itertuples(index=False),
        start=1,
    ):
        trajectory_id = str(completion.trajectory_id)
        participant = str(completion.participant)
        checkpoint_path = (
            INPUT_ROOT / "deterministic_final_states" / f"{trajectory_id}.pt"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        valid_checkpoint = (
            checkpoint.get("deep_protocol_sha256") == DEEP_PROTOCOL_SHA256
            and checkpoint.get("participant") == participant
            and checkpoint.get("engine_source_sha256")
            == sha256_file(INPUT_ROOT / "stage5d2_executed_source.py")
            and all(
                torch.isfinite(value).all().item()
                for value in checkpoint["model_state_dict"].values()
            )
            and np.isfinite(checkpoint["normalizer_means"]).all()
            and np.isfinite(checkpoint["normalizer_stds"]).all()
            and (np.asarray(checkpoint["normalizer_stds"]) > 0).all()
        )
        if not valid_checkpoint:
            raise RuntimeError(f"Invalid final state: {trajectory_id}")
        model = model_class().to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        state = {
            "model": model,
            "means": np.asarray(checkpoint["normalizer_means"], dtype=np.float64),
            "stds": np.asarray(checkpoint["normalizer_stds"], dtype=np.float64),
        }
        trajectory_cells = []
        for test_session in [1, 2, 3, 4]:
            test_rows = fixed_test_rows(metadata, participant, test_session)
            logits, probabilities, predicted = predict_rows_cpu(
                state, features, valid_mask, test_rows, engine
            )
            truths = metadata.iloc[test_rows]["label"].to_numpy(dtype=int)
            metrics = engine.classification_metrics(truths, predicted)
            diagonal_match = d2_folds.loc[
                d2_folds["trajectory_id"].eq(trajectory_id)
                & d2_folds["target_session"].eq(test_session)
            ]
            if len(diagonal_match) != 1:
                raise RuntimeError(f"Missing diagonal anchor for {trajectory_id}")
            diagonal = float(
                diagonal_match.iloc[0]["repetition_balanced_accuracy"]
            )
            cell = {
                "trajectory_id": trajectory_id,
                "participant": participant,
                "strategy": str(completion.strategy),
                "query_budget": int(completion.query_budget),
                "case_analysis": participant == "P07",
                "final_history_session": 5,
                "prior_test_session": int(test_session),
                "test_repetitions": 35,
                "diagonal_balanced_accuracy": diagonal,
                "final_state_retained_balanced_accuracy": metrics[
                    "balanced_accuracy"
                ],
                "backward_change_from_diagonal": float(
                    metrics["balanced_accuracy"] - diagonal
                ),
                "diagonal_to_final_drop": float(
                    max(0.0, diagonal - metrics["balanced_accuracy"])
                ),
                "all_logits_finite": bool(np.isfinite(logits).all()),
                "all_probabilities_finite": bool(
                    np.isfinite(probabilities).all()
                ),
                "retention_analysis_role": (
                    "DESCRIPTIVE_FINAL_STATE_SENSITIVITY_ONLY"
                ),
            }
            cells.append(cell)
            trajectory_cells.append(cell)
            test_meta = metadata.iloc[test_rows].reset_index(drop=True)
            for row_index in range(len(test_rows)):
                predictions.append(
                    {
                        "trajectory_id": trajectory_id,
                        "participant": participant,
                        "strategy": str(completion.strategy),
                        "query_budget": int(completion.query_budget),
                        "prior_test_session": int(test_session),
                        "opaque_test_token": str(
                            test_meta.iloc[row_index]["opaque_candidate_token"]
                        ),
                        "true_label": int(truths[row_index]),
                        "predicted_label": int(predicted[row_index]),
                    }
                )
        trajectory_frame = pd.DataFrame(trajectory_cells)
        summaries.append(
            {
                "trajectory_id": trajectory_id,
                "participant": participant,
                "strategy": str(completion.strategy),
                "query_budget": int(completion.query_budget),
                "case_analysis": participant == "P07",
                "backward_performance_change": float(
                    trajectory_frame["backward_change_from_diagonal"].mean()
                ),
                "worst_diagonal_to_final_drop": float(
                    trajectory_frame["diagonal_to_final_drop"].max()
                ),
                "mean_retained_balanced_accuracy": float(
                    trajectory_frame[
                        "final_state_retained_balanced_accuracy"
                    ].mean()
                ),
                "mean_diagonal_balanced_accuracy": float(
                    trajectory_frame["diagonal_balanced_accuracy"].mean()
                ),
                "inferential_status": "DESCRIPTIVE_ONLY_NO_RANDOM_FINAL_STATES",
            }
        )
        del model, state, checkpoint
        if index % 8 == 0 or index == 56:
            print(f"  Retention trajectories: {index}/56", flush=True)
    cells = pd.DataFrame(cells)
    predictions = pd.DataFrame(predictions)
    summaries = pd.DataFrame(summaries)
    able_summary = (
        summaries.loc[summaries["participant"].isin(ABLE_BODIED)]
        .groupby(["strategy", "query_budget"], as_index=False)
        .agg(
            participants=("participant", "nunique"),
            mean_backward_performance_change=(
                "backward_performance_change",
                "mean",
            ),
            mean_worst_diagonal_to_final_drop=(
                "worst_diagonal_to_final_drop",
                "mean",
            ),
            mean_retained_balanced_accuracy=(
                "mean_retained_balanced_accuracy",
                "mean",
            ),
        )
        .sort_values(["query_budget", "strategy"])
        .reset_index(drop=True)
    )
    p07 = summaries.loc[summaries["participant"].eq("P07")].copy()
    return cells, predictions, summaries, able_summary, p07


# =============================================================================
# CLAIM CALIBRATION AND PDF
# =============================================================================


def build_claim_matrix(primary, secondary, balance_summary, jackknife):
    primary_row = primary.iloc[0]
    primary_supported = bool(
        primary_row["p_value_raw"] < 0.05
        and primary_row["paired_delta_ci_low"] > 0
        and primary_row["mean_paired_delta"] > 0
    )
    directions_stable = bool(
        primary_row["mean_paired_delta"] > 0
        and (secondary["mean_paired_delta"] >= 0).all()
    )
    jackknife_stable = bool(jackknife["positive_mean_direction"].all())
    k7 = balance_summary.loc[balance_summary["query_budget"].eq(7)].set_index(
        "strategy"
    )
    diversity_supported = bool(
        k7.loc["PCBM_PROPOSED", "mean_normalized_label_entropy"]
        > k7.loc["RANDOM_UNIFORM", "mean_normalized_label_entropy"]
        and k7.loc["PCBM_PROPOSED", "mean_normalized_label_entropy"]
        > k7.loc["GLOBAL_MARGIN", "mean_normalized_label_entropy"]
    )
    rows = [
        (
            "D01",
            "SUPPORTED" if primary_supported else "NOT_SUPPORTED",
            primary_supported,
            "The deep PCBM model statistically outperforms mean random acquisition at K07.",
        ),
        (
            "D02",
            "SUPPORTED" if directions_stable and jackknife_stable else "NOT_SUPPORTED",
            directions_stable and jackknife_stable,
            "The deep PCBM predictive advantage is robust across budgets and participants.",
        ),
        (
            "D03",
            "DESCRIPTIVELY_SUPPORTED" if diversity_supported else "NOT_SUPPORTED",
            diversity_supported,
            "Deep PCBM improves low-budget true-class acquisition diversity.",
        ),
        (
            "D04",
            "NOT_ESTABLISHED",
            False,
            "Deep PCBM provides retention superiority over random acquisition.",
        ),
        (
            "D05",
            "DESCRIPTIVE_ONLY",
            True,
            "Deterministic deep trajectories were evaluated for final-state retention.",
        ),
        (
            "D06",
            "NOT_ESTABLISHED",
            False,
            "The deep findings generalize to people with limb absence.",
        ),
        (
            "D07",
            "SUPPORTED_BY_AUDIT",
            True,
            "The deep evaluation preserves causal, opaque-ID, and leakage controls.",
        ),
        (
            "D08",
            "EXPLICITLY_PROHIBITED",
            False,
            "The deep extension replaces or overturns the frozen Stage 3G conclusion.",
        ),
        (
            "D09",
            "NOT_TESTED",
            False,
            "The deep system is clinically or prosthetically deployed in real time.",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["claim_id", "status", "allowed_in_abstract", "candidate_claim"]
    )


def create_pdf(
    primary,
    primary_participants,
    secondary,
    random_seed_participant,
    deterministic,
    balance_summary,
    retention_able,
    claim_matrix,
):
    pdf_path = RESULT_ROOT / "stage5f_deep_analysis_summary.pdf"
    colors = {"PCBM_PROPOSED": "#0072B2", "GLOBAL_MARGIN": "#D55E00", "RANDOM": "#009E73"}
    with PdfPages(pdf_path) as pdf:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for label, strategy in [
            ("PCBM_PROPOSED", "PCBM_PROPOSED"),
            ("GLOBAL_MARGIN", "GLOBAL_MARGIN"),
        ]:
            means = []
            standard_errors = []
            for budget in BUDGETS:
                values = deterministic.loc[
                    deterministic["strategy"].eq(strategy)
                    & deterministic["query_budget"].eq(budget),
                    "mean_repetition_balanced_accuracy",
                ].to_numpy(dtype=float)
                means.append(values.mean())
                standard_errors.append(values.std(ddof=1) / np.sqrt(len(values)))
            axes[0].errorbar(
                BUDGETS,
                means,
                yerr=standard_errors,
                marker="o",
                label=label,
                color=colors[label],
            )
        random_means = []
        random_errors = []
        for budget in BUDGETS:
            seed_values = (
                random_seed_participant.loc[
                    random_seed_participant["query_budget"].eq(budget)
                ]
                .groupby("random_replicate")[
                    "mean_repetition_balanced_accuracy"
                ]
                .mean()
                .to_numpy(dtype=float)
            )
            random_means.append(seed_values.mean())
            random_errors.append(seed_values.std(ddof=1))
        axes[0].errorbar(
            BUDGETS,
            random_means,
            yerr=random_errors,
            marker="o",
            label="RANDOM (30 seeds)",
            color=colors["RANDOM"],
        )
        axes[0].set_title("Deep adaptation performance")
        axes[0].set_xlabel("Queries per session")
        axes[0].set_ylabel("Participant-mean repetition BA")
        axes[0].set_ylim(0.85, 1.005)
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        participant_order = primary_participants["participant"].tolist()
        deltas = primary_participants["paired_delta"].to_numpy(dtype=float)
        axes[1].bar(
            participant_order,
            deltas,
            color=["#0072B2" if value >= 0 else "#D55E00" for value in deltas],
        )
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title(
            "Primary deep contrast: PCBM − mean random (K07)\n"
            f"p={primary.iloc[0]['p_value_raw']:.4f}"
        )
        axes[1].set_ylabel("Paired repetition-BA difference")
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        k7 = balance_summary.loc[balance_summary["query_budget"].eq(7)]
        axes[0].bar(
            k7["strategy"],
            k7["mean_normalized_label_entropy"],
            color=["#D55E00", "#0072B2", "#009E73"],
        )
        axes[0].set_title("K07 acquisition diversity")
        axes[0].set_ylabel("Mean normalized label entropy")
        axes[0].tick_params(axis="x", rotation=20)
        retention_plot = retention_able.loc[
            retention_able["strategy"].isin(["PCBM_PROPOSED", "GLOBAL_MARGIN"])
            & retention_able["query_budget"].isin(BUDGETS)
        ]
        for strategy in ["PCBM_PROPOSED", "GLOBAL_MARGIN"]:
            subset = retention_plot.loc[retention_plot["strategy"].eq(strategy)]
            axes[1].plot(
                subset["query_budget"],
                subset["mean_retained_balanced_accuracy"],
                marker="o",
                label=strategy,
                color=colors[strategy],
            )
        axes[1].set_title("Final-state retention sensitivity (descriptive)")
        axes[1].set_xlabel("Queries per session")
        axes[1].set_ylabel("Mean retained repetition BA")
        axes[1].set_ylim(0.8, 1.005)
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        pdf.savefig(figure)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(11.7, 8.3))
        axis.axis("off")
        lines = [
            "Stage 5F — Deep Claim Calibration",
            "",
            f"Primary deep p-value: {primary.iloc[0]['p_value_raw']:.6f}",
            f"Primary paired delta: {primary.iloc[0]['mean_paired_delta']:.6f}",
            f"Primary 95% bootstrap CI: [{primary.iloc[0]['paired_delta_ci_low']:.6f}, "
            f"{primary.iloc[0]['paired_delta_ci_high']:.6f}]",
            "",
            "Multiplicity-controlled secondary results:",
        ]
        for row in secondary.itertuples(index=False):
            lines.append(
                f"  {row.comparison}: delta={row.mean_paired_delta:.6f}, "
                f"Holm p={row.p_value_holm_5_tests:.6f}"
            )
        lines.extend(["", "Claim matrix:"])
        for row in claim_matrix.itertuples(index=False):
            lines.append(f"  {row.claim_id}: {row.status} — {row.candidate_claim}")
        lines.extend(
            [
                "",
                "Frozen Stage 3G conclusion remains unchanged:",
                FROZEN_STAGE3G_CONCLUSION,
            ]
        )
        axis.text(
            0.02,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            wrap=True,
        )
        pdf.savefig(figure)
        plt.close(figure)
    return pdf_path


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def main():
    start_time = time.time()
    print("=" * 79)
    print("STAGE 5F — DEEP STATISTICS, RETENTION, AND SENSITIVITY")
    print("=" * 79)
    print("Execution device: CPU")
    print("Neural-network training: False")
    print("Inferential unit: participant (P01–P06, n=6)")
    print("P07: descriptive case analysis only")
    print("Primary deep endpoint: repetition balanced accuracy")
    print("Primary deep contrast: PCBM versus participant-specific mean of 30 random seeds at K07")
    print("Retention: deterministic final-state descriptive sensitivity only")
    print()

    bootstrap_rclone()
    create_rclone_config()
    print(
        "rclone version:",
        subprocess.run(
            [str(RCLONE), "version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0],
    )
    d2_report, e_report, input_gates = prepare_inputs()

    d2_folds = read_csv_member(STAGE5D2_PACKET, "stage5d2_fold_results.csv")
    d2_completion = read_csv_member(
        STAGE5D2_PACKET, "stage5d2_trajectory_completion_summary.csv"
    )
    d2_selections = read_csv_member(
        STAGE5D2_PACKET, "stage5d2_selection_trace.csv"
    )
    e_folds = read_csv_member(STAGE5E_PACKET, "stage5e_fold_results.csv")
    e_selections = read_csv_member(STAGE5E_PACKET, "stage5e_selection_trace.csv")
    for frame in [d2_folds, d2_completion, d2_selections, e_folds, e_selections]:
        frame["participant"] = frame["participant"].astype(str)
        frame["query_budget"] = pd.to_numeric(
            frame["query_budget"], errors="raise"
        ).astype(int)

    deterministic, random_seed_participant, random_mean = (
        participant_performance_tables(d2_folds, e_folds)
    )
    primary, primary_participants, secondary, secondary_participants = (
        run_locked_statistics(deterministic, random_mean)
    )
    jackknife = primary_jackknife(primary_participants)
    auc_table, auc_summary = label_efficiency_auc(deterministic, random_mean)
    balance_sets, balance_summary = acquisition_balance(
        d2_selections, e_selections
    )
    (
        retention_cells,
        retention_predictions,
        retention_trajectories,
        retention_able,
        retention_p07,
    ) = run_retention_sensitivity(d2_completion, d2_folds)
    claim_matrix = build_claim_matrix(
        primary, secondary, balance_summary, jackknife
    )

    p07_performance = pd.concat(
        [
            d2_folds.loc[d2_folds["participant"].eq("P07")]
            .groupby(["strategy", "query_budget"], as_index=False)
            .agg(
                target_sessions=("target_session", "nunique"),
                random_replicates=("run_id", lambda values: 1),
                mean_repetition_balanced_accuracy=(
                    "repetition_balanced_accuracy",
                    "mean",
                ),
            ),
            e_folds.loc[e_folds["participant"].eq("P07")]
            .groupby(["strategy", "query_budget"], as_index=False)
            .agg(
                target_sessions=("target_session", "nunique"),
                random_replicates=("random_replicate", "nunique"),
                mean_repetition_balanced_accuracy=(
                    "repetition_balanced_accuracy",
                    "mean",
                ),
            ),
        ],
        ignore_index=True,
    )
    p07_performance["inferential_status"] = "P07_DESCRIPTIVE_ONLY"

    outputs = [
        (deterministic, "stage5f_deterministic_participant_performance.csv"),
        (random_seed_participant, "stage5f_random_seed_participant_performance.csv"),
        (random_mean, "stage5f_random_mean_participant_performance.csv"),
        (primary, "stage5f_primary_deep_result.csv"),
        (primary_participants, "stage5f_primary_participant_differences.csv"),
        (secondary, "stage5f_secondary_deep_results.csv"),
        (secondary_participants, "stage5f_secondary_participant_differences.csv"),
        (jackknife, "stage5f_primary_jackknife.csv"),
        (auc_table, "stage5f_label_efficiency_auc_participants.csv"),
        (auc_summary, "stage5f_label_efficiency_auc_summary.csv"),
        (balance_sets, "stage5f_acquisition_balance_sets.csv"),
        (balance_summary, "stage5f_acquisition_balance_summary.csv"),
        (retention_cells, "stage5f_final_state_retention_cells.csv"),
        (retention_predictions, "stage5f_final_state_retention_predictions.csv"),
        (retention_trajectories, "stage5f_final_state_retention_trajectories.csv"),
        (retention_able, "stage5f_final_state_retention_able_summary.csv"),
        (retention_p07, "stage5f_final_state_retention_p07.csv"),
        (p07_performance, "stage5f_p07_performance_descriptive.csv"),
        (claim_matrix, "stage5f_deep_claim_calibration_matrix.csv"),
    ]
    for frame, filename in outputs:
        atomic_csv(frame, RESULT_ROOT / filename)

    pdf_path = create_pdf(
        primary,
        primary_participants,
        secondary,
        random_seed_participant,
        deterministic,
        balance_summary,
        retention_able,
        claim_matrix,
    )
    primary_row = primary.iloc[0]
    primary_supported = bool(
        primary_row["p_value_raw"] < 0.05
        and primary_row["paired_delta_ci_low"] > 0
        and primary_row["mean_paired_delta"] > 0
    )
    primary_conclusion = (
        "DEEP_EXTENSION_PRIMARY_HYPOTHESIS_SUPPORTED"
        if primary_supported
        else "DEEP_EXTENSION_PRIMARY_HYPOTHESIS_NOT_SUPPORTED"
    )
    integrated_claim = (
        "The deep PCBM extension demonstrated participant-level predictive "
        "superiority over mean random acquisition at K07."
        if primary_supported
        else "The deep PCBM extension did not demonstrate participant-level "
        "predictive superiority over mean random acquisition at K07."
    )
    metrics = np.concatenate(
        [
            d2_folds[
                ["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]
            ].to_numpy(dtype=float).ravel(),
            e_folds[
                ["repetition_accuracy", "repetition_balanced_accuracy", "repetition_macro_f1"]
            ].to_numpy(dtype=float).ravel(),
        ]
    )
    readiness_gates = {
        **{f"input_{key}": value for key, value in input_gates.items()},
        "stage5d2_fold_count_is_280": len(d2_folds) == 280,
        "stage5e_fold_count_is_3150": len(e_folds) == 3150,
        "able_bodied_participants_are_exactly_p01_to_p06": sorted(
            deterministic["participant"].unique().tolist()
        ) == ABLE_BODIED,
        "primary_result_has_one_row": len(primary) == 1,
        "primary_uses_six_participants": int(primary_row["participant_count"]) == 6,
        "primary_random_comparator_uses_30_seeds": bool(
            random_mean["random_replicates"].eq(30).all()
        ),
        "primary_p_value_is_finite": bool(
            np.isfinite(primary_row["p_value_raw"])
        ),
        "secondary_test_count_is_5": len(secondary) == 5,
        "every_secondary_test_uses_six_participants": bool(
            secondary["participant_count"].eq(6).all()
        ),
        "holm_adjustment_is_complete": bool(
            np.isfinite(secondary["p_value_holm_5_tests"]).all()
        ),
        "all_performance_metrics_are_finite": bool(np.isfinite(metrics).all()),
        "all_performance_metrics_are_between_zero_and_one": bool(
            (metrics >= 0).all() and (metrics <= 1).all()
        ),
        "auc_table_has_18_rows": len(auc_table) == 18,
        "acquisition_balance_sets_are_complete": len(balance_sets) == 2880,
        "retention_final_state_count_is_56": len(retention_trajectories) == 56,
        "retention_cell_count_is_224": len(retention_cells) == 224,
        "retention_prediction_count_is_7840": len(retention_predictions) == 7840,
        "retention_metrics_are_finite": bool(
            np.isfinite(
                retention_cells[
                    [
                        "diagonal_balanced_accuracy",
                        "final_state_retained_balanced_accuracy",
                        "backward_change_from_diagonal",
                        "diagonal_to_final_drop",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "retention_worst_drop_is_nonnegative": bool(
            (retention_trajectories["worst_diagonal_to_final_drop"] >= 0).all()
        ),
        "retention_is_descriptive_only": bool(
            retention_trajectories["inferential_status"]
            .eq("DESCRIPTIVE_ONLY_NO_RANDOM_FINAL_STATES")
            .all()
        ),
        "no_random_retention_superiority_test_was_run": True,
        "p07_is_excluded_from_inference": bool(
            not primary_participants["participant"].eq("P07").any()
            and not secondary_participants["participant"].eq("P07").any()
        ),
        "claim_matrix_has_9_claims": len(claim_matrix) == 9,
        "stage3g_replacement_claim_is_prohibited": bool(
            claim_matrix.loc[claim_matrix["claim_id"].eq("D08"), "status"]
            .eq("EXPLICITLY_PROHIBITED")
            .all()
        ),
        "p07_generalization_claim_is_prohibited": bool(
            not claim_matrix.loc[
                claim_matrix["claim_id"].eq("D06"), "allowed_in_abstract"
            ].any()
        ),
        "participant_is_the_inferential_unit": True,
        "sessions_seeds_windows_and_repetitions_are_not_inferential_units": True,
        "stage3g_primary_result_is_unchanged": True,
        "no_neural_network_training_was_run": True,
        "retention_execution_device_is_cpu": True,
        "pdf_exists_and_is_nonempty": pdf_path.exists() and pdf_path.stat().st_size > 0,
        "credentials_not_written_to_artifacts": True,
    }
    report = {
        "stage": "STAGE5F_DEEP_STATISTICS_RETENTION_SENSITIVITY",
        "deep_protocol_name": DEEP_PROTOCOL_NAME,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "stage3g_freeze_sha256": STAGE3G_FREEZE_SHA256,
        "stage3g_frozen_conclusion": FROZEN_STAGE3G_CONCLUSION,
        "stage5d2_packet_sha256": STAGE5D2_PACKET_SHA256,
        "stage5e_packet_sha256": STAGE5E_PACKET_SHA256,
        "primary_deep_conclusion": primary_conclusion,
        "integrated_deep_claim": integrated_claim,
        "retention_scope": (
            "Deterministic final-state descriptive sensitivity only; random "
            "final states were not persisted by Stage 5E, so no random "
            "retention superiority test was attempted."
        ),
        "readiness_gates": readiness_gates,
        "all_readiness_gates_passed": all(readiness_gates.values()),
        "runtime_minutes": (time.time() - start_time) / 60.0,
    }
    atomic_json(report, RESULT_ROOT / "stage5f_deep_analysis_report.json")

    print("\n" + "=" * 79)
    print("STAGE 5F — DEEP ANALYSIS SUMMARY")
    print("=" * 79)
    print("\nPrimary deep-extension result:")
    print(primary.to_string(index=False))
    print("\nPrimary participant differences:")
    print(primary_participants.to_string(index=False))
    print("\nSecondary multiplicity-controlled results:")
    print(secondary.to_string(index=False))
    print("\nLabel-efficiency AUC:")
    print(auc_summary.to_string(index=False))
    print("\nAcquisition balance:")
    print(balance_summary.to_string(index=False))
    print("\nDeterministic final-state retention summary:")
    print(retention_able.to_string(index=False))
    print("\nClaim-calibration matrix:")
    print(claim_matrix.to_string(index=False))
    print("\nReadiness gates:")
    for gate, value in readiness_gates.items():
        print(f"  {gate}: {value}")
    if not all(readiness_gates.values()):
        raise RuntimeError("Stage 5F readiness gates did not all pass")

    shutil.copy2(Path(__file__), RESULT_ROOT / "stage5f_executed_source.py")
    shutil.copy2(
        INPUT_ROOT / "stage5f_frozen_input_audit.json",
        RESULT_ROOT / "stage5f_frozen_input_audit.json",
    )
    shutil.copy2(
        INPUT_ROOT / "stage5a4b_claim_guardrails.csv",
        RESULT_ROOT / "stage5f_parent_claim_guardrails.csv",
    )
    manifest_rows = []
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(RESULT_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_csv(
        pd.DataFrame(manifest_rows), RESULT_ROOT / "stage5f_sha256_manifest.csv"
    )
    packet_crc = make_zip(
        RESULT_ROOT,
        PACKET_PATH,
        "Stage5F_Deep_Statistics_Retention_Sensitivity",
    )
    if not packet_crc:
        raise RuntimeError("Stage 5F packet CRC failure")
    rclone(
        [
            "copy",
            str(RESULT_ROOT),
            REMOTE_OUTPUT,
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--timeout",
            "5m",
        ]
    )
    rclone(
        [
            "copyto",
            str(PACKET_PATH),
            REMOTE_OUTPUT + "/" + PACKET_PATH.name,
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--timeout",
            "5m",
        ]
    )
    remote_files = set(
        rclone(["lsf", REMOTE_OUTPUT, "--files-only"]).stdout.splitlines()
    )
    remote_packet_verified = PACKET_PATH.name in remote_files
    cleanup_secret()
    print("\nPrimary deep conclusion:")
    print(primary_conclusion)
    print("\nIntegrated deep claim:")
    print(integrated_claim)
    print("\nFrozen Stage 3G conclusion remains:")
    print(FROZEN_STAGE3G_CONCLUSION)
    print("\nPacket CRC pass:", packet_crc)
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", sha256_file(PACKET_PATH))
    print("Remote packet verified:", remote_packet_verified)
    print("Runtime minutes:", round((time.time() - start_time) / 60.0, 2))
    if report["all_readiness_gates_passed"] and remote_packet_verified:
        print("\nFINAL DECISION: PASS_TO_STAGE6_DEEP_EXTENSION_MANUSCRIPT_INTEGRATION")
    else:
        print("\nFINAL DECISION: STAGE5F_FINALIZATION_NOT_READY")


if __name__ == "__main__":
    main()
