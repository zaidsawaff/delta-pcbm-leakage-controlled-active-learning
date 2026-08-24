import argparse
import atexit
import base64
import configparser
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# STAGE 5E — 30-SEED DEEP RANDOM TRAJECTORIES
# =============================================================================

PARENT_PROTOCOL_SHA256 = (
    "f548b1ca6f2831c29ea8fecb764557efed49f229eb72322f98632edcf0aeb221"
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
STAGE5C_PACKET_SHA256 = (
    "85ea2e8a8440369a77d43f00b5d509ea2f2978d2a60ab2f24fb828ce9ca6b9d4"
)
STAGE5D2_PACKET_SHA256 = (
    "fc8ac364bac0344639a50977d5f8725b1e5b5b2875758e01587de8c083a1f914"
)
AMP_AMENDMENT_SHA256 = (
    "d303d3a7059855c95af1106db6ebf440d46a9ebe75165498748b178ee4b5c9a9"
)

PARTICIPANTS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07"]
ABLE_BODIED = ["P01", "P02", "P03", "P04", "P05", "P06"]
BUDGETS = [7, 14, 21]
BUDGET_TO_ROUNDS = {7: 1, 14: 2, 21: 3}
RANDOM_REPLICATES = 30
SELECTOR_COLUMNS = ["opaque_candidate_token", "predicted_label", "margin"]
FORBIDDEN_SELECTOR_COLUMNS = [
    "repetition_uid",
    "participant",
    "session",
    "label",
    "true_label",
    "repetition_number",
    "protocol_role",
]

EXPECTED_TRAJECTORIES = 630
EXPECTED_FOLDS = 3150
EXPECTED_PREDICTIONS = 110250
EXPECTED_SELECTIONS = 44100
EXPECTED_SELECTOR_CALLS = 6300
EXPECTED_CANDIDATE_AUDITS = 191100
EXPECTED_FIT_AUDITS = 6930

SYNC_INTERVAL_SECONDS = 300
STATUS_INTERVAL_SECONDS = 60

WORKING = Path("/kaggle/working")
TOOLS = WORKING / "_stage5_tools"
TOOLS.mkdir(parents=True, exist_ok=True)
RCLONE = TOOLS / "rclone"

INPUT_ROOT = WORKING / "STAGE5E_FROZEN_INPUTS"
RESULT_ROOT = (
    WORKING
    / "DELTA_STAGE5_DEEP_RESULTS"
    / "Stage5E_30_Seed_Deep_Random_Trajectories"
)
CACHE_ROOT = WORKING / "STAGE5E_STATE_CACHE"
PACKET_STAGING = WORKING / "STAGE5E_PACKET_STAGING"
PACKET_PATH = WORKING / "stage5e_30_seed_deep_random_trajectories_packet.zip"
for directory in [INPUT_ROOT, RESULT_ROOT, CACHE_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

EVIDENCE_ROOT = Path(
    "/kaggle/input/datasets/zaidalsawaff/delta-q1-stage5-evidence-archives-v1"
)
STAGE3A_PACKET = EVIDENCE_ROOT / "stage3a_v1_1_protocol_amendment_packet.zip.bin"
STAGE5A4B_PACKET = WORKING / "stage5a4b_deep_protocol_lock_packet.zip"
STAGE5B_PACKET = WORKING / "stage5b_deep_sequence_assembly_packet.zip"
STAGE5C_PACKET = WORKING / "stage5c1_dual_gpu_loso_pretraining_packet.zip"
STAGE5D2_PACKET = WORKING / "stage5d2_full_deterministic_deep_trajectories_packet.zip"

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
    STAGE5C_PACKET: (
        REMOTE_BASE
        + "/Deep_Training/Stage5C_LOSO_Pretraining/"
        + STAGE5C_PACKET.name
    ),
    STAGE5D2_PACKET: (
        REMOTE_BASE
        + "/Deep_Training/Stage5D2_Full_Deterministic_Deep_Trajectories/"
        + STAGE5D2_PACKET.name
    ),
}
REMOTE_OUTPUT = (
    REMOTE_BASE
    + "/Deep_Training/Stage5E_30_Seed_Deep_Random_Trajectories"
)

CONFIG_PATH = None

SELECTION_TRACE_COLUMNS = [
    "trajectory_id",
    "participant",
    "target_session",
    "strategy",
    "query_budget",
    "random_replicate",
    "random_seed",
    "round_index",
    "query_subseed",
    "selection_order_in_round",
    "opaque_candidate_token",
    "sequence_row_internal",
    "true_label_revealed_after_selection",
    "selector_visible_columns",
]
CANDIDATE_AUDIT_COLUMNS = [
    "call_id",
    "trajectory_id",
    "participant",
    "target_session",
    "strategy",
    "query_budget",
    "random_replicate",
    "random_seed",
    "round_index",
    "query_subseed",
    "candidate_position",
    "opaque_candidate_token",
    "predicted_label",
    "margin",
    "selected_this_round",
    "true_label_visible_to_selector",
    "semantic_uid_visible_to_selector",
]
SELECTOR_AUDIT_COLUMNS = [
    "call_id",
    "trajectory_id",
    "participant",
    "query_budget",
    "random_replicate",
    "random_seed",
    "selector_name",
    "rows_received",
    "columns_received",
    "exact_schema",
    "forbidden_columns_present",
    "tokens_are_opaque_hex",
    "true_labels_used_for_selection",
]
FIT_AUDIT_COLUMNS = [
    "participant",
    "strategy",
    "query_budget",
    "random_replicate",
    "random_seed",
    "target_session",
    "round_index",
    "history_repetitions",
    "history_fingerprint",
    "history_seed",
    "maximum_history_session",
    "minimum_normalizer_count",
    "all_normalizer_values_finite",
    "all_normalizer_stds_positive",
    "fixed_test_in_history",
    "cache_source",
    "target_epochs",
    "final_train_loss",
]
FOLD_RESULT_COLUMNS = [
    "run_id",
    "trajectory_id",
    "participant",
    "target_session",
    "strategy",
    "query_budget",
    "random_replicate",
    "random_seed",
    "case_analysis",
    "source_repetitions",
    "test_repetitions",
    "repetition_accuracy",
    "repetition_balanced_accuracy",
    "repetition_macro_f1",
    "repetition_confusion_matrix",
    "all_logits_finite",
    "all_probabilities_finite",
    "maximum_history_session",
    "fixed_test_used_for_training",
    "fixed_test_used_for_normalization",
    "future_session_used",
]
PREDICTION_COLUMNS = [
    "run_id",
    "trajectory_id",
    "participant",
    "target_session",
    "strategy",
    "query_budget",
    "random_replicate",
    "random_seed",
    "opaque_test_token",
    "true_label",
    "predicted_label",
] + [f"logit_label_{label}" for label in range(7)] + [
    f"probability_label_{label}" for label in range(7)
]


# =============================================================================
# GENERIC UTILITIES
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
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


def atomic_torch_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
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


def extract_member_by_basename(packet, basename, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {basename} in {packet}; found {matches}")
        with archive.open(matches[0]) as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target)
    return destination


def extract_checkpoint(packet, participant, destination):
    # The frozen Stage 5C packet stores checkpoints as ``.../P01/best.pt``
    # through ``.../P07/best.pt``.  The basename alone is therefore shared by
    # all seven folds and cannot identify the target participant.
    suffix = f"/{participant}/best.pt"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one pretrained checkpoint for {participant}; "
                f"found {matches}"
            )
        with archive.open(matches[0]) as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target)
    return destination


def make_zip(source_directory, destination, archive_root):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(source_directory).rglob("*")):
            if path.is_file():
                arcname = Path(archive_root) / path.relative_to(source_directory)
                archive.write(path, arcname.as_posix())
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
    temporary_root = Path(tempfile.mkdtemp(prefix="stage5e_rclone_", dir="/tmp"))
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
    if not parser.has_section("gdrive_stage5"):
        raise RuntimeError("Restricted remote gdrive_stage5 is unavailable")
    if parser.get("gdrive_stage5", "type", fallback="") != "drive":
        raise RuntimeError("gdrive_stage5 is not a Google Drive remote")
    if parser.get("gdrive_stage5", "scope", fallback="") != "drive.file":
        raise RuntimeError("Google Drive scope is not restricted to drive.file")
    temporary = tempfile.NamedTemporaryFile(
        prefix="stage5e_rclone_", suffix=".conf", delete=False
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


def load_engine(input_root):
    path = Path(input_root) / "stage5d2_executed_source.py"
    spec = importlib.util.spec_from_file_location("stage5d2_frozen_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_hash(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


RANDOM_IMPLEMENTATION_LOCK = {
    "stage": "STAGE5E_RANDOM_IMPLEMENTATION_LOCK",
    "scientific_strategy": "RANDOM_UNIFORM",
    "locked_seed_source": "stage3a_v1_1_random_seeds.csv",
    "replicates": 30,
    "sampling": "uniform_without_replacement",
    "round_size": 7,
    "candidate_order_before_sampling": "lexicographic_opaque_candidate_token",
    "subseed_payload": (
        "deep_protocol_sha256|locked_random_seed|participant|query_budget|"
        "target_session|round_index"
    ),
    "subseed_rule": "first_16_hex_sha256_as_unsigned_integer",
    "selector_visible_schema": SELECTOR_COLUMNS,
    "true_labels_used_before_selection": False,
    "semantic_identifiers_used_before_selection": False,
    "training_engine": "exact Stage5D2 executed source",
    "scientific_method_changed": False,
    "stage3g_primary_result_changed": False,
}
RANDOM_IMPLEMENTATION_SHA256 = canonical_hash(RANDOM_IMPLEMENTATION_LOCK)


def derive_query_subseed(
    locked_random_seed, participant, query_budget, target_session, round_index
):
    payload = "|".join(
        [
            DEEP_PROTOCOL_SHA256,
            str(int(locked_random_seed)),
            str(participant),
            str(int(query_budget)),
            str(int(target_session)),
            str(int(round_index)),
        ]
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def select_random_uniform(visible_frame, query_subseed, engine):
    engine.validate_selector_frame(visible_frame)
    ordered = visible_frame.sort_values(
        "opaque_candidate_token", kind="mergesort"
    ).reset_index(drop=True)
    generator = np.random.default_rng(int(query_subseed))
    positions = generator.choice(len(ordered), size=7, replace=False)
    selected = ordered.iloc[positions]["opaque_candidate_token"].astype(str).tolist()
    if len(selected) != 7 or len(set(selected)) != 7:
        raise RuntimeError("Random selector did not return seven unique tokens")
    return selected


# =============================================================================
# FROZEN INPUT PREPARATION
# =============================================================================


def prepare_frozen_inputs():
    print("Restoring and verifying frozen inputs...", flush=True)
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
        STAGE5C_PACKET: STAGE5C_PACKET_SHA256,
        STAGE5D2_PACKET: STAGE5D2_PACKET_SHA256,
    }
    hash_gates = {
        path.name: sha256_file(path) == expected
        for path, expected in expected_hashes.items()
    }
    if not all(hash_gates.values()):
        raise RuntimeError(f"Frozen input hash failure: {hash_gates}")
    for packet in [
        STAGE3A_PACKET,
        STAGE5A4B_PACKET,
        STAGE5B_PACKET,
        STAGE5C_PACKET,
        STAGE5D2_PACKET,
    ]:
        with zipfile.ZipFile(packet, "r") as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"CRC failure: {packet}")

    stage5d2_report = read_json_member(
        STAGE5D2_PACKET, "stage5d2_full_deterministic_report.json"
    )
    if not stage5d2_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("Stage 5D-2 packet did not pass every readiness gate")

    for basename in [
        "stage5b_rms_repetition_sequences.npy",
        "stage5b_main_valid_repetition_sequences.npy",
        "stage5b_repetition_metadata.csv",
        "stage5b_mask_aware_rms_tcn.py",
        "stage5b_sequence_assembly_report.json",
    ]:
        extract_member_by_basename(STAGE5B_PACKET, basename, INPUT_ROOT / basename)
    for participant in PARTICIPANTS:
        extract_checkpoint(
            STAGE5C_PACKET,
            participant,
            INPUT_ROOT / "pretrained" / f"{participant}_best.pt",
        )
    engine_path = extract_member_by_basename(
        STAGE5D2_PACKET,
        "stage5d2_executed_source.py",
        INPUT_ROOT / "stage5d2_executed_source.py",
    )
    d2_manifest = read_csv_member(STAGE5D2_PACKET, "stage5d2_sha256_manifest.csv")
    source_manifest = d2_manifest.loc[
        d2_manifest["relative_path"].astype(str).str.endswith(
            "stage5d2_executed_source.py"
        )
    ]
    if len(source_manifest) != 1:
        raise RuntimeError("Stage 5D-2 executed source is not uniquely manifested")
    engine_source_hash_matches = (
        sha256_file(engine_path) == str(source_manifest.iloc[0]["sha256"])
    )
    if not engine_source_hash_matches:
        raise RuntimeError("Extracted Stage 5D-2 engine hash mismatch")

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
                "repetition_uid",
                "protocol_role",
                "eligible_for_query",
                "eligible_for_training",
                "fixed_test_never_query",
                "case_analysis",
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
        or aligned["repetition_uid"].isna().any()
        or aligned["sequence_row"].tolist() != list(range(2940))
    ):
        raise RuntimeError("Metadata-protocol alignment failure")
    atomic_csv(aligned, INPUT_ROOT / "stage5e_metadata_protocol_aligned.csv")

    seed_schedule = read_csv_member(
        STAGE5A4B_PACKET, "stage5a4b_training_seed_schedule.csv"
    )
    primary_seeds = seed_schedule.loc[
        seed_schedule["seed_role"].eq("PRIMARY_DETERMINISTIC"),
        ["participant", "seed"],
    ].copy()
    primary_seeds["participant"] = primary_seeds["participant"].astype(str)
    primary_seeds["seed"] = pd.to_numeric(
        primary_seeds["seed"], errors="raise"
    ).astype(np.int64)
    random_seeds = read_csv_member(
        STAGE3A_PACKET, "stage3a_v1_1_random_seeds.csv"
    )
    random_seeds["replicate_index"] = pd.to_numeric(
        random_seeds["replicate_index"], errors="raise"
    ).astype(int)
    random_seeds["random_seed"] = pd.to_numeric(
        random_seeds["random_seed"], errors="raise"
    ).astype(np.int64)
    random_seeds = random_seeds.sort_values("replicate_index").reset_index(drop=True)
    atomic_csv(primary_seeds, INPUT_ROOT / "stage5e_primary_seeds.csv")
    atomic_csv(random_seeds, INPUT_ROOT / "stage5e_locked_random_seeds.csv")
    atomic_json(
        {
            **RANDOM_IMPLEMENTATION_LOCK,
            "implementation_sha256": RANDOM_IMPLEMENTATION_SHA256,
        },
        INPUT_ROOT / "stage5e_random_implementation_lock.json",
    )

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
    engine = load_engine(INPUT_ROOT)
    synthetic_visible = pd.DataFrame(
        {
            "opaque_candidate_token": [f"{index:024x}" for index in range(35)],
            "predicted_label": [index % 7 for index in range(35)],
            "margin": np.linspace(0.01, 0.99, 35),
        },
        columns=SELECTOR_COLUMNS,
    )
    first_seed = int(random_seeds.iloc[0]["random_seed"])
    second_seed = int(random_seeds.iloc[1]["random_seed"])
    first_subseed = derive_query_subseed(first_seed, "P01", 7, 1, 1)
    second_subseed = derive_query_subseed(second_seed, "P01", 7, 1, 1)
    synthetic_first = select_random_uniform(
        synthetic_visible.copy(), first_subseed, engine
    )
    synthetic_repeat = select_random_uniform(
        synthetic_visible.copy(), first_subseed, engine
    )
    synthetic_different = select_random_uniform(
        synthetic_visible.copy(), second_subseed, engine
    )
    gates = {
        "parent_protocol_hash_is_preserved": len(PARENT_PROTOCOL_SHA256) == 64,
        "deep_protocol_hash_is_preserved": len(DEEP_PROTOCOL_SHA256) == 64,
        "stage5a4b_hash_matches": hash_gates[STAGE5A4B_PACKET.name],
        "stage5b_hash_matches": hash_gates[STAGE5B_PACKET.name],
        "stage5c_hash_matches": hash_gates[STAGE5C_PACKET.name],
        "stage5d2_hash_matches": hash_gates[STAGE5D2_PACKET.name],
        "stage5d2_all_gates_passed": bool(
            stage5d2_report["all_readiness_gates_passed"]
        ),
        "stage5d2_engine_source_hash_matches": engine_source_hash_matches,
        "feature_shape_is_2940_by_37_by_64": features.shape == (2940, 37, 64),
        "main_mask_shape_matches": valid_mask.shape == features.shape,
        "metadata_has_2940_aligned_rows": len(aligned) == 2940,
        "primary_seeds_are_complete": sorted(primary_seeds["participant"]) == PARTICIPANTS,
        "random_seed_count_is_30": len(random_seeds) == RANDOM_REPLICATES,
        "random_replicates_are_1_to_30": random_seeds[
            "replicate_index"
        ].tolist() == list(range(1, 31)),
        "locked_random_seeds_loaded_from_hashed_stage3a": True,
        "random_seeds_are_unique": random_seeds["random_seed"].nunique() == 30,
        "random_same_seed_is_reproducible": synthetic_first == synthetic_repeat,
        "random_different_seed_changes_selection": synthetic_first
        != synthetic_different,
        "synthetic_selector_schema_is_exact": synthetic_visible.columns.tolist()
        == SELECTOR_COLUMNS,
        "random_implementation_hash_is_valid": len(
            RANDOM_IMPLEMENTATION_SHA256
        ) == 64,
        "stage3g_primary_result_is_unchanged": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Stage 5E frozen-input gates failed: {gates}")
    atomic_json(
        {
            "stage": "STAGE5E_FROZEN_INPUT_AUDIT",
            "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
            "stage5d2_packet_sha256": STAGE5D2_PACKET_SHA256,
            "random_implementation_sha256": RANDOM_IMPLEMENTATION_SHA256,
            "readiness_gates": gates,
            "all_readiness_gates_passed": all(gates.values()),
        },
        INPUT_ROOT / "stage5e_frozen_input_audit.json",
    )
    return gates


# =============================================================================
# RANDOM TRAJECTORY WORKER
# =============================================================================


def trajectory_id_for(participant, budget, replicate):
    return f"{participant}_RANDOM_UNIFORM_K{budget:02d}_R{replicate:02d}"


def trajectory_directory_for(result_root, participant, budget, replicate):
    return (
        Path(result_root)
        / f"R{replicate:02d}"
        / participant
        / trajectory_id_for(participant, budget, replicate)
    )


def new_progress(participant, budget, replicate, random_seed, initial_rows):
    return {
        "version": 1,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "stage5d2_packet_sha256": STAGE5D2_PACKET_SHA256,
        "random_implementation_sha256": RANDOM_IMPLEMENTATION_SHA256,
        "trajectory_id": trajectory_id_for(participant, budget, replicate),
        "participant": participant,
        "strategy": "RANDOM_UNIFORM",
        "query_budget": int(budget),
        "random_replicate": int(replicate),
        "random_seed": int(random_seed),
        "case_analysis": participant == "P07",
        "history_rows": list(map(int, initial_rows)),
        "next_session": 1,
        "active_session": None,
        "completed_rounds_in_active_session": 0,
        "remaining_rows": [],
        "current_state_fingerprint": None,
        "selection_trace": [],
        "candidate_audit": [],
        "selector_audit": [],
        "fit_audit": [],
        "fold_results": [],
        "predictions": [],
    }


def load_or_create_progress(
    directory, participant, budget, replicate, random_seed, initial_rows
):
    progress_path = Path(directory) / "progress.json"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            valid = (
                progress.get("deep_protocol_sha256") == DEEP_PROTOCOL_SHA256
                and progress.get("stage5d2_packet_sha256") == STAGE5D2_PACKET_SHA256
                and progress.get("random_implementation_sha256")
                == RANDOM_IMPLEMENTATION_SHA256
                and progress.get("participant") == participant
                and int(progress.get("query_budget")) == int(budget)
                and int(progress.get("random_replicate")) == int(replicate)
                and int(progress.get("random_seed")) == int(random_seed)
            )
            if valid:
                return progress
        except Exception:
            pass
    return new_progress(participant, budget, replicate, random_seed, initial_rows)


def save_progress(progress, directory, state=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if state is not None:
        atomic_torch_save(state["checkpoint_payload"], directory / "current_state.pt")
        progress["current_state_fingerprint"] = state["fingerprint"]
    atomic_json(progress, directory / "progress.json")
    for key, columns, filename in [
        ("selection_trace", SELECTION_TRACE_COLUMNS, "selection_trace.csv"),
        ("candidate_audit", CANDIDATE_AUDIT_COLUMNS, "candidate_score_audit.csv"),
        ("selector_audit", SELECTOR_AUDIT_COLUMNS, "selector_schema_audit.csv"),
        ("fit_audit", FIT_AUDIT_COLUMNS, "fit_normalizer_audit.csv"),
        ("fold_results", FOLD_RESULT_COLUMNS, "fold_results.csv"),
        ("predictions", PREDICTION_COLUMNS, "repetition_predictions.csv"),
    ]:
        atomic_csv(pd.DataFrame(progress[key], columns=columns), directory / filename)


def append_fit_audit(progress, state, metadata, session, round_index):
    if any(
        row.get("history_fingerprint") == state["fingerprint"]
        for row in progress["fit_audit"]
    ):
        return
    history_meta = metadata.iloc[state["history_rows"]]
    progress["fit_audit"].append(
        {
            "participant": progress["participant"],
            "strategy": "RANDOM_UNIFORM",
            "query_budget": int(progress["query_budget"]),
            "random_replicate": int(progress["random_replicate"]),
            "random_seed": int(progress["random_seed"]),
            "target_session": int(session),
            "round_index": int(round_index),
            "history_repetitions": int(len(state["history_rows"])),
            "history_fingerprint": state["fingerprint"],
            "history_seed": int(state["seed"]),
            "maximum_history_session": int(history_meta["session"].max()),
            "minimum_normalizer_count": int(state["counts"].min()),
            "all_normalizer_values_finite": bool(
                np.isfinite(state["means"]).all()
                and np.isfinite(state["stds"]).all()
            ),
            "all_normalizer_stds_positive": bool((state["stds"] > 0).all()),
            "fixed_test_in_history": bool(
                history_meta["fixed_test_never_query"].astype(bool).any()
            ),
            "cache_source": state["cache_source"],
            "target_epochs": 40,
            "final_train_loss": float(state["train_losses"][-1]),
        }
    )


def state_for_history(
    progress,
    session,
    round_index,
    history,
    features,
    valid_mask,
    metadata,
    primary_seeds,
    pretrained_path,
    model_class,
    device,
    cache_root,
    trajectory_directory,
    engine,
):
    participant = progress["participant"]
    status = (
        f"{participant} RANDOM K{progress['query_budget']:02d} "
        f"R{progress['random_replicate']:02d} S{session:02d} Q{round_index:02d}"
    )
    state = engine.fit_or_load_history_state(
        participant,
        history,
        features,
        valid_mask,
        metadata,
        primary_seeds,
        pretrained_path,
        model_class,
        device,
        cache_root,
        Path(trajectory_directory) / "current_state.pt",
        status,
    )
    append_fit_audit(progress, state, metadata, session, round_index)
    return state


def evaluate_fixed_test(
    state, progress, features, valid_mask, metadata, session, device, engine
):
    participant = progress["participant"]
    test_rows = engine.fixed_test_rows(metadata, participant, session)
    if np.intersect1d(test_rows, state["history_rows"]).size:
        raise RuntimeError("Fixed test entered random trajectory history")
    logits, probabilities, predictions, _ = engine.predict_rows(
        state, features, valid_mask, test_rows, device
    )
    truths = metadata.iloc[test_rows]["label"].to_numpy(dtype=int)
    metrics = engine.classification_metrics(truths, predictions)
    run_id = f"{progress['trajectory_id']}_S{session:02d}"
    progress["fold_results"].append(
        {
            "run_id": run_id,
            "trajectory_id": progress["trajectory_id"],
            "participant": participant,
            "target_session": int(session),
            "strategy": "RANDOM_UNIFORM",
            "query_budget": int(progress["query_budget"]),
            "random_replicate": int(progress["random_replicate"]),
            "random_seed": int(progress["random_seed"]),
            "case_analysis": participant == "P07",
            "source_repetitions": int(len(state["history_rows"])),
            "test_repetitions": 35,
            "repetition_accuracy": metrics["accuracy"],
            "repetition_balanced_accuracy": metrics["balanced_accuracy"],
            "repetition_macro_f1": metrics["macro_f1"],
            "repetition_confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            "all_logits_finite": bool(np.isfinite(logits).all()),
            "all_probabilities_finite": bool(np.isfinite(probabilities).all()),
            "maximum_history_session": int(
                metadata.iloc[state["history_rows"]]["session"].max()
            ),
            "fixed_test_used_for_training": False,
            "fixed_test_used_for_normalization": False,
            "future_session_used": False,
        }
    )
    test_meta = metadata.iloc[test_rows].reset_index(drop=True)
    for index in range(len(test_rows)):
        row = {
            "run_id": run_id,
            "trajectory_id": progress["trajectory_id"],
            "participant": participant,
            "target_session": int(session),
            "strategy": "RANDOM_UNIFORM",
            "query_budget": int(progress["query_budget"]),
            "random_replicate": int(progress["random_replicate"]),
            "random_seed": int(progress["random_seed"]),
            "opaque_test_token": str(
                test_meta.iloc[index]["opaque_candidate_token"]
            ),
            "true_label": int(truths[index]),
            "predicted_label": int(predictions[index]),
        }
        for label in range(7):
            row[f"logit_label_{label}"] = float(logits[index, label])
            row[f"probability_label_{label}"] = float(
                probabilities[index, label]
            )
        progress["predictions"].append(row)


def complete_and_valid(directory, trajectory_id, replicate, random_seed):
    path = Path(directory) / "complete.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            payload.get("complete") is True
            and payload.get("trajectory_id") == trajectory_id
            and payload.get("deep_protocol_sha256") == DEEP_PROTOCOL_SHA256
            and payload.get("stage5d2_packet_sha256") == STAGE5D2_PACKET_SHA256
            and payload.get("random_implementation_sha256")
            == RANDOM_IMPLEMENTATION_SHA256
            and int(payload.get("random_replicate")) == int(replicate)
            and int(payload.get("random_seed")) == int(random_seed)
            and int(payload.get("fold_count")) == 5
            and payload.get("all_gates_passed") is True
        )
    except Exception:
        return False


def run_random_trajectory(
    participant,
    budget,
    replicate,
    random_seed,
    features,
    valid_mask,
    metadata,
    primary_seeds,
    pretrained_path,
    model_class,
    device,
    cache_root,
    result_root,
    engine,
):
    trajectory_id = trajectory_id_for(participant, budget, replicate)
    directory = trajectory_directory_for(
        result_root, participant, budget, replicate
    )
    directory.mkdir(parents=True, exist_ok=True)
    if complete_and_valid(directory, trajectory_id, replicate, random_seed):
        current_state = directory / "current_state.pt"
        if current_state.exists():
            current_state.unlink()
        print(f"SKIP COMPLETE | {trajectory_id}", flush=True)
        return

    initial_rows = engine.initial_history_rows(metadata, participant)
    progress = load_or_create_progress(
        directory,
        participant,
        budget,
        replicate,
        random_seed,
        initial_rows,
    )
    history = list(map(int, progress["history_rows"]))
    state = state_for_history(
        progress,
        max(0, int(progress["next_session"]) - 1),
        0,
        history,
        features,
        valid_mask,
        metadata,
        primary_seeds,
        pretrained_path,
        model_class,
        device,
        cache_root,
        directory,
        engine,
    )
    save_progress(progress, directory, state)
    rounds = BUDGET_TO_ROUNDS[int(budget)]

    while int(progress["next_session"]) <= 5:
        session = int(progress["next_session"])
        if progress["active_session"] is None:
            progress["active_session"] = session
            progress["completed_rounds_in_active_session"] = 0
            progress["remaining_rows"] = engine.candidate_rows(
                metadata, participant, session
            ).tolist()
            save_progress(progress, directory, state)
        elif int(progress["active_session"]) != session:
            raise RuntimeError("Inconsistent active random-session checkpoint")

        remaining = np.asarray(progress["remaining_rows"], dtype=np.int64)
        completed_rounds = int(progress["completed_rounds_in_active_session"])
        for round_index in range(completed_rounds + 1, rounds + 1):
            logits, probabilities, predictions, margins = engine.predict_rows(
                state, features, valid_mask, remaining, device
            )
            visible = pd.DataFrame(
                {
                    "opaque_candidate_token": metadata.iloc[remaining][
                        "opaque_candidate_token"
                    ].astype(str).to_numpy(),
                    "predicted_label": predictions,
                    "margin": margins,
                },
                columns=SELECTOR_COLUMNS,
            )
            engine.validate_selector_frame(visible)
            query_subseed = derive_query_subseed(
                random_seed, participant, budget, session, round_index
            )
            selected_tokens = select_random_uniform(visible, query_subseed, engine)
            selected_rows = engine.reveal_selected_rows(
                metadata, selected_tokens, remaining
            )
            new_history = history + selected_rows.tolist()
            new_state = state_for_history(
                progress,
                session,
                round_index,
                new_history,
                features,
                valid_mask,
                metadata,
                primary_seeds,
                pretrained_path,
                model_class,
                device,
                cache_root,
                directory,
                engine,
            )
            call_id = f"{trajectory_id}_S{session:02d}_Q{round_index:02d}"
            forbidden = sorted(
                set(visible.columns).intersection(FORBIDDEN_SELECTOR_COLUMNS)
            )
            progress["selector_audit"].append(
                {
                    "call_id": call_id,
                    "trajectory_id": trajectory_id,
                    "participant": participant,
                    "query_budget": int(budget),
                    "random_replicate": int(replicate),
                    "random_seed": int(random_seed),
                    "selector_name": "RANDOM_UNIFORM",
                    "rows_received": len(visible),
                    "columns_received": json.dumps(visible.columns.tolist()),
                    "exact_schema": visible.columns.tolist() == SELECTOR_COLUMNS,
                    "forbidden_columns_present": json.dumps(forbidden),
                    "tokens_are_opaque_hex": bool(
                        visible["opaque_candidate_token"]
                        .str.fullmatch(r"[0-9a-f]{24}")
                        .all()
                    ),
                    "true_labels_used_for_selection": False,
                }
            )
            selected_set = set(selected_tokens)
            for candidate_position, row in visible.reset_index(drop=True).iterrows():
                progress["candidate_audit"].append(
                    {
                        "call_id": call_id,
                        "trajectory_id": trajectory_id,
                        "participant": participant,
                        "target_session": int(session),
                        "strategy": "RANDOM_UNIFORM",
                        "query_budget": int(budget),
                        "random_replicate": int(replicate),
                        "random_seed": int(random_seed),
                        "round_index": int(round_index),
                        "query_subseed": int(query_subseed),
                        "candidate_position": int(candidate_position),
                        "opaque_candidate_token": row["opaque_candidate_token"],
                        "predicted_label": int(row["predicted_label"]),
                        "margin": float(row["margin"]),
                        "selected_this_round": row["opaque_candidate_token"]
                        in selected_set,
                        "true_label_visible_to_selector": False,
                        "semantic_uid_visible_to_selector": False,
                    }
                )
            for order, (token, row_index) in enumerate(
                zip(selected_tokens, selected_rows), start=1
            ):
                progress["selection_trace"].append(
                    {
                        "trajectory_id": trajectory_id,
                        "participant": participant,
                        "target_session": int(session),
                        "strategy": "RANDOM_UNIFORM",
                        "query_budget": int(budget),
                        "random_replicate": int(replicate),
                        "random_seed": int(random_seed),
                        "round_index": int(round_index),
                        "query_subseed": int(query_subseed),
                        "selection_order_in_round": int(order),
                        "opaque_candidate_token": token,
                        "sequence_row_internal": int(row_index),
                        "true_label_revealed_after_selection": int(
                            metadata.iloc[row_index]["label"]
                        ),
                        "selector_visible_columns": "|".join(SELECTOR_COLUMNS),
                    }
                )
            selected_row_set = set(map(int, selected_rows))
            remaining = np.asarray(
                [row for row in remaining if int(row) not in selected_row_set],
                dtype=np.int64,
            )
            history = new_history
            state = new_state
            progress["history_rows"] = history
            progress["remaining_rows"] = remaining.tolist()
            progress["completed_rounds_in_active_session"] = int(round_index)
            save_progress(progress, directory, state)
            del logits, probabilities

        evaluate_fixed_test(
            state,
            progress,
            features,
            valid_mask,
            metadata,
            session,
            device,
            engine,
        )
        expected_source = 35 + int(budget) * int(session)
        if len(history) != expected_source:
            raise RuntimeError(
                f"Source schedule mismatch for {trajectory_id} S{session}: "
                f"{len(history)} versus {expected_source}"
            )
        progress["next_session"] = session + 1
        progress["active_session"] = None
        progress["completed_rounds_in_active_session"] = 0
        progress["remaining_rows"] = []
        save_progress(progress, directory, state)
        print(
            f"SESSION COMPLETE | {trajectory_id} | S{session:02d} | "
            f"history={len(history)} | "
            f"BA={progress['fold_results'][-1]['repetition_balanced_accuracy']:.6f}",
            flush=True,
        )

    folds = pd.DataFrame(progress["fold_results"], columns=FOLD_RESULT_COLUMNS)
    selections = pd.DataFrame(
        progress["selection_trace"], columns=SELECTION_TRACE_COLUMNS
    )
    candidates = pd.DataFrame(
        progress["candidate_audit"], columns=CANDIDATE_AUDIT_COLUMNS
    )
    selectors = pd.DataFrame(
        progress["selector_audit"], columns=SELECTOR_AUDIT_COLUMNS
    )
    fits = pd.DataFrame(progress["fit_audit"], columns=FIT_AUDIT_COLUMNS)
    predictions_frame = pd.DataFrame(
        progress["predictions"], columns=PREDICTION_COLUMNS
    )
    expected_candidates = 5 * sum(
        35 - 7 * prior_round for prior_round in range(rounds)
    )
    metrics = folds[
        [
            "repetition_accuracy",
            "repetition_balanced_accuracy",
            "repetition_macro_f1",
        ]
    ].to_numpy(dtype=float)
    complete_gates = {
        "five_sessions_completed": len(folds) == 5,
        "every_fold_has_35_tests": bool(folds["test_repetitions"].eq(35).all()),
        "prediction_count_is_175": len(predictions_frame) == 175,
        "selection_count_matches_budget": len(selections) == int(budget) * 5,
        "selected_tokens_are_unique_within_session": bool(
            selections.groupby("target_session")["opaque_candidate_token"]
            .nunique()
            .eq(int(budget))
            .all()
        ),
        "selector_call_count_matches": len(selectors) == int(rounds) * 5,
        "candidate_audit_count_matches": len(candidates) == expected_candidates,
        "fit_audit_count_matches": len(fits) == 1 + int(rounds) * 5,
        "selector_schema_is_exact": bool(selectors["exact_schema"].all()),
        "no_forbidden_selector_columns": bool(
            selectors["forbidden_columns_present"].eq("[]").all()
        ),
        "no_true_label_used_for_selection": bool(
            not bool(selectors["true_labels_used_for_selection"].any())
        ),
        "all_metrics_are_finite": bool(np.isfinite(metrics).all()),
        "all_metrics_are_between_zero_and_one": bool(
            (metrics >= 0).all() and (metrics <= 1).all()
        ),
        "normalizers_are_finite": bool(
            fits["all_normalizer_values_finite"].all()
        ),
        "normalizer_stds_are_positive": bool(
            fits["all_normalizer_stds_positive"].all()
        ),
        "no_fixed_test_in_history": bool(
            not bool(fits["fixed_test_in_history"].any())
        ),
        "no_future_session_used": bool(
            (folds["maximum_history_session"] <= folds["target_session"]).all()
        ),
        "source_counts_match_schedule": bool(
            folds.apply(
                lambda row: int(row["source_repetitions"])
                == 35 + int(budget) * int(row["target_session"]),
                axis=1,
            ).all()
        ),
        "final_history_count_matches": len(history) == 35 + int(budget) * 5,
        "final_model_values_are_finite": all(
            torch.isfinite(value).all().item()
            for value in state["checkpoint_payload"]["model_state_dict"].values()
        ),
        "p07_case_flag_is_correct": bool(
            (participant == "P07") == progress["case_analysis"]
        ),
    }
    complete = {
        "complete": True,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "stage5d2_packet_sha256": STAGE5D2_PACKET_SHA256,
        "random_implementation_sha256": RANDOM_IMPLEMENTATION_SHA256,
        "trajectory_id": trajectory_id,
        "participant": participant,
        "strategy": "RANDOM_UNIFORM",
        "query_budget": int(budget),
        "random_replicate": int(replicate),
        "random_seed": int(random_seed),
        "case_analysis": participant == "P07",
        "fold_count": len(folds),
        "selection_count": len(selections),
        "candidate_audit_count": len(candidates),
        "selector_call_count": len(selectors),
        "prediction_count": len(predictions_frame),
        "fit_state_count": len(fits),
        "final_history_repetitions": len(history),
        "mean_repetition_balanced_accuracy": float(
            folds["repetition_balanced_accuracy"].mean()
        ),
        "readiness_gates": complete_gates,
        "all_gates_passed": all(complete_gates.values()),
    }
    if not complete["all_gates_passed"]:
        raise RuntimeError(f"Random trajectory gates failed: {trajectory_id}")
    atomic_json(complete, directory / "complete.json")
    current_state = directory / "current_state.pt"
    if current_state.exists():
        current_state.unlink()
    print(
        f"TRAJECTORY COMPLETE | {trajectory_id} | "
        f"mean_BA={complete['mean_repetition_balanced_accuracy']:.6f}",
        flush=True,
    )
    del state
    torch.cuda.empty_cache()


def worker_main(arguments):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Isolated worker expected one visible GPU; found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_properties(0).name
    if "T4" not in gpu_name:
        raise RuntimeError(f"Expected Tesla T4; observed {gpu_name}")
    input_root = Path(arguments.input_root)
    result_root = Path(arguments.result_root)
    cache_root = Path(arguments.cache_root)
    tasks = pd.read_csv(arguments.task_file)
    features = np.load(
        input_root / "stage5b_rms_repetition_sequences.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    valid_mask = np.load(
        input_root / "stage5b_main_valid_repetition_sequences.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    metadata = pd.read_csv(input_root / "stage5e_metadata_protocol_aligned.csv")
    metadata["participant"] = metadata["participant"].astype(str)
    for column in ["session", "label", "repetition", "sequence_row"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(int)
    primary_seed_frame = pd.read_csv(input_root / "stage5e_primary_seeds.csv")
    primary_seeds = dict(
        zip(primary_seed_frame["participant"], primary_seed_frame["seed"])
    )
    engine = load_engine(input_root)
    model_class = engine.load_model_class(input_root)
    print(
        f"WORKER START | physical_gpu={arguments.physical_gpu_label} | "
        f"visible_gpu={gpu_name} | participant-seed tasks={len(tasks)}",
        flush=True,
    )
    for task in tasks.itertuples(index=False):
        participant = str(task.participant)
        replicate = int(task.random_replicate)
        random_seed = int(task.random_seed)
        pretrained_path = input_root / "pretrained" / f"{participant}_best.pt"
        for budget in BUDGETS:
            run_random_trajectory(
                participant,
                budget,
                replicate,
                random_seed,
                features,
                valid_mask,
                metadata,
                primary_seeds,
                pretrained_path,
                model_class,
                device,
                cache_root,
                result_root,
                engine,
            )
    print(
        f"WORKER COMPLETE | physical_gpu={arguments.physical_gpu_label}",
        flush=True,
    )


# =============================================================================
# AGGREGATION AND ORCHESTRATION
# =============================================================================


def last_log_line(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return "NO LOG OUTPUT YET"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-1] if lines else "EMPTY LOG"


def build_task_manifest():
    random_seeds = pd.read_csv(INPUT_ROOT / "stage5e_locked_random_seeds.csv")
    rows = []
    task_index = 0
    for replicate_row in random_seeds.itertuples(index=False):
        for participant in PARTICIPANTS:
            rows.append(
                {
                    "task_index": task_index,
                    "participant": participant,
                    "random_replicate": int(replicate_row.replicate_index),
                    "random_seed": int(replicate_row.random_seed),
                    "physical_gpu": task_index % 2,
                }
            )
            task_index += 1
    manifest = pd.DataFrame(rows)
    if (
        len(manifest) != 210
        or manifest.groupby("physical_gpu").size().to_dict() != {0: 105, 1: 105}
    ):
        raise RuntimeError("Stage 5E GPU task manifest is not balanced")
    atomic_csv(manifest, INPUT_ROOT / "stage5e_gpu_task_manifest.csv")
    for gpu in [0, 1]:
        atomic_csv(
            manifest.loc[manifest["physical_gpu"].eq(gpu)].reset_index(drop=True),
            INPUT_ROOT / f"stage5e_gpu{gpu}_tasks.csv",
        )
    return manifest


def read_nonempty_frames(pattern):
    paths = [path for path in sorted(RESULT_ROOT.rglob(pattern)) if path.stat().st_size]
    if not paths:
        raise RuntimeError(f"No files found for {pattern}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def aggregate_and_finalize(worker_exit_codes, sync_successes, sync_failures, start_time):
    completion_paths = sorted(RESULT_ROOT.rglob("complete.json"))
    completion = pd.DataFrame(
        [json.loads(path.read_text(encoding="utf-8")) for path in completion_paths]
    )
    folds = read_nonempty_frames("fold_results.csv")
    selections = read_nonempty_frames("selection_trace.csv")
    candidates = read_nonempty_frames("candidate_score_audit.csv")
    selectors = read_nonempty_frames("selector_schema_audit.csv")
    fits = read_nonempty_frames("fit_normalizer_audit.csv")
    predictions = read_nonempty_frames("repetition_predictions.csv")

    aggregate_files = [
        (completion, "stage5e_trajectory_completion_summary.csv"),
        (folds, "stage5e_fold_results.csv"),
        (selections, "stage5e_selection_trace.csv"),
        (candidates, "stage5e_candidate_score_audit.csv"),
        (selectors, "stage5e_selector_schema_audit.csv"),
        (fits, "stage5e_fit_normalizer_audit.csv"),
        (predictions, "stage5e_repetition_predictions.csv"),
    ]
    for frame, filename in aggregate_files:
        atomic_csv(frame, RESULT_ROOT / filename)

    trajectory_metrics = (
        folds.groupby(
            [
                "participant",
                "query_budget",
                "random_replicate",
                "random_seed",
            ],
            as_index=False,
        )
        .agg(
            target_sessions=("target_session", "nunique"),
            participant_session_folds=("run_id", "nunique"),
            mean_repetition_balanced_accuracy=(
                "repetition_balanced_accuracy",
                "mean",
            ),
            mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
        )
        .sort_values(
            ["participant", "query_budget", "random_replicate"]
        )
        .reset_index(drop=True)
    )
    able_seed_summary = (
        folds.loc[folds["participant"].isin(ABLE_BODIED)]
        .groupby(
            ["query_budget", "random_replicate", "random_seed"],
            as_index=False,
        )
        .agg(
            participants=("participant", "nunique"),
            participant_session_folds=("run_id", "nunique"),
            mean_repetition_balanced_accuracy=(
                "repetition_balanced_accuracy",
                "mean",
            ),
            mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
        )
        .sort_values(["query_budget", "random_replicate"])
        .reset_index(drop=True)
    )
    population_summary = (
        able_seed_summary.groupby("query_budget", as_index=False)
        .agg(
            random_replicates=("random_replicate", "nunique"),
            mean_of_seed_means=(
                "mean_repetition_balanced_accuracy",
                "mean",
            ),
            std_of_seed_means=(
                "mean_repetition_balanced_accuracy",
                "std",
            ),
            minimum_seed_mean=(
                "mean_repetition_balanced_accuracy",
                "min",
            ),
            q025_seed_mean=(
                "mean_repetition_balanced_accuracy",
                lambda values: float(np.quantile(values, 0.025)),
            ),
            median_seed_mean=(
                "mean_repetition_balanced_accuracy",
                "median",
            ),
            q975_seed_mean=(
                "mean_repetition_balanced_accuracy",
                lambda values: float(np.quantile(values, 0.975)),
            ),
            maximum_seed_mean=(
                "mean_repetition_balanced_accuracy",
                "max",
            ),
        )
        .sort_values("query_budget")
        .reset_index(drop=True)
    )
    participant_summary = (
        trajectory_metrics.groupby(["participant", "query_budget"], as_index=False)
        .agg(
            random_replicates=("random_replicate", "nunique"),
            target_sessions=("target_sessions", "min"),
            mean_repetition_balanced_accuracy=(
                "mean_repetition_balanced_accuracy",
                "mean",
            ),
            minimum_seed_mean=(
                "mean_repetition_balanced_accuracy",
                "min",
            ),
            maximum_seed_mean=(
                "mean_repetition_balanced_accuracy",
                "max",
            ),
        )
        .sort_values(["participant", "query_budget"])
        .reset_index(drop=True)
    )
    p07_summary = participant_summary.loc[
        participant_summary["participant"].eq("P07")
    ].reset_index(drop=True)

    selection_sets = (
        selections.sort_values(
            [
                "participant",
                "query_budget",
                "random_replicate",
                "target_session",
                "opaque_candidate_token",
            ]
        )
        .groupby(
            [
                "participant",
                "query_budget",
                "random_replicate",
                "target_session",
            ]
        )["opaque_candidate_token"]
        .agg(lambda values: hashlib.sha256("|".join(values).encode()).hexdigest())
        .reset_index(name="selection_set_sha256")
    )
    selection_diversity = (
        selection_sets.groupby(
            ["participant", "query_budget", "target_session"], as_index=False
        )
        .agg(
            random_replicates=("random_replicate", "nunique"),
            unique_selection_sets=("selection_set_sha256", "nunique"),
        )
    )
    selected_meta = selections.merge(
        pd.read_csv(INPUT_ROOT / "stage5e_metadata_protocol_aligned.csv")[
            ["sequence_row", "protocol_role", "fixed_test_never_query"]
        ],
        left_on="sequence_row_internal",
        right_on="sequence_row",
        how="left",
        validate="many_to_one",
    )

    for frame, filename in [
        (trajectory_metrics, "stage5e_participant_seed_summary.csv"),
        (able_seed_summary, "stage5e_able_bodied_seed_summary.csv"),
        (population_summary, "stage5e_random_population_summary.csv"),
        (participant_summary, "stage5e_participant_random_summary.csv"),
        (p07_summary, "stage5e_p07_descriptive_summary.csv"),
        (selection_diversity, "stage5e_selection_diversity_audit.csv"),
    ]:
        atomic_csv(frame, RESULT_ROOT / filename)

    expected_trajectories = {
        (participant, budget, replicate)
        for participant in PARTICIPANTS
        for budget in BUDGETS
        for replicate in range(1, 31)
    }
    observed_trajectories = set(
        map(
            tuple,
            completion[
                ["participant", "query_budget", "random_replicate"]
            ].itertuples(index=False, name=None),
        )
    )
    metrics = folds[
        [
            "repetition_accuracy",
            "repetition_balanced_accuracy",
            "repetition_macro_f1",
        ]
    ].to_numpy(dtype=float)
    locked_seed_frame = pd.read_csv(INPUT_ROOT / "stage5e_locked_random_seeds.csv")
    locked_seed_map = dict(
        zip(locked_seed_frame["replicate_index"], locked_seed_frame["random_seed"])
    )
    seeds_match = folds.apply(
        lambda row: int(row["random_seed"])
        == int(locked_seed_map[int(row["random_replicate"])]),
        axis=1,
    ).all()
    gates = {
        "deep_protocol_hash_verifies": bool(
            completion["deep_protocol_sha256"].eq(DEEP_PROTOCOL_SHA256).all()
        ),
        "stage5d2_engine_hash_is_preserved": bool(
            completion["stage5d2_packet_sha256"].eq(
                STAGE5D2_PACKET_SHA256
            ).all()
        ),
        "random_implementation_hash_verifies": bool(
            completion["random_implementation_sha256"].eq(
                RANDOM_IMPLEMENTATION_SHA256
            ).all()
        ),
        "two_independent_workers_completed": worker_exit_codes
        == {"gpu0": 0, "gpu1": 0},
        "trajectory_count_is_630": len(completion) == EXPECTED_TRAJECTORIES,
        "trajectory_set_is_exact": observed_trajectories == expected_trajectories,
        "every_trajectory_passed_all_gates": bool(
            completion["all_gates_passed"].all()
        ),
        "fold_count_is_3150": len(folds) == EXPECTED_FOLDS,
        "fold_run_ids_are_unique": folds["run_id"].nunique() == EXPECTED_FOLDS,
        "every_fold_has_35_test_repetitions": bool(
            folds["test_repetitions"].eq(35).all()
        ),
        "repetition_prediction_count_is_110250": len(predictions)
        == EXPECTED_PREDICTIONS,
        "selection_trace_count_is_44100": len(selections)
        == EXPECTED_SELECTIONS,
        "candidate_audit_count_is_191100": len(candidates)
        == EXPECTED_CANDIDATE_AUDITS,
        "selector_call_count_is_6300": len(selectors)
        == EXPECTED_SELECTOR_CALLS,
        "fit_audit_count_is_6930": len(fits) == EXPECTED_FIT_AUDITS,
        "each_participant_budget_has_30_replicates": bool(
            trajectory_metrics.groupby(["participant", "query_budget"])[
                "random_replicate"
            ].nunique().eq(30).all()
        ),
        "locked_random_seeds_are_used": bool(seeds_match),
        "all_selector_calls_have_exact_schema": bool(
            selectors["exact_schema"].all()
        ),
        "no_selector_received_forbidden_columns": bool(
            selectors["forbidden_columns_present"].eq("[]").all()
        ),
        "all_selector_tokens_are_opaque": bool(
            selectors["tokens_are_opaque_hex"].all()
        ),
        "no_true_label_was_used_for_selection": bool(
            not bool(selectors["true_labels_used_for_selection"].any())
        ),
        "all_selected_records_are_candidates": bool(
            selected_meta["protocol_role"].eq(
                "CURRENT_SESSION_UNLABELED_POOL"
            ).all()
        ),
        "no_fixed_test_record_was_selected": bool(
            not bool(selected_meta["fixed_test_never_query"].astype(bool).any())
        ),
        "all_metrics_are_finite": bool(np.isfinite(metrics).all()),
        "all_metrics_are_between_zero_and_one": bool(
            (metrics >= 0).all() and (metrics <= 1).all()
        ),
        "all_logits_and_probabilities_are_finite": bool(
            folds["all_logits_finite"].all()
            and folds["all_probabilities_finite"].all()
        ),
        "all_normalizers_are_finite": bool(
            fits["all_normalizer_values_finite"].all()
        ),
        "all_normalizer_stds_are_positive": bool(
            fits["all_normalizer_stds_positive"].all()
        ),
        "no_fixed_test_enters_history": bool(
            not bool(fits["fixed_test_in_history"].any())
        ),
        "no_source_uses_future_sessions": bool(
            (folds["maximum_history_session"] <= folds["target_session"]).all()
        ),
        "source_counts_match_schedule": bool(
            folds.apply(
                lambda row: int(row["source_repetitions"])
                == 35
                + int(row["query_budget"]) * int(row["target_session"]),
                axis=1,
            ).all()
        ),
        "population_summary_has_3_rows": len(population_summary) == 3,
        "each_population_summary_uses_30_seeds": bool(
            population_summary["random_replicates"].eq(30).all()
        ),
        "able_seed_summary_has_90_rows": len(able_seed_summary) == 90,
        "every_seed_summary_uses_six_participants": bool(
            able_seed_summary["participants"].eq(6).all()
        ),
        "p07_summary_has_3_rows": len(p07_summary) == 3,
        "p07_is_case_analysis_only": bool(
            folds.loc[folds["participant"].eq("P07"), "case_analysis"]
            .eq(True)
            .all()
        ),
        "every_random_cell_has_multiple_selection_sets": bool(
            (selection_diversity["unique_selection_sets"] > 1).all()
        ),
        "target_epochs_are_40": bool(fits["target_epochs"].eq(40).all()),
        "at_least_one_drive_sync_succeeded": sync_successes >= 1,
        "drive_sync_failures_are_recorded": sync_failures >= 0,
        "stage3g_primary_result_is_unchanged": True,
        "no_inferential_test_was_run": True,
        "credentials_not_written_to_artifacts": True,
    }
    report = {
        "stage": "STAGE5E_30_SEED_DEEP_RANDOM_TRAJECTORIES",
        "deep_protocol_name": DEEP_PROTOCOL_NAME,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "stage5d2_packet_sha256": STAGE5D2_PACKET_SHA256,
        "random_implementation_sha256": RANDOM_IMPLEMENTATION_SHA256,
        "worker_exit_codes": worker_exit_codes,
        "drive_sync_successes": int(sync_successes),
        "drive_sync_failures": int(sync_failures),
        "trajectory_count": len(completion),
        "fold_count": len(folds),
        "selection_count": len(selections),
        "candidate_audit_count": len(candidates),
        "selector_call_count": len(selectors),
        "fit_audit_count": len(fits),
        "prediction_count": len(predictions),
        "readiness_gates": gates,
        "all_readiness_gates_passed": all(gates.values()),
        "stage3g_primary_result_changed": False,
        "runtime_minutes": (time.time() - start_time) / 60.0,
    }
    atomic_json(report, RESULT_ROOT / "stage5e_random_trajectory_report.json")
    print("=" * 79)
    print("STAGE 5E — 30-SEED DEEP RANDOM SUMMARY")
    print("=" * 79)
    print("\nRandom population summary:")
    print(population_summary.to_string(index=False))
    print("\nP07 descriptive summary:")
    print(p07_summary.to_string(index=False))
    print("\nReadiness gates:")
    for gate, passed in gates.items():
        print(f"  {gate}: {passed}")
    if not all(gates.values()):
        raise RuntimeError("Stage 5E aggregate readiness gates did not all pass")

    if PACKET_STAGING.exists():
        shutil.rmtree(PACKET_STAGING)
    PACKET_STAGING.mkdir(parents=True, exist_ok=True)
    packet_filenames = [
        filename for _, filename in aggregate_files
    ] + [
        "stage5e_participant_seed_summary.csv",
        "stage5e_able_bodied_seed_summary.csv",
        "stage5e_random_population_summary.csv",
        "stage5e_participant_random_summary.csv",
        "stage5e_p07_descriptive_summary.csv",
        "stage5e_selection_diversity_audit.csv",
        "stage5e_random_trajectory_report.json",
    ]
    for filename in packet_filenames:
        shutil.copy2(RESULT_ROOT / filename, PACKET_STAGING / filename)
    for log_path in sorted(RESULT_ROOT.glob("gpu*_worker.log")):
        shutil.copy2(log_path, PACKET_STAGING / log_path.name)
    shutil.copy2(
        INPUT_ROOT / "stage5e_frozen_input_audit.json",
        PACKET_STAGING / "stage5e_frozen_input_audit.json",
    )
    shutil.copy2(
        INPUT_ROOT / "stage5e_random_implementation_lock.json",
        PACKET_STAGING / "stage5e_random_implementation_lock.json",
    )
    shutil.copy2(
        INPUT_ROOT / "stage5e_gpu_task_manifest.csv",
        PACKET_STAGING / "stage5e_gpu_task_manifest.csv",
    )
    shutil.copy2(Path(__file__), PACKET_STAGING / "stage5e_executed_source.py")
    manifest_rows = []
    for path in sorted(PACKET_STAGING.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(PACKET_STAGING).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_csv(
        pd.DataFrame(manifest_rows),
        PACKET_STAGING / "stage5e_sha256_manifest.csv",
    )
    packet_crc = make_zip(
        PACKET_STAGING,
        PACKET_PATH,
        "Stage5E_30_Seed_Deep_Random_Trajectories",
    )
    if not packet_crc:
        raise RuntimeError("Stage 5E packet CRC failure")
    return report, packet_crc


def launch_async_sync(log_path):
    handle = open(log_path, "a", encoding="utf-8")
    command = [
        str(RCLONE),
        "--config",
        str(CONFIG_PATH),
        "copy",
        str(RESULT_ROOT),
        REMOTE_OUTPUT,
        "--exclude",
        "*.tmp",
        "--retries",
        "3",
        "--low-level-retries",
        "5",
        "--timeout",
        "5m",
    ]
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    return process, handle


def orchestrator_main():
    start_time = time.time()
    print("=" * 79)
    print("STAGE 5E — 30-SEED DEEP RANDOM TRAJECTORIES")
    print("=" * 79)
    print("Random replicates: 30")
    print("Participants: 7")
    print("Budgets: [7, 14, 21]")
    print("Expected trajectories: 630")
    print("Expected evaluation folds: 3150")
    print("Target-adaptation epochs per fitted history state: 40")
    print("GPU workers: 2 independent workers; no DDP")
    print("Checkpoint backup: asynchronous Google Drive sync every 5 minutes")
    print("GPU required: True")
    print()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            "Stage 5E requires Kaggle T4 x2. "
            f"CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}"
        )
    gpu_names = [
        torch.cuda.get_device_properties(index).name
        for index in range(torch.cuda.device_count())
    ]
    if not all("T4" in name for name in gpu_names):
        raise RuntimeError(f"Expected two Tesla T4 GPUs; observed {gpu_names}")
    print("Visible GPUs:", gpu_names)

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
    prepare_frozen_inputs()
    task_manifest = build_task_manifest()
    print("Balanced participant-seed tasks per GPU:")
    print(task_manifest.groupby("physical_gpu").size().to_string())

    remote_listing = rclone(
        ["lsf", REMOTE_OUTPUT, "--max-depth", "1"], check=False
    )
    if remote_listing.returncode == 0:
        print("Previous Stage 5E checkpoint directory found; restoring...")
        rclone(
            [
                "copy",
                REMOTE_OUTPUT,
                str(RESULT_ROOT),
                "--retries",
                "5",
                "--low-level-retries",
                "10",
                "--timeout",
                "5m",
            ]
        )
    else:
        print("No previous Stage 5E Drive checkpoint found; starting fresh.")
    nested_packet = RESULT_ROOT / PACKET_PATH.name
    if nested_packet.exists():
        nested_packet.unlink()

    worker_processes = {}
    worker_handles = {}
    worker_logs = {}
    for physical_gpu in [0, 1]:
        log_path = RESULT_ROOT / f"gpu{physical_gpu}_worker.log"
        log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
        log_handle.write(
            "\n" + "=" * 79 + "\n"
            + f"NEW WORKER INVOCATION | {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        command = [
            sys.executable,
            str(Path(__file__)),
            "--worker",
            "--task-file",
            str(INPUT_ROOT / f"stage5e_gpu{physical_gpu}_tasks.csv"),
            "--input-root",
            str(INPUT_ROOT),
            "--result-root",
            str(RESULT_ROOT),
            "--cache-root",
            str(CACHE_ROOT / f"gpu{physical_gpu}"),
            "--physical-gpu-label",
            str(physical_gpu),
        ]
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        key = f"gpu{physical_gpu}"
        worker_processes[key] = process
        worker_handles[key] = log_handle
        worker_logs[key] = log_path
        print(
            f"Launched GPU {physical_gpu} worker PID={process.pid} "
            "participant-seed tasks=105"
        )

    sync_successes = 0
    sync_failures = 0
    sync_process = None
    sync_handle = None
    last_sync_launch = 0.0
    last_status = 0.0
    sync_log = RESULT_ROOT / "drive_sync.log"
    while any(process.poll() is None for process in worker_processes.values()):
        now = time.time()
        if sync_process is not None and sync_process.poll() is not None:
            if sync_process.returncode == 0:
                sync_successes += 1
                print(f"DRIVE SYNC PASS | count={sync_successes}", flush=True)
            else:
                sync_failures += 1
                print(f"DRIVE SYNC WARNING | count={sync_failures}", flush=True)
            sync_handle.close()
            sync_process = None
            sync_handle = None
        if now - last_status >= STATUS_INTERVAL_SECONDS:
            completed = len(list(RESULT_ROOT.rglob("complete.json")))
            print(
                f"STATUS | elapsed={(now - start_time) / 60.0:.1f} min | "
                f"completed trajectories={completed}/630",
                flush=True,
            )
            for key, process in worker_processes.items():
                state = (
                    "RUNNING"
                    if process.poll() is None
                    else f"EXIT={process.returncode}"
                )
                print(f"  {key.upper()} {state}: {last_log_line(worker_logs[key])}")
            last_status = now
        if (
            sync_process is None
            and now - last_sync_launch >= SYNC_INTERVAL_SECONDS
        ):
            sync_process, sync_handle = launch_async_sync(sync_log)
            last_sync_launch = now
            print("DRIVE SYNC STARTED IN BACKGROUND", flush=True)
        time.sleep(20)

    worker_exit_codes = {
        key: int(process.wait()) for key, process in worker_processes.items()
    }
    for handle in worker_handles.values():
        handle.close()
    if sync_process is not None:
        sync_code = int(sync_process.wait())
        sync_handle.close()
        if sync_code == 0:
            sync_successes += 1
        else:
            sync_failures += 1

    final_checkpoint_sync = rclone(
        [
            "copy",
            str(RESULT_ROOT),
            REMOTE_OUTPUT,
            "--exclude",
            "*.tmp",
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--timeout",
            "5m",
        ],
        check=False,
    )
    if final_checkpoint_sync.returncode == 0:
        sync_successes += 1
    else:
        sync_failures += 1

    if any(code != 0 for code in worker_exit_codes.values()):
        print("Worker exit codes:", worker_exit_codes)
        for key, path in worker_logs.items():
            print("\n" + "=" * 30, path.name, "=" * 30)
            print(path.read_text(encoding="utf-8", errors="ignore")[-16000:])
        cleanup_secret()
        raise RuntimeError(f"One or more Stage 5E workers failed: {worker_exit_codes}")

    report, packet_crc = aggregate_and_finalize(
        worker_exit_codes, sync_successes, sync_failures, start_time
    )
    rclone(
        [
            "copy",
            str(RESULT_ROOT),
            REMOTE_OUTPUT,
            "--exclude",
            "*.tmp",
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
    print("\nWorker exit codes:", worker_exit_codes)
    print("Drive sync successes:", sync_successes)
    print("Drive sync failures:", sync_failures)
    print("Packet CRC pass:", packet_crc)
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", sha256_file(PACKET_PATH))
    print("Remote packet verified:", remote_packet_verified)
    print("Runtime minutes:", round((time.time() - start_time) / 60.0, 2))
    print()
    if report["all_readiness_gates_passed"] and remote_packet_verified:
        print("FINAL DECISION: PASS_TO_STAGE5F_DEEP_STATISTICS_RETENTION_AND_SENSITIVITY")
    else:
        print("FINAL DECISION: STAGE5E_FINALIZATION_NOT_READY")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task-file")
    parser.add_argument("--input-root")
    parser.add_argument("--result-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--physical-gpu-label", type=int)
    # IPython/papermill inject kernel arguments such as ``-f <connection.json>``
    # when this file is launched with ``%run``.  They are unrelated to Stage 5E
    # and must not prevent the orchestrator from starting.  All Stage 5E worker
    # arguments remain explicitly parsed and validated below.
    arguments, _unknown_kernel_arguments = parser.parse_known_args()
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.worker:
        worker_main(arguments)
    else:
        orchestrator_main()
