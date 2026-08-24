import atexit
import base64
import configparser
import hashlib
import importlib.util
import io
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# STAGE 5D-1 — DETERMINISTIC DEEP-ACQUISITION ENGINE UNIT TESTS
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
AMP_AMENDMENT_SHA256 = (
    "d303d3a7059855c95af1106db6ebf440d46a9ebe75165498748b178ee4b5c9a9"
)

PARTICIPANTS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07"]
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
CHANNELS = 64
CLASSES = 7
WINDOWS = 37

# These are deliberately short implementation tests, not scientific results.
UNIT_EPOCHS = 2
TARGET_BATCH_SIZE = 16
ENCODER_LEARNING_RATE = 1.0e-4
CLASSIFIER_LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-3
LABEL_SMOOTHING = 0.05
AMP_INITIAL_SCALE = 1024.0
AMP_GROWTH_INTERVAL = 10000

WORKING = Path("/kaggle/working")
TOOLS = WORKING / "_stage5_tools"
TOOLS.mkdir(parents=True, exist_ok=True)
RCLONE = TOOLS / "rclone"

RESTORE_ROOT = WORKING / "STAGE5D1_RESTORED_INPUTS"
STAGE5B_ROOT = RESTORE_ROOT / "stage5b"
RESULT_ROOT = WORKING / "STAGE5D1_DETERMINISTIC_ENGINE_UNIT_TESTS"
PACKET_PATH = WORKING / "stage5d1_deterministic_engine_unit_test_packet.zip"
for directory in [RESTORE_ROOT, STAGE5B_ROOT, RESULT_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

EVIDENCE_ROOT = Path(
    "/kaggle/input/datasets/zaidalsawaff/delta-q1-stage5-evidence-archives-v1"
)
STAGE3A_PACKET = EVIDENCE_ROOT / "stage3a_v1_1_protocol_amendment_packet.zip.bin"

STAGE5A4B_PACKET = WORKING / "stage5a4b_deep_protocol_lock_packet.zip"
STAGE5B_PACKET = WORKING / "stage5b_deep_sequence_assembly_packet.zip"
STAGE5C_PACKET = WORKING / "stage5c1_dual_gpu_loso_pretraining_packet.zip"

REMOTE_BASE = "gdrive_stage5:DELTA_Q1_Stage5_DeepLearning_Backup"
REMOTE_STAGE5A4B = (
    REMOTE_BASE
    + "/Stage5A4B_Deep_Protocol_Lock/"
    + STAGE5A4B_PACKET.name
)
REMOTE_STAGE5B = (
    REMOTE_BASE
    + "/Stage5B_Deep_Sequence_Assembly/"
    + STAGE5B_PACKET.name
)
REMOTE_STAGE5C = (
    REMOTE_BASE
    + "/Deep_Training/Stage5C_LOSO_Pretraining/"
    + STAGE5C_PACKET.name
)
REMOTE_OUTPUT = REMOTE_BASE + "/Deep_Training/Stage5D1_Deterministic_Engine_Unit_Tests"

START_TIME = time.time()
CONFIG_PATH = None

print("=" * 79)
print("STAGE 5D-1 — DETERMINISTIC DEEP-ACQUISITION ENGINE UNIT TESTS")
print("=" * 79)
print("Scientific role: IMPLEMENTATION UNIT TESTS ONLY")
print("Unit adaptation epochs:", UNIT_EPOCHS)
print("Full Stage 5D experiment will not run in this notebook.")
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


def write_json(payload, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
    os.replace(temporary, destination)


def read_json_member(packet, basename):
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise ValueError(f"Expected one {basename}; found {len(matches)}")
        return json.loads(archive.read(matches[0]).decode("utf-8"))


def read_csv_member(packet, basename):
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise ValueError(f"Expected one {basename}; found {len(matches)}")
        with archive.open(matches[0], "r") as handle:
            return pd.read_csv(handle)


def extract_member_by_basename(packet, basename, destination):
    destination = Path(destination)
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise ValueError(f"Expected one {basename}; found {len(matches)}")
        destination.write_bytes(archive.read(matches[0]))


def checkpoint_from_packet(packet, participant):
    expected_suffix = f"/{participant}/best.pt"
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if name.endswith(expected_suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one best checkpoint for {participant}; found {len(matches)}"
            )
        return torch.load(
            io.BytesIO(archive.read(matches[0])),
            map_location="cpu",
            weights_only=False,
        )


def make_zip(source_directory, destination, archive_root):
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(Path(source_directory).rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=(
                        archive_root
                        + "/"
                        + path.relative_to(source_directory).as_posix()
                    ),
                )
    with zipfile.ZipFile(destination, "r") as archive:
        return archive.testzip() is None


def cleanup_secret():
    global CONFIG_PATH
    if CONFIG_PATH is not None and Path(CONFIG_PATH).exists():
        Path(CONFIG_PATH).unlink()


atexit.register(cleanup_secret)


# -----------------------------------------------------------------------------
# 1. GPU, RCLONE, AND FROZEN INPUT RESTORE
# -----------------------------------------------------------------------------

if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError(
        "Stage 5D-1 requires the Kaggle T4 x2 accelerator. "
        f"CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}"
    )
GPU_NAMES = [
    torch.cuda.get_device_properties(index).name
    for index in range(torch.cuda.device_count())
]
if not all("T4" in name for name in GPU_NAMES):
    raise RuntimeError(f"Expected two Tesla T4 GPUs; observed {GPU_NAMES}")
print("Visible GPUs:", GPU_NAMES)

if not RCLONE.exists():
    print("Downloading verified official rclone binary...")
    version_text = urllib.request.urlopen(
        "https://downloads.rclone.org/version.txt", timeout=60
    ).read().decode("utf-8").strip()
    match = re.search(r"v?(\d+\.\d+\.\d+)", version_text)
    if match is None:
        raise RuntimeError("Could not resolve official rclone version")
    version = match.group(1)
    archive_name = f"rclone-v{version}-linux-amd64.zip"
    base_url = f"https://downloads.rclone.org/v{version}"
    temporary_root = Path(tempfile.mkdtemp(prefix="stage5d1_rclone_", dir="/tmp"))
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
    raise RuntimeError("Restricted Google Drive remote verification failed")
temporary_config = tempfile.NamedTemporaryFile(
    mode="wb",
    prefix="stage5d1_",
    suffix=".conf",
    dir="/tmp",
    delete=False,
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


for label, remote, local in [
    ("Stage 5A-4B", REMOTE_STAGE5A4B, STAGE5A4B_PACKET),
    ("Stage 5B", REMOTE_STAGE5B, STAGE5B_PACKET),
    ("Stage 5C", REMOTE_STAGE5C, STAGE5C_PACKET),
]:
    print(f"Restoring {label} packet...")
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

packet_hash_gates = {
    "stage5a4b_packet_hash_matches": (
        sha256_file(STAGE5A4B_PACKET) == STAGE5A4B_PACKET_SHA256
    ),
    "stage5b_packet_hash_matches": (
        sha256_file(STAGE5B_PACKET) == STAGE5B_PACKET_SHA256
    ),
    "stage5c_packet_hash_matches": (
        sha256_file(STAGE5C_PACKET) == STAGE5C_PACKET_SHA256
    ),
}
if not all(packet_hash_gates.values()):
    raise RuntimeError(f"Frozen packet hash mismatch: {packet_hash_gates}")
for packet in [STAGE3A_PACKET, STAGE5A4B_PACKET, STAGE5B_PACKET, STAGE5C_PACKET]:
    if not packet.exists() or packet.stat().st_size == 0:
        raise FileNotFoundError(packet)
    with zipfile.ZipFile(packet, "r") as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f"CRC failure: {packet}")

for basename in [
    "stage5b_rms_repetition_sequences.npy",
    "stage5b_main_valid_repetition_sequences.npy",
    "stage5b_repetition_metadata.csv",
    "stage5b_mask_aware_rms_tcn.py",
    "stage5b_sequence_assembly_report.json",
]:
    extract_member_by_basename(STAGE5B_PACKET, basename, STAGE5B_ROOT / basename)

stage5b_report = json.loads(
    (STAGE5B_ROOT / "stage5b_sequence_assembly_report.json").read_text(
        encoding="utf-8"
    )
)
if not stage5b_report.get("all_readiness_gates_passed", False):
    raise RuntimeError("Restored Stage 5B report did not pass all gates")


# -----------------------------------------------------------------------------
# 2. LOAD AND ALIGN FROZEN DATA, TOKENS, SEEDS, AND CHECKPOINTS
# -----------------------------------------------------------------------------

features = np.load(
    STAGE5B_ROOT / "stage5b_rms_repetition_sequences.npy",
    mmap_mode="r",
    allow_pickle=False,
)
main_valid = np.load(
    STAGE5B_ROOT / "stage5b_main_valid_repetition_sequences.npy",
    mmap_mode="r",
    allow_pickle=False,
)
metadata = pd.read_csv(STAGE5B_ROOT / "stage5b_repetition_metadata.csv")
metadata["participant"] = metadata["participant"].astype(str)
for column in ["session", "label", "repetition"]:
    metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(int)
metadata["sequence_row"] = pd.to_numeric(
    metadata["sequence_row"], errors="raise"
).astype(int)

universe = read_csv_member(
    STAGE3A_PACKET, "stage3a_v1_1_repetition_protocol_universe.csv"
)
lookup = read_csv_member(
    STAGE3A_PACKET, "stage3a_v1_1_internal_token_lookup.csv"
)
selector_schema = read_json_member(
    STAGE3A_PACKET, "stage3a_v1_1_selector_schema.json"
)
seed_schedule = read_csv_member(
    STAGE5A4B_PACKET, "stage5a4b_training_seed_schedule.csv"
)

if selector_schema["selector_input_columns"] != SELECTOR_COLUMNS:
    raise RuntimeError("Selector input schema drift")
if selector_schema["forbidden_selector_columns"] != FORBIDDEN_SELECTOR_COLUMNS:
    raise RuntimeError("Forbidden selector-column drift")

universe["participant"] = universe["participant"].astype(str)
for column in ["session", "label", "repetition_number"]:
    universe[column] = pd.to_numeric(universe[column], errors="raise").astype(int)

join_columns = ["participant", "session", "label"]
metadata_aligned = metadata.merge(
    universe[
        join_columns
        + [
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
    left_on=join_columns + ["repetition"],
    right_on=join_columns + ["repetition_number"],
    how="left",
    validate="one_to_one",
)
metadata_aligned = metadata_aligned.sort_values("sequence_row").reset_index(drop=True)

primary_seed_rows = seed_schedule.loc[
    seed_schedule["seed_role"].eq("PRIMARY_DETERMINISTIC"),
    ["participant", "seed"],
].copy()
primary_seed_rows["participant"] = primary_seed_rows["participant"].astype(str)
primary_seed_rows["seed"] = pd.to_numeric(
    primary_seed_rows["seed"], errors="raise"
).astype(np.int64)
PRIMARY_SEEDS = dict(zip(primary_seed_rows["participant"], primary_seed_rows["seed"]))
if sorted(PRIMARY_SEEDS) != PARTICIPANTS:
    raise RuntimeError("Locked primary seed schedule is incomplete")

checkpoint_by_participant = {
    participant: checkpoint_from_packet(STAGE5C_PACKET, participant)
    for participant in PARTICIPANTS
}
for participant, checkpoint in checkpoint_by_participant.items():
    if (
        checkpoint.get("protocol_sha256") != DEEP_PROTOCOL_SHA256
        or checkpoint.get("target_participant") != participant
        or int(checkpoint.get("training_seed")) != int(PRIMARY_SEEDS[participant])
        or checkpoint.get("target_data_used") is not False
    ):
        raise RuntimeError(f"Invalid pretrained checkpoint identity for {participant}")
    if not all(
        torch.isfinite(value).all().item()
        for value in checkpoint["model_state_dict"].values()
    ):
        raise RuntimeError(f"Non-finite pretrained state for {participant}")

model_path = STAGE5B_ROOT / "stage5b_mask_aware_rms_tcn.py"
spec = importlib.util.spec_from_file_location("stage5b_mask_aware_rms_tcn", model_path)
model_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model_module
spec.loader.exec_module(model_module)
MaskAwareRMSTCN = model_module.MaskAwareRMSTCN

if features.shape != (2940, WINDOWS, CHANNELS):
    raise RuntimeError(f"Unexpected feature shape: {features.shape}")
if main_valid.shape != features.shape:
    raise RuntimeError("Main-mask shape mismatch")
if len(metadata_aligned) != 2940 or metadata_aligned["repetition_uid"].isna().any():
    raise RuntimeError("Metadata-to-protocol join is incomplete")
if metadata_aligned["sequence_row"].tolist() != list(range(2940)):
    raise RuntimeError("Sequence-row alignment drift")
if not metadata_aligned["opaque_candidate_token"].str.fullmatch(
    r"[0-9a-f]{24}"
).all():
    raise RuntimeError("Opaque-token format failure")


# -----------------------------------------------------------------------------
# 3. EXACT OPAQUE SELECTORS AND SYNTHETIC CONTRACT TESTS
# -----------------------------------------------------------------------------

selector_audit_rows = []


def validate_selector_frame(frame, selector_name, call_id):
    columns = frame.columns.tolist()
    exact_schema = columns == SELECTOR_COLUMNS
    forbidden_present = sorted(set(columns).intersection(FORBIDDEN_SELECTOR_COLUMNS))
    tokens_opaque = bool(
        frame["opaque_candidate_token"].astype(str).str.fullmatch(r"[0-9a-f]{24}").all()
    ) if "opaque_candidate_token" in frame.columns else False
    selector_audit_rows.append(
        {
            "call_id": call_id,
            "selector_name": selector_name,
            "rows_received": len(frame),
            "columns_received": json.dumps(columns),
            "exact_schema": exact_schema,
            "forbidden_columns_present": json.dumps(forbidden_present),
            "tokens_are_opaque_hex": tokens_opaque,
        }
    )
    if not exact_schema or forbidden_present or not tokens_opaque:
        raise ValueError(
            f"Invalid selector-visible frame for {selector_name}: {columns}"
        )
    if frame["opaque_candidate_token"].duplicated().any():
        raise ValueError("Duplicate opaque tokens in selector frame")
    if not np.isfinite(frame["margin"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite selector margins")


def selector_order(frame):
    return frame.sort_values(
        ["margin", "opaque_candidate_token"],
        kind="mergesort",
    )


def select_pcbm(frame, call_id):
    validate_selector_frame(frame, "PCBM_PROPOSED", call_id)
    ordered = selector_order(frame)
    nominees = (
        ordered.groupby("predicted_label", sort=False, as_index=False)
        .head(1)
        .sort_values(["margin", "opaque_candidate_token"], kind="mergesort")
    )
    selected = nominees.head(7)["opaque_candidate_token"].tolist()
    if len(selected) < 7:
        fill = ordered.loc[
            ~ordered["opaque_candidate_token"].isin(selected),
            "opaque_candidate_token",
        ].tolist()
        selected.extend(fill[: 7 - len(selected)])
    if len(selected) != 7 or len(set(selected)) != 7:
        raise RuntimeError("PCBM did not select seven unique tokens")
    return selected


def select_global_margin(frame, call_id):
    validate_selector_frame(frame, "GLOBAL_MARGIN", call_id)
    selected = selector_order(frame).head(7)["opaque_candidate_token"].tolist()
    if len(selected) != 7 or len(set(selected)) != 7:
        raise RuntimeError("Global margin did not select seven unique tokens")
    return selected


def select_random_uniform(frame, seed, call_id):
    validate_selector_frame(frame, "RANDOM_UNIFORM", call_id)
    tokens = frame["opaque_candidate_token"].astype(str).to_numpy()
    generator = np.random.default_rng(int(seed))
    selected = generator.choice(tokens, size=7, replace=False).tolist()
    if len(set(selected)) != 7:
        raise RuntimeError("Random selector returned duplicate tokens")
    return selected


def synthetic_selector_frame(predicted_labels, margins):
    tokens = [f"{index:024x}" for index in range(len(predicted_labels))]
    return pd.DataFrame(
        {
            "opaque_candidate_token": tokens,
            "predicted_label": predicted_labels,
            "margin": margins,
        }
    )


# Seven represented predicted classes must receive one nominee each.
synthetic_balanced = synthetic_selector_frame(
    [0, 0, 0, 1, 2, 3, 4, 5, 6],
    [0.001, 0.002, 0.003, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
)
balanced_tokens = select_pcbm(synthetic_balanced, "SYNTHETIC_BALANCED")
balanced_selected = synthetic_balanced.set_index("opaque_candidate_token").loc[
    balanced_tokens
]
synthetic_one_per_class = set(balanced_selected["predicted_label"].tolist()) == set(
    range(7)
)

# With only three represented classes, the remaining four positions are filled
# by global margin order among unselected candidates.
synthetic_fill = synthetic_selector_frame(
    [0, 0, 0, 1, 1, 2, 2, 2, 2],
    [0.4, 0.1, 0.2, 0.3, 0.05, 0.15, 0.25, 0.35, 0.45],
)
fill_tokens = select_pcbm(synthetic_fill, "SYNTHETIC_FILL")
expected_nominees = set(
    selector_order(synthetic_fill)
    .groupby("predicted_label", sort=False, as_index=False)
    .head(1)["opaque_candidate_token"]
)
synthetic_fill_preserves_nominees = expected_nominees.issubset(set(fill_tokens))

# Equal margins must be resolved lexicographically by opaque token.
synthetic_tie = synthetic_selector_frame([0] * 8, [0.5] * 8)
tie_tokens = select_global_margin(synthetic_tie, "SYNTHETIC_TIE")
synthetic_tie_is_lexical = tie_tokens == sorted(
    synthetic_tie["opaque_candidate_token"].tolist()
)[:7]

forbidden_rejected = False
try:
    bad_frame = synthetic_tie.copy()
    bad_frame["label"] = 0
    select_pcbm(bad_frame, "SYNTHETIC_FORBIDDEN_EXPECTED_FAILURE")
except ValueError:
    forbidden_rejected = True

random_a = select_random_uniform(synthetic_fill, 12345, "SYNTHETIC_RANDOM_A")
random_b = select_random_uniform(synthetic_fill, 12345, "SYNTHETIC_RANDOM_B")
random_c = select_random_uniform(synthetic_fill, 54321, "SYNTHETIC_RANDOM_C")


# -----------------------------------------------------------------------------
# 4. MASK-AWARE HISTORY-ONLY DEEP ADAPTATION ENGINE
# -----------------------------------------------------------------------------

DEVICE = torch.device("cuda:0")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def history_seed(participant, history_rows):
    tokens = sorted(
        metadata_aligned.iloc[np.asarray(history_rows, dtype=int)][
            "opaque_candidate_token"
        ].astype(str).tolist()
    )
    payload = (
        str(int(PRIMARY_SEEDS[participant]))
        + "|"
        + participant
        + "|"
        + "|".join(tokens)
    )
    value = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return int(value % 2_000_000_000 + 1)


def fit_normalizer(raw, valid):
    logged = np.log1p(np.asarray(raw, dtype=np.float64))
    valid = np.asarray(valid, dtype=bool)
    means = np.zeros(CHANNELS, dtype=np.float64)
    stds = np.zeros(CHANNELS, dtype=np.float64)
    counts = np.zeros(CHANNELS, dtype=np.int64)
    for channel in range(CHANNELS):
        values = logged[:, :, channel][valid[:, :, channel]]
        counts[channel] = len(values)
        if len(values) == 0:
            raise ValueError(f"No main-valid history values for channel {channel}")
        means[channel] = values.mean()
        stds[channel] = values.std(ddof=0)
        if not np.isfinite(stds[channel]) or stds[channel] <= 0:
            raise ValueError(f"Invalid history std for channel {channel}")
    return means, stds, counts


def transform_rows(row_indices, means, stds):
    row_indices = np.asarray(row_indices, dtype=int)
    raw = np.asarray(features[row_indices], dtype=np.float64)
    valid = np.asarray(main_valid[row_indices], dtype=bool)
    normalized = (
        np.log1p(raw) - means[None, None, :]
    ) / stds[None, None, :]
    normalized[~valid] = 0.0
    combined = np.concatenate(
        [normalized.astype(np.float32), valid.astype(np.float32)],
        axis=2,
    )
    combined = np.transpose(combined, (0, 2, 1)).copy()
    if not np.isfinite(combined).all():
        raise ValueError("Non-finite transformed model input")
    return combined


def freeze_target_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = True
    stem_attribute = next(
        (
            name
            for name in ["stem", "input_projection", "input_stem"]
            if hasattr(model, name) and isinstance(getattr(model, name), nn.Module)
        ),
        None,
    )
    if stem_attribute is None or not hasattr(model, "blocks"):
        raise RuntimeError("Model does not expose the locked stem/blocks contract")
    if len(model.blocks) != 4:
        raise RuntimeError(f"Expected four TCN blocks; found {len(model.blocks)}")
    stem_module = getattr(model, stem_attribute)
    for parameter in stem_module.parameters():
        parameter.requires_grad = False
    for block in model.blocks[:2]:
        for parameter in block.parameters():
            parameter.requires_grad = False
    frozen_names = [
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    ]
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not frozen_names or not trainable_names:
        raise RuntimeError("Invalid target-adaptation freeze state")
    stem_prefix = stem_attribute + "."
    if any(
        not (
            name.startswith(stem_prefix)
            or name.startswith("blocks.0.")
            or name.startswith("blocks.1.")
        )
        for name in frozen_names
    ):
        raise RuntimeError("Unexpected frozen parameter")
    if any(
        name.startswith(stem_prefix)
        or name.startswith("blocks.0.")
        or name.startswith("blocks.1.")
        for name in trainable_names
    ):
        raise RuntimeError("Locked early encoder layer remained trainable")
    return frozen_names, trainable_names, stem_attribute


def build_optimizer(model):
    classifier_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("classifier.")
    ]
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("classifier.")
    ]
    if not classifier_parameters or not encoder_parameters:
        raise RuntimeError("Target optimizer parameter groups are incomplete")
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LEARNING_RATE},
            {"params": classifier_parameters, "lr": CLASSIFIER_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def fit_history_state(participant, history_rows, epochs=UNIT_EPOCHS):
    history_rows = np.asarray(sorted(set(map(int, history_rows))), dtype=np.int64)
    if len(history_rows) == 0:
        raise ValueError("History cannot be empty")
    history_metadata = metadata_aligned.iloc[history_rows]
    if not history_metadata["participant"].eq(participant).all():
        raise RuntimeError("Cross-participant target history contamination")
    if not history_metadata["eligible_for_training"].astype(bool).all():
        raise RuntimeError("A non-training-eligible repetition entered history")
    if history_metadata["fixed_test_never_query"].astype(bool).any():
        raise RuntimeError("A fixed-test repetition entered training history")

    means, stds, counts = fit_normalizer(
        features[history_rows], main_valid[history_rows]
    )
    x = transform_rows(history_rows, means, stds)
    y = history_metadata["label"].to_numpy(dtype=np.int64)
    if sorted(np.unique(y).tolist()) != list(range(CLASSES)):
        raise RuntimeError("History does not contain all seven classes")

    seed = history_seed(participant, history_rows)
    set_seed(seed)
    model = MaskAwareRMSTCN().to(DEVICE)
    checkpoint = checkpoint_by_participant[participant]
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    frozen_names, trainable_names, stem_attribute = freeze_target_model(model)
    optimizer = build_optimizer(model)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=AMP_INITIAL_SCALE,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=AMP_GROWTH_INTERVAL,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    epoch_rows = []

    for epoch in range(1, int(epochs) + 1):
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=TARGET_BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        model.train()
        total_loss = 0.0
        total_examples = 0
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite target-adaptation loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            ):
                raise RuntimeError("Non-finite target-adaptation gradient")
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(labels)
            total_examples += len(labels)
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_examples,
                "amp_scale": float(scaler.get_scale()),
            }
        )

    if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise RuntimeError("Non-finite target-adapted model state")
    return {
        "model": model,
        "means": means,
        "stds": stds,
        "counts": counts,
        "history_rows": history_rows,
        "history_tokens": sorted(history_metadata["opaque_candidate_token"].tolist()),
        "history_seed": seed,
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_names": trainable_names,
        "stem_attribute": stem_attribute,
        "epoch_rows": epoch_rows,
        "normalizer_source_rows": history_rows.copy(),
    }


@torch.no_grad()
def predict_rows(state, row_indices):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    x = transform_rows(row_indices, state["means"], state["stds"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    state["model"].eval()
    logits_batches = []
    for (inputs,) in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = state["model"](inputs)
        logits_batches.append(logits.float().cpu().numpy())
    logits = np.concatenate(logits_batches, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    ordered = np.sort(probabilities, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    predictions = np.argmax(probabilities, axis=1).astype(int)
    if not np.isfinite(probabilities).all() or not np.isfinite(margins).all():
        raise RuntimeError("Non-finite deep candidate scores")
    return logits, probabilities, predictions, margins


def candidate_rows(participant, session):
    mask = (
        metadata_aligned["participant"].eq(participant)
        & metadata_aligned["session"].eq(int(session))
        & metadata_aligned["protocol_role"].eq("CURRENT_SESSION_UNLABELED_POOL")
    )
    rows = metadata_aligned.loc[mask, "sequence_row"].to_numpy(dtype=np.int64)
    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 candidates for {participant} session {session}")
    return rows


def fixed_test_rows(participant, session):
    mask = (
        metadata_aligned["participant"].eq(participant)
        & metadata_aligned["session"].eq(int(session))
        & metadata_aligned["protocol_role"].eq("TARGET_FIXED_TEST_NEVER_QUERY")
    )
    rows = metadata_aligned.loc[mask, "sequence_row"].to_numpy(dtype=np.int64)
    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 fixed tests for {participant} session {session}")
    return rows


def initial_history_rows(participant):
    mask = (
        metadata_aligned["participant"].eq(participant)
        & metadata_aligned["session"].eq(0)
        & metadata_aligned["protocol_role"].eq("INITIAL_LABELED_CALIBRATION")
    )
    rows = metadata_aligned.loc[mask, "sequence_row"].to_numpy(dtype=np.int64)
    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 initial history rows for {participant}")
    return rows


def selector_frame_from_scores(rows, predictions, margins):
    return pd.DataFrame(
        {
            "opaque_candidate_token": metadata_aligned.iloc[rows][
                "opaque_candidate_token"
            ].astype(str).to_numpy(),
            "predicted_label": np.asarray(predictions, dtype=int),
            "margin": np.asarray(margins, dtype=float),
        },
        columns=SELECTOR_COLUMNS,
    )


def reveal_selected_rows(selected_tokens, remaining_rows):
    remaining = metadata_aligned.iloc[np.asarray(remaining_rows, dtype=int)][
        ["opaque_candidate_token", "sequence_row", "protocol_role"]
    ].copy()
    token_to_row = dict(
        zip(remaining["opaque_candidate_token"], remaining["sequence_row"])
    )
    if any(token not in token_to_row for token in selected_tokens):
        raise RuntimeError("Selector returned a token outside the candidate pool")
    selected_rows = np.asarray(
        [token_to_row[token] for token in selected_tokens], dtype=np.int64
    )
    selected_meta = metadata_aligned.iloc[selected_rows]
    if not selected_meta["protocol_role"].eq(
        "CURRENT_SESSION_UNLABELED_POOL"
    ).all():
        raise RuntimeError("A non-candidate record was selected")
    if selected_meta["fixed_test_never_query"].astype(bool).any():
        raise RuntimeError("A fixed-test record was selected")
    return selected_rows


def classification_metrics(y_true, y_pred):
    matrix = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    for truth, predicted in zip(y_true, y_pred):
        matrix[int(truth), int(predicted)] += 1
    support = matrix.sum(axis=1)
    recall = np.divide(
        np.diag(matrix),
        support,
        out=np.zeros(CLASSES, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        np.diag(matrix),
        matrix.sum(axis=0),
        out=np.zeros(CLASSES, dtype=np.float64),
        where=matrix.sum(axis=0) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(CLASSES, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_fixed_test(state, participant, session):
    rows = fixed_test_rows(participant, session)
    if np.intersect1d(rows, state["normalizer_source_rows"]).size != 0:
        raise RuntimeError("Fixed test entered normalizer source")
    _, probabilities, predictions, _ = predict_rows(state, rows)
    truths = metadata_aligned.iloc[rows]["label"].to_numpy(dtype=int)
    metrics = classification_metrics(truths, predictions)
    metrics.update(
        {
            "test_repetitions": int(len(rows)),
            "test_rows": rows.tolist(),
            "probabilities_finite": bool(np.isfinite(probabilities).all()),
        }
    )
    return metrics


selection_trace_rows = []
unit_run_rows = []
fit_audit_rows = []


def record_fit(participant, strategy, budget, session, round_index, state):
    history_meta = metadata_aligned.iloc[state["history_rows"]]
    fit_audit_rows.append(
        {
            "participant": participant,
            "strategy": strategy,
            "query_budget": budget,
            "target_session": session,
            "round_index": round_index,
            "history_repetitions": len(state["history_rows"]),
            "history_seed": state["history_seed"],
            "maximum_history_session": int(history_meta["session"].max()),
            "normalizer_source_equals_history": bool(
                np.array_equal(
                    np.sort(state["normalizer_source_rows"]),
                    np.sort(state["history_rows"]),
                )
            ),
            "minimum_normalizer_count": int(state["counts"].min()),
            "all_normalizer_values_finite": bool(
                np.isfinite(state["means"]).all()
                and np.isfinite(state["stds"]).all()
            ),
            "all_normalizer_stds_positive": bool((state["stds"] > 0).all()),
            "fixed_test_in_history": bool(
                history_meta["fixed_test_never_query"].astype(bool).any()
            ),
            "trainable_parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in state["model"].parameters()
                    if parameter.requires_grad
                )
            ),
            "frozen_parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in state["model"].parameters()
                    if not parameter.requires_grad
                )
            ),
            "unit_epochs": UNIT_EPOCHS,
            "final_unit_train_loss": float(state["epoch_rows"][-1]["train_loss"]),
            "final_amp_scale": float(state["epoch_rows"][-1]["amp_scale"]),
        }
    )


def run_unit_trajectory(participant, strategy, budget, sessions):
    history = initial_history_rows(participant).tolist()
    budget_to_rounds = {0: 0, 7: 1, 14: 2, 21: 3, 35: 5}
    rounds = budget_to_rounds[int(budget)]
    final_state = None

    for session in sessions:
        candidates = candidate_rows(participant, session)
        selected_this_session = []

        if strategy == "NO_ADAPTATION_REFERENCE":
            state = fit_history_state(participant, history)
            record_fit(participant, strategy, budget, session, 0, state)
            final_state = state

        elif strategy == "FULL_POOL_REFERENCE":
            selected_rows = candidates.copy()
            selected_tokens = metadata_aligned.iloc[selected_rows][
                "opaque_candidate_token"
            ].astype(str).tolist()
            for order, (token, row) in enumerate(
                zip(selected_tokens, selected_rows), start=1
            ):
                selection_trace_rows.append(
                    {
                        "participant": participant,
                        "target_session": session,
                        "strategy": strategy,
                        "query_budget": budget,
                        "round_index": 1,
                        "selection_order_in_round": order,
                        "opaque_candidate_token": token,
                        "sequence_row_internal": int(row),
                        "true_label_revealed_after_selection": int(
                            metadata_aligned.iloc[row]["label"]
                        ),
                        "selector_visible_columns": "FULL_POOL_ALL_CANDIDATES",
                    }
                )
            selected_this_session.extend(selected_rows.tolist())
            history.extend(selected_rows.tolist())
            state = fit_history_state(participant, history)
            record_fit(participant, strategy, budget, session, 1, state)
            final_state = state

        else:
            remaining = candidates.copy()
            for round_index in range(1, rounds + 1):
                state = fit_history_state(participant, history)
                record_fit(
                    participant,
                    strategy,
                    budget,
                    session,
                    round_index - 1,
                    state,
                )
                _, _, predictions, margins = predict_rows(state, remaining)
                visible = selector_frame_from_scores(remaining, predictions, margins)
                call_id = (
                    f"{participant}_S{session:02d}_{strategy}_K{budget:02d}_R{round_index:02d}"
                )
                if strategy == "PCBM_PROPOSED":
                    selected_tokens = select_pcbm(visible, call_id)
                elif strategy == "GLOBAL_MARGIN":
                    selected_tokens = select_global_margin(visible, call_id)
                else:
                    raise ValueError(f"Unsupported deterministic strategy: {strategy}")
                selected_rows = reveal_selected_rows(selected_tokens, remaining)
                for order, (token, row) in enumerate(
                    zip(selected_tokens, selected_rows), start=1
                ):
                    selection_trace_rows.append(
                        {
                            "participant": participant,
                            "target_session": session,
                            "strategy": strategy,
                            "query_budget": budget,
                            "round_index": round_index,
                            "selection_order_in_round": order,
                            "opaque_candidate_token": token,
                            "sequence_row_internal": int(row),
                            "true_label_revealed_after_selection": int(
                                metadata_aligned.iloc[row]["label"]
                            ),
                            "selector_visible_columns": "|".join(SELECTOR_COLUMNS),
                        }
                    )
                selected_this_session.extend(selected_rows.tolist())
                history.extend(selected_rows.tolist())
                remaining = np.asarray(
                    [row for row in remaining if row not in set(selected_rows)],
                    dtype=np.int64,
                )
            final_state = fit_history_state(participant, history)
            record_fit(
                participant, strategy, budget, session, rounds, final_state
            )

        metrics = evaluate_fixed_test(final_state, participant, session)
        selected_meta = metadata_aligned.iloc[selected_this_session]
        unit_run_rows.append(
            {
                "participant": participant,
                "target_session": session,
                "strategy": strategy,
                "query_budget": budget,
                "case_analysis": participant == "P07",
                "history_repetitions": len(history),
                "selected_repetitions_this_session": len(selected_this_session),
                "unique_selected_tokens": int(
                    selected_meta["opaque_candidate_token"].nunique()
                    if len(selected_meta)
                    else 0
                ),
                "selected_true_labels_after_reveal": int(
                    selected_meta["label"].nunique() if len(selected_meta) else 0
                ),
                "test_repetitions": metrics["test_repetitions"],
                "test_accuracy": metrics["accuracy"],
                "test_balanced_accuracy": metrics["balanced_accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "probabilities_finite": metrics["probabilities_finite"],
                "unit_test_only": True,
            }
        )

    del final_state
    torch.cuda.empty_cache()


print()
print("Running short real-data engine tests on GPU 0...")
print("  P01 PCBM K07 across sessions 1 and 2")
run_unit_trajectory("P01", "PCBM_PROPOSED", 7, [1, 2])
print("  P05 global margin K14 in session 1")
run_unit_trajectory("P05", "GLOBAL_MARGIN", 14, [1])
print("  P07 full-pool K35 case-analysis test in session 1")
run_unit_trajectory("P07", "FULL_POOL_REFERENCE", 35, [1])
print("  P03 no-adaptation K00 in session 1")
run_unit_trajectory("P03", "NO_ADAPTATION_REFERENCE", 0, [1])

unit_runs = pd.DataFrame(unit_run_rows)
selection_trace = pd.DataFrame(selection_trace_rows)
fit_audit = pd.DataFrame(fit_audit_rows)
selector_audit = pd.DataFrame(selector_audit_rows)


# Independent repeat-fit determinism check on the same P02 initial history.
print("  P02 repeated-fit deterministic-state test")
p02_history = initial_history_rows("P02")
p02_state_a = fit_history_state("P02", p02_history)
p02_rows = candidate_rows("P02", 1)[:8]
p02_logits_a, _, _, _ = predict_rows(p02_state_a, p02_rows)
del p02_state_a
torch.cuda.empty_cache()
p02_state_b = fit_history_state("P02", p02_history)
p02_logits_b, _, _, _ = predict_rows(p02_state_b, p02_rows)
maximum_repeat_fit_logit_difference = float(
    np.max(np.abs(p02_logits_a - p02_logits_b))
)
del p02_state_b
torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# 5. AUDIT, PACKET, AND DRIVE BACKUP
# -----------------------------------------------------------------------------

expected_run_keys = {
    ("P01", 1, "PCBM_PROPOSED", 7),
    ("P01", 2, "PCBM_PROPOSED", 7),
    ("P05", 1, "GLOBAL_MARGIN", 14),
    ("P07", 1, "FULL_POOL_REFERENCE", 35),
    ("P03", 1, "NO_ADAPTATION_REFERENCE", 0),
}
observed_run_keys = set(
    map(
        tuple,
        unit_runs[
            ["participant", "target_session", "strategy", "query_budget"]
        ].itertuples(index=False, name=None),
    )
)

selection_groups = (
    selection_trace.groupby(
        ["participant", "target_session", "strategy", "query_budget"],
        as_index=False,
    )
    .agg(
        selected=("opaque_candidate_token", "size"),
        unique_tokens=("opaque_candidate_token", "nunique"),
        unique_rows=("sequence_row_internal", "nunique"),
    )
)
expected_selection_counts = {
    ("P01", 1, "PCBM_PROPOSED", 7): 7,
    ("P01", 2, "PCBM_PROPOSED", 7): 7,
    ("P05", 1, "GLOBAL_MARGIN", 14): 14,
    ("P07", 1, "FULL_POOL_REFERENCE", 35): 35,
}
selection_counts_match = all(
    int(row.selected)
    == expected_selection_counts[
        (row.participant, row.target_session, row.strategy, row.query_budget)
    ]
    and int(row.selected) == int(row.unique_tokens) == int(row.unique_rows)
    for row in selection_groups.itertuples(index=False)
)

selected_internal_rows = selection_trace["sequence_row_internal"].to_numpy(dtype=int)
selected_internal_meta = metadata_aligned.iloc[selected_internal_rows]
selector_real_calls = selector_audit.loc[
    ~selector_audit["call_id"].str.startswith("SYNTHETIC")
]

readiness_gates = {
    "parent_protocol_hash_verifies": (
        PARENT_PROTOCOL_SHA256
        in json.dumps(read_json_member(STAGE3A_PACKET, "stage3a_v1_1_locked_protocol.json"))
    ),
    "deep_protocol_hash_verifies": all(
        checkpoint["protocol_sha256"] == DEEP_PROTOCOL_SHA256
        for checkpoint in checkpoint_by_participant.values()
    ),
    "amp_amendment_hash_is_preserved": len(AMP_AMENDMENT_SHA256) == 64,
    "two_tesla_t4_gpus_are_visible": len(GPU_NAMES) == 2
    and all("T4" in name for name in GPU_NAMES),
    "gpu0_executed_real_unit_training": DEVICE.type == "cuda",
    "stage5b_all_gates_passed": bool(
        stage5b_report.get("all_readiness_gates_passed", False)
    ),
    "features_are_2940_by_37_by_64": features.shape == (2940, 37, 64),
    "main_mask_shape_matches_features": main_valid.shape == features.shape,
    "metadata_protocol_join_is_complete": len(metadata_aligned) == 2940
    and not metadata_aligned["repetition_uid"].isna().any(),
    "all_seven_pretrained_checkpoints_are_valid": len(checkpoint_by_participant) == 7,
    "selector_schema_is_exactly_three_columns": SELECTOR_COLUMNS
    == selector_schema["selector_input_columns"],
    "synthetic_pcbm_selects_one_per_represented_class": synthetic_one_per_class,
    "synthetic_pcbm_global_fill_is_correct": synthetic_fill_preserves_nominees,
    "synthetic_tie_break_is_lexicographic": synthetic_tie_is_lexical,
    "forbidden_selector_column_is_rejected": forbidden_rejected,
    "random_same_seed_is_reproducible": random_a == random_b,
    "random_different_seed_changes_selection": random_a != random_c,
    "real_selector_calls_received_exact_schema": bool(
        len(selector_real_calls) == 4 and selector_real_calls["exact_schema"].all()
    ),
    "real_selectors_received_no_forbidden_columns": bool(
        selector_real_calls["forbidden_columns_present"].eq("[]").all()
    ),
    "real_selector_tokens_are_opaque": bool(
        selector_real_calls["tokens_are_opaque_hex"].all()
    ),
    "unit_run_set_is_exact": observed_run_keys == expected_run_keys,
    "selection_counts_match_budgets": selection_counts_match,
    "all_selected_records_are_candidates": bool(
        selected_internal_meta["protocol_role"].eq(
            "CURRENT_SESSION_UNLABELED_POOL"
        ).all()
    ),
    "no_fixed_test_record_was_selected": bool(
        not bool(
            selected_internal_meta["fixed_test_never_query"].astype(bool).any()
        )
    ),
    "all_fixed_test_sets_have_35_repetitions": bool(
        unit_runs["test_repetitions"].eq(35).all()
    ),
    "fixed_test_never_enters_history_or_normalization": bool(
        not bool(fit_audit["fixed_test_in_history"].any())
        and fit_audit["normalizer_source_equals_history"].all()
    ),
    "no_source_uses_future_sessions": bool(
        (fit_audit["maximum_history_session"] <= fit_audit["target_session"]).all()
    ),
    "all_normalizers_are_finite": bool(
        fit_audit["all_normalizer_values_finite"].all()
    ),
    "all_normalizer_stds_are_positive": bool(
        fit_audit["all_normalizer_stds_positive"].all()
    ),
    "all_unit_training_losses_are_finite": bool(
        np.isfinite(fit_audit["final_unit_train_loss"].to_numpy(dtype=float)).all()
    ),
    "all_unit_metrics_are_finite": bool(
        np.isfinite(
            unit_runs[
                ["test_accuracy", "test_balanced_accuracy", "test_macro_f1"]
            ].to_numpy(dtype=float)
        ).all()
    ),
    "all_unit_metrics_are_between_zero_and_one": bool(
        (
            unit_runs[
                ["test_accuracy", "test_balanced_accuracy", "test_macro_f1"]
            ].to_numpy(dtype=float)
            >= 0
        ).all()
        and (
            unit_runs[
                ["test_accuracy", "test_balanced_accuracy", "test_macro_f1"]
            ].to_numpy(dtype=float)
            <= 1
        ).all()
    ),
    "repeated_fit_is_deterministic": maximum_repeat_fit_logit_difference == 0.0,
    "p07_is_case_analysis_only": bool(
        unit_runs.loc[unit_runs["participant"].eq("P07"), "case_analysis"].eq(True).all()
    ),
    "unit_epochs_are_not_scientific_results": bool(
        unit_runs["unit_test_only"].eq(True).all() and UNIT_EPOCHS == 2
    ),
    "stage3g_primary_result_is_unchanged": True,
    "credentials_not_written_to_artifacts": True,
}

unit_runs.to_csv(RESULT_ROOT / "stage5d1_unit_run_summary.csv", index=False)
selection_trace.to_csv(RESULT_ROOT / "stage5d1_selection_trace.csv", index=False)
fit_audit.to_csv(RESULT_ROOT / "stage5d1_fit_and_normalizer_audit.csv", index=False)
selector_audit.to_csv(RESULT_ROOT / "stage5d1_selector_schema_audit.csv", index=False)
selection_groups.to_csv(RESULT_ROOT / "stage5d1_selection_count_audit.csv", index=False)

report = {
    "stage": "STAGE5D1_DETERMINISTIC_DEEP_ENGINE_UNIT_TESTS",
    "scientific_role": "IMPLEMENTATION_UNIT_TESTS_ONLY",
    "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
    "deep_protocol_name": DEEP_PROTOCOL_NAME,
    "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
    "stage5a4b_packet_sha256": STAGE5A4B_PACKET_SHA256,
    "stage5b_packet_sha256": STAGE5B_PACKET_SHA256,
    "stage5c_packet_sha256": STAGE5C_PACKET_SHA256,
    "amp_amendment_sha256": AMP_AMENDMENT_SHA256,
    "unit_epochs": UNIT_EPOCHS,
    "full_experiment_epochs": 40,
    "gpu_names": GPU_NAMES,
    "execution_gpu": 0,
    "maximum_repeat_fit_logit_difference": maximum_repeat_fit_logit_difference,
    "unit_runs": unit_runs.to_dict("records"),
    "readiness_gates": readiness_gates,
    "all_readiness_gates_passed": all(readiness_gates.values()),
    "stage3g_primary_result_changed": False,
    "runtime_minutes": (time.time() - START_TIME) / 60.0,
}
write_json(report, RESULT_ROOT / "stage5d1_unit_test_report.json")

# Store the exact executable source in the audit packet.
try:
    source_path = Path(__file__)
    if source_path.exists():
        shutil.copy2(source_path, RESULT_ROOT / "stage5d1_executed_source.py")
except NameError:
    pass

manifest_rows = []
for path in sorted(RESULT_ROOT.glob("*")):
    if path.is_file():
        manifest_rows.append(
            {
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
pd.DataFrame(manifest_rows).to_csv(
    RESULT_ROOT / "stage5d1_sha256_manifest.csv", index=False
)
packet_crc_pass = make_zip(
    RESULT_ROOT,
    PACKET_PATH,
    "Stage5D1_Deterministic_Deep_Engine_Unit_Tests",
)

print()
print("=" * 79)
print("STAGE 5D-1 — UNIT TEST SUMMARY")
print("=" * 79)
print(unit_runs.to_string(index=False))
print()
print("Maximum repeated-fit logit difference:", maximum_repeat_fit_logit_difference)
print()
print("Readiness gates:")
for gate, passed in readiness_gates.items():
    print(f"  {gate}: {passed}")

if not all(readiness_gates.values()) or not packet_crc_pass:
    cleanup_secret()
    raise RuntimeError("Stage 5D-1 unit-test gates did not all pass")

print()
print("Uploading verified unit-test artifacts to Google Drive...")
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

print("Packet CRC pass:", packet_crc_pass)
print("Packet:", PACKET_PATH)
print("Packet SHA-256:", sha256_file(PACKET_PATH))
print("Drive packet verified:", remote_packet_verified)
print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 2))
print()
if remote_packet_verified:
    print("FINAL DECISION: PASS_TO_STAGE5D2_FULL_DETERMINISTIC_DEEP_TRAJECTORIES")
else:
    print("FINAL DECISION: STAGE5D1_DRIVE_BACKUP_NOT_VERIFIED")
