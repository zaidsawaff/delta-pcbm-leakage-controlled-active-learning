import argparse
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
# STAGE 5D-2 — FULL DETERMINISTIC DEEP TRAJECTORIES
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
STAGE5D1_PACKET_SHA256 = (
    "64e505b15225ad92ac647c33d63f96b40ce0db0cdca75630faefc4b843e10de6"
)
AMP_AMENDMENT_SHA256 = (
    "d303d3a7059855c95af1106db6ebf440d46a9ebe75165498748b178ee4b5c9a9"
)

PARTICIPANTS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07"]
ABLE_BODIED = ["P01", "P02", "P03", "P04", "P05", "P06"]
GPU_ASSIGNMENTS = {
    0: ["P01", "P03", "P05", "P07"],
    1: ["P02", "P04", "P06"],
}
TRAJECTORY_PLAN = [
    ("NO_ADAPTATION_REFERENCE", 0),
    ("PCBM_PROPOSED", 7),
    ("PCBM_PROPOSED", 14),
    ("PCBM_PROPOSED", 21),
    ("GLOBAL_MARGIN", 7),
    ("GLOBAL_MARGIN", 14),
    ("GLOBAL_MARGIN", 21),
    ("FULL_POOL_REFERENCE", 35),
]
BUDGET_TO_ROUNDS = {0: 0, 7: 1, 14: 2, 21: 3, 35: 5}
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
SELECTION_TRACE_COLUMNS = [
    "trajectory_id",
    "participant",
    "target_session",
    "strategy",
    "query_budget",
    "round_index",
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
    "round_index",
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
    "selector_name",
    "rows_received",
    "columns_received",
    "exact_schema",
    "forbidden_columns_present",
    "tokens_are_opaque_hex",
]
FIT_AUDIT_COLUMNS = [
    "participant",
    "strategy",
    "query_budget",
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
    "opaque_test_token",
    "true_label",
    "predicted_label",
] + [f"logit_label_{label}" for label in range(7)] + [
    f"probability_label_{label}" for label in range(7)
]

CHANNELS = 64
WINDOWS = 37
CLASSES = 7
TARGET_EPOCHS = 40
TARGET_BATCH_SIZE = 16
ENCODER_LEARNING_RATE = 1.0e-4
CLASSIFIER_LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-3
LABEL_SMOOTHING = 0.05
AMP_INITIAL_SCALE = 1024.0
AMP_GROWTH_INTERVAL = 10000
SYNC_INTERVAL_SECONDS = 90
STATUS_INTERVAL_SECONDS = 60

WORKING = Path("/kaggle/working")
TOOLS = WORKING / "_stage5_tools"
TOOLS.mkdir(parents=True, exist_ok=True)
RCLONE = TOOLS / "rclone"

INPUT_ROOT = WORKING / "STAGE5D2_FROZEN_INPUTS"
RESULT_ROOT = WORKING / "DELTA_STAGE5_DEEP_RESULTS" / "Stage5D2_Deterministic"
CACHE_ROOT = WORKING / "STAGE5D2_STATE_CACHE"
PACKET_PATH = WORKING / "stage5d2_full_deterministic_deep_trajectories_packet.zip"
for directory in [INPUT_ROOT, RESULT_ROOT, CACHE_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

EVIDENCE_ROOT = Path(
    "/kaggle/input/datasets/zaidalsawaff/delta-q1-stage5-evidence-archives-v1"
)
STAGE3A_PACKET = EVIDENCE_ROOT / "stage3a_v1_1_protocol_amendment_packet.zip.bin"
STAGE5A4B_PACKET = WORKING / "stage5a4b_deep_protocol_lock_packet.zip"
STAGE5B_PACKET = WORKING / "stage5b_deep_sequence_assembly_packet.zip"
STAGE5C_PACKET = WORKING / "stage5c1_dual_gpu_loso_pretraining_packet.zip"
STAGE5D1_PACKET = WORKING / "stage5d1_deterministic_engine_unit_test_packet.zip"

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
    STAGE5D1_PACKET: (
        REMOTE_BASE
        + "/Deep_Training/Stage5D1_Deterministic_Engine_Unit_Tests/"
        + STAGE5D1_PACKET.name
    ),
}
REMOTE_OUTPUT = (
    REMOTE_BASE + "/Deep_Training/Stage5D2_Full_Deterministic_Deep_Trajectories"
)

CONFIG_PATH = None


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


def atomic_json(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_csv(dataframe, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def atomic_torch_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise ValueError(f"Expected one {basename}; found {len(matches)}")
        destination.write_bytes(archive.read(matches[0]))


def extract_checkpoint(packet, participant, destination):
    suffix = f"/{participant}/best.pt"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one pretrained checkpoint for {participant}; found {len(matches)}"
            )
        destination.write_bytes(archive.read(matches[0]))


def make_zip(source_directory, destination, archive_root):
    destination = Path(destination)
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
    temporary_root = Path(tempfile.mkdtemp(prefix="stage5d2_rclone_", dir="/tmp"))
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
    decoded = base64.b64decode(encoded, validate=True)
    del encoded
    parser = configparser.ConfigParser()
    parser.read_string(decoded.decode("utf-8"))
    valid = (
        "gdrive_stage5" in parser.sections()
        and parser.get("gdrive_stage5", "type", fallback="") == "drive"
        and parser.get("gdrive_stage5", "scope", fallback="") == "drive.file"
    )
    if not valid:
        raise RuntimeError("Restricted Google Drive remote verification failed")
    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="stage5d2_",
        suffix=".conf",
        dir="/tmp",
        delete=False,
    )
    temporary.write(decoded)
    temporary.close()
    del decoded
    CONFIG_PATH = Path(temporary.name)
    os.chmod(CONFIG_PATH, 0o600)
    return True


def rclone(arguments, check=True):
    return subprocess.run(
        [str(RCLONE), "--config", str(CONFIG_PATH), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


# =============================================================================
# WORKER IMPLEMENTATION
# =============================================================================


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


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


def transform_data(features, valid_mask, row_indices, means, stds):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    raw = np.asarray(features[row_indices], dtype=np.float64)
    valid = np.asarray(valid_mask[row_indices], dtype=bool)
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


def history_fingerprint(metadata, participant, history_rows):
    tokens = sorted(
        metadata.iloc[np.asarray(history_rows, dtype=np.int64)][
            "opaque_candidate_token"
        ].astype(str).tolist()
    )
    payload = participant + "|" + "|".join(tokens)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def history_seed(metadata, primary_seeds, participant, history_rows):
    tokens = sorted(
        metadata.iloc[np.asarray(history_rows, dtype=np.int64)][
            "opaque_candidate_token"
        ].astype(str).tolist()
    )
    payload = (
        str(int(primary_seeds[participant]))
        + "|"
        + participant
        + "|"
        + "|".join(tokens)
    )
    value = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return int(value % 2_000_000_000 + 1)


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
    for parameter in getattr(model, stem_attribute).parameters():
        parameter.requires_grad = False
    for block in model.blocks[:2]:
        for parameter in block.parameters():
            parameter.requires_grad = False
    stem_prefix = stem_attribute + "."
    frozen_names = [
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    ]
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not frozen_names or not trainable_names:
        raise RuntimeError("Invalid target-adaptation freeze state")
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
    return stem_attribute, frozen_names, trainable_names


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
        raise RuntimeError("Target optimizer groups are incomplete")
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": ENCODER_LEARNING_RATE},
            {"params": classifier_parameters, "lr": CLASSIFIER_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def load_model_class(input_root):
    model_path = Path(input_root) / "stage5b_mask_aware_rms_tcn.py"
    spec = importlib.util.spec_from_file_location(
        "stage5b_mask_aware_rms_tcn", model_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MaskAwareRMSTCN


def state_checkpoint_payload(
    model,
    participant,
    history_rows,
    fingerprint,
    seed,
    means,
    stds,
    counts,
    stem_attribute,
    train_losses,
):
    return {
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "engine_source_sha256": sha256_file(Path(__file__)),
        "participant": participant,
        "history_rows": np.asarray(history_rows, dtype=np.int64),
        "history_fingerprint": fingerprint,
        "history_seed": int(seed),
        "normalizer_means": means,
        "normalizer_stds": stds,
        "normalizer_counts": counts,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "stem_attribute": stem_attribute,
        "target_epochs": TARGET_EPOCHS,
        "amp_initial_scale": AMP_INITIAL_SCALE,
        "amp_growth_interval": AMP_GROWTH_INTERVAL,
        "train_losses": train_losses,
    }


def load_fitted_state(
    checkpoint_path,
    expected_participant,
    expected_history_rows,
    expected_fingerprint,
    model_class,
    device,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        valid = (
            checkpoint.get("deep_protocol_sha256") == DEEP_PROTOCOL_SHA256
            and checkpoint.get("engine_source_sha256")
            == sha256_file(Path(__file__))
            and checkpoint.get("participant") == expected_participant
            and checkpoint.get("history_fingerprint") == expected_fingerprint
            and int(checkpoint.get("target_epochs")) == TARGET_EPOCHS
            and np.array_equal(
                np.asarray(checkpoint["history_rows"], dtype=np.int64),
                np.asarray(expected_history_rows, dtype=np.int64),
            )
            and np.isfinite(checkpoint["normalizer_means"]).all()
            and np.isfinite(checkpoint["normalizer_stds"]).all()
            and (np.asarray(checkpoint["normalizer_stds"]) > 0).all()
            and (np.asarray(checkpoint["normalizer_counts"]) > 0).all()
            and all(
                torch.isfinite(value).all().item()
                for value in checkpoint["model_state_dict"].values()
            )
        )
        if not valid:
            return None
        model = model_class().to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        freeze_target_model(model)
        return {
            "model": model,
            "means": np.asarray(checkpoint["normalizer_means"], dtype=np.float64),
            "stds": np.asarray(checkpoint["normalizer_stds"], dtype=np.float64),
            "counts": np.asarray(checkpoint["normalizer_counts"], dtype=np.int64),
            "history_rows": np.asarray(checkpoint["history_rows"], dtype=np.int64),
            "fingerprint": checkpoint["history_fingerprint"],
            "seed": int(checkpoint["history_seed"]),
            "stem_attribute": checkpoint["stem_attribute"],
            "train_losses": [float(value) for value in checkpoint["train_losses"]],
            "checkpoint_payload": checkpoint,
        }
    except Exception:
        return None


def fit_or_load_history_state(
    participant,
    history_rows,
    features,
    valid_mask,
    metadata,
    primary_seeds,
    pretrained_path,
    model_class,
    device,
    cache_root,
    trajectory_state_path,
    status_context,
):
    history_rows = np.asarray(sorted(set(map(int, history_rows))), dtype=np.int64)
    if len(history_rows) == 0:
        raise ValueError("History cannot be empty")
    history_meta = metadata.iloc[history_rows]
    if not history_meta["participant"].eq(participant).all():
        raise RuntimeError("Cross-participant target-history contamination")
    if not history_meta["eligible_for_training"].astype(bool).all():
        raise RuntimeError("A non-training-eligible repetition entered history")
    if history_meta["fixed_test_never_query"].astype(bool).any():
        raise RuntimeError("A fixed-test repetition entered training history")
    if sorted(history_meta["label"].unique().tolist()) != list(range(CLASSES)):
        raise RuntimeError("History does not contain all seven classes")

    fingerprint = history_fingerprint(metadata, participant, history_rows)
    cache_path = Path(cache_root) / participant / f"{fingerprint}.pt"
    for candidate, source in [
        (trajectory_state_path, "TRAJECTORY_CHECKPOINT"),
        (cache_path, "CONTENT_ADDRESSED_CACHE"),
    ]:
        state = load_fitted_state(
            candidate,
            participant,
            history_rows,
            fingerprint,
            model_class,
            device,
        )
        if state is not None:
            state["cache_source"] = source
            return state

    means, stds, counts = fit_normalizer(
        features[history_rows], valid_mask[history_rows]
    )
    x = transform_data(features, valid_mask, history_rows, means, stds)
    y = history_meta["label"].to_numpy(dtype=np.int64)
    seed = history_seed(metadata, primary_seeds, participant, history_rows)
    set_seed(seed)

    pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if (
        pretrained.get("protocol_sha256") != DEEP_PROTOCOL_SHA256
        or pretrained.get("target_participant") != participant
        or pretrained.get("target_data_used") is not False
    ):
        raise RuntimeError(f"Invalid pretrained checkpoint for {participant}")

    model = model_class().to(device)
    model.load_state_dict(pretrained["model_state_dict"], strict=True)
    stem_attribute, frozen_names, trainable_names = freeze_target_model(model)
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
    train_losses = []

    for epoch in range(1, TARGET_EPOCHS + 1):
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
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
        train_losses.append(float(total_loss / total_examples))
        if epoch in [1, 10, 20, 30, 40]:
            print(
                f"FIT | {status_context} | epoch={epoch:02d}/40 | "
                f"history={len(history_rows)} | loss={train_losses[-1]:.6f}",
                flush=True,
            )

    if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise RuntimeError("Non-finite adapted model state")
    payload = state_checkpoint_payload(
        model,
        participant,
        history_rows,
        fingerprint,
        seed,
        means,
        stds,
        counts,
        stem_attribute,
        train_losses,
    )
    atomic_torch_save(payload, cache_path)
    state = {
        "model": model,
        "means": means,
        "stds": stds,
        "counts": counts,
        "history_rows": history_rows,
        "fingerprint": fingerprint,
        "seed": seed,
        "stem_attribute": stem_attribute,
        "train_losses": train_losses,
        "checkpoint_payload": payload,
        "cache_source": "NEW_FIT",
        "frozen_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if not parameter.requires_grad
            )
        ),
        "trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_names": trainable_names,
    }
    return state


@torch.no_grad()
def predict_rows(state, features, valid_mask, row_indices, device):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    x = transform_data(
        features,
        valid_mask,
        row_indices,
        state["means"],
        state["stds"],
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    state["model"].eval()
    batches = []
    for (inputs,) in loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = state["model"](inputs)
        batches.append(logits.float().cpu().numpy())
    logits = np.concatenate(batches, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    ordered = np.sort(probabilities, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    predictions = np.argmax(probabilities, axis=1).astype(int)
    if not (
        np.isfinite(logits).all()
        and np.isfinite(probabilities).all()
        and np.isfinite(margins).all()
    ):
        raise RuntimeError("Non-finite deep scores")
    return logits, probabilities, predictions, margins


def classification_metrics(y_true, y_pred):
    matrix = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
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
        2.0 * precision * recall,
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


def selector_order(frame):
    return frame.sort_values(
        ["margin", "opaque_candidate_token"], kind="mergesort"
    )


def validate_selector_frame(frame):
    if frame.columns.tolist() != SELECTOR_COLUMNS:
        raise ValueError(f"Selector schema drift: {frame.columns.tolist()}")
    if set(frame.columns).intersection(FORBIDDEN_SELECTOR_COLUMNS):
        raise ValueError("Forbidden selector column was exposed")
    if frame["opaque_candidate_token"].duplicated().any():
        raise ValueError("Duplicate selector token")
    if not frame["opaque_candidate_token"].str.fullmatch(r"[0-9a-f]{24}").all():
        raise ValueError("Non-opaque selector token")
    if not np.isfinite(frame["margin"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite selector margin")


def select_pcbm(frame):
    validate_selector_frame(frame)
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


def select_global_margin(frame):
    validate_selector_frame(frame)
    selected = selector_order(frame).head(7)["opaque_candidate_token"].tolist()
    if len(selected) != 7 or len(set(selected)) != 7:
        raise RuntimeError("Global margin did not select seven unique tokens")
    return selected


def rows_for_role(metadata, participant, session, role):
    mask = (
        metadata["participant"].eq(participant)
        & metadata["session"].eq(int(session))
        & metadata["protocol_role"].eq(role)
    )
    rows = metadata.loc[mask, "sequence_row"].to_numpy(dtype=np.int64)
    expected = 35
    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} {role} rows for {participant} session {session}; "
            f"found {len(rows)}"
        )
    return rows


def initial_history_rows(metadata, participant):
    return rows_for_role(
        metadata, participant, 0, "INITIAL_LABELED_CALIBRATION"
    )


def candidate_rows(metadata, participant, session):
    return rows_for_role(
        metadata, participant, session, "CURRENT_SESSION_UNLABELED_POOL"
    )


def fixed_test_rows(metadata, participant, session):
    return rows_for_role(
        metadata, participant, session, "TARGET_FIXED_TEST_NEVER_QUERY"
    )


def reveal_selected_rows(metadata, selected_tokens, remaining_rows):
    remaining = metadata.iloc[np.asarray(remaining_rows, dtype=np.int64)][
        ["opaque_candidate_token", "sequence_row"]
    ]
    mapping = dict(zip(remaining["opaque_candidate_token"], remaining["sequence_row"]))
    if any(token not in mapping for token in selected_tokens):
        raise RuntimeError("Selector returned a token outside the remaining pool")
    selected_rows = np.asarray([mapping[token] for token in selected_tokens], dtype=np.int64)
    selected_meta = metadata.iloc[selected_rows]
    if not selected_meta["protocol_role"].eq(
        "CURRENT_SESSION_UNLABELED_POOL"
    ).all():
        raise RuntimeError("A non-candidate record was selected")
    if selected_meta["fixed_test_never_query"].astype(bool).any():
        raise RuntimeError("A fixed-test record was selected")
    return selected_rows


def save_progress(progress, trajectory_directory, state=None):
    trajectory_directory = Path(trajectory_directory)
    if state is not None:
        atomic_torch_save(
            state["checkpoint_payload"], trajectory_directory / "current_state.pt"
        )
        progress["current_state_fingerprint"] = state["fingerprint"]
    atomic_json(progress, trajectory_directory / "progress.json")
    atomic_csv(
        pd.DataFrame(progress["selection_trace"], columns=SELECTION_TRACE_COLUMNS),
        trajectory_directory / "selection_trace.csv",
    )
    atomic_csv(
        pd.DataFrame(progress["candidate_audit"], columns=CANDIDATE_AUDIT_COLUMNS),
        trajectory_directory / "candidate_score_audit.csv",
    )
    atomic_csv(
        pd.DataFrame(progress["selector_audit"], columns=SELECTOR_AUDIT_COLUMNS),
        trajectory_directory / "selector_schema_audit.csv",
    )
    atomic_csv(
        pd.DataFrame(progress["fit_audit"], columns=FIT_AUDIT_COLUMNS),
        trajectory_directory / "fit_normalizer_audit.csv",
    )
    atomic_csv(
        pd.DataFrame(progress["fold_results"], columns=FOLD_RESULT_COLUMNS),
        trajectory_directory / "fold_results.csv",
    )
    atomic_csv(
        pd.DataFrame(progress["predictions"], columns=PREDICTION_COLUMNS),
        trajectory_directory / "repetition_predictions.csv",
    )


def append_fit_audit(progress, state, metadata, participant, session, round_index):
    if any(
        row.get("history_fingerprint") == state["fingerprint"]
        for row in progress["fit_audit"]
    ):
        return
    history_meta = metadata.iloc[state["history_rows"]]
    progress["fit_audit"].append(
        {
            "participant": participant,
            "strategy": progress["strategy"],
            "query_budget": progress["query_budget"],
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
            "target_epochs": TARGET_EPOCHS,
            "final_train_loss": float(state["train_losses"][-1]),
        }
    )


def state_for_history(
    progress,
    participant,
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
):
    status = (
        f"{participant} {progress['strategy']} K{progress['query_budget']:02d} "
        f"S{session:02d} R{round_index:02d}"
    )
    state = fit_or_load_history_state(
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
        trajectory_directory / "current_state.pt",
        status,
    )
    append_fit_audit(progress, state, metadata, participant, session, round_index)
    return state


def evaluate_fixed_test(
    state,
    progress,
    features,
    valid_mask,
    metadata,
    participant,
    session,
    device,
):
    test_rows = fixed_test_rows(metadata, participant, session)
    if np.intersect1d(test_rows, state["history_rows"]).size:
        raise RuntimeError("Fixed test entered history")
    logits, probabilities, predictions, _ = predict_rows(
        state, features, valid_mask, test_rows, device
    )
    truths = metadata.iloc[test_rows]["label"].to_numpy(dtype=int)
    metrics = classification_metrics(truths, predictions)
    run_id = f"{progress['trajectory_id']}_S{session:02d}"
    progress["fold_results"].append(
        {
            "run_id": run_id,
            "trajectory_id": progress["trajectory_id"],
            "participant": participant,
            "target_session": int(session),
            "strategy": progress["strategy"],
            "query_budget": int(progress["query_budget"]),
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
            "strategy": progress["strategy"],
            "query_budget": int(progress["query_budget"]),
            "opaque_test_token": str(test_meta.iloc[index]["opaque_candidate_token"]),
            "true_label": int(truths[index]),
            "predicted_label": int(predictions[index]),
        }
        for label in range(CLASSES):
            row[f"logit_label_{label}"] = float(logits[index, label])
            row[f"probability_label_{label}"] = float(probabilities[index, label])
        progress["predictions"].append(row)


def trajectory_complete_and_valid(directory, trajectory_id):
    complete_path = Path(directory) / "complete.json"
    if not complete_path.exists():
        return False
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        return bool(
            complete.get("complete") is True
            and complete.get("trajectory_id") == trajectory_id
            and complete.get("deep_protocol_sha256") == DEEP_PROTOCOL_SHA256
            and int(complete.get("fold_count")) == 5
            and complete.get("all_gates_passed") is True
        )
    except Exception:
        return False


def new_progress(participant, strategy, budget, history_rows):
    return {
        "version": 1,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "trajectory_id": f"{participant}_{strategy}_K{budget:02d}",
        "participant": participant,
        "strategy": strategy,
        "query_budget": int(budget),
        "case_analysis": participant == "P07",
        "history_rows": list(map(int, history_rows)),
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
    trajectory_directory, participant, strategy, budget, initial_rows
):
    progress_path = Path(trajectory_directory) / "progress.json"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            valid = (
                progress.get("deep_protocol_sha256") == DEEP_PROTOCOL_SHA256
                and progress.get("participant") == participant
                and progress.get("strategy") == strategy
                and int(progress.get("query_budget")) == int(budget)
                and progress.get("trajectory_id")
                == f"{participant}_{strategy}_K{budget:02d}"
            )
            if valid:
                return progress
        except Exception:
            pass
    return new_progress(participant, strategy, budget, initial_rows)


def run_trajectory(
    participant,
    strategy,
    budget,
    features,
    valid_mask,
    metadata,
    primary_seeds,
    pretrained_path,
    model_class,
    device,
    cache_root,
    result_root,
):
    trajectory_id = f"{participant}_{strategy}_K{budget:02d}"
    trajectory_directory = Path(result_root) / participant / trajectory_id
    trajectory_directory.mkdir(parents=True, exist_ok=True)
    if trajectory_complete_and_valid(trajectory_directory, trajectory_id):
        current_state = trajectory_directory / "current_state.pt"
        if current_state.exists():
            current_state.unlink()
        print(f"SKIP COMPLETE | {trajectory_id}", flush=True)
        return

    initial_rows = initial_history_rows(metadata, participant)
    progress = load_or_create_progress(
        trajectory_directory, participant, strategy, budget, initial_rows
    )
    history = list(map(int, progress["history_rows"]))
    state = state_for_history(
        progress,
        participant,
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
        trajectory_directory,
    )
    save_progress(progress, trajectory_directory, state)
    rounds = BUDGET_TO_ROUNDS[int(budget)]

    while int(progress["next_session"]) <= 5:
        session = int(progress["next_session"])
        if progress["active_session"] is None:
            progress["active_session"] = session
            progress["completed_rounds_in_active_session"] = 0
            progress["remaining_rows"] = candidate_rows(
                metadata, participant, session
            ).tolist()
            save_progress(progress, trajectory_directory, state)
        elif int(progress["active_session"]) != session:
            raise RuntimeError("Inconsistent active-session checkpoint")

        remaining = np.asarray(progress["remaining_rows"], dtype=np.int64)
        completed_rounds = int(progress["completed_rounds_in_active_session"])

        if strategy == "NO_ADAPTATION_REFERENCE":
            if budget != 0 or rounds != 0:
                raise RuntimeError("Invalid no-adaptation trajectory")

        elif strategy == "FULL_POOL_REFERENCE":
            if completed_rounds == 0:
                selected_rows = remaining.copy()
                selected_tokens = metadata.iloc[selected_rows][
                    "opaque_candidate_token"
                ].astype(str).tolist()
                new_history = history + selected_rows.tolist()
                new_state = state_for_history(
                    progress,
                    participant,
                    session,
                    5,
                    new_history,
                    features,
                    valid_mask,
                    metadata,
                    primary_seeds,
                    pretrained_path,
                    model_class,
                    device,
                    cache_root,
                    trajectory_directory,
                )
                for order, (token, row_index) in enumerate(
                    zip(selected_tokens, selected_rows), start=1
                ):
                    progress["selection_trace"].append(
                        {
                            "trajectory_id": trajectory_id,
                            "participant": participant,
                            "target_session": session,
                            "strategy": strategy,
                            "query_budget": budget,
                            "round_index": 1,
                            "selection_order_in_round": order,
                            "opaque_candidate_token": token,
                            "sequence_row_internal": int(row_index),
                            "true_label_revealed_after_selection": int(
                                metadata.iloc[row_index]["label"]
                            ),
                            "selector_visible_columns": "FULL_POOL_ALL_CANDIDATES",
                        }
                    )
                history = new_history
                state = new_state
                progress["history_rows"] = history
                progress["remaining_rows"] = []
                progress["completed_rounds_in_active_session"] = 5
                save_progress(progress, trajectory_directory, state)

        else:
            for round_index in range(completed_rounds + 1, rounds + 1):
                logits, probabilities, predictions, margins = predict_rows(
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
                validate_selector_frame(visible)
                selected_tokens = (
                    select_pcbm(visible)
                    if strategy == "PCBM_PROPOSED"
                    else select_global_margin(visible)
                )
                selected_rows = reveal_selected_rows(
                    metadata, selected_tokens, remaining
                )
                new_history = history + selected_rows.tolist()
                new_state = state_for_history(
                    progress,
                    participant,
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
                    trajectory_directory,
                )
                call_id = (
                    f"{trajectory_id}_S{session:02d}_R{round_index:02d}"
                )
                progress["selector_audit"].append(
                    {
                        "call_id": call_id,
                        "trajectory_id": trajectory_id,
                        "selector_name": strategy,
                        "rows_received": len(visible),
                        "columns_received": json.dumps(visible.columns.tolist()),
                        "exact_schema": visible.columns.tolist() == SELECTOR_COLUMNS,
                        "forbidden_columns_present": json.dumps(
                            sorted(
                                set(visible.columns).intersection(
                                    FORBIDDEN_SELECTOR_COLUMNS
                                )
                            )
                        ),
                        "tokens_are_opaque_hex": bool(
                            visible["opaque_candidate_token"]
                            .str.fullmatch(r"[0-9a-f]{24}")
                            .all()
                        ),
                    }
                )
                selected_set = set(selected_tokens)
                for candidate_position, row in visible.reset_index(drop=True).iterrows():
                    progress["candidate_audit"].append(
                        {
                            "call_id": call_id,
                            "trajectory_id": trajectory_id,
                            "participant": participant,
                            "target_session": session,
                            "strategy": strategy,
                            "query_budget": budget,
                            "round_index": round_index,
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
                            "target_session": session,
                            "strategy": strategy,
                            "query_budget": budget,
                            "round_index": round_index,
                            "selection_order_in_round": order,
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
                progress["completed_rounds_in_active_session"] = round_index
                save_progress(progress, trajectory_directory, state)

        evaluate_fixed_test(
            state,
            progress,
            features,
            valid_mask,
            metadata,
            participant,
            session,
            device,
        )
        expected_source = 35 + int(budget) * session
        if len(history) != expected_source:
            raise RuntimeError(
                f"Source schedule mismatch for {trajectory_id} session {session}: "
                f"{len(history)} versus {expected_source}"
            )
        progress["next_session"] = session + 1
        progress["active_session"] = None
        progress["completed_rounds_in_active_session"] = 0
        progress["remaining_rows"] = []
        save_progress(progress, trajectory_directory, state)
        print(
            f"SESSION COMPLETE | {trajectory_id} | S{session:02d} | "
            f"history={len(history)} | "
            f"BA={progress['fold_results'][-1]['repetition_balanced_accuracy']:.6f}",
            flush=True,
        )

    final_state_path = trajectory_directory / "final_state.pt"
    atomic_torch_save(state["checkpoint_payload"], final_state_path)
    fold_frame = pd.DataFrame(progress["fold_results"])
    selection_frame = pd.DataFrame(progress["selection_trace"])
    candidate_frame = pd.DataFrame(progress["candidate_audit"])
    selector_frame = pd.DataFrame(progress["selector_audit"])
    fit_frame = pd.DataFrame(progress["fit_audit"])
    prediction_frame = pd.DataFrame(progress["predictions"])

    expected_selection_count = int(budget) * 5
    expected_selector_calls = (
        int(BUDGET_TO_ROUNDS[int(budget)]) * 5
        if strategy in ["PCBM_PROPOSED", "GLOBAL_MARGIN"]
        else 0
    )
    expected_candidate_rows = 0
    if strategy in ["PCBM_PROPOSED", "GLOBAL_MARGIN"]:
        expected_candidate_rows = 5 * sum(
            35 - 7 * prior_round
            for prior_round in range(BUDGET_TO_ROUNDS[int(budget)])
        )
    complete_gates = {
        "five_sessions_completed": len(fold_frame) == 5,
        "every_fold_has_35_tests": bool(fold_frame["test_repetitions"].eq(35).all()),
        "prediction_count_is_175": len(prediction_frame) == 175,
        "selection_count_matches_budget": len(selection_frame)
        == expected_selection_count,
        "selected_tokens_are_unique_within_session": bool(
            selection_frame.empty
            or selection_frame.groupby("target_session")[
                "opaque_candidate_token"
            ].nunique().eq(int(budget)).all()
        ),
        "selector_call_count_matches": len(selector_frame) == expected_selector_calls,
        "candidate_audit_count_matches": len(candidate_frame)
        == expected_candidate_rows,
        "selector_schema_is_exact": bool(
            selector_frame.empty or selector_frame["exact_schema"].all()
        ),
        "no_forbidden_selector_columns": bool(
            selector_frame.empty
            or selector_frame["forbidden_columns_present"].eq("[]").all()
        ),
        "all_metrics_are_finite": bool(
            np.isfinite(
                fold_frame[
                    [
                        "repetition_accuracy",
                        "repetition_balanced_accuracy",
                        "repetition_macro_f1",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "all_metrics_are_between_zero_and_one": bool(
            (
                fold_frame[
                    [
                        "repetition_accuracy",
                        "repetition_balanced_accuracy",
                        "repetition_macro_f1",
                    ]
                ].to_numpy(dtype=float)
                >= 0
            ).all()
            and (
                fold_frame[
                    [
                        "repetition_accuracy",
                        "repetition_balanced_accuracy",
                        "repetition_macro_f1",
                    ]
                ].to_numpy(dtype=float)
                <= 1
            ).all()
        ),
        "normalizers_are_finite": bool(
            fit_frame["all_normalizer_values_finite"].all()
        ),
        "normalizer_stds_are_positive": bool(
            fit_frame["all_normalizer_stds_positive"].all()
        ),
        "no_fixed_test_in_history": bool(
            not bool(fit_frame["fixed_test_in_history"].any())
        ),
        "no_future_session_used": bool(
            (
                fold_frame["maximum_history_session"]
                <= fold_frame["target_session"]
            ).all()
        ),
        "source_counts_match_schedule": bool(
            fold_frame.apply(
                lambda row: int(row["source_repetitions"])
                == 35 + int(budget) * int(row["target_session"]),
                axis=1,
            ).all()
        ),
        "final_history_count_matches": len(history) == 35 + int(budget) * 5,
        "final_state_is_finite": all(
            torch.isfinite(value).all().item()
            for value in state["checkpoint_payload"]["model_state_dict"].values()
        ),
        "p07_case_flag_is_correct": bool((participant == "P07") == progress["case_analysis"]),
    }
    complete = {
        "complete": True,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "trajectory_id": trajectory_id,
        "participant": participant,
        "strategy": strategy,
        "query_budget": int(budget),
        "case_analysis": participant == "P07",
        "fold_count": len(fold_frame),
        "selection_count": len(selection_frame),
        "candidate_audit_count": len(candidate_frame),
        "selector_call_count": len(selector_frame),
        "prediction_count": len(prediction_frame),
        "fit_state_count": len(fit_frame),
        "final_history_repetitions": len(history),
        "final_state_sha256": sha256_file(final_state_path),
        "mean_repetition_balanced_accuracy": float(
            fold_frame["repetition_balanced_accuracy"].mean()
        ),
        "readiness_gates": complete_gates,
        "all_gates_passed": all(complete_gates.values()),
    }
    if not complete["all_gates_passed"]:
        raise RuntimeError(f"Trajectory gates failed: {trajectory_id}: {complete_gates}")
    atomic_json(complete, trajectory_directory / "complete.json")
    current_state = trajectory_directory / "current_state.pt"
    if current_state.exists():
        current_state.unlink()
    print(
        f"TRAJECTORY COMPLETE | {trajectory_id} | "
        f"mean_BA={complete['mean_repetition_balanced_accuracy']:.6f}",
        flush=True,
    )


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
    participants = [value for value in arguments.participants.split(",") if value]
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
    metadata = pd.read_csv(input_root / "stage5d2_metadata_protocol_aligned.csv")
    metadata["participant"] = metadata["participant"].astype(str)
    for column in ["session", "label", "repetition", "sequence_row"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(int)
    primary_seed_frame = pd.read_csv(input_root / "stage5d2_primary_seeds.csv")
    primary_seeds = dict(
        zip(primary_seed_frame["participant"], primary_seed_frame["seed"])
    )
    model_class = load_model_class(input_root)
    print(
        f"WORKER START | physical_gpu={arguments.physical_gpu_label} | "
        f"visible_gpu={gpu_name} | participants={participants}",
        flush=True,
    )
    for participant in participants:
        pretrained_path = input_root / "pretrained" / f"{participant}_best.pt"
        for strategy, budget in TRAJECTORY_PLAN:
            run_trajectory(
                participant,
                strategy,
                budget,
                features,
                valid_mask,
                metadata,
                primary_seeds,
                pretrained_path,
                model_class,
                device,
                cache_root,
                result_root,
            )
    print(
        f"WORKER COMPLETE | physical_gpu={arguments.physical_gpu_label}",
        flush=True,
    )


# =============================================================================
# ORCHESTRATOR IMPLEMENTATION
# =============================================================================


def last_log_line(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return "NO LOG OUTPUT YET"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-1] if lines else "EMPTY LOG"


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
        STAGE5D1_PACKET: STAGE5D1_PACKET_SHA256,
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
        STAGE5D1_PACKET,
    ]:
        with zipfile.ZipFile(packet, "r") as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"CRC failure: {packet}")

    stage5d1_report = read_json_member(STAGE5D1_PACKET, "stage5d1_unit_test_report.json")
    if not stage5d1_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("Stage 5D-1 unit-test packet did not pass all gates")

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
    atomic_csv(aligned, INPUT_ROOT / "stage5d2_metadata_protocol_aligned.csv")

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
    if sorted(primary_seeds["participant"].tolist()) != PARTICIPANTS:
        raise RuntimeError("Primary deterministic seed schedule is incomplete")
    atomic_csv(primary_seeds, INPUT_ROOT / "stage5d2_primary_seeds.csv")

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
    checkpoint_gates = {}
    for participant in PARTICIPANTS:
        checkpoint = torch.load(
            INPUT_ROOT / "pretrained" / f"{participant}_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_gates[participant] = bool(
            checkpoint.get("protocol_sha256") == DEEP_PROTOCOL_SHA256
            and checkpoint.get("target_participant") == participant
            and checkpoint.get("target_data_used") is False
            and all(
                torch.isfinite(value).all().item()
                for value in checkpoint["model_state_dict"].values()
            )
        )
    gates = {
        "stage5a4b_hash_matches": hash_gates[STAGE5A4B_PACKET.name],
        "stage5b_hash_matches": hash_gates[STAGE5B_PACKET.name],
        "stage5c_hash_matches": hash_gates[STAGE5C_PACKET.name],
        "stage5d1_hash_matches": hash_gates[STAGE5D1_PACKET.name],
        "stage5d1_all_gates_passed": bool(
            stage5d1_report["all_readiness_gates_passed"]
        ),
        "feature_shape_is_2940_by_37_by_64": features.shape == (2940, 37, 64),
        "main_mask_shape_matches": valid_mask.shape == features.shape,
        "metadata_has_2940_aligned_rows": len(aligned) == 2940,
        "all_seven_pretrained_checkpoints_are_valid": all(
            checkpoint_gates.values()
        ),
        "primary_seeds_are_complete": len(primary_seeds) == 7,
        "amp_amendment_hash_is_preserved": len(AMP_AMENDMENT_SHA256) == 64,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Frozen-input readiness failure: {gates}")
    atomic_json(
        {
            "stage": "STAGE5D2_FROZEN_INPUT_AUDIT",
            "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
            "input_hash_gates": hash_gates,
            "checkpoint_gates": checkpoint_gates,
            "readiness_gates": gates,
            "all_readiness_gates_passed": all(gates.values()),
        },
        INPUT_ROOT / "stage5d2_frozen_input_audit.json",
    )
    return gates


def aggregate_and_finalize(worker_exit_codes, sync_successes, sync_failures, start_time):
    complete_files = sorted(RESULT_ROOT.glob("P*/**/complete.json"))
    complete_rows = [
        json.loads(path.read_text(encoding="utf-8")) for path in complete_files
    ]
    completion = pd.DataFrame(complete_rows)
    fold_frames = [
        pd.read_csv(path)
        for path in sorted(RESULT_ROOT.glob("P*/**/fold_results.csv"))
    ]
    selection_frames = [
        pd.read_csv(path)
        for path in sorted(RESULT_ROOT.glob("P*/**/selection_trace.csv"))
        if path.stat().st_size > 0
    ]
    candidate_frames = [
        pd.read_csv(path)
        for path in sorted(RESULT_ROOT.glob("P*/**/candidate_score_audit.csv"))
        if path.stat().st_size > 0
    ]
    selector_frames = [
        pd.read_csv(path)
        for path in sorted(RESULT_ROOT.glob("P*/**/selector_schema_audit.csv"))
        if path.stat().st_size > 0
    ]
    fit_frames = [
        pd.read_csv(path)
        for path in sorted(RESULT_ROOT.glob("P*/**/fit_normalizer_audit.csv"))
        if path.stat().st_size > 0
    ]
    prediction_frames = [
        pd.read_csv(path)
        for path in sorted(RESULT_ROOT.glob("P*/**/repetition_predictions.csv"))
        if path.stat().st_size > 0
    ]
    folds = pd.concat(fold_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    selectors = pd.concat(selector_frames, ignore_index=True)
    fits = pd.concat(fit_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    for frame, filename in [
        (completion, "stage5d2_trajectory_completion_summary.csv"),
        (folds, "stage5d2_fold_results.csv"),
        (selections, "stage5d2_selection_trace.csv"),
        (candidates, "stage5d2_candidate_score_audit.csv"),
        (selectors, "stage5d2_selector_schema_audit.csv"),
        (fits, "stage5d2_fit_normalizer_audit.csv"),
        (predictions, "stage5d2_repetition_predictions.csv"),
    ]:
        atomic_csv(frame, RESULT_ROOT / filename)

    able_folds = folds.loc[folds["participant"].isin(ABLE_BODIED)].copy()
    able_summary = (
        able_folds.groupby(["strategy", "query_budget"], as_index=False)
        .agg(
            participant_session_folds=("run_id", "nunique"),
            participants=("participant", "nunique"),
            mean_repetition_balanced_accuracy=(
                "repetition_balanced_accuracy",
                "mean",
            ),
            std_repetition_balanced_accuracy=(
                "repetition_balanced_accuracy",
                "std",
            ),
            mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
            total_repetition_errors=(
                "repetition_accuracy",
                lambda values: int(np.rint(((1.0 - values) * 35).sum())),
            ),
        )
        .sort_values(["query_budget", "strategy"])
        .reset_index(drop=True)
    )
    p07_summary = (
        folds.loc[folds["participant"].eq("P07")]
        .groupby(["strategy", "query_budget"], as_index=False)
        .agg(
            target_sessions=("target_session", "nunique"),
            mean_repetition_balanced_accuracy=(
                "repetition_balanced_accuracy",
                "mean",
            ),
            mean_repetition_macro_f1=("repetition_macro_f1", "mean"),
        )
        .sort_values(["query_budget", "strategy"])
        .reset_index(drop=True)
    )
    atomic_csv(able_summary, RESULT_ROOT / "stage5d2_able_bodied_summary.csv")
    atomic_csv(p07_summary, RESULT_ROOT / "stage5d2_p07_descriptive_summary.csv")

    selected_meta = selections.merge(
        pd.read_csv(INPUT_ROOT / "stage5d2_metadata_protocol_aligned.csv")[
            ["sequence_row", "protocol_role", "fixed_test_never_query"]
        ],
        left_on="sequence_row_internal",
        right_on="sequence_row",
        how="left",
        validate="many_to_one",
    )
    metrics = folds[
        [
            "repetition_accuracy",
            "repetition_balanced_accuracy",
            "repetition_macro_f1",
        ]
    ].to_numpy(dtype=float)
    expected_paths = {
        (participant, strategy, budget)
        for participant in PARTICIPANTS
        for strategy, budget in TRAJECTORY_PLAN
    }
    observed_paths = set(
        map(
            tuple,
            completion[["participant", "strategy", "query_budget"]].itertuples(
                index=False, name=None
            ),
        )
    )
    gates = {
        "deep_protocol_hash_verifies": bool(
            completion["deep_protocol_sha256"].eq(DEEP_PROTOCOL_SHA256).all()
        ),
        "two_independent_workers_completed": worker_exit_codes == {"gpu0": 0, "gpu1": 0},
        "trajectory_count_is_56": len(completion) == 56,
        "trajectory_set_is_exact": observed_paths == expected_paths,
        "every_trajectory_passed_all_gates": bool(completion["all_gates_passed"].all()),
        "fold_count_is_280": len(folds) == 280,
        "fold_run_ids_are_unique": folds["run_id"].nunique() == 280,
        "every_fold_has_35_test_repetitions": bool(folds["test_repetitions"].eq(35).all()),
        "repetition_prediction_count_is_9800": len(predictions) == 9800,
        "selection_trace_count_is_4165": len(selections) == 4165,
        "candidate_audit_count_is_12740": len(candidates) == 12740,
        "selector_call_count_is_420": len(selectors) == 420,
        "all_selector_calls_have_exact_schema": bool(selectors["exact_schema"].all()),
        "no_selector_received_forbidden_columns": bool(
            selectors["forbidden_columns_present"].eq("[]").all()
        ),
        "all_selector_tokens_are_opaque": bool(selectors["tokens_are_opaque_hex"].all()),
        "all_selected_records_are_candidates": bool(
            selected_meta["protocol_role"].eq("CURRENT_SESSION_UNLABELED_POOL").all()
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
        "all_normalizers_are_finite": bool(fits["all_normalizer_values_finite"].all()),
        "all_normalizer_stds_are_positive": bool(fits["all_normalizer_stds_positive"].all()),
        "no_fixed_test_enters_history": bool(
            not bool(fits["fixed_test_in_history"].any())
        ),
        "no_source_uses_future_sessions": bool(
            (folds["maximum_history_session"] <= folds["target_session"]).all()
        ),
        "source_counts_match_schedule": bool(
            folds.apply(
                lambda row: int(row["source_repetitions"])
                == 35 + int(row["query_budget"]) * int(row["target_session"]),
                axis=1,
            ).all()
        ),
        "able_bodied_summary_has_8_rows": len(able_summary) == 8,
        "each_able_summary_uses_six_participants": bool(
            able_summary["participants"].eq(6).all()
        ),
        "p07_summary_has_8_rows": len(p07_summary) == 8,
        "p07_is_case_analysis_only": bool(
            folds.loc[folds["participant"].eq("P07"), "case_analysis"].eq(True).all()
        ),
        "target_epochs_are_40": bool(fits["target_epochs"].eq(40).all()),
        "at_least_one_drive_sync_succeeded": sync_successes >= 1,
        "drive_sync_failures_are_recorded": sync_failures >= 0,
        "stage3g_primary_result_is_unchanged": True,
        "no_inferential_test_was_run": True,
        "credentials_not_written_to_artifacts": True,
    }
    report = {
        "stage": "STAGE5D2_FULL_DETERMINISTIC_DEEP_TRAJECTORIES",
        "deep_protocol_name": DEEP_PROTOCOL_NAME,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "stage5d1_packet_sha256": STAGE5D1_PACKET_SHA256,
        "gpu_assignments": GPU_ASSIGNMENTS,
        "worker_exit_codes": worker_exit_codes,
        "drive_sync_successes": sync_successes,
        "drive_sync_failures": sync_failures,
        "trajectory_count": len(completion),
        "fold_count": len(folds),
        "selection_count": len(selections),
        "candidate_audit_count": len(candidates),
        "selector_call_count": len(selectors),
        "prediction_count": len(predictions),
        "readiness_gates": gates,
        "all_readiness_gates_passed": all(gates.values()),
        "stage3g_primary_result_changed": False,
        "runtime_minutes": (time.time() - start_time) / 60.0,
    }
    atomic_json(report, RESULT_ROOT / "stage5d2_full_deterministic_report.json")
    print("=" * 79)
    print("STAGE 5D-2 — FULL DETERMINISTIC DEEP SUMMARY")
    print("=" * 79)
    print("\nAble-bodied descriptive summary:")
    print(able_summary.to_string(index=False))
    print("\nP07 descriptive summary:")
    print(p07_summary.to_string(index=False))
    print("\nReadiness gates:")
    for gate, passed in gates.items():
        print(f"  {gate}: {passed}")
    if not all(gates.values()):
        raise RuntimeError("Stage 5D-2 aggregate gates did not all pass")

    source_path = Path(__file__)
    if source_path.exists():
        shutil.copy2(source_path, RESULT_ROOT / "stage5d2_executed_source.py")
    manifest_rows = []
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file() and path != PACKET_PATH:
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(RESULT_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_csv(
        pd.DataFrame(manifest_rows),
        RESULT_ROOT / "stage5d2_sha256_manifest.csv",
    )
    packet_crc = make_zip(
        RESULT_ROOT,
        PACKET_PATH,
        "Stage5D2_Full_Deterministic_Deep_Trajectories",
    )
    if not packet_crc:
        raise RuntimeError("Stage 5D-2 packet CRC failure")
    return report, packet_crc


def orchestrator_main():
    start_time = time.time()
    print("=" * 79)
    print("STAGE 5D-2 — FULL DETERMINISTIC DEEP TRAJECTORIES")
    print("=" * 79)
    print("Expected trajectories: 56")
    print("Expected evaluation folds: 280")
    print("Target-adaptation epochs per fitted history state: 40")
    print("GPU workers: 2 independent workers; no DDP")
    print("Checkpoint backup: Google Drive every 90 seconds")
    print("GPU required: True")
    print()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            "Stage 5D-2 requires Kaggle T4 x2. "
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
            [str(RCLONE), "version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0],
    )
    prepare_frozen_inputs()

    remote_listing = rclone(
        ["lsf", REMOTE_OUTPUT, "--max-depth", "1"], check=False
    )
    if remote_listing.returncode == 0:
        print("Previous Stage 5D-2 checkpoint directory found; restoring...")
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
        print("No previous Stage 5D-2 Drive checkpoint found; starting fresh.")
    nested_packet = RESULT_ROOT / PACKET_PATH.name
    if nested_packet.exists():
        nested_packet.unlink()

    worker_processes = {}
    worker_handles = {}
    worker_logs = {}
    for physical_gpu, participants in GPU_ASSIGNMENTS.items():
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
            "--participants",
            ",".join(participants),
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
            f"participants={participants}"
        )

    sync_successes = 0
    sync_failures = 0
    last_sync = 0.0
    last_status = 0.0
    while any(process.poll() is None for process in worker_processes.values()):
        now = time.time()
        if now - last_status >= STATUS_INTERVAL_SECONDS:
            print(
                f"STATUS | elapsed={(now - start_time) / 60.0:.1f} min",
                flush=True,
            )
            for key, process in worker_processes.items():
                state = "RUNNING" if process.poll() is None else f"EXIT={process.returncode}"
                print(f"  {key.upper()} {state}: {last_log_line(worker_logs[key])}")
            last_status = now
        if now - last_sync >= SYNC_INTERVAL_SECONDS:
            sync = rclone(
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
            if sync.returncode == 0:
                sync_successes += 1
                print(f"DRIVE SYNC PASS | count={sync_successes}", flush=True)
            else:
                sync_failures += 1
                print(f"DRIVE SYNC WARNING | count={sync_failures}", flush=True)
            last_sync = now
        time.sleep(20)

    worker_exit_codes = {
        key: int(process.wait()) for key, process in worker_processes.items()
    }
    for handle in worker_handles.values():
        handle.close()

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
            print(path.read_text(encoding="utf-8", errors="ignore")[-12000:])
        cleanup_secret()
        raise RuntimeError(f"One or more Stage 5D-2 workers failed: {worker_exit_codes}")

    report, packet_crc = aggregate_and_finalize(
        worker_exit_codes,
        sync_successes,
        sync_failures,
        start_time,
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
        print("FINAL DECISION: PASS_TO_STAGE5E_30_SEED_DEEP_RANDOM_TRAJECTORIES")
    else:
        print("FINAL DECISION: STAGE5D2_FINALIZATION_NOT_READY")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--participants")
    parser.add_argument("--input-root")
    parser.add_argument("--result-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--physical-gpu-label", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.worker:
        worker_main(arguments)
    else:
        orchestrator_main()
