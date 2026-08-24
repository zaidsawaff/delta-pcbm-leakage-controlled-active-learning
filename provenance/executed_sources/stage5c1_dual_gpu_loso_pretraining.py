import base64
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================================
# STAGE 5C-1 — DUAL-GPU LOSO TRANSFER PRETRAINING
# ============================================================================

PROTOCOL_NAME = "DELTA_MASK_AWARE_RMS_TCN_TRANSFER_v1"
PROTOCOL_SHA256 = (
    "abe15812c1a52b0f4e917b5b6ad39b0df"
    "de50e5bb2d58dfcc35b3cacb22e3bd2"
)

PARTICIPANTS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07"]
ABLE_BODIED = ["P01", "P02", "P03", "P04", "P05", "P06"]

STAGE5A4B_PACKET = Path(
    "/kaggle/working/stage5a4b_deep_protocol_lock_packet.zip"
)
STAGE5B_ROOT = Path(
    "/kaggle/working/STAGE5B_DEEP_SEQUENCE_ASSEMBLY"
)
RESULT_ROOT = Path(
    "/kaggle/working/DELTA_STAGE5_DEEP_RESULTS/Stage5C_LOSO_Pretraining"
)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

PACKET_PATH = Path(
    "/kaggle/working/stage5c1_dual_gpu_loso_pretraining_packet.zip"
)

RCLONE_BINARY = Path("/kaggle/working/_stage5_tools/rclone")
DRIVE_REMOTE_DIRECTORY = (
    "gdrive_stage5:DELTA_Q1_Stage5_DeepLearning_Backup/"
    "Deep_Training/Stage5C_LOSO_Pretraining"
)

MAX_EPOCHS = 100
MIN_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 15
BATCH_SIZE = 64
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-3
LABEL_SMOOTHING = 0.05
ETA_MIN = 1.0e-6
AMP_INITIAL_SCALE = 1024.0
AMP_GROWTH_INTERVAL = 10000
SYNC_INTERVAL_SECONDS = 90

START_TIME = time.time()

print("=" * 79)
print("STAGE 5C-1 — DUAL-GPU LOSO TRANSFER PRETRAINING")
print("=" * 79)
print("Protocol:", PROTOCOL_NAME)
print("Protocol SHA-256:", PROTOCOL_SHA256)
print("Expected LOSO folds: 7")
print("Maximum epochs per fold:", MAX_EPOCHS)
print("Minimum epochs before early stopping:", MIN_EPOCHS)
print("Early-stopping patience:", EARLY_STOPPING_PATIENCE)
print("GPU workers: 2 independent workers; no DDP")
print("AMP initial scale:", AMP_INITIAL_SCALE)
print("AMP growth interval:", AMP_GROWTH_INTERVAL)
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
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def atomic_write_json(payload, destination):
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


def run_rclone(arguments, config_path, check=True):
    return subprocess.run(
        [
            str(RCLONE_BINARY),
            "--config",
            str(config_path),
            *arguments,
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def make_temporary_rclone_config():
    from kaggle_secrets import UserSecretsClient

    encoded = UserSecretsClient().get_secret("RCLONE_CONFIG_B64")
    if not encoded:
        raise RuntimeError("Kaggle secret RCLONE_CONFIG_B64 is unavailable.")
    decoded = base64.b64decode(encoded, validate=True)
    del encoded
    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="stage5c_rclone_",
        suffix=".conf",
        dir="/tmp",
        delete=False,
    )
    temporary.write(decoded)
    temporary.close()
    del decoded
    path = Path(temporary.name)
    os.chmod(path, 0o600)
    return path


# ----------------------------------------------------------------------------
# 1. INPUT AND GPU GATES
# ----------------------------------------------------------------------------

required_stage5b = {
    "features": STAGE5B_ROOT / "stage5b_rms_repetition_sequences.npy",
    "mask": STAGE5B_ROOT / "stage5b_main_valid_repetition_sequences.npy",
    "metadata": STAGE5B_ROOT / "stage5b_repetition_metadata.csv",
    "model": STAGE5B_ROOT / "stage5b_mask_aware_rms_tcn.py",
    "report": STAGE5B_ROOT / "stage5b_sequence_assembly_report.json",
}

if not STAGE5A4B_PACKET.exists():
    raise FileNotFoundError(f"Missing {STAGE5A4B_PACKET}")
for name, path in required_stage5b.items():
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing Stage 5B {name}: {path}")

if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError(
        "Stage 5C-1 requires exactly two visible Kaggle T4 GPUs. "
        f"CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}"
    )

gpu_names = [
    torch.cuda.get_device_properties(index).name
    for index in range(torch.cuda.device_count())
]
if not all("T4" in name for name in gpu_names):
    raise RuntimeError(f"Expected two Tesla T4 GPUs; observed {gpu_names}")
if not RCLONE_BINARY.exists():
    raise FileNotFoundError(
        f"Missing {RCLONE_BINARY}; run Stage 5C-0B before this stage."
    )

with open(required_stage5b["report"], "r", encoding="utf-8") as handle:
    stage5b_report = json.load(handle)
if not stage5b_report.get("all_readiness_gates_passed", False):
    raise RuntimeError("The restored Stage 5B report did not pass all gates.")

with zipfile.ZipFile(STAGE5A4B_PACKET, "r") as archive:
    if archive.testzip() is not None:
        raise RuntimeError("Stage 5A-4B packet failed CRC verification.")
    text = "\n".join(
        archive.read(member).decode("utf-8", errors="ignore")
        for member in archive.namelist()
        if member.lower().endswith((".json", ".csv", ".txt"))
    )
    if PROTOCOL_SHA256 not in text:
        raise RuntimeError("Deep protocol hash verification failed.")
    seed_members = [
        member
        for member in archive.namelist()
        if Path(member).name == "stage5a4b_training_seed_schedule.csv"
    ]
    if len(seed_members) != 1:
        raise RuntimeError("Locked training seed schedule did not resolve once.")
    with archive.open(seed_members[0], "r") as handle:
        seed_schedule = pd.read_csv(handle)

primary_seeds = seed_schedule.loc[
    seed_schedule["seed_role"].eq("PRIMARY_DETERMINISTIC"),
    ["participant", "seed"],
].copy()
primary_seeds["participant"] = primary_seeds["participant"].astype(str)
primary_seeds["seed"] = pd.to_numeric(primary_seeds["seed"], errors="raise").astype(np.int64)
if sorted(primary_seeds["participant"].tolist()) != PARTICIPANTS:
    raise RuntimeError("Primary deterministic seed schedule is incomplete.")
if primary_seeds["seed"].nunique() != 7:
    raise RuntimeError("Primary deterministic seeds are not unique.")
seed_by_target = dict(zip(primary_seeds["participant"], primary_seeds["seed"]))

metadata = pd.read_csv(required_stage5b["metadata"])
metadata["participant"] = metadata["participant"].astype(str)
metadata["session"] = pd.to_numeric(metadata["session"], errors="raise").astype(int)
metadata["label"] = pd.to_numeric(metadata["label"], errors="raise").astype(int)
if len(metadata) != 2940:
    raise RuntimeError("Unexpected Stage 5B metadata row count.")

fold_plan_rows = []
for target in PARTICIPANTS:
    source_participants = (
        [participant for participant in ABLE_BODIED if participant != target]
        if target in ABLE_BODIED
        else ABLE_BODIED.copy()
    )
    train = metadata["participant"].isin(source_participants) & metadata["session"].isin([0, 1, 2, 3, 4])
    validation = metadata["participant"].isin(source_participants) & metadata["session"].eq(5)
    expected_train = 1750 if target in ABLE_BODIED else 2100
    expected_validation = 350 if target in ABLE_BODIED else 420
    if int(train.sum()) != expected_train or int(validation.sum()) != expected_validation:
        raise RuntimeError(f"Unexpected source counts for {target}.")
    if train.any() and metadata.loc[train, "participant"].eq(target).any():
        raise RuntimeError(f"Target participant entered {target} training data.")
    if validation.any() and metadata.loc[validation, "participant"].eq(target).any():
        raise RuntimeError(f"Target participant entered {target} validation data.")
    fold_plan_rows.append(
        {
            "target_participant": target,
            "training_seed": int(seed_by_target[target]),
            "source_participants": "|".join(source_participants),
            "source_train_sessions": "0|1|2|3|4",
            "source_validation_session": 5,
            "train_repetitions": int(train.sum()),
            "validation_repetitions": int(validation.sum()),
            "target_data_used": False,
            "gpu_assignment": 0 if target in ["P01", "P03", "P05", "P07"] else 1,
        }
    )

fold_plan = pd.DataFrame(fold_plan_rows)
fold_plan_path = RESULT_ROOT / "stage5c1_loso_fold_plan.csv"
fold_plan.to_csv(fold_plan_path, index=False)

# Locked-protocol-preserving numerical implementation patch. The diagnostic
# demonstrated finite inputs/logits and isolated the overflow to the default
# GradScaler scale of 65536. A scale of 1024 was finite for both independently
# tested LOSO folds. No scientific data, model, loss, optimizer, or endpoint is
# changed by this patch.
amp_patch_record = {
    "amendment_role": "NUMERICAL_IMPLEMENTATION_PATCH_ONLY",
    "parent_protocol_name": PROTOCOL_NAME,
    "parent_protocol_sha256": PROTOCOL_SHA256,
    "diagnostic_stage": "STAGE5C1A_AMP_NUMERICAL_STABILITY_DIAGNOSTIC",
    "failure_reproduced_at_scale": 65536.0,
    "affected_parameter_in_both_tested_folds": "classifier.weight",
    "tested_targets": ["P01", "P02"],
    "maximum_absolute_transformed_input": {
        "P01": 9.838401794433594,
        "P02": 8.5670166015625,
    },
    "values_exceeding_fp16_maximum": 0,
    "finite_tested_scales_for_both_folds": [1024.0, 256.0, 32.0],
    "selected_amp_initial_scale": AMP_INITIAL_SCALE,
    "selected_amp_growth_interval": AMP_GROWTH_INTERVAL,
    "gradient_clipping_added": False,
    "data_changed": False,
    "normalization_changed": False,
    "model_architecture_changed": False,
    "loss_changed": False,
    "optimizer_changed": False,
    "endpoint_changed": False,
    "scientific_protocol_changed": False,
}
amp_patch_record["amendment_sha256"] = hashlib.sha256(
    json.dumps(
        amp_patch_record,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
atomic_write_json(
    amp_patch_record,
    RESULT_ROOT / "stage5c1a_amp_numerical_patch.json",
)

print("GPU 0:", gpu_names[0], "targets=P01,P03,P05,P07")
print("GPU 1:", gpu_names[1], "targets=P02,P04,P06")
print("Locked primary seeds:")
print(primary_seeds.to_string(index=False))
print()


# ----------------------------------------------------------------------------
# 2. RESTORE ANY PREVIOUS DRIVE CHECKPOINTS
# ----------------------------------------------------------------------------

rclone_config = make_temporary_rclone_config()
restore_result = run_rclone(
    ["lsf", DRIVE_REMOTE_DIRECTORY, "--max-depth", "1"],
    rclone_config,
    check=False,
)
if restore_result.returncode == 0:
    print("Previous Stage 5C checkpoint directory found; restoring...")
    run_rclone(
        [
            "copy",
            DRIVE_REMOTE_DIRECTORY,
            str(RESULT_ROOT),
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--timeout",
            "5m",
        ],
        rclone_config,
        check=True,
    )
else:
    print("No previous Stage 5C Drive checkpoint found; starting fresh.")

# The finalized packet is stored at the remote directory root. It is not a
# training input and must never become recursively embedded in a new packet.
restored_nested_packet = RESULT_ROOT / PACKET_PATH.name
if restored_nested_packet.exists():
    restored_nested_packet.unlink()

# Preserve the diagnosed pre-patch failure as audit evidence while ensuring
# the corrected execution starts with clean worker logs.
for physical_gpu in [0, 1]:
    prior_log = RESULT_ROOT / f"gpu{physical_gpu}_worker.log"
    archived_log = RESULT_ROOT / f"gpu{physical_gpu}_prepatch_failure.log"
    if prior_log.exists():
        prior_text = prior_log.read_text(encoding="utf-8", errors="ignore")
        if "Non-finite gradient" in prior_text:
            if archived_log.exists():
                archived_text = archived_log.read_text(
                    encoding="utf-8", errors="ignore"
                )
                archived_log.write_text(
                    archived_text + "\n" + prior_text,
                    encoding="utf-8",
                )
                prior_log.unlink()
            else:
                os.replace(prior_log, archived_log)


# ----------------------------------------------------------------------------
# 3. WRITE THE ISOLATED CUDA WORKER
# ----------------------------------------------------------------------------

worker_source = r'''
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROTOCOL_NAME = "DELTA_MASK_AWARE_RMS_TCN_TRANSFER_v1"
PROTOCOL_SHA256 = "abe15812c1a52b0f4e917b5b6ad39b0dfde50e5bb2d58dfcc35b3cacb22e3bd2"
PARTICIPANTS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07"]
ABLE_BODIED = ["P01", "P02", "P03", "P04", "P05", "P06"]
CHANNELS = 64
MAX_EPOCHS = 100
MIN_EPOCHS = 20
PATIENCE = 15
BATCH_SIZE = 64
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-3
LABEL_SMOOTHING = 0.05
ETA_MIN = 1.0e-6
AMP_INITIAL_SCALE = 1024.0
AMP_GROWTH_INTERVAL = 10000


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_json(payload, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, destination)


def atomic_csv(dataframe, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
            raise ValueError(f"No valid source-training values for channel {channel}")
        means[channel] = values.mean()
        stds[channel] = values.std(ddof=0)
        if not np.isfinite(stds[channel]) or stds[channel] <= 0:
            raise ValueError(f"Invalid source-training std for channel {channel}")
    return means, stds, counts


def transform(raw, valid, means, stds):
    raw = np.asarray(raw, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    normalized = (np.log1p(raw) - means[None, None, :]) / stds[None, None, :]
    normalized[~valid] = 0.0
    combined = np.concatenate(
        [normalized.astype(np.float32), valid.astype(np.float32)],
        axis=2,
    )
    combined = np.transpose(combined, (0, 2, 1)).copy()
    if not np.isfinite(combined).all():
        raise ValueError("Non-finite transformed values")
    return combined


def classification_metrics(y_true, y_pred, classes=7):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
    support = matrix.sum(axis=1)
    recall = np.divide(
        np.diag(matrix), support,
        out=np.zeros(classes, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        np.diag(matrix), matrix.sum(axis=0),
        out=np.zeros(classes, dtype=np.float64),
        where=matrix.sum(axis=0) > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    truths = []
    predictions = []
    probabilities = []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(inputs)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)
        predicted = torch.argmax(probs, dim=1)
        total_loss += float(loss.item()) * len(labels)
        total_examples += len(labels)
        truths.append(labels.cpu().numpy())
        predictions.append(predicted.cpu().numpy())
        probabilities.append(probs.cpu().numpy())
    y_true = np.concatenate(truths)
    y_pred = np.concatenate(predictions)
    probability = np.concatenate(probabilities)
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = float(total_loss / total_examples)
    return metrics, y_true, y_pred, probability


def checkpoint_payload(
    target, seed, epoch, model, optimizer, scheduler, scaler,
    best_metric, best_epoch, no_improvement, means, stds, counts,
    train_indices, validation_indices,
):
    return {
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": PROTOCOL_SHA256,
        "target_participant": target,
        "training_seed": int(seed),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_validation_balanced_accuracy": float(best_metric),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(no_improvement),
        "normalizer_means": means,
        "normalizer_stds": stds,
        "normalizer_counts": counts,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "scientific_role": "LOSO_SOURCE_PRETRAINING",
        "target_data_used": False,
    }


def run_fold(target, seed, data_root, result_root, physical_gpu_label):
    fold_start = time.time()
    fold_directory = result_root / target
    fold_directory.mkdir(parents=True, exist_ok=True)
    complete_path = fold_directory / "complete.json"
    best_path = fold_directory / "best.pt"
    last_path = fold_directory / "last.pt"
    metrics_path = fold_directory / "epoch_metrics.csv"
    predictions_path = fold_directory / "best_validation_predictions.csv"
    normalizer_path = fold_directory / "source_train_normalizer.npz"

    if complete_path.exists() and best_path.exists():
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                complete.get("complete") is True
                and complete.get("protocol_sha256") == PROTOCOL_SHA256
                and complete.get("best_checkpoint_sha256") == sha256_file(best_path)
            ):
                print(f"SKIP COMPLETE | {target} | best epoch {complete['best_epoch']}", flush=True)
                return
        except Exception:
            pass

    set_seed(seed)
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_properties(0).name
    if "T4" not in gpu_name:
        raise RuntimeError(f"Worker expected T4, observed {gpu_name}")

    sys.path.insert(0, str(data_root))
    from stage5b_mask_aware_rms_tcn import MaskAwareRMSTCN

    features = np.load(
        data_root / "stage5b_rms_repetition_sequences.npy",
        mmap_mode="r", allow_pickle=False,
    )
    valid_mask = np.load(
        data_root / "stage5b_main_valid_repetition_sequences.npy",
        mmap_mode="r", allow_pickle=False,
    )
    metadata = pd.read_csv(data_root / "stage5b_repetition_metadata.csv")
    metadata["participant"] = metadata["participant"].astype(str)
    metadata["session"] = pd.to_numeric(metadata["session"], errors="raise").astype(int)
    metadata["label"] = pd.to_numeric(metadata["label"], errors="raise").astype(int)

    source_participants = (
        [participant for participant in ABLE_BODIED if participant != target]
        if target in ABLE_BODIED else ABLE_BODIED.copy()
    )
    train_boolean = metadata["participant"].isin(source_participants) & metadata["session"].isin([0, 1, 2, 3, 4])
    validation_boolean = metadata["participant"].isin(source_participants) & metadata["session"].eq(5)
    train_indices = np.flatnonzero(train_boolean.to_numpy()).astype(np.int64)
    validation_indices = np.flatnonzero(validation_boolean.to_numpy()).astype(np.int64)
    expected_train = 1750 if target in ABLE_BODIED else 2100
    expected_validation = 350 if target in ABLE_BODIED else 420
    if len(train_indices) != expected_train or len(validation_indices) != expected_validation:
        raise RuntimeError(f"Source counts failed for {target}")
    if target in set(metadata.iloc[train_indices]["participant"]):
        raise RuntimeError(f"Target leakage into training for {target}")
    if target in set(metadata.iloc[validation_indices]["participant"]):
        raise RuntimeError(f"Target leakage into validation for {target}")

    means, stds, counts = fit_normalizer(features[train_indices], valid_mask[train_indices])
    if not (np.isfinite(means).all() and np.isfinite(stds).all() and (stds > 0).all() and (counts > 0).all()):
        raise RuntimeError(f"Normalizer audit failed for {target}")
    np.savez(
        normalizer_path,
        means=means,
        stds=stds,
        counts=counts,
        source_participants=np.asarray(source_participants),
        source_train_sessions=np.asarray([0, 1, 2, 3, 4]),
        source_validation_session=np.asarray([5]),
        target_participant=np.asarray([target]),
    )

    train_x = transform(features[train_indices], valid_mask[train_indices], means, stds)
    validation_x = transform(features[validation_indices], valid_mask[validation_indices], means, stds)
    train_y = metadata.iloc[train_indices]["label"].to_numpy(dtype=np.int64)
    validation_y = metadata.iloc[validation_indices]["label"].to_numpy(dtype=np.int64)
    if sorted(np.unique(train_y).tolist()) != list(range(7)):
        raise RuntimeError("Training classes are incomplete")
    if sorted(np.unique(validation_y).tolist()) != list(range(7)):
        raise RuntimeError("Validation classes are incomplete")

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    validation_dataset = TensorDataset(torch.from_numpy(validation_x), torch.from_numpy(validation_y))
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    model = MaskAwareRMSTCN().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != 118536:
        raise RuntimeError(f"Model parameter drift: {parameter_count}")
    if any(isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for module in model.modules()):
        raise RuntimeError("BatchNorm is prohibited")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=ETA_MIN,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=AMP_INITIAL_SCALE,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=AMP_GROWTH_INTERVAL,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    start_epoch = 1
    best_metric = -np.inf
    best_epoch = 0
    no_improvement = 0
    history = []
    resumed = False

    if metrics_path.exists():
        try:
            history = pd.read_csv(metrics_path).to_dict("records")
        except Exception:
            history = []

    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol_sha256") != PROTOCOL_SHA256:
            raise RuntimeError(f"Checkpoint protocol mismatch for {target}")
        if checkpoint.get("target_participant") != target or int(checkpoint.get("training_seed")) != int(seed):
            raise RuntimeError(f"Checkpoint identity mismatch for {target}")
        if not (
            np.array_equal(checkpoint["train_indices"], train_indices)
            and np.array_equal(checkpoint["validation_indices"], validation_indices)
            and np.array_equal(checkpoint["normalizer_means"], means)
            and np.array_equal(checkpoint["normalizer_stds"], stds)
            and np.array_equal(checkpoint["normalizer_counts"], counts)
        ):
            raise RuntimeError(f"Checkpoint data/normalizer mismatch for {target}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        best_metric = float(checkpoint["best_validation_balanced_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        no_improvement = int(checkpoint["epochs_without_improvement"])
        start_epoch = int(checkpoint["epoch"]) + 1
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())
        history = [row for row in history if int(row["epoch"]) < start_epoch]
        resumed = True
        print(f"RESUME | {target} | epoch {start_epoch}", flush=True)

    stopped_early = False
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = time.time()
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(int(seed) + int(epoch))
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            generator=epoch_generator, num_workers=0,
            pin_memory=True, drop_last=False,
        )
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_examples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss for {target} epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"Non-finite gradient for {target} epoch {epoch}")
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += float(loss.item()) * len(labels)
            total_train_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total_train_examples += len(labels)

        validation_metrics, _, _, _ = evaluate(model, validation_loader, criterion, device)
        current_metric = float(validation_metrics["balanced_accuracy"])
        improved = current_metric > best_metric + 1.0e-12
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        row = {
            "target_participant": target,
            "epoch": epoch,
            "training_seed": int(seed),
            "learning_rate": current_lr,
            "train_loss": float(total_train_loss / total_train_examples),
            "train_accuracy": float(total_train_correct / total_train_examples),
            "validation_loss": float(validation_metrics["loss"]),
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_balanced_accuracy": current_metric,
            "validation_macro_f1": float(validation_metrics["macro_f1"]),
            "is_new_best": bool(improved),
            "best_validation_balanced_accuracy": float(best_metric),
            "best_epoch": int(best_epoch),
            "epochs_without_improvement": int(no_improvement),
            "epoch_seconds": float(time.time() - epoch_start),
            "gpu_physical_assignment": int(physical_gpu_label),
            "gpu_visible_name": gpu_name,
        }
        history.append(row)
        atomic_csv(pd.DataFrame(history), metrics_path)
        payload = checkpoint_payload(
            target, seed, epoch, model, optimizer, scheduler, scaler,
            best_metric, best_epoch, no_improvement, means, stds, counts,
            train_indices, validation_indices,
        )
        if improved:
            atomic_torch_save(payload, best_path)
        atomic_torch_save(payload, last_path)
        if epoch % 5 == 0:
            atomic_json(
                {
                    "target_participant": target,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_balanced_accuracy": best_metric,
                    "checkpoint_ready_for_drive_sync": True,
                },
                fold_directory / f"sync_epoch_{epoch:03d}.json",
            )
        print(
            f"EPOCH | {target} | {epoch:03d}/{MAX_EPOCHS} | "
            f"train_loss={row['train_loss']:.5f} | "
            f"val_BA={current_metric:.6f} | best={best_metric:.6f}@{best_epoch} | "
            f"wait={no_improvement}/{PATIENCE}",
            flush=True,
        )
        if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
            stopped_early = True
            print(f"EARLY STOP | {target} | epoch {epoch}", flush=True)
            break

    if not best_path.exists():
        raise RuntimeError(f"No best checkpoint was created for {target}")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_metrics, y_true, y_pred, probabilities = evaluate(model, validation_loader, criterion, device)
    prediction_frame = metadata.iloc[validation_indices].copy().reset_index(drop=True)
    prediction_frame.insert(0, "target_pretraining_fold", target)
    prediction_frame["true_label"] = y_true
    prediction_frame["predicted_label"] = y_pred
    for label in range(7):
        prediction_frame[f"probability_label_{label}"] = probabilities[:, label]
    atomic_csv(prediction_frame, predictions_path)

    completed_epochs = int(pd.DataFrame(history)["epoch"].max())
    complete_payload = {
        "complete": True,
        "protocol_name": PROTOCOL_NAME,
        "protocol_sha256": PROTOCOL_SHA256,
        "target_participant": target,
        "case_analysis": target == "P07",
        "training_seed": int(seed),
        "physical_gpu_assignment": int(physical_gpu_label),
        "gpu_visible_name": gpu_name,
        "source_participants": source_participants,
        "source_train_sessions": [0, 1, 2, 3, 4],
        "source_validation_session": 5,
        "train_repetitions": int(len(train_indices)),
        "validation_repetitions": int(len(validation_indices)),
        "target_data_used": False,
        "parameter_count": int(parameter_count),
        "batch_normalization_modules": 0,
        "amp_fp16_used": True,
        "amp_initial_scale": float(AMP_INITIAL_SCALE),
        "amp_growth_interval": int(AMP_GROWTH_INTERVAL),
        "resumed_from_checkpoint": bool(resumed),
        "completed_epochs": completed_epochs,
        "stopped_early": bool(stopped_early),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_loss": float(final_metrics["loss"]),
        "best_validation_accuracy": float(final_metrics["accuracy"]),
        "best_validation_balanced_accuracy": float(final_metrics["balanced_accuracy"]),
        "best_validation_macro_f1": float(final_metrics["macro_f1"]),
        "best_validation_confusion_matrix": final_metrics["confusion_matrix"],
        "all_normalizer_values_finite": bool(np.isfinite(means).all() and np.isfinite(stds).all()),
        "all_normalizer_stds_positive": bool((stds > 0).all()),
        "all_normalizer_counts_positive": bool((counts > 0).all()),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "runtime_minutes": float((time.time() - fold_start) / 60.0),
    }
    atomic_json(complete_payload, complete_path)
    print(
        f"COMPLETE | {target} | epochs={completed_epochs} | "
        f"best={complete_payload['best_validation_balanced_accuracy']:.6f}@{complete_payload['best_epoch']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--seed-plan", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--physical-gpu-label", required=True, type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Isolated worker expected one visible CUDA GPU; observed {torch.cuda.device_count()}"
        )
    seed_plan = pd.read_csv(args.seed_plan)
    seed_by_target = dict(zip(seed_plan["target_participant"], seed_plan["training_seed"]))
    targets = [item for item in args.targets.split(",") if item]
    print(
        f"WORKER START | physical_gpu={args.physical_gpu_label} | "
        f"visible_gpu={torch.cuda.get_device_properties(0).name} | targets={targets}",
        flush=True,
    )
    for target in targets:
        run_fold(
            target=target,
            seed=int(seed_by_target[target]),
            data_root=Path(args.data_root),
            result_root=Path(args.result_root),
            physical_gpu_label=args.physical_gpu_label,
        )
    print(f"WORKER COMPLETE | physical_gpu={args.physical_gpu_label}", flush=True)


if __name__ == "__main__":
    main()
'''

worker_path = RESULT_ROOT / "stage5c1_cuda_worker.py"
worker_path.write_text(worker_source, encoding="utf-8")

# Compile before consuming GPU time.
compile_result = subprocess.run(
    [sys.executable, "-m", "py_compile", str(worker_path)],
    capture_output=True,
    text=True,
)
if compile_result.returncode != 0:
    raise RuntimeError("Generated CUDA worker failed compilation:\n" + compile_result.stderr)


# ----------------------------------------------------------------------------
# 4. LAUNCH TWO INDEPENDENT GPU WORKERS
# ----------------------------------------------------------------------------

worker_assignments = [
    (0, ["P01", "P03", "P05", "P07"]),
    (1, ["P02", "P04", "P06"]),
]
process_records = []

for physical_gpu, targets in worker_assignments:
    log_path = RESULT_ROOT / f"gpu{physical_gpu}_worker.log"
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        str(worker_path),
        "--targets", ",".join(targets),
        "--seed-plan", str(fold_plan_path),
        "--data-root", str(STAGE5B_ROOT),
        "--result-root", str(RESULT_ROOT),
        "--physical-gpu-label", str(physical_gpu),
    ]
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    process_records.append(
        {
            "gpu": physical_gpu,
            "targets": targets,
            "process": process,
            "log_path": log_path,
            "log_handle": log_handle,
        }
    )
    print(f"Launched GPU {physical_gpu} worker PID={process.pid} targets={targets}")


def last_nonempty_line(path):
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in reversed(lines):
            if line.strip():
                return line.strip()
    except Exception:
        pass
    return "NO LOG OUTPUT YET"


sync_successes = 0
sync_failures = 0
last_sync_time = 0.0
last_status_time = 0.0

while any(record["process"].poll() is None for record in process_records):
    now = time.time()
    if now - last_status_time >= 30:
        elapsed = (now - START_TIME) / 60.0
        statuses = []
        for record in process_records:
            status = "RUNNING" if record["process"].poll() is None else f"EXIT={record['process'].returncode}"
            statuses.append(f"GPU{record['gpu']} {status}: {last_nonempty_line(record['log_path'])}")
        print(f"STATUS | elapsed={elapsed:.1f} min")
        for status in statuses:
            print(" ", status)
        last_status_time = now

    if now - last_sync_time >= SYNC_INTERVAL_SECONDS:
        try:
            run_rclone(
                [
                    "copy",
                    str(RESULT_ROOT),
                    DRIVE_REMOTE_DIRECTORY,
                    "--retries", "3",
                    "--low-level-retries", "5",
                    "--timeout", "5m",
                    "--transfers", "4",
                    "--checkers", "8",
                ],
                rclone_config,
                check=True,
            )
            sync_successes += 1
            print(f"DRIVE SYNC PASS | count={sync_successes}")
        except Exception as error:
            sync_failures += 1
            print(f"DRIVE SYNC WARNING | {type(error).__name__}: {error}")
        last_sync_time = now
    time.sleep(15)

for record in process_records:
    record["log_handle"].close()

worker_exit_codes = {f"gpu{record['gpu']}": int(record["process"].returncode) for record in process_records}

# Always attempt a final checkpoint upload, including logs from a failed worker.
try:
    run_rclone(
        [
            "copy", str(RESULT_ROOT), DRIVE_REMOTE_DIRECTORY,
            "--retries", "5", "--low-level-retries", "10", "--timeout", "5m",
            "--transfers", "4", "--checkers", "8",
        ],
        rclone_config,
        check=True,
    )
    sync_successes += 1
except Exception as error:
    sync_failures += 1
    print(f"FINAL DRIVE SYNC WARNING | {type(error).__name__}: {error}")

if any(code != 0 for code in worker_exit_codes.values()):
    for record in process_records:
        print("\n", "=" * 30, record["log_path"].name, "=" * 30)
        lines = record["log_path"].read_text(encoding="utf-8", errors="ignore").splitlines()
        print("\n".join(lines[-40:]))
    if rclone_config.exists():
        rclone_config.unlink()
    raise RuntimeError(f"One or more GPU workers failed: {worker_exit_codes}")


# ----------------------------------------------------------------------------
# 5. FULL COMPLETION AND CHECKPOINT AUDIT
# ----------------------------------------------------------------------------

completion_rows = []
checkpoint_state_finite = True
checkpoint_identity_valid = True
normalizer_valid = True
best_hashes_valid = True

for target in PARTICIPANTS:
    fold_directory = RESULT_ROOT / target
    complete_path = fold_directory / "complete.json"
    best_path = fold_directory / "best.pt"
    last_path = fold_directory / "last.pt"
    metrics_path = fold_directory / "epoch_metrics.csv"
    predictions_path = fold_directory / "best_validation_predictions.csv"
    if not all(path.exists() and path.stat().st_size > 0 for path in [complete_path, best_path, last_path, metrics_path, predictions_path]):
        raise RuntimeError(f"Incomplete Stage 5C artifacts for {target}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    checkpoint_identity_valid &= bool(
        checkpoint.get("protocol_sha256") == PROTOCOL_SHA256
        and checkpoint.get("target_participant") == target
        and int(checkpoint.get("training_seed")) == int(seed_by_target[target])
        and checkpoint.get("target_data_used") is False
    )
    checkpoint_state_finite &= all(
        not torch.is_tensor(value) or torch.isfinite(value).all().item()
        for value in checkpoint["model_state_dict"].values()
    )
    means = np.asarray(checkpoint["normalizer_means"])
    stds = np.asarray(checkpoint["normalizer_stds"])
    counts = np.asarray(checkpoint["normalizer_counts"])
    normalizer_valid &= bool(
        np.isfinite(means).all() and np.isfinite(stds).all()
        and (stds > 0).all() and (counts > 0).all()
    )
    best_hashes_valid &= complete["best_checkpoint_sha256"] == sha256_file(best_path)
    completion_rows.append(complete)

completion_summary = pd.DataFrame(completion_rows).sort_values("target_participant").reset_index(drop=True)
summary_path = RESULT_ROOT / "stage5c1_loso_pretraining_summary.csv"
completion_summary.to_csv(summary_path, index=False)

readiness_gates = {
    "deep_protocol_hash_verifies": True,
    "two_tesla_t4_gpus_used": gpu_names == ["Tesla T4", "Tesla T4"],
    "two_independent_workers_completed": worker_exit_codes == {"gpu0": 0, "gpu1": 0},
    "loso_fold_count_is_7": len(completion_summary) == 7,
    "targets_are_exactly_p01_to_p07": completion_summary["target_participant"].tolist() == PARTICIPANTS,
    "locked_primary_seeds_used": set(completion_summary["training_seed"].astype(int)) == set(primary_seeds["seed"].astype(int)),
    "every_fold_has_a_best_checkpoint": all((RESULT_ROOT / target / "best.pt").exists() for target in PARTICIPANTS),
    "every_fold_has_a_last_checkpoint": all((RESULT_ROOT / target / "last.pt").exists() for target in PARTICIPANTS),
    "checkpoint_identities_are_valid": bool(checkpoint_identity_valid),
    "all_checkpoint_model_values_are_finite": bool(checkpoint_state_finite),
    "all_source_normalizers_are_valid": bool(normalizer_valid),
    "all_best_checkpoint_hashes_match": bool(best_hashes_valid),
    "target_data_is_never_used": bool(completion_summary["target_data_used"].eq(False).all()),
    "all_folds_use_amp_fp16": bool(completion_summary["amp_fp16_used"].eq(True).all()),
    "all_folds_use_diagnostically_safe_amp_scale": bool(
        completion_summary["amp_initial_scale"].eq(AMP_INITIAL_SCALE).all()
        and completion_summary["amp_growth_interval"].eq(AMP_GROWTH_INTERVAL).all()
    ),
    "amp_numerical_patch_is_documented": bool(
        (RESULT_ROOT / "stage5c1a_amp_numerical_patch.json").exists()
        and len(amp_patch_record["amendment_sha256"]) == 64
    ),
    "all_models_have_118536_parameters": bool(completion_summary["parameter_count"].eq(118536).all()),
    "all_models_have_no_batch_normalization": bool(completion_summary["batch_normalization_modules"].eq(0).all()),
    "all_validation_metrics_are_finite": bool(np.isfinite(completion_summary[["best_validation_loss", "best_validation_accuracy", "best_validation_balanced_accuracy", "best_validation_macro_f1"]].to_numpy(dtype=float)).all()),
    "all_validation_metrics_are_in_range": bool(completion_summary[["best_validation_accuracy", "best_validation_balanced_accuracy", "best_validation_macro_f1"]].apply(lambda column: column.between(0.0, 1.0).all()).all()),
    "p07_is_marked_case_analysis": bool(completion_summary.loc[completion_summary["target_participant"].eq("P07"), "case_analysis"].eq(True).all()),
    "at_least_one_drive_sync_succeeded": sync_successes >= 1,
    "credentials_not_written_to_artifacts": True,
}

report = {
    "stage": "STAGE5C1_DUAL_GPU_LOSO_PRETRAINING",
    "protocol_name": PROTOCOL_NAME,
    "protocol_sha256": PROTOCOL_SHA256,
    "gpu_names": gpu_names,
    "worker_assignments": {
        "gpu0": ["P01", "P03", "P05", "P07"],
        "gpu1": ["P02", "P04", "P06"],
    },
    "worker_exit_codes": worker_exit_codes,
    "maximum_epochs": MAX_EPOCHS,
    "minimum_epochs": MIN_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "batch_size": BATCH_SIZE,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
    "scheduler": "CosineAnnealingLR",
    "amp_fp16": True,
    "amp_initial_scale": AMP_INITIAL_SCALE,
    "amp_growth_interval": AMP_GROWTH_INTERVAL,
    "amp_numerical_patch_sha256": amp_patch_record["amendment_sha256"],
    "drive_sync_successes": sync_successes,
    "drive_sync_failures": sync_failures,
    "fold_count": len(completion_summary),
    "readiness_gates": readiness_gates,
    "all_readiness_gates_passed": bool(all(readiness_gates.values())),
    "runtime_minutes": float((time.time() - START_TIME) / 60.0),
    "credentials_displayed": False,
    "credentials_written_to_artifacts": False,
}

report_path = RESULT_ROOT / "stage5c1_loso_pretraining_report.json"
atomic_write_json(report, report_path)
if not report["all_readiness_gates_passed"]:
    failed = [gate for gate, passed in readiness_gates.items() if not passed]
    raise RuntimeError(f"Stage 5C-1 readiness failure: {failed}")


# ----------------------------------------------------------------------------
# 6. MANIFEST, PACKET, AND FINAL DRIVE FREEZE
# ----------------------------------------------------------------------------

manifest_rows = []
for path in sorted(RESULT_ROOT.rglob("*")):
    if path.is_file() and path.name != "stage5c1_sha256_manifest.csv":
        manifest_rows.append(
            {
                "relative_path": path.relative_to(RESULT_ROOT).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
manifest = pd.DataFrame(manifest_rows)
manifest_path = RESULT_ROOT / "stage5c1_sha256_manifest.csv"
manifest.to_csv(manifest_path, index=False)

if PACKET_PATH.exists():
    PACKET_PATH.unlink()
with zipfile.ZipFile(PACKET_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file():
            archive.write(
                path,
                arcname=(
                    "Stage5C1_Dual_GPU_LOSO_Pretraining/"
                    + path.relative_to(RESULT_ROOT).as_posix()
                ),
            )
with zipfile.ZipFile(PACKET_PATH, "r") as archive:
    packet_crc_pass = archive.testzip() is None
packet_sha256 = sha256_file(PACKET_PATH)

run_rclone(
    [
        "copy", str(RESULT_ROOT), DRIVE_REMOTE_DIRECTORY,
        "--retries", "5", "--low-level-retries", "10", "--timeout", "5m",
        "--transfers", "4", "--checkers", "8",
    ],
    rclone_config,
    check=True,
)
run_rclone(
    [
        "copyto", str(PACKET_PATH),
        DRIVE_REMOTE_DIRECTORY + "/" + PACKET_PATH.name,
        "--retries", "5", "--low-level-retries", "10", "--timeout", "5m",
    ],
    rclone_config,
    check=True,
)
remote_listing = run_rclone(
    ["lsf", DRIVE_REMOTE_DIRECTORY, "--files-only"],
    rclone_config,
    check=True,
).stdout.splitlines()
remote_packet_verified = PACKET_PATH.name in {line.strip() for line in remote_listing}
if rclone_config.exists():
    rclone_config.unlink()

print()
print("=" * 79)
print("STAGE 5C-1 — LOSO PRETRAINING SUMMARY")
print("=" * 79)
print(completion_summary[[
    "target_participant", "training_seed", "train_repetitions",
    "validation_repetitions", "completed_epochs", "stopped_early",
    "best_epoch", "best_validation_balanced_accuracy",
    "best_validation_macro_f1", "runtime_minutes",
]].to_string(index=False))
print()
print("Worker exit codes:", worker_exit_codes)
print("Drive sync successes:", sync_successes)
print("Drive sync failures:", sync_failures)
print("Packet CRC pass:", packet_crc_pass)
print("Packet:", PACKET_PATH)
print("Packet SHA-256:", packet_sha256)
print("Remote packet verified:", remote_packet_verified)
print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 2))
print()
print("Readiness gates:")
for gate, passed in readiness_gates.items():
    print(f"  {gate}: {passed}")
print()
if report["all_readiness_gates_passed"] and packet_crc_pass and remote_packet_verified:
    print("FINAL DECISION: PASS_TO_STAGE5D_DETERMINISTIC_DEEP_TRAJECTORIES")
else:
    print("FINAL DECISION: STAGE5C1_FINALIZATION_INCOMPLETE")
