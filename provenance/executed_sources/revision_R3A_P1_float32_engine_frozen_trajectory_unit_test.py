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
from sklearn.linear_model import RidgeClassifier


REVISION_PROTOCOL_SHA256 = (
    "6807b71de18ca82013cfa4360d760e0daf9a920a1acc0625dcb13bd8f4d07249"
)
R1_PACKET_SHA256 = (
    "2ec8cff608a765d20807e2d57249bf091768d97ff747ee2de7b44bcb17475ec8"
)
R2A_PACKET_SHA256 = (
    "554270b1d5bcc0bf7791020b64c44b72a267d71a2ee24afb5f43029957d01a8a"
)
STAGE5B_PACKET_SHA256 = (
    "1c0fbc63f6412362f3ae7cd22609ea6a7fcb23236cdf688ad5fe0578ebaab84d"
)
STAGE5D2_PACKET_SHA256 = (
    "fc8ac364bac0344639a50977d5f8725b1e5b5b2875758e01587de8c083a1f914"
)
R3A_D3_PACKET_SHA256 = "edb4307b9315e318d93111948f318f39cac20955f821d7ea0df4ecbe6f218a28"
NUMERICAL_ENGINE_CONTRACT = "FLOAT32_END_TO_END_RIDGE_ALPHA_1_SOLVER_AUTO"

PARTICIPANTS = [f"P{index:02d}" for index in range(1, 8)]
CHANNELS = 64
WINDOWS = 37
CLASSES = 7
SELECTOR_COLUMNS = ["opaque_candidate_token", "predicted_label", "margin"]
FORBIDDEN_SELECTOR_COLUMNS = {
    "participant",
    "session",
    "label",
    "true_label",
    "repetition",
    "repetition_uid",
    "relative_path",
    "dataset_key",
}
CONFIGURATIONS = [
    ("NO_ADAPTATION_REFERENCE", 0),
    ("PCBM_PROPOSED", 7),
    ("PCBM_PROPOSED", 14),
    ("PCBM_PROPOSED", 21),
    ("GLOBAL_MARGIN", 7),
    ("GLOBAL_MARGIN", 14),
    ("GLOBAL_MARGIN", 21),
    ("FULL_POOL_REFERENCE", 35),
]

WORKING = Path(os.environ.get("REVISION_R3A_WORKING", "/kaggle/working"))
TOOLS = WORKING / "_stage5_tools"
RCLONE = TOOLS / "rclone"
INPUT_ROOT = WORKING / "REVISION_R3A_P1_FROZEN_INPUTS"
RESULT_ROOT = (
    WORKING
    / "DELTA_REVIEWER_REVISION"
    / "Revision_R3A_P1_Float32_Engine_Frozen_Trajectory_Unit_Test"
)
PACKET_PATH = WORKING / "revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip"
for directory in (TOOLS, INPUT_ROOT, RESULT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

REMOTE_BASE = "gdrive_stage5:DELTA_Q1_Stage5_DeepLearning_Backup"
REMOTE_OUTPUT = (
    REMOTE_BASE
    + "/Reviewer_Revision/Revision_R3A_P1_Float32_Engine_Frozen_Trajectory_Unit_Test"
)
CONFIG_PATH = None
REMOTE_LISTING = None
START_TIME = time.time()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_crc_passes(path):
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def archive_matches(packet, basename):
    with zipfile.ZipFile(packet, "r") as archive:
        return [name for name in archive.namelist() if Path(name).name == basename]


def archive_member(packet, basename):
    matches = archive_matches(packet, basename)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {basename} in {packet}; found {matches}")
    with zipfile.ZipFile(packet, "r") as archive:
        return archive.read(matches[0])


def read_json_member(packet, basename):
    return json.loads(archive_member(packet, basename).decode("utf-8"))


def read_csv_member(packet, basename):
    return pd.read_csv(io.BytesIO(archive_member(packet, basename)))


def extract_member(packet, basename, destination):
    data = archive_member(packet, basename)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, destination)
    return destination


def atomic_json(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_csv(frame, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def make_zip(source_directory, destination, archive_root):
    destination = Path(destination)
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(source_directory).rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=(Path(archive_root) / path.relative_to(source_directory)).as_posix(),
                )
    return archive_crc_passes(destination)


def persist_executed_source(destination):
    source_name = globals().get("__file__")
    if source_name and Path(source_name).is_file():
        shutil.copy2(source_name, destination)
        return "PYTHON_FILE"
    raise RuntimeError("Could not capture Revision R3A executed source")


def cleanup_secret():
    global CONFIG_PATH
    if CONFIG_PATH and Path(CONFIG_PATH).exists():
        try:
            Path(CONFIG_PATH).unlink()
        except OSError:
            pass


atexit.register(cleanup_secret)


def bootstrap_rclone():
    if RCLONE.exists():
        RCLONE.chmod(0o755)
        return
    print("Downloading verified official rclone binary...", flush=True)
    version_text = urllib.request.urlopen(
        "https://downloads.rclone.org/version.txt", timeout=60
    ).read().decode("utf-8")
    match = re.search(r"v?(\d+\.\d+\.\d+)", version_text)
    if match is None:
        raise RuntimeError("Could not resolve official rclone version")
    version = match.group(1)
    archive_name = f"rclone-v{version}-linux-amd64.zip"
    base_url = f"https://downloads.rclone.org/v{version}"
    temporary_root = Path(tempfile.mkdtemp(prefix="revision_r3a_rclone_", dir="/tmp"))
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
        members = [name for name in archive.namelist() if name.endswith("/rclone")]
        if archive.testzip() is not None or len(members) != 1:
            raise RuntimeError("rclone archive verification failed")
        with archive.open(members[0]) as source, open(RCLONE, "wb") as target:
            shutil.copyfileobj(source, target)
    RCLONE.chmod(0o755)
    shutil.rmtree(temporary_root, ignore_errors=True)


def create_rclone_config():
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
    temporary = tempfile.NamedTemporaryFile(
        prefix="revision_r3a_", suffix=".conf", dir="/tmp", delete=False
    )
    temporary.write(decoded)
    temporary.flush()
    temporary.close()
    os.chmod(temporary.name, 0o600)
    CONFIG_PATH = Path(temporary.name)


def rclone(arguments, check=True):
    return subprocess.run(
        [str(RCLONE), "--config", str(CONFIG_PATH)] + list(arguments),
        check=check,
        capture_output=True,
        text=True,
    )


def remote_listing():
    global REMOTE_LISTING
    if REMOTE_LISTING is None:
        REMOTE_LISTING = rclone(
            ["lsf", REMOTE_BASE, "--recursive", "--files-only"]
        ).stdout.splitlines()
    return REMOTE_LISTING


def choose_remote(matches):
    priorities = [
        "Reviewer_Revision/Classical_Detail_Packets/",
        "Reviewer_Revision/",
        "Evidence/",
        "Deep_Analysis/",
        "Deep_Training/",
    ]
    for prefix in priorities:
        selected = [path for path in matches if path.startswith(prefix)]
        if selected:
            return sorted(selected)[0]
    return sorted(matches)[0]


def resolve_packet(basename, expected_sha256):
    destination = INPUT_ROOT / basename
    if (
        destination.exists()
        and sha256_file(destination) == expected_sha256
        and archive_crc_passes(destination)
    ):
        return destination, "EXISTING_VERIFIED_COPY"
    accepted_names = {basename, basename + ".bin"}
    matches = [path for path in remote_listing() if Path(path).name in accepted_names]
    if not matches:
        raise FileNotFoundError(f"Required frozen packet unavailable: {basename}")
    ordered = [choose_remote(matches)] + sorted(
        set(matches).difference({choose_remote(matches)})
    )
    for remote_path in ordered:
        temporary = destination.with_suffix(".download")
        rclone(
            [
                "copyto",
                REMOTE_BASE + "/" + remote_path,
                str(temporary),
                "--retries",
                "5",
                "--timeout",
                "5m",
            ]
        )
        if sha256_file(temporary) == expected_sha256 and archive_crc_passes(temporary):
            os.replace(temporary, destination)
            return destination, "GOOGLE_DRIVE"
        temporary.unlink(missing_ok=True)
    raise RuntimeError(f"No verified copy matched the frozen hash: {basename}")


def roundtrip_remote_file(local_path, remote_path, expected_hash):
    rclone(["copyto", str(local_path), remote_path, "--retries", "5", "--timeout", "5m"])
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="revision_r3a_roundtrip_", suffix=".zip", dir="/tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        rclone(["copyto", remote_path, str(temporary), "--retries", "5", "--timeout", "5m"])
        return sha256_file(temporary) == expected_hash
    finally:
        temporary.unlink(missing_ok=True)


def extract_stage3g_hash_map(stage3g_packet):
    mapping = {}
    with zipfile.ZipFile(stage3g_packet, "r") as archive:
        for member in archive.namelist():
            if Path(member).suffix.lower() != ".csv":
                continue
            try:
                frame = pd.read_csv(io.BytesIO(archive.read(member)))
            except Exception:
                continue
            for filename_column in frame.columns:
                filenames = frame[filename_column].astype(str)
                if not filenames.str.contains(r"\.zip(?:\.bin)?$", regex=True).any():
                    continue
                for hash_column in frame.columns:
                    hashes = frame[hash_column].astype(str).str.lower()
                    if not hashes.str.fullmatch(r"[0-9a-f]{64}").any():
                        continue
                    for filename, digest in zip(filenames, hashes):
                        filename = Path(filename).name.removesuffix(".bin")
                        if filename.endswith(".zip") and re.fullmatch(r"[0-9a-f]{64}", digest):
                            mapping[filename] = digest
    return mapping


def integrity_hash_from_r1(r1_packet, basename):
    integrity = read_csv_member(r1_packet, "revision_R1_packet_integrity.csv")
    rows = integrity[integrity["packet"].astype(str).eq(basename)]
    if len(rows) != 1:
        raise RuntimeError(f"R1 does not expose one integrity row for {basename}")
    digest = str(rows.iloc[0]["observed_sha256"]).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError(f"Invalid R1 hash for {basename}")
    return digest


def find_csv_member(packet, required_name_tokens, preferred_basename=None):
    candidates = []
    with zipfile.ZipFile(packet, "r") as archive:
        for member in archive.namelist():
            basename = Path(member).name.lower()
            if not basename.endswith(".csv"):
                continue
            if all(token in basename for token in required_name_tokens):
                try:
                    frame = pd.read_csv(io.BytesIO(archive.read(member)))
                except Exception:
                    continue
                candidates.append((member, frame))
    if preferred_basename is not None:
        preferred = [
            candidate
            for candidate in candidates
            if Path(candidate[0]).name.lower() == preferred_basename.lower()
        ]
        if len(preferred) == 1:
            return preferred[0]
        if len(preferred) > 1:
            raise RuntimeError(
                f"Expected one exact {preferred_basename} in {packet}; "
                f"found {[name for name, _ in preferred]}"
            )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one CSV containing {required_name_tokens} in {packet}; "
            f"found {[name for name, _ in candidates]}"
        )
    return candidates[0]


def resolve_column(frame, aliases, contains=None, required=True):
    lower = {str(column).lower(): column for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    if contains:
        matches = [
            column
            for column in frame.columns
            if all(token.lower() in str(column).lower() for token in contains)
        ]
        if len(matches) == 1:
            return matches[0]
    if required:
        raise RuntimeError(
            f"Could not resolve column aliases={aliases}, contains={contains}; "
            f"columns={frame.columns.tolist()}"
        )
    return None


def normalize_boolean_series(series, name):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise")
        if not values.isin([0, 1]).all():
            raise RuntimeError(f"Non-binary numeric values in {name}")
        return values.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    )
    if mapped.isna().any():
        raise RuntimeError(f"Could not normalize Boolean column {name}")
    return mapped.astype(bool)


def normalize_selection_table(frame):
    columns = {
        "participant": resolve_column(frame, ["participant"]),
        "target_session": resolve_column(frame, ["target_session", "session"]),
        "strategy": resolve_column(frame, ["strategy", "method"]),
        "query_budget": resolve_column(frame, ["query_budget", "budget"]),
        "opaque_candidate_token": resolve_column(
            frame,
            ["opaque_candidate_token", "selected_opaque_candidate_token", "selected_token"],
            contains=["opaque", "token"],
        ),
        "repetition_uid": resolve_column(
            frame,
            [
                "repetition_uid",
                "selected_repetition_uid",
                "internal_repetition_uid",
            ],
            contains=["repetition", "uid"],
            required=False,
        ),
        "sequence_row": resolve_column(
            frame,
            [
                "sequence_row_internal",
                "sequence_row",
                "selected_sequence_row",
            ],
            contains=["sequence", "row"],
            required=False,
        ),
        "query_round": resolve_column(
            frame,
            ["query_round", "round_index", "selection_round"],
            contains=["round"],
            required=False,
        ),
        "position": resolve_column(
            frame,
            [
                "selection_position",
                "selection_position_in_round",
                "position_in_round",
                "query_position",
                "selection_index",
            ],
            contains=["position"],
            required=False,
        ),
    }
    normalized = pd.DataFrame(
        {
            name: frame[column] if column is not None else np.nan
            for name, column in columns.items()
        }
    )
    normalized["_frozen_row_order"] = np.arange(len(normalized), dtype=int)
    normalized["participant"] = normalized["participant"].astype(str)
    normalized["strategy"] = normalized["strategy"].astype(str)
    normalized["opaque_candidate_token"] = normalized[
        "opaque_candidate_token"
    ].astype(str)
    normalized["repetition_uid"] = normalized["repetition_uid"].astype(str)
    for column in ["target_session", "query_budget"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(int)
    if columns["query_round"] is not None:
        normalized["query_round"] = pd.to_numeric(
            normalized["query_round"], errors="raise"
        ).astype(int)
    if columns["position"] is not None:
        normalized["position"] = pd.to_numeric(
            normalized["position"], errors="raise"
        ).astype(int)
    if columns["sequence_row"] is not None:
        normalized["sequence_row"] = pd.to_numeric(
            normalized["sequence_row"], errors="raise"
        ).astype(int)
    return normalized, columns


def normalize_prediction_table(frame):
    columns = {
        "participant": resolve_column(frame, ["participant"]),
        "target_session": resolve_column(frame, ["target_session", "session"]),
        "strategy": resolve_column(frame, ["strategy", "method"]),
        "query_budget": resolve_column(frame, ["query_budget", "budget"]),
        "repetition_uid": resolve_column(
            frame,
            ["repetition_uid", "test_repetition_uid", "internal_repetition_uid"],
            contains=["repetition", "uid"],
            required=False,
        ),
        "true_label": resolve_column(
            frame, ["true_label", "label", "y_true", "true_label_internal"]
        ),
        "predicted_label": resolve_column(
            frame,
            ["predicted_label", "prediction", "y_pred", "repetition_prediction"],
            contains=["predicted", "label"],
        ),
    }
    normalized = pd.DataFrame(
        {
            name: frame[column] if column is not None else ""
            for name, column in columns.items()
        }
    )
    normalized["_frozen_row_order"] = np.arange(len(normalized), dtype=int)
    normalized["participant"] = normalized["participant"].astype(str)
    normalized["strategy"] = normalized["strategy"].astype(str)
    normalized["repetition_uid"] = normalized["repetition_uid"].astype(str)
    for column in ["target_session", "query_budget", "true_label", "predicted_label"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(int)
    grouping = [
        "participant",
        "target_session",
        "strategy",
        "query_budget",
        "true_label",
    ]
    normalized["class_position"] = normalized.groupby(
        grouping, sort=False
    ).cumcount()
    return normalized, columns


def normalize_fold_table(frame):
    columns = {
        "participant": resolve_column(frame, ["participant"]),
        "target_session": resolve_column(frame, ["target_session", "session"]),
        "strategy": resolve_column(frame, ["strategy", "method"]),
        "query_budget": resolve_column(frame, ["query_budget", "budget"]),
        "repetition_balanced_accuracy": resolve_column(
            frame,
            ["repetition_balanced_accuracy", "balanced_accuracy"],
            contains=["repetition", "balanced", "accuracy"],
        ),
    }
    normalized = frame[[column for column in columns.values()]].copy()
    normalized.columns = list(columns)
    normalized["participant"] = normalized["participant"].astype(str)
    normalized["strategy"] = normalized["strategy"].astype(str)
    for column in ["target_session", "query_budget"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(int)
    normalized["repetition_balanced_accuracy"] = pd.to_numeric(
        normalized["repetition_balanced_accuracy"], errors="raise"
    ).astype(float)
    return normalized, columns


def validate_selector_frame(frame):
    if frame.columns.tolist() != SELECTOR_COLUMNS:
        raise ValueError(f"Selector schema drift: {frame.columns.tolist()}")
    if set(frame.columns).intersection(FORBIDDEN_SELECTOR_COLUMNS):
        raise ValueError("Forbidden selector column exposed")
    if frame["opaque_candidate_token"].duplicated().any():
        raise ValueError("Duplicate opaque token")
    if not frame["opaque_candidate_token"].astype(str).str.fullmatch(r"[0-9a-f]{24}").all():
        raise ValueError("Non-opaque selector token")
    if not np.isfinite(frame["margin"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite selector margin")


def selector_order(frame):
    return frame.sort_values(
        ["margin", "opaque_candidate_token"], kind="mergesort"
    )


def select_pcbm(frame):
    validate_selector_frame(frame)
    ordered = selector_order(frame)
    nominees = (
        ordered.groupby("predicted_label", sort=False, as_index=False)
        .head(1)
        .sort_values(["margin", "opaque_candidate_token"], kind="mergesort")
    )
    selected = nominees.head(7)["opaque_candidate_token"].astype(str).tolist()
    if len(selected) < 7:
        fill = ordered.loc[
            ~ordered["opaque_candidate_token"].isin(selected),
            "opaque_candidate_token",
        ].astype(str).tolist()
        selected.extend(fill[: 7 - len(selected)])
    if len(selected) != 7 or len(set(selected)) != 7:
        raise RuntimeError("PCBM failed to select seven unique tokens")
    return selected


def select_global_margin(frame):
    validate_selector_frame(frame)
    selected = selector_order(frame).head(7)["opaque_candidate_token"].astype(str).tolist()
    if len(selected) != 7 or len(set(selected)) != 7:
        raise RuntimeError("Global margin failed to select seven unique tokens")
    return selected


def fit_normalizer(raw, valid):
    # R3A-D3 identified the immutable Stage 3C-1 numerical path as native
    # float32 from log1p through source-only statistics and Ridge input.
    logged = np.log1p(np.asarray(raw, dtype=np.float32))
    valid = np.asarray(valid, dtype=bool)
    means = np.zeros(CHANNELS, dtype=np.float32)
    stds = np.zeros(CHANNELS, dtype=np.float32)
    counts = np.zeros(CHANNELS, dtype=np.int64)
    for channel in range(CHANNELS):
        values = logged[:, :, channel][valid[:, :, channel]]
        counts[channel] = len(values)
        if len(values) == 0:
            raise ValueError(f"No main-valid history values for channel {channel}")
        means[channel] = values.mean()
        stds[channel] = values.std(ddof=0)
        if not np.isfinite(stds[channel]) or stds[channel] <= 0:
            raise ValueError(f"Invalid main-mask history std for channel {channel}")
    return means, stds, counts


def transform_repetitions(features, main_valid, rows, means, stds):
    rows = np.asarray(rows, dtype=int)
    raw = np.asarray(features[rows], dtype=np.float32)
    valid = np.asarray(main_valid[rows], dtype=bool)
    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)
    transformed = (np.log1p(raw) - means[None, None, :]) / stds[None, None, :]
    transformed[~valid] = np.float32(0.0)
    if transformed.dtype != np.float32:
        raise RuntimeError("Float32 numerical contract drift")
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite transformed feature")
    return transformed


def fit_history_state(features, main_valid, metadata, history_rows):
    history_rows = np.asarray(history_rows, dtype=int)
    if len(history_rows) == 0 or len(np.unique(history_rows)) != len(history_rows):
        raise RuntimeError("History rows must be nonempty and unique")
    history_meta = metadata.iloc[history_rows]
    if history_meta["fixed_test_never_query"].astype(bool).any():
        raise RuntimeError("Fixed test entered history")
    if not history_meta["eligible_for_training"].astype(bool).all():
        raise RuntimeError("Non-training-eligible repetition entered history")
    if sorted(history_meta["label"].unique().tolist()) != list(range(CLASSES)):
        raise RuntimeError("History lacks one or more classes")
    means, stds, counts = fit_normalizer(
        features[history_rows], main_valid[history_rows]
    )
    transformed = transform_repetitions(
        features, main_valid, history_rows, means, stds
    )
    x = transformed.reshape(-1, CHANNELS)
    y = np.repeat(history_meta["label"].to_numpy(dtype=int), WINDOWS)
    model = RidgeClassifier(alpha=1.0, solver="auto")
    model.fit(x, y)
    return {
        "model": model,
        "means": means,
        "stds": stds,
        "counts": counts,
        "history_rows": history_rows.copy(),
        "numerical_engine_contract": NUMERICAL_ENGINE_CONTRACT,
        "training_array_dtype": str(x.dtype),
        "model_coefficient_dtype": str(np.asarray(model.coef_).dtype),
    }


def score_repetitions(state, features, main_valid, rows):
    rows = np.asarray(rows, dtype=int)
    transformed = transform_repetitions(
        features, main_valid, rows, state["means"], state["stds"]
    )
    window_scores = state["model"].decision_function(
        transformed.reshape(-1, CHANNELS)
    )
    window_scores = np.asarray(window_scores, dtype=np.float64).reshape(
        len(rows), WINDOWS, CLASSES
    )
    repetition_scores = window_scores.mean(axis=1)
    predicted = np.argmax(repetition_scores, axis=1).astype(int)
    ordered = np.sort(repetition_scores, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    if not np.isfinite(repetition_scores).all() or not np.isfinite(margins).all():
        raise RuntimeError("Non-finite Ridge repetition scores")
    return repetition_scores, predicted, margins


def balanced_accuracy(y_true, y_pred):
    recalls = []
    for label in range(CLASSES):
        mask = np.asarray(y_true) == label
        if mask.sum() != 5:
            raise RuntimeError("Fixed test is not five-per-class balanced")
        recalls.append(float((np.asarray(y_pred)[mask] == label).mean()))
    return float(np.mean(recalls))


def initial_history_rows(metadata, participant):
    mask = (
        metadata["participant"].eq(participant)
        & metadata["session"].eq(0)
        & metadata["protocol_role"].eq("INITIAL_LABELED_CALIBRATION")
    )
    rows = metadata.loc[mask, "sequence_row"].to_numpy(dtype=int)
    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 initial rows for {participant}")
    return rows


def candidate_rows(metadata, participant, session):
    mask = (
        metadata["participant"].eq(participant)
        & metadata["session"].eq(int(session))
        & metadata["protocol_role"].eq("CURRENT_SESSION_UNLABELED_POOL")
    )
    rows = metadata.loc[mask, "sequence_row"].to_numpy(dtype=int)
    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 candidates for {participant} session {session}")
    return rows


def fixed_test_rows(metadata, participant, session):
    mask = (
        metadata["participant"].eq(participant)
        & metadata["session"].eq(int(session))
        & metadata["protocol_role"].eq("TARGET_FIXED_TEST_NEVER_QUERY")
    )
    rows = metadata.loc[mask, "sequence_row"].to_numpy(dtype=int)
    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 fixed-test rows for {participant} session {session}")
    return rows


def reveal_rows(metadata, selected_tokens, remaining_rows):
    remaining_meta = metadata.iloc[np.asarray(remaining_rows, dtype=int)]
    mapping = dict(
        zip(
            remaining_meta["opaque_candidate_token"].astype(str),
            remaining_meta["sequence_row"].astype(int),
        )
    )
    if any(token not in mapping for token in selected_tokens):
        raise RuntimeError("Selector returned token outside remaining candidates")
    rows = np.asarray([mapping[token] for token in selected_tokens], dtype=int)
    selected_meta = metadata.iloc[rows]
    if not selected_meta["protocol_role"].eq("CURRENT_SESSION_UNLABELED_POOL").all():
        raise RuntimeError("Non-candidate selected")
    if selected_meta["fixed_test_never_query"].astype(bool).any():
        raise RuntimeError("Fixed test selected")
    return rows


def selector_frame(metadata, rows, predicted, margins):
    return pd.DataFrame(
        {
            "opaque_candidate_token": metadata.iloc[np.asarray(rows, dtype=int)][
                "opaque_candidate_token"
            ].astype(str).to_numpy(),
            "predicted_label": np.asarray(predicted, dtype=int),
            "margin": np.asarray(margins, dtype=float),
        },
        columns=SELECTOR_COLUMNS,
    )


def run_deterministic_reconstruction(features, main_valid, metadata):
    selection_rows = []
    prediction_rows = []
    fold_rows = []
    normalizer_rows = []
    for participant in PARTICIPANTS:
        for strategy, budget in CONFIGURATIONS:
            history = initial_history_rows(metadata, participant).tolist()
            for session in range(1, 6):
                remaining = candidate_rows(metadata, participant, session).tolist()
                selected_session = []
                rounds = budget // 7
                if strategy == "FULL_POOL_REFERENCE":
                    ordered_tokens = sorted(
                        metadata.iloc[remaining]["opaque_candidate_token"].astype(str).tolist()
                    )
                    for round_index in range(1, 6):
                        tokens = ordered_tokens[(round_index - 1) * 7 : round_index * 7]
                        rows = reveal_rows(metadata, tokens, remaining)
                        for position, (token, row) in enumerate(zip(tokens, rows), start=1):
                            selection_rows.append(
                                {
                                    "participant": participant,
                                    "target_session": session,
                                    "strategy": strategy,
                                    "query_budget": budget,
                                    "query_round": round_index,
                                    "position": position,
                                    "opaque_candidate_token": token,
                                    "sequence_row": int(row),
                                    "repetition_uid": str(
                                        metadata.iloc[int(row)]["repetition_uid"]
                                    ),
                                }
                            )
                        selected_session.extend(rows.tolist())
                        remaining = [row for row in remaining if row not in set(rows.tolist())]
                elif strategy in {"PCBM_PROPOSED", "GLOBAL_MARGIN"}:
                    for round_index in range(1, rounds + 1):
                        state = fit_history_state(features, main_valid, metadata, history)
                        _, predicted, margins = score_repetitions(
                            state, features, main_valid, remaining
                        )
                        visible = selector_frame(metadata, remaining, predicted, margins)
                        tokens = (
                            select_pcbm(visible)
                            if strategy == "PCBM_PROPOSED"
                            else select_global_margin(visible)
                        )
                        rows = reveal_rows(metadata, tokens, remaining)
                        for position, (token, row) in enumerate(zip(tokens, rows), start=1):
                            selection_rows.append(
                                {
                                    "participant": participant,
                                    "target_session": session,
                                    "strategy": strategy,
                                    "query_budget": budget,
                                    "query_round": round_index,
                                    "position": position,
                                    "opaque_candidate_token": token,
                                    "sequence_row": int(row),
                                    "repetition_uid": str(
                                        metadata.iloc[int(row)]["repetition_uid"]
                                    ),
                                }
                            )
                        history.extend(rows.tolist())
                        selected_session.extend(rows.tolist())
                        selected_set = set(rows.tolist())
                        remaining = [row for row in remaining if row not in selected_set]
                elif strategy != "NO_ADAPTATION_REFERENCE":
                    raise RuntimeError(f"Unsupported deterministic strategy: {strategy}")

                if strategy == "FULL_POOL_REFERENCE":
                    history.extend(selected_session)
                if len(selected_session) != budget:
                    raise RuntimeError(
                        f"Selection count mismatch: {participant} {strategy} K{budget:02d} S{session}"
                    )
                final_state = fit_history_state(features, main_valid, metadata, history)
                test_rows = fixed_test_rows(metadata, participant, session)
                if np.intersect1d(test_rows, final_state["history_rows"]).size:
                    raise RuntimeError("Fixed test entered fit/normalization history")
                _, predicted, _ = score_repetitions(
                    final_state, features, main_valid, test_rows
                )
                truth = metadata.iloc[test_rows]["label"].to_numpy(dtype=int)
                ba = balanced_accuracy(truth, predicted)
                fold_rows.append(
                    {
                        "participant": participant,
                        "target_session": session,
                        "strategy": strategy,
                        "query_budget": budget,
                        "history_repetitions": len(history),
                        "selected_repetitions_this_session": len(selected_session),
                        "repetition_balanced_accuracy": ba,
                    }
                )
                normalizer_rows.append(
                    {
                        "participant": participant,
                        "target_session": session,
                        "strategy": strategy,
                        "query_budget": budget,
                        "history_repetitions": len(history),
                        "minimum_valid_count": int(final_state["counts"].min()),
                        "all_means_finite": bool(np.isfinite(final_state["means"]).all()),
                        "all_stds_finite": bool(np.isfinite(final_state["stds"]).all()),
                        "all_stds_positive": bool((final_state["stds"] > 0).all()),
                        "normalizer_mean_dtype": str(final_state["means"].dtype),
                        "normalizer_std_dtype": str(final_state["stds"].dtype),
                        "training_array_dtype": final_state["training_array_dtype"],
                        "model_coefficient_dtype": final_state["model_coefficient_dtype"],
                        "numerical_engine_contract": final_state["numerical_engine_contract"],
                    }
                )
                test_meta = metadata.iloc[test_rows]
                for index, row in enumerate(test_meta.itertuples(index=False)):
                    prediction_rows.append(
                        {
                            "participant": participant,
                            "target_session": session,
                            "strategy": strategy,
                            "query_budget": budget,
                            "repetition_uid": str(row.repetition_uid),
                            "true_label": int(row.label),
                            "predicted_label": int(predicted[index]),
                        }
                    )
            print(
                f"Reconstructed | {participant} | {strategy} K{budget:02d}",
                flush=True,
            )
    predictions = pd.DataFrame(prediction_rows)
    predictions["class_position"] = predictions.groupby(
        [
            "participant",
            "target_session",
            "strategy",
            "query_budget",
            "true_label",
        ],
        sort=False,
    ).cumcount()
    return (
        pd.DataFrame(selection_rows),
        predictions,
        pd.DataFrame(fold_rows),
        pd.DataFrame(normalizer_rows),
    )


def compare_frozen_selections(observed, frozen):
    group_columns = ["participant", "target_session", "strategy", "query_budget"]
    rows = []
    keys = sorted(
        set(map(tuple, observed[group_columns].drop_duplicates().to_numpy().tolist()))
        | set(map(tuple, frozen[group_columns].drop_duplicates().to_numpy().tolist()))
    )
    for key in keys:
        observed_group = observed
        frozen_group = frozen
        for column, value in zip(group_columns, key):
            observed_group = observed_group[observed_group[column].eq(value)]
            frozen_group = frozen_group[frozen_group[column].eq(value)]
        observed_tokens = observed_group["opaque_candidate_token"].astype(str).tolist()
        frozen_tokens = frozen_group["opaque_candidate_token"].astype(str).tolist()
        set_match = sorted(observed_tokens) == sorted(frozen_tokens)
        identity_available = bool(
            frozen_group["sequence_row"].notna().all()
            and observed_group["sequence_row"].notna().all()
        )
        identity_match = None
        if identity_available:
            observed_identities = sorted(
                observed_group["sequence_row"].astype(int).tolist()
            )
            frozen_identities = sorted(
                frozen_group["sequence_row"].astype(int).tolist()
            )
            identity_match = observed_identities == frozen_identities
        round_match = None
        round_identity_match = None
        if frozen_group["query_round"].notna().all():
            observed_pairs = sorted(
                zip(observed_group["query_round"], observed_group["opaque_candidate_token"])
            )
            frozen_pairs = sorted(
                zip(frozen_group["query_round"], frozen_group["opaque_candidate_token"])
            )
            round_match = observed_pairs == frozen_pairs
            if identity_available:
                observed_identity_pairs = sorted(
                    zip(
                        observed_group["query_round"].astype(int),
                        observed_group["sequence_row"].astype(int),
                    )
                )
                frozen_identity_pairs = sorted(
                    zip(
                        frozen_group["query_round"].astype(int),
                        frozen_group["sequence_row"].astype(int),
                    )
                )
                round_identity_match = (
                    observed_identity_pairs == frozen_identity_pairs
                )
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "observed_count": len(observed_tokens),
                "frozen_count": len(frozen_tokens),
                "selected_token_set_matches": set_match,
                "selected_identity_available": identity_available,
                "selected_repetition_identity_set_matches": identity_match,
                "selected_round_membership_matches": round_match,
                "selected_round_repetition_identity_membership_matches": (
                    round_identity_match
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_frozen_predictions(observed, frozen):
    keys = [
        "participant",
        "target_session",
        "strategy",
        "query_budget",
        "true_label",
        "class_position",
    ]
    merged = observed.merge(
        frozen,
        on=keys,
        how="outer",
        suffixes=("_observed", "_frozen"),
        indicator=True,
        validate="one_to_one",
    )
    merged["predicted_label_matches"] = (
        merged["predicted_label_observed"] == merged["predicted_label_frozen"]
    )
    return merged


def compare_frozen_folds(observed, frozen):
    keys = ["participant", "target_session", "strategy", "query_budget"]
    merged = observed.merge(
        frozen,
        on=keys,
        how="outer",
        suffixes=("_observed", "_frozen"),
        indicator=True,
        validate="one_to_one",
    )
    merged["absolute_difference"] = (
        merged["repetition_balanced_accuracy_observed"]
        - merged["repetition_balanced_accuracy_frozen"]
    ).abs()
    return merged


def main():
    # Prevent stale files from an interrupted/repeated Kaggle invocation from
    # entering the evidence packet.
    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    PACKET_PATH.unlink(missing_ok=True)
    print("=" * 92)
    print("REVISION R3A-P1 — FLOAT32 ENGINE PATCH AND FROZEN-TRAJECTORY UNIT TEST")
    print("=" * 92)
    print("Execution device: CPU")
    print("Raw HDF5 data accessed: False")
    print("Scientific role: IMPLEMENTATION VALIDATION ONLY")
    print("Frozen fixed-test inference replay: True")
    print("New reviewer experiment executed: False")
    print("New statistical test executed: False")
    print("Frozen deterministic trajectories to reproduce: 56")
    print("Frozen deterministic folds to reproduce: 280")
    print()

    bootstrap_rclone()
    create_rclone_config()
    print("rclone version:", rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified parents and frozen engine artifacts...")

    d3_packet, d3_source = resolve_packet(
        "revision_R3A_D3_numerical_implementation_variant_audit_packet.zip",
        R3A_D3_PACKET_SHA256,
    )
    d3_report = read_json_member(d3_packet, "revision_R3A_D3_report.json")
    if not d3_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("Revision R3A-D3 parent did not pass all readiness gates")
    if d3_report.get("revision_protocol_sha256") != REVISION_PROTOCOL_SHA256:
        raise RuntimeError("Revision protocol hash drift at R3A-D3")
    if d3_report.get("classification") != "NUMERICAL_VARIANT_REPRODUCES_FROZEN_BOUNDARY":
        raise RuntimeError("R3A-D3 did not authorize the float32 numerical patch")
    d3_best = d3_report.get("best_configuration", {})
    expected_d3_best = {
        "stats_dtype": "float32",
        "train_dtype": "float32",
        "score_dtype": "float32",
        "solver": "auto",
        "thread_mode": "runtime_default",
    }
    if any(d3_best.get(key) != value for key, value in expected_d3_best.items()):
        raise RuntimeError(f"R3A-D3 best configuration drift: {d3_best}")

    r1_packet, r1_source = resolve_packet(
        "revision_R1_frozen_artifact_engine_audit_packet.zip", R1_PACKET_SHA256
    )
    r2a_packet, r2a_source = resolve_packet(
        "revision_R2A_classical_detail_packet_migration_packet.zip", R2A_PACKET_SHA256
    )
    r2a_report = read_json_member(r2a_packet, "revision_R2A_migration_report.json")
    if not r2a_report.get("all_readiness_gates_passed", False):
        raise RuntimeError("Revision R2A parent did not pass all readiness gates")
    if r2a_report.get("revision_protocol_sha256") != REVISION_PROTOCOL_SHA256:
        raise RuntimeError("Revision protocol hash drift at R2A")

    stage3g_hash = integrity_hash_from_r1(
        r1_packet, "stage3g_final_results_freeze_packet.zip"
    )
    stage3g_packet, stage3g_source = resolve_packet(
        "stage3g_final_results_freeze_packet.zip", stage3g_hash
    )
    stage5b_packet, stage5b_source = resolve_packet(
        "stage5b_deep_sequence_assembly_packet.zip", STAGE5B_PACKET_SHA256
    )
    stage5d2_packet, stage5d2_source = resolve_packet(
        "stage5d2_full_deterministic_deep_trajectories_packet.zip",
        STAGE5D2_PACKET_SHA256,
    )
    stage3g_hashes = extract_stage3g_hash_map(stage3g_packet)
    stage3c1_hash = stage3g_hashes["stage3c1_deterministic_experiment_packet.zip"]
    stage3c2_hash = stage3g_hashes["stage3c2_random_control_packet.zip"]
    stage3c1_packet, stage3c1_source = resolve_packet(
        "stage3c1_deterministic_experiment_packet.zip", stage3c1_hash
    )
    stage3c2_packet, stage3c2_source = resolve_packet(
        "stage3c2_random_control_packet.zip", stage3c2_hash
    )

    input_audit = pd.DataFrame(
        [
            ("revision_R3A_D3_numerical_implementation_variant_audit_packet.zip", R3A_D3_PACKET_SHA256, d3_source),
            ("revision_R1_frozen_artifact_engine_audit_packet.zip", R1_PACKET_SHA256, r1_source),
            ("revision_R2A_classical_detail_packet_migration_packet.zip", R2A_PACKET_SHA256, r2a_source),
            ("stage3g_final_results_freeze_packet.zip", stage3g_hash, stage3g_source),
            ("stage5b_deep_sequence_assembly_packet.zip", STAGE5B_PACKET_SHA256, stage5b_source),
            ("stage5d2_full_deterministic_deep_trajectories_packet.zip", STAGE5D2_PACKET_SHA256, stage5d2_source),
            ("stage3c1_deterministic_experiment_packet.zip", stage3c1_hash, stage3c1_source),
            ("stage3c2_random_control_packet.zip", stage3c2_hash, stage3c2_source),
        ],
        columns=["packet", "sha256", "source"],
    )
    input_audit["crc_passes"] = input_audit["packet"].map(
        lambda name: archive_crc_passes(INPUT_ROOT / name)
    )
    atomic_csv(input_audit, RESULT_ROOT / "revision_R3A_P1_input_packet_audit.csv")

    for basename in [
        "stage5b_rms_repetition_sequences.npy",
        "stage5b_main_valid_repetition_sequences.npy",
        "stage5b_repetition_metadata.csv",
        "stage5b_sequence_assembly_report.json",
    ]:
        extract_member(stage5b_packet, basename, INPUT_ROOT / basename)
    features = np.load(
        INPUT_ROOT / "stage5b_rms_repetition_sequences.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    main_valid = np.load(
        INPUT_ROOT / "stage5b_main_valid_repetition_sequences.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    metadata = pd.read_csv(INPUT_ROOT / "stage5b_repetition_metadata.csv")
    metadata["participant"] = metadata["participant"].astype(str)
    for column in ["session", "label", "repetition", "sequence_row"]:
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype(int)
    metadata = metadata.sort_values("sequence_row").reset_index(drop=True)
    if metadata["sequence_row"].tolist() != list(range(2940)):
        raise RuntimeError("Stage 5B sequence-row order drift")

    # Stage 3A is not required as a standalone Drive file. Protocol roles are
    # deterministic functions of the frozen session/repetition design, and the
    # complete opaque-token-to-sequence-row map is recovered from the Stage
    # 5D-2 FULL_POOL_REFERENCE trace, which selected all 1,225 target candidates.
    initial_mask = metadata["session"].eq(0) & metadata["repetition"].le(5)
    candidate_mask = metadata["session"].between(1, 5) & metadata["repetition"].le(5)
    fixed_mask = metadata["session"].between(1, 5) & metadata["repetition"].ge(6)
    metadata["protocol_role"] = "INITIAL_SESSION_UNUSED"
    metadata.loc[initial_mask, "protocol_role"] = "INITIAL_LABELED_CALIBRATION"
    metadata.loc[candidate_mask, "protocol_role"] = "CURRENT_SESSION_UNLABELED_POOL"
    metadata.loc[fixed_mask, "protocol_role"] = "TARGET_FIXED_TEST_NEVER_QUERY"
    metadata["eligible_for_query"] = candidate_mask.astype(bool)
    metadata["eligible_for_training"] = (initial_mask | candidate_mask).astype(bool)
    metadata["fixed_test_never_query"] = fixed_mask.astype(bool)
    metadata["case_analysis"] = metadata["participant"].eq("P07")

    # Reconstruct the exact Stage 3A semantic UID from frozen metadata fields.
    # The Stage 5B composite key is a join key, not the Stage 3A repetition UID.
    metadata["repetition_uid"] = metadata.apply(
        lambda row: (
            f"{row.participant}_S{int(row.session):02d}_"
            f"L{int(row.label)}_R{int(row.repetition):02d}"
        ),
        axis=1,
    )

    deep_selection_member, deep_selection = find_csv_member(
        stage5d2_packet,
        ["selection", "trace"],
        preferred_basename="stage5d2_selection_trace.csv",
    )
    deep_strategy_column = resolve_column(deep_selection, ["strategy"])
    deep_row_column = resolve_column(
        deep_selection,
        ["sequence_row_internal", "sequence_row"],
        contains=["sequence", "row"],
    )
    deep_token_column = resolve_column(
        deep_selection,
        ["opaque_candidate_token", "selected_opaque_candidate_token"],
        contains=["opaque", "token"],
    )
    full_pool_tokens = deep_selection.loc[
        deep_selection[deep_strategy_column].astype(str).eq("FULL_POOL_REFERENCE"),
        [deep_row_column, deep_token_column],
    ].copy()
    full_pool_tokens[deep_row_column] = pd.to_numeric(
        full_pool_tokens[deep_row_column], errors="raise"
    ).astype(int)
    full_pool_tokens[deep_token_column] = full_pool_tokens[
        deep_token_column
    ].astype(str)
    full_pool_tokens = full_pool_tokens.drop_duplicates()
    if (
        len(full_pool_tokens) != 1225
        or full_pool_tokens[deep_row_column].nunique() != 1225
        or full_pool_tokens[deep_token_column].nunique() != 1225
    ):
        raise RuntimeError(
            "Stage 5D-2 full-pool trace does not expose one token for every candidate"
        )
    token_by_row = dict(
        zip(full_pool_tokens[deep_row_column], full_pool_tokens[deep_token_column])
    )
    metadata["opaque_candidate_token"] = metadata["sequence_row"].map(token_by_row)
    if metadata.loc[candidate_mask, "opaque_candidate_token"].isna().any():
        raise RuntimeError("One or more target candidates lack an opaque token")

    selector_schema = {
        "selector_input_columns": SELECTOR_COLUMNS,
        "source": "STAGE5D2_FROZEN_SELECTOR_CONTRACT",
    }

    selection_member, frozen_selection_raw = find_csv_member(
        stage3c1_packet, ["selection", "trace"]
    )
    prediction_member, frozen_prediction_raw = find_csv_member(
        stage3c1_packet, ["repetition", "prediction"]
    )
    fold_member, frozen_fold_raw = find_csv_member(stage3c1_packet, ["fold"])
    frozen_selection, selection_columns = normalize_selection_table(
        frozen_selection_raw
    )
    row_by_uid = dict(
        zip(metadata["repetition_uid"].astype(str), metadata["sequence_row"].astype(int))
    )
    row_by_token = {str(token): int(row) for row, token in token_by_row.items()}
    frozen_identity_source = "UNAVAILABLE"
    if selection_columns["repetition_uid"] is not None:
        frozen_selection["sequence_row"] = frozen_selection["repetition_uid"].map(
            row_by_uid
        )
        frozen_identity_source = "FROZEN_REPETITION_UID"
    elif selection_columns["sequence_row"] is not None:
        frozen_identity_source = "FROZEN_SEQUENCE_ROW"
    else:
        frozen_selection["sequence_row"] = frozen_selection[
            "opaque_candidate_token"
        ].map(row_by_token)
        if frozen_selection["sequence_row"].notna().all():
            frozen_identity_source = "CROSS_PACKET_OPAQUE_TOKEN"
    frozen_prediction, prediction_columns = normalize_prediction_table(
        frozen_prediction_raw
    )
    frozen_fold, fold_columns = normalize_fold_table(frozen_fold_raw)
    random_selection_member, random_selection_raw = find_csv_member(
        stage3c2_packet, ["selection", "trace"]
    )

    table_discovery = {
        "stage5d2_full_pool_token_member": deep_selection_member,
        "stage5d2_full_pool_candidate_token_count": len(full_pool_tokens),
        "stage3c1_selection_member": selection_member,
        "stage3c1_prediction_member": prediction_member,
        "stage3c1_fold_member": fold_member,
        "stage3c2_selection_member": random_selection_member,
        "selection_column_mapping": selection_columns,
        "stage3c1_selection_identity_source": frozen_identity_source,
        "stage3c1_selection_identity_rows_resolved": int(
            frozen_selection["sequence_row"].notna().sum()
        ),
        "prediction_column_mapping": prediction_columns,
        "fold_column_mapping": fold_columns,
        "stage3c2_selection_rows": len(random_selection_raw),
    }
    atomic_json(table_discovery, RESULT_ROOT / "revision_R3A_P1_frozen_table_discovery.json")

    synthetic = pd.DataFrame(
        {
            "opaque_candidate_token": [f"{index:024x}" for index in range(9)],
            "predicted_label": [0, 0, 0, 1, 2, 3, 4, 5, 6],
            "margin": [0.001, 0.002, 0.003, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
        },
        columns=SELECTOR_COLUMNS,
    )
    synthetic_pcbm = select_pcbm(synthetic)
    synthetic_labels = synthetic.set_index("opaque_candidate_token").loc[
        synthetic_pcbm, "predicted_label"
    ].tolist()
    synthetic_one_per_class = set(synthetic_labels) == set(range(7))
    forbidden_rejected = False
    try:
        invalid = synthetic.copy()
        invalid["label"] = 0
        select_pcbm(invalid)
    except ValueError:
        forbidden_rejected = True

    print("Reconstructing the 56 frozen deterministic trajectories...", flush=True)
    selections, predictions, folds, normalizers = run_deterministic_reconstruction(
        features, main_valid, metadata
    )
    selection_comparison = compare_frozen_selections(selections, frozen_selection)
    prediction_comparison = compare_frozen_predictions(predictions, frozen_prediction)
    fold_comparison = compare_frozen_folds(folds, frozen_fold)

    atomic_csv(selections, RESULT_ROOT / "revision_R3A_P1_reconstructed_selection_trace.csv")
    atomic_csv(predictions, RESULT_ROOT / "revision_R3A_P1_reconstructed_repetition_predictions.csv")
    atomic_csv(folds, RESULT_ROOT / "revision_R3A_P1_reconstructed_folds.csv")
    atomic_csv(normalizers, RESULT_ROOT / "revision_R3A_P1_normalizer_audit.csv")
    atomic_csv(selection_comparison, RESULT_ROOT / "revision_R3A_P1_selection_comparison.csv")
    atomic_csv(prediction_comparison, RESULT_ROOT / "revision_R3A_P1_prediction_comparison.csv")
    atomic_csv(fold_comparison, RESULT_ROOT / "revision_R3A_P1_fold_comparison.csv")

    token_mismatches = selection_comparison.loc[
        ~selection_comparison["selected_token_set_matches"]
    ].copy()
    identity_mismatches = selection_comparison.loc[
        selection_comparison["selected_identity_available"]
        & ~selection_comparison["selected_repetition_identity_set_matches"].fillna(False)
    ].copy()
    print()
    print("Frozen selection comparison diagnostic:")
    print("  Stage 3C-1 identity source:", frozen_identity_source)
    print("  Token-set mismatch groups:", len(token_mismatches))
    print("  Repetition-identity mismatch groups:", len(identity_mismatches))
    if len(token_mismatches):
        print(token_mismatches.to_string(index=False))

    readiness_gates = {
        "revision_r3a_d3_packet_hash_matches": sha256_file(d3_packet) == R3A_D3_PACKET_SHA256,
        "revision_r3a_d3_all_gates_passed": bool(d3_report.get("all_readiness_gates_passed")),
        "revision_r3a_d3_classification_authorizes_patch": d3_report.get("classification") == "NUMERICAL_VARIANT_REPRODUCES_FROZEN_BOUNDARY",
        "revision_r3a_d3_best_configuration_is_float32_auto": all(
            d3_best.get(key) == value for key, value in expected_d3_best.items()
        ),
        "revision_r2a_packet_hash_matches": sha256_file(r2a_packet) == R2A_PACKET_SHA256,
        "revision_r2a_all_gates_passed": bool(r2a_report.get("all_readiness_gates_passed")),
        "revision_protocol_hash_matches": r2a_report.get("revision_protocol_sha256") == REVISION_PROTOCOL_SHA256,
        "all_eight_input_packets_pass_crc": bool(input_audit["crc_passes"].all()),
        "stage3c1_and_stage3c2_hashes_match_stage3g": bool(
            sha256_file(stage3c1_packet) == stage3c1_hash
            and sha256_file(stage3c2_packet) == stage3c2_hash
        ),
        "feature_shape_is_2940_by_37_by_64": features.shape == (2940, 37, 64),
        "main_mask_shape_matches_features": main_valid.shape == features.shape,
        "metadata_join_has_2940_rows": len(metadata) == 2940,
        "metadata_has_no_missing_repetition_keys": bool(metadata["repetition_uid"].notna().all()),
        "protocol_roles_are_derived_from_frozen_design": bool(
            initial_mask.sum() == 245
            and candidate_mask.sum() == 1225
            and fixed_mask.sum() == 1225
        ),
        "stage5d2_full_pool_maps_all_1225_candidate_tokens": bool(
            len(full_pool_tokens) == 1225
            and metadata.loc[candidate_mask, "opaque_candidate_token"].notna().all()
        ),
        "candidate_opaque_tokens_are_valid": bool(
            metadata.loc[candidate_mask, "opaque_candidate_token"]
            .astype(str)
            .str.fullmatch(r"[0-9a-f]{24}")
            .all()
        ),
        "standalone_stage3a_packet_is_not_required": True,
        "selector_schema_is_exactly_three_columns": selector_schema["selector_input_columns"] == SELECTOR_COLUMNS,
        "synthetic_pcbm_selects_one_per_class": synthetic_one_per_class,
        "forbidden_selector_column_is_rejected": forbidden_rejected,
        "reconstructed_trajectory_count_is_56": len(folds.groupby(["participant", "strategy", "query_budget"])) == 56,
        "reconstructed_fold_count_is_280": len(folds) == 280,
        "reconstructed_prediction_count_is_9800": len(predictions) == 9800,
        "reconstructed_selection_count_is_4165": len(selections) == 4165,
        "frozen_selection_count_is_4165": len(frozen_selection) == 4165,
        "all_stage3c1_selected_repetition_identities_are_resolved": bool(
            selection_comparison["selected_identity_available"].all()
        ),
        "all_selected_repetition_identity_sets_match_stage3c1": bool(
            selection_comparison[
                "selected_repetition_identity_set_matches"
            ].fillna(False).all()
        ),
        "opaque_token_text_match_is_recorded": True,
        "all_active_selector_round_repetition_identity_memberships_match_stage3c1": bool(
            selection_comparison.loc[
                selection_comparison["strategy"].isin(
                    ["PCBM_PROPOSED", "GLOBAL_MARGIN"]
                )
                & selection_comparison[
                    "selected_round_repetition_identity_membership_matches"
                ].notna(),
                "selected_round_repetition_identity_membership_matches",
            ].fillna(False).all()
        ),
        "frozen_prediction_count_is_9800": len(frozen_prediction) == 9800,
        "prediction_join_is_complete": bool(prediction_comparison["_merge"].eq("both").all()),
        "all_true_labels_and_class_positions_match_stage3c1": bool(
            prediction_comparison["_merge"].eq("both").all()
        ),
        "all_predicted_labels_match_stage3c1": bool(prediction_comparison["predicted_label_matches"].all()),
        "frozen_fold_count_is_280": len(frozen_fold) == 280,
        "fold_join_is_complete": bool(fold_comparison["_merge"].eq("both").all()),
        "maximum_fold_metric_difference_is_below_1e_12": bool(fold_comparison["absolute_difference"].max() < 1e-12),
        "all_normalizers_are_finite": bool(normalizers[["all_means_finite", "all_stds_finite"]].all().all()),
        "all_normalizer_stds_are_positive": bool(normalizers["all_stds_positive"].all()),
        "all_normalizer_valid_counts_are_positive": bool((normalizers["minimum_valid_count"] > 0).all()),
        "all_normalizer_means_are_float32": bool(normalizers["normalizer_mean_dtype"].eq("float32").all()),
        "all_normalizer_stds_are_float32": bool(normalizers["normalizer_std_dtype"].eq("float32").all()),
        "all_training_arrays_are_float32": bool(normalizers["training_array_dtype"].eq("float32").all()),
        "all_model_coefficients_are_float32": bool(normalizers["model_coefficient_dtype"].eq("float32").all()),
        "all_folds_use_locked_numerical_contract": bool(
            normalizers["numerical_engine_contract"].eq(NUMERICAL_ENGINE_CONTRACT).all()
        ),
        "random_frozen_selection_trace_is_available_for_r3b": len(random_selection_raw) > 0,
        "raw_hdf5_data_was_not_accessed": True,
        "no_new_reviewer_experiment_was_run": True,
        "no_new_statistical_test_was_run": True,
        "stage3g_and_stage5f_conclusions_cannot_be_replaced": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [name for name, value in readiness_gates.items() if not bool(value)]
    if failed:
        failure_decision = (
            "HOLD_FOR_REVISION_R3A_P1_FLOAT32_RECONSTRUCTION_MISMATCH_DIAGNOSTIC"
        )
        atomic_json(
            {
                "stage": "REVISION_R3A_P1_FLOAT32_ENGINE_FROZEN_TRAJECTORY_UNIT_TEST",
                "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
                "r3a_d3_packet_sha256": R3A_D3_PACKET_SHA256,
                "numerical_engine_contract": NUMERICAL_ENGINE_CONTRACT,
                "readiness_gates": readiness_gates,
                "failed_gates": failed,
                "all_readiness_gates_passed": False,
                "final_decision": failure_decision,
            },
            RESULT_ROOT / "revision_R3A_P1_failure_report.json",
        )
        persist_executed_source(RESULT_ROOT / "revision_R3A_P1_executed_source.py")
        failure_manifest = []
        for path in sorted(RESULT_ROOT.rglob("*")):
            if path.is_file():
                failure_manifest.append(
                    {
                        "relative_path": path.relative_to(RESULT_ROOT).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        atomic_csv(
            pd.DataFrame(failure_manifest),
            RESULT_ROOT / "revision_R3A_P1_output_sha256_manifest.csv",
        )
        packet_crc = make_zip(
            RESULT_ROOT,
            PACKET_PATH,
            "Revision_R3A_P1_Float32_Engine_Frozen_Trajectory_Unit_Test",
        )
        packet_hash = sha256_file(PACKET_PATH)
        remote_verified = roundtrip_remote_file(
            PACKET_PATH, REMOTE_OUTPUT + "/" + PACKET_PATH.name, packet_hash
        )
        print()
        print("R3A-P1 mismatch evidence was preserved before stopping.")
        print("Failed readiness gates:", failed)
        print("Packet CRC pass:", packet_crc)
        print("Packet:", PACKET_PATH)
        print("Packet SHA-256:", packet_hash)
        print("Remote round-trip verified:", remote_verified)
        print()
        print("FINAL DECISION:", failure_decision)
        return

    source_mode = persist_executed_source(
        RESULT_ROOT / "revision_R3A_P1_executed_source.py"
    )
    report = {
        "stage": "REVISION_R3A_P1_FLOAT32_ENGINE_FROZEN_TRAJECTORY_UNIT_TEST",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "r3a_d3_packet_sha256": R3A_D3_PACKET_SHA256,
        "r2a_packet_sha256": R2A_PACKET_SHA256,
        "numerical_engine_contract": NUMERICAL_ENGINE_CONTRACT,
        "r3a_d3_best_configuration": d3_best,
        "reconstructed_trajectories": 56,
        "reconstructed_folds": 280,
        "reconstructed_predictions": 9800,
        "reconstructed_selections": 4165,
        "maximum_fold_metric_difference": float(fold_comparison["absolute_difference"].max()),
        "source_capture_method": source_mode,
        "readiness_gates": readiness_gates,
        "all_readiness_gates_passed": True,
        "raw_hdf5_data_accessed": False,
        "new_reviewer_experiment_performed": False,
        "new_statistical_tests_performed": False,
        "runtime_minutes": round((time.time() - START_TIME) / 60, 3),
        "final_decision": "PASS_TO_REVISION_R3B_NEW_SELECTOR_IMPLEMENTATION_AND_UNIT_TESTS",
    }
    atomic_json(report, RESULT_ROOT / "revision_R3A_P1_float32_reconstruction_report.json")
    manifest_rows = []
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(RESULT_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_csv(
        pd.DataFrame(manifest_rows),
        RESULT_ROOT / "revision_R3A_P1_output_sha256_manifest.csv",
    )
    packet_crc = make_zip(
        RESULT_ROOT,
        PACKET_PATH,
        "Revision_R3A_P1_Float32_Engine_Frozen_Trajectory_Unit_Test",
    )
    packet_hash = sha256_file(PACKET_PATH)
    remote_verified = roundtrip_remote_file(
        PACKET_PATH, REMOTE_OUTPUT + "/" + PACKET_PATH.name, packet_hash
    )
    if not remote_verified:
        raise RuntimeError("Revision R3A Drive round-trip verification failed")

    print()
    print("=" * 92)
    print("REVISION R3A-P1 — FLOAT32 RECONSTRUCTION SUMMARY")
    print("=" * 92)
    print("Reconstructed trajectories: 56")
    print("Reconstructed folds: 280")
    print("Reconstructed repetition predictions: 9800")
    print("Reconstructed selections: 4165")
    print("Maximum fold metric difference:", report["maximum_fold_metric_difference"])
    print("Readiness gates:")
    for name, value in readiness_gates.items():
        print(f"  {name}: {value}")
    print()
    print("Packet CRC pass:", packet_crc)
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", packet_hash)
    print("Remote round-trip verified:", remote_verified)
    print("Runtime minutes:", report["runtime_minutes"])
    print()
    print("FINAL DECISION: PASS_TO_REVISION_R3B_NEW_SELECTOR_IMPLEMENTATION_AND_UNIT_TESTS")


if __name__ == "__main__":
    main()
