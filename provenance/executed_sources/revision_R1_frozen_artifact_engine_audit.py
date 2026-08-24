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
import textwrap
import time
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/revision_r1_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# REVISION R1 — FROZEN ARTIFACT AND ENGINE AUDIT
# =============================================================================

REVISION_PROTOCOL_NAME = "DELTA_PCBM_REVIEWER_REQUESTED_REVISION_v1"
REVISION_PROTOCOL_SHA256 = (
    "6807b71de18ca82013cfa4360d760e0daf9a920a1acc0625dcb13bd8f4d07249"
)
PARENT_PROTOCOL_SHA256 = (
    "f548b1ca6f2831c29ea8fecb764557efed49f229eb72322f98632edcf0aeb221"
)
STAGE3G_FREEZE_SHA256 = (
    "dfcdbbade7bf1032c3250626439689e08e95249ec49d5fb90465c4444a998fbe"
)
DEEP_PROTOCOL_SHA256 = (
    "abe15812c1a52b0f4e917b5b6ad39b0dfde50e5bb2d58dfcc35b3cacb22e3bd2"
)
FROZEN_STAGE3G_CONCLUSION = (
    "PCBM increased low-budget acquisition diversity, but did not demonstrate "
    "robust predictive or retention superiority."
)
FROZEN_DEEP_CONCLUSION = "DEEP_EXTENSION_PRIMARY_HYPOTHESIS_NOT_SUPPORTED"

# Frozen DELTA cohort used throughout Stages 3 and 5. P01–P06 form the
# inferential population and P07 remains a descriptive case analysis, but all
# seven participants must have one LOSO pretrained checkpoint in Stage 5C.
PARTICIPANTS = [f"P{index:02d}" for index in range(1, 8)]

KNOWN_PACKET_SHA256 = {
    "stage5a4b_deep_protocol_lock_packet.zip": "46d4b99b4ee0a222b3facca2aff99dc4b8242afa249a0fc398bacf184c4ca4b4",
    "stage5b_deep_sequence_assembly_packet.zip": "1c0fbc63f6412362f3ae7cd22609ea6a7fcb23236cdf688ad5fe0578ebaab84d",
    "stage5c1_dual_gpu_loso_pretraining_packet.zip": "85ea2e8a8440369a77d43f00b5d509ea2f2978d2a60ab2f24fb828ce9ca6b9d4",
    "stage5d1_deterministic_engine_unit_test_packet.zip": "64e505b15225ad92ac647c33d63f96b40ce0db0cdca75630faefc4b843e10de6",
    "stage5d2_full_deterministic_deep_trajectories_packet.zip": "fc8ac364bac0344639a50977d5f8725b1e5b5b2875758e01587de8c083a1f914",
    "stage5e_30_seed_deep_random_trajectories_packet.zip": "7277a2847ee5f8c07554a155ff1c9bf7ef6e967b70998bfe3b261276710e5b78",
    "stage5f_deep_statistics_retention_sensitivity_packet.zip": "833ec70085c5cb841d49c2d9523074e25d0a51250a81729442434520a5a48afe",
    "stageR0_reviewer_revision_protocol_lock_packet.zip": "0800e315a29b81934095ba56deaea3f8b6600fd0df13db348d7ea72d3b82df78",
}

CORE_PACKETS = [
    "stage1a_index_audit_packet.zip",
    "stage1c2_full_rms_audit_packet.zip",
    "stage3a_v1_1_protocol_amendment_packet.zip",
    "stage3g_final_results_freeze_packet.zip",
    "stage5a4b_deep_protocol_lock_packet.zip",
    "stage5b_deep_sequence_assembly_packet.zip",
    "stage5c1_dual_gpu_loso_pretraining_packet.zip",
    "stage5d1_deterministic_engine_unit_test_packet.zip",
    "stage5d2_full_deterministic_deep_trajectories_packet.zip",
    "stage5e_30_seed_deep_random_trajectories_packet.zip",
    "stage5f_deep_statistics_retention_sensitivity_packet.zip",
    "stageR0_reviewer_revision_protocol_lock_packet.zip",
]

REVISION_DETAIL_PACKETS = [
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

REPORT_BASENAMES = {
    "stage3g_final_results_freeze_packet.zip": "stage3g_report.json",
    "stage5a4b_deep_protocol_lock_packet.zip": "stage5a4b_protocol_lock_report.json",
    "stage5b_deep_sequence_assembly_packet.zip": "stage5b_sequence_assembly_report.json",
    "stage5c1_dual_gpu_loso_pretraining_packet.zip": "stage5c1_loso_pretraining_report.json",
    "stage5d1_deterministic_engine_unit_test_packet.zip": "stage5d1_unit_test_report.json",
    "stage5d2_full_deterministic_deep_trajectories_packet.zip": "stage5d2_full_deterministic_report.json",
    "stage5e_30_seed_deep_random_trajectories_packet.zip": "stage5e_random_trajectory_report.json",
    "stage5f_deep_statistics_retention_sensitivity_packet.zip": "stage5f_deep_analysis_report.json",
    "stageR0_reviewer_revision_protocol_lock_packet.zip": "stageR0_protocol_lock_report.json",
}

WORKING = Path(os.environ.get("REVISION_R1_WORKING", "/kaggle/working"))
TOOLS = WORKING / "_stage5_tools"
TOOLS.mkdir(parents=True, exist_ok=True)
RCLONE = TOOLS / "rclone"
INPUT_ROOT = WORKING / "REVISION_R1_FROZEN_INPUTS"
RESULT_ROOT = (
    WORKING
    / "DELTA_REVIEWER_REVISION"
    / "Revision_R1_Frozen_Artifact_Engine_Audit"
)
CACHE_ROOT = WORKING / "REVISION_R1_AUDIT_CACHE"
for directory in [INPUT_ROOT, RESULT_ROOT, CACHE_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

PACKET_PATH = WORKING / "revision_R1_frozen_artifact_engine_audit_packet.zip"
REMOTE_BASE = "gdrive_stage5:DELTA_Q1_Stage5_DeepLearning_Backup"
REMOTE_OUTPUT = (
    REMOTE_BASE
    + "/Reviewer_Revision/Revision_R1_Frozen_Artifact_Engine_Audit"
)

CONFIG_PATH = None
REMOTE_LISTING = None
START_TIME = time.time()


# =============================================================================
# UTILITIES
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


def canonical_hash(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def atomic_json(payload, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def atomic_csv(dataframe, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def archive_crc_passes(path):
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def archive_member_matches(packet, basename):
    with zipfile.ZipFile(packet, "r") as archive:
        return [name for name in archive.namelist() if Path(name).name == basename]


def archive_member(packet, basename):
    matches = archive_member_matches(packet, basename)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {basename} in {packet}; found {matches}")
    with zipfile.ZipFile(packet, "r") as archive:
        return archive.read(matches[0])


def read_json_member(packet, basename):
    return json.loads(archive_member(packet, basename).decode("utf-8"))


def archive_text(packet):
    chunks = []
    with zipfile.ZipFile(packet, "r") as archive:
        for name in archive.namelist():
            if Path(name).suffix.lower() in {".json", ".csv", ".txt", ".md", ".py"}:
                chunks.append(archive.read(name).decode("utf-8", errors="ignore"))
    return "\n".join(chunks)


def extract_member(packet, basename, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matches = archive_member_matches(packet, basename)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {basename} in {packet}; found {matches}")
    with zipfile.ZipFile(packet, "r") as archive:
        with archive.open(matches[0]) as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target)
    return destination


def make_zip(source_directory, destination, archive_root):
    if Path(destination).exists():
        Path(destination).unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(source_directory).rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=(
                        Path(archive_root) / path.relative_to(source_directory)
                    ).as_posix(),
                )
    return archive_crc_passes(destination)


def persist_executed_source(destination):
    source_name = globals().get("__file__")
    if source_name and Path(source_name).is_file():
        shutil.copy2(Path(source_name), destination)
        return "PYTHON_FILE"
    cell_source = ""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        history = getattr(shell.history_manager, "input_hist_raw", [])
        if history:
            cell_source = str(history[-1])
    except Exception:
        cell_source = ""
    if not cell_source.strip():
        raise RuntimeError("Could not capture Revision R1 executed source")
    Path(destination).write_text(cell_source, encoding="utf-8")
    return "IPYTHON_CELL"


# =============================================================================
# RESTRICTED DRIVE
# =============================================================================


def cleanup_secret():
    global CONFIG_PATH
    if CONFIG_PATH is not None and Path(CONFIG_PATH).exists():
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
    ).read().decode("utf-8").strip()
    match = re.search(r"v?(\d+\.\d+\.\d+)", version_text)
    if match is None:
        raise RuntimeError("Could not resolve official rclone version")
    version = match.group(1)
    archive_name = f"rclone-v{version}-linux-amd64.zip"
    base_url = f"https://downloads.rclone.org/v{version}"
    temporary_root = Path(tempfile.mkdtemp(prefix="revision_r1_rclone_", dir="/tmp"))
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
    temporary = tempfile.NamedTemporaryFile(
        prefix="revision_r1_", suffix=".conf", dir="/tmp", delete=False
    )
    temporary.write(decoded)
    temporary.flush()
    temporary.close()
    os.chmod(temporary.name, 0o600)
    CONFIG_PATH = Path(temporary.name)
    parser = configparser.ConfigParser()
    parser.read_string(decoded.decode("utf-8"))
    if not parser.has_section("gdrive_stage5"):
        raise RuntimeError("gdrive_stage5 remote is missing")
    if parser.get("gdrive_stage5", "type", fallback="") != "drive":
        raise RuntimeError("gdrive_stage5 is not a Drive remote")
    if parser.get("gdrive_stage5", "scope", fallback="") != "drive.file":
        raise RuntimeError("Drive scope is not restricted to drive.file")


def rclone(arguments, capture=True, check=True):
    return subprocess.run(
        [str(RCLONE), "--config", str(CONFIG_PATH)] + list(arguments),
        check=check,
        capture_output=capture,
        text=True,
    )


def get_remote_listing():
    global REMOTE_LISTING
    if REMOTE_LISTING is None:
        REMOTE_LISTING = rclone(
            ["lsf", REMOTE_BASE, "--recursive", "--files-only"]
        ).stdout.splitlines()
    return REMOTE_LISTING


# =============================================================================
# PACKET DISCOVERY AND VERIFICATION
# =============================================================================


def local_packet_candidates(basename):
    candidates = []
    roots = [Path("/kaggle/input"), WORKING]
    target_names = {basename, basename + ".bin"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name in target_names:
                if INPUT_ROOT not in path.parents and RESULT_ROOT not in path.parents:
                    candidates.append(path)
    return sorted(set(candidates), key=lambda item: str(item))


def remote_packet_candidates(basename):
    return sorted(
        path for path in get_remote_listing() if Path(path).name == basename
    )


def choose_remote(matches):
    priorities = [
        "Reviewer_Revision/",
        "Evidence/",
        "Deep_Training/",
        "Deep_Analysis/",
        "Stage5",
    ]
    for prefix in priorities:
        selected = [path for path in matches if path.startswith(prefix)]
        if selected:
            return sorted(selected)[0]
    return sorted(matches)[0]


def resolve_packet(basename, required):
    destination = INPUT_ROOT / basename
    if destination.exists() and destination.stat().st_size > 0:
        return destination, "EXISTING_WORKING_COPY", str(destination)
    local_matches = local_packet_candidates(basename)
    if local_matches:
        shutil.copy2(local_matches[0], destination)
        return destination, "KAGGLE_INPUT_OR_WORKING", str(local_matches[0])
    remote_matches = remote_packet_candidates(basename)
    if remote_matches:
        selected = choose_remote(remote_matches)
        rclone(
            [
                "copyto",
                REMOTE_BASE + "/" + selected,
                str(destination),
                "--retries",
                "5",
                "--low-level-retries",
                "10",
                "--timeout",
                "5m",
            ]
        )
        return destination, "GOOGLE_DRIVE", selected
    if required:
        raise FileNotFoundError(f"Required frozen packet is unavailable: {basename}")
    return None, "MISSING", ""


def extract_stage3g_hash_map(stage3g_packet):
    mapping = {}
    with zipfile.ZipFile(stage3g_packet, "r") as archive:
        for member in archive.namelist():
            suffix = Path(member).suffix.lower()
            if suffix == ".csv":
                try:
                    frame = pd.read_csv(io.BytesIO(archive.read(member)))
                except Exception:
                    continue
                for filename_column in frame.columns:
                    values = frame[filename_column].astype(str)
                    if not values.str.contains(r"\.zip(?:\.bin)?$", regex=True).any():
                        continue
                    for hash_column in frame.columns:
                        hashes = frame[hash_column].astype(str).str.lower()
                        if not hashes.str.fullmatch(r"[0-9a-f]{64}").any():
                            continue
                        for filename, digest in zip(values, hashes):
                            filename = Path(filename).name.removesuffix(".bin")
                            if filename.endswith(".zip") and re.fullmatch(r"[0-9a-f]{64}", digest):
                                mapping[filename] = digest
            elif suffix == ".json":
                text = archive.read(member).decode("utf-8", errors="ignore")
                for filename, digest in re.findall(
                    r'([A-Za-z0-9_\-]+\.zip)(?:[^0-9a-fA-F]{1,100})([0-9a-fA-F]{64})',
                    text,
                ):
                    mapping.setdefault(filename, digest.lower())
    return mapping


def report_all_gates_passed(packet, basename):
    matches = archive_member_matches(packet, basename)
    if len(matches) != 1:
        return False, f"REPORT_MATCH_COUNT_{len(matches)}"
    report = read_json_member(packet, basename)
    if "all_readiness_gates_passed" in report:
        return bool(report["all_readiness_gates_passed"]), "DIRECT_FLAG"
    gates = report.get("readiness_gates") or report.get("gates")
    if isinstance(gates, dict):
        return bool(gates) and all(bool(value) for value in gates.values()), "GATE_DICTIONARY"
    return False, "NO_GATE_FIELD"


def packet_member_inventory(packet, packet_name):
    rows = []
    with zipfile.ZipFile(packet, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            rows.append(
                {
                    "packet": packet_name,
                    "member": info.filename,
                    "basename": Path(info.filename).name,
                    "suffix": Path(info.filename).suffix.lower(),
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                }
            )
    return rows


def audit_packets():
    resolved = {}
    discovery_rows = []
    for basename in CORE_PACKETS:
        path, source, source_path = resolve_packet(basename, required=True)
        resolved[basename] = path
        discovery_rows.append(
            {
                "packet": basename,
                "role": "CORE_REQUIRED",
                "available": True,
                "source": source,
                "source_path": source_path,
            }
        )
    for basename in REVISION_DETAIL_PACKETS:
        path, source, source_path = resolve_packet(basename, required=False)
        if path is not None:
            resolved[basename] = path
        discovery_rows.append(
            {
                "packet": basename,
                "role": "REVISION_DETAIL_REQUIRED_BEFORE_R3_OR_R7",
                "available": path is not None,
                "source": source,
                "source_path": source_path,
            }
        )

    stage3g = resolved["stage3g_final_results_freeze_packet.zip"]
    stage3g_hashes = extract_stage3g_hash_map(stage3g)
    for name in [
        "stage1a_index_audit_packet.zip",
        "stage1c2_full_rms_audit_packet.zip",
        "stage3a_v1_1_protocol_amendment_packet.zip",
    ]:
        if name not in stage3g_hashes:
            raise RuntimeError(f"Stage 3G did not expose a hash for {name}")

    integrity_rows = []
    member_rows = []
    report_rows = []
    for basename, path in resolved.items():
        observed_hash = sha256_file(path)
        expected_hash = KNOWN_PACKET_SHA256.get(basename) or stage3g_hashes.get(basename)
        hash_status = (
            observed_hash == expected_hash if expected_hash is not None else None
        )
        crc_status = archive_crc_passes(path)
        integrity_rows.append(
            {
                "packet": basename,
                "role": "CORE_REQUIRED" if basename in CORE_PACKETS else "REVISION_DETAIL",
                "bytes": path.stat().st_size,
                "crc_passes": crc_status,
                "observed_sha256": observed_hash,
                "expected_sha256": expected_hash or "INVENTORY_ONLY_NO_PARENT_HASH",
                "sha256_matches_expected": hash_status,
            }
        )
        if not crc_status:
            raise RuntimeError(f"CRC failure: {basename}")
        if expected_hash is not None and not hash_status:
            raise RuntimeError(f"SHA-256 mismatch: {basename}")
        member_rows.extend(packet_member_inventory(path, basename))
        report_basename = REPORT_BASENAMES.get(basename)
        if report_basename:
            passed, method = report_all_gates_passed(path, report_basename)
            report_rows.append(
                {
                    "packet": basename,
                    "report_basename": report_basename,
                    "report_resolved_once": len(archive_member_matches(path, report_basename)) == 1,
                    "all_parent_gates_passed": passed,
                    "gate_resolution_method": method,
                }
            )
            if not passed:
                raise RuntimeError(f"Parent report did not pass: {basename}")
    return (
        resolved,
        pd.DataFrame(discovery_rows),
        pd.DataFrame(integrity_rows),
        pd.DataFrame(member_rows),
        pd.DataFrame(report_rows),
        stage3g_hashes,
    )


# =============================================================================
# ENGINE AND ARTIFACT CAPABILITY AUDIT
# =============================================================================


def find_source_members(resolved):
    rows = []
    source_texts = {}
    for packet_name, packet in resolved.items():
        with zipfile.ZipFile(packet, "r") as archive:
            for member in archive.namelist():
                if Path(member).suffix.lower() != ".py":
                    continue
                content = archive.read(member)
                digest = hashlib.sha256(content).hexdigest()
                text = content.decode("utf-8", errors="ignore")
                key = f"{packet_name}|{Path(member).name}"
                source_texts[key] = text
                rows.append(
                    {
                        "packet": packet_name,
                        "member": member,
                        "basename": Path(member).name,
                        "bytes": len(content),
                        "lines": len(text.splitlines()),
                        "sha256": digest,
                    }
                )
    return pd.DataFrame(rows), source_texts


def combined_source(source_texts, packet_filter):
    return "\n".join(
        text
        for key, text in source_texts.items()
        if any(token in key for token in packet_filter)
    )


def audit_stage5b_contract(stage5b_packet):
    basenames = [
        "stage5b_rms_repetition_sequences.npy",
        "stage5b_main_valid_repetition_sequences.npy",
        "stage5b_strict_valid_repetition_sequences.npy",
        "stage5b_repetition_metadata.csv",
        "stage5b_mask_aware_rms_tcn.py",
        "stage5b_sequence_assembly_report.json",
    ]
    rows = []
    for basename in basenames:
        matches = archive_member_matches(stage5b_packet, basename)
        rows.append(
            {
                "artifact": basename,
                "match_count": len(matches),
                "available_exactly_once": len(matches) == 1,
            }
        )
    required = basenames[:2] + basenames[3:]
    if not all(
        len(archive_member_matches(stage5b_packet, basename)) == 1
        for basename in required
    ):
        raise RuntimeError("Stage 5B minimum artifact contract is incomplete")

    metadata_bytes = archive_member(stage5b_packet, "stage5b_repetition_metadata.csv")
    metadata_header = pd.read_csv(io.BytesIO(metadata_bytes), nrows=0)
    required_columns = {
        "sequence_row",
        "participant",
        "session",
        "label",
        "repetition",
        "acquisition_order",
    }
    metadata_columns = set(metadata_header.columns)

    shape_rows = []
    for basename in [
        "stage5b_rms_repetition_sequences.npy",
        "stage5b_main_valid_repetition_sequences.npy",
    ]:
        destination = CACHE_ROOT / basename
        extract_member(stage5b_packet, basename, destination)
        array = np.load(destination, mmap_mode="r")
        shape_rows.append(
            {
                "artifact": basename,
                "shape": "x".join(str(value) for value in array.shape),
                "dtype": str(array.dtype),
                "payload_values_read": False,
            }
        )
        del array
        destination.unlink()

    return (
        pd.DataFrame(rows),
        pd.DataFrame(shape_rows),
        sorted(metadata_header.columns.tolist()),
        required_columns.issubset(metadata_columns),
    )


def build_capability_matrix(resolved, source_texts, metadata_columns):
    deep_det = combined_source(source_texts, ["stage5d1_", "stage5d2_"])
    deep_random = combined_source(source_texts, ["stage5e_"])
    deep_model = combined_source(source_texts, ["stage5b_"])
    classical = combined_source(
        source_texts,
        ["stage3c1_", "stage3c2_", "stage3e2a_", "stage3e2b_", "stage3e2c_"],
    )
    stage3a_packet = resolved["stage3a_v1_1_protocol_amendment_packet.zip"]
    checks = [
        (
            "OPAQUE_SELECTOR_SCHEMA",
            len(
                archive_member_matches(
                    stage3a_packet,
                    "stage3a_v1_1_selector_schema.json",
                )
            )
            == 1,
            "REUSE_LOCKED",
        ),
        ("PCBM_SELECTOR", "PCBM_PROPOSED" in deep_det and "predicted_label" in deep_det and "margin" in deep_det, "REUSE_AND_GENERALIZE"),
        ("GLOBAL_MARGIN_SELECTOR", "GLOBAL_MARGIN" in deep_det, "REUSE"),
        ("RANDOM_SELECTOR", "RANDOM_UNIFORM" in deep_random, "REUSE"),
        ("HISTORY_ONLY_NORMALIZATION", "normalizer" in deep_det.lower() and "history" in deep_det.lower(), "REUSE_WITH_ANCHOR_TESTS"),
        ("FIXED_TEST_EVALUATION", "fixed_test" in deep_det.lower(), "REUSE"),
        ("FUTURE_SESSION_GUARD", "future_session" in deep_det.lower(), "REUSE"),
        ("TCN_MODEL_SOURCE", "Conv1d" in deep_model and "BatchNorm" not in deep_model, "REUSE"),
        ("TCN_RESIDUAL_BLOCKS", "blocks" in deep_model and "dilation" in deep_model, "REUSE_AND_DOCUMENT"),
        ("ALTERNATIVE_SPLIT_METADATA", {"repetition", "acquisition_order"}.issubset(set(metadata_columns)), "NEW_SPLIT_SCHEDULER"),
        ("CLASSICAL_ENGINE_SOURCE", len(classical) > 0, "REUSE_IF_AVAILABLE_ELSE_MIGRATE"),
        ("RIDGE_PROBABILITY_CALIBRATOR", "CalibratedClassifier" in classical or "calibration" in classical.lower(), "NEW_IMPLEMENTATION_REQUIRED"),
        ("ENTROPY_SELECTOR", "entropy" in classical.lower(), "NEW_IMPLEMENTATION_REQUIRED"),
        ("LEAST_CONFIDENCE_SELECTOR", "least_confidence" in classical.lower(), "NEW_IMPLEMENTATION_REQUIRED"),
        ("RBMAL_SELECTOR", "rbmal" in classical.lower() or "ranked batch" in classical.lower(), "NEW_IMPLEMENTATION_REQUIRED"),
        ("CORE_SET_SELECTOR", "core_set" in classical.lower() or "k-center" in classical.lower(), "NEW_IMPLEMENTATION_REQUIRED"),
        ("BADGE_SELECTOR", "badge" in deep_det.lower(), "NEW_IMPLEMENTATION_REQUIRED"),
        ("COMPUTE_TELEMETRY", "peak_memory" in deep_det.lower() and "latency" in deep_det.lower(), "NEW_IMPLEMENTATION_REQUIRED"),
    ]
    return pd.DataFrame(
        checks,
        columns=["capability", "currently_available", "revision_action"],
    )


def build_reuse_matrix():
    rows = [
        ("DATA_SEQUENCE", "Stage5B RMS sequence and masks", "Reuse unchanged", "R3-R6"),
        ("METADATA_JOIN", "Stage5B composite repetition metadata", "Reuse and add locked split roles", "R3-R6"),
        ("OPAQUE_TOKENS", "Stage3A v1.1 schema and hashing", "Reuse unchanged", "R3-R6"),
        ("PCBM_GLOBAL", "Stage5D1/5D2 deterministic selectors", "Reuse exact rules", "R3-R6"),
        ("RANDOM", "Stage5E random selector and seed handling", "Reuse with R0 seeds", "R3-R6"),
        ("TCN", "Stage5B model plus Stage5C pretrained checkpoints", "Reuse architecture/checkpoints", "R6"),
        ("NORMALIZATION", "Stage5D history-only masked transform", "Reuse and anchor", "R3-R6"),
        ("FIXED_TEST", "Stage5D evaluation contract", "Reuse without model selection", "R3-R6"),
        ("RETENTION", "Stage3F and Stage5F definitions", "Reuse for supplement; no new primary role", "R7"),
        ("STATISTICS", "Exact sign-enumeration and seeded bootstrap", "Expose details; add BCa sensitivity", "R7"),
    ]
    return pd.DataFrame(
        rows,
        columns=["component", "frozen_source", "locked_revision_use", "stages"],
    )


def build_new_implementation_matrix():
    rows = [
        ("RIDGE_OOF_CALIBRATION", "Five-fold source-history repetition-level calibration", "R3", "CPU"),
        ("LEAST_CONFIDENCE", "Probability-based selector with opaque schema wrapper", "R3", "CPU/GPU"),
        ("PREDICTIVE_ENTROPY", "Probability entropy selector with opaque schema wrapper", "R3", "CPU/GPU"),
        ("RBMAL", "Locked 0.5 uncertainty + 0.5 Euclidean novelty", "R3", "CPU/GPU"),
        ("CORE_SET", "Greedy k-center on source-normalized repetition embeddings", "R3", "CPU/GPU"),
        ("IMBALANCE_BUILDER", "R0 count patterns, rotations, and five subset seeds", "R4", "CPU"),
        ("TEMPORAL_SPLIT_SCHEDULER", "Four disjoint candidate/test schedules", "R5", "CPU"),
        ("DRIFT_AUDIT", "Acquisition-order feature and performance drift", "R5", "CPU"),
        ("BADGE", "TCN last-layer gradient embeddings and k-means++", "R6", "GPU"),
        ("TCN_MULTISEED", "Fixed-history and end-to-end six-seed orchestration", "R6", "2xT4"),
        ("COST_TELEMETRY", "Selection/refit/wall-time/RAM/VRAM measurement", "R3-R6", "CPU/GPU"),
        ("MC_CONVERGENCE", "Random seed-prefix/subsample convergence and extension gate", "R7", "CPU"),
        ("BCa_INTERVALS", "Participant-level BCa with percentile sensitivity", "R7", "CPU"),
        ("FULL_SUPPLEMENT", "Participant/session/class/retention/sensitivity exports", "R7", "CPU"),
    ]
    return pd.DataFrame(
        rows,
        columns=["implementation_id", "locked_scope", "first_stage", "compute"],
    )


def make_pdf(report, discovery, integrity, capability, missing_detail, destination):
    def page(pdf, title, lines):
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        fig.text(0.07, 0.95, title, fontsize=15, weight="bold", va="top")
        y = 0.91
        for line in lines:
            for part in textwrap.wrap(str(line), width=105) or [""]:
                fig.text(0.07, y, part, fontsize=8.3, va="top")
                y -= 0.019
                if y < 0.06:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig = plt.figure(figsize=(8.5, 11))
                    plt.axis("off")
                    y = 0.95
            y -= 0.004
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(destination) as pdf:
        page(
            pdf,
            "Revision R1 — Frozen Artifact and Engine Audit",
            [
                f"Revision protocol SHA-256: {REVISION_PROTOCOL_SHA256}",
                f"Core packets resolved: {int((discovery.role == 'CORE_REQUIRED').sum())}",
                f"Revision-detail packets available: {int(((discovery.role != 'CORE_REQUIRED') & discovery.available).sum())}/{len(REVISION_DETAIL_PACKETS)}",
                f"Missing detail packets: {', '.join(missing_detail) if missing_detail else 'None'}",
                "No raw HDF5 file was opened. No model was trained and no test prediction was generated.",
                f"Decision: {report['final_decision']}",
            ],
        )
        page(
            pdf,
            "Packet Integrity",
            [
                f"{row.packet}: CRC={row.crc_passes}, SHA match={row.sha256_matches_expected}, MB={row.bytes/1e6:.2f}"
                for row in integrity.itertuples(index=False)
            ],
        )
        page(
            pdf,
            "Engine Capabilities",
            [
                f"{row.capability}: available={row.currently_available}; action={row.revision_action}"
                for row in capability.itertuples(index=False)
            ],
        )
        page(
            pdf,
            "Readiness Gates",
            [f"{key}: {value}" for key, value in report["readiness_gates"].items()],
        )


def manifest_for_directory(directory):
    rows = []
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.name != "revision_R1_sha256_manifest.csv":
            rows.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def main():
    print("=" * 88)
    print("REVISION R1 — FROZEN ARTIFACT AND ENGINE AUDIT")
    print("=" * 88)
    print("Execution device: CPU")
    print("Raw HDF5 data accessed: False")
    print("Model training: False")
    print("Test-set inference: False")
    print("New statistical tests: False")
    print()

    bootstrap_rclone()
    create_rclone_config()
    print("rclone version:", rclone(["version"]).stdout.splitlines()[0])
    print("Discovering frozen packets in Kaggle Inputs, working storage, and Drive...")

    (
        resolved,
        discovery,
        integrity,
        member_inventory,
        parent_reports,
        stage3g_hashes,
    ) = audit_packets()

    r0_packet = resolved["stageR0_reviewer_revision_protocol_lock_packet.zip"]
    r0_protocol = read_json_member(r0_packet, "stageR0_locked_revision_protocol.json")
    r0_hash_valid = (
        canonical_hash({k: v for k, v in r0_protocol.items() if k != "protocol_sha256"})
        == r0_protocol["protocol_sha256"]
        == REVISION_PROTOCOL_SHA256
    )

    stage3a_text = archive_text(
        resolved["stage3a_v1_1_protocol_amendment_packet.zip"]
    )
    stage3g_text = archive_text(
        resolved["stage3g_final_results_freeze_packet.zip"]
    )
    stage5a4b_text = archive_text(
        resolved["stage5a4b_deep_protocol_lock_packet.zip"]
    )

    source_inventory, source_texts = find_source_members(resolved)
    (
        stage5b_contract,
        stage5b_shapes,
        metadata_columns,
        metadata_supports_revision,
    ) = audit_stage5b_contract(
        resolved["stage5b_deep_sequence_assembly_packet.zip"]
    )
    capability = build_capability_matrix(resolved, source_texts, metadata_columns)
    reuse = build_reuse_matrix()
    new_implementation = build_new_implementation_matrix()

    stage5c_packet = resolved["stage5c1_dual_gpu_loso_pretraining_packet.zip"]
    with zipfile.ZipFile(stage5c_packet, "r") as archive:
        # Stage 5C stores each checkpoint under a participant directory as
        # ``.../P01/best.pt`` rather than flattening it to ``P01_best.pt``.
        # Audit the frozen archive using that original contract and require
        # exactly one best checkpoint for every participant.
        checkpoint_rows = []
        for participant in PARTICIPANTS:
            suffix = f"/{participant}/best.pt"
            matches = [
                name
                for name in archive.namelist()
                if name.endswith(suffix)
            ]
            checkpoint_rows.append(
                {
                    "participant": participant,
                    "expected_archive_suffix": suffix,
                    "match_count": len(matches),
                    "archive_member": matches[0] if len(matches) == 1 else None,
                    "available_exactly_once": len(matches) == 1,
                }
            )
        checkpoint_audit = pd.DataFrame(checkpoint_rows)
        best_checkpoints = checkpoint_audit.loc[
            checkpoint_audit["available_exactly_once"],
            "archive_member",
        ].tolist()

    detail_discovery = discovery[discovery["role"] != "CORE_REQUIRED"].copy()
    missing_detail = detail_discovery.loc[
        ~detail_discovery["available"], "packet"
    ].tolist()
    available_detail = detail_discovery.loc[
        detail_discovery["available"], "packet"
    ].tolist()

    core_integrity = integrity[integrity["role"] == "CORE_REQUIRED"]
    known_hash_rows = core_integrity[
        core_integrity["expected_sha256"] != "INVENTORY_ONLY_NO_PARENT_HASH"
    ]
    readiness_gates = {
        "revision_r0_protocol_hash_verifies": r0_hash_valid,
        "parent_v11_protocol_hash_is_present": PARENT_PROTOCOL_SHA256 in stage3a_text,
        "stage3g_freeze_hash_is_present": STAGE3G_FREEZE_SHA256 in stage3g_text,
        "deep_protocol_hash_is_present": DEEP_PROTOCOL_SHA256 in stage5a4b_text,
        "all_twelve_core_packets_are_resolved": len(core_integrity) == len(CORE_PACKETS),
        "all_core_packets_pass_crc": bool(core_integrity["crc_passes"].all()),
        "all_core_packets_with_parent_hashes_match": bool(known_hash_rows["sha256_matches_expected"].all()),
        "all_resolved_parent_reports_pass": bool(parent_reports["all_parent_gates_passed"].all()),
        "stage1a_hash_is_anchored_by_stage3g": "stage1a_index_audit_packet.zip" in stage3g_hashes,
        "stage1c2_hash_is_anchored_by_stage3g": "stage1c2_full_rms_audit_packet.zip" in stage3g_hashes,
        "stage3a_hash_is_anchored_by_stage3g": "stage3a_v1_1_protocol_amendment_packet.zip" in stage3g_hashes,
        "stage5b_minimum_artifact_contract_is_complete": bool(stage5b_contract.loc[stage5b_contract.artifact != "stage5b_strict_valid_repetition_sequences.npy", "available_exactly_once"].all()),
        "stage5b_sequence_shape_is_2940_by_37_by_64": set(stage5b_shapes["shape"]) == {"2940x37x64"},
        "metadata_supports_locked_temporal_splits": metadata_supports_revision,
        "seven_pretrained_best_checkpoints_are_available": bool(
            len(best_checkpoints) == 7
            and checkpoint_audit["available_exactly_once"].all()
        ),
        "engine_source_inventory_is_nonempty": len(source_inventory) > 0,
        "opaque_selector_contract_is_available": bool(capability.loc[capability.capability == "OPAQUE_SELECTOR_SCHEMA", "currently_available"].iloc[0]),
        "pcbm_and_global_selectors_are_available": bool(capability.loc[capability.capability.isin(["PCBM_SELECTOR", "GLOBAL_MARGIN_SELECTOR"]), "currently_available"].all()),
        "random_selector_is_available": bool(capability.loc[capability.capability == "RANDOM_SELECTOR", "currently_available"].iloc[0]),
        "history_only_normalization_is_available": bool(capability.loc[capability.capability == "HISTORY_ONLY_NORMALIZATION", "currently_available"].iloc[0]),
        "fixed_test_and_future_session_guards_are_available": bool(capability.loc[capability.capability.isin(["FIXED_TEST_EVALUATION", "FUTURE_SESSION_GUARD"]), "currently_available"].all()),
        "new_comparator_implementations_are_explicitly_identified": len(new_implementation) == 14,
        "missing_revision_detail_packets_are_explicitly_recorded": len(detail_discovery) == len(REVISION_DETAIL_PACKETS),
        "raw_hdf5_data_was_not_opened": True,
        "array_payload_values_were_not_read": bool((stage5b_shapes["payload_values_read"] == False).all()),
        "no_model_was_trained": True,
        "no_test_inference_was_run": True,
        "no_statistical_test_was_run": True,
        "frozen_stage3g_and_stage5f_conclusions_cannot_be_replaced": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    if not all(readiness_gates.values()):
        failed = [key for key, value in readiness_gates.items() if not value]
        raise RuntimeError(f"Revision R1 audit failed readiness gates: {failed}")

    if missing_detail:
        final_decision = (
            "PASS_TO_REVISION_R2_MANUSCRIPT_CORRECTIONS_WITH_"
            "CLASSICAL_DETAIL_PACKET_MIGRATION_REQUIRED_BEFORE_R3_R7"
        )
    else:
        final_decision = "PASS_TO_REVISION_R2_MANUSCRIPT_CORRECTIONS"

    outputs = {
        "revision_R1_packet_discovery.csv": discovery,
        "revision_R1_packet_integrity.csv": integrity,
        "revision_R1_archive_member_inventory.csv": member_inventory,
        "revision_R1_parent_report_audit.csv": parent_reports,
        "revision_R1_source_inventory.csv": source_inventory,
        "revision_R1_stage5b_artifact_contract.csv": stage5b_contract,
        "revision_R1_stage5b_array_header_audit.csv": stage5b_shapes,
        "revision_R1_stage5c_best_checkpoint_audit.csv": checkpoint_audit,
        "revision_R1_engine_capability_matrix.csv": capability,
        "revision_R1_frozen_component_reuse_matrix.csv": reuse,
        "revision_R1_new_implementation_matrix.csv": new_implementation,
    }
    for filename, dataframe in outputs.items():
        atomic_csv(dataframe, RESULT_ROOT / filename)
    atomic_json(
        {
            "metadata_columns": metadata_columns,
            "required_temporal_fields_present": metadata_supports_revision,
        },
        RESULT_ROOT / "revision_R1_metadata_contract.json",
    )
    atomic_json(
        {
            "available_revision_detail_packets": available_detail,
            "missing_revision_detail_packets": missing_detail,
            "migration_required_before": ["R3", "R7"] if missing_detail else [],
        },
        RESULT_ROOT / "revision_R1_detail_packet_migration_status.json",
    )

    report = {
        "stage": "REVISION_R1_FROZEN_ARTIFACT_AND_ENGINE_AUDIT",
        "revision_protocol_name": REVISION_PROTOCOL_NAME,
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "core_packet_count": len(CORE_PACKETS),
        "resolved_core_packet_count": int((discovery.role == "CORE_REQUIRED").sum()),
        "revision_detail_packet_count": len(REVISION_DETAIL_PACKETS),
        "available_revision_detail_packet_count": len(available_detail),
        "missing_revision_detail_packets": missing_detail,
        "source_file_count": len(source_inventory),
        "stage5c_best_checkpoint_count": len(best_checkpoints),
        "raw_hdf5_data_accessed": False,
        "array_payload_values_read": False,
        "model_training_performed": False,
        "test_inference_performed": False,
        "statistical_tests_performed": False,
        "readiness_gates": readiness_gates,
        "all_readiness_gates_passed": all(readiness_gates.values()),
        "final_decision": final_decision,
    }
    atomic_json(report, RESULT_ROOT / "revision_R1_audit_report.json")
    make_pdf(
        report,
        discovery,
        integrity,
        capability,
        missing_detail,
        RESULT_ROOT / "revision_R1_audit_summary.pdf",
    )
    source_mode = persist_executed_source(
        RESULT_ROOT / "revision_R1_executed_source.py"
    )
    report["executed_source_capture_mode"] = source_mode
    atomic_json(report, RESULT_ROOT / "revision_R1_audit_report.json")
    atomic_csv(
        manifest_for_directory(RESULT_ROOT),
        RESULT_ROOT / "revision_R1_sha256_manifest.csv",
    )

    packet_crc = make_zip(
        RESULT_ROOT,
        PACKET_PATH,
        "Revision_R1_Frozen_Artifact_Engine_Audit",
    )
    packet_sha256 = sha256_file(PACKET_PATH)

    print()
    print("=" * 88)
    print("REVISION R1 — AUDIT SUMMARY")
    print("=" * 88)
    print("Core packets resolved:", len(CORE_PACKETS), "/", len(CORE_PACKETS))
    print("Revision-detail packets available:", len(available_detail), "/", len(REVISION_DETAIL_PACKETS))
    print("Missing revision-detail packets:")
    if missing_detail:
        for item in missing_detail:
            print("  ", item)
    else:
        print("  None")
    print("Source files inventoried:", len(source_inventory))
    print("Stage 5C best checkpoints:", len(best_checkpoints))
    print("Stage 5B array shapes:")
    print(stage5b_shapes.to_string(index=False))
    print()
    print("Engine capability matrix:")
    print(capability.to_string(index=False))
    print()
    print("Readiness gates:")
    for key, value in readiness_gates.items():
        print(f"  {key}: {value}")

    print()
    print("Uploading Revision R1 audit and packet to Google Drive...")
    rclone(["mkdir", REMOTE_OUTPUT])
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
    remote_packet = REMOTE_OUTPUT + "/" + PACKET_PATH.name
    rclone(
        [
            "copyto",
            str(PACKET_PATH),
            remote_packet,
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--timeout",
            "5m",
        ]
    )
    roundtrip_packet = WORKING / "_revision_R1_roundtrip_packet.zip"
    if roundtrip_packet.exists():
        roundtrip_packet.unlink()
    rclone(
        [
            "copyto",
            remote_packet,
            str(roundtrip_packet),
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--timeout",
            "5m",
        ]
    )
    roundtrip_hash = sha256_file(roundtrip_packet)
    remote_verified = (
        roundtrip_hash == packet_sha256 and archive_crc_passes(roundtrip_packet)
    )
    roundtrip_packet.unlink()
    if not remote_verified:
        raise RuntimeError("Revision R1 Drive round-trip verification failed")

    verification_path = RESULT_ROOT / "revision_R1_drive_roundtrip_verification.json"
    atomic_json(
        {
            "packet_filename": PACKET_PATH.name,
            "local_sha256": packet_sha256,
            "roundtrip_sha256": roundtrip_hash,
            "roundtrip_sha256_matches": remote_verified,
            "credentials_displayed": False,
        },
        verification_path,
    )
    rclone(
        [
            "copyto",
            str(verification_path),
            REMOTE_OUTPUT + "/" + verification_path.name,
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--timeout",
            "5m",
        ]
    )
    cleanup_secret()

    print()
    print("Packet CRC pass:", packet_crc)
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", packet_sha256)
    print("Remote round-trip verified:", remote_verified)
    print("Runtime minutes:", round((time.time() - START_TIME) / 60.0, 2))
    print()
    print("FINAL DECISION:", final_decision)


if __name__ == "__main__":
    main()
