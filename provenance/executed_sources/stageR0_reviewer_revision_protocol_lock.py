import atexit
import base64
import configparser
import hashlib
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# REVISION R0 — REVIEWER-RESPONSE EXPERIMENT PROTOCOL LOCK
# =============================================================================

REVISION_PROTOCOL_NAME = "DELTA_PCBM_REVIEWER_REQUESTED_REVISION_v1"
PARENT_PROTOCOL_SHA256 = (
    "f548b1ca6f2831c29ea8fecb764557efed49f229eb72322f98632edcf0aeb221"
)
STAGE3G_FREEZE_SHA256 = (
    "dfcdbbade7bf1032c3250626439689e08e95249ec49d5fb90465c4444a998fbe"
)
STAGE5F_PACKET_SHA256 = (
    "833ec70085c5cb841d49c2d9523074e25d0a51250a81729442434520a5a48afe"
)
DEEP_PROTOCOL_SHA256 = (
    "abe15812c1a52b0f4e917b5b6ad39b0dfde50e5bb2d58dfcc35b3cacb22e3bd2"
)
FROZEN_STAGE3G_CONCLUSION = (
    "PCBM increased low-budget acquisition diversity, but did not demonstrate "
    "robust predictive or retention superiority."
)
FROZEN_DEEP_CONCLUSION = "DEEP_EXTENSION_PRIMARY_HYPOTHESIS_NOT_SUPPORTED"

PARTICIPANTS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07"]
ABLE_BODIED = ["P01", "P02", "P03", "P04", "P05", "P06"]
TARGET_SESSIONS = [1, 2, 3, 4, 5]
LABELS = list(range(7))
BUDGETS = [0, 7, 14, 21]

WORKING = Path(os.environ.get("STAGER0_WORKING", "/kaggle/working"))
TOOLS = WORKING / "_stage5_tools"
TOOLS.mkdir(parents=True, exist_ok=True)
RCLONE = TOOLS / "rclone"
INPUT_ROOT = WORKING / "STAGER0_FROZEN_INPUTS"
RESULT_ROOT = (
    WORKING
    / "DELTA_REVIEWER_REVISION"
    / "StageR0_Reviewer_Revision_Protocol_Lock"
)
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

STAGE3G_PACKET = INPUT_ROOT / "stage3g_final_results_freeze_packet.zip"
STAGE5F_PACKET = INPUT_ROOT / "stage5f_deep_statistics_retention_sensitivity_packet.zip"
PACKET_PATH = WORKING / "stageR0_reviewer_revision_protocol_lock_packet.zip"

REMOTE_BASE = "gdrive_stage5:DELTA_Q1_Stage5_DeepLearning_Backup"
REMOTE_OUTPUT = (
    REMOTE_BASE
    + "/Reviewer_Revision/StageR0_Reviewer_Revision_Protocol_Lock"
)

CONFIG_PATH = None
START_TIME = time.time()


# =============================================================================
# GENERAL UTILITIES
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
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(namespace):
    value = int(hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16], 16)
    return int(value % 2_000_000_000 + 1)


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
    with zipfile.ZipFile(path, "r") as archive:
        return archive.testzip() is None


def archive_member(packet, basename):
    with zipfile.ZipFile(packet, "r") as archive:
        matches = [name for name in archive.namelist() if Path(name).name == basename]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {basename} in {packet}; found {matches}"
            )
        return archive.read(matches[0])


def read_json_member(packet, basename):
    return json.loads(archive_member(packet, basename).decode("utf-8"))


def archive_text(packet):
    chunks = []
    with zipfile.ZipFile(packet, "r") as archive:
        for name in archive.namelist():
            if Path(name).suffix.lower() in {".json", ".csv", ".txt", ".md"}:
                chunks.append(
                    archive.read(name).decode("utf-8", errors="ignore")
                )
    return "\n".join(chunks)


def make_zip(source_directory, destination, archive_root):
    source_directory = Path(source_directory)
    destination = Path(destination)
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(source_directory.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=(
                        Path(archive_root) / path.relative_to(source_directory)
                    ).as_posix(),
                )
    return archive_crc_passes(destination)


def persist_executed_source(destination):
    destination = Path(destination)
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
        raise RuntimeError("Could not capture executed Revision R0 source")
    destination.write_text(cell_source, encoding="utf-8")
    return "IPYTHON_CELL"


# =============================================================================
# RESTRICTED GOOGLE DRIVE BRIDGE
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
    temporary_root = Path(tempfile.mkdtemp(prefix="stageR0_rclone_", dir="/tmp"))
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
        prefix="stageR0_", suffix=".conf", dir="/tmp", delete=False
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


def rclone(arguments, capture=True):
    return subprocess.run(
        [str(RCLONE), "--config", str(CONFIG_PATH)] + list(arguments),
        check=True,
        capture_output=capture,
        text=True,
    )


def locate_remote_basename(basename):
    listing = rclone(
        ["lsf", REMOTE_BASE, "--recursive", "--files-only"]
    ).stdout.splitlines()
    matches = sorted(path for path in listing if Path(path).name == basename)
    if not matches:
        raise RuntimeError(f"Could not locate {basename} under {REMOTE_BASE}")
    preferred_prefixes = ["Evidence/", "Deep_Analysis/"]
    selected = None
    for prefix in preferred_prefixes:
        preferred = [path for path in matches if path.startswith(prefix)]
        if preferred:
            selected = preferred[0]
            break
    if selected is None:
        selected = matches[0]
    print(
        f"Remote {basename} candidates: {len(matches)}; selected: {selected}",
        flush=True,
    )
    return REMOTE_BASE + "/" + selected


def restore_and_verify_parents():
    for destination in [STAGE3G_PACKET, STAGE5F_PACKET]:
        remote = locate_remote_basename(destination.name)
        rclone(
            [
                "copyto",
                remote,
                str(destination),
                "--retries",
                "5",
                "--low-level-retries",
                "10",
                "--timeout",
                "5m",
            ]
        )
    stage3g_text = archive_text(STAGE3G_PACKET)
    stage5f_report = read_json_member(
        STAGE5F_PACKET, "stage5f_deep_analysis_report.json"
    )
    gates = {
        "stage3g_packet_crc_passes": archive_crc_passes(STAGE3G_PACKET),
        "stage5f_packet_crc_passes": archive_crc_passes(STAGE5F_PACKET),
        "stage3g_internal_freeze_hash_is_present": (
            STAGE3G_FREEZE_SHA256 in stage3g_text
        ),
        "stage5f_packet_sha256_matches": (
            sha256_file(STAGE5F_PACKET) == STAGE5F_PACKET_SHA256
        ),
        "stage5f_all_readiness_gates_passed": bool(
            stage5f_report["all_readiness_gates_passed"]
        ),
        "deep_protocol_hash_matches": (
            stage5f_report["deep_protocol_sha256"] == DEEP_PROTOCOL_SHA256
        ),
        "stage3g_conclusion_is_preserved": (
            stage5f_report["stage3g_frozen_conclusion"]
            == FROZEN_STAGE3G_CONCLUSION
        ),
        "deep_conclusion_is_preserved": (
            stage5f_report["primary_deep_conclusion"]
            == FROZEN_DEEP_CONCLUSION
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Frozen-parent verification failed: {gates}")
    return stage5f_report, gates


# =============================================================================
# LOCKED REVISION TABLES
# =============================================================================


def build_action_matrix():
    rows = [
        ("NOVELTY_01", "MAJOR", "LITERATURE", "Add and directly compare Szymaniak et al. 2022", "TEXT_AND_TABLE", "R2"),
        ("NOVELTY_02", "MAJOR", "LITERATURE", "Discuss class-balanced active learning and delimit PCBM novelty", "TEXT_AND_TABLE", "R2"),
        ("NOVELTY_03", "MAJOR", "CLAIMS", "Delete first-framework claim; describe PCBM as an acquisition heuristic", "TEXT", "R2"),
        ("EXPERIMENT_01", "MAJOR", "IMBALANCE", "Evaluate four locked candidate-pool imbalance levels", "NEW_EXPERIMENT", "R4"),
        ("EXPERIMENT_02", "MAJOR", "BASELINES", "Add least confidence, entropy, RBMAL, core-set, and BADGE", "NEW_EXPERIMENT", "R3_R6"),
        ("EXPERIMENT_03", "MAJOR", "EFFICIENCY", "Report normalized AULC over K00/K07/K14/K21", "NEW_ANALYSIS", "R7"),
        ("EXPERIMENT_04", "MAJOR", "COST", "Measure selector, refit, wall-clock, GPU, and memory cost", "NEW_ANALYSIS", "R3_R6"),
        ("TEMPORAL_01", "MAJOR", "SPLITS", "Run first/second-half and both odd/even locked splits", "NEW_EXPERIMENT", "R5"),
        ("TEMPORAL_02", "MAJOR", "DRIFT", "Analyze acquisition-order feature and performance drift", "NEW_ANALYSIS", "R5"),
        ("ORACLE_01", "MAJOR", "DEPLOYMENT", "Define offline oracle and annotation/acquisition burden", "TEXT", "R2"),
        ("DEEP_01", "MAJOR", "REPRODUCIBILITY", "Report complete TCN architecture and optimization settings", "TEXT_AND_TABLE", "R2"),
        ("DEEP_02", "MAJOR", "STOCHASTICITY", "Use six pre-existing locked training seeds per participant", "NEW_EXPERIMENT", "R6"),
        ("MATH_01", "MAJOR", "EQUATIONS", "Correct window index, normalization domain, epsilon, and symbols", "TEXT_AND_EQUATIONS", "R2"),
        ("MATH_02", "MAJOR", "EQUATIONS", "Represent K14/K21 as sequential rounds with refitting", "TEXT_AND_EQUATIONS", "R2"),
        ("STATS_01", "MAJOR", "STATISTICS", "Expose Wilcoxon statistic, zero handling, bootstrap type and seed", "TEXT_AND_TABLE", "R7"),
        ("STATS_02", "MAJOR", "STATISTICS", "Quantify random-policy Monte Carlo error and convergence", "NEW_ANALYSIS", "R7"),
        ("STATS_03", "MAJOR", "STATISTICS", "Report participant effects and confidence intervals; prohibit equivalence claims", "TEXT_AND_SUPPLEMENT", "R7_R8"),
        ("RESULTS_01", "MAJOR", "SUPPLEMENT", "Release all 18 retention and five classical secondary tests", "SUPPLEMENT", "R7"),
        ("RESULTS_02", "MAJOR", "SUPPLEMENT", "Add participant/session results, confusion matrices, recall, coverage, entropy", "SUPPLEMENT", "R7"),
        ("REPRO_01", "MAJOR", "REPRODUCIBILITY", "Use internally prespecified and frozen unless public timestamp exists", "TEXT", "R2"),
        ("REPRO_02", "MAJOR", "AVAILABILITY", "Archive protocol, code, and evidence packets with persistent links", "REPOSITORY", "R8"),
        ("FORMAT_01", "MODERATE", "FORMAT", "Renumber tables, figures, equations and keep captions with objects", "DOCUMENT_QA", "R8"),
        ("FORMAT_02", "MODERATE", "FORMAT", "Remove manuscript map, editorial cover text, and internal stage labels", "DOCUMENT_QA", "R8"),
        ("FORMAT_03", "MODERATE", "FORMAT", "Repair figure overlaps, whitespace, reference and footer crowding", "DOCUMENT_QA", "R8"),
        ("JOURNAL_01", "MODERATE", "JOURNAL", "Adopt leakage-controlled title and BSPC highlight limit", "TEXT", "R8"),
        ("JOURNAL_02", "MODERATE", "JOURNAL", "Complete authorship, conflicts, CRediT, code, and AI declaration", "AUTHOR_ACTION", "R8"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "issue_id",
            "priority",
            "domain",
            "locked_response_action",
            "action_type",
            "execution_stage",
        ],
    )


def build_literature_matrix():
    rows = [
        {
            "study_id": "SZYMANIAK_2022",
            "citation": "Szymaniak K, Krasoulis A, Nazarpour K. Recalibration of myoelectric control with active learning. Front Neurorobot. 2022;16:1061201.",
            "doi": "10.3389/fnbot.2022.1061201",
            "direct_relevance": "Closest prior active-learning myoelectric recalibration study",
            "methods_to_compare": "LDA; least confidence; smallest margin; entropy; naive batch; ranked batch; random",
            "required_revision_use": "Direct comparison table; novelty correction; oracle/deployment discussion",
        },
        {
            "study_id": "BENGAR_2022",
            "citation": "Bengar JZ, van de Weijer J, Lopez Fuentes L, Raducanu B. Class-Balanced Active Learning for Image Classification. WACV. 2022.",
            "doi": "10.1109/WACV51458.2022.00376",
            "direct_relevance": "Explicit class balancing under imbalanced active-learning pools",
            "methods_to_compare": "Optimization-based class balance combined with informative or representative acquisition",
            "required_revision_use": "Conceptual and mathematical delimitation from PCBM",
        },
        {
            "study_id": "CARDOSO_2017",
            "citation": "Cardoso TN, Silva RM, Canuto S, Moro MM, Goncalves MA. Ranked batch-mode active learning. Inf Sci. 2017;379:313-337.",
            "doi": "10.1016/j.ins.2016.10.037",
            "direct_relevance": "Uncertainty-diversity batch acquisition used by Szymaniak et al.",
            "methods_to_compare": "Ranked uncertainty plus Euclidean diversity",
            "required_revision_use": "Implement locked RBMAL comparator",
        },
        {
            "study_id": "SENER_SAVARESE_2018",
            "citation": "Sener O, Savarese S. Active Learning for Convolutional Neural Networks: A Core-Set Approach. ICLR. 2018.",
            "doi": "10.48550/arXiv.1708.00489",
            "direct_relevance": "Representative diversity baseline for batch deep active learning",
            "methods_to_compare": "Greedy k-center core-set selection",
            "required_revision_use": "Implement representation-based comparator",
        },
        {
            "study_id": "ASH_2020",
            "citation": "Ash JT, Zhang C, Krishnamurthy A, Langford J, Agarwal A. Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds. ICLR. 2020.",
            "doi": "10.48550/arXiv.1906.03671",
            "direct_relevance": "Modern deep uncertainty-diversity acquisition",
            "methods_to_compare": "BADGE gradient embeddings and k-means++ seeding",
            "required_revision_use": "Implement TCN-only comparator",
        },
    ]
    return pd.DataFrame(rows)


def build_pool_patterns():
    patterns = {
        "BALANCED_35": [5, 5, 5, 5, 5, 5, 5],
        "MILD_32": [5, 5, 5, 5, 4, 4, 4],
        "MODERATE_28": [5, 5, 5, 4, 4, 3, 2],
        "SEVERE_21": [5, 5, 4, 3, 2, 1, 1],
    }
    rows = []
    for level_order, (level, base_counts) in enumerate(patterns.items()):
        for rotation in range(7):
            rotated = base_counts[-rotation:] + base_counts[:-rotation] if rotation else base_counts
            row = {
                "imbalance_level": level,
                "level_order": level_order,
                "rotation_index": rotation,
                "total_candidates": int(sum(rotated)),
                "minimum_class_count": int(min(rotated)),
                "maximum_class_count": int(max(rotated)),
                "k21_is_full_pool_control": bool(sum(rotated) == 21),
            }
            for label, count in zip(LABELS, rotated):
                row[f"class_{label}_count"] = int(count)
            rows.append(row)
    return pd.DataFrame(rows)


def build_temporal_splits():
    rows = [
        ("FIRST_HALF_ORIGINAL", "1|2|3|4|5", "6|7|8|9|10", True),
        ("SECOND_HALF_REVERSED", "6|7|8|9|10", "1|2|3|4|5", False),
        ("ODD_CANDIDATE_EVEN_TEST", "1|3|5|7|9", "2|4|6|8|10", False),
        ("EVEN_CANDIDATE_ODD_TEST", "2|4|6|8|10", "1|3|5|7|9", False),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "split_id",
            "candidate_repetition_numbers",
            "fixed_test_repetition_numbers",
            "is_frozen_original_split",
        ],
    )


def build_strategy_definitions():
    rows = [
        ("PCBM_ORIGINAL", "Ridge|LDA|TCN", "Original raw top-two decision-score margin; one nominee per represented predicted class then global fill", "DEPLOYABLE", "All"),
        ("GLOBAL_MARGIN_ORIGINAL", "Ridge|LDA|TCN", "Globally smallest original raw top-two margins", "DEPLOYABLE", "All"),
        ("RANDOM_UNIFORM", "Ridge|LDA|TCN", "Uniform without replacement; locked acquisition seed", "DEPLOYABLE", "All"),
        ("LEAST_CONFIDENCE", "Ridge|LDA|TCN", "Smallest maximum class probability", "DEPLOYABLE", "Ridge uses source-only OOF calibration; LDA/TCN native probabilities"),
        ("PREDICTIVE_ENTROPY", "Ridge|LDA|TCN", "Largest Shannon entropy of class probabilities", "DEPLOYABLE", "Same probability contract as least confidence"),
        ("RBMAL_MARGIN_DIVERSITY", "Ridge|LDA|TCN", "Sequential batch rank with 0.5 normalized uncertainty plus 0.5 normalized Euclidean novelty", "DEPLOYABLE", "Diversity uses source-normalized repetition embedding"),
        ("CORE_SET_GREEDY", "Ridge|LDA|TCN", "Greedy k-center distance from labeled history and already selected batch members", "DEPLOYABLE", "No labels used before selection"),
        ("BADGE", "TCN", "Last-layer pseudo-label gradient embeddings with locked-seed k-means++", "DEPLOYABLE", "Deep comparison only"),
        ("TRUE_CLASS_BALANCED_ORACLE", "Reference", "Uses hidden true labels to select equal counts", "NON_DEPLOYABLE_REFERENCE", "Descriptive ceiling only; excluded from superiority inference"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "strategy",
            "applicable_engines",
            "locked_definition",
            "deployability",
            "implementation_note",
        ],
    )


def build_seed_schedule():
    rows = []
    namespace = REVISION_PROTOCOL_NAME + "|SEED_NAMESPACE_V1"
    for seed_index in range(1, 61):
        rows.append(
            {
                "seed_family": "RANDOM_ACQUISITION",
                "participant": "ALL",
                "seed_index": seed_index,
                "seed": stable_seed(f"{namespace}|RANDOM|{seed_index:02d}"),
                "use_rule": (
                    "INITIAL_30" if seed_index <= 30 else "PRESPECIFIED_EXTENSION_TO_60"
                ),
            }
        )
    for seed_index in range(1, 6):
        rows.append(
            {
                "seed_family": "POOL_SUBSET",
                "participant": "ALL",
                "seed_index": seed_index,
                "seed": stable_seed(f"{namespace}|POOL|{seed_index:02d}"),
                "use_rule": "ALL_FIVE_LOCKED_POOL_REALIZATIONS",
            }
        )
    locked_tcn = {
        "P01": [618490612, 1207316626, 1718339452, 901025671, 1928051441, 514154804],
        "P02": [1637964855, 955924008, 1910009012, 475435245, 197316294, 162000216],
        "P03": [873020370, 1077187824, 493443598, 1583128803, 420546654, 2085004370],
        "P04": [540151429, 135166150, 1576045209, 1327984691, 1067573097, 1986386293],
        "P05": [1533881370, 519741990, 1270390308, 1079046655, 280787894, 1121317831],
        "P06": [1567911598, 677159535, 791443996, 455208921, 292488224, 46667793],
        "P07": [440546248, 1973700858, 1471957582, 1030133353, 1444706075, 148061126],
    }
    for participant, seeds in locked_tcn.items():
        for seed_index, seed in enumerate(seeds):
            rows.append(
                {
                    "seed_family": "TCN_TRAINING",
                    "participant": participant,
                    "seed_index": seed_index,
                    "seed": seed,
                    "use_rule": (
                        "PREEXISTING_PRIMARY_DETERMINISTIC"
                        if seed_index == 0
                        else "PREEXISTING_STOCHASTIC_SENSITIVITY"
                    ),
                }
            )
    for item in ["BOOTSTRAP_BCA", "MC_CONVERGENCE", "BADGE_KMEANS_PP", "RBMAL_TIES"]:
        rows.append(
            {
                "seed_family": item,
                "participant": "ALL",
                "seed_index": 1,
                "seed": stable_seed(f"{namespace}|{item}"),
                "use_rule": "LOCKED_ANALYSIS_SEED",
            }
        )
    return pd.DataFrame(rows)


def build_experiment_manifest():
    rows = [
        ("R0", "Reviewer-response protocol lock", False, "No raw features, labels, or numerical outcomes", "CURRENT_STAGE"),
        ("R1", "Frozen artifact restoration and engine audit", False, "No new scientific result", "NEXT"),
        ("R2", "Literature, novelty, equations, oracle, and TCN methods correction", False, "Manuscript text only", "PENDING"),
        ("R3", "Selector implementations and balanced-pool comparator extension", True, "Unit tests then locked full run", "PENDING"),
        ("R4", "Four-level candidate-pool imbalance stress test", False, "Primary Ridge; LDA sensitivity", "PENDING"),
        ("R5", "Alternative temporal splits and within-session drift audit", False, "Primary Ridge; no future leakage", "PENDING"),
        ("R6", "TCN six-seed stability, BADGE, and computational-cost audit", True, "Two Tesla T4 workers; checkpointed", "PENDING"),
        ("R7", "Locked statistics, AULC, Monte Carlo convergence, and supplement", False, "Participant is inferential unit", "PENDING"),
        ("R8", "Point-by-point response, revised manuscript, formatting audit, freeze", False, "No new unplanned tests", "PENDING"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "stage",
            "task",
            "gpu_required",
            "scope_guardrail",
            "status_at_lock",
        ],
    )


def build_statistical_plan():
    rows = [
        ("REV_FOCAL_01", "Imbalance effect modification at K07", "[(PCBM-RANDOM) mean over MODERATE+SEVERE] - [(PCBM-RANDOM) BALANCED]", "Exact two-sided participant sign permutation", "BCa 95%; percentile sensitivity", "NONE_SINGLE_FOCAL", 6),
        ("REV_SECONDARY_IMBALANCE", "K07 PCBM comparisons within each imbalance level", "PCBM minus RANDOM/GLOBAL/LC/ENTROPY/RBMAL/CORE_SET", "Exact two-sided participant sign permutation", "BCa 95%; standardized and rank-biserial effects", "HOLM_WITHIN_IMBALANCE_FAMILY", 6),
        ("REV_SECONDARY_AULC", "Normalized AULC over K00/K07/K14/K21", "PCBM minus each deployable comparator", "Exact two-sided participant sign permutation", "BCa 95%", "HOLM_WITHIN_AULC_FAMILY", 6),
        ("REV_SPLIT_STABILITY", "Direction stability across four locked temporal splits", "PCBM minus RANDOM and GLOBAL at K07", "Exact two-sided participant sign permutation", "BCa 95%", "HOLM_WITHIN_SPLIT_FAMILY", 6),
        ("REV_DRIFT", "Within-session temporal drift", "Late-minus-early no-adaptation accuracy and feature-distance slope", "Exact two-sided participant sign permutation", "BCa 95%", "HOLM_TWO_DIAGNOSTICS", 6),
        ("REV_DEEP_STABILITY", "TCN training-seed stability", "Fixed-history and end-to-end K07 contrasts over six locked seeds", "Participant-level after averaging training seeds", "Seed distribution plus BCa 95% participant CI", "HOLM_LOCKED_DEEP_FAMILY", 6),
        ("REV_MC_RANDOM", "Random-policy Monte Carlo convergence", "Deviation of m-seed mean from full schedule for m=1,2,5,10,15,20,25,30", "Deterministic subsampling diagnostic", "MCSE and 95% interval", "DESCRIPTIVE_DIAGNOSTIC", 6),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "analysis_id",
            "analysis_question",
            "estimand",
            "test",
            "interval_and_effect",
            "multiplicity_policy",
            "inferential_participant_count",
        ],
    )


def build_equation_spec():
    rows = [
        ("EQ01", "Window extraction", "x[r,w,c,n] = s[r,c,n + (w-1)H], w=1,...,37", "Correct zero/one-based offset and prevent last-window overrun"),
        ("EQ02", "History-only mean", "mu[t,c] = sum_{r in H_t,w} m[r,w,c] log(1+x[r,w,c]) / sum_{r in H_t,w} m[r,w,c]", "Put r in H_t inside both sums"),
        ("EQ03", "Safe normalization", "z[r,w,c] = m[r,w,c]{log(1+x[r,w,c])-mu[t,c]} / max(sigma[t,c], epsilon)", "epsilon locked to 1e-8 for numerical definition; report zero-count policy separately"),
        ("EQ04", "TCN index", "Use q for within-repetition time/window and t only for session", "Remove overloaded symbol"),
        ("EQ05", "Causal convolution wording", "No future windows/time steps q are used", "Do not say future repetitions"),
        ("EQ09", "Round-aware selection", "Q_t^(a) = SELECT_7(U_t^(a), f_{theta_t^(a)}), a=1,...,K/7", "K14 and K21 are sequential rounds"),
        ("EQ10", "History update and refit", "H_t^(a+1)=H_t^(a) union {(r,y_r): r in Q_t^(a)}; theta_t^(a+1)=Fit(H_t^(a+1))", "True labels revealed only after the round"),
        ("EQ11", "Layout requirement", "Equation and number must remain in one Word equation paragraph", "Prevent number orphaning across pages"),
    ]
    return pd.DataFrame(
        rows,
        columns=["equation_id", "topic", "locked_corrected_form", "reason"],
    )


def build_reporting_requirements():
    rows = [
        ("TCN_ARCHITECTURE", "Four residual TCN blocks; channels, kernels, dilations, dropout, residual paths, mask planes, pooling, classifier, 118536 parameters"),
        ("TCN_PRETRAIN", "AdamW; lr=3e-4; weight_decay=1e-3; batch=64; cross-entropy with label smoothing=0.05; max=100; min=20; patience=15; best validation balanced accuracy"),
        ("TCN_ADAPT", "40 epochs; batch=16; encoder lr=1e-4; classifier lr=3e-4; weight_decay=1e-3; label smoothing=0.05; frozen stem and first two blocks"),
        ("RANDOMNESS", "All acquisition, initialization, mini-batch, worker, and BADGE seeds; matched-seed policy across strategies"),
        ("WILCOXON", "Statistic, nonzero-pair count, 1e-12 zero tolerance, zero differences discarded, exact sign enumeration, two-sided p-value"),
        ("BOOTSTRAP", "Existing analyses: percentile, 100000 resamples, SHA-derived seed. Revision: BCa primary interval and percentile sensitivity, both fully seeded"),
        ("MONTE_CARLO", "Random-seed convergence, MCSE, 25-to-30 change, and prespecified extension to 60 if adequacy thresholds fail"),
        ("EFFECTS", "Raw paired delta, standardized paired effect, rank-biserial correlation, CI, participant sign counts"),
        ("FULL_RESULTS", "All participant/session cells, all 18 retention tests, five classical secondary tests, Ridge/LDA/Strict-QC/TCN tables"),
        ("CLASS_RESULTS", "Confusion matrices, per-class recall, class coverage and normalized entropy distributions"),
        ("BALANCED_TEST", "State that balanced accuracy equals ordinary accuracy when all seven test classes have equal support"),
        ("COMPUTE", "Selection latency, model-score latency, refit time, end-to-end session time, peak CPU RAM, peak GPU VRAM, hardware and software"),
        ("ORACLE", "Dataset labels are retrospectively hidden; simulated oracle reveals intended movement after selection; protocol measures annotation/review burden and requires all 35 candidates to have been recorded"),
        ("AVAILABILITY", "Persistent code/evidence links must replace placeholders before submission"),
        ("AI_DISCLOSURE", "Journal-specific disclosure of generative-AI assistance, purpose, human verification, and author responsibility"),
    ]
    return pd.DataFrame(rows, columns=["requirement_id", "locked_requirement"])


def build_claim_guardrails():
    rows = [
        ("FIRST_ACTIVE_LEARNING_EMG", "PROHIBITED", "Szymaniak et al. 2022 directly predates this work"),
        ("PCBM_PREDICTIVE_SUPERIORITY", "NOT_ESTABLISHED", "May change only if locked revision evidence supports a narrowly defined new claim"),
        ("PCBM_EQUIVALENCE", "PROHIBITED", "No equivalence margin or equivalence test is locked"),
        ("ORIGINAL_STAGE3G_REPLACEMENT", "PROHIBITED", "Revision extensions cannot replace frozen primary conclusions"),
        ("DEEP_STAGE5F_REPLACEMENT", "PROHIBITED", "Training-seed sensitivity cannot rewrite the frozen deep result"),
        ("P07_GENERALIZATION", "PROHIBITED", "P07 remains a descriptive limb-difference case analysis"),
        ("CAUSAL_INFERENCE", "PROHIBITED", "Leakage-controlled temporal simulation is not causal-effect identification"),
        ("CLINICAL_DEPLOYMENT", "PROHIBITED", "Offline retrospective evaluation does not establish real-time clinical utility"),
        ("NEGATIVE_RESULT", "ALLOWED", "A well-powered stress test may remain a bounded negative result"),
        ("ACQUISITION_DIVERSITY", "FROZEN_DESCRIPTIVE", "Original diversity finding remains descriptive and unchanged"),
    ]
    return pd.DataFrame(rows, columns=["claim_id", "status", "rationale"])


def build_protocol(
    action_matrix,
    literature,
    pools,
    splits,
    strategies,
    seeds,
    experiments,
    statistics,
    equations,
    reporting,
    claims,
):
    protocol = {
        "protocol_name": REVISION_PROTOCOL_NAME,
        "protocol_stage": "REVISION_R0_LOCK",
        "lock_date_utc": "2026-08-12",
        "scientific_status": "REVIEWER_REQUESTED_EXTENSION_INTERNALLY_PRESPECIFIED_AND_FROZEN",
        "parent_evidence": {
            "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
            "stage3g_freeze_sha256": STAGE3G_FREEZE_SHA256,
            "stage5f_packet_sha256": STAGE5F_PACKET_SHA256,
            "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
            "stage3g_conclusion": FROZEN_STAGE3G_CONCLUSION,
            "deep_conclusion": FROZEN_DEEP_CONCLUSION,
            "replacement_by_revision_is_allowed": False,
        },
        "population": {
            "inferential_participants": ABLE_BODIED,
            "inferential_unit": "participant",
            "p07_role": "DESCRIPTIVE_CASE_ANALYSIS_ONLY",
            "target_sessions": TARGET_SESSIONS,
            "labels": LABELS,
        },
        "focal_revision_question": {
            "question": "Does the relative K07 performance of PCBM versus mean random acquisition improve when the candidate pool becomes moderately or severely class imbalanced?",
            "estimand": "[(PCBM-RANDOM) averaged over MODERATE_28 and SEVERE_21] minus [(PCBM-RANDOM) under BALANCED_35]",
            "endpoint": "participant mean repetition balanced accuracy across five target sessions, rotations, and locked pool realizations",
            "test": "two-sided exact participant sign permutation",
            "confidence_interval": "BCa 95% with percentile sensitivity",
            "multiplicity": "none for the single revision focal contrast",
        },
        "candidate_pool_stress": {
            "imbalance_levels": sorted(pools["imbalance_level"].unique().tolist()),
            "rotations_per_level": 7,
            "pool_subset_realizations": 5,
            "maximum_available_repetitions_per_true_class": 5,
            "common_budgets": BUDGETS,
            "k07_role": "REVISION_FOCAL_LOW_BUDGET",
            "k14_role": "SECONDARY",
            "k21_role": "SECONDARY_AND_FULL_POOL_CONTROL_FOR_SEVERE_21",
            "true_labels_used_to_construct_pool": True,
            "true_labels_visible_to_selector": False,
            "fixed_test_changed_by_pool_stress": False,
        },
        "temporal_split_sensitivity": {
            "split_ids": splits["split_id"].tolist(),
            "session0_initial_source_repetitions": [1, 2, 3, 4, 5],
            "target_candidate_and_test_are_disjoint": True,
            "future_sessions_accessible": False,
        },
        "probability_contract": {
            "ridge": "Five-fold stratified out-of-fold repetition-score calibration using labeled history only; multinomial L2 logistic calibration C=1, lbfgs, max_iter=5000; Ridge refit on full history",
            "lda": "Native shrinkage-LDA posterior probability",
            "tcn": "Native softmax probability",
            "unlabeled_pool_used_for_calibration": False,
            "fixed_test_used_for_calibration": False,
        },
        "aulc": {
            "grid": BUDGETS,
            "formula": "trapezoidal integral of balanced accuracy over K=0,7,14,21 divided by 21",
            "same_grid_required_for_all_strategies": True,
        },
        "random_policy_convergence": {
            "initial_seed_count": 30,
            "prefix_sizes": [1, 2, 5, 10, 15, 20, 25, 30],
            "deterministic_subsamples_per_size": 1000,
            "adequacy_threshold_absolute_25_to_30_change": 0.005,
            "adequacy_threshold_95pct_mc_halfwidth": 0.01,
            "if_threshold_fails": "Run prespecified seeds 31-60 before final analysis",
        },
        "deep_stochasticity": {
            "training_seeds_per_participant": 6,
            "seed_source": "pre-existing Stage5A4B locked schedule",
            "matched_initialization_and_minibatch_seed_across_strategies": True,
            "fixed_history_refit_analysis": True,
            "end_to_end_sensitivity": True,
            "inference_after_seed_averaging_uses_participant": True,
        },
        "statistical_rules": {
            "zero_tolerance": 1e-12,
            "zero_difference_policy": "discard before exact sign enumeration; report tied count",
            "two_sided": True,
            "bca_bootstrap_replicates": 100000,
            "percentile_sensitivity_replicates": 100000,
            "random_seeds_are_not_inferential_units": True,
            "sessions_are_not_inferential_units": True,
            "windows_and_repetitions_are_not_inferential_units": True,
            "equivalence_language_allowed": False,
        },
        "oracle_interpretation": {
            "offline_oracle": "The stored intended-movement label is hidden until selection and then revealed by the simulator",
            "real_world_analogue": "The user confirms or corrects intended movement after a query",
            "all_35_candidate_repetitions_recorded_before_pool_selection": True,
            "budget_interpretation": "annotation/review burden, not total signal-acquisition burden",
            "prompted_online_acquisition_claimed": False,
        },
        "table_row_counts": {
            "action_matrix": len(action_matrix),
            "literature_matrix": len(literature),
            "pool_pattern_rotations": len(pools),
            "temporal_splits": len(splits),
            "strategies": len(strategies),
            "seed_rows": len(seeds),
            "experiment_stages": len(experiments),
            "statistical_analyses": len(statistics),
            "equation_corrections": len(equations),
            "reporting_requirements": len(reporting),
            "claim_guardrails": len(claims),
        },
    }
    protocol_sha256 = canonical_hash(protocol)
    protocol["protocol_sha256"] = protocol_sha256
    return protocol


def make_summary_pdf(protocol, tables, readiness_gates, destination):
    def add_text_page(pdf, title, lines):
        figure = plt.figure(figsize=(8.5, 11))
        figure.patch.set_facecolor("white")
        plt.axis("off")
        figure.text(0.07, 0.95, title, fontsize=15, weight="bold", va="top")
        y = 0.91
        for line in lines:
            wrapped = textwrap.wrap(str(line), width=102) or [""]
            for part in wrapped:
                figure.text(0.07, y, part, fontsize=8.4, va="top", family="DejaVu Sans")
                y -= 0.019
                if y < 0.06:
                    pdf.savefig(figure, bbox_inches="tight")
                    plt.close(figure)
                    figure = plt.figure(figsize=(8.5, 11))
                    figure.patch.set_facecolor("white")
                    plt.axis("off")
                    y = 0.95
            y -= 0.004
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    with PdfPages(destination) as pdf:
        add_text_page(
            pdf,
            "Revision R0 — Reviewer-Response Protocol Lock",
            [
                f"Protocol: {protocol['protocol_name']}",
                f"Protocol SHA-256: {protocol['protocol_sha256']}",
                "Status: internally prespecified and frozen before reviewer-requested experiments.",
                "The Stage 3G classical conclusion and Stage 5F deep conclusion remain immutable.",
                "No raw DELTA feature, label, or outcome table was accessed in this stage.",
                "",
                "Focal revision question:",
                protocol["focal_revision_question"]["question"],
                "",
                "Candidate-pool levels: BALANCED_35, MILD_32, MODERATE_28, SEVERE_21; seven rotations and five locked pool realizations.",
                "Temporal splits: original halves, reversed halves, odd/even, and even/odd.",
                "New comparators: least confidence, entropy, RBMAL, core-set, and TCN-only BADGE.",
                "TCN stability: six pre-existing locked training seeds per participant.",
            ],
        )
        add_text_page(
            pdf,
            "Execution Plan",
            [
                f"{row.stage}: {row.task} | GPU={row.gpu_required} | {row.scope_guardrail}"
                for row in tables["experiments"].itertuples(index=False)
            ],
        )
        add_text_page(
            pdf,
            "Locked Statistical Analyses",
            [
                f"{row.analysis_id}: {row.analysis_question} | {row.test} | {row.multiplicity_policy}"
                for row in tables["statistics"].itertuples(index=False)
            ],
        )
        add_text_page(
            pdf,
            "Claim Guardrails and Readiness",
            [
                *[
                    f"{row.claim_id}: {row.status} — {row.rationale}"
                    for row in tables["claims"].itertuples(index=False)
                ],
                "",
                *[f"{key}: {value}" for key, value in readiness_gates.items()],
            ],
        )


def manifest_for_directory(directory, excluded_names=None):
    excluded_names = set(excluded_names or [])
    rows = []
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.name not in excluded_names:
            rows.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return pd.DataFrame(rows)


def main():
    print("=" * 79)
    print("REVISION R0 — REVIEWER-RESPONSE EXPERIMENT PROTOCOL LOCK")
    print("=" * 79)
    print("Execution device: CPU")
    print("Raw DELTA data accessed: False")
    print("New model training: False")
    print("New statistical tests executed: False")
    print("Purpose: freeze all reviewer-requested extensions before outcomes")
    print()

    bootstrap_rclone()
    create_rclone_config()
    print(
        "rclone version:",
        rclone(["version"]).stdout.splitlines()[0],
    )
    print("Restoring immutable Stage 3G and Stage 5F parents...")
    _, parent_gates = restore_and_verify_parents()

    action_matrix = build_action_matrix()
    literature = build_literature_matrix()
    pools = build_pool_patterns()
    splits = build_temporal_splits()
    strategies = build_strategy_definitions()
    seeds = build_seed_schedule()
    experiments = build_experiment_manifest()
    statistics = build_statistical_plan()
    equations = build_equation_spec()
    reporting = build_reporting_requirements()
    claims = build_claim_guardrails()

    protocol = build_protocol(
        action_matrix,
        literature,
        pools,
        splits,
        strategies,
        seeds,
        experiments,
        statistics,
        equations,
        reporting,
        claims,
    )

    split_checks = []
    for row in splits.itertuples(index=False):
        candidate = {int(value) for value in row.candidate_repetition_numbers.split("|")}
        test = {int(value) for value in row.fixed_test_repetition_numbers.split("|")}
        split_checks.append(
            candidate.isdisjoint(test)
            and candidate | test == set(range(1, 11))
            and len(candidate) == len(test) == 5
        )

    tcn_seeds = seeds[seeds["seed_family"] == "TCN_TRAINING"]
    random_seeds = seeds[seeds["seed_family"] == "RANDOM_ACQUISITION"]
    pool_seeds = seeds[seeds["seed_family"] == "POOL_SUBSET"]
    all_seed_values = seeds["seed"].astype(int)

    readiness_gates = {
        **parent_gates,
        "protocol_name_is_locked": protocol["protocol_name"] == REVISION_PROTOCOL_NAME,
        "protocol_hash_is_valid": (
            canonical_hash({k: v for k, v in protocol.items() if k != "protocol_sha256"})
            == protocol["protocol_sha256"]
        ),
        "stage3g_and_stage5f_cannot_be_replaced": (
            protocol["parent_evidence"]["replacement_by_revision_is_allowed"] is False
        ),
        "inferential_participants_are_exactly_p01_to_p06": ABLE_BODIED == ["P01", "P02", "P03", "P04", "P05", "P06"],
        "p07_is_descriptive_only": protocol["population"]["p07_role"] == "DESCRIPTIVE_CASE_ANALYSIS_ONLY",
        "four_imbalance_levels_are_locked": pools["imbalance_level"].nunique() == 4,
        "each_imbalance_level_has_seven_rotations": bool((pools.groupby("imbalance_level").size() == 7).all()),
        "all_pool_class_counts_are_between_one_and_five": bool((pools.filter(like="class_").to_numpy() >= 1).all() and (pools.filter(like="class_").to_numpy() <= 5).all()),
        "every_pool_supports_k21": bool((pools["total_candidates"] >= 21).all()),
        "five_pool_subset_seeds_are_locked": len(pool_seeds) == 5,
        "four_temporal_splits_are_locked": len(splits) == 4,
        "all_temporal_splits_are_disjoint_complete_and_balanced": all(split_checks),
        "required_new_selector_set_is_present": {"LEAST_CONFIDENCE", "PREDICTIVE_ENTROPY", "RBMAL_MARGIN_DIVERSITY", "CORE_SET_GREEDY", "BADGE"}.issubset(set(strategies["strategy"])),
        "ridge_probability_calibration_is_source_only": protocol["probability_contract"]["unlabeled_pool_used_for_calibration"] is False and protocol["probability_contract"]["fixed_test_used_for_calibration"] is False,
        "initial_30_random_seeds_are_locked": len(random_seeds[random_seeds["seed_index"] <= 30]) == 30,
        "extension_seeds_31_to_60_are_prespecified": len(random_seeds[random_seeds["seed_index"] > 30]) == 30,
        "all_seed_values_are_unique": all_seed_values.nunique() == len(all_seed_values),
        "six_tcn_training_seeds_per_participant_are_locked": bool((tcn_seeds.groupby("participant").size() == 6).all()) and set(tcn_seeds["participant"]) == set(PARTICIPANTS),
        "random_seeds_are_not_inferential_units": protocol["statistical_rules"]["random_seeds_are_not_inferential_units"] is True,
        "sessions_are_not_inferential_units": protocol["statistical_rules"]["sessions_are_not_inferential_units"] is True,
        "equivalence_claim_is_prohibited": protocol["statistical_rules"]["equivalence_language_allowed"] is False,
        "first_active_learning_emg_claim_is_prohibited": claims.loc[claims["claim_id"] == "FIRST_ACTIVE_LEARNING_EMG", "status"].iloc[0] == "PROHIBITED",
        "oracle_burden_is_defined_as_annotation_review": protocol["oracle_interpretation"]["budget_interpretation"] == "annotation/review burden, not total signal-acquisition burden",
        "action_matrix_covers_at_least_25_items": len(action_matrix) >= 25,
        "equation_correction_spec_has_eight_rows": len(equations) == 8,
        "execution_plan_ends_in_revised_manuscript_freeze": experiments.iloc[-1]["stage"] == "R8",
        "no_raw_data_was_accessed": True,
        "no_model_was_trained": True,
        "no_statistical_test_was_executed": True,
    }
    if not all(readiness_gates.values()):
        failed = [key for key, value in readiness_gates.items() if not value]
        raise RuntimeError(f"Revision R0 lock failed gates: {failed}")

    tables = {
        "actions": action_matrix,
        "literature": literature,
        "pools": pools,
        "splits": splits,
        "strategies": strategies,
        "seeds": seeds,
        "experiments": experiments,
        "statistics": statistics,
        "equations": equations,
        "reporting": reporting,
        "claims": claims,
    }
    filenames = {
        "actions": "stageR0_reviewer_action_matrix.csv",
        "literature": "stageR0_literature_gap_matrix.csv",
        "pools": "stageR0_candidate_pool_imbalance_patterns.csv",
        "splits": "stageR0_temporal_split_schedule.csv",
        "strategies": "stageR0_strategy_definitions.csv",
        "seeds": "stageR0_seed_schedule.csv",
        "experiments": "stageR0_execution_manifest.csv",
        "statistics": "stageR0_statistical_analysis_plan.csv",
        "equations": "stageR0_equation_correction_spec.csv",
        "reporting": "stageR0_reporting_requirements.csv",
        "claims": "stageR0_claim_guardrails.csv",
    }
    for key, dataframe in tables.items():
        atomic_csv(dataframe, RESULT_ROOT / filenames[key])
    atomic_json(protocol, RESULT_ROOT / "stageR0_locked_revision_protocol.json")

    report = {
        "stage": "REVISION_R0_REVIEWER_RESPONSE_PROTOCOL_LOCK",
        "protocol_name": REVISION_PROTOCOL_NAME,
        "protocol_sha256": protocol["protocol_sha256"],
        "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
        "stage3g_freeze_sha256": STAGE3G_FREEZE_SHA256,
        "stage5f_packet_sha256": STAGE5F_PACKET_SHA256,
        "deep_protocol_sha256": DEEP_PROTOCOL_SHA256,
        "raw_delta_data_accessed": False,
        "new_model_training_performed": False,
        "new_statistical_test_performed": False,
        "frozen_stage3g_conclusion": FROZEN_STAGE3G_CONCLUSION,
        "frozen_deep_conclusion": FROZEN_DEEP_CONCLUSION,
        "readiness_gates": readiness_gates,
        "all_readiness_gates_passed": bool(all(readiness_gates.values())),
        "final_decision": "PASS_TO_REVISION_R1_FROZEN_ARTIFACT_AND_ENGINE_AUDIT",
    }
    atomic_json(report, RESULT_ROOT / "stageR0_protocol_lock_report.json")

    make_summary_pdf(
        protocol,
        tables,
        readiness_gates,
        RESULT_ROOT / "stageR0_protocol_lock_summary.pdf",
    )
    source_mode = persist_executed_source(
        RESULT_ROOT / "stageR0_executed_source.py"
    )
    report["executed_source_capture_mode"] = source_mode
    atomic_json(report, RESULT_ROOT / "stageR0_protocol_lock_report.json")

    manifest = manifest_for_directory(
        RESULT_ROOT,
        excluded_names={"stageR0_sha256_manifest.csv"},
    )
    atomic_csv(manifest, RESULT_ROOT / "stageR0_sha256_manifest.csv")

    packet_crc = make_zip(
        RESULT_ROOT,
        PACKET_PATH,
        "StageR0_Reviewer_Revision_Protocol_Lock",
    )
    packet_sha256 = sha256_file(PACKET_PATH)

    print()
    print("=" * 79)
    print("REVISION R0 — LOCK SUMMARY")
    print("=" * 79)
    print("Protocol:", REVISION_PROTOCOL_NAME)
    print("Protocol SHA-256:", protocol["protocol_sha256"])
    print("Reviewer actions:", len(action_matrix))
    print("Imbalance levels:", pools["imbalance_level"].nunique())
    print("Pool rotations:", len(pools))
    print("Temporal splits:", len(splits))
    print("Acquisition strategies/references:", len(strategies))
    print("Locked seed rows:", len(seeds))
    print("TCN training seeds per participant: 6")
    print("Raw data accessed: False")
    print("Model training performed: False")
    print("Statistical tests executed: False")
    print()
    print("Readiness gates:")
    for key, value in readiness_gates.items():
        print(f"  {key}: {value}")

    print()
    print("Uploading locked protocol packet to Google Drive...")
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
    # Google Drive exposes MD5 but not a native SHA-256 remote hash through
    # rclone.  Verify the exact uploaded bytes by downloading the packet to a
    # separate local path and computing SHA-256 locally.
    roundtrip_packet = WORKING / "_stageR0_roundtrip_packet.zip"
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
    remote_hash = sha256_file(roundtrip_packet)
    remote_verified = remote_hash == packet_sha256
    roundtrip_packet.unlink()
    if not remote_verified:
        raise RuntimeError("Remote Revision R0 packet hash mismatch")

    roundtrip_report = RESULT_ROOT / "stageR0_drive_roundtrip_verification.json"
    atomic_json(
        {
            "stage": "REVISION_R0_DRIVE_ROUNDTRIP_VERIFICATION",
            "protocol_sha256": protocol["protocol_sha256"],
            "packet_filename": PACKET_PATH.name,
            "local_packet_sha256": packet_sha256,
            "roundtrip_packet_sha256": remote_hash,
            "roundtrip_sha256_matches": remote_verified,
            "credentials_displayed": False,
        },
        roundtrip_report,
    )
    rclone(
        [
            "copyto",
            str(roundtrip_report),
            REMOTE_OUTPUT + "/" + roundtrip_report.name,
            "--retries",
            "5",
            "--low-level-retries",
            "10",
            "--timeout",
            "5m",
        ]
    )

    cleanup_secret()
    runtime_minutes = (time.time() - START_TIME) / 60.0
    print()
    print("Packet CRC pass:", packet_crc)
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", packet_sha256)
    print("Remote packet verified:", remote_verified)
    print("Runtime minutes:", round(runtime_minutes, 2))
    print()
    print("FINAL DECISION: PASS_TO_REVISION_R1_FROZEN_ARTIFACT_AND_ENGINE_AUDIT")


if __name__ == "__main__":
    main()
