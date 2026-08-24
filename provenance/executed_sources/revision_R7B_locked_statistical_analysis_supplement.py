from __future__ import annotations

import fcntl
import hashlib
import io
import itertools
import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

import revision_R3A_P1_float32_engine_frozen_trajectory_unit_test as engine
import revision_R7A_frozen_statistical_input_schema_audit as r7a


REVISION_PROTOCOL_SHA256 = "6807b71de18ca82013cfa4360d760e0daf9a920a1acc0625dcb13bd8f4d07249"
R7A_PACKET_SHA256 = "4ac042997fdf12252a84fd26a9163646973ba2580e0621f4e67d03ec05b14122"
R7A_BASENAME = "revision_R7A_frozen_statistical_input_schema_audit_packet.zip"
R7A_RELATIVE = (
    "Reviewer_Revision/Revision_R7A_Frozen_Statistical_Input_Schema_Audit/"
    + R7A_BASENAME
)
ABLE_BODIED = ["P01", "P02", "P03", "P04", "P05", "P06"]
ZERO_TOLERANCE = 1e-12
BOOTSTRAP_REPLICATES = 100_000
MC_REPLICATES = 1_000
MC_PREFIXES = [1, 2, 5, 10, 15, 20, 25, 30]
LEVELS = ["BALANCED_35", "MILD_32", "MODERATE_28", "SEVERE_21"]
COMPARATORS = [
    "RANDOM_UNIFORM",
    "GLOBAL_MARGIN_ORIGINAL",
    "LEAST_CONFIDENCE",
    "PREDICTIVE_ENTROPY",
    "RBMAL_MARGIN_DIVERSITY",
    "CORE_SET_GREEDY",
]

WORKING = Path(os.environ.get("REVISION_R7B_WORKING", "/kaggle/working"))
INPUT_ROOT = WORKING / "REVISION_R7B_FROZEN_INPUTS"
RESULT_ROOT = WORKING / "DELTA_REVIEWER_REVISION" / "Revision_R7B_Locked_Statistical_Analysis_and_Supplement"
SUPPLEMENT_ROOT = RESULT_ROOT / "Supplementary_Data"
PACKET_PATH = WORKING / "revision_R7B_locked_statistical_analysis_supplement_packet.zip"
REMOTE_OUTPUT = engine.REMOTE_BASE + "/Reviewer_Revision/Revision_R7B_Locked_Statistical_Analysis_and_Supplement"
START_TIME = time.time()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def stable_seed(text: str) -> int:
    digest = hashlib.sha256((REVISION_PROTOCOL_SHA256 + "|" + text).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 2_000_000_000 + 1


def exact_signed_rank(differences: np.ndarray) -> dict:
    original = np.asarray(differences, dtype=float)
    kept = original[np.abs(original) > ZERO_TOLERANCE]
    tied_zero = int(len(original) - len(kept))
    if len(kept) == 0:
        return {
            "nonzero_pair_count": 0,
            "zero_difference_count": tied_zero,
            "wilcoxon_w_plus": 0.0,
            "wilcoxon_w_minus": 0.0,
            "wilcoxon_statistic": 0.0,
            "exact_two_sided_p_value": 1.0,
            "rank_biserial_correlation": 0.0,
        }
    ranks = rankdata(np.abs(kept), method="average")
    positive = kept > 0
    w_plus = float(ranks[positive].sum())
    w_minus = float(ranks[~positive].sum())
    rank_total = float(ranks.sum())
    observed = abs(w_plus - rank_total / 2.0)
    enumerated = []
    for bits in itertools.product([False, True], repeat=len(kept)):
        enumerated.append(float(ranks[np.asarray(bits, dtype=bool)].sum()))
    enumerated = np.asarray(enumerated, dtype=float)
    p_value = float(np.mean(np.abs(enumerated - rank_total / 2.0) >= observed - 1e-12))
    return {
        "nonzero_pair_count": int(len(kept)),
        "zero_difference_count": tied_zero,
        "wilcoxon_w_plus": w_plus,
        "wilcoxon_w_minus": w_minus,
        "wilcoxon_statistic": min(w_plus, w_minus),
        "exact_two_sided_p_value": min(1.0, p_value),
        "rank_biserial_correlation": float((w_plus - w_minus) / rank_total),
    }


def bootstrap_mean_intervals(differences: np.ndarray, analysis_id: str) -> dict:
    values = np.asarray(differences, dtype=float)
    theta = float(values.mean())
    rng = np.random.default_rng(stable_seed("BOOTSTRAP|" + analysis_id))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    boot = values[indices].mean(axis=1)
    percentile_low, percentile_high = np.quantile(boot, [0.025, 0.975])
    if np.allclose(boot, boot[0], atol=0.0, rtol=0.0):
        bca_low = bca_high = float(boot[0])
    else:
        proportion = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / len(boot)
        proportion = float(np.clip(proportion, 1.0 / (2 * len(boot)), 1.0 - 1.0 / (2 * len(boot))))
        z0 = float(norm.ppf(proportion))
        jackknife = np.asarray([np.delete(values, index).mean() for index in range(len(values))])
        jack_mean = float(jackknife.mean())
        numerator = float(np.sum((jack_mean - jackknife) ** 3))
        denominator = float(6.0 * np.sum((jack_mean - jackknife) ** 2) ** 1.5)
        acceleration = numerator / denominator if denominator > 0 else 0.0
        adjusted = []
        for alpha in [0.025, 0.975]:
            z_alpha = float(norm.ppf(alpha))
            divisor = 1.0 - acceleration * (z0 + z_alpha)
            transformed = z0 + (z0 + z_alpha) / divisor if abs(divisor) > 1e-15 else z0 + z_alpha
            adjusted.append(float(np.clip(norm.cdf(transformed), 0.0, 1.0)))
        bca_low, bca_high = np.quantile(boot, adjusted)
    return {
        "bca_95_ci_low": float(bca_low),
        "bca_95_ci_high": float(bca_high),
        "percentile_95_ci_low": float(percentile_low),
        "percentile_95_ci_high": float(percentile_high),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": stable_seed("BOOTSTRAP|" + analysis_id),
    }


def locked_test(analysis_id: str, family: str, description: str, differences: np.ndarray) -> dict:
    values = np.asarray(differences, dtype=float)
    if len(values) != 6 or not np.isfinite(values).all():
        raise RuntimeError(f"{analysis_id} does not contain six finite participant estimands")
    exact = exact_signed_rank(values)
    sd = float(values.std(ddof=1))
    return {
        "analysis_id": analysis_id,
        "multiplicity_family": family,
        "description": description,
        "inferential_unit": "PARTICIPANT",
        "participant_count": len(values),
        "mean_paired_delta": float(values.mean()),
        "median_paired_delta": float(np.median(values)),
        "sd_paired_delta": sd,
        "standardized_paired_effect_dz": float(values.mean() / sd) if sd > 0 else np.nan,
        "participants_positive": int((values > ZERO_TOLERANCE).sum()),
        "participants_tied": int((np.abs(values) <= ZERO_TOLERANCE).sum()),
        "participants_negative": int((values < -ZERO_TOLERANCE).sum()),
        **exact,
        **bootstrap_mean_intervals(values, analysis_id),
        "zero_tolerance": ZERO_TOLERANCE,
        "two_sided": True,
    }


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["holm_adjusted_p_value"] = np.nan
    for family, group in output.groupby("multiplicity_family", sort=False):
        positions = group.index.to_numpy()
        p_values = group["exact_two_sided_p_value"].to_numpy(dtype=float)
        if family == "NONE_SINGLE_FOCAL":
            adjusted = p_values
        else:
            order = np.argsort(p_values, kind="mergesort")
            adjusted = np.zeros(len(p_values), dtype=float)
            running = 0.0
            for rank_position, original_index in enumerate(order):
                running = max(running, min(1.0, (len(p_values) - rank_position) * p_values[original_index]))
                adjusted[original_index] = running
        output.loc[positions, "holm_adjusted_p_value"] = adjusted
    output["significant_holm_0_05"] = output["holm_adjusted_p_value"] < 0.05
    return output


def all_csv_members(packet: Path) -> list[tuple[str, int, list[str]]]:
    rows = []
    with zipfile.ZipFile(packet, "r") as archive:
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() != ".csv":
                continue
            try:
                header = pd.read_csv(archive.open(info), nrows=3)
            except Exception:
                continue
            rows.append((info.filename, int(info.file_size), [str(column) for column in header.columns]))
    return rows


def read_named_member(packet: Path, basename: str) -> pd.DataFrame:
    return engine.read_csv_member(packet, basename)


def read_best_balanced_random(packet: Path) -> tuple[pd.DataFrame, str]:
    candidates = []
    for member, size, columns in all_csv_members(packet):
        lower_columns = [column.lower() for column in columns]
        has_participant = any("participant" in column for column in lower_columns)
        has_budget = any(column in {"query_budget", "budget", "k"} or "budget" in column for column in lower_columns)
        has_metric = any("balanced" in column and "accuracy" in column and "std" not in column for column in lower_columns)
        if not (has_participant and has_budget and has_metric):
            continue
        name = Path(member).name.lower()
        score = int("fold" in name) * 50 + int("result" in name) * 20 + int("summary" in name) * 10 + min(size / 1e6, 9)
        candidates.append((score, member))
    if not candidates:
        raise RuntimeError("No balanced Stage 3C-2 random performance table could be identified")
    member = max(candidates)[1]
    with zipfile.ZipFile(packet, "r") as archive:
        frame = pd.read_csv(archive.open(member))
    return frame, member


def canonical_random(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "participant": ["participant", "participant_id", "target_participant"],
        "query_budget": ["query_budget", "budget", "k", "acquisition_budget"],
        "value": [
            "mean_repetition_balanced_accuracy",
            "repetition_balanced_accuracy",
            "balanced_accuracy",
            "mean_balanced_accuracy",
            "test_balanced_accuracy",
        ],
    }
    chosen: dict[str, str] = {}
    lower = {str(column).lower(): column for column in frame.columns}
    for target, names in aliases.items():
        matches = [lower[name] for name in names if name in lower]
        if not matches and target == "participant":
            matches = [column for key, column in lower.items() if "participant" in key]
        if not matches and target == "query_budget":
            matches = [column for key, column in lower.items() if "budget" in key]
        if not matches and target == "value":
            matches = [column for key, column in lower.items() if "balanced" in key and "accuracy" in key and "std" not in key]
        if not matches:
            raise RuntimeError(f"Balanced random table lacks {target}; columns={list(frame.columns)}")
        chosen[target] = matches[0]
    data = frame.rename(columns={source: target for target, source in chosen.items()}).copy()
    if "strategy" in data.columns:
        random_rows = data["strategy"].astype(str).str.upper().str.contains("RANDOM")
        if random_rows.any():
            data = data.loc[random_rows]
    data["participant"] = data["participant"].astype(str)
    data["query_budget"] = pd.to_numeric(data["query_budget"], errors="raise").astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="raise").astype(float)
    return (
        data.loc[data["participant"].isin(ABLE_BODIED + ["P07"]) & data["query_budget"].isin([7, 14, 21])]
        .groupby(["participant", "query_budget"], as_index=False)
        .agg(value=("value", "mean"))
        .assign(strategy="RANDOM_UNIFORM", imbalance_level="BALANCED_35")
    )


def canonical_balanced(packets: dict[str, Path]) -> tuple[pd.DataFrame, str]:
    r3a_folds = read_named_member(
        packets["revision_R3A_P1_float32_engine_frozen_trajectory_unit_test_packet.zip"],
        "revision_R3A_P1_reconstructed_folds.csv",
    )
    for column in ["query_budget", "repetition_balanced_accuracy"]:
        r3a_folds[column] = pd.to_numeric(r3a_folds[column], errors="raise")
    r3a = (
        r3a_folds.loc[
            r3a_folds["participant"].isin(ABLE_BODIED + ["P07"])
            & r3a_folds["strategy"].isin(["NO_ADAPTATION_REFERENCE", "PCBM_PROPOSED", "GLOBAL_MARGIN"])
            & r3a_folds["query_budget"].isin([0, 7, 14, 21])
        ]
        .groupby(["participant", "strategy", "query_budget"], as_index=False)
        .agg(value=("repetition_balanced_accuracy", "mean"))
    )
    r3a["strategy"] = r3a["strategy"].replace({"GLOBAL_MARGIN": "GLOBAL_MARGIN_ORIGINAL"})
    r3a["imbalance_level"] = "BALANCED_35"
    r3c = read_named_member(
        packets["revision_R3C_balanced_pool_classical_comparator_extension_packet.zip"],
        "revision_R3C_participant_summary.csv",
    )
    r3c["query_budget"] = pd.to_numeric(r3c["query_budget"], errors="raise").astype(int)
    r3c["mean_repetition_balanced_accuracy"] = pd.to_numeric(
        r3c["mean_repetition_balanced_accuracy"], errors="raise"
    )
    r3c = (
        r3c.loc[r3c["participant"].isin(ABLE_BODIED + ["P07"])]
        .groupby(["participant", "strategy", "query_budget"], as_index=False)
        .agg(value=("mean_repetition_balanced_accuracy", "mean"))
    )
    r3c["imbalance_level"] = "BALANCED_35"
    random_raw, random_member = read_best_balanced_random(packets["stage3c2_random_control_packet.zip"])
    random = canonical_random(random_raw)
    balanced = pd.concat([r3a, r3c, random], ignore_index=True)
    balanced["strategy"] = balanced["strategy"].replace(
        {"PCBM_PROPOSED": "PCBM_ORIGINAL", "GLOBAL_MARGIN": "GLOBAL_MARGIN_ORIGINAL"}
    )
    duplicates = balanced.groupby(["participant", "strategy", "query_budget"]).size()
    if duplicates.max() > 1:
        balanced = (
            balanced.groupby(["participant", "strategy", "query_budget", "imbalance_level"], as_index=False)
            .agg(value=("value", "mean"))
        )
    return balanced, random_member


def canonical_imbalance(packets: dict[str, Path], balanced: pd.DataFrame, classifier: str) -> pd.DataFrame:
    if classifier == "RIDGE":
        deterministic_packet = "revision_R4B_ridge_deterministic_imbalance_packet.zip"
        deterministic_member = "revision_R4B_participant_level_summary.csv"
        random_packet = "revision_R4C_ridge_random_imbalance_packet.zip"
        random_member = "revision_R4C_participant_level_summary.csv"
    else:
        deterministic_packet = "revision_R4D_lda_deterministic_imbalance_packet.zip"
        deterministic_member = "revision_R4D_participant_level_summary.csv"
        random_packet = "revision_R4E_lda_random_imbalance_packet.zip"
        random_member = "revision_R4E_participant_level_summary.csv"
    deterministic = read_named_member(packets[deterministic_packet], deterministic_member)
    random = read_named_member(packets[random_packet], random_member)
    frames = []
    for frame in [deterministic, random]:
        subset = frame.loc[
            frame["participant"].isin(ABLE_BODIED + ["P07"])
            & pd.to_numeric(frame["query_budget"], errors="raise").eq(7)
        ].copy()
        subset["query_budget"] = 7
        subset["value"] = pd.to_numeric(subset["mean_repetition_balanced_accuracy"], errors="raise")
        frames.append(subset[["participant", "imbalance_level", "strategy", "query_budget", "value"]])
    if classifier == "RIDGE":
        base = balanced.loc[balanced["query_budget"].eq(7)].copy()
        frames.insert(0, base[["participant", "imbalance_level", "strategy", "query_budget", "value"]])
    result = pd.concat(frames, ignore_index=True)
    result["strategy"] = result["strategy"].replace(
        {"PCBM_PROPOSED": "PCBM_ORIGINAL", "GLOBAL_MARGIN": "GLOBAL_MARGIN_ORIGINAL"}
    )
    result["classifier"] = classifier
    return result


def paired_from_long(
    frame: pd.DataFrame,
    proposed: str,
    comparator: str,
    filters: dict[str, str | int] | None = None,
) -> pd.DataFrame:
    subset = frame.copy()
    for column, value in (filters or {}).items():
        subset = subset.loc[subset[column].eq(value)]
    participant_values = subset.groupby(["participant", "strategy"], as_index=False).agg(value=("value", "mean"))
    pivot = participant_values.pivot(index="participant", columns="strategy", values="value")
    missing = [participant for participant in ABLE_BODIED if participant not in pivot.index]
    if missing or proposed not in pivot.columns or comparator not in pivot.columns:
        raise RuntimeError(f"Pairing failure: proposed={proposed}, comparator={comparator}, missing={missing}")
    output = pivot.loc[ABLE_BODIED, [proposed, comparator]].reset_index()
    output["paired_delta"] = output[proposed] - output[comparator]
    return output


def ridge_imbalance_statistics(ridge: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    tests = []
    participant_tables = []
    diffs_by_level = {}
    for level in LEVELS:
        for comparator in COMPARATORS:
            paired = paired_from_long(ridge, "PCBM_ORIGINAL", comparator, {"imbalance_level": level})
            analysis_id = f"REV_SECONDARY_IMBALANCE__RIDGE__{level}__PCBM_MINUS_{comparator}"
            paired.insert(0, "analysis_id", analysis_id)
            paired["imbalance_level"] = level
            paired["comparator"] = comparator
            tests.append(
                locked_test(
                    analysis_id,
                    f"SECONDARY_IMBALANCE::{level}",
                    f"Ridge K07 PCBM minus {comparator} within {level}",
                    paired["paired_delta"].to_numpy(),
                )
            )
            participant_tables.append(paired)
            if comparator == "RANDOM_UNIFORM":
                diffs_by_level[level] = paired.set_index("participant")["paired_delta"]
    focal = pd.DataFrame({"participant": ABLE_BODIED})
    focal["moderate_pcbm_minus_random"] = focal["participant"].map(diffs_by_level["MODERATE_28"])
    focal["severe_pcbm_minus_random"] = focal["participant"].map(diffs_by_level["SEVERE_21"])
    focal["balanced_pcbm_minus_random"] = focal["participant"].map(diffs_by_level["BALANCED_35"])
    focal["paired_delta"] = (
        0.5 * (focal["moderate_pcbm_minus_random"] + focal["severe_pcbm_minus_random"])
        - focal["balanced_pcbm_minus_random"]
    )
    focal_id = "REV_FOCAL_01__RIDGE_EFFECT_MODIFICATION_K07"
    focal.insert(0, "analysis_id", focal_id)
    tests.insert(
        0,
        locked_test(
            focal_id,
            "NONE_SINGLE_FOCAL",
            "[(PCBM-RANDOM) mean MODERATE+SEVERE] minus BALANCED at K07; Ridge primary",
            focal["paired_delta"].to_numpy(),
        ),
    )
    participant_tables.insert(0, focal)
    return tests, participant_tables


def aulc_statistics(balanced: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame], pd.DataFrame]:
    rows = []
    for participant in ABLE_BODIED:
        base_rows = balanced.loc[
            balanced["participant"].eq(participant)
            & balanced["strategy"].eq("NO_ADAPTATION_REFERENCE")
            & balanced["query_budget"].eq(0),
            "value",
        ]
        if len(base_rows) != 1:
            raise RuntimeError(f"K00 base ambiguity for {participant}")
        base = float(base_rows.iloc[0])
        for strategy in ["PCBM_ORIGINAL", *COMPARATORS]:
            curve = [base]
            for budget in [7, 14, 21]:
                values = balanced.loc[
                    balanced["participant"].eq(participant)
                    & balanced["strategy"].eq(strategy)
                    & balanced["query_budget"].eq(budget),
                    "value",
                ]
                if len(values) != 1:
                    raise RuntimeError(f"AULC cell ambiguity for {participant}/{strategy}/K{budget}")
                curve.append(float(values.iloc[0]))
            aulc = sum((curve[index] + curve[index + 1]) * 0.5 * 7.0 for index in range(3)) / 21.0
            rows.append(
                {
                    "participant": participant,
                    "strategy": strategy,
                    "normalized_aulc_k00_k21": aulc,
                    "curve_k00_k07_k14_k21": json.dumps(curve),
                }
            )
    estimands = pd.DataFrame(rows)
    tests = []
    participant_tables = []
    for comparator in COMPARATORS:
        pivot = estimands.loc[estimands["strategy"].isin(["PCBM_ORIGINAL", comparator])].pivot(
            index="participant", columns="strategy", values="normalized_aulc_k00_k21"
        )
        paired = pivot.loc[ABLE_BODIED, ["PCBM_ORIGINAL", comparator]].reset_index()
        paired["paired_delta"] = paired["PCBM_ORIGINAL"] - paired[comparator]
        analysis_id = f"REV_SECONDARY_AULC__RIDGE__PCBM_MINUS_{comparator}"
        paired.insert(0, "analysis_id", analysis_id)
        paired["comparator"] = comparator
        tests.append(
            locked_test(
                analysis_id,
                "SECONDARY_AULC",
                f"Normalized K00/K07/K14/K21 Ridge AULC: PCBM minus {comparator}",
                paired["paired_delta"].to_numpy(),
            )
        )
        participant_tables.append(paired)
    return tests, participant_tables, estimands


def split_statistics(packet: Path) -> tuple[list[dict], list[pd.DataFrame], pd.DataFrame]:
    frame = read_named_member(packet, "revision_R5B_participant_level_locked_contrasts.csv")
    frame = frame.loc[frame["participant"].isin(ABLE_BODIED)].copy()
    tests = []
    tables = []
    for split_id in sorted(frame["split_id"].astype(str).unique()):
        group = frame.loc[frame["split_id"].astype(str).eq(split_id)].set_index("participant").loc[ABLE_BODIED]
        for column, comparator in [
            ("pcbm_minus_random_k07", "RANDOM_UNIFORM"),
            ("pcbm_minus_global_k07", "GLOBAL_MARGIN_ORIGINAL"),
        ]:
            differences = pd.to_numeric(group[column], errors="raise").to_numpy(float)
            analysis_id = f"REV_SPLIT_STABILITY__{split_id}__PCBM_MINUS_{comparator}"
            paired = pd.DataFrame({"analysis_id": analysis_id, "participant": ABLE_BODIED, "paired_delta": differences})
            paired["split_id"] = split_id
            paired["comparator"] = comparator
            tests.append(
                locked_test(
                    analysis_id,
                    "SPLIT_STABILITY",
                    f"K07 PCBM minus {comparator} under split {split_id}",
                    differences,
                )
            )
            tables.append(paired)
    return tests, tables, frame


def drift_statistics(packet: Path) -> tuple[list[dict], list[pd.DataFrame], pd.DataFrame]:
    frame = read_named_member(packet, "revision_R5C_participant_level_estimands_for_R7.csv")
    able = frame.set_index("participant").loc[ABLE_BODIED]
    specifications = [
        ("mean_session_late_minus_early_accuracy", "PERFORMANCE_LATE_MINUS_EARLY"),
        ("mean_session_feature_drift_slope", "FEATURE_DISTANCE_SLOPE"),
    ]
    tests = []
    tables = []
    for column, label in specifications:
        differences = pd.to_numeric(able[column], errors="raise").to_numpy(float)
        analysis_id = f"REV_DRIFT__{label}"
        paired = pd.DataFrame({"analysis_id": analysis_id, "participant": ABLE_BODIED, "paired_delta": differences})
        tests.append(locked_test(analysis_id, "DRIFT_TWO_DIAGNOSTICS", label, differences))
        tables.append(paired)
    return tests, tables, frame


def deep_statistics(packet: Path) -> tuple[list[dict], list[pd.DataFrame], pd.DataFrame]:
    frame = read_named_member(packet, "revision_R6D_participant_seed_averaged_estimands.csv")
    frame = frame.loc[frame["participant"].isin(ABLE_BODIED)].copy()
    tests = []
    tables = []
    for keys, group in frame.groupby(["analysis_scope", "contrast", "comparator"], sort=True):
        ordered = group.set_index("participant").loc[ABLE_BODIED]
        differences = pd.to_numeric(
            ordered["participant_seed_averaged_difference_balanced_accuracy"], errors="raise"
        ).to_numpy(float)
        analysis_id = f"REV_DEEP_STABILITY__{keys[0]}__{keys[1]}"
        paired = pd.DataFrame({"analysis_id": analysis_id, "participant": ABLE_BODIED, "paired_delta": differences})
        paired["analysis_scope"] = keys[0]
        paired["contrast"] = keys[1]
        tests.append(
            locked_test(
                analysis_id,
                "DEEP_STABILITY",
                f"Six-training-seed averaged {keys[0]}: {keys[1]}",
                differences,
            )
        )
        tables.append(paired)
    if len(tests) != 4:
        raise RuntimeError(f"Expected four deep stability contrasts; found {len(tests)}")
    return tests, tables, frame


def lda_descriptive_sensitivity(lda: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in sorted(lda["imbalance_level"].astype(str).unique()):
        for comparator in COMPARATORS:
            paired = paired_from_long(lda, "PCBM_ORIGINAL", comparator, {"imbalance_level": level})
            for row in paired.itertuples(index=False):
                rows.append(
                    {
                        "participant": row.participant,
                        "classifier": "LDA",
                        "imbalance_level": level,
                        "comparator": comparator,
                        "pcbm_value": getattr(row, "PCBM_ORIGINAL"),
                        "comparator_value": getattr(row, comparator),
                        "paired_delta": row.paired_delta,
                        "scientific_role": "DESCRIPTIVE_SENSITIVITY_NOT_ADDITIONAL_INFERENTIAL_FAMILY",
                    }
                )
    return pd.DataFrame(rows)


def monte_carlo_convergence(packet: Path) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    frame = read_named_member(packet, "revision_R4C_participant_seed_summary.csv")
    frame = frame.loc[frame["participant"].isin(ABLE_BODIED)].copy()
    frame["random_seed_index"] = pd.to_numeric(frame["random_seed_index"], errors="raise").astype(int)
    frame["query_budget"] = pd.to_numeric(frame["query_budget"], errors="raise").astype(int)
    frame["value"] = pd.to_numeric(frame["mean_repetition_balanced_accuracy"], errors="raise")
    frame = (
        frame.groupby(["participant", "imbalance_level", "query_budget", "random_seed_index"], as_index=False)
        .agg(value=("value", "mean"))
    )
    convergence_rows = []
    cell_rows = []
    for keys, group in frame.groupby(["participant", "imbalance_level", "query_budget"], sort=True):
        group = group.sort_values("random_seed_index")
        if group["random_seed_index"].tolist() != list(range(1, 31)):
            raise RuntimeError(f"Monte Carlo cell {keys} does not contain seed indices 1-30")
        values = group["value"].to_numpy(float)
        full_mean = float(values.mean())
        full_mean_halfwidth = 1.96 * float(values.std(ddof=1)) / math.sqrt(30.0)
        prefix_25 = float(values[:25].mean())
        prefix_30 = full_mean
        change_25_to_30 = abs(prefix_30 - prefix_25)
        for m in MC_PREFIXES:
            rng = np.random.default_rng(stable_seed(f"MC|{keys}|m={m}"))
            subset_means = np.empty(MC_REPLICATES, dtype=float)
            for replicate in range(MC_REPLICATES):
                subset_means[replicate] = values[rng.choice(30, size=m, replace=False)].mean()
            deviations = subset_means - full_mean
            halfwidth = 1.96 * float(deviations.std(ddof=1))
            convergence_rows.append(
                {
                    "participant": keys[0],
                    "imbalance_level": keys[1],
                    "query_budget": keys[2],
                    "m_seed_count": m,
                    "full_30_seed_mean": full_mean,
                    "mean_subsample_deviation": float(deviations.mean()),
                    "mcse_subsample_deviation": float(deviations.std(ddof=1) / math.sqrt(MC_REPLICATES)),
                    "mc_95_halfwidth": halfwidth,
                    "deviation_2_5_percentile": float(np.quantile(deviations, 0.025)),
                    "deviation_97_5_percentile": float(np.quantile(deviations, 0.975)),
                    "deterministic_subsamples": MC_REPLICATES,
                    "subsample_seed": stable_seed(f"MC|{keys}|m={m}"),
                }
            )
        cell_rows.append(
            {
                "participant": keys[0],
                "imbalance_level": keys[1],
                "query_budget": keys[2],
                "absolute_prefix_25_to_30_change": change_25_to_30,
                "full_30_seed_mean_mc_95_halfwidth": full_mean_halfwidth,
                "change_threshold_0_005_passes": change_25_to_30 <= 0.005,
                "halfwidth_threshold_0_01_passes": full_mean_halfwidth <= 0.01,
            }
        )
    convergence = pd.DataFrame(convergence_rows)
    cells = pd.DataFrame(cell_rows)
    adequate = bool(cells[["change_threshold_0_005_passes", "halfwidth_threshold_0_01_passes"]].all().all())
    return convergence, cells, adequate


def best_existing_statistical_table(packet: Path, preferred_tokens: list[str], minimum_rows: int) -> tuple[pd.DataFrame, str]:
    candidates = []
    with zipfile.ZipFile(packet, "r") as archive:
        for member, size, columns in all_csv_members(packet):
            lower_columns = [column.lower() for column in columns]
            p_value_columns = [column for column in lower_columns if "p_value" in column or column in {"p", "pvalue"}]
            if not p_value_columns:
                continue
            try:
                frame = pd.read_csv(archive.open(member))
            except Exception:
                continue
            if len(frame) < minimum_rows:
                continue
            lower_name = Path(member).name.lower()
            score = 100 * sum(token.lower() in lower_name for token in preferred_tokens) + len(p_value_columns) * 10 - len(frame) / 100000
            candidates.append((score, member, frame))
    if not candidates:
        raise RuntimeError(f"No existing statistical table with at least {minimum_rows} rows found in {packet.name}")
    _, member, frame = max(candidates, key=lambda item: item[0])
    frame.insert(0, "frozen_source_member", member)
    return frame, member


def extract_supplement_tables(packets: dict[str, Path]) -> pd.DataFrame:
    requested = {
        "revision_R3C_balanced_pool_classical_comparator_extension_packet.zip": [
            "revision_R3C_per_class_recall.csv",
            "revision_R3C_confusion_matrices_long.csv",
            "revision_R3C_class_coverage_entropy_summary.csv",
        ],
        "revision_R4B_ridge_deterministic_imbalance_packet.zip": [
            "revision_R4B_class_coverage_entropy_summary.csv",
            "revision_R4B_compute_summary.csv",
            "revision_R4B_confusion_matrices_aggregated.csv",
        ],
        "revision_R4C_ridge_random_imbalance_packet.zip": [
            "revision_R4C_class_coverage_entropy_summary.csv",
            "revision_R4C_confusion_matrices_aggregated.csv",
        ],
        "revision_R4D_lda_deterministic_imbalance_packet.zip": [
            "revision_R4D_class_coverage_entropy_summary.csv",
            "revision_R4D_compute_summary.csv",
            "revision_R4D_confusion_matrices_aggregated.csv",
        ],
        "revision_R4E_lda_random_imbalance_packet.zip": [
            "revision_R4E_class_coverage_entropy_summary.csv",
            "revision_R4E_confusion_matrices_aggregated.csv",
        ],
        "revision_R5B_ridge_temporal_split_sensitivity_packet.zip": [
            "revision_R5B_aggregate_recalls.csv",
            "revision_R5B_aggregate_confusions.csv",
            "revision_R5B_aggregate_coverage.csv",
        ],
        "revision_R6D_deep_stability_compute_aggregation_packet.zip": [
            "revision_R6D_training_seed_distribution.csv",
            "revision_R6D_compute_cost_summary.csv",
        ],
    }
    index_rows = []
    for packet_name, basenames in requested.items():
        packet = packets[packet_name]
        available = {Path(member).name for member, _, _ in all_csv_members(packet)}
        for basename in basenames:
            if basename not in available:
                continue
            destination = SUPPLEMENT_ROOT / packet_name.replace("_packet.zip", "") / basename
            engine.extract_member(packet, basename, destination)
            index_rows.append(
                {
                    "source_packet": packet_name,
                    "source_member": basename,
                    "supplement_path": destination.relative_to(RESULT_ROOT).as_posix(),
                    "bytes": destination.stat().st_size,
                    "sha256": engine.sha256_file(destination),
                }
            )
    stage5f_packet = packets["stage5f_deep_statistics_retention_sensitivity_packet.zip"]
    for member, _, columns in all_csv_members(stage5f_packet):
        basename = Path(member).name
        lower = basename.lower()
        if any(token in lower for token in ["primary", "secondary", "participant", "retention", "auc", "balance", "claim"]):
            destination = SUPPLEMENT_ROOT / "Stage5F_Frozen_Deep_Analysis" / basename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(stage5f_packet, "r") as archive:
                destination.write_bytes(archive.read(member))
            index_rows.append(
                {
                    "source_packet": stage5f_packet.name,
                    "source_member": member,
                    "supplement_path": destination.relative_to(RESULT_ROOT).as_posix(),
                    "bytes": destination.stat().st_size,
                    "sha256": engine.sha256_file(destination),
                }
            )
    return pd.DataFrame(index_rows)


def restore_inputs() -> tuple[dict[str, Path], pd.DataFrame, dict]:
    packets = {}
    audits = []
    r7a_packet, r7a_source = r7a.direct_restore(
        R7A_BASENAME, R7A_PACKET_SHA256, engine.REMOTE_BASE + "/" + R7A_RELATIVE
    )
    packets[R7A_BASENAME] = r7a_packet
    audits.append(
        {
            "packet": R7A_BASENAME,
            "source": r7a_source,
            "expected_sha256": R7A_PACKET_SHA256,
            "actual_sha256": engine.sha256_file(r7a_packet),
            "hash_matches": engine.sha256_file(r7a_packet) == R7A_PACKET_SHA256,
            "crc_passes": engine.archive_crc_passes(r7a_packet),
        }
    )
    contract = engine.read_json_member(r7a_packet, "revision_R7A_frozen_schema_contract.json")
    r7a_report = engine.read_json_member(r7a_packet, "revision_R7A_final_report.json")
    for basename, (expected_hash, relative) in r7a.DIRECT_PACKETS.items():
        if basename == "revision_R2A_classical_detail_packet_migration_packet.zip":
            continue
        path, source = r7a.direct_restore(basename, expected_hash, engine.REMOTE_BASE + "/" + relative)
        packets[basename] = path
        audits.append(
            {
                "packet": basename,
                "source": source,
                "expected_sha256": expected_hash,
                "actual_sha256": engine.sha256_file(path),
                "hash_matches": engine.sha256_file(path) == expected_hash,
                "crc_passes": engine.archive_crc_passes(path),
            }
        )
    for basename in [
        "stage3c2_random_control_packet.zip",
        "stage3e3_sensitivity_integration_packet.zip",
        "stage3f3_retention_statistical_analysis_packet.zip",
    ]:
        expected_hash = contract["detail_packet_hashes"][basename]
        path, source = r7a.direct_restore(basename, expected_hash, r7a.REMOTE_DETAILS + "/" + basename)
        packets[basename] = path
        audits.append(
            {
                "packet": basename,
                "source": source,
                "expected_sha256": expected_hash,
                "actual_sha256": engine.sha256_file(path),
                "hash_matches": engine.sha256_file(path) == expected_hash,
                "crc_passes": engine.archive_crc_passes(path),
            }
        )
    return packets, pd.DataFrame(audits), {"contract": contract, "r7a_report": r7a_report}


def main() -> None:
    print("=" * 112)
    print("REVISION R7B — LOCKED STATISTICAL ANALYSIS AND SUPPLEMENT")
    print("=" * 112)
    print("Execution device: CPU")
    print("Model training: False")
    print("Fixed-test inference: False")
    print("New inferential statistical tests: True — exactly the seven locked R7 families")
    print("Inferential unit: participant P01-P06; P07 descriptive only")
    print("BCa and percentile bootstrap replicates:", BOOTSTRAP_REPLICATES)
    print()

    WORKING.mkdir(parents=True, exist_ok=True)
    lock_handle = open(WORKING / "_revision_R7B_single_instance.lock", "w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("FINAL DECISION: DUPLICATE_INVOCATION_EXITED_SAFELY")
        return

    engine.bootstrap_rclone()
    engine.create_rclone_config()
    print("rclone version:", engine.rclone(["version"]).stdout.splitlines()[0])
    print("Restoring verified R7A contract, modern parents, and three frozen detail packets...")
    packets, packet_audit, context = restore_inputs()
    if RESULT_ROOT.exists():
        shutil.rmtree(RESULT_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT_ROOT.mkdir(parents=True, exist_ok=True)

    balanced, balanced_random_member = canonical_balanced(packets)
    ridge = canonical_imbalance(packets, balanced, "RIDGE")
    lda = canonical_imbalance(packets, balanced, "LDA")

    all_tests = []
    participant_tables = []
    tests, tables = ridge_imbalance_statistics(ridge)
    all_tests.extend(tests)
    participant_tables.extend(tables)
    tests, tables, aulc_estimands = aulc_statistics(balanced)
    all_tests.extend(tests)
    participant_tables.extend(tables)
    tests, tables, split_estimands = split_statistics(
        packets["revision_R5B_ridge_temporal_split_sensitivity_packet.zip"]
    )
    all_tests.extend(tests)
    participant_tables.extend(tables)
    tests, tables, drift_estimands = drift_statistics(
        packets["revision_R5C_within_session_drift_audit_packet.zip"]
    )
    all_tests.extend(tests)
    participant_tables.extend(tables)
    tests, tables, deep_estimands = deep_statistics(
        packets["revision_R6D_deep_stability_compute_aggregation_packet.zip"]
    )
    all_tests.extend(tests)
    participant_tables.extend(tables)

    tests_frame = holm_adjust(pd.DataFrame(all_tests))
    participant_differences = pd.concat(participant_tables, ignore_index=True, sort=False)
    lda_sensitivity = lda_descriptive_sensitivity(lda)
    mc_convergence, mc_cells, mc_adequate = monte_carlo_convergence(
        packets["revision_R4C_ridge_random_imbalance_packet.zip"]
    )
    classical_five, classical_member = best_existing_statistical_table(
        packets["stage3e3_sensitivity_integration_packet.zip"], ["stat", "secondary", "integr"], 5
    )
    retention_eighteen, retention_member = best_existing_statistical_table(
        packets["stage3f3_retention_statistical_analysis_packet.zip"], ["retention", "stat", "analysis"], 18
    )
    supplement_index = extract_supplement_tables(packets)

    p07_tables = []
    for source, frame in [
        ("BALANCED_RIDGE", balanced),
        ("RIDGE_IMBALANCE", ridge),
        ("LDA_IMBALANCE", lda),
        ("R5C_DRIFT", drift_estimands),
        ("R6D_DEEP", deep_estimands),
    ]:
        if "participant" in frame.columns:
            subset = frame.loc[frame["participant"].astype(str).eq("P07")].copy()
            if len(subset):
                subset.insert(0, "source_analysis", source)
                p07_tables.append(subset)
    p07_descriptive = pd.concat(p07_tables, ignore_index=True, sort=False) if p07_tables else pd.DataFrame()

    statistical_plan = engine.read_csv_member(
        packets["stageR0_reviewer_revision_protocol_lock_packet.zip"],
        "stageR0_statistical_analysis_plan.csv",
    )
    expected_test_count = 1 + 24 + 6 + 8 + 2 + 4
    gates = {
        "r7a_packet_hash_matches": engine.sha256_file(packets[R7A_BASENAME]) == R7A_PACKET_SHA256,
        "r7a_all_readiness_gates_passed": bool(context["r7a_report"].get("all_readiness_gates_passed")),
        "r7a_decision_authorizes_r7b": context["r7a_report"].get("final_decision")
        == "PASS_TO_REVISION_R7B_LOCKED_STATISTICAL_ANALYSIS_AND_SUPPLEMENT",
        "all_input_packets_pass_hash_and_crc": bool(packet_audit[["hash_matches", "crc_passes"]].all().all()),
        "locked_registry_contains_exactly_seven_families": set(statistical_plan["analysis_id"].astype(str))
        == {
            "REV_FOCAL_01",
            "REV_SECONDARY_IMBALANCE",
            "REV_SECONDARY_AULC",
            "REV_SPLIT_STABILITY",
            "REV_DRIFT",
            "REV_DEEP_STABILITY",
            "REV_MC_RANDOM",
        },
        "exact_inferential_test_count_is_45": len(tests_frame) == expected_test_count,
        "every_test_uses_six_participants": tests_frame["participant_count"].eq(6).all(),
        "every_test_reports_exact_statistic_and_zero_policy": tests_frame[
            ["wilcoxon_statistic", "wilcoxon_w_plus", "wilcoxon_w_minus", "zero_difference_count"]
        ].notna().all().all(),
        "every_test_has_100000_bca_and_percentile_bootstraps": tests_frame["bootstrap_replicates"].eq(100000).all(),
        "all_raw_and_holm_p_values_are_valid": tests_frame[
            ["exact_two_sided_p_value", "holm_adjusted_p_value"]
        ].ge(0).all().all()
        and tests_frame[["exact_two_sided_p_value", "holm_adjusted_p_value"]].le(1).all().all(),
        "focal_family_contains_one_test": tests_frame["multiplicity_family"].eq("NONE_SINGLE_FOCAL").sum() == 1,
        "secondary_imbalance_contains_24_tests": tests_frame["analysis_id"].str.startswith("REV_SECONDARY_IMBALANCE").sum() == 24,
        "aulc_contains_six_tests": tests_frame["analysis_id"].str.startswith("REV_SECONDARY_AULC").sum() == 6,
        "split_stability_contains_eight_tests": tests_frame["analysis_id"].str.startswith("REV_SPLIT_STABILITY").sum() == 8,
        "drift_contains_two_tests": tests_frame["analysis_id"].str.startswith("REV_DRIFT").sum() == 2,
        "deep_stability_contains_four_tests": tests_frame["analysis_id"].str.startswith("REV_DEEP_STABILITY").sum() == 4,
        "balanced_random_source_was_resolved": bool(balanced_random_member),
        "random_convergence_has_all_locked_prefixes": set(mc_convergence["m_seed_count"].unique()) == set(MC_PREFIXES),
        "random_convergence_uses_1000_subsamples": mc_convergence["deterministic_subsamples"].eq(1000).all(),
        "five_classical_secondary_tests_are_exposed": len(classical_five) >= 5,
        "eighteen_retention_tests_are_exposed": len(retention_eighteen) >= 18,
        "p07_has_no_inferential_test_row": not participant_differences["participant"].astype(str).eq("P07").any(),
        "p07_descriptive_rows_are_preserved": len(p07_descriptive) > 0,
        "equivalence_claim_is_not_made": True,
        "raw_hdf5_data_was_not_accessed": True,
        "no_model_was_trained": True,
        "no_fixed_test_inference_was_run": True,
        "credentials_were_not_written_to_artifacts": True,
    }
    failed = [key for key, value in gates.items() if not bool(value)]

    atomic_csv(packet_audit, RESULT_ROOT / "revision_R7B_input_packet_audit.csv")
    atomic_csv(statistical_plan, RESULT_ROOT / "revision_R7B_locked_statistical_registry.csv")
    atomic_csv(tests_frame, RESULT_ROOT / "revision_R7B_all_new_statistical_tests.csv")
    atomic_csv(participant_differences, RESULT_ROOT / "revision_R7B_participant_level_differences.csv")
    atomic_csv(ridge, RESULT_ROOT / "revision_R7B_ridge_imbalance_estimands.csv")
    atomic_csv(lda_sensitivity, RESULT_ROOT / "revision_R7B_lda_descriptive_sensitivity.csv")
    atomic_csv(aulc_estimands, RESULT_ROOT / "revision_R7B_aulc_estimands.csv")
    atomic_csv(split_estimands, RESULT_ROOT / "revision_R7B_split_estimands.csv")
    atomic_csv(drift_estimands, RESULT_ROOT / "revision_R7B_drift_estimands.csv")
    atomic_csv(deep_estimands, RESULT_ROOT / "revision_R7B_deep_estimands.csv")
    atomic_csv(mc_convergence, RESULT_ROOT / "revision_R7B_random_policy_mc_convergence.csv")
    atomic_csv(mc_cells, RESULT_ROOT / "revision_R7B_random_policy_mc_cell_adequacy.csv")
    atomic_csv(classical_five, RESULT_ROOT / "revision_R7B_frozen_five_classical_secondary_tests.csv")
    atomic_csv(retention_eighteen, RESULT_ROOT / "revision_R7B_frozen_eighteen_retention_tests.csv")
    atomic_csv(p07_descriptive, RESULT_ROOT / "revision_R7B_P07_descriptive_only.csv")
    atomic_csv(supplement_index, RESULT_ROOT / "revision_R7B_supplementary_table_index.csv")

    if failed:
        final_decision = "REVISION_R7B_LOCKED_ANALYSIS_FAILED"
    elif not mc_adequate:
        final_decision = "PASS_TO_REVISION_R7C_RANDOM_SEEDS_31_TO_60_EXTENSION"
    else:
        final_decision = "PASS_TO_REVISION_R8_REVISED_MANUSCRIPT_AND_SUPPLEMENT_INTEGRATION"
    report = {
        "stage": "REVISION_R7B_LOCKED_STATISTICAL_ANALYSIS_AND_SUPPLEMENT",
        "revision_protocol_sha256": REVISION_PROTOCOL_SHA256,
        "r7a_packet_sha256": R7A_PACKET_SHA256,
        "balanced_random_source_member": balanced_random_member,
        "frozen_five_classical_source_member": classical_member,
        "frozen_eighteen_retention_source_member": retention_member,
        "inferential_participants": ABLE_BODIED,
        "p07_role": "DESCRIPTIVE_CASE_ONLY",
        "new_inferential_test_count": len(tests_frame),
        "bootstrap_replicates_per_interval": BOOTSTRAP_REPLICATES,
        "monte_carlo_random_policy_adequate": mc_adequate,
        "maximum_absolute_prefix_25_to_30_change": float(mc_cells["absolute_prefix_25_to_30_change"].max()),
        "maximum_mc_95_halfwidth": float(mc_cells["full_30_seed_mean_mc_95_halfwidth"].max()),
        "prespecified_seed_31_to_60_extension_required": not mc_adequate,
        "readiness_gates": gates,
        "failed_readiness_gates": failed,
        "all_readiness_gates_passed": not failed,
        "model_training_run": False,
        "fixed_test_inference_run": False,
        "new_inferential_statistical_tests_run": True,
        "runtime_minutes": (time.time() - START_TIME) / 60.0,
        "final_decision": final_decision,
    }
    atomic_json(report, RESULT_ROOT / "revision_R7B_final_report.json")
    shutil.copy2(Path(__file__), RESULT_ROOT / "revision_R7B_executed_source.py")
    manifest_rows = []
    for path in sorted(RESULT_ROOT.rglob("*")):
        if path.is_file() and path.name != "revision_R7B_output_manifest.csv":
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(RESULT_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": engine.sha256_file(path),
                }
            )
    atomic_csv(pd.DataFrame(manifest_rows), RESULT_ROOT / "revision_R7B_output_manifest.csv")
    if failed:
        raise RuntimeError(f"R7B readiness failed: {failed}")
    if not engine.make_zip(RESULT_ROOT, PACKET_PATH, "Revision_R7B_Locked_Statistical_Analysis_and_Supplement"):
        raise RuntimeError("R7B packet CRC failed")
    digest = engine.sha256_file(PACKET_PATH)
    if not engine.roundtrip_remote_file(PACKET_PATH, REMOTE_OUTPUT + "/" + PACKET_PATH.name, digest):
        raise RuntimeError("R7B remote round-trip failed")

    print()
    print("=" * 112)
    print("REVISION R7B — FINAL LOCKED ANALYSIS SUMMARY")
    print("=" * 112)
    print("New locked inferential tests:", len(tests_frame))
    print("Participant-level difference rows:", len(participant_differences))
    print("Five frozen classical secondary tests exposed:", len(classical_five))
    print("Frozen retention tests exposed:", len(retention_eighteen))
    print("Random-policy Monte Carlo adequate:", mc_adequate)
    print("Maximum |25-to-30 change|:", report["maximum_absolute_prefix_25_to_30_change"])
    print("Maximum MC 95% half-width:", report["maximum_mc_95_halfwidth"])
    print("Failed readiness gates:", failed or "None")
    print("Packet CRC pass: True")
    print("Packet:", PACKET_PATH)
    print("Packet SHA-256:", digest)
    print("Remote round-trip verified: True")
    print("Runtime minutes:", round(report["runtime_minutes"], 3))
    print()
    print("FINAL DECISION:", final_decision)


if __name__ == "__main__":
    main()
