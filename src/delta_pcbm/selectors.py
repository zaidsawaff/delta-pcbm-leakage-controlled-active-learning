"""Data-free reference implementation of the locked acquisition rules.

This module exposes no labels, participant identifiers, session identifiers,
fixed-test membership, or future information to the selectors.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


SELECTOR_COLUMNS = [
    "opaque_candidate_token",
    "predicted_label",
    "margin",
]
TOKEN_PATTERN = re.compile(r"^[0-9a-f]{24}$")


def _validate_selector_frame(frame: pd.DataFrame, batch_size: int) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.columns.tolist() != SELECTOR_COLUMNS:
        raise ValueError(
            f"selector schema must be exactly {SELECTOR_COLUMNS}; "
            f"received {frame.columns.tolist()}"
        )
    if not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    if len(frame) < int(batch_size):
        raise ValueError("candidate pool is smaller than batch_size")
    tokens = frame["opaque_candidate_token"].astype(str)
    if tokens.duplicated().any():
        raise ValueError("duplicate opaque candidate token")
    if not tokens.map(lambda token: TOKEN_PATTERN.fullmatch(token) is not None).all():
        raise ValueError("opaque candidate tokens must be 24 lowercase hex characters")
    margins = frame["margin"].to_numpy(dtype=float)
    if not np.isfinite(margins).all():
        raise ValueError("selector margins must be finite")
    if frame["predicted_label"].isna().any():
        raise ValueError("predicted labels must not be missing")


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["margin", "opaque_candidate_token"],
        kind="mergesort",
    )


def select_pcbm(frame: pd.DataFrame, batch_size: int = 7) -> list[str]:
    """Select a deterministic predicted-class-balanced margin batch.

    The lowest-margin item in each represented predicted class is nominated
    first. Nominees are ordered globally by margin and opaque token. Remaining
    batch positions are filled by the lowest-margin unselected candidates.
    """

    _validate_selector_frame(frame, batch_size)
    ordered = _ordered(frame)
    nominees = (
        ordered.groupby("predicted_label", sort=False, as_index=False)
        .head(1)
        .sort_values(["margin", "opaque_candidate_token"], kind="mergesort")
    )
    selected = nominees.head(int(batch_size))["opaque_candidate_token"].tolist()
    if len(selected) < int(batch_size):
        fill = ordered.loc[
            ~ordered["opaque_candidate_token"].isin(selected),
            "opaque_candidate_token",
        ].tolist()
        selected.extend(fill[: int(batch_size) - len(selected)])
    if len(selected) != int(batch_size) or len(set(selected)) != int(batch_size):
        raise RuntimeError("PCBM failed to select a unique complete batch")
    return [str(token) for token in selected]


def select_global_margin(frame: pd.DataFrame, batch_size: int = 7) -> list[str]:
    """Select the globally lowest-margin candidates deterministically."""

    _validate_selector_frame(frame, batch_size)
    selected = _ordered(frame).head(int(batch_size))["opaque_candidate_token"].tolist()
    if len(selected) != int(batch_size) or len(set(selected)) != int(batch_size):
        raise RuntimeError("global margin failed to select a unique complete batch")
    return [str(token) for token in selected]

