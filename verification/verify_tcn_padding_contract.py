"""Synthetic verification of the audited TCN padding and parameter contract.

This file is a reviewer-facing verification utility, not a claim that the
standalone executed Stage 5B model source was recovered from the frozen packet.
"""

from __future__ import annotations

import torch
from torch import nn


class AuditedResidualBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            128, 128, kernel_size=3, dilation=dilation,
            padding=dilation, groups=128, bias=False,
        )
        self.pointwise = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, 128, eps=1e-5)
        self.dropout = nn.Dropout(0.15)


class AuditedParameterContract(nn.Module):
    """Parameter-bearing modules listed in manuscript Table 2."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128, eps=1e-5),
        )
        self.blocks = nn.ModuleList(
            [AuditedResidualBlock(d) for d in (1, 2, 4, 8)]
        )
        self.attention_score = nn.Conv1d(128, 1, kernel_size=1, bias=True)
        self.classifier = nn.Linear(128, 7, bias=True)


def demonstrate_future_within_repetition_dependency() -> float:
    """Return output at q=10 caused solely by an input at q=11."""

    conv = nn.Conv1d(1, 1, kernel_size=3, dilation=1, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 2] = 1.0
    x = torch.zeros(1, 1, 37)
    x[0, 0, 11] = 1.0
    return float(conv(x)[0, 0, 10])


def main() -> None:
    model = AuditedParameterContract()
    parameter_count = sum(p.numel() for p in model.parameters())
    future_dependency = demonstrate_future_within_repetition_dependency()
    assert parameter_count == 118_536, parameter_count
    assert future_dependency == 1.0, future_dependency
    print("parameter_count=118536")
    print("future_within_repetition_dependency_at_q10_from_q11=1.0")
    print("classification=DILATED_NONCAUSAL_WITHIN_COMPLETE_REPETITION")


if __name__ == "__main__":
    main()

