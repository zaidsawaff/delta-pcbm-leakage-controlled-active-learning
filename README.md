# Leakage-Controlled Active Learning for Longitudinal HD-sEMG

This repository accompanies the study **“Predicted-Class-Balanced Active Learning for Leakage-Controlled Longitudinal Adaptation of High-Density Surface Electromyography: Classical and Deep-Learning Evidence from the DELTA Dataset.”**

The study evaluates Predicted-Class-Balanced Margin (PCBM), a deterministic low-budget acquisition heuristic for longitudinal adaptation of high-density surface electromyography (HD-sEMG) classifiers. The protocol isolates every fixed test from model fitting, normalization, calibration, and query selection.

## Main result

PCBM increased low-budget acquisition coverage and label entropy, but the complete participant-level evidence did **not** establish a robust predictive or retention advantage over the prespecified comparators. None of the 45 locked analyses met its declared significance criterion. The repository is therefore intended to support transparent verification of a bounded, predominantly negative result rather than a superiority claim.

## Repository contents

| Path | Purpose |
|---|---|
| `src/delta_pcbm/` | Clean reference implementation of PCBM and global-margin selection |
| `tests/` | Synthetic, data-free selector tests |
| `audit/` | TCN architecture and padding audit |
| `verification/` | Synthetic verification of the locked TCN padding contract |
| `provenance/executed_sources/` | Available executed analysis sources preserved with their historical names |
| `provenance/SOURCE_INDEX.csv` | Role of each preserved source program |
| `DATA_ACCESS.md` | DELTA dataset access and non-redistribution statement |
| `REPRODUCIBILITY.md` | Reproduction levels, limitations, and commands |
| `THIRD_PARTY_NOTICES.md` | License boundaries for external data and dependencies |

Raw DELTA recordings, participant-level source files, credentials, private storage configuration, manuscript files, and reviewer correspondence are not included.

## Quick verification

Create the environment:

```bash
conda env create -f environment.yml
conda activate delta-pcbm
```

Run the data-free selector tests:

```bash
python -m unittest discover -s tests -v
```

Verify the TCN layer contract:

```bash
python verification/verify_tcn_padding_contract.py
```

Verify the release checksums:

```bash
sha256sum -c SHA256SUMS.txt
```

## PCBM rule

At each seven-query round, candidates are ordered by ascending top-two decision-score margin. The first pass nominates at most one lowest-margin candidate from each represented predicted class. If fewer than seven classes are represented, the remaining positions are filled by the globally lowest-margin unselected candidates. Ties are resolved deterministically by opaque candidate token.

The selector receives only:

- an opaque candidate token;
- the current model's predicted class; and
- the top-two decision-score margin.

True labels, fixed-test membership, participant identity, session identity, and future information are not selector-visible.

## TCN reporting contract

The audited temporal convolutional network uses ordinary PyTorch `Conv1d` layers with kernel size 3 and `padding=dilation` at dilations 1, 2, 4, and 8. This is symmetric left/right zero padding. The model is therefore **dilated and noncausal within each complete 37-window repetition**. It does not use future repetitions, fixed-test labels, or future sessions, and no online or real-time causal inference claim is made.

## Reproducibility boundary

The clean selector implementation and TCN contract can be verified without private data. The preserved historical stage programs require named, hash-verified parent evidence packets that are not redistributed here. They are provided as provenance of the available executed source, not as a claim of a self-contained raw-data-to-paper workflow. See `REPRODUCIBILITY.md`.

## Data

The public DELTA dataset is available from Zenodo:

- DOI: [10.5281/zenodo.10801000](https://doi.org/10.5281/zenodo.10801000)

Users must obtain the data directly from the original provider and comply with its license and access conditions.

## License

Study-generated software in this repository is released under the MIT License. The license does not apply to the DELTA dataset, third-party software, manuscript text, or external annotations. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Citation

A version-specific DOI will be minted through Zenodo for the GitHub release `v1.0.0`. Until the DOI is issued, cite the software metadata in `CITATION.cff`:

> Al-Sawaff, Z. H. (2026). *DELTA PCBM: Leakage-Controlled Active Learning for Longitudinal HD-sEMG* (Version 1.0.0) [Computer software].

Author ORCID: [0000-0001-8789-4905](https://orcid.org/0000-0001-8789-4905).
