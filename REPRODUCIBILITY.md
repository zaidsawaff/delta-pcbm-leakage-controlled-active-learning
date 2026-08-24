# Reproducibility guide

This repository distinguishes three reproducibility levels so that verification claims remain precise.

## Level 1 — release integrity

No data or Python environment is required.

```bash
sha256sum -c SHA256SUMS.txt
```

This verifies that every versioned release file matches the release manifest.

## Level 2 — data-free method verification

Create the environment and run:

```bash
python -m unittest discover -s tests -v
python verification/verify_tcn_padding_contract.py
```

The selector tests verify:

- strict selector-visible columns;
- deterministic tie handling;
- one first-pass nominee per represented predicted class;
- global-margin filling when fewer than seven classes are represented;
- seven unique selected opaque tokens; and
- rejection of duplicate, malformed, or non-finite inputs.

The TCN utility verifies the locked 118,536-parameter architecture and demonstrates that symmetric padding permits a later window in the same complete repetition to affect an earlier output.

## Level 3 — historical analysis replay

The programs in `provenance/executed_sources/` preserve the available source files from the scientific execution lineage. Many stages restore named, hash-verified parent packets from the original execution environment. Those parent packets and private remote-storage configuration are not included.

Consequently, the preserved programs support source inspection, contract verification, and replay when the required parent evidence is available; they do not constitute a self-contained raw-data-to-manuscript workflow.

The standalone executed module historically named `stage5b_mask_aware_rms_tcn.py` was not present in the final local assembly workspace. It has not been reconstructed or falsely represented as the executed file. The supplied architecture audit and synthetic verifier reproduce the locked layer contract and its padding semantics.

## Environment

The environment records the principal versions used by the packaged audit and analysis sources:

- Python 3.12
- NumPy 2.0.2
- pandas 2.2.2
- SciPy 1.16.3
- scikit-learn 1.6.1
- PyTorch 2.11.0

Some historical programs were executed in Kaggle or a private synchronized-storage environment and contain calls that request a secret named `RCLONE_CONFIG_B64`. No secret value is included. Do not provide credentials to untrusted copies of these programs.

## Statistical scope

The inferential unit is the participant. The primary focal contrast is unadjusted; secondary families are Holm-controlled. BCa and percentile intervals use 100,000 deterministic bootstrap resamples. P07 remains descriptive. No equivalence claim is made.

