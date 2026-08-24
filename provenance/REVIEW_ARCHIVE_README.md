# Anonymized reviewer code and environment archive

This archive accompanies the anonymized manuscript and statistical supplement. It contains the available executed Python analysis sources for the locked protocol, classical comparators, candidate-pool construction, temporal-split analyses, deep-model seed analyses, statistical aggregation, and manuscript integration.

## Scientific correction documented by this archive

The audited TCN layer contract uses ordinary PyTorch `Conv1d` layers with `kernel_size=3` and `padding=dilation` for dilation values 1, 2, 4, and 8. PyTorch applies that integer padding symmetrically to the left and right. The available contract contains no left-only `F.pad`, `Chomp1d`, cropping, or output-shift operation. The manuscript therefore describes the network as a **dilated, noncausal TCN within each complete 37-window repetition**.

This is a reporting correction. It changes neither model weights nor predictions, requires no retraining, and does not affect the primary Ridge analysis. It also does not introduce leakage across repetitions, fixed tests, or future sessions.

See:

- `audit/tcn_architecture_padding_audit.csv`
- `audit/tcn_causality_audit.json`
- `reference/verify_tcn_padding_contract.py`

## Source inventory and provenance

The `source/` directory contains the analysis programs available in the reviewer-package assembly workspace. `SOURCE_INDEX.csv` describes their roles. Each file is listed in `SHA256SUMS.txt`.

The byte-identical standalone module named `stage5b_mask_aware_rms_tcn.py` was not present in this local assembly workspace. The deep-analysis drivers restore that module from a hash-verified frozen parent packet and validate its expected 118,536-parameter contract. To avoid misrepresenting a reconstruction as the executed source, this archive does not fabricate that missing file. The supplied audit and verification script reproduce the padding semantics and parameter count from the locked layer contract.

## Reproduction levels

1. **Integrity check (no data):** run `sha256sum -c SHA256SUMS.txt` from the archive root.
2. **Padding-contract check (synthetic data only):** create the environment and run `python reference/verify_tcn_padding_contract.py`. It verifies the 118,536-parameter layer contract and demonstrates that a future within-repetition input can affect an earlier output under symmetric padding.
3. **Analysis replay:** the stage programs require their named frozen parent packets. They verify packet identities before use. Raw DELTA data are not included; the public dataset is available from Zenodo under DOI `10.5281/zenodo.10801000`.

Remote-storage credentials, Kaggle secrets, raw participant data, author names, affiliations, email addresses, and private configuration files are excluded. Several stage programs contain calls that *request* a secret named `RCLONE_CONFIG_B64`; no secret value is embedded in this archive.

## Environment

Use either:

```bash
conda env create -f environment.yml
conda activate delta-pcbm-review
```

or:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The scientific executions used CPU for the final audit/statistical stages. Deep training scripts can use CPU or CUDA as documented internally; the manuscript correction itself requires no retraining.

