# Data access

## DELTA dataset

The analyses use the publicly available DELTA dataset:

- **DELTA: Dense Electromyography for Long-Term Adaptive Control**
- Zenodo DOI: [10.5281/zenodo.10801000](https://doi.org/10.5281/zenodo.10801000)

Download the dataset directly from Zenodo. This repository does not redistribute the raw HD-sEMG recordings, labels, participant files, or a modified copy of the source dataset.

## Study cohort used in the analysis

The archived dataset contains seven de-identified participants, six recording sessions per participant, seven movement labels, and 2,940 repetitions. The population-inference cohort comprises P01-P06. P07, the participant with limb absence, is retained as a descriptive case and does not enter population confidence intervals or hypothesis tests.

## Expected local data

The preserved historical programs were written against frozen evidence packets and private execution paths. They are intentionally not configured to download private storage or credentials. A user who wishes to reconstruct the full workflow must:

1. obtain DELTA from the original Zenodo record;
2. recreate the documented repetition/window structure and masks;
3. provide the named intermediate evidence packets expected by the preserved stage programs; and
4. verify every packet against the hashes embedded in those programs.

The repository does not claim that the historical pipeline is self-contained without those evidence packets.

## Privacy

Only de-identified participant codes used by the public DELTA dataset appear in the code. No author credentials, email addresses, raw participant data, or private configuration values are included.

