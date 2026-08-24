# Repository structure and provenance policy

The repository separates public reference code from preserved execution provenance.

## Public reference layer

`src/delta_pcbm/` contains a compact, dependency-light implementation of the acquisition rule described in the study. It is intended for inspection, reuse, and synthetic testing. It is behaviorally aligned with the selector implementation in the locked deep-analysis drivers, but it is not presented as a byte-identical historical source file.

## Preserved execution layer

`provenance/executed_sources/` contains the available analysis source files exactly as assembled for the final reviewer code archive. Historical stage and revision names are retained because renaming or silently rewriting them would break provenance and embedded hash/path contracts.

The original archive checksums are retained in `provenance/ORIGINAL_REVIEW_ARCHIVE_SHA256SUMS.txt`. Paths in that checksum file are relative to the original reviewer archive layout; the release-level `SHA256SUMS.txt` is authoritative for the public repository tree.

## Exclusions

The public repository excludes:

- raw DELTA recordings and labels;
- private intermediate evidence packets;
- cloud-storage credentials and configuration values;
- manuscript and response-letter files;
- author affiliations and private contact information; and
- publisher-formatted content.

