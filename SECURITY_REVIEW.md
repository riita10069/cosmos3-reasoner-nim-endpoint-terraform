# Publication security review

## License scope

The release uses an explicit mixed-license structure:

- paper and authored documentation: CC BY 4.0;
- derived benchmark and research JSON inputs: CC BY-NC-SA 4.0 for the
  author's contributions, subject to upstream rights;
- OpenStreetMap-derived database content: ODbL 1.0 where applicable;
- source datasets and model artifacts: not redistributed;
- software: no software license is granted by the Creative Commons notices.

The release verifier requires all license and third-party notice files to be
present.

Review date: 2026-08-10

## Scope

This fresh-history export was created from an allowlist. It does not inherit
the source working repository's Git history.

## Checks

- exactly 21 paper input JSON files are present;
- all input JSON files parse successfully;
- every input and generated artifact matches the SHA-256 manifest;
- no video, frame image, JSONL, CSV, Parquet, NumPy array, credential, private
  key, Terraform state, Terraform plan, or local environment file is present;
- no local absolute home-directory path remains;
- no symlink resolves outside the repository;
- extracted PDF text is included in the text-content scan;
- the paper generator's focused tests pass.

Run `make verify` to repeat the machine checks.
