# ROAD Cosmos 3 paper bundle

This directory contains the final paper and the locked, sanitized JSON inputs
used to generate it.

## License boundary

The authored paper, TeX, Markdown, adversarial review, and reproduction
documentation are licensed under CC BY 4.0. See
[`../LICENSE-PAPER.md`](../LICENSE-PAPER.md).

The files under `inputs/` include a deterministic clip-level benchmark and
ground-truth metadata derived from ROAD annotations. The repository author's
contributions to these JSON files are licensed under CC BY-NC-SA 4.0, subject
to all applicable upstream rights. See
[`../LICENSE-DERIVED-DATA.md`](../LICENSE-DERIVED-DATA.md).

No original ROAD, Oxford RobotCar, or OPR source files are included. No rights
in those source datasets are granted by the paper's CC BY 4.0 license.
OpenStreetMap-derived database content remains subject to ODbL 1.0 where
applicable. See
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Contents

- `ROAD_COSMOS3_SCENE_LABELING_PAPER.pdf`: final compiled paper
- `ROAD_COSMOS3_SCENE_LABELING_PAPER.tex`: generated TeX source
- `ROAD_COSMOS3_SCENE_LABELING_PAPER.md`: generated Markdown version
- `ADVERSARIAL_SELF_REVIEW.md`: adversarial review record
- `REPRODUCTION.md`: regeneration command
- `inputs/*.json`: 21 locked ROAD input artifacts
- `road-paper-artifact-manifest.json`: SHA-256 digests for all inputs and
  generated artifacts

The bundle contains no videos, frame images, prediction JSONL files, customer
ground truth, credentials, AWS state, or private S3 references. Absolute local
paths in the original public ROAD experiment artifacts were replaced with:

- `${REPO_ROOT}`
- `${ROAD_DATA_ROOT}`
- `${ROAD_ARTIFACT_ROOT}`
- `${USER_HOME}`

These placeholders are provenance metadata only. Paper generation reads the
metrics and locked summaries directly from `inputs/` and does not dereference
the paths.
