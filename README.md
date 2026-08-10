# Cosmos 3 ROAD scene-labeling paper

This repository is the publication-safe supplementary package for the final
ROAD scene-labeling paper.

Paper DOI: <https://doi.org/10.5281/zenodo.21863163>

It intentionally contains only:

- the final paper in PDF, TeX, and Markdown formats;
- the 21 sanitized JSON inputs used by the paper generator;
- the SHA-256 artifact manifest;
- the minimal Python generator and focused tests;
- a security verifier for the release tree.

It does not contain videos, extracted frames, customer data, prediction JSONL
files, ground-truth source files, AWS credentials, Terraform state, or
intermediate experiment runs.

## Licenses and dataset rights

This is a mixed-license repository. The license applied to the paper does not
apply automatically to data, source datasets, or software.

| Material | License or status |
|---|---|
| Final paper and authored paper documentation | CC BY 4.0 |
| `paper/inputs/*.json` derived benchmark and research inputs | CC BY-NC-SA 4.0 for the author's contributions, subject to upstream rights |
| OpenStreetMap-derived content in `map-context-lock.json` | ODbL 1.0 also applies where relevant |
| Python code, tests, Makefile, and verification scripts | No software license granted |
| ROAD, Oxford RobotCar, and OPR source materials | Not redistributed; upstream licenses remain controlling |

This paper is licensed under CC BY 4.0. The ROAD dataset, Oxford RobotCar
Dataset, and OPR-derived source materials are not redistributed as part of
this work and remain subject to their respective licenses, including CC
BY-NC-SA 4.0.

No rights in the ROAD, Oxford RobotCar, or OPR source datasets are granted by
the license applied to this paper.

The clip-level benchmark and ground-truth metadata under `paper/inputs/` were
created by deterministic transformation of ROAD annotations. Because those
files may constitute adapted or derived dataset material, they are distributed
under CC BY-NC-SA 4.0 to the extent the repository author has authority to
license them.

See:

- [`LICENSE`](LICENSE): repository-wide license boundaries
- [`LICENSE-PAPER.md`](LICENSE-PAPER.md): CC BY 4.0 paper notice
- [`LICENSE-DERIVED-DATA.md`](LICENSE-DERIVED-DATA.md): derived JSON notice
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): source-dataset licenses
  and attribution

## Verify

```bash
make verify
```

## Regenerate the paper

Tectonic and the fonts declared in the TeX source are required.

```bash
make paper
```

The generated artifact hashes are recorded in
`paper/road-paper-artifact-manifest.json`.
