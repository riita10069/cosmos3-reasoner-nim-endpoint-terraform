# Third-party notices

This repository distributes no original video, frame image, annotation
archive, sensor log, trajectory source file, map snapshot, or model weight
from the projects listed below. The JSON paper inputs contain deterministic
transformations, aggregate measurements, or feature summaries derived during
the research.

The repository's licenses do not grant rights in third-party source
materials. Users must comply with each applicable upstream license.

## ROAD: The ROad event Awareness Dataset

- Project: <https://github.com/gurkirt/road-dataset>
- License: [CC BY-NC-SA 4.0][cc-by-nc-sa-4.0]
- Upstream notice: ROAD is built on the Oxford RobotCar Dataset and is
  intended for non-commercial academic use.
- Citation:
  - Gurkirt Singh et al., "ROAD: The ROad event Awareness Dataset for
    Autonomous Driving," IEEE Transactions on Pattern Analysis and Machine
    Intelligence, 2022.

The clip-level benchmark and ground-truth metadata in `paper/inputs/` were
created by deterministic transformation of ROAD annotations. They are
distributed under the terms described in
[`LICENSE-DERIVED-DATA.md`](LICENSE-DERIVED-DATA.md).

## Oxford RobotCar Dataset

- Project: <https://robotcar-dataset.robots.ox.ac.uk/>
- License: [CC BY-NC-SA 4.0][cc-by-nc-sa-4.0]
- Intended use stated by the project: non-commercial academic use.
- Privacy and download terms remain applicable.
- Citation:
  - Will Maddern, Geoffrey Pascoe, Chris Linegar, and Paul Newman,
    "1 Year, 1000 km: The Oxford RobotCar Dataset," The International Journal
    of Robotics Research, 2017.

## OPR-Project OxfordRobotCar_OpenPlaceRecognition

- Dataset:
  <https://huggingface.co/datasets/OPR-Project/OxfordRobotCar_OpenPlaceRecognition>
- License declared by the dataset card: [CC BY-NC-SA 4.0][cc-by-nc-sa-4.0]
- Description: a pre-processed Oxford RobotCar subset for place recognition.

The original OPR images, description CSV files, and source trajectories are
not included. Some locked JSON inputs contain low-frequency numerical
summaries derived from timestamp-aligned public OPR trajectories.

## OpenStreetMap

- Project and attribution information:
  <https://www.openstreetmap.org/copyright>
- Copyright: OpenStreetMap contributors.
- Database license: [Open Data Commons Open Database License 1.0][odbl-1.0].

The raw OpenStreetMap snapshot is not included. The file
`paper/inputs/map-context-lock.json` contains derived feature summaries such
as intersection and signal-infrastructure distances. ODbL obligations apply
to the extent those summaries exercise OpenStreetMap database rights.

## NVIDIA Cosmos and related software

No model weights, container images, or NVIDIA software are redistributed by
this repository. Any use of NVIDIA Cosmos, NIM, or related software is subject
to the terms supplied by NVIDIA and the relevant service provider.

[cc-by-nc-sa-4.0]: https://creativecommons.org/licenses/by-nc-sa/4.0/
[odbl-1.0]: https://opendatacommons.org/licenses/odbl/1-0/
