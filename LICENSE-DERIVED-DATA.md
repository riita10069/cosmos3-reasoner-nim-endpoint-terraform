# Derived benchmark and research data license

## Scope

This notice applies to the JSON files under `paper/inputs/`.

Those files contain the locked inputs used to generate the paper, including a
deterministically generated clip-level benchmark manifest, derived ground
truth metadata, aggregate evaluation results, sampling audits, serving
measurements, and derived motion or map-context summaries.

They do **not** contain the original:

- ROAD videos or annotation files;
- Oxford RobotCar images, sensor logs, GPS/INS files, or RTK files;
- OPR images, description CSV files, or source trajectories;
- OpenStreetMap snapshot or raw map database.

## License

The repository author's copyrightable contributions to these files are
licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
license][cc-by-nc-sa-4.0] (CC BY-NC-SA 4.0).

This license choice reflects that the clip-level benchmark and ground-truth
metadata are deterministic transformations of ROAD annotations and may be
considered adapted or derived dataset material.

You may use and adapt the covered files for non-commercial purposes if you:

1. give appropriate attribution to this work and the applicable source
   datasets;
2. provide links to the applicable licenses;
3. identify modifications; and
4. distribute adapted covered material under CC BY-NC-SA 4.0.

## Upstream rights are not relicensed

This notice licenses only rights that the repository author is authorized to
license. It does not replace, weaken, or broaden any upstream license.

ROAD, Oxford RobotCar, and the OPR-derived dataset remain subject to CC
BY-NC-SA 4.0 and their attribution, privacy, academic-use, and other
conditions. No rights in those source datasets are granted by this notice.

`paper/inputs/map-context-lock.json` contains feature summaries derived using
OpenStreetMap data. OpenStreetMap is copyright OpenStreetMap contributors and
is licensed under ODbL 1.0. To the extent ODbL database rights apply, users
must comply with ODbL 1.0 in addition to this notice.

## Recommended attribution

> Ryota Yamada, "公開自動運転動画に対する視覚言語モデルを用いた時系列シーンラベリング評価," 2026.  
> DOI: <https://doi.org/10.5281/zenodo.21863163>  
> Derived benchmark inputs licensed under CC BY-NC-SA 4.0.

The source-dataset citations and license links are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

[cc-by-nc-sa-4.0]: https://creativecommons.org/licenses/by-nc-sa/4.0/
