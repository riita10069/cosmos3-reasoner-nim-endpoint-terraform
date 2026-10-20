from __future__ import annotations

import json
from pathlib import Path

from benchmark_tool.road import (
    _benchmark_split,
    _contiguous_ranges,
    build_road_ground_truth,
    frame_label_sets,
    select_road_smoke,
)
from benchmark_tool.util import load_json, write_json


def test_frame_label_sets_collects_av_and_actor_labels() -> None:
    frame = {
        "av_action_ids": [1, 4],
        "annos": {
            "b1": {
                "agent_ids": [0],
                "action_ids": [3, 4],
                "loc_ids": [2],
                "duplex_ids": [7],
                "triplet_ids": [11],
            },
            "b2": {
                "agent_ids": [1],
                "action_ids": [3],
                "loc_ids": [2],
                "duplex_ids": [8],
                "triplet_ids": [12],
            },
        },
    }

    labels = frame_label_sets(frame)

    assert labels["av_action"] == {1, 4}
    assert labels["agent"] == {0, 1}
    assert labels["action"] == {3, 4}
    assert labels["location"] == {2}
    assert labels["duplex"] == {7, 8}
    assert labels["triplet"] == {11, 12}


def test_contiguous_ranges_and_official_fold_mapping() -> None:
    assert _contiguous_ranges([1, 2, 4, 5, 6, 9]) == [
        (1, 2),
        (4, 6),
        (9, 9),
    ]
    assert _benchmark_split(["all", "val_1"], 1) == "development"
    assert _benchmark_split(["all", "train_1"], 1) == "evaluation"
    assert _benchmark_split(["all", "train_2"], 1) == "unassigned"


def test_build_road_ground_truth_uses_three_value_contract(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "road.json"
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    config = tmp_path / "config.json"
    output = tmp_path / "output"
    frames = {}
    for frame_id in range(1, 9):
        av_actions = [0] if frame_id <= 3 else [1]
        duplex = [0] if frame_id in {5, 6} else []
        frames[str(frame_id)] = {
            "annotated": 1,
            "av_action_ids": av_actions,
            "annos": {
                "b1": {
                    "agent_ids": [0],
                    "action_ids": [],
                    "loc_ids": [],
                    "duplex_ids": duplex,
                    "triplet_ids": [],
                }
            },
        }
    write_json(
        annotations,
        {
            "all_av_action_labels": ["AV-Stop", "AV-Mov"],
            "all_agent_labels": ["Ped"],
            "all_action_labels": [],
            "all_loc_labels": [],
            "all_duplex_labels": ["Ped-Xing"],
            "all_triplet_labels": [],
            "db": {
                "video": {
                    "numf": 8,
                    "split_ids": ["all", "val_1"],
                    "frames": frames,
                    "av_action_tubes": {},
                    "agent_tubes": {},
                    "action_tubes": {},
                    "loc_tubes": {},
                    "duplex_tubes": {},
                    "triplet_tubes": {},
                }
            },
        },
    )
    write_json(
        config,
        {
            "fps": 2,
            "window_seconds": 2,
            "stride_seconds": 2,
            "minimum_positive_frames": 3,
            "fold": 1,
            "require_fully_annotated": True,
            "tags": [
                {
                    "id": "ego_stopped",
                    "label_type": "av_action",
                    "labels": ["AV-Stop"],
                },
                {
                    "id": "pedestrian_crossing",
                    "label_type": "duplex",
                    "labels": ["Ped-Xing"],
                },
            ],
        },
    )

    manifest_path, ground_truth_path = build_road_ground_truth(
        annotations,
        videos_dir,
        config,
        output,
    )

    manifest = load_json(manifest_path)
    records = [
        json.loads(line)
        for line in ground_truth_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["clip_count"] == 2
    assert manifest["split_clip_counts"] == {"development": 2}
    assert records[0]["labels"] == {
        "ego_stopped": "positive",
        "pedestrian_crossing": "negative",
    }
    assert records[1]["labels"] == {
        "ego_stopped": "negative",
        "pedestrian_crossing": "unknown",
    }
    assert records[1]["events"]["pedestrian_crossing"] == [
        {
            "start_frame": 5,
            "end_frame": 6,
            "start_seconds": 2.0,
            "end_seconds": 3.0,
        }
    ]


def test_select_road_smoke_meets_positive_and_negative_quotas(
    tmp_path: Path,
) -> None:
    ground_truth_path = tmp_path / "gt.jsonl"
    rows = [
        {
            "clip_id": f"c{index}",
            "source_video_id": f"v{index % 2}",
            "split": "development",
            "labels": {
                "a": "positive" if index in {0, 1} else "negative",
                "b": "positive" if index in {1, 2} else "negative",
            },
            "events": {
                "a": (
                    [{"start_seconds": 0, "end_seconds": 1 + index}]
                    if index in {0, 1}
                    else []
                ),
                "b": (
                    [{"start_seconds": 2, "end_seconds": 3}]
                    if index in {1, 2}
                    else []
                ),
            },
        }
        for index in range(4)
    ]
    ground_truth_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest_path, ids_path = select_road_smoke(
        ground_truth_path,
        tmp_path / "out",
        split="development",
        positives_per_tag=1,
        negatives_per_tag=1,
        max_clips=3,
        seed=7,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_clip_count"] <= 3
    assert all(
        values["positive"] >= 1 and values["negative"] >= 1
        for values in manifest["label_counts"].values()
    )
    assert len(ids_path.read_text(encoding="utf-8").splitlines()) == (
        manifest["selected_clip_count"]
    )
