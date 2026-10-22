from __future__ import annotations

import pytest
import numpy as np

from benchmark_tool.road_media import (
    _cap_timestamps,
    _clip_motion_peaks,
    _gt_oracle_timestamps,
    _load_video_ids,
    _profile_tag_groups,
    _transform_frame,
    road_profile_timestamps,
)


def test_uniform_road_timestamps_do_not_cross_clip_boundary() -> None:
    timestamps, peaks = road_profile_timestamps(
        {
            "start_seconds": 8,
            "end_seconds": 16,
            "fps": 12,
        },
        {
            "kind": "uniform",
            "fps": 2,
            "max_frames": 16,
        },
    )

    assert len(timestamps) == 16
    assert timestamps[0] == 8
    assert timestamps[-1] == 15.5
    assert peaks == []


def test_motion_peaks_are_selected_within_clip_and_nms() -> None:
    observations = [
        {"timestamp": 8.0, "score": 0.5},
        {"timestamp": 9.0, "score": 1.0},
        {"timestamp": 10.0, "score": 0.9},
        {"timestamp": 13.0, "score": 0.8},
        {"timestamp": 16.0, "score": 2.0},
    ]

    peaks = _clip_motion_peaks(
        observations,
        start=8,
        end=16,
        nms_seconds=2,
        max_events=2,
    )

    assert peaks == [9.0, 13.0]


def test_timestamp_cap_preserves_endpoints() -> None:
    assert _cap_timestamps([0, 1, 2, 3, 4], 3) == [0, 2, 4]


def test_profile_tag_groups_support_grouped_requests() -> None:
    assert _profile_tag_groups({}) == ["all"]
    assert _profile_tag_groups({"tag_group": "ego"}) == ["ego"]
    assert _profile_tag_groups({"tag_groups": ["ego", "objects"]}) == [
        "ego",
        "objects",
    ]


def test_load_video_ids_ignores_comments_and_blanks(tmp_path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("# smoke\n\nclip-a\nclip-b\n", encoding="utf-8")

    assert _load_video_ids(path) == {"clip-a", "clip-b"}


def test_hybrid_sampling_preserves_global_grid_and_adds_burst() -> None:
    timestamps, peaks = road_profile_timestamps(
        {
            "start_seconds": 0,
            "end_seconds": 8,
            "fps": 12,
        },
        {
            "kind": "adaptive",
            "analysis_fps": 8,
            "nms_seconds": 2,
            "max_events": 1,
            "base_fps": 2,
            "shoulder_seconds": 1,
            "shoulder_fps": 2,
            "event_seconds": 0.5,
            "event_fps": 8,
            "max_frames": 24,
        },
        motion_observations=[
            {"timestamp": 4.0, "score": 10.0},
            {"timestamp": 6.0, "score": 1.0},
        ],
    )

    assert peaks == [4.0]
    assert len(timestamps) <= 24
    assert 4.125 in timestamps
    assert 0.0 in timestamps
    assert 7.916667 in timestamps


def test_gt_oracle_prioritizes_event_boundaries_and_center() -> None:
    timestamps = _gt_oracle_timestamps(
        {
            "start_seconds": 0,
            "end_seconds": 8,
            "fps": 12,
            "labels": {"event": "positive", "absent": "negative"},
            "events": {
                "event": [
                    {"start_seconds": 2.0, "end_seconds": 3.0}
                ],
                "absent": [],
            },
        },
        {"max_frames": 8, "base_fps": 1},
    )

    assert 2.0 in timestamps
    assert 2.5 in timestamps
    assert 3.0 in timestamps
    assert 0.0 in timestamps
    assert 7.916667 in timestamps


def test_gt_oracle_is_rejected_for_evaluation_split() -> None:
    with pytest.raises(ValueError, match="development"):
        road_profile_timestamps(
            {
                "start_seconds": 0,
                "end_seconds": 8,
                "fps": 12,
                "split": "evaluation",
                "labels": {"event": "positive"},
                "events": {
                    "event": [
                        {"start_seconds": 2.0, "end_seconds": 3.0}
                    ]
                },
            },
            {"kind": "gt_oracle", "max_frames": 8},
        )


def test_multiscale_transform_keeps_shape_and_magnifies_views() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :100] = (10, 20, 30)
    frame[:, 100:] = (100, 110, 120)

    transformed = _transform_frame(frame, "road-multiscale-mosaic")

    assert transformed.shape == frame.shape
    assert transformed[:50, :100].mean() > 0
    assert transformed[50:, :100].mean() > 0
    assert transformed[50:, 100:].mean() > transformed[50:, :100].mean()
