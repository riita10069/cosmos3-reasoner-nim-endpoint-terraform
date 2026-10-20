from __future__ import annotations

import json
from pathlib import Path

from benchmark_tool.road_diagnostics import (
    analyze_road_errors,
    audit_road_sampling,
    evaluate_road_cascades,
    summarize_road_serving_repeats,
    temporal_coverage,
)
from benchmark_tool.util import write_json


def test_temporal_coverage_reports_hit_span_and_context() -> None:
    coverage = temporal_coverage(
        [0.0, 1.0, 1.75, 3.0],
        [{"start_seconds": 0.75, "end_seconds": 2.0}],
        clip_start=0,
        clip_end=4,
    )

    assert coverage["event_sample_count"] == 2
    assert coverage["event_sample_span_seconds"] == 0.75
    assert coverage["event_span_at_least_0_5s"]
    assert coverage["has_both_boundary_contexts"]


def test_error_analysis_separates_sampling_sparse_and_covered_misses(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.jsonl"
    rows = [
        _gt("miss", "ego_lane_change", 1.0, 1.4),
        _gt("sparse", "ego_lane_change", 1.0, 1.4),
        _gt("covered", "ego_lane_change", 1.0, 2.0),
    ]
    gt.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    run = tmp_path / "run"
    run.mkdir()
    write_json(
        run / "run-metadata.json",
        {
            "experiment": {"model": "Nano", "instance": {}},
            "execution": {"profiles": ["p"]},
        },
    )
    timestamps = {
        "miss": [0.0, 2.0],
        "sparse": [1.1],
        "covered": [1.0, 1.75],
    }
    with (run / "results.jsonl").open("w", encoding="utf-8") as handle:
        for clip_id, selected in timestamps.items():
            handle.write(
                json.dumps(
                    {
                        "profile": "p",
                        "video_id": clip_id,
                        "success": True,
                        "parsed_tags": {"ego_lane_change": "no"},
                        "evaluated_tags": ["ego_lane_change"],
                        "selected_timestamps_seconds": selected,
                    }
                )
                + "\n"
            )

    summary_path, detail_path, _ = analyze_road_errors(
        [run], gt, tmp_path / "out", split="evaluation"
    )

    profile = json.loads(summary_path.read_text())["profiles"][0]
    assert profile["temporal_sampling_miss"] == 1
    assert profile["temporally_sparse_miss"] == 1
    assert profile["temporally_covered_miss"] == 1
    assert detail_path.is_file()


def test_sampling_audit_compares_media_profiles(tmp_path: Path) -> None:
    gt = tmp_path / "gt.jsonl"
    gt.write_text(
        json.dumps(_gt("c1", "event", 1.0, 2.0)) + "\n",
        encoding="utf-8",
    )
    media_lock = tmp_path / "media.json"
    write_json(
        media_lock,
        {
            "records": [
                {
                    "video_id": "c1",
                    "profile": "miss",
                    "selected_frame_count": 2,
                    "selected_timestamps_seconds": [0.0, 3.0],
                },
                {
                    "video_id": "c1",
                    "profile": "hit",
                    "selected_frame_count": 3,
                    "selected_timestamps_seconds": [0.0, 1.0, 1.75],
                },
            ]
        },
    )

    summary_path, _ = audit_road_sampling(
        [media_lock], gt, tmp_path / "audit", split="evaluation"
    )

    profiles = {
        row["profile"]: row
        for row in json.loads(summary_path.read_text())["profiles"]
    }
    assert profiles["miss"]["any_event_hit_rate"] == 0
    assert profiles["hit"]["event_span_at_least_0_5s_rate"] == 1


def test_serving_repeat_summary_reports_mean_and_sample_sd(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evaluation.json"
    write_json(
        source,
        {
            "profiles": [
                {
                    "model": "Nano",
                    "serving_profile": "nano-server-default",
                    "concurrency": 1,
                    "profile": "p",
                    "run_id": f"r{index}",
                    "runtime": {
                        "wall_seconds_per_clip": wall,
                        "request_e2e_seconds_p95": wall,
                        "prefill_seconds_per_clip": 1,
                        "decode_seconds_per_clip": 2,
                        "estimated_usd_per_1000_clips": wall,
                        "clips_per_gpu_hour": 3600 / wall,
                        "preemptions": 0,
                        "failed_requests": 0,
                    },
                }
                for index, wall in enumerate((1.0, 2.0, 3.0), 1)
            ]
        },
    )

    summary_path, report_path = summarize_road_serving_repeats(
        source, tmp_path / "repeat"
    )

    group = json.loads(summary_path.read_text())["groups"][0]
    assert group["replicate_count"] == 3
    assert group["metrics"]["wall_seconds_per_clip"]["mean"] == 2
    assert (
        group["metrics"]["wall_seconds_per_clip"][
            "sample_standard_deviation"
        ]
        == 1
    )
    assert report_path.is_file()


def test_cascade_routes_from_nano_predictions_without_gt(
    tmp_path: Path,
) -> None:
    gt = tmp_path / "gt.jsonl"
    rows = [
        {
            **_gt("c1", "pedestrian_crossing", 1.0, 2.0),
            "labels": {
                "pedestrian_crossing": "positive",
                "ego_moving": "negative",
            },
            "events": {
                "pedestrian_crossing": [
                    {"start_seconds": 1.0, "end_seconds": 2.0}
                ],
                "ego_moving": [],
            },
        }
    ]
    gt.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    nano = _run(
        tmp_path / "nano",
        "Nano",
        {"pedestrian_crossing": "yes", "ego_moving": "yes"},
    )
    super_run = _run(
        tmp_path / "super",
        "Super",
        {"pedestrian_crossing": "yes", "ego_moving": "no"},
    )

    summary_path, _ = evaluate_road_cascades(
        nano,
        super_run,
        gt,
        tmp_path / "cascade",
        split="evaluation",
        nano_usd_per_1000=1,
        super_usd_per_1000=10,
    )

    policies = {
        row["policy"]: row
        for row in json.loads(summary_path.read_text())["policies"]
    }
    assert policies["nano_pedestrian_positive"]["routed_clip_count"] == 1
    assert policies["nano_pedestrian_positive"]["metrics"]["strict_f1"] == 1
    assert (
        policies["nano_pedestrian_positive"][
            "estimated_usd_per_1000_clips"
        ]
        == 11
    )


def _gt(
    clip_id: str, tag: str, event_start: float, event_end: float
) -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "source_video_id": "v",
        "split": "evaluation",
        "start_seconds": 0.0,
        "end_seconds": 4.0,
        "labels": {tag: "positive"},
        "events": {
            tag: [
                {
                    "start_seconds": event_start,
                    "end_seconds": event_end,
                }
            ]
        },
    }


def _run(
    path: Path, model: str, predictions: dict[str, str]
) -> Path:
    path.mkdir()
    write_json(
        path / "run-metadata.json",
        {
            "experiment": {"model": model, "instance": {}},
            "execution": {"profiles": ["p"]},
        },
    )
    (path / "results.jsonl").write_text(
        json.dumps(
            {
                "profile": "p",
                "video_id": "c1",
                "success": True,
                "parsed_tags": predictions,
                "evaluated_tags": list(predictions),
                "selected_timestamps_seconds": [1.5],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path
