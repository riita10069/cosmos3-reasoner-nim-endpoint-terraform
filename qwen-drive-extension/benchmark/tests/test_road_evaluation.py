from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_tool.road_evaluation import (
    _js_divergence,
    _profile_metrics,
    _rare_scene_retrieval,
    aggregate_road_records,
    evaluate_road,
    load_road_ground_truth,
)
from benchmark_tool.util import write_json


TAGS = ["event"]


def _ground_truth() -> dict[str, dict[str, object]]:
    return {
        "c1": {
            "clip_id": "c1",
            "source_video_id": "v1",
            "split": "evaluation",
            "labels": {"event": "positive"},
            "events": {
                "event": [{"start_seconds": 1.0, "end_seconds": 2.0}]
            },
        },
        "c2": {
            "clip_id": "c2",
            "source_video_id": "v1",
            "split": "evaluation",
            "labels": {"event": "negative"},
            "events": {"event": []},
        },
        "c3": {
            "clip_id": "c3",
            "source_video_id": "v2",
            "split": "evaluation",
            "labels": {"event": "positive"},
            "events": {
                "event": [{"start_seconds": 5.0, "end_seconds": 6.0}]
            },
        },
        "c4": {
            "clip_id": "c4",
            "source_video_id": "v2",
            "split": "evaluation",
            "labels": {"event": "negative"},
            "events": {"event": []},
        },
    }


def _records() -> list[dict[str, object]]:
    predictions = {
        "c1": {"event": "yes"},
        "c2": {"event": "yes"},
        "c3": None,
        "c4": {"event": "no"},
    }
    return [
        {
            "success": prediction is not None,
            "profile": "p",
            "benchmark_key": "run::p",
            "run_id": "run",
            "model": "model",
            "video_id": clip_id,
            "evaluated_tags": TAGS,
            "parsed_tags": prediction,
            "selected_timestamps_seconds": [1.5],
            "e2e_seconds": 1.0,
            "instance": {"gpu_count": 1, "hourly_usd": 3.6},
        }
        for clip_id, prediction in predictions.items()
    ]


def test_complete_gt_metrics_penalize_abstention_without_inventing_fp() -> None:
    records = _records()
    predictions = aggregate_road_records(records, TAGS)

    metrics = _profile_metrics(
        "run::p",
        predictions["run::p"],
        _ground_truth(),
        TAGS,
        records,
        bootstrap_samples=100,
        seed=7,
    )

    tag = metrics["tags"][0]
    assert tag["true_positive"] == 1
    assert tag["false_positive"] == 1
    assert tag["true_negative"] == 1
    assert tag["false_negative"] == 0
    assert tag["abstained_positive"] == 1
    assert tag["abstained_negative"] == 0
    assert tag["precision"] == 0.5
    assert tag["strict_recall"] == 0.5
    assert tag["strict_f1"] == 0.5
    assert tag["overall_accuracy"] == 0.5
    assert metrics["output_pair_coverage"] == 0.75
    assert metrics["evidence"]["evidence_hit_rate"] == 0.5
    assert not metrics["output_validity"]["passed"]


def test_runtime_normalizes_wall_time_by_clip_repetition() -> None:
    records = [
        {
            "success": True,
            "profile": "p",
            "benchmark_key": "run::p",
            "run_id": "run",
            "model": "model",
            "video_id": "c1",
            "repetition": repetition,
            "evaluated_tags": TAGS,
            "parsed_tags": {"event": "yes"},
            "selected_timestamps_seconds": [1.5],
            "e2e_seconds": 1.0,
            "instance": {"gpu_count": 1, "hourly_usd": 3.6},
            "run_wall_seconds": 4.0,
            "run_profile_count": 1,
            "run_metrics_delta": {},
        }
        for repetition in (1, 2)
    ]
    predictions = aggregate_road_records(records, TAGS)

    metrics = _profile_metrics(
        "run::p",
        predictions["run::p"],
        {"c1": _ground_truth()["c1"]},
        TAGS,
        records,
        bootstrap_samples=0,
        seed=7,
    )

    assert metrics["runtime"]["evaluated_clip_instances"] == 2
    assert metrics["runtime"]["wall_seconds_per_clip"] == 2.0


def test_majority_vote_tie_is_abstention() -> None:
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": "c1",
            "parsed_tags": {"event": "yes"},
        },
        {
            "success": True,
            "profile": "p",
            "video_id": "c1",
            "parsed_tags": {"event": "no"},
        },
    ]

    assert aggregate_road_records(records, TAGS)["p"]["c1"] == {
        "event": "unknown"
    }


def test_load_ground_truth_filters_split_and_validates_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gt.jsonl"
    rows = list(_ground_truth().values())
    rows[0] = {**rows[0], "split": "development"}
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    records, tags = load_road_ground_truth(path, split="development")

    assert list(records) == ["c1"]
    assert tags == TAGS


def test_evaluate_road_writes_complete_gt_outputs(
    tmp_path: Path,
) -> None:
    gt_path = tmp_path / "gt.jsonl"
    gt_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in _ground_truth().values()
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(
        run_dir / "run-metadata.json",
        {
            "experiment": {
                "model": "model",
                "instance": {"gpu_count": 1, "hourly_usd": 3.6},
            },
            "kubernetes": {
                "pod_template_labels": {
                    "benchmark-profile": "nano-chunked-2048"
                }
            },
            "execution": {
                "profiles": ["p"],
                "wall_seconds": 4,
                "concurrency": 1,
            },
        },
    )
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in _records():
            record = {
                key: value
                for key, value in record.items()
                if key not in {"benchmark_key", "run_id", "instance"}
            }
            handle.write(json.dumps(record) + "\n")

    summary_path, csv_path, report_path = evaluate_road(
        [run_dir],
        gt_path,
        tmp_path / "output",
        split="evaluation",
        bootstrap_samples=10,
        seed=7,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["clip_count"] == 4
    assert summary["profiles"][0]["micro"]["strict_f1"] == 0.5
    assert (
        summary["profiles"][0]["serving_profile"]
        == "nano-chunked-2048"
    )
    assert csv_path.is_file()
    assert "ROAD complete-GT benchmark" in report_path.read_text(
        encoding="utf-8"
    )


def test_js_divergence_is_zero_for_equal_distributions() -> None:
    assert _js_divergence([1, 3], [2, 6]) == pytest.approx(0)


def test_rare_scene_retrieval_reports_review_enrichment() -> None:
    ground_truth = {
        f"c{index}": {
            "source_video_id": "v",
            "labels": {
                "rare": "positive" if index == 0 else "negative",
                "common": "positive" if index < 8 else "negative",
            },
        }
        for index in range(10)
    }
    predictions = {
        clip_id: {
            "rare": "yes" if clip_id == "c0" else "no",
            "common": "yes",
        }
        for clip_id in ground_truth
    }

    result = _rare_scene_retrieval(
        list(ground_truth),
        predictions,
        ground_truth,
        ["rare", "common"],
        seed=7,
        tie_samples=20,
    )

    assert result is not None
    assert result["rare_tags"] == ["rare"]
    assert result["budgets"][0]["precision_mean"] == 1
    assert result["budgets"][0]["enrichment_mean"] == 10
