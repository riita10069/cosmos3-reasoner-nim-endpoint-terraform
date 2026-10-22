from __future__ import annotations

import csv
import json
from pathlib import Path

import httpx
import pytest

from benchmark_tool.vlm import (
    load_ground_truth_labels,
    load_positive_ground_truth,
    reuse_vlm_media_lock,
    timestamp_manifest,
    uniform_timestamps,
    window_bounds,
)
from benchmark_tool.vlm_evaluation import (
    _gpu_telemetry_summary,
    _percentile,
    _profile_metrics,
    _read_run_records,
    aggregate_records,
    evaluate_vlm,
    quality_gate,
)
from benchmark_tool.vlm_runner import (
    _load_video_ids,
    _merge_grouped_repair_result,
    _merge_repair_result,
    _repair_reason,
    _resolve_repair_tag_groups,
    _run_task,
)
from benchmark_tool.client import parse_tags
from benchmark_tool.util import load_json
from benchmark_tool.util import resolve_media_lock_root
from benchmark_tool.util import write_json


def test_window_bounds_cover_timeline_and_pin_final_window() -> None:
    windows = window_bounds(47, 20, 0.5)
    assert windows == [
        (0.0, 20.0),
        (10.0, 30.0),
        (20.0, 40.0),
        (27.0, 47),
    ]
    assert windows[0][0] == 0
    assert windows[-1][1] == 47
    assert all(current[1] >= following[0] for current, following in zip(windows, windows[1:]))


def test_uniform_timestamps_and_manifest_preserve_source_time() -> None:
    timestamps = uniform_timestamps(10, 12, 2)
    assert timestamps == [10, 10.5, 11, 11.5, 12]
    assert timestamp_manifest(timestamps).splitlines() == [
        "F0001=10.000s",
        "F0002=10.500s",
        "F0003=11.000s",
        "F0004=11.500s",
        "F0005=12.000s",
    ]


def test_vlm_sampler_rejects_invalid_worker_count(tmp_path: Path) -> None:
    from benchmark_tool.vlm import build_vlm_media_lock

    try:
        build_vlm_media_lock(
            tmp_path / "dataset.json",
            tmp_path / "profiles.json",
            tmp_path / "output",
            selected_profiles=None,
            workers=0,
        )
    except ValueError as error:
        assert str(error) == "workers must be at least 1"
    else:
        raise AssertionError("Expected invalid worker count to fail")


def test_media_lock_root_resolves_relative_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work_dir = tmp_path / "work"
    lock_path = work_dir / "data" / "media-lock.json"
    lock_path.parent.mkdir(parents=True)
    monkeypatch.chdir(work_dir)

    assert resolve_media_lock_root({}, Path("data/media-lock.json")) == (
        lock_path.parent.resolve()
    )


def test_ground_truth_duplicate_tag_uses_any_yes(tmp_path: Path) -> None:
    path = tmp_path / "gt.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["動画名", "煙", "煙", "坂"])
        writer.writerow(["000001.mp4", "no", "yes", "no"])
    ground_truth, tags = load_positive_ground_truth(path)
    assert tags == ["煙", "坂"]
    assert ground_truth["000001.mp4"] == {"煙": "yes", "坂": "unknown"}
    raw_ground_truth, _ = load_ground_truth_labels(path)
    assert raw_ground_truth["000001.mp4"] == {"煙": "yes", "坂": "no"}


def test_aggregate_records_ors_windows_and_preserves_unknown() -> None:
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": "one.mp4",
            "parsed_tags": {"a": "no", "b": "no"},
        },
        {
            "success": True,
            "profile": "p",
            "video_id": "one.mp4",
            "parsed_tags": {"a": "yes"},
        },
        {
            "success": False,
            "profile": "p",
            "video_id": "two.mp4",
        },
    ]
    result = aggregate_records(records, ["a", "b"])
    assert result["p"]["one.mp4"] == {"a": "yes", "b": "no"}
    assert result["p"]["two.mp4"] == {"a": "unknown", "b": "unknown"}


def test_aggregate_records_does_not_leak_videos_across_profiles() -> None:
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": "one.mp4",
            "parsed_tags": {"a": "yes"},
        },
        {
            "success": True,
            "profile": "q",
            "video_id": "two.mp4",
            "parsed_tags": {"a": "yes"},
        },
    ]

    result = aggregate_records(records, ["a"])

    assert set(result["p"]) == {"one.mp4"}
    assert set(result["q"]) == {"two.mp4"}


def test_profile_metrics_limit_ground_truth_to_run_videos() -> None:
    predictions = {"one.mp4": {"a": "yes"}}
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": "one.mp4",
            "parsed_tags": {"a": "yes"},
            "e2e_seconds": 1,
        }
    ]

    metrics = _profile_metrics(
        "p",
        predictions,
        {"one.mp4": {"a": "yes"}, "not-run.mp4": {"a": "yes"}},
        ["a"],
        records,
    )

    assert metrics["micro_coverage"] == 1
    assert metrics["micro_lower_bound_recall"] == 1
    assert metrics["tags"][0]["gt_positive"] == 1


def test_profile_metrics_count_excluded_positive_as_not_evaluated() -> None:
    predictions = {"one.mp4": {"a": "yes"}}
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": "one.mp4",
            "parsed_tags": {"a": "yes"},
            "e2e_seconds": 1,
            "dataset_excluded_video_ids": ["broken.mp4"],
        }
    ]

    metrics = _profile_metrics(
        "p",
        predictions,
        {"one.mp4": {"a": "yes"}, "broken.mp4": {"a": "yes"}},
        ["a"],
        records,
    )

    assert metrics["micro_coverage"] == 0.5
    assert metrics["micro_lower_bound_recall"] == 0.5
    assert metrics["requested_video_count"] == 2
    assert metrics["dataset_excluded_video_count"] == 1


def test_profile_metrics_use_wall_time_for_concurrent_cost() -> None:
    predictions = {
        "one.mp4": {"a": "yes"},
        "two.mp4": {"a": "yes"},
    }
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": video_id,
            "parsed_tags": labels,
            "recorded_at": "2026-08-04T00:00:00+00:00",
            "e2e_seconds": 10,
            "instance": {"gpu_count": 2, "hourly_usd": 4},
        }
        for video_id, labels in predictions.items()
    ]

    metrics = _profile_metrics(
        "p",
        predictions,
        {video_id: {"a": "yes"} for video_id in predictions},
        ["a"],
        records,
    )

    assert metrics["request_e2e_seconds_per_video"] == 10
    assert metrics["wall_seconds"] == 10
    assert metrics["wall_seconds_per_video"] == 5
    assert metrics["gpu_seconds_per_video"] == 10
    assert metrics["scenes_per_gpu_hour"] == 360
    assert metrics["estimated_usd_per_1000_videos"] == 4 * 5 / 3600 * 1000


def test_profile_metrics_use_run_counters_for_concurrent_stages(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(
        run_dir / "run-metadata.json",
        {
            "experiment": {
                "model": "model",
                "instance": {"gpu_count": 1, "hourly_usd": 1},
            },
            "execution": {
                "profiles": ["p"],
                "concurrency": 8,
                "wall_seconds": 10,
                "metrics_delta": {
                    "vllm:request_prefill_time_seconds_sum": 12,
                    "vllm:request_decode_time_seconds_sum": 4,
                    "vllm:mm_cache_hits_total": 3,
                    "vllm:mm_cache_queries_total": 4,
                    "vllm:prefix_cache_hits_total": 8,
                    "vllm:prefix_cache_queries_total": 10,
                    "vllm:num_preemptions_total": 2,
                },
            },
            "kubernetes": {
                "pod_template_labels": {
                    "benchmark-profile": "nano-chunked-2048"
                }
            },
        },
    )
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": video_id,
            "parsed_tags": {"a": "yes"},
            "e2e_seconds": 8,
        }
        for video_id in ["one.mp4", "two.mp4"]
    ]
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    loaded = _read_run_records(run_dir)

    metrics = _profile_metrics(
        "run::p",
        {
            "one.mp4": {"a": "yes"},
            "two.mp4": {"a": "yes"},
        },
        {
            "one.mp4": {"a": "yes"},
            "two.mp4": {"a": "yes"},
        },
        ["a"],
        loaded,
    )

    assert metrics["prefill_seconds_per_video"] == 6
    assert metrics["decode_seconds_per_video"] == 2
    assert metrics["mm_cache_hit_rate"] == 0.75
    assert metrics["prefix_cache_token_hit_rate"] == 0.8
    assert metrics["preemptions"] == 2
    assert metrics["concurrency"] == 8
    assert metrics["serving_profile"] == "nano-chunked-2048"


def test_profile_metrics_reconstruct_resumed_serial_wall_time() -> None:
    predictions = {
        "one.mp4": {"a": "yes"},
        "two.mp4": {"a": "yes"},
    }
    records = [
        {
            "success": True,
            "benchmark_key": "run::p",
            "profile": "p",
            "run_id": "run",
            "run_wall_seconds": 10,
            "run_profile_count": 1,
            "concurrency": 1,
            "video_id": video_id,
            "parsed_tags": labels,
            "recorded_at": recorded_at,
            "e2e_seconds": 8,
            "instance": {"gpu_count": 1, "hourly_usd": 1},
        }
        for video_id, labels, recorded_at in [
            ("one.mp4", {"a": "yes"}, "2026-08-04T00:00:00+00:00"),
            ("two.mp4", {"a": "yes"}, "2026-08-04T00:00:08+00:00"),
        ]
    ]

    metrics = _profile_metrics(
        "run::p",
        predictions,
        {video_id: {"a": "yes"} for video_id in predictions},
        ["a"],
        records,
    )

    assert metrics["wall_seconds"] == 16
    assert metrics["wall_seconds_reconstructed"] is True


def test_profile_metrics_allocate_multi_profile_run_wall_time() -> None:
    records = [
        {
            "success": True,
            "benchmark_key": profile,
            "profile": profile,
            "run_id": "run",
            "run_wall_seconds": 33,
            "video_id": f"{profile}.mp4",
            "parsed_tags": {"a": "yes"},
            "recorded_at": recorded_at,
            "e2e_seconds": duration,
            "instance": {"gpu_count": 1, "hourly_usd": 1},
        }
        for profile, recorded_at, duration in [
            ("p", "2026-08-04T00:00:00+00:00", 10),
            ("q", "2026-08-04T00:00:10+00:00", 20),
        ]
    ]

    metrics = _profile_metrics(
        "p",
        {"p.mp4": {"a": "yes"}},
        {"p.mp4": {"a": "yes"}},
        ["a"],
        records,
    )

    assert metrics["wall_seconds"] == 11
    assert metrics["estimated_usd_per_1000_videos"] == 11 / 3600 * 1000


def test_profile_metrics_reject_incomplete_output_coverage() -> None:
    predictions = {
        "one.mp4": {"a": "yes"},
        "two.mp4": {"a": "unknown"},
    }
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": "one.mp4",
            "parsed_tags": {"a": "yes"},
            "e2e_seconds": 1,
        },
        {
            "success": True,
            "profile": "p",
            "video_id": "two.mp4",
            "parsed_tags": None,
            "e2e_seconds": 1,
        },
    ]

    metrics = _profile_metrics(
        "p",
        predictions,
        {"one.mp4": {"a": "yes"}, "two.mp4": {"a": "unknown"}},
        ["a"],
        records,
    )

    assert metrics["request_output_coverage"] == 0.5
    assert not metrics["output_validity"]["passed"]


def test_profile_metrics_reject_partial_grouped_output() -> None:
    predictions = {"one.mp4": {"a": "yes", "b": "unknown"}}
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": "one.mp4",
            "evaluated_tags": ["a", "b"],
            "parsed_tags": {"a": "yes"},
            "e2e_seconds": 1,
        }
    ]

    metrics = _profile_metrics(
        "p",
        predictions,
        {"one.mp4": {"a": "yes", "b": "yes"}},
        ["a", "b"],
        records,
    )

    assert metrics["request_output_coverage"] == 0
    assert metrics["micro_coverage"] == 0.5
    assert not metrics["output_validity"]["passed"]


def test_profile_metrics_count_recovered_retry_as_complete_output() -> None:
    predictions = {"one.mp4": {"a": "yes"}}
    records = [
        {
            "success": False,
            "profile": "p",
            "video_id": "one.mp4",
            "error": "transport failed",
        },
        {
            "success": True,
            "profile": "p",
            "video_id": "one.mp4",
            "parsed_tags": {"a": "yes"},
            "e2e_seconds": 1,
        },
    ]

    metrics = _profile_metrics(
        "p",
        predictions,
        {"one.mp4": {"a": "yes"}},
        ["a"],
        records,
    )

    assert metrics["failed_requests"] == 1
    assert metrics["requested_video_count"] == 1
    assert metrics["request_output_coverage"] == 1
    assert metrics["output_validity"]["passed"]


def test_profile_metrics_count_failed_videos_in_requested_scope() -> None:
    predictions = {
        "one.mp4": {"a": "yes"},
        "two.mp4": {"a": "unknown"},
    }
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": "one.mp4",
            "parsed_tags": {"a": "yes"},
            "e2e_seconds": 1,
        },
        {
            "success": False,
            "profile": "p",
            "video_id": "two.mp4",
            "error": "endpoint failed",
        },
    ]

    metrics = _profile_metrics(
        "p",
        predictions,
        {"one.mp4": {"a": "yes"}, "two.mp4": {"a": "yes"}},
        ["a"],
        records,
    )

    assert metrics["video_count"] == 1
    assert metrics["requested_video_count"] == 2
    assert metrics["failed_requests"] == 1
    assert metrics["request_output_coverage"] == 0.5


def test_reference_precision_and_accuracy_use_assumed_negatives_only() -> None:
    predictions = {
        "one.mp4": {"a": "yes", "b": "no"},
        "two.mp4": {"a": "yes", "b": "yes"},
        "three.mp4": {"a": "no", "b": "unknown"},
    }
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": video_id,
            "parsed_tags": labels,
            "e2e_seconds": 1,
        }
        for video_id, labels in predictions.items()
    ]
    positive_ground_truth = {
        "one.mp4": {"a": "yes", "b": "unknown"},
        "two.mp4": {"a": "unknown", "b": "yes"},
        "three.mp4": {"a": "unknown", "b": "unknown"},
    }
    raw_ground_truth = {
        "one.mp4": {"a": "yes", "b": "no"},
        "two.mp4": {"a": "no", "b": "yes"},
        "three.mp4": {"a": "no", "b": "unknown"},
    }

    metrics = _profile_metrics(
        "p",
        predictions,
        positive_ground_truth,
        ["a", "b"],
        records,
        reference_ground_truth=raw_ground_truth,
    )

    assert metrics["micro_lower_bound_recall"] == 1
    assert metrics["reference_precision"] == 2 / 3
    assert metrics["reference_accuracy"] == 4 / 5
    assert metrics["assumed_false_positive"] == 1
    assert metrics["assumed_true_negative"] == 2


def test_reference_metrics_do_not_affect_quality_gate() -> None:
    reference = {
        "micro_lower_bound_recall": 0.8,
        "macro_lower_bound_recall": 0.8,
        "reference_precision": 1.0,
        "reference_accuracy": 1.0,
        "tags": [],
    }
    candidate = {
        **reference,
        "reference_precision": 0.0,
        "reference_accuracy": 0.0,
    }

    assert quality_gate(candidate, reference)["passed"]


def test_profile_metrics_reject_systematic_all_positive_output() -> None:
    predictions = {
        f"{index}.mp4": {"a": "yes", "b": "yes"}
        for index in range(10)
    }
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": video_id,
            "parsed_tags": labels,
            "e2e_seconds": 1,
        }
        for video_id, labels in predictions.items()
    ]

    metrics = _profile_metrics(
        "p",
        predictions,
        {
            video_id: {"a": "yes", "b": "unknown"}
            for video_id in predictions
        },
        ["a", "b"],
        records,
    )

    assert not metrics["output_validity"]["passed"]
    assert metrics["all_positive_video_fraction"] == 1


def test_quality_gate_uses_balanced_thresholds() -> None:
    reference = {
        "micro_lower_bound_recall": 0.9,
        "macro_lower_bound_recall": 0.8,
        "tags": [{"tag": "a", "gt_positive": 20, "lower_bound_recall": 0.8}],
    }
    passing = {
        "micro_lower_bound_recall": 0.88,
        "macro_lower_bound_recall": 0.75,
        "tags": [{"tag": "a", "gt_positive": 20, "lower_bound_recall": 0.7}],
    }
    failing = {
        **passing,
        "tags": [{"tag": "a", "gt_positive": 20, "lower_bound_recall": 0.69}],
    }
    assert quality_gate(passing, reference)["passed"]
    assert not quality_gate(failing, reference)["passed"]


def test_reasoned_sparse_parser_requires_valid_final_indices() -> None:
    tags, error = parse_tags(
        "Reasoning text.\nFINAL_POSITIVE_INDICES: [0, 2]",
        ["a", "b", "c"],
    )
    assert error is None
    assert tags == {"a": "yes", "b": "no", "c": "yes"}

    tags, error = parse_tags(
        "FINAL_POSITIVE_INDICES: [3]", ["a", "b", "c"]
    )
    assert tags is None
    assert error == "Invalid positive tag indices: [3]"


def test_run_reader_recovers_bare_positive_indices(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(
        run_dir / "run-metadata.json",
        {"experiment": {"model": "model", "instance": {"gpu_count": 1}}},
    )
    (run_dir / "results.jsonl").write_text(
        '{"success":true,"profile":"p","video_id":"one.mp4",'
        '"content":"[]","evaluated_tags":["a","b"],'
        '"prompt_variant":"reasoned-sparse",'
        '"parsed_tags":null,"parse_error":"Response has no tags object"}\n',
        encoding="utf-8",
    )

    record = _read_run_records(run_dir)[0]

    assert record["parsed_tags"] == {"a": "no", "b": "no"}
    assert record["parse_recovered"] is True
    assert record["original_parse_error"] == "Response has no tags object"


def test_media_first_prompt_is_part_of_vlm_task_identity() -> None:
    from benchmark_tool.vlm_runner import _task_key

    experiment = {"model": "model"}
    record = {
        "profile": "p",
        "video_id": "one.mp4",
        "clip_id": "w0000",
        "tag_group": "group",
        "prompt_variant": "decision-only-sparse",
        "media_sha256": "abc",
    }

    media_first = _task_key(
        experiment,
        {**record, "media_first_prompt": True},
        1,
        "labeling",
    )
    text_first = _task_key(
        experiment,
        {**record, "media_first_prompt": False},
        1,
        "labeling",
    )

    assert media_first != text_first


def test_reuse_media_lock_requires_identical_sampling(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.json"
    write_json(
        profiles_path,
        {
            "profiles": {
                "source": {
                    "kind": "adaptive",
                    "base_fps": 1,
                    "timestamp_mode": "text-manifest",
                    "tag_groups": ["all"],
                    "prompt_variant": "reasoned-sparse",
                    "max_tokens": 4096,
                },
                "target": {
                    "kind": "adaptive",
                    "base_fps": 1,
                    "timestamp_mode": "implicit-control",
                    "tag_groups": ["a", "b"],
                    "prompt_variant": "decision-only-sparse",
                    "max_tokens": 256,
                },
            }
        },
    )
    source_lock_path = tmp_path / "source-lock.json"
    write_json(
        source_lock_path,
        {
            "dataset_lock": "dataset.json",
            "dataset_lock_sha256": "dataset-sha",
            "records": [
                {
                    "profile": "source",
                    "video_id": "one.mp4",
                    "clip_id": "w0000",
                    "media_sha256": "media-sha",
                    "tag_group": "all",
                }
            ],
        },
    )

    output = reuse_vlm_media_lock(
        source_lock_path,
        profiles_path,
        tmp_path / "target-lock.json",
        source_profile="source",
        target_profile="target",
    )

    lock = load_json(output)
    assert [row["tag_group"] for row in lock["records"]] == ["a", "b"]
    assert all(row["media_sha256"] == "media-sha" for row in lock["records"])
    assert all(
        row["prompt_variant"] == "decision-only-sparse"
        for row in lock["records"]
    )


def test_percentile_uses_linear_interpolation() -> None:
    assert _percentile([], 0.95) is None
    assert _percentile([1, 2, 3, 4], 0.50) == 2.5
    assert round(_percentile([1, 2, 3, 4], 0.95) or 0, 2) == 3.85


def test_gpu_telemetry_summary(tmp_path: Path) -> None:
    path = tmp_path / "gpu-telemetry.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "gpu_utilization_percent": 90,
                        "memory_used_mib": 40_000,
                        "memory_total_mib": 46_000,
                        "power_watts": 300,
                        "temperature_c": 70,
                    }
                ),
                json.dumps(
                    {
                        "gpu_utilization_percent": 100,
                        "memory_used_mib": 45_000,
                        "memory_total_mib": 46_000,
                        "power_watts": 350,
                        "temperature_c": 80,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _gpu_telemetry_summary(path)

    assert summary["gpu_utilization_percent_mean"] == 95
    assert summary["gpu_utilization_percent_p95"] == 99.5
    assert summary["gpu_memory_used_mib_max"] == 45_000
    assert summary["gpu_power_watts_mean"] == 325
    assert summary["gpu_temperature_c_max"] == 80


def test_profile_metrics_accept_null_request_metrics() -> None:
    records = [
        {
            "success": True,
            "profile": "p",
            "video_id": "one.mp4",
            "parsed_tags": {"a": "yes"},
            "evaluated_tags": ["a"],
            "metrics_delta": None,
            "e2e_seconds": 1,
        }
    ]

    metrics = _profile_metrics(
        "p",
        {"one.mp4": {"a": "yes"}},
        {"one.mp4": {"a": "yes"}},
        ["a"],
        records,
    )

    assert metrics["prefill_seconds_per_video"] == 0
    assert metrics["preemptions"] == 0


def test_repair_result_preserves_primary_and_sums_cost() -> None:
    primary = {
        "content": "FINAL_NEGATIVE_INDICES: []",
        "parsed_tags": None,
        "parse_error": "invalid",
        "finish_reason": "stop",
        "e2e_seconds": 1,
        "ttft_seconds": 0.2,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
        },
        "metrics_delta": {"decode": 0.5},
    }
    repair = {
        "content": "FINAL_POSITIVE_INDICES: []",
        "parsed_tags": {"a": "no"},
        "parse_error": None,
        "finish_reason": "stop",
        "e2e_seconds": 2,
        "ttft_seconds": 0.3,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 4,
            "total_tokens": 104,
        },
        "metrics_delta": {"decode": 0.6, "prefill": 0.2},
    }

    result = _merge_repair_result(
        primary, repair, repair_variant="repair"
    )

    assert result["parsed_tags"] == {"a": "no"}
    assert result["attempt_count"] == 2
    assert result["e2e_seconds"] == 3
    assert result["usage"]["total_tokens"] == 209
    assert result["metrics_delta"] == {"decode": 1.1, "prefill": 0.2}
    assert result["primary_parse_error"] == "invalid"
    assert result["repair_reason"] == "parse_failure"


def test_grouped_repair_merges_tags_and_sums_cost() -> None:
    primary = {
        "content": "FINAL_POSITIVE_INDICES: [0, 1]",
        "parsed_tags": {"a": "yes", "b": "yes"},
        "parse_error": None,
        "finish_reason": "stop",
        "e2e_seconds": 1,
        "ttft_seconds": 0.2,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "total_tokens": 105,
        },
        "metrics_delta": {"decode": 0.5},
    }
    repairs = [
        (
            "first",
            {
                "request_sha256": "one",
                "content": "FINAL_POSITIVE_INDICES: []",
                "parsed_tags": {"a": "no"},
                "parse_error": None,
                "finish_reason": "stop",
                "e2e_seconds": 2,
                "ttft_seconds": 0.3,
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 4,
                    "total_tokens": 84,
                },
                "metrics_delta": {"decode": 0.6},
            },
        ),
        (
            "second",
            {
                "request_sha256": "two",
                "content": "FINAL_POSITIVE_INDICES: [0]",
                "parsed_tags": {"b": "yes"},
                "parse_error": None,
                "finish_reason": "stop",
                "e2e_seconds": 3,
                "ttft_seconds": 0.4,
                "usage": {
                    "prompt_tokens": 90,
                    "completion_tokens": 4,
                    "total_tokens": 94,
                },
                "metrics_delta": {"decode": 0.7, "prefill": 0.2},
            },
        ),
    ]

    result = _merge_grouped_repair_result(
        primary,
        repairs,
        expected_tags=["a", "b"],
        repair_variant="repair",
        repair_reason="all_positive",
    )

    assert result["parsed_tags"] == {"a": "no", "b": "yes"}
    assert result["attempt_count"] == 3
    assert result["e2e_seconds"] == 6
    assert result["usage"]["total_tokens"] == 283
    assert result["metrics_delta"] == {"decode": 1.8, "prefill": 0.2}
    assert result["repair_success"] is True
    assert result["repair_group_count"] == 2


def test_grouped_repair_requires_exact_tag_partition() -> None:
    groups = {"first": ["a"], "second": ["b"]}
    assert _resolve_repair_tag_groups(
        groups, ["first", "second"], ["a", "b"]
    ) == [("first", ["a"]), ("second", ["b"])]
    with pytest.raises(ValueError, match="must partition"):
        _resolve_repair_tag_groups(
            groups, ["first"], ["a", "b"]
        )


def test_load_video_ids_ignores_comments_and_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "video-ids.txt"
    path.write_text(
        "# retry set\none.mp4\n\ntwo.mp4\none.mp4\n",
        encoding="utf-8",
    )

    assert _load_video_ids(path) == {"one.mp4", "two.mp4"}


def test_repair_reason_retries_only_enabled_degenerate_output() -> None:
    variant = {
        "repair_prompt_variant": "repair",
        "repair_all_positive": True,
        "all_positive_repair_prompt_variant": "all-positive-repair",
    }

    assert (
        _repair_reason(
            {"parsed_tags": {"a": "yes", "b": "yes"}}, variant
        )
        == "all_positive"
    )
    assert (
        _repair_reason(
            {"parsed_tags": {"a": "yes", "b": "no"}}, variant
        )
        is None
    )
    assert (
        _repair_reason(
            {"parsed_tags": None}, variant
        )
        == "parse_failure"
    )


def test_parse_repair_receives_primary_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_invoke_video(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {
                "success": True,
                "content": "draft explicitly supports index 0",
                "parsed_tags": None,
                "parse_error": "truncated",
                "finish_reason": "length",
                "e2e_seconds": 1,
                "ttft_seconds": 0.5,
                "usage": {},
                "metrics_delta": None,
            }
        return {
            "success": True,
            "content": "FINAL_POSITIVE_INDICES: [0]",
            "parsed_tags": {"tag": "yes"},
            "parse_error": None,
            "finish_reason": "stop",
            "e2e_seconds": 1,
            "ttft_seconds": 0.5,
            "usage": {},
            "metrics_delta": None,
            "request_sha256": "repair",
        }

    monkeypatch.setattr(
        "benchmark_tool.vlm_runner.invoke_video", fake_invoke_video
    )
    record = {
        "video_id": "one.mp4",
        "profile": "draft-aware",
        "evaluation_profile": "draft-aware",
        "media_path": str(tmp_path / "one.mp4"),
        "media_io_kwargs": {"video": {"num_frames": -1}},
        "media_sha256": "abc",
        "clip_id": "one",
        "tag_group": "all",
        "prompt_variant": "reasoned-draft-repair",
        "media_first_prompt": True,
        "timestamp_mode": "text-manifest",
        "timestamp_manifest": "F0001=0.000s",
        "clip_start_seconds": 0,
        "clip_end_seconds": 1,
        "selected_frame_count": 1,
        "selected_timestamps_seconds": [0],
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.8,
    }
    result = _run_task(
        (1, record),
        index=1,
        endpoint="http://unused",
        experiment={
            "model": "model",
            "max_tokens": 256,
            "temperature": 0.2,
            "seed": 1,
        },
        labeling={
            "tag_definitions": {"tag": "criterion"},
            "tag_groups": {},
            "prompt_variants": {
                "reasoned-draft-repair": {
                        "repair_prompt_variant": "draft-repair",
                        "repair_include_primary_draft": True,
                        "repair_text_only": True,
                        "repair_max_tokens": 64,
                }
            },
        },
        video={"video_id": "one.mp4"},
        media_lock_root=tmp_path,
        server_media_root=None,
        capture_metrics=False,
        labeling_digest="labels",
    )

    assert result["parsed_tags"] == {"tag": "yes"}
    assert calls[0].get("draft_context") is None
    assert calls[1]["draft_context"] == "draft explicitly supports index 0"
    assert calls[1]["include_video"] is False


def test_output_validity_failure_blocks_quality_gate() -> None:
    reference = {
        "micro_lower_bound_recall": 0.5,
        "macro_lower_bound_recall": 0.5,
        "tags": [],
    }
    candidate = {
        **reference,
        "output_validity": {
            "passed": False,
            "warnings": ["all-positive output"],
        },
    }
    gate = quality_gate(candidate, reference)
    assert not gate["passed"]
    assert gate["failures"] == ["all-positive output"]


def test_evaluation_can_use_cross_run_reference(tmp_path: Path) -> None:
    ground_truth = tmp_path / "gt.csv"
    ground_truth.write_text(
        "動画名,a\none.mp4,yes\ntwo.mp4,yes\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    partial = tmp_path / "partial"
    for run_dir, profile, predictions in [
        (baseline, "reference", ["yes", "yes"]),
        (candidate, "candidate", ["yes", "no"]),
        (partial, "partial", ["yes"]),
    ]:
        run_dir.mkdir()
        write_json(
            run_dir / "run-metadata.json",
            {
                "experiment": {
                    "model": "model",
                    "instance": {"gpu_count": 1, "hourly_usd": 1},
                }
            },
        )
        lines = [
            {
                "success": True,
                "profile": profile,
                "video_id": video_id,
                "parsed_tags": {"a": prediction},
                "e2e_seconds": 1,
            }
            for video_id, prediction in zip(
                ["one.mp4", "two.mp4"][: len(predictions)],
                predictions,
                strict=True,
            )
        ]
        (run_dir / "results.jsonl").write_text(
            "".join(f"{json.dumps(row)}\n" for row in lines),
            encoding="utf-8",
        )

    summary_path, csv_path, _ = evaluate_vlm(
        [baseline, candidate, partial],
        ground_truth,
        tmp_path / "output",
        reference_profile="reference",
        reference_run_id="baseline",
        bootstrap_samples=100,
        seed=7,
    )

    summary = load_json(summary_path)
    candidate_metrics = next(
        row for row in summary["profiles"] if row["run_id"] == "candidate"
    )
    assert candidate_metrics["comparison_reference"] == "baseline::reference"
    assert candidate_metrics["micro_delta_ci95"] is not None
    assert not candidate_metrics["gate"]["passed"]
    partial_metrics = next(
        row for row in summary["profiles"] if row["run_id"] == "partial"
    )
    assert "comparison_reference" not in partial_metrics
    assert "micro_delta_ci95" not in partial_metrics
    assert csv_path.read_text(encoding="utf-8-sig").splitlines()[0].startswith(
        "benchmark_key,run_id,profile"
    )


