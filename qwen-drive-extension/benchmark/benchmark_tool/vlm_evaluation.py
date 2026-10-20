from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .client import parse_tags
from .vlm import load_ground_truth_labels, load_positive_ground_truth
from .util import load_json, sha256_file, utc_now, write_json


def evaluate_vlm(
    run_dirs: list[Path],
    ground_truth_path: Path,
    output_dir: Path,
    *,
    reference_profile: str,
    bootstrap_samples: int,
    seed: int,
    reference_run_id: str | None = None,
) -> tuple[Path, Path, Path]:
    ground_truth, tags = load_positive_ground_truth(ground_truth_path)
    reference_ground_truth, reference_tags = load_ground_truth_labels(
        ground_truth_path
    )
    if reference_tags != tags:
        raise ValueError("Ground-truth tag order changed while loading labels")
    records = [
        record for run_dir in run_dirs for record in _read_run_records(run_dir)
    ]
    predictions = aggregate_records(records, tags)
    profile_metrics = [
        _profile_metrics(
            benchmark_key,
            values,
            ground_truth,
            tags,
            records,
            reference_ground_truth=reference_ground_truth,
        )
        for benchmark_key, values in sorted(predictions.items())
    ]
    references_by_run = {
        row["run_id"]: row
        for row in profile_metrics
        if row["profile"] == reference_profile
    }
    global_reference = None
    if reference_run_id:
        global_reference = references_by_run.get(reference_run_id)
        if global_reference is None:
            available = ", ".join(sorted(references_by_run)) or "none"
            raise ValueError(
                "Reference profile "
                f"{reference_profile!r} was not found in run "
                f"{reference_run_id!r}; available reference runs: {available}"
            )
    for row in profile_metrics:
        reference = references_by_run.get(row["run_id"])
        if (
            global_reference
            and set(predictions[row["benchmark_key"]])
            == set(predictions[global_reference["benchmark_key"]])
        ):
            reference = global_reference
        if reference:
            row["comparison_reference"] = reference["benchmark_key"]
            row["gate"] = quality_gate(row, reference)
            row["micro_delta_ci95"] = _bootstrap_delta(
                predictions[row["benchmark_key"]],
                predictions[reference["benchmark_key"]],
                ground_truth,
                tags,
                samples=bootstrap_samples,
                seed=seed,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "vlm-summary.json"
    summary_csv = output_dir / "vlm-tag-metrics.csv"
    report_md = output_dir / "VLM_BENCHMARK_REPORT.md"
    write_json(
        summary_json,
        {
            "schema_version": 2,
            "generated_at": utc_now(),
            "ground_truth_sha256": sha256_file(ground_truth_path),
            "reference_profile": reference_profile,
            "reference_run_id": reference_run_id,
            "profiles": profile_metrics,
        },
    )
    _write_tag_csv(summary_csv, _quality_profiles(profile_metrics))
    report_md.write_text(
        _render_markdown(profile_metrics, reference_profile), encoding="utf-8"
    )
    return summary_json, summary_csv, report_md


def aggregate_records(
    records: list[dict[str, Any]], tags: list[str]
) -> dict[str, dict[str, dict[str, str]]]:
    observations: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    profile_videos: dict[str, set[str]] = defaultdict(set)
    for record in records:
        benchmark_key = record.get("benchmark_key", record["profile"])
        profile_videos[benchmark_key].add(record["video_id"])
        if not record.get("success") or not record.get("parsed_tags"):
            continue
        for tag, value in record["parsed_tags"].items():
            observations[(benchmark_key, record["video_id"], tag)].append(
                value
            )
    result: dict[str, dict[str, dict[str, str]]] = {}
    for profile, videos in profile_videos.items():
        result[profile] = {}
        for video_id in videos:
            result[profile][video_id] = {}
            for tag in tags:
                values = observations.get((profile, video_id, tag), [])
                result[profile][video_id][tag] = (
                    "yes"
                    if "yes" in values
                    else "no"
                    if values and all(value == "no" for value in values)
                    else "unknown"
                )
    return result


def quality_gate(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    failures = []
    epsilon = 1e-12
    validity = candidate.get("output_validity", {})
    if not validity.get("passed", True):
        failures.extend(validity.get("warnings", ["output validity failed"]))
    if candidate["micro_lower_bound_recall"] < (
        reference["micro_lower_bound_recall"] - 0.02 - epsilon
    ):
        failures.append("micro lower-bound recall loss > 2pp")
    if candidate["macro_lower_bound_recall"] < (
        reference["macro_lower_bound_recall"] - 0.05 - epsilon
    ):
        failures.append("macro lower-bound recall loss > 5pp")
    reference_by_tag = {row["tag"]: row for row in reference["tags"]}
    for row in candidate["tags"]:
        baseline = reference_by_tag[row["tag"]]
        if (
            row["gt_positive"] >= 20
            and row["lower_bound_recall"]
            < baseline["lower_bound_recall"] - 0.10 - epsilon
        ):
            failures.append(f"{row['tag']} recall loss > 10pp")
    return {"passed": not failures, "failures": failures}


def _profile_metrics(
    benchmark_key: str,
    predictions: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, str]],
    tags: list[str],
    records: list[dict[str, Any]],
    *,
    reference_ground_truth: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    rows = []
    total_gt = total_evaluated = total_tp = 0
    total_reference_evaluated = 0
    total_reference_predicted_positive = 0
    total_assumed_true_negative = 0
    total_assumed_false_positive = 0
    prediction_videos = set(predictions)
    matching = [
        record
        for record in records
        if record.get("benchmark_key", record.get("profile")) == benchmark_key
    ]
    excluded_video_ids = {
        video_id
        for record in matching
        for video_id in record.get("dataset_excluded_video_ids", [])
    }
    evaluation_scope_videos = prediction_videos | excluded_video_ids
    reference_ground_truth = reference_ground_truth or ground_truth
    for tag in tags:
        positives = {
            video_id
            for video_id, labels in ground_truth.items()
            if video_id in evaluation_scope_videos
            and labels.get(tag) == "yes"
        }
        evaluated = {
            video_id
            for video_id in positives
            if predictions.get(video_id, {}).get(tag) in {"yes", "no"}
        }
        true_positive = {
            video_id
            for video_id in evaluated
            if predictions[video_id][tag] == "yes"
        }
        predicted_positive = {
            video_id
            for video_id, labels in predictions.items()
            if labels.get(tag) == "yes"
        }
        reference_evaluated = {
            video_id
            for video_id in prediction_videos
            if reference_ground_truth.get(video_id, {}).get(tag)
            in {"yes", "no"}
            and predictions.get(video_id, {}).get(tag) in {"yes", "no"}
        }
        reference_predicted_positive = {
            video_id
            for video_id in reference_evaluated
            if predictions[video_id][tag] == "yes"
        }
        assumed_false_positive = {
            video_id
            for video_id in reference_predicted_positive
            if reference_ground_truth[video_id][tag] == "no"
        }
        assumed_true_negative = {
            video_id
            for video_id in reference_evaluated
            if predictions[video_id][tag] == "no"
            and reference_ground_truth[video_id][tag] == "no"
        }
        gt_count = len(positives)
        evaluated_count = len(evaluated)
        tp_count = len(true_positive)
        total_gt += gt_count
        total_evaluated += evaluated_count
        total_tp += tp_count
        total_reference_evaluated += len(reference_evaluated)
        total_reference_predicted_positive += len(
            reference_predicted_positive
        )
        total_assumed_true_negative += len(assumed_true_negative)
        total_assumed_false_positive += len(assumed_false_positive)
        rows.append(
            {
                "tag": tag,
                "gt_positive": gt_count,
                "evaluated_positive": evaluated_count,
                "true_positive": tp_count,
                "false_negative": evaluated_count - tp_count,
                "not_evaluated": gt_count - evaluated_count,
                "recall_on_evaluated": (
                    tp_count / evaluated_count if evaluated_count else None
                ),
                "coverage": evaluated_count / gt_count if gt_count else None,
                "lower_bound_recall": tp_count / gt_count if gt_count else None,
                "predicted_positive": len(predicted_positive),
                "unverified_positive": sum(
                    ground_truth.get(video_id, {}).get(tag) != "yes"
                    for video_id in predicted_positive
                ),
                "assumed_negative_evaluated": sum(
                    reference_ground_truth.get(video_id, {}).get(tag) == "no"
                    for video_id in reference_evaluated
                ),
                "assumed_true_negative": len(assumed_true_negative),
                "assumed_false_positive": len(assumed_false_positive),
                "reference_precision": (
                    tp_count / len(reference_predicted_positive)
                    if reference_predicted_positive
                    else None
                ),
                "reference_accuracy": (
                    (tp_count + len(assumed_true_negative))
                    / len(reference_evaluated)
                    if reference_evaluated
                    else None
                ),
            }
        )
    macro_values = [
        row["lower_bound_recall"]
        for row in rows
        if row["lower_bound_recall"] is not None
    ]
    completed = [record for record in matching if record.get("success")]
    usable = [record for record in completed if _has_complete_output(record)]
    videos = {record["video_id"] for record in completed}
    requested_videos = {
        record["video_id"] for record in matching
    } | excluded_video_ids
    total_e2e = sum(float(record.get("e2e_seconds", 0)) for record in completed)
    request_e2e_per_video = total_e2e / len(videos) if videos else None
    wall_seconds = _allocated_profile_wall_seconds(
        benchmark_key, completed, records
    )
    run_wall_values = {
        float(record["run_wall_seconds"])
        for record in completed
        if record.get("run_wall_seconds") is not None
    }
    wall_seconds_reconstructed = bool(
        _common_value(completed, "concurrency") == 1
        and len(run_wall_values) == 1
        and wall_seconds is not None
        and wall_seconds > next(iter(run_wall_values)) + 1e-9
    )
    wall_seconds_per_video = (
        wall_seconds / len(videos)
        if wall_seconds is not None and videos
        else None
    )
    models = {str(record.get("model", "unknown")) for record in matching}
    run_ids = {str(record.get("run_id", "unknown")) for record in matching}
    profiles = {str(record["profile"]) for record in matching}
    gpu_counts = {
        int(record["instance"]["gpu_count"])
        for record in matching
        if record.get("instance", {}).get("gpu_count") is not None
    }
    hourly_prices = {
        float(record["instance"]["hourly_usd"])
        for record in matching
        if record.get("instance", {}).get("hourly_usd") is not None
    }
    instance_types = {
        str(record["instance"]["type"])
        for record in matching
        if record.get("instance", {}).get("type")
    }
    gpu_count = next(iter(gpu_counts)) if len(gpu_counts) == 1 else None
    hourly_usd = next(iter(hourly_prices)) if len(hourly_prices) == 1 else None
    prompt_tokens = sum(
        int(record.get("usage", {}).get("prompt_tokens", 0))
        for record in completed
    )
    completion_tokens = sum(
        int(record.get("usage", {}).get("completion_tokens", 0))
        for record in completed
    )
    completion_token_values = [
        float(record.get("usage", {}).get("completion_tokens", 0))
        for record in completed
    ]
    e2e_values = [
        float(record.get("e2e_seconds", 0)) for record in completed
    ]
    selected_frame_values = [
        float(record["selected_frame_count"])
        for record in completed
        if record.get("selected_frame_count") is not None
    ]
    clip_duration_values = [
        max(
            0.0,
            float(record["clip_end_seconds"])
            - float(record["clip_start_seconds"]),
        )
        for record in completed
        if record.get("clip_start_seconds") is not None
        and record.get("clip_end_seconds") is not None
    ]
    ttft_values = [
        float(record["ttft_seconds"])
        for record in completed
        if record.get("ttft_seconds") is not None
    ]
    telemetry_values = [
        record["run_gpu_telemetry"]
        for record in completed
        if record.get("run_gpu_telemetry")
    ]
    telemetry = (
        telemetry_values[0]
        if telemetry_values
        and all(value == telemetry_values[0] for value in telemetry_values)
        else {}
    )
    prefill_seconds = _metric_total(
        completed, "vllm:request_prefill_time_seconds_sum"
    )
    decode_seconds = _metric_total(
        completed, "vllm:request_decode_time_seconds_sum"
    )
    mm_cache_hits = _metric_total(completed, "vllm:mm_cache_hits_total")
    mm_cache_queries = _metric_total(
        completed, "vllm:mm_cache_queries_total"
    )
    prefix_cache_hits = _metric_total(
        completed, "vllm:prefix_cache_hits_total"
    )
    prefix_cache_queries = _metric_total(
        completed, "vllm:prefix_cache_queries_total"
    )
    preemptions = _metric_total(
        completed, "vllm:num_preemptions_total"
    )
    length_finished = sum(
        record.get("finish_reason") == "length" for record in completed
    )
    api_attempts = sum(
        int(record.get("attempt_count", 1)) for record in matching
    )
    repaired_requests = sum(
        bool(record.get("repair_attempted")) for record in matching
    )
    pipeline_group_requests = sum(
        int(record.get("pipeline_group_request_count", 0))
        for record in matching
    )
    pipeline_confirmation_requests = sum(
        bool(record.get("pipeline_confirmation_attempted"))
        for record in matching
    )
    complete_predictions = [
        predictions[video_id]
        for video_id in videos
        if all(
            predictions.get(video_id, {}).get(tag) in {"yes", "no"}
            for tag in tags
        )
    ]
    positive_counts = [
        sum(labels[tag] == "yes" for tag in tags)
        for labels in complete_predictions
    ]
    all_positive_fraction = (
        sum(count == len(tags) for count in positive_counts)
        / len(positive_counts)
        if positive_counts
        else None
    )
    all_negative_fraction = (
        sum(count == 0 for count in positive_counts) / len(positive_counts)
        if positive_counts
        else None
    )
    degenerate_all_positive = bool(
        len(positive_counts) >= 5
        and all_positive_fraction is not None
        and all_positive_fraction >= 0.1
    )
    usable_output_videos = {
        record["video_id"] for record in usable
    }
    request_output_coverage = (
        len(usable_output_videos) / len(requested_videos)
        if requested_videos
        else None
    )
    validity_warnings = []
    if degenerate_all_positive:
        validity_warnings.append(
            "all-positive output on at least 10% of complete videos"
        )
    if (
        request_output_coverage is not None
        and request_output_coverage < 0.99
    ):
        validity_warnings.append("parseable output coverage below 99%")
    return {
        "benchmark_key": benchmark_key,
        "run_id": next(iter(run_ids)) if len(run_ids) == 1 else "mixed",
        "model": next(iter(models)) if len(models) == 1 else "mixed",
        "profile": next(iter(profiles)) if len(profiles) == 1 else benchmark_key,
        "serving_profile": _common_value(
            completed, "serving_profile"
        ),
        "concurrency": _common_value(completed, "concurrency"),
        "video_count": len(videos),
        "requested_video_count": len(requested_videos),
        "dataset_excluded_video_count": len(excluded_video_ids),
        "requests": len(completed),
        "api_attempts": api_attempts,
        "api_attempts_per_video": (
            api_attempts / len(videos) if videos else None
        ),
        "repaired_requests": repaired_requests,
        "pipeline_routing": _common_value(
            completed, "pipeline_routing"
        ),
        "pipeline_confirmation": _common_value(
            completed, "pipeline_confirmation"
        ),
        "pipeline_group_requests": pipeline_group_requests,
        "pipeline_group_requests_per_video": (
            pipeline_group_requests / len(videos)
            if videos
            else None
        ),
        "pipeline_confirmation_requests": (
            pipeline_confirmation_requests
        ),
        "failed_requests": len(matching) - len(completed),
        "parse_failed_requests": len(completed) - len(usable),
        "parse_recovered_requests": sum(
            bool(record.get("parse_recovered")) for record in completed
        ),
        "request_output_coverage": request_output_coverage,
        "requests_per_video": (
            len(completed) / len(videos) if videos else None
        ),
        "micro_recall_on_evaluated": (
            total_tp / total_evaluated if total_evaluated else None
        ),
        "micro_coverage": total_evaluated / total_gt if total_gt else None,
        "micro_lower_bound_recall": total_tp / total_gt if total_gt else None,
        "macro_lower_bound_recall": (
            statistics.fmean(macro_values) if macro_values else None
        ),
        "reference_precision": (
            total_tp / total_reference_predicted_positive
            if total_reference_predicted_positive
            else None
        ),
        "reference_accuracy": (
            (total_tp + total_assumed_true_negative)
            / total_reference_evaluated
            if total_reference_evaluated
            else None
        ),
        "reference_evaluated_pairs": total_reference_evaluated,
        "assumed_false_positive": total_assumed_false_positive,
        "assumed_true_negative": total_assumed_true_negative,
        # Retained for compatibility with earlier summaries. This is summed
        # request latency, not elapsed instance time under concurrency.
        "e2e_seconds_per_video": request_e2e_per_video,
        "request_e2e_seconds_per_video": request_e2e_per_video,
        "wall_seconds": wall_seconds,
        "wall_seconds_per_video": wall_seconds_per_video,
        "wall_seconds_reconstructed": wall_seconds_reconstructed,
        "request_e2e_seconds_p50": _percentile(e2e_values, 0.50),
        "request_e2e_seconds_p95": _percentile(e2e_values, 0.95),
        "request_e2e_seconds_p99": _percentile(e2e_values, 0.99),
        "request_e2e_seconds_max": max(e2e_values) if e2e_values else None,
        "selected_frames_p50": _percentile(
            selected_frame_values, 0.50
        ),
        "selected_frames_p95": _percentile(
            selected_frame_values, 0.95
        ),
        "selected_frames_max": (
            max(selected_frame_values)
            if selected_frame_values
            else None
        ),
        "clip_duration_seconds_p50": _percentile(
            clip_duration_values, 0.50
        ),
        "clip_duration_seconds_p95": _percentile(
            clip_duration_values, 0.95
        ),
        "ttft_seconds_p50": (
            statistics.median(ttft_values) if ttft_values else None
        ),
        "ttft_seconds_p95": _percentile(ttft_values, 0.95),
        "completion_tokens_p50": _percentile(
            completion_token_values, 0.50
        ),
        "completion_tokens_p95": _percentile(
            completion_token_values, 0.95
        ),
        "completion_tokens_p99": _percentile(
            completion_token_values, 0.99
        ),
        "completion_tokens_max": (
            max(completion_token_values)
            if completion_token_values
            else None
        ),
        "length_finished_requests": length_finished,
        "length_finish_fraction": (
            length_finished / len(completed) if completed else None
        ),
        "prefill_seconds_per_video": (
            prefill_seconds / len(videos) if videos else None
        ),
        "decode_seconds_per_video": (
            decode_seconds / len(videos) if videos else None
        ),
        "mm_cache_hit_rate": (
            mm_cache_hits / mm_cache_queries if mm_cache_queries else None
        ),
        "prefix_cache_token_hit_rate": (
            prefix_cache_hits / prefix_cache_queries
            if prefix_cache_queries
            else None
        ),
        "preemptions": int(preemptions),
        "gpu_utilization_percent_mean": telemetry.get(
            "gpu_utilization_percent_mean"
        ),
        "gpu_utilization_percent_p95": telemetry.get(
            "gpu_utilization_percent_p95"
        ),
        "gpu_memory_used_mib_max": telemetry.get(
            "gpu_memory_used_mib_max"
        ),
        "gpu_memory_total_mib": telemetry.get(
            "gpu_memory_total_mib"
        ),
        "gpu_power_watts_mean": telemetry.get(
            "gpu_power_watts_mean"
        ),
        "gpu_temperature_c_max": telemetry.get(
            "gpu_temperature_c_max"
        ),
        "prompt_tokens_per_video": (
            prompt_tokens / len(videos) if videos else None
        ),
        "completion_tokens_per_video": (
            completion_tokens / len(videos) if videos else None
        ),
        "predicted_positive_labels_per_video": (
            statistics.fmean(positive_counts) if positive_counts else None
        ),
        "all_positive_video_fraction": all_positive_fraction,
        "all_negative_video_fraction": all_negative_fraction,
        "unverified_positive_labels_per_video": (
            sum(row["unverified_positive"] for row in rows) / len(videos)
            if videos
            else None
        ),
        "output_validity": {
            "passed": not validity_warnings,
            "warnings": validity_warnings,
        },
        "gpu_count": gpu_count,
        "instance_type": (
            next(iter(instance_types))
            if len(instance_types) == 1
            else None
        ),
        "capacity_reservation_id": _common_value(
            completed, "capacity_reservation_id"
        ),
        "container_image": _common_value(completed, "container_image"),
        "vllm_version": _common_value(completed, "vllm_version"),
        "git_commit": _common_value(completed, "git_commit"),
        "git_dirty": _common_value(completed, "git_dirty"),
        "experiment_sha256": _common_value(
            completed, "experiment_sha256"
        ),
        "dataset_lock_sha256": _common_value(
            completed, "dataset_lock_sha256"
        ),
        "media_lock_sha256": _common_value(
            completed, "media_lock_sha256"
        ),
        "labeling_config_sha256": _common_value(
            completed, "labeling_config_sha256"
        ),
        "gpu_seconds_per_video": (
            wall_seconds_per_video * gpu_count
            if wall_seconds_per_video is not None and gpu_count is not None
            else None
        ),
        "scenes_per_gpu_hour": (
            3600 / (wall_seconds_per_video * gpu_count)
            if wall_seconds_per_video and gpu_count
            else None
        ),
        "instance_hourly_usd": hourly_usd,
        "estimated_usd_per_1000_videos": (
            hourly_usd * wall_seconds_per_video / 3600 * 1000
            if hourly_usd is not None and wall_seconds_per_video is not None
            else None
        ),
        "tags": rows,
    }


def _bootstrap_delta(
    candidate: dict[str, dict[str, str]],
    reference: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, str]],
    tags: list[str],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    paired_video_ids = set(candidate) & set(reference)
    pairs_by_video = {
        video_id: [
            (
                1 if candidate.get(video_id, {}).get(tag) == "yes" else 0,
                1 if reference.get(video_id, {}).get(tag) == "yes" else 0,
            )
            for tag in tags
            if labels.get(tag) == "yes"
        ]
        for video_id, labels in ground_truth.items()
        if video_id in paired_video_ids
        and any(labels.get(tag) == "yes" for tag in tags)
    }
    video_ids = sorted(pairs_by_video)
    if not video_ids or samples <= 0:
        return None
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        drawn_videos = [
            video_ids[rng.randrange(len(video_ids))] for _ in video_ids
        ]
        draw = [
            pair
            for video_id in drawn_videos
            for pair in pairs_by_video[video_id]
        ]
        deltas.append(
            sum(candidate_value - reference_value for candidate_value, reference_value in draw)
            / len(draw)
        )
    deltas.sort()
    return [
        deltas[int(0.025 * (len(deltas) - 1))],
        deltas[int(0.975 * (len(deltas) - 1))],
    ]


def _write_tag_csv(path: Path, profiles: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark_key",
        "run_id",
        "profile",
        "tag",
        "gt_positive",
        "evaluated_positive",
        "true_positive",
        "false_negative",
        "not_evaluated",
        "recall_on_evaluated",
        "coverage",
        "lower_bound_recall",
        "predicted_positive",
        "unverified_positive",
        "assumed_negative_evaluated",
        "assumed_true_negative",
        "assumed_false_positive",
        "reference_precision",
        "reference_accuracy",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            for row in profile["tags"]:
                writer.writerow(
                    {
                        "benchmark_key": profile["benchmark_key"],
                        "run_id": profile["run_id"],
                        "profile": profile["profile"],
                        **row,
                    }
                )


def _render_markdown(
    profiles: list[dict[str, Any]], reference_profile: str
) -> str:
    quality_profiles = _quality_profiles(profiles)
    lines = [
        "# Cosmos 3 VLM シーンラベリング評価",
        "",
        f"Reference profile: `{reference_profile}`",
        "",
        "Ground Truth の `yes` のみを信頼し、`no` は未確認として扱う。"
        "正式比較と gate は LBR、coverage、positive-label recall のみを使用する。",
        "",
        "| Model | Run | Profile | Videos | Requests | Micro LBR | Coverage | Macro LBR | Frames p50/p95 | GPU util. | Wall sec/video | Scenes/GPUh | USD/1k videos | Gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in quality_profiles:
        gate = row.get("gate")
        gate_text = "reference" if gate is None else "PASS" if gate["passed"] else "FAIL"
        lines.append(
            f"| {row['model']} | {row['run_id']} | {row['profile']} | "
            f"{row['video_count']} | {row['requests']} | "
            f"{_pct(row['micro_lower_bound_recall'])} | "
            f"{_pct(row['micro_coverage'])} | "
            f"{_pct(row['macro_lower_bound_recall'])} | "
            f"{_number(row.get('selected_frames_p50'))}/"
            f"{_number(row.get('selected_frames_p95'))} | "
            f"{_number(row.get('gpu_utilization_percent_mean'))}% | "
            f"{row.get('wall_seconds_per_video') or 0:.2f} | "
            f"{_number(row.get('scenes_per_gpu_hour'))} | "
            f"{_number(row['estimated_usd_per_1000_videos'])} | {gate_text} |"
        )
    lines.extend(
        [
            "",
            "## 参考診断（暫定 negative 仮定）",
            "",
            "`no` を暫定的な negative と仮定した参考値であり、"
            "正式比較、順位付け、quality gate には使用しない。",
            "",
            "| Model | Run | Profile | Ref. Precision | Ref. Accuracy | "
            "Predicted yes/video | Unverified yes/video | All-positive videos |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quality_profiles:
        lines.append(
            f"| {row['model']} | {row['run_id']} | {row['profile']} | "
            f"{_pct(row['reference_precision'])} | "
            f"{_pct(row['reference_accuracy'])} | "
            f"{_number(row['predicted_positive_labels_per_video'])} | "
            f"{_number(row['unverified_positive_labels_per_video'])} | "
            f"{_pct(row['all_positive_video_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "Ref. Precision/Accuracy は GT=`no` が真の負例だった場合の参考値である。",
            "",
            "## 性能計測",
            "",
            "Smoke は品質の順位付けに使用せず、serving、latency、stage 時間、"
            "cache 効率の比較に限定する。",
            "",
            "| Scope | Model | Run | Videos | P50 | P95 | P99 | "
            "Prefill/video | Decode/video | MM cache | Prefix cache |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    max_scope = max(
        (int(row.get("video_count", 0)) for row in profiles),
        default=0,
    )
    for row in profiles:
        scope = (
            "Full"
            if int(row.get("video_count", 0)) == max_scope
            else "Smoke"
        )
        lines.append(
            f"| {scope} | {row['model']} | {row['run_id']} | "
            f"{row['video_count']} | "
            f"{_number(row.get('request_e2e_seconds_p50'))} | "
            f"{_number(row.get('request_e2e_seconds_p95'))} | "
            f"{_number(row.get('request_e2e_seconds_p99'))} | "
            f"{_number(row.get('prefill_seconds_per_video'))} | "
            f"{_number(row.get('decode_seconds_per_video'))} | "
            f"{_pct(row.get('mm_cache_hit_rate'))} | "
            f"{_pct(row.get('prefix_cache_token_hit_rate'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _quality_profiles(
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    max_scope = max(
        (int(row.get("video_count", 0)) for row in profiles),
        default=0,
    )
    return [
        row
        for row in profiles
        if int(row.get("video_count", 0)) == max_scope
    ]


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _active_wall_seconds(records: list[dict[str, Any]]) -> float | None:
    intervals: list[tuple[float, float]] = []
    for record in records:
        recorded_at = record.get("recorded_at")
        if not recorded_at:
            continue
        try:
            started_at = datetime.fromisoformat(str(recorded_at))
            duration = max(0.0, float(record.get("e2e_seconds", 0)))
        except (TypeError, ValueError):
            continue
        started_timestamp = started_at.timestamp()
        intervals.append((started_timestamp, started_timestamp + duration))
    if intervals:
        active_seconds = 0.0
        current_start, current_end = sorted(intervals)[0]
        for started_at, finished_at in sorted(intervals)[1:]:
            if started_at <= current_end:
                current_end = max(current_end, finished_at)
                continue
            active_seconds += current_end - current_start
            current_start, current_end = started_at, finished_at
        return active_seconds + current_end - current_start
    if not records:
        return None
    return sum(float(record.get("e2e_seconds", 0)) for record in records)


def _allocated_profile_wall_seconds(
    benchmark_key: str,
    completed: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
) -> float | None:
    run_ids = {str(record.get("run_id")) for record in completed}
    run_wall_values = {
        float(record["run_wall_seconds"])
        for record in completed
        if record.get("run_wall_seconds") is not None
    }
    if len(run_ids) != 1 or len(run_wall_values) != 1:
        return _active_wall_seconds(completed)

    run_id = next(iter(run_ids))
    run_records = [
        record
        for record in all_records
        if record.get("success") and str(record.get("run_id")) == run_id
    ]
    records_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        key = str(record.get("benchmark_key", record.get("profile")))
        records_by_profile[key].append(record)
    if len(records_by_profile) == 1:
        run_wall_seconds = next(iter(run_wall_values))
        active_wall_seconds = _active_wall_seconds(completed)
        concurrency = _common_value(completed, "concurrency")
        if (
            concurrency == 1
            and active_wall_seconds is not None
            and active_wall_seconds > run_wall_seconds
        ):
            return active_wall_seconds
        return run_wall_seconds

    active_by_profile = {
        key: _active_wall_seconds(profile_records) or 0.0
        for key, profile_records in records_by_profile.items()
    }
    total_active = sum(active_by_profile.values())
    if total_active <= 0 or benchmark_key not in active_by_profile:
        return _active_wall_seconds(completed)
    return (
        next(iter(run_wall_values))
        * active_by_profile[benchmark_key]
        / total_active
    )


def _sum_metric(records: list[dict[str, Any]], name: str) -> float:
    return sum(
        float((record.get("metrics_delta") or {}).get(name, 0) or 0)
        for record in records
    )


def _common_value(
    records: list[dict[str, Any]], name: str
) -> Any | None:
    values = {
        record[name]
        for record in records
        if record.get(name) is not None
    }
    return next(iter(values)) if len(values) == 1 else None


def _has_complete_output(record: dict[str, Any]) -> bool:
    parsed_tags = record.get("parsed_tags")
    if not isinstance(parsed_tags, dict):
        return False
    evaluated_tags = record.get("evaluated_tags")
    if not evaluated_tags:
        return True
    return all(
        parsed_tags.get(tag) in {"yes", "no"}
        for tag in evaluated_tags
    )


def _metric_total(records: list[dict[str, Any]], name: str) -> float:
    run_ids = {str(record.get("run_id")) for record in records}
    profile_counts = {
        int(record["run_profile_count"])
        for record in records
        if record.get("run_profile_count") is not None
    }
    run_deltas = [
        record["run_metrics_delta"]
        for record in records
        if isinstance(record.get("run_metrics_delta"), dict)
    ]
    if len(run_ids) == 1 and profile_counts == {1} and run_deltas:
        return float(run_deltas[0].get(name, 0) or 0)
    return _sum_metric(records, name)


def _read_run_records(run_dir: Path) -> list[dict[str, Any]]:
    metadata_path = run_dir / "run-metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    experiment = metadata.get("experiment", {})
    execution = metadata.get("execution", {})
    kubernetes = metadata.get("kubernetes", {})
    server = metadata.get("server", {})
    environment = metadata.get("environment", {})
    serving_profile = (
        kubernetes.get("pod_template_labels", {}).get(
            "benchmark-profile"
        )
    )
    model = experiment.get("model", "unknown")
    instance = experiment.get("instance", {})
    records = _read_jsonl(run_dir / "results.jsonl")
    telemetry = _gpu_telemetry_summary(
        run_dir / "gpu-telemetry.jsonl"
    )
    for record in records:
        if (
            record.get("success")
            and record.get("parsed_tags") is None
            and isinstance(record.get("content"), str)
            and record.get("evaluated_tags")
        ):
            parsed_tags, parse_error = parse_tags(
                record["content"],
                record["evaluated_tags"],
                allow_bare_positive_indices=record.get("prompt_variant")
                in {"reasoned-sparse", "decision-only-sparse"},
            )
            if parsed_tags is not None:
                record["original_parse_error"] = record.get("parse_error")
                record["parsed_tags"] = parsed_tags
                record["parse_error"] = parse_error
                record["parse_recovered"] = True
        profile = record["profile"]
        record.setdefault("model", model)
        record["run_id"] = run_dir.name
        record["instance"] = instance
        record["capacity_reservation_id"] = instance.get(
            "capacity_reservation_id"
        )
        record["container_image"] = kubernetes.get("image")
        record["vllm_version"] = server.get("version", {}).get("version")
        record["git_commit"] = environment.get("git_commit")
        record["git_dirty"] = environment.get("git_dirty")
        record["experiment_sha256"] = metadata.get("experiment_sha256")
        record["dataset_lock_sha256"] = metadata.get(
            "dataset_lock_sha256"
        )
        record["media_lock_sha256"] = metadata.get("media_lock_sha256")
        record["labeling_config_sha256"] = metadata.get(
            "labeling_config_sha256"
        )
        record["dataset_excluded_video_ids"] = execution.get(
            "excluded_video_ids", []
        )
        record["run_wall_seconds"] = execution.get("wall_seconds")
        record["run_metrics_delta"] = execution.get("metrics_delta")
        record["run_profile_count"] = len(execution.get("profiles", []))
        record["concurrency"] = execution.get("concurrency")
        record["serving_profile"] = serving_profile
        record["run_gpu_telemetry"] = telemetry
        record["benchmark_key"] = f"{run_dir.name}::{profile}"
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _gpu_telemetry_summary(
    path: Path,
) -> dict[str, float | None]:
    records = _read_jsonl(path)
    if not records:
        return {}
    utilization = [
        float(record["gpu_utilization_percent"])
        for record in records
        if record.get("gpu_utilization_percent") is not None
    ]
    memory_used = [
        float(record["memory_used_mib"])
        for record in records
        if record.get("memory_used_mib") is not None
    ]
    memory_total = [
        float(record["memory_total_mib"])
        for record in records
        if record.get("memory_total_mib") is not None
    ]
    power = [
        float(record["power_watts"])
        for record in records
        if record.get("power_watts") is not None
    ]
    temperature = [
        float(record["temperature_c"])
        for record in records
        if record.get("temperature_c") is not None
    ]
    return {
        "gpu_utilization_percent_mean": (
            statistics.fmean(utilization) if utilization else None
        ),
        "gpu_utilization_percent_p95": _percentile(
            utilization, 0.95
        ),
        "gpu_memory_used_mib_max": (
            max(memory_used) if memory_used else None
        ),
        "gpu_memory_total_mib": (
            max(memory_total) if memory_total else None
        ),
        "gpu_power_watts_mean": (
            statistics.fmean(power) if power else None
        ),
        "gpu_temperature_c_max": (
            max(temperature) if temperature else None
        ),
    }
