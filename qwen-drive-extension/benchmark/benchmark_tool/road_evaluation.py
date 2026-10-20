from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .vlm_evaluation import (
    _allocated_profile_wall_seconds,
    _common_value,
    _gpu_telemetry_summary,
    _has_complete_output,
    _metric_total,
    _percentile,
    _read_run_records,
)
from .util import sha256_file, utc_now, write_json


def evaluate_road(
    run_dirs: list[Path],
    ground_truth_path: Path,
    output_dir: Path,
    *,
    split: str | None,
    bootstrap_samples: int,
    seed: int,
    reference_profile: str | None = None,
    reference_benchmark_key: str | None = None,
) -> tuple[Path, Path, Path]:
    ground_truth, tags = load_road_ground_truth(
        ground_truth_path, split=split
    )
    records = [
        record for run_dir in run_dirs for record in _read_run_records(run_dir)
    ]
    predictions = aggregate_road_records(records, tags)
    profiles = [
        _profile_metrics(
            benchmark_key,
            profile_predictions,
            ground_truth,
            tags,
            records,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for benchmark_key, profile_predictions in sorted(predictions.items())
    ]
    _add_paired_comparisons(
        profiles,
        predictions,
        ground_truth,
        tags,
        reference_profile=reference_profile,
        reference_benchmark_key=reference_benchmark_key,
        samples=bootstrap_samples,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "road-complete-gt-summary.json"
    tag_csv_path = output_dir / "road-complete-gt-tag-metrics.csv"
    report_path = output_dir / "ROAD_OPEN_BENCHMARK.md"
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "dataset": "ROAD",
            "ground_truth_path": str(ground_truth_path.resolve()),
            "ground_truth_sha256": sha256_file(ground_truth_path),
            "split": split,
            "clip_count": len(ground_truth),
            "source_video_count": len(
                {
                    record["source_video_id"]
                    for record in ground_truth.values()
                }
            ),
            "tags": tags,
            "bootstrap": {
                "unit": "source_video_id",
                "samples": bootstrap_samples,
                "seed": seed,
            },
            "reference_profile": reference_profile,
            "reference_benchmark_key": reference_benchmark_key,
            "profiles": profiles,
        },
    )
    _write_tag_csv(tag_csv_path, profiles)
    report_path.write_text(
        _render_markdown(profiles, split), encoding="utf-8"
    )
    return summary_path, tag_csv_path, report_path


def load_road_ground_truth(
    path: Path, *, split: str | None = None
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    tags: list[str] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if split and record["split"] != split:
                continue
            clip_id = str(record["clip_id"])
            if clip_id in records:
                raise ValueError(f"Duplicate ROAD clip ID: {clip_id}")
            record_tags = list(record["labels"])
            if tags is None:
                tags = record_tags
            elif record_tags != tags:
                raise ValueError(
                    f"ROAD tag order differs for clip {clip_id}"
                )
            invalid = sorted(
                set(record["labels"].values())
                - {"positive", "negative", "unknown"}
            )
            if invalid:
                raise ValueError(
                    f"Invalid ROAD labels for {clip_id}: {invalid}"
                )
            records[clip_id] = record
    if not records:
        raise ValueError(f"No ROAD ground truth records for split={split!r}")
    return records, tags or []


def aggregate_road_records(
    records: list[dict[str, Any]], tags: list[str]
) -> dict[str, dict[str, dict[str, str]]]:
    observations: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    profile_clips: dict[str, set[str]] = defaultdict(set)
    for record in records:
        benchmark_key = str(
            record.get("benchmark_key", record["profile"])
        )
        clip_id = str(record["video_id"])
        profile_clips[benchmark_key].add(clip_id)
        if not record.get("success"):
            continue
        parsed_tags = record.get("parsed_tags")
        if not isinstance(parsed_tags, dict):
            continue
        for tag in tags:
            value = parsed_tags.get(tag)
            if value in {"yes", "no"}:
                observations[(benchmark_key, clip_id, tag)].append(value)

    predictions: dict[str, dict[str, dict[str, str]]] = {}
    for benchmark_key, clip_ids in profile_clips.items():
        predictions[benchmark_key] = {}
        for clip_id in clip_ids:
            predictions[benchmark_key][clip_id] = {
                tag: _majority_vote(
                    observations.get((benchmark_key, clip_id, tag), [])
                )
                for tag in tags
            }
    return predictions


def _majority_vote(values: list[str]) -> str:
    yes = values.count("yes")
    no = values.count("no")
    if yes > no:
        return "yes"
    if no > yes:
        return "no"
    return "unknown"


def _profile_metrics(
    benchmark_key: str,
    predictions: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
    records: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    matching = [
        record
        for record in records
        if str(record.get("benchmark_key", record.get("profile")))
        == benchmark_key
    ]
    scope = sorted(set(predictions) & set(ground_truth))
    ignored = sorted(set(predictions) - set(ground_truth))
    if not scope:
        raise ValueError(
            f"{benchmark_key} has no predictions matching ROAD ground truth"
        )

    tag_rows = [
        _tag_metrics(
            tag,
            scope,
            predictions,
            ground_truth,
        )
        for tag in tags
    ]
    totals = {
        key: sum(int(row[key]) for row in tag_rows)
        for key in (
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
            "abstained_positive",
            "abstained_negative",
            "known_pairs",
        )
    }
    micro = _metrics_from_counts(totals)
    macro_keys = (
        "precision",
        "strict_recall",
        "strict_f1",
        "overall_accuracy",
        "strict_balanced_accuracy",
        "selective_mcc",
    )
    macro = {
        key: _mean(
            [row[key] for row in tag_rows if row[key] is not None]
        )
        for key in macro_keys
    }
    coverage = (
        (
            totals["true_positive"]
            + totals["false_positive"]
            + totals["true_negative"]
            + totals["false_negative"]
        )
        / totals["known_pairs"]
        if totals["known_pairs"]
        else None
    )
    complete_clips = sum(
        all(predictions[clip_id].get(tag) in {"yes", "no"} for tag in tags)
        for clip_id in scope
    )
    evidence = _evidence_metrics(
        scope, matching, ground_truth, predictions, tags
    )
    distribution = _distribution_metrics(
        scope, predictions, ground_truth, tags
    )
    rare_scene_retrieval = _rare_scene_retrieval(
        scope,
        predictions,
        ground_truth,
        tags,
        seed=seed,
    )
    runtime = _runtime_metrics(benchmark_key, matching, records)
    bootstrap = _block_bootstrap_ci(
        scope,
        predictions,
        ground_truth,
        tags,
        samples=bootstrap_samples,
        seed=seed,
    )
    run_ids = {str(record.get("run_id", "unknown")) for record in matching}
    models = {str(record.get("model", "unknown")) for record in matching}
    profiles = {str(record.get("profile", "unknown")) for record in matching}
    concurrencies = {
        int(record["concurrency"])
        for record in matching
        if record.get("concurrency") is not None
    }
    serving_profiles = {
        str(record["serving_profile"])
        for record in matching
        if record.get("serving_profile") is not None
    }
    return {
        "benchmark_key": benchmark_key,
        "run_id": next(iter(run_ids)) if len(run_ids) == 1 else "mixed",
        "model": next(iter(models)) if len(models) == 1 else "mixed",
        "profile": (
            next(iter(profiles)) if len(profiles) == 1 else "mixed"
        ),
        "serving_profile": (
            next(iter(serving_profiles))
            if len(serving_profiles) == 1
            else None
        ),
        "concurrency": (
            next(iter(concurrencies)) if len(concurrencies) == 1 else None
        ),
        "clip_count": len(scope),
        "source_video_count": len(
            {ground_truth[clip_id]["source_video_id"] for clip_id in scope}
        ),
        "ignored_prediction_clip_count": len(ignored),
        "known_pairs": totals["known_pairs"],
        "output_pair_coverage": coverage,
        "complete_output_clip_fraction": complete_clips / len(scope),
        "output_validity": {
            "passed": bool(
                coverage is not None
                and coverage >= 0.99
                and complete_clips / len(scope) >= 0.99
            ),
            "minimum_required": 0.99,
        },
        "micro": micro,
        "macro": macro,
        "bootstrap_ci95": bootstrap,
        "distribution": distribution,
        "rare_scene_retrieval": rare_scene_retrieval,
        "evidence": evidence,
        "runtime": runtime,
        "tags": tag_rows,
    }


def _tag_metrics(
    tag: str,
    clip_ids: list[str],
    predictions: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "abstained_positive": 0,
        "abstained_negative": 0,
        "known_pairs": 0,
    }
    for clip_id in clip_ids:
        truth = ground_truth[clip_id]["labels"][tag]
        if truth == "unknown":
            continue
        counts["known_pairs"] += 1
        prediction = predictions.get(clip_id, {}).get(tag, "unknown")
        if prediction not in {"yes", "no"}:
            counts[
                "abstained_positive"
                if truth == "positive"
                else "abstained_negative"
            ] += 1
        elif truth == "positive" and prediction == "yes":
            counts["true_positive"] += 1
        elif truth == "positive":
            counts["false_negative"] += 1
        elif prediction == "yes":
            counts["false_positive"] += 1
        else:
            counts["true_negative"] += 1
    return {"tag": tag, **counts, **_metrics_from_counts(counts)}


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float | None]:
    tp = int(counts["true_positive"])
    fp = int(counts["false_positive"])
    tn = int(counts["true_negative"])
    fn = int(counts["false_negative"])
    abstained_positive = int(counts["abstained_positive"])
    abstained_negative = int(counts["abstained_negative"])
    known = int(counts["known_pairs"])
    emitted = tp + fp + tn + fn
    positives = tp + fn + abstained_positive
    negatives = tn + fp + abstained_negative
    precision = _divide(tp, tp + fp)
    selective_recall = _divide(tp, tp + fn)
    strict_recall = _divide(tp, positives)
    selective_specificity = _divide(tn, tn + fp)
    strict_specificity = _divide(tn, negatives)
    selective_f1 = _f1(precision, selective_recall)
    strict_f1 = _f1(precision, strict_recall)
    return {
        "coverage": _divide(emitted, known),
        "precision": precision,
        "selective_recall": selective_recall,
        "strict_recall": strict_recall,
        "selective_specificity": selective_specificity,
        "strict_specificity": strict_specificity,
        "selective_f1": selective_f1,
        "strict_f1": strict_f1,
        "selective_accuracy": _divide(tp + tn, emitted),
        "overall_accuracy": _divide(tp + tn, known),
        "selective_balanced_accuracy": _mean(
            [
                value
                for value in (selective_recall, selective_specificity)
                if value is not None
            ]
        ),
        "strict_balanced_accuracy": _mean(
            [
                value
                for value in (strict_recall, strict_specificity)
                if value is not None
            ]
        ),
        "selective_mcc": _mcc(tp, fp, tn, fn),
    }


def _evidence_metrics(
    clip_ids: list[str],
    records: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, str]],
    tags: list[str],
) -> dict[str, Any]:
    timestamps_by_clip: dict[str, list[float]] = {}
    for record in records:
        clip_id = str(record["video_id"])
        if clip_id not in timestamps_by_clip:
            timestamps_by_clip[clip_id] = [
                float(value)
                for value in record.get(
                    "selected_timestamps_seconds", []
                )
            ]
    event_pairs = hits = true_positive_hits = true_positive_misses = 0
    per_tag = []
    for tag in tags:
        tag_pairs = tag_hits = 0
        for clip_id in clip_ids:
            truth = ground_truth[clip_id]
            if truth["labels"][tag] != "positive":
                continue
            events = truth.get("events", {}).get(tag, [])
            if not events:
                continue
            tag_pairs += 1
            event_pairs += 1
            hit = any(
                float(event["start_seconds"]) <= timestamp
                < float(event["end_seconds"]) + 1e-9
                for event in events
                for timestamp in timestamps_by_clip.get(clip_id, [])
            )
            if hit:
                tag_hits += 1
                hits += 1
                if predictions[clip_id].get(tag) == "yes":
                    true_positive_hits += 1
            elif predictions[clip_id].get(tag) == "yes":
                true_positive_misses += 1
        per_tag.append(
            {
                "tag": tag,
                "positive_event_pairs": tag_pairs,
                "evidence_hit_pairs": tag_hits,
                "evidence_hit_rate": _divide(tag_hits, tag_pairs),
            }
        )
    return {
        "positive_event_pairs": event_pairs,
        "evidence_hit_pairs": hits,
        "evidence_hit_rate": _divide(hits, event_pairs),
        "true_positive_with_evidence_hit": true_positive_hits,
        "true_positive_without_evidence_hit": true_positive_misses,
        "tags": per_tag,
    }


def _distribution_metrics(
    clip_ids: list[str],
    predictions: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
) -> dict[str, Any]:
    rows = []
    true_counts = []
    predicted_counts = []
    for tag in tags:
        covered = [
            clip_id
            for clip_id in clip_ids
            if ground_truth[clip_id]["labels"][tag]
            in {"positive", "negative"}
            and predictions[clip_id].get(tag) in {"yes", "no"}
        ]
        true_count = sum(
            ground_truth[clip_id]["labels"][tag] == "positive"
            for clip_id in covered
        )
        predicted_count = sum(
            predictions[clip_id][tag] == "yes" for clip_id in covered
        )
        true_prevalence = _divide(true_count, len(covered))
        predicted_prevalence = _divide(predicted_count, len(covered))
        rows.append(
            {
                "tag": tag,
                "covered_clips": len(covered),
                "true_prevalence": true_prevalence,
                "predicted_prevalence": predicted_prevalence,
                "absolute_error": (
                    abs(predicted_prevalence - true_prevalence)
                    if true_prevalence is not None
                    and predicted_prevalence is not None
                    else None
                ),
            }
        )
        true_counts.append(float(true_count))
        predicted_counts.append(float(predicted_count))
    errors = [
        row["absolute_error"]
        for row in rows
        if row["absolute_error"] is not None
    ]
    return {
        "prevalence_mae": _mean(errors),
        "label_distribution_js_divergence": _js_divergence(
            true_counts, predicted_counts
        ),
        "tags": rows,
    }


def _runtime_metrics(
    benchmark_key: str,
    matching: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [record for record in matching if record.get("success")]
    usable = [record for record in completed if _has_complete_output(record)]
    clip_ids = {str(record["video_id"]) for record in matching}
    successful_clip_ids = {
        str(record["video_id"]) for record in completed
    }
    successful_clip_instances = {
        (
            str(record["video_id"]),
            int(record.get("repetition") or 1),
        )
        for record in completed
    }
    e2e = [float(record.get("e2e_seconds", 0)) for record in completed]
    ttft = [
        float(record["ttft_seconds"])
        for record in completed
        if record.get("ttft_seconds") is not None
    ]
    wall_seconds = _allocated_profile_wall_seconds(
        benchmark_key, completed, all_records
    )
    wall_per_clip = (
        wall_seconds / len(successful_clip_instances)
        if wall_seconds is not None and successful_clip_instances
        else None
    )
    instances = [
        record["instance"]
        for record in completed
        if isinstance(record.get("instance"), dict)
    ]
    instance = (
        instances[0]
        if instances
        and all(value == instances[0] for value in instances)
        else None
    )
    hourly_usd = (
        float(instance["hourly_usd"])
        if isinstance(instance, dict)
        and instance.get("hourly_usd") is not None
        else None
    )
    gpu_count = (
        int(instance["gpu_count"])
        if isinstance(instance, dict)
        and instance.get("gpu_count") is not None
        else None
    )
    telemetry_values = [
        record["run_gpu_telemetry"]
        for record in completed
        if record.get("run_gpu_telemetry")
    ]
    telemetry = telemetry_values[0] if telemetry_values else {}
    prefill = _metric_total(
        completed, "vllm:request_prefill_time_seconds_sum"
    )
    decode = _metric_total(
        completed, "vllm:request_decode_time_seconds_sum"
    )
    mm_hits = _metric_total(completed, "vllm:mm_cache_hits_total")
    mm_queries = _metric_total(completed, "vllm:mm_cache_queries_total")
    prefix_hits = _metric_total(
        completed, "vllm:prefix_cache_hits_total"
    )
    prefix_queries = _metric_total(
        completed, "vllm:prefix_cache_queries_total"
    )
    return {
        "request_count": len(matching),
        "successful_requests": len(completed),
        "failed_requests": len(matching) - len(completed),
        "parseable_requests": len(usable),
        "requested_clip_count": len(clip_ids),
        "successful_clip_count": len(successful_clip_ids),
        "evaluated_clip_instances": len(successful_clip_instances),
        "request_e2e_seconds_p50": _percentile(e2e, 0.50),
        "request_e2e_seconds_p95": _percentile(e2e, 0.95),
        "request_e2e_seconds_p99": _percentile(e2e, 0.99),
        "ttft_seconds_p50": _percentile(ttft, 0.50),
        "ttft_seconds_p95": _percentile(ttft, 0.95),
        "wall_seconds": wall_seconds,
        "wall_seconds_per_clip": wall_per_clip,
        "prefill_seconds_per_clip": (
            prefill / len(successful_clip_ids)
            if successful_clip_ids
            else None
        ),
        "decode_seconds_per_clip": (
            decode / len(successful_clip_ids)
            if successful_clip_ids
            else None
        ),
        "mm_cache_hit_rate": _divide(mm_hits, mm_queries),
        "prefix_cache_token_hit_rate": _divide(
            prefix_hits, prefix_queries
        ),
        "preemptions": int(
            _metric_total(completed, "vllm:num_preemptions_total")
        ),
        "gpu_count": gpu_count,
        "gpu_utilization_percent_mean": telemetry.get(
            "gpu_utilization_percent_mean"
        ),
        "gpu_utilization_percent_p95": telemetry.get(
            "gpu_utilization_percent_p95"
        ),
        "gpu_memory_used_mib_max": telemetry.get(
            "gpu_memory_used_mib_max"
        ),
        "instance_hourly_usd": hourly_usd,
        "estimated_usd_per_1000_clips": (
            hourly_usd * wall_per_clip / 3600 * 1000
            if hourly_usd is not None and wall_per_clip is not None
            else None
        ),
        "clips_per_gpu_hour": (
            3600 / (wall_per_clip * gpu_count)
            if wall_per_clip and gpu_count
            else None
        ),
    }


def _block_bootstrap_ci(
    clip_ids: list[str],
    predictions: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]] | None:
    by_video: dict[str, list[str]] = defaultdict(list)
    for clip_id in clip_ids:
        by_video[str(ground_truth[clip_id]["source_video_id"])].append(
            clip_id
        )
    video_ids = sorted(by_video)
    if samples <= 0 or not video_ids:
        return None
    draws: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(samples):
        sampled_clips = [
            clip_id
            for _ in video_ids
            for clip_id in by_video[
                video_ids[rng.randrange(len(video_ids))]
            ]
        ]
        values = _micro_metrics_for_clips(
            sampled_clips, predictions, ground_truth, tags
        )
        for key in (
            "strict_f1",
            "overall_accuracy",
            "strict_balanced_accuracy",
            "selective_mcc",
        ):
            if values[key] is not None:
                draws[key].append(float(values[key]))
    return {
        key: _ci95(values) for key, values in sorted(draws.items())
    }


def _add_paired_comparisons(
    profiles: list[dict[str, Any]],
    predictions: dict[str, dict[str, dict[str, str]]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
    *,
    reference_profile: str | None,
    reference_benchmark_key: str | None,
    samples: int,
    seed: int,
) -> None:
    exact_reference = next(
        (
            row
            for row in profiles
            if row["benchmark_key"] == reference_benchmark_key
        ),
        None,
    )
    if reference_benchmark_key and exact_reference is None:
        raise ValueError(
            f"Unknown ROAD reference benchmark key: "
            f"{reference_benchmark_key}"
        )
    references_by_run = {
        row["run_id"]: row
        for row in profiles
        if reference_profile and row["profile"] == reference_profile
    }
    for row in profiles:
        reference = exact_reference or references_by_run.get(row["run_id"])
        if reference is None or reference is row:
            continue
        candidate_predictions = predictions[row["benchmark_key"]]
        reference_predictions = predictions[reference["benchmark_key"]]
        paired_clips = sorted(
            set(candidate_predictions)
            & set(reference_predictions)
            & set(ground_truth)
        )
        row["comparison_reference"] = reference["benchmark_key"]
        row["paired_delta_ci95"] = _paired_block_bootstrap_delta(
            paired_clips,
            candidate_predictions,
            reference_predictions,
            ground_truth,
            tags,
            samples=samples,
            seed=seed,
        )
        row["disagreement_analysis"] = _disagreement_analysis(
            paired_clips,
            candidate_predictions,
            reference_predictions,
            ground_truth,
            tags,
        )


def _paired_block_bootstrap_delta(
    clip_ids: list[str],
    candidate: dict[str, dict[str, str]],
    reference: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]] | None:
    by_video: dict[str, list[str]] = defaultdict(list)
    for clip_id in clip_ids:
        by_video[str(ground_truth[clip_id]["source_video_id"])].append(
            clip_id
        )
    video_ids = sorted(by_video)
    if samples <= 0 or not video_ids:
        return None
    draws: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(samples):
        sampled_clips = [
            clip_id
            for _ in video_ids
            for clip_id in by_video[
                video_ids[rng.randrange(len(video_ids))]
            ]
        ]
        candidate_metrics = _micro_metrics_for_clips(
            sampled_clips, candidate, ground_truth, tags
        )
        reference_metrics = _micro_metrics_for_clips(
            sampled_clips, reference, ground_truth, tags
        )
        for key in (
            "strict_f1",
            "overall_accuracy",
            "strict_balanced_accuracy",
            "selective_mcc",
        ):
            left = candidate_metrics[key]
            right = reference_metrics[key]
            if left is not None and right is not None:
                draws[key].append(float(left) - float(right))
    return {
        key: _ci95(values) for key, values in sorted(draws.items())
    }


def _micro_metrics_for_clips(
    clip_ids: list[str],
    predictions: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
) -> dict[str, float | None]:
    totals = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "abstained_positive": 0,
        "abstained_negative": 0,
        "known_pairs": 0,
    }
    for clip_id in clip_ids:
        for tag in tags:
            truth = ground_truth[clip_id]["labels"][tag]
            if truth == "unknown":
                continue
            totals["known_pairs"] += 1
            prediction = predictions.get(clip_id, {}).get(tag, "unknown")
            if prediction not in {"yes", "no"}:
                totals[
                    "abstained_positive"
                    if truth == "positive"
                    else "abstained_negative"
                ] += 1
            elif truth == "positive" and prediction == "yes":
                totals["true_positive"] += 1
            elif truth == "positive":
                totals["false_negative"] += 1
            elif prediction == "yes":
                totals["false_positive"] += 1
            else:
                totals["true_negative"] += 1
    return _metrics_from_counts(totals)


def _write_tag_csv(
    path: Path, profiles: list[dict[str, Any]]
) -> None:
    fields = [
        "benchmark_key",
        "run_id",
        "model",
        "profile",
        "serving_profile",
        "tag",
        "known_pairs",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "abstained_positive",
        "abstained_negative",
        "coverage",
        "precision",
        "strict_recall",
        "strict_f1",
        "overall_accuracy",
        "strict_balanced_accuracy",
        "selective_mcc",
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
                        "model": profile["model"],
                        "profile": profile["profile"],
                        "serving_profile": profile.get("serving_profile"),
                        **{field: row.get(field) for field in fields[5:]},
                    }
                )


def _render_markdown(
    profiles: list[dict[str, Any]], split: str | None
) -> str:
    lines = [
        "# ROAD complete-GT benchmark",
        "",
        f"Split: `{split or 'all'}`",
        "",
        "Unknown GT pairs are excluded. Model abstentions are reported "
        "separately, reduce strict recall and overall accuracy, and are "
        "guarded by a 99% output-coverage threshold.",
        "",
        "| Model | Run | Profile | Serving | Clips | Coverage | Precision | "
        "Recall | F1 | Accuracy | Balanced Acc. | MCC | Evidence hit | "
        "USD/1k clips |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in profiles:
        runtime = row["runtime"]
        lines.append(
            f"| {row['model']} | {row['run_id']} | {row['profile']} | "
            f"{row.get('serving_profile') or '--'} | "
            f"{row['clip_count']} | {_pct(row['output_pair_coverage'])} | "
            f"{_pct(row['micro']['precision'])} | "
            f"{_pct(row['micro']['strict_recall'])} | "
            f"{_pct(row['micro']['strict_f1'])} | "
            f"{_pct(row['micro']['overall_accuracy'])} | "
            f"{_pct(row['micro']['strict_balanced_accuracy'])} | "
            f"{_number(row['micro']['selective_mcc'])} | "
            f"{_pct(row['evidence']['evidence_hit_rate'])} | "
            f"{_number(runtime['estimated_usd_per_1000_clips'])} |"
        )
    lines.extend(
        [
            "",
            "## Latency and cache",
            "",
            "| Model | Profile | P50 | P95 | P99 | Prefill/clip | "
            "Decode/clip | MM cache | Prefix cache | GPU util. |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in profiles:
        runtime = row["runtime"]
        lines.append(
            f"| {row['model']} | {row['profile']} | "
            f"{_number(runtime['request_e2e_seconds_p50'])} | "
            f"{_number(runtime['request_e2e_seconds_p95'])} | "
            f"{_number(runtime['request_e2e_seconds_p99'])} | "
            f"{_number(runtime['prefill_seconds_per_clip'])} | "
            f"{_number(runtime['decode_seconds_per_clip'])} | "
            f"{_pct(runtime['mm_cache_hit_rate'])} | "
            f"{_pct(runtime['prefix_cache_token_hit_rate'])} | "
            f"{_number(runtime['gpu_utilization_percent_mean'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _rare_scene_retrieval(
    clip_ids: list[str],
    predictions: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
    *,
    seed: int,
    rarity_threshold: float = 0.10,
    tie_samples: int = 1000,
) -> dict[str, Any] | None:
    prevalence = {}
    positive_count = {}
    for tag in tags:
        known = [
            ground_truth[clip_id]["labels"][tag]
            for clip_id in clip_ids
            if ground_truth[clip_id]["labels"][tag]
            in {"positive", "negative"}
        ]
        positive_count[tag] = known.count("positive")
        prevalence[tag] = _divide(positive_count[tag], len(known))
    rare_tags = [
        tag
        for tag in tags
        if prevalence[tag] is not None
        and prevalence[tag] <= rarity_threshold
        and positive_count[tag] > 0
    ]
    if not rare_tags:
        return None
    targets = {
        clip_id
        for clip_id in clip_ids
        if any(
            ground_truth[clip_id]["labels"][tag] == "positive"
            for tag in rare_tags
        )
    }
    if not targets:
        return None
    scores = {
        clip_id: sum(
            predictions.get(clip_id, {}).get(tag) == "yes"
            for tag in rare_tags
        )
        for clip_id in clip_ids
    }
    budgets = sorted(
        {
            max(1, min(len(clip_ids), math.ceil(len(clip_ids) * fraction)))
            for fraction in (0.01, 0.05, 0.10, 0.20)
        }
    )
    rng = random.Random(seed)
    metric_draws: dict[int, dict[str, list[float]]] = {
        budget: defaultdict(list) for budget in budgets
    }
    score_groups: dict[int, list[str]] = defaultdict(list)
    for clip_id, score in scores.items():
        score_groups[score].append(clip_id)
    for _ in range(tie_samples):
        ranking = []
        for score in sorted(score_groups, reverse=True):
            tied = list(score_groups[score])
            rng.shuffle(tied)
            ranking.extend(tied)
        for budget in budgets:
            selected = set(ranking[:budget])
            discovered = len(selected & targets)
            rare_pairs = sum(
                ground_truth[clip_id]["labels"][tag] == "positive"
                for clip_id in selected
                for tag in rare_tags
            )
            all_rare_pairs = sum(
                ground_truth[clip_id]["labels"][tag] == "positive"
                for clip_id in clip_ids
                for tag in rare_tags
            )
            metric_draws[budget]["precision"].append(
                discovered / budget
            )
            metric_draws[budget]["recall"].append(
                discovered / len(targets)
            )
            metric_draws[budget]["rare_label_recall"].append(
                rare_pairs / all_rare_pairs
            )
    base_prevalence = len(targets) / len(clip_ids)
    return {
        "definition": (
            "A rare scene contains at least one GT-positive tag whose "
            f"prevalence is at most {rarity_threshold:.0%}."
        ),
        "rarity_threshold": rarity_threshold,
        "rare_tags": rare_tags,
        "rare_scene_count": len(targets),
        "rare_scene_prevalence": base_prevalence,
        "ranking_score": "count of predicted-positive rare tags",
        "tie_break_samples": tie_samples,
        "budgets": [
            {
                "review_count": budget,
                "review_fraction": budget / len(clip_ids),
                "precision_mean": _mean(
                    metric_draws[budget]["precision"]
                ),
                "precision_ci95": _ci95(
                    metric_draws[budget]["precision"]
                ),
                "recall_mean": _mean(metric_draws[budget]["recall"]),
                "recall_ci95": _ci95(
                    metric_draws[budget]["recall"]
                ),
                "rare_label_recall_mean": _mean(
                    metric_draws[budget]["rare_label_recall"]
                ),
                "rare_label_recall_ci95": _ci95(
                    metric_draws[budget]["rare_label_recall"]
                ),
                "enrichment_mean": (
                    _mean(metric_draws[budget]["precision"])
                    / base_prevalence
                    if base_prevalence
                    else None
                ),
                "random_precision_expectation": base_prevalence,
                "random_recall_expectation": budget / len(clip_ids),
            }
            for budget in budgets
        ],
    }


def _disagreement_analysis(
    clip_ids: list[str],
    candidate: dict[str, dict[str, str]],
    reference: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
) -> dict[str, Any]:
    disagreement_pairs = agreement_pairs = 0
    candidate_disagreement_errors = candidate_agreement_errors = 0
    reference_disagreement_errors = reference_agreement_errors = 0
    clip_scores = {}
    for clip_id in clip_ids:
        clip_disagreements = 0
        for tag in tags:
            truth = ground_truth[clip_id]["labels"][tag]
            left = candidate.get(clip_id, {}).get(tag)
            right = reference.get(clip_id, {}).get(tag)
            if truth not in {"positive", "negative"}:
                continue
            if left not in {"yes", "no"} or right not in {"yes", "no"}:
                continue
            truth_value = "yes" if truth == "positive" else "no"
            if left != right:
                disagreement_pairs += 1
                clip_disagreements += 1
                candidate_disagreement_errors += left != truth_value
                reference_disagreement_errors += right != truth_value
            else:
                agreement_pairs += 1
                candidate_agreement_errors += left != truth_value
                reference_agreement_errors += right != truth_value
        clip_scores[clip_id] = clip_disagreements
    return {
        "paired_clip_count": len(clip_ids),
        "clips_with_disagreement": sum(
            score > 0 for score in clip_scores.values()
        ),
        "disagreement_pairs": disagreement_pairs,
        "agreement_pairs": agreement_pairs,
        "candidate_error_rate_on_disagreement": _divide(
            candidate_disagreement_errors, disagreement_pairs
        ),
        "candidate_error_rate_on_agreement": _divide(
            candidate_agreement_errors, agreement_pairs
        ),
        "reference_error_rate_on_disagreement": _divide(
            reference_disagreement_errors, disagreement_pairs
        ),
        "reference_error_rate_on_agreement": _divide(
            reference_agreement_errors, agreement_pairs
        ),
    }


def _mcc(tp: int, fp: int, tn: int, fn: int) -> float | None:
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    return (tp * tn - fp * fn) / denominator if denominator else None


def _f1(
    precision: float | None, recall: float | None
) -> float | None:
    if precision is None or recall is None:
        return None
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _ci95(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [
        float(ordered[int(0.025 * (len(ordered) - 1))]),
        float(ordered[int(0.975 * (len(ordered) - 1))]),
    ]


def _js_divergence(
    left_counts: list[float], right_counts: list[float]
) -> float | None:
    left_total = sum(left_counts)
    right_total = sum(right_counts)
    if left_total <= 0 or right_total <= 0:
        return None
    left = [value / left_total for value in left_counts]
    right = [value / right_total for value in right_counts]
    midpoint = [(a + b) / 2 for a, b in zip(left, right)]

    def kl(values: list[float], reference: list[float]) -> float:
        return sum(
            value * math.log2(value / target)
            for value, target in zip(values, reference)
            if value > 0 and target > 0
        )

    return (kl(left, midpoint) + kl(right, midpoint)) / 2


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"
