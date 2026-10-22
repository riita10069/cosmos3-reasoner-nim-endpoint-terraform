from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .vlm_evaluation import _read_run_records
from .road_evaluation import (
    _metrics_from_counts,
    aggregate_road_records,
    load_road_ground_truth,
)
from .util import load_json, sha256_file, utc_now, write_json


TEMPORAL_TAGS = {
    "ego_moving",
    "ego_stopped",
    "ego_turn_left",
    "ego_turn_right",
    "ego_lane_change",
    "car_braking",
    "car_stopped",
    "pedestrian_waiting_to_cross",
    "pedestrian_crossing",
}
PEDESTRIAN_TAGS = {
    "pedestrian_waiting_to_cross",
    "pedestrian_crossing",
}


def analyze_road_errors(
    run_dirs: list[Path],
    ground_truth_path: Path,
    output_dir: Path,
    *,
    split: str | None,
) -> tuple[Path, Path, Path]:
    ground_truth, tags = load_road_ground_truth(
        ground_truth_path, split=split
    )
    records = [
        record for run_dir in run_dirs for record in _read_run_records(run_dir)
    ]
    predictions = aggregate_road_records(records, tags)
    details: list[dict[str, Any]] = []
    profiles = []
    for benchmark_key, profile_predictions in sorted(predictions.items()):
        matching = [
            record
            for record in records
            if record["benchmark_key"] == benchmark_key
        ]
        timestamps = _timestamps_by_clip(matching)
        profile_details = _error_details(
            benchmark_key,
            profile_predictions,
            ground_truth,
            tags,
            timestamps,
        )
        details.extend(profile_details)
        profiles.append(
            _error_profile_summary(
                benchmark_key, matching, profile_details, tags
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "road-error-analysis.json"
    detail_path = output_dir / "road-error-details.csv"
    report_path = output_dir / "ROAD_ERROR_ANALYSIS.md"
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "dataset": "ROAD",
            "split": split,
            "ground_truth_path": str(ground_truth_path.resolve()),
            "ground_truth_sha256": sha256_file(ground_truth_path),
            "classification_note": (
                "temporally_covered_miss includes model, visual resolution, "
                "occlusion, and prompt/representation failures; it does not "
                "isolate model capacity"
            ),
            "profiles": profiles,
        },
    )
    _write_error_csv(detail_path, details)
    report_path.write_text(
        _render_error_markdown(profiles), encoding="utf-8"
    )
    return summary_path, detail_path, report_path


def audit_road_sampling(
    media_lock_paths: list[Path],
    ground_truth_path: Path,
    output_dir: Path,
    *,
    split: str | None,
) -> tuple[Path, Path]:
    ground_truth, tags = load_road_ground_truth(
        ground_truth_path, split=split
    )
    observations: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    sources = []
    for path in media_lock_paths:
        lock = load_json(path)
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "profiles_config_sha256": lock.get(
                    "profiles_config_sha256"
                ),
            }
        )
        for record in lock["records"]:
            clip_id = str(record["video_id"])
            if clip_id not in ground_truth:
                continue
            profile = str(
                record.get("evaluation_profile", record["profile"])
            )
            candidate = {
                "timestamps": [
                    float(value)
                    for value in record["selected_timestamps_seconds"]
                ],
                "frame_count": int(record["selected_frame_count"]),
            }
            current = observations[profile].get(clip_id)
            if current is not None and current != candidate:
                raise ValueError(
                    f"Conflicting sampling records for {profile}/{clip_id}"
                )
            observations[profile][clip_id] = candidate

    profiles = [
        _sampling_profile_summary(
            profile, by_clip, ground_truth, tags
        )
        for profile, by_clip in sorted(observations.items())
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "road-sampling-audit.json"
    report_path = output_dir / "ROAD_SAMPLING_AUDIT.md"
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "dataset": "ROAD",
            "split": split,
            "ground_truth_path": str(ground_truth_path.resolve()),
            "ground_truth_sha256": sha256_file(ground_truth_path),
            "media_locks": sources,
            "profiles": profiles,
        },
    )
    report_path.write_text(
        _render_sampling_markdown(profiles), encoding="utf-8"
    )
    return summary_path, report_path


def summarize_road_serving_repeats(
    evaluation_summary_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    source = load_json(evaluation_summary_path)
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in source["profiles"]:
        key = (
            str(row["model"]),
            str(row.get("serving_profile") or "unknown"),
            int(row.get("concurrency") or 0),
            str(row["profile"]),
        )
        grouped[key].append(row)
    groups = []
    for (model, serving_profile, concurrency, profile), rows in sorted(
        grouped.items()
    ):
        groups.append(
            {
                "model": model,
                "serving_profile": serving_profile,
                "concurrency": concurrency,
                "profile": profile,
                "replicate_count": len(rows),
                "run_ids": sorted(str(row["run_id"]) for row in rows),
                "metrics": {
                    key: _repeat_metric(
                        [
                            row["runtime"].get(key)
                            for row in rows
                            if row["runtime"].get(key) is not None
                        ]
                    )
                    for key in (
                        "wall_seconds_per_clip",
                        "request_e2e_seconds_p95",
                        "prefill_seconds_per_clip",
                        "decode_seconds_per_clip",
                        "estimated_usd_per_1000_clips",
                        "clips_per_gpu_hour",
                    )
                },
                "preemptions": [
                    int(row["runtime"]["preemptions"]) for row in rows
                ],
                "failed_requests": [
                    int(row["runtime"]["failed_requests"]) for row in rows
                ],
            }
        )
    baseline = next(
        (
            row
            for row in groups
            if row["serving_profile"] == "nano-server-default"
            and row["concurrency"] == 1
        ),
        None,
    )
    if baseline:
        baseline_wall = baseline["metrics"]["wall_seconds_per_clip"][
            "mean"
        ]
        for row in groups:
            wall = row["metrics"]["wall_seconds_per_clip"]["mean"]
            row["speedup_vs_default_c1"] = (
                baseline_wall / wall if baseline_wall and wall else None
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "road-serving-repeat-summary.json"
    report_path = output_dir / "ROAD_SERVING_REPEAT_SUMMARY.md"
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "source_path": str(evaluation_summary_path.resolve()),
            "source_sha256": sha256_file(evaluation_summary_path),
            "groups": groups,
        },
    )
    report_path.write_text(
        _render_repeat_markdown(groups), encoding="utf-8"
    )
    return summary_path, report_path


def evaluate_road_cascades(
    nano_run_dir: Path,
    super_run_dir: Path,
    ground_truth_path: Path,
    output_dir: Path,
    *,
    split: str | None,
    nano_usd_per_1000: float,
    super_usd_per_1000: float,
) -> tuple[Path, Path]:
    ground_truth, tags = load_road_ground_truth(
        ground_truth_path, split=split
    )
    nano = _single_run_predictions(nano_run_dir, tags)
    super_predictions = _single_run_predictions(super_run_dir, tags)
    scope = sorted(set(ground_truth) & set(nano) & set(super_predictions))
    if not scope:
        raise ValueError("Nano and Super runs have no common ROAD clips")
    complex_temporal = TEMPORAL_TAGS - {
        "ego_moving",
        "ego_stopped",
    }
    policies = [
        ("nano_only", lambda values: False),
        (
            "nano_positive_count_at_least_3",
            lambda values: _positive_count(values) >= 3,
        ),
        (
            "nano_pedestrian_positive",
            lambda values: _any_positive(values, PEDESTRIAN_TAGS),
        ),
        (
            "nano_complex_temporal_positive",
            lambda values: _any_positive(values, complex_temporal),
        ),
        ("super_all", lambda values: True),
    ]
    rows = []
    for policy_name, route in policies:
        routed = {
            clip_id for clip_id in scope if route(nano[clip_id])
        }
        predictions = {
            clip_id: (
                super_predictions[clip_id]
                if clip_id in routed
                else nano[clip_id]
            )
            for clip_id in scope
        }
        counts = _prediction_counts(
            scope, predictions, ground_truth, tags
        )
        metrics = _metrics_from_counts(counts)
        if policy_name == "nano_only":
            cost = nano_usd_per_1000
        elif policy_name == "super_all":
            cost = super_usd_per_1000
        else:
            cost = nano_usd_per_1000 + (
                super_usd_per_1000 * len(routed) / len(scope)
            )
        rows.append(
            {
                "policy": policy_name,
                "clip_count": len(scope),
                "routed_clip_count": len(routed),
                "routing_fraction": len(routed) / len(scope),
                "estimated_usd_per_1000_clips": cost,
                "counts": counts,
                "metrics": metrics,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "road-cascade-summary.json"
    report_path = output_dir / "ROAD_CASCADE_SUMMARY.md"
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "dataset": "ROAD",
            "split": split,
            "ground_truth_sha256": sha256_file(ground_truth_path),
            "nano_run": {
                "path": str(nano_run_dir.resolve()),
                "results_sha256": sha256_file(
                    nano_run_dir / "results.jsonl"
                ),
                "usd_per_1000_clips": nano_usd_per_1000,
            },
            "super_run": {
                "path": str(super_run_dir.resolve()),
                "results_sha256": sha256_file(
                    super_run_dir / "results.jsonl"
                ),
                "usd_per_1000_clips": super_usd_per_1000,
            },
            "routing_constraint": (
                "Routing uses Nano predictions only; ROAD ground truth is "
                "used only for post-hoc evaluation"
            ),
            "policies": rows,
        },
    )
    report_path.write_text(
        _render_cascade_markdown(rows), encoding="utf-8"
    )
    return summary_path, report_path


def temporal_coverage(
    timestamps: list[float],
    events: list[dict[str, Any]],
    *,
    clip_start: float,
    clip_end: float,
) -> dict[str, Any]:
    selected = sorted(set(float(value) for value in timestamps))
    inside = [
        timestamp
        for timestamp in selected
        if any(
            float(event["start_seconds"]) <= timestamp
            < float(event["end_seconds"]) + 1e-9
            for event in events
        )
    ]
    starts = [float(event["start_seconds"]) for event in events]
    ends = [float(event["end_seconds"]) for event in events]
    span = inside[-1] - inside[0] if len(inside) >= 2 else 0.0
    context_eligible = bool(
        events
        and min(starts) > clip_start + 1e-9
        and max(ends) < clip_end - 1e-9
    )
    has_before = bool(events and any(value < min(starts) for value in selected))
    has_after = bool(events and any(value >= max(ends) for value in selected))
    nearest_distance = None
    if events and not inside and selected:
        nearest_distance = min(
            _distance_to_interval(
                timestamp,
                float(event["start_seconds"]),
                float(event["end_seconds"]),
            )
            for timestamp in selected
            for event in events
        )
    return {
        "event_duration_seconds": round(
            sum(
                max(0.0, end - start)
                for start, end in zip(starts, ends, strict=True)
            ),
            6,
        ),
        "event_sample_count": len(inside),
        "event_sample_span_seconds": round(span, 6),
        "any_event_hit": bool(inside),
        "two_event_frames": len(inside) >= 2,
        "event_span_at_least_0_5s": span >= 0.5 - 1e-9,
        "boundary_context_eligible": context_eligible,
        "has_pre_event_context": has_before,
        "has_post_event_context": has_after,
        "has_both_boundary_contexts": context_eligible
        and has_before
        and has_after,
        "nearest_event_distance_seconds": (
            round(nearest_distance, 6)
            if nearest_distance is not None
            else None
        ),
    }


def _error_details(
    benchmark_key: str,
    predictions: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
    timestamps: dict[str, list[float]],
) -> list[dict[str, Any]]:
    details = []
    for clip_id in sorted(set(predictions) & set(ground_truth)):
        truth = ground_truth[clip_id]
        for tag in tags:
            label = truth["labels"][tag]
            prediction = predictions[clip_id].get(tag, "unknown")
            coverage = temporal_coverage(
                timestamps.get(clip_id, []),
                truth.get("events", {}).get(tag, []),
                clip_start=float(truth["start_seconds"]),
                clip_end=float(truth["end_seconds"]),
            )
            details.append(
                {
                    "benchmark_key": benchmark_key,
                    "clip_id": clip_id,
                    "source_video_id": truth["source_video_id"],
                    "tag": tag,
                    "ground_truth": label,
                    "prediction": prediction,
                    "outcome": _outcome(
                        label, prediction, tag, coverage
                    ),
                    "selected_frame_count": len(
                        timestamps.get(clip_id, [])
                    ),
                    **coverage,
                }
            )
    return details


def _outcome(
    label: str,
    prediction: str,
    tag: str,
    coverage: dict[str, Any],
) -> str:
    if label == "unknown":
        return "unknown_ground_truth"
    if prediction == "unknown":
        return (
            "abstained_positive"
            if label == "positive"
            else "abstained_negative"
        )
    if label == "negative":
        return "false_positive" if prediction == "yes" else "true_negative"
    if prediction == "yes":
        return "true_positive"
    if not coverage["any_event_hit"]:
        return "temporal_sampling_miss"
    if tag in TEMPORAL_TAGS and (
        not coverage["two_event_frames"]
        or not coverage["event_span_at_least_0_5s"]
    ):
        return "temporally_sparse_miss"
    return "temporally_covered_miss"


def _error_profile_summary(
    benchmark_key: str,
    records: list[dict[str, Any]],
    details: list[dict[str, Any]],
    tags: list[str],
) -> dict[str, Any]:
    counts = Counter(row["outcome"] for row in details)
    fn_names = {
        "temporal_sampling_miss",
        "temporally_sparse_miss",
        "temporally_covered_miss",
        "abstained_positive",
    }
    false_negatives = sum(counts[name] for name in fn_names)
    per_tag = []
    for tag in tags:
        tag_counts = Counter(
            row["outcome"] for row in details if row["tag"] == tag
        )
        tag_fn = sum(tag_counts[name] for name in fn_names)
        per_tag.append(
            {
                "tag": tag,
                "false_negative_count": tag_fn,
                "temporal_sampling_miss": tag_counts[
                    "temporal_sampling_miss"
                ],
                "temporally_sparse_miss": tag_counts[
                    "temporally_sparse_miss"
                ],
                "temporally_covered_miss": tag_counts[
                    "temporally_covered_miss"
                ],
                "sampling_or_sparse_fraction": _divide(
                    tag_counts["temporal_sampling_miss"]
                    + tag_counts["temporally_sparse_miss"],
                    tag_fn,
                ),
            }
        )
    first = records[0] if records else {}
    return {
        "benchmark_key": benchmark_key,
        "run_id": first.get("run_id"),
        "model": first.get("model"),
        "profile": first.get("profile"),
        "clip_count": len({row["clip_id"] for row in details}),
        "counts": dict(sorted(counts.items())),
        "false_negative_count": false_negatives,
        "temporal_sampling_miss": counts["temporal_sampling_miss"],
        "temporally_sparse_miss": counts["temporally_sparse_miss"],
        "temporally_covered_miss": counts["temporally_covered_miss"],
        "sampling_or_sparse_fraction": _divide(
            counts["temporal_sampling_miss"]
            + counts["temporally_sparse_miss"],
            false_negatives,
        ),
        "tags": per_tag,
    }


def _sampling_profile_summary(
    profile: str,
    by_clip: dict[str, dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
) -> dict[str, Any]:
    frame_counts = [row["frame_count"] for row in by_clip.values()]
    tag_rows = []
    all_coverages = []
    for tag in tags:
        coverages = [
            temporal_coverage(
                sample["timestamps"],
                ground_truth[clip_id]["events"][tag],
                clip_start=float(ground_truth[clip_id]["start_seconds"]),
                clip_end=float(ground_truth[clip_id]["end_seconds"]),
            )
            for clip_id, sample in by_clip.items()
            if ground_truth[clip_id]["labels"][tag] == "positive"
        ]
        all_coverages.extend(coverages)
        tag_rows.append(_coverage_summary(tag, coverages))
    return {
        "profile": profile,
        "clip_count": len(by_clip),
        "mean_selected_frames": (
            statistics.fmean(frame_counts) if frame_counts else None
        ),
        "max_selected_frames": max(frame_counts) if frame_counts else None,
        **_coverage_summary(None, all_coverages),
        "tags": tag_rows,
    }


def _coverage_summary(
    tag: str | None, coverages: list[dict[str, Any]]
) -> dict[str, Any]:
    eligible = [
        row for row in coverages if row["boundary_context_eligible"]
    ]
    result = {
        "positive_event_pairs": len(coverages),
        "any_event_hit_rate": _mean_bool(coverages, "any_event_hit"),
        "two_event_frame_rate": _mean_bool(
            coverages, "two_event_frames"
        ),
        "event_span_at_least_0_5s_rate": _mean_bool(
            coverages, "event_span_at_least_0_5s"
        ),
        "mean_event_samples": (
            statistics.fmean(
                float(row["event_sample_count"]) for row in coverages
            )
            if coverages
            else None
        ),
        "boundary_context_eligible_pairs": len(eligible),
        "both_boundary_contexts_rate": _mean_bool(
            eligible, "has_both_boundary_contexts"
        ),
    }
    return {"tag": tag, **result} if tag is not None else result


def _timestamps_by_clip(
    records: list[dict[str, Any]],
) -> dict[str, list[float]]:
    result = {}
    for record in records:
        clip_id = str(record["video_id"])
        timestamps = [
            float(value)
            for value in record.get("selected_timestamps_seconds", [])
        ]
        previous = result.get(clip_id)
        if previous is not None and previous != timestamps:
            raise ValueError(
                f"Conflicting timestamps for {clip_id} within one profile"
            )
        result[clip_id] = timestamps
    return result


def _single_run_predictions(
    run_dir: Path, tags: list[str]
) -> dict[str, dict[str, str]]:
    aggregated = aggregate_road_records(_read_run_records(run_dir), tags)
    if len(aggregated) != 1:
        raise ValueError(
            f"Expected one benchmark profile in {run_dir}, "
            f"found {len(aggregated)}"
        )
    return next(iter(aggregated.values()))


def _prediction_counts(
    clip_ids: list[str],
    predictions: dict[str, dict[str, str]],
    ground_truth: dict[str, dict[str, Any]],
    tags: list[str],
) -> dict[str, int]:
    counts = Counter(
        {
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
            "abstained_positive": 0,
            "abstained_negative": 0,
            "known_pairs": 0,
        }
    )
    for clip_id in clip_ids:
        for tag in tags:
            truth = ground_truth[clip_id]["labels"][tag]
            if truth == "unknown":
                continue
            counts["known_pairs"] += 1
            prediction = predictions[clip_id].get(tag, "unknown")
            if prediction == "unknown":
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
    return dict(counts)


def _positive_count(values: dict[str, str]) -> int:
    return sum(value == "yes" for value in values.values())


def _any_positive(values: dict[str, str], tags: set[str]) -> bool:
    return any(values.get(tag) == "yes" for tag in tags)


def _distance_to_interval(value: float, start: float, end: float) -> float:
    if start <= value <= end:
        return 0.0
    return min(abs(value - start), abs(value - end))


def _divide(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    return (
        statistics.fmean(bool(row[key]) for row in rows)
        if rows
        else None
    )


def _repeat_metric(values: list[float | int]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric) if numeric else None,
        "sample_standard_deviation": (
            statistics.stdev(numeric) if len(numeric) >= 2 else None
        ),
        "minimum": min(numeric) if numeric else None,
        "maximum": max(numeric) if numeric else None,
    }


def _write_error_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("ROAD error analysis produced no rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_error_markdown(profiles: list[dict[str, Any]]) -> str:
    lines = [
        "# ROAD error analysis",
        "",
        "| Model | FN | sampling miss | sparse miss | covered miss | sampling+sparse/FN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in profiles:
        lines.append(
            f"| {row['model']} | {row['false_negative_count']} | "
            f"{row['temporal_sampling_miss']} | "
            f"{row['temporally_sparse_miss']} | "
            f"{row['temporally_covered_miss']} | "
            f"{_percent(row['sampling_or_sparse_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "`temporally_covered_miss` also includes visibility, resolution, "
            "representation, and prompt failures.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_sampling_markdown(profiles: list[dict[str, Any]]) -> str:
    lines = [
        "# ROAD sampling audit",
        "",
        "| Profile | Frames | any hit | 2 frames | span >= 0.5s | boundary context |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in profiles:
        lines.append(
            f"| {row['profile']} | {row['mean_selected_frames']:.1f} | "
            f"{_percent(row['any_event_hit_rate'])} | "
            f"{_percent(row['two_event_frame_rate'])} | "
            f"{_percent(row['event_span_at_least_0_5s_rate'])} | "
            f"{_percent(row['both_boundary_contexts_rate'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_repeat_markdown(groups: list[dict[str, Any]]) -> str:
    lines = [
        "# ROAD serving repeat summary",
        "",
        "| Serving profile | c | n | wall/clip [s] | P95 [s] | prefill [s] | decode [s] | USD/1k | speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        metrics = row["metrics"]
        lines.append(
            f"| {row['serving_profile']} | {row['concurrency']} | "
            f"{row['replicate_count']} | "
            f"{_mean_sd(metrics['wall_seconds_per_clip'])} | "
            f"{_mean_sd(metrics['request_e2e_seconds_p95'])} | "
            f"{_mean_sd(metrics['prefill_seconds_per_clip'])} | "
            f"{_mean_sd(metrics['decode_seconds_per_clip'])} | "
            f"{_mean_sd(metrics['estimated_usd_per_1000_clips'])} | "
            f"{_number(row.get('speedup_vs_default_c1'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_cascade_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ROAD Nano/Super cascade",
        "",
        "| Policy | Routed | Precision | Recall | F1 | MCC | USD/1k |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['policy']} | {_percent(row['routing_fraction'])} | "
            f"{_percent(metrics['precision'])} | "
            f"{_percent(metrics['strict_recall'])} | "
            f"{_percent(metrics['strict_f1'])} | "
            f"{_number(metrics['selective_mcc'])} | "
            f"{row['estimated_usd_per_1000_clips']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _mean_sd(metric: dict[str, Any]) -> str:
    mean = metric["mean"]
    standard_deviation = metric["sample_standard_deviation"]
    if mean is None:
        return "--"
    if standard_deviation is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {standard_deviation:.3f}"


def _number(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}%"
