from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .dataset import probe_video
from .util import load_json, sha256_file, utc_now, write_json

LABEL_TYPES = {
    "av_action": {
        "labels_key": "all_av_action_labels",
        "frame_key": "av_action_ids",
        "tubes_key": "av_action_tubes",
    },
    "agent": {
        "labels_key": "all_agent_labels",
        "frame_key": "agent_ids",
        "tubes_key": "agent_tubes",
    },
    "action": {
        "labels_key": "all_action_labels",
        "frame_key": "action_ids",
        "tubes_key": "action_tubes",
    },
    "location": {
        "labels_key": "all_loc_labels",
        "frame_key": "loc_ids",
        "tubes_key": "loc_tubes",
    },
    "duplex": {
        "labels_key": "all_duplex_labels",
        "frame_key": "duplex_ids",
        "tubes_key": "duplex_tubes",
    },
    "triplet": {
        "labels_key": "all_triplet_labels",
        "frame_key": "triplet_ids",
        "tubes_key": "triplet_tubes",
    },
}


def audit_road_annotations(
    annotations_path: Path,
    videos_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    data = load_json(annotations_path)
    labels = _label_names(data)
    frame_counts = {
        label_type: Counter() for label_type in LABEL_TYPES
    }
    instance_counts = {
        label_type: Counter() for label_type in LABEL_TYPES
    }
    tube_counts = {
        label_type: Counter() for label_type in LABEL_TYPES
    }
    video_counts: dict[str, Counter[int]] = {
        label_type: Counter() for label_type in LABEL_TYPES
    }
    videos: list[dict[str, Any]] = []

    for video_id, video in sorted(data["db"].items()):
        per_video_seen = {
            label_type: set() for label_type in LABEL_TYPES
        }
        annotated_frames = 0
        for frame in video["frames"].values():
            if int(frame.get("annotated", 0)) != 1:
                continue
            annotated_frames += 1
            frame_sets = frame_label_sets(frame)
            for label_type, values in frame_sets.items():
                frame_counts[label_type].update(values)
                per_video_seen[label_type].update(values)
            instance_counts["av_action"].update(
                int(value) for value in frame.get("av_action_ids", [])
            )
            for annotation in frame.get("annos", {}).values():
                for label_type, spec in LABEL_TYPES.items():
                    if label_type == "av_action":
                        continue
                    instance_counts[label_type].update(
                        int(value)
                        for value in annotation.get(spec["frame_key"], [])
                    )
        for label_type, seen in per_video_seen.items():
            video_counts[label_type].update(seen)
        for label_type, spec in LABEL_TYPES.items():
            for tube in video.get(spec["tubes_key"], {}).values():
                tube_counts[label_type][int(tube["label_id"])] += 1

        video_path = videos_dir / f"{video_id}.mp4"
        videos.append(
            {
                "video_id": video_id,
                "split_ids": list(video["split_ids"]),
                "annotation_frame_count": int(video["numf"]),
                "annotated_frame_count": annotated_frames,
                "unannotated_frame_count": (
                    int(video["numf"]) - annotated_frames
                ),
                "video_path": (
                    str(video_path.resolve()) if video_path.is_file() else None
                ),
                "video_probe": (
                    probe_video(video_path) if video_path.is_file() else None
                ),
            }
        )

    rows = []
    for label_type, names in labels.items():
        for label_id, name in enumerate(names):
            rows.append(
                {
                    "label_type": label_type,
                    "label_id": label_id,
                    "label": name,
                    "frame_count": frame_counts[label_type][label_id],
                    "video_count": video_counts[label_type][label_id],
                    "instance_count": instance_counts[label_type][label_id],
                    "tube_count": tube_counts[label_type][label_id],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "road-annotation-audit.json"
    csv_path = output_dir / "road-label-frequency.csv"
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "dataset": "ROAD",
            "annotations_path": str(annotations_path.resolve()),
            "annotations_sha256": sha256_file(annotations_path),
            "videos_dir": str(videos_dir.resolve()),
            "video_count": len(videos),
            "annotation_frame_count": sum(
                video["annotation_frame_count"] for video in videos
            ),
            "annotated_frame_count": sum(
                video["annotated_frame_count"] for video in videos
            ),
            "unannotated_frame_count": sum(
                video["unannotated_frame_count"] for video in videos
            ),
            "missing_video_count": sum(
                video["video_path"] is None for video in videos
            ),
            "videos": videos,
            "labels": rows,
        },
    )
    _write_csv(csv_path, rows)
    return summary_path, csv_path


def build_road_ground_truth(
    annotations_path: Path,
    videos_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    data = load_json(annotations_path)
    config = load_json(config_path)
    labels = _label_names(data)
    tag_specs = _resolve_tag_specs(config["tags"], labels)
    fps = float(config["fps"])
    window_frames = _seconds_to_frames(
        float(config["window_seconds"]), fps
    )
    stride_frames = _seconds_to_frames(
        float(config["stride_seconds"]), fps
    )
    default_minimum = int(config["minimum_positive_frames"])
    fold = int(config["fold"])
    require_complete = bool(config.get("require_fully_annotated", True))

    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = output_dir / "road-ground-truth.jsonl"
    records: list[dict[str, Any]] = []
    label_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for video_id, video in sorted(data["db"].items()):
        split = _benchmark_split(video["split_ids"], fold)
        source_path = videos_dir / f"{video_id}.mp4"
        for start_frame in range(
            1,
            int(video["numf"]) - window_frames + 2,
            stride_frames,
        ):
            end_frame = start_frame + window_frames - 1
            frames = [
                video["frames"].get(str(frame_id))
                for frame_id in range(start_frame, end_frame + 1)
            ]
            complete = all(
                frame is not None and int(frame.get("annotated", 0)) == 1
                for frame in frames
            )
            if require_complete and not complete:
                continue
            frame_sets = [
                frame_label_sets(frame) if frame is not None else _empty_sets()
                for frame in frames
            ]
            tag_labels: dict[str, str] = {}
            tag_events: dict[str, list[dict[str, float | int]]] = {}
            tag_active_frames: dict[str, int] = {}
            for tag in tag_specs:
                active_offsets = [
                    offset
                    for offset, values in enumerate(frame_sets)
                    if set(tag["label_ids"]) & values[tag["label_type"]]
                ]
                minimum = int(
                    tag.get("minimum_positive_frames", default_minimum)
                )
                if not complete:
                    value = "unknown"
                elif len(active_offsets) >= minimum:
                    value = "positive"
                elif not active_offsets:
                    value = "negative"
                else:
                    value = "unknown"
                tag_labels[tag["id"]] = value
                tag_active_frames[tag["id"]] = len(active_offsets)
                tag_events[tag["id"]] = _event_intervals(
                    active_offsets,
                    source_start_frame=start_frame,
                    fps=fps,
                )
                label_counts[split][tag["id"]][value] += 1

            clip_id = (
                f"{video_id}__f{start_frame:06d}-f{end_frame:06d}"
            )
            records.append(
                {
                    "schema_version": 1,
                    "clip_id": clip_id,
                    "source_video_id": video_id,
                    "source_video_path": str(source_path.resolve()),
                    "split": split,
                    "official_split_ids": list(video["split_ids"]),
                    "fps": fps,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_seconds": round((start_frame - 1) / fps, 6),
                    "end_seconds": round(end_frame / fps, 6),
                    "frame_count": window_frames,
                    "fully_annotated": complete,
                    "labels": tag_labels,
                    "active_frame_counts": tag_active_frames,
                    "events": tag_events,
                }
            )

    with ground_truth_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    manifest_path = output_dir / "road-benchmark-manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "dataset": "ROAD",
            "annotations_path": str(annotations_path.resolve()),
            "annotations_sha256": sha256_file(annotations_path),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "ground_truth_path": str(ground_truth_path.resolve()),
            "ground_truth_sha256": sha256_file(ground_truth_path),
            "videos_dir": str(videos_dir.resolve()),
            "fold": fold,
            "fps": fps,
            "window_seconds": float(config["window_seconds"]),
            "stride_seconds": float(config["stride_seconds"]),
            "minimum_positive_frames": default_minimum,
            "require_fully_annotated": require_complete,
            "tags": tag_specs,
            "clip_count": len(records),
            "source_video_count": len(
                {record["source_video_id"] for record in records}
            ),
            "split_clip_counts": dict(
                Counter(record["split"] for record in records)
            ),
            "label_counts": {
                split: {
                    tag: dict(values)
                    for tag, values in sorted(tags.items())
                }
                for split, tags in sorted(label_counts.items())
            },
        },
    )
    return manifest_path, ground_truth_path


def select_road_smoke(
    ground_truth_path: Path,
    output_dir: Path,
    *,
    split: str,
    positives_per_tag: int,
    negatives_per_tag: int,
    max_clips: int,
    seed: int,
) -> tuple[Path, Path]:
    if positives_per_tag < 1 or negatives_per_tag < 1:
        raise ValueError("ROAD smoke quotas must be positive")
    if max_clips < 1:
        raise ValueError("ROAD smoke max_clips must be positive")
    records = [
        record
        for record in _read_jsonl(ground_truth_path)
        if record["split"] == split
    ]
    if not records:
        raise ValueError(f"No ROAD clips for split={split!r}")
    tags = list(records[0]["labels"])
    class_counts = {
        tag: Counter(record["labels"][tag] for record in records)
        for tag in tags
    }
    quotas = {
        (tag, label): min(
            positives_per_tag if label == "positive" else negatives_per_tag,
            class_counts[tag][label],
        )
        for tag in tags
        for label in ("positive", "negative")
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    covered: Counter[tuple[str, str]] = Counter()
    tie_order = list(records)
    random.Random(seed).shuffle(tie_order)
    tie_rank = {
        record["clip_id"]: index for index, record in enumerate(tie_order)
    }

    while len(selected) < max_clips:
        best = max(
            (
                record
                for record in records
                if record["clip_id"] not in selected_ids
            ),
            key=lambda record: (
                _smoke_selection_score(
                    record, tags, quotas, covered, class_counts
                ),
                -tie_rank[record["clip_id"]],
            ),
            default=None,
        )
        if best is None:
            break
        score = _smoke_selection_score(
            best, tags, quotas, covered, class_counts
        )
        if score <= 0:
            break
        selected.append(best)
        selected_ids.add(best["clip_id"])
        for tag in tags:
            label = best["labels"][tag]
            if label in {"positive", "negative"}:
                covered[(tag, label)] += 1

    unmet = {
        f"{tag}:{label}": quota - covered[(tag, label)]
        for (tag, label), quota in quotas.items()
        if covered[(tag, label)] < quota
    }
    if unmet:
        raise ValueError(
            f"ROAD smoke quotas cannot be met within {max_clips} clips: "
            f"{unmet}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "road-smoke-clip-ids.txt"
    ids_path.write_text(
        "".join(f"{record['clip_id']}\n" for record in selected),
        encoding="utf-8",
    )
    manifest_path = output_dir / "road-smoke-selection.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "ground_truth_path": str(ground_truth_path.resolve()),
            "ground_truth_sha256": sha256_file(ground_truth_path),
            "split": split,
            "seed": seed,
            "positive_quota_per_tag": positives_per_tag,
            "negative_quota_per_tag": negatives_per_tag,
            "max_clips": max_clips,
            "selected_clip_count": len(selected),
            "source_video_count": len(
                {record["source_video_id"] for record in selected}
            ),
            "selected_clip_ids_sha256": sha256_file(ids_path),
            "label_counts": {
                tag: dict(
                    Counter(record["labels"][tag] for record in selected)
                )
                for tag in tags
            },
            "selected_clips": [
                {
                    "clip_id": record["clip_id"],
                    "source_video_id": record["source_video_id"],
                    "labels": record["labels"],
                    "positive_event_duration_seconds": {
                        tag: round(
                            sum(
                                float(event["end_seconds"])
                                - float(event["start_seconds"])
                                for event in record["events"].get(tag, [])
                            ),
                            6,
                        )
                        for tag in tags
                        if record["labels"][tag] == "positive"
                    },
                }
                for record in selected
            ],
        },
    )
    return manifest_path, ids_path


def _smoke_selection_score(
    record: dict[str, Any],
    tags: list[str],
    quotas: dict[tuple[str, str], int],
    covered: Counter[tuple[str, str]],
    class_counts: dict[str, Counter[str]],
) -> float:
    score = 0.0
    for tag in tags:
        label = record["labels"][tag]
        key = (tag, label)
        if label not in {"positive", "negative"}:
            continue
        if covered[key] >= quotas[key]:
            continue
        rarity = 1 / max(1, class_counts[tag][label])
        sensitivity = 1.0
        if label == "positive":
            duration = sum(
                float(event["end_seconds"])
                - float(event["start_seconds"])
                for event in record["events"].get(tag, [])
            )
            if duration > 0:
                sensitivity += min(2.0, 1 / max(duration, 0.5))
        score += rarity * sensitivity
    return score


def frame_label_sets(frame: dict[str, Any]) -> dict[str, set[int]]:
    values = _empty_sets()
    values["av_action"].update(
        int(value) for value in frame.get("av_action_ids", [])
    )
    for annotation in frame.get("annos", {}).values():
        for label_type, spec in LABEL_TYPES.items():
            if label_type == "av_action":
                continue
            values[label_type].update(
                int(value)
                for value in annotation.get(spec["frame_key"], [])
            )
    return values


def _label_names(data: dict[str, Any]) -> dict[str, list[str]]:
    return {
        label_type: list(data[spec["labels_key"]])
        for label_type, spec in LABEL_TYPES.items()
    }


def _resolve_tag_specs(
    tags: list[dict[str, Any]],
    labels: dict[str, list[str]],
) -> list[dict[str, Any]]:
    resolved = []
    seen_ids: set[str] = set()
    for tag in tags:
        tag_id = str(tag["id"])
        if tag_id in seen_ids:
            raise ValueError(f"Duplicate ROAD tag ID: {tag_id}")
        seen_ids.add(tag_id)
        label_type = str(tag["label_type"])
        if label_type not in labels:
            raise ValueError(f"Unknown ROAD label type: {label_type}")
        name_to_id = {
            name: index for index, name in enumerate(labels[label_type])
        }
        unknown = sorted(set(tag["labels"]) - set(name_to_id))
        if unknown:
            raise ValueError(
                f"Unknown {label_type} labels for {tag_id}: {unknown}"
            )
        resolved.append(
            {
                **tag,
                "id": tag_id,
                "label_type": label_type,
                "label_ids": sorted(
                    name_to_id[name] for name in tag["labels"]
                ),
            }
        )
    return resolved


def _event_intervals(
    active_offsets: list[int],
    *,
    source_start_frame: int,
    fps: float,
) -> list[dict[str, float | int]]:
    intervals = []
    for start, end in _contiguous_ranges(active_offsets):
        source_start = source_start_frame + start
        source_end = source_start_frame + end
        intervals.append(
            {
                "start_frame": source_start,
                "end_frame": source_end,
                "start_seconds": round((source_start - 1) / fps, 6),
                "end_seconds": round(source_end / fps, 6),
            }
        )
    return intervals


def _contiguous_ranges(values: Iterable[int]) -> list[tuple[int, int]]:
    sorted_values = sorted(set(values))
    if not sorted_values:
        return []
    ranges = []
    start = previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value != previous + 1:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return ranges


def _benchmark_split(split_ids: list[str], fold: int) -> str:
    if f"val_{fold}" in split_ids:
        return "development"
    if f"train_{fold}" in split_ids:
        return "evaluation"
    return "unassigned"


def _seconds_to_frames(seconds: float, fps: float) -> int:
    frames = round(seconds * fps)
    if seconds <= 0 or fps <= 0 or frames < 1:
        raise ValueError("ROAD window, stride, and fps must be positive")
    return frames


def _empty_sets() -> dict[str, set[int]]:
    return {label_type: set() for label_type in LABEL_TYPES}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label_type",
        "label_id",
        "label",
        "frame_count",
        "video_count",
        "instance_count",
        "tube_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
