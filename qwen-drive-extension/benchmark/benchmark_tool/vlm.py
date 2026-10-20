from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .sampler import detect_event_peaks, select_adaptive_timestamps
from .util import load_json, run, sha256_file, utc_now, write_json


def load_ground_truth_labels(
    path: Path,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Empty ground truth: {path}")
    headers = [value.strip() for value in rows[0]]
    if "動画名" not in headers:
        raise ValueError("Ground truth has no 動画名 column")
    video_index = headers.index("動画名")
    tag_indexes: dict[str, list[int]] = defaultdict(list)
    for index, tag in enumerate(headers):
        if index != video_index and tag:
            tag_indexes[tag].append(index)

    ground_truth: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        if not row or video_index >= len(row) or not row[video_index].strip():
            continue
        video_id = Path(row[video_index].strip()).name
        labels: dict[str, str] = {}
        for tag, indexes in tag_indexes.items():
            values = {
                row[index].strip().lower()
                for index in indexes
                if index < len(row)
            }
            labels[tag] = (
                "yes"
                if "yes" in values
                else "no"
                if "no" in values
                else "unknown"
            )
        ground_truth[video_id] = labels
    return ground_truth, list(tag_indexes)


def load_positive_ground_truth(
    path: Path,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    ground_truth, tags = load_ground_truth_labels(path)
    return (
        {
            video_id: {
                tag: "yes" if value == "yes" else "unknown"
                for tag, value in labels.items()
            }
            for video_id, labels in ground_truth.items()
        },
        tags,
    )


def create_vlm_manifest(
    ground_truth_path: Path,
    tags_path: Path,
    output_path: Path,
    *,
    source_bucket: str,
    source_region: str,
    source_prefix: str,
    selection: str,
    per_tag: int,
    seed: int,
) -> Path:
    ground_truth, tags = load_positive_ground_truth(ground_truth_path)
    tag_config = load_json(tags_path)
    categories = tag_config["categories"]
    category_by_tag = {
        tag: category for category, values in categories.items() for tag in values
    }
    positive_by_tag = {
        tag: sorted(
            video_id
            for video_id, labels in ground_truth.items()
            if labels.get(tag) == "yes"
        )
        for tag in tags
    }

    selected: list[tuple[str, str | None]] = []
    if selection == "all":
        for video_id in sorted(ground_truth):
            positives = [
                tag for tag in tags if ground_truth[video_id].get(tag) == "yes"
            ]
            selected.append((video_id, positives[0] if positives else None))
    elif selection == "positive-smoke":
        # Offset each tag independently while preserving a deterministic,
        # reviewable order. Rare tags retain every available positive.
        for tag in tags:
            values = positive_by_tag[tag]
            if not values:
                continue
            offset = seed % len(values)
            rotated = values[offset:] + values[:offset]
            selected.extend((video_id, tag) for video_id in rotated[:per_tag])
    else:
        raise ValueError(f"Unknown selection: {selection}")

    seen: set[str] = set()
    videos = []
    for video_id, expected_positive in selected:
        if video_id in seen:
            continue
        seen.add(video_id)
        videos.append(
            {
                "video_id": video_id,
                "scenario": (
                    category_by_tag.get(expected_positive, "unverified")
                    if expected_positive
                    else "unverified"
                ),
                "expected_positive": expected_positive,
            }
        )

    write_json(
        output_path,
        {
            "schema_version": 2,
            "name": f"vlm-{selection}",
            "source": {
                "bucket": source_bucket,
                "region": source_region,
                "prefix": source_prefix.strip("/"),
            },
            "selection": {
                "kind": selection,
                "per_tag": per_tag if selection == "positive-smoke" else None,
                "seed": seed,
                "ground_truth_sha256": sha256_file(ground_truth_path),
                "positive_counts": {
                    tag: len(values) for tag, values in positive_by_tag.items()
                },
                "selected_video_count": len(videos),
            },
            "videos": videos,
        },
    )
    return output_path


def window_bounds(
    duration: float, window_seconds: float, overlap_fraction: float
) -> list[tuple[float, float]]:
    if duration <= 0 or window_seconds <= 0:
        raise ValueError("duration and window_seconds must be positive")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    if duration <= window_seconds:
        return [(0.0, round(duration, 6))]
    stride = window_seconds * (1 - overlap_fraction)
    starts: list[float] = []
    value = 0.0
    while value + window_seconds < duration:
        starts.append(value)
        value += stride
    final_start = max(0.0, duration - window_seconds)
    if not starts or not math.isclose(starts[-1], final_start, abs_tol=1e-6):
        starts.append(final_start)
    return [
        (round(start, 6), round(min(duration, start + window_seconds), 6))
        for start in starts
    ]


def uniform_timestamps(start: float, end: float, fps: float) -> list[float]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    count = math.floor((end - start) * fps) + 1
    values = [round(start + index / fps, 6) for index in range(count)]
    if values[-1] < end and end - values[-1] < 1 / fps * 0.05:
        values[-1] = round(end, 6)
    return values


def timestamp_manifest(timestamps: list[float]) -> str:
    return "\n".join(
        f"F{index:04d}={timestamp:.3f}s"
        for index, timestamp in enumerate(timestamps, 1)
    )


def build_vlm_media_lock(
    dataset_lock_path: Path,
    profiles_path: Path,
    output_dir: Path,
    *,
    selected_profiles: list[str] | None,
    workers: int = 1,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    dataset = load_json(dataset_lock_path)
    config = load_json(profiles_path)
    profiles = config["profiles"]
    profile_names = selected_profiles or list(profiles)
    unknown = sorted(set(profile_names) - set(profiles))
    if unknown:
        raise ValueError(f"Unknown VLM profiles: {unknown}")

    def build_video(video: dict[str, Any]) -> list[dict[str, Any]]:
        result = _build_vlm_video_records(
            video, profile_names, profiles, output_dir
        )
        print(f"sampled {video['video_id']}", flush=True)
        return result

    if workers == 1:
        record_batches = [build_video(video) for video in dataset["videos"]]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            record_batches = list(executor.map(build_video, dataset["videos"]))
    records = [record for batch in record_batches for record in batch]

    lock_path = output_dir / "vlm-media-lock.json"
    write_json(
        lock_path,
        {
            "schema_version": 2,
            "created_at": utc_now(),
            "dataset_lock": str(dataset_lock_path.resolve()),
            "dataset_lock_sha256": sha256_file(dataset_lock_path),
            "profiles_config": str(profiles_path.resolve()),
            "profiles_config_sha256": sha256_file(profiles_path),
            "media_root": str(output_dir.resolve()),
            "sampling_workers": workers,
            "frame_extraction": "sequential-nearest-frame",
            "records": records,
        },
    )
    return lock_path


def _build_vlm_video_records(
    video: dict[str, Any],
    profile_names: list[str],
    profiles: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    source = Path(video["path"])
    probe = video["probe"]
    duration = float(probe["duration_seconds"])
    if probe.get("frame_count") and probe.get("fps"):
        duration = min(
            duration,
            (int(probe["frame_count"]) - 1) / float(probe["fps"]),
        )
    records: list[dict[str, Any]] = []
    event_cache: dict[tuple[float, float, int], list[float]] = {}
    media_cache: dict[tuple[tuple[float, ...], float, bool], Path] = {}
    for name in profile_names:
        profile = profiles[name]
        timestamp_mode = profile.get("timestamp_mode", "text-manifest")
        if profile["kind"] == "uniform_windows":
            bounds = window_bounds(
                duration,
                float(profile["window_seconds"]),
                float(profile["overlap_fraction"]),
            )
            timestamp_sets = [
                uniform_timestamps(start, end, float(profile["fps"]))
                for start, end in bounds
            ]
            peaks: list[float] = []
        elif profile["kind"] == "adaptive":
            max_events = min(
                int(profile["max_events"]),
                max(
                    1,
                    math.ceil(
                        duration / float(profile["seconds_per_event"])
                    ),
                ),
            )
            event_key = (
                float(profile["analysis_fps"]),
                float(profile["nms_seconds"]),
                max_events,
            )
            if event_key not in event_cache:
                event_cache[event_key], _ = detect_event_peaks(
                    source,
                    analysis_fps=event_key[0],
                    nms_seconds=event_key[1],
                    max_events=event_key[2],
                )
            peaks = event_cache[event_key]
            timestamp_sets = [
                select_adaptive_timestamps(duration, peaks, profile)
            ]
            bounds = [(0.0, duration)]
        elif profile["kind"] == "uniform_full":
            timestamps = uniform_timestamps(
                0.0, duration, float(profile["fps"])
            )
            timestamp_sets = [
                _cap_timestamps(timestamps, int(profile["max_frames"]))
            ]
            bounds = [(0.0, duration)]
            peaks = []
        else:
            raise ValueError(f"Unknown profile kind: {profile['kind']}")

        for index, (bounds_item, timestamps) in enumerate(
            zip(bounds, timestamp_sets, strict=True)
        ):
            clip_id = f"w{index:04d}"
            target = output_dir / "derived" / name / video["video_id"]
            if len(timestamp_sets) > 1:
                target = target.with_name(
                    f"{target.stem}-{clip_id}{target.suffix}"
                )
            display_fps = float(
                profile.get("display_fps", profile.get("fps", 8))
            )
            burn_in = timestamp_mode == "text-manifest-burn-in"
            media_key = (tuple(timestamps), display_fps, burn_in)
            if media_key in media_cache:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    target.unlink(missing_ok=True)
                    os.link(media_cache[media_key], target)
                except OSError:
                    shutil.copyfile(media_cache[media_key], target)
            else:
                _build_timestamped_video(
                    source,
                    target,
                    timestamps,
                    output_fps=display_fps,
                    burn_in=burn_in,
                )
                media_cache[media_key] = target
            base_record = {
                "video_id": video["video_id"],
                "profile": name,
                "evaluation_profile": profile.get(
                    "evaluation_profile", name
                ),
                "kind": profile["kind"],
                "clip_id": clip_id,
                "prompt_variant": profile.get(
                    "prompt_variant", "direct"
                ),
                "media_first_prompt": bool(
                    profile.get("media_first_prompt", True)
                ),
                "max_tokens": profile.get("max_tokens"),
                "temperature": profile.get("temperature"),
                "top_p": profile.get("top_p"),
                "timestamp_mode": timestamp_mode,
                "source_path": str(source.resolve()),
                "source_sha256": video["sha256"],
                "source_duration_seconds": duration,
                "clip_start_seconds": round(bounds_item[0], 6),
                "clip_end_seconds": round(bounds_item[1], 6),
                "media_path": str(target.resolve()),
                "media_sha256": sha256_file(target),
                "media_io_kwargs": {"video": {"num_frames": -1}},
                "selected_frame_count": len(timestamps),
                "selected_timestamps_seconds": timestamps,
                "timestamp_manifest": (
                    timestamp_manifest(timestamps)
                    if timestamp_mode != "implicit-control"
                    else None
                ),
                "event_peak_timestamps_seconds": peaks,
            }
            tag_groups = profile.get(
                "tag_groups", [profile.get("tag_group", "all")]
            )
            for tag_group in tag_groups:
                records.append({**base_record, "tag_group": tag_group})
    return records


def reuse_vlm_media_lock(
    source_media_lock_path: Path,
    profiles_path: Path,
    output_path: Path,
    *,
    source_profile: str,
    target_profile: str,
) -> Path:
    source_lock = load_json(source_media_lock_path)
    profiles = load_json(profiles_path)["profiles"]
    if source_profile not in profiles or target_profile not in profiles:
        raise ValueError("Source and target profiles must exist in the profile config")

    prompt_fields = {
        "evaluation_profile",
        "media_first_prompt",
        "max_tokens",
        "prompt_variant",
        "tag_group",
        "tag_groups",
        "temperature",
        "timestamp_mode",
        "top_p",
    }
    source_media = {
        key: value
        for key, value in profiles[source_profile].items()
        if key not in prompt_fields
    }
    target_media = {
        key: value
        for key, value in profiles[target_profile].items()
        if key not in prompt_fields
    }
    source_media["burn_in"] = (
        profiles[source_profile].get("timestamp_mode")
        == "text-manifest-burn-in"
    )
    target_media["burn_in"] = (
        profiles[target_profile].get("timestamp_mode")
        == "text-manifest-burn-in"
    )
    if source_media != target_media:
        raise ValueError(
            "Cannot reuse media because source and target sampling parameters differ"
        )

    source_records = [
        record
        for record in source_lock["records"]
        if record["profile"] == source_profile
    ]
    if not source_records:
        raise ValueError(f"No media records found for profile: {source_profile}")

    target = profiles[target_profile]
    tag_groups = target.get(
        "tag_groups", [target.get("tag_group", "all")]
    )
    records = []
    seen_media = set()
    for source in source_records:
        media_key = (
            source["video_id"],
            source["clip_id"],
            source["media_sha256"],
        )
        if media_key in seen_media:
            continue
        seen_media.add(media_key)
        base = {
            **source,
            "profile": target_profile,
            "evaluation_profile": target.get(
                "evaluation_profile", target_profile
            ),
            "prompt_variant": target.get("prompt_variant", "direct"),
            "media_first_prompt": bool(
                target.get("media_first_prompt", True)
            ),
            "max_tokens": target.get("max_tokens"),
            "temperature": target.get("temperature"),
            "top_p": target.get("top_p"),
        }
        for tag_group in tag_groups:
            records.append({**base, "tag_group": tag_group})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output_path,
        {
            "schema_version": 2,
            "created_at": utc_now(),
            "dataset_lock": source_lock["dataset_lock"],
            "dataset_lock_sha256": source_lock["dataset_lock_sha256"],
            "profiles_config": str(profiles_path.resolve()),
            "profiles_config_sha256": sha256_file(profiles_path),
            "source_media_lock": str(source_media_lock_path.resolve()),
            "source_media_lock_sha256": sha256_file(source_media_lock_path),
            "source_profile": source_profile,
            "target_profile": target_profile,
            "media_root": source_lock.get(
                "media_root", str(source_media_lock_path.parent.resolve())
            ),
            "records": records,
        },
    )
    return output_path


def _cap_timestamps(values: list[float], maximum: int) -> list[float]:
    if len(values) <= maximum:
        return values
    indexes = np.linspace(0, len(values) - 1, maximum, dtype=int)
    return [values[index] for index in indexes]


def _build_timestamped_video(
    source: Path,
    target: Path,
    timestamps: list[float],
    *,
    output_fps: float,
    burn_in: bool,
) -> None:
    if not timestamps:
        raise ValueError("Cannot build a video without timestamps")
    target.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"Invalid video timing metadata: {source}")
    target_frames = [
        min(
            frame_count - 1,
            int(math.floor(timestamp * source_fps + 0.5)),
        )
        for timestamp in timestamps
    ]
    with tempfile.TemporaryDirectory(prefix="cosmos-vlm-") as temporary:
        frames_dir = Path(temporary)
        source_frame_index = -1
        frame: np.ndarray | None = None
        for index, (timestamp, target_frame) in enumerate(
            zip(timestamps, target_frames, strict=True), 1
        ):
            while source_frame_index < target_frame:
                ok, frame = capture.read()
                if not ok:
                    capture.release()
                    raise ValueError(
                        f"Cannot decode {source} at {timestamp:.3f}s"
                    )
                source_frame_index += 1
            if frame is None:
                capture.release()
                raise ValueError(f"Cannot decode first frame: {source}")
            output_frame = frame.copy() if burn_in else frame
            if burn_in:
                label = f"source t={timestamp:.3f}s"
                cv2.rectangle(
                    output_frame, (12, 12), (310, 52), (0, 0, 0), -1
                )
                cv2.putText(
                    output_frame,
                    label,
                    (22, 41),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            destination = frames_dir / f"frame-{index:06d}.png"
            if not cv2.imwrite(
                str(destination),
                output_frame,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            ):
                capture.release()
                raise OSError(f"Cannot write frame: {destination}")
        capture.release()
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(output_fps),
                "-i",
                str(frames_dir / "frame-%06d.png"),
                "-map_metadata",
                "-1",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ]
        )
