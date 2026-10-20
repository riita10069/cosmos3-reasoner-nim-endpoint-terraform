from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2

from .vlm import timestamp_manifest
from .sampler import detect_event_peaks, select_adaptive_timestamps
from .util import (
    load_json,
    run,
    sha256_file,
    utc_now,
    write_json,
)


def build_road_media_lock(
    ground_truth_path: Path,
    profiles_path: Path,
    output_dir: Path,
    *,
    selected_profiles: list[str] | None,
    split: str | None,
    workers: int,
    limit: int | None = None,
    video_ids_file: Path | None = None,
) -> tuple[Path, Path]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    profiles_config_sha256 = sha256_file(profiles_path)
    profiles_config = load_json(profiles_path)
    profiles = profiles_config["profiles"]
    profile_names = selected_profiles or list(profiles)
    unknown_profiles = sorted(set(profile_names) - set(profiles))
    if unknown_profiles:
        raise ValueError(f"Unknown ROAD profiles: {unknown_profiles}")

    clips = _read_jsonl(ground_truth_path)
    if split:
        clips = [clip for clip in clips if clip["split"] == split]
    selected_video_ids = _load_video_ids(video_ids_file)
    if selected_video_ids is not None:
        available_video_ids = {clip["clip_id"] for clip in clips}
        unknown_video_ids = sorted(
            selected_video_ids - available_video_ids
        )
        if unknown_video_ids:
            raise ValueError(
                f"Unknown ROAD clip IDs in {video_ids_file}: "
                f"{unknown_video_ids}"
            )
        clips = [
            clip
            for clip in clips
            if clip["clip_id"] in selected_video_ids
        ]
    if limit is not None:
        if limit < 1:
            raise ValueError("ROAD sample limit must be positive")
        clips = clips[:limit]
    by_video: dict[str, list[dict[str, Any]]] = {}
    for clip in clips:
        by_video.setdefault(clip["source_video_id"], []).append(clip)

    output_dir.mkdir(parents=True, exist_ok=True)

    def build_video(item: tuple[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        video_id, video_clips = item
        records = _build_source_media(
            video_id,
            video_clips,
            profile_names,
            profiles,
            output_dir,
        )
        print(f"sampled {video_id}", flush=True)
        return records

    items = sorted(by_video.items())
    if workers == 1:
        batches = [build_video(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            batches = list(executor.map(build_video, items))
    records = sorted(
        (record for batch in batches for record in batch),
        key=lambda value: (
            value["video_id"],
            value["profile"],
            value["tag_group"],
        ),
    )
    if sha256_file(profiles_path) != profiles_config_sha256:
        raise ValueError(
            "ROAD sampling profiles changed during media generation"
        )

    dataset_lock_path = output_dir / "road-dataset-lock.json"
    source_sha256 = {
        record["source_video_id"]: record["source_sha256"]
        for record in records
    }
    write_json(
        dataset_lock_path,
        {
            "schema_version": 1,
            "name": f"road-open-{split or 'all'}",
            "source": {"type": "ROAD-public"},
            "prepared_at": utc_now(),
            "ground_truth_path": str(ground_truth_path.resolve()),
            "ground_truth_sha256": sha256_file(ground_truth_path),
            "videos": [
                {
                    "video_id": clip["clip_id"],
                    "source_video_id": clip["source_video_id"],
                    "split": clip["split"],
                    "path": clip["source_video_path"],
                    "sha256": source_sha256[clip["source_video_id"]],
                    "clip_start_seconds": clip["start_seconds"],
                    "clip_end_seconds": clip["end_seconds"],
                    "labels": clip["labels"],
                    "events": clip["events"],
                }
                for clip in clips
            ],
        },
    )

    media_lock_path = output_dir / "road-media-lock.json"
    write_json(
        media_lock_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_lock": str(dataset_lock_path.resolve()),
            "dataset_lock_sha256": sha256_file(dataset_lock_path),
            "ground_truth_path": str(ground_truth_path.resolve()),
            "ground_truth_sha256": sha256_file(ground_truth_path),
            "profiles_config": str(profiles_path.resolve()),
            "profiles_config_sha256": profiles_config_sha256,
            "selected_profiles": profile_names,
            "media_root": str(output_dir.resolve()),
            "split": split,
            "limit": limit,
            "video_ids_file": (
                str(video_ids_file.resolve()) if video_ids_file else None
            ),
            "video_ids_file_sha256": (
                sha256_file(video_ids_file) if video_ids_file else None
            ),
            "sampling_workers": workers,
            "frame_extraction": "single-sequential-decode-per-source-video",
            "records": records,
        },
    )
    return dataset_lock_path, media_lock_path


def road_profile_timestamps(
    clip: dict[str, Any],
    profile: dict[str, Any],
    *,
    motion_observations: list[dict[str, float]] | None = None,
) -> tuple[list[float], list[float]]:
    start = float(clip["start_seconds"])
    end = float(clip["end_seconds"])
    source_fps = float(clip["fps"])
    final_timestamp = end - 1 / source_fps
    if profile["kind"] == "uniform":
        timestamps = _grid_exclusive(start, end, float(profile["fps"]))
        maximum = int(profile.get("max_frames", len(timestamps)))
        return _cap_timestamps(timestamps, maximum), []
    if profile["kind"] == "gt_oracle":
        if clip.get("split") != "development":
            raise ValueError(
                "GT oracle sampling is diagnostic-only and restricted to "
                "the ROAD development split"
            )
        return _gt_oracle_timestamps(clip, profile), []
    if profile["kind"] != "adaptive":
        raise ValueError(f"Unknown ROAD sampling kind: {profile['kind']}")
    if motion_observations is None:
        raise ValueError("Adaptive ROAD sampling requires motion observations")

    peaks = _clip_motion_peaks(
        motion_observations,
        start=start,
        end=end,
        nms_seconds=float(profile["nms_seconds"]),
        max_events=int(profile["max_events"]),
    )
    relative_peaks = [peak - start for peak in peaks]
    relative = select_adaptive_timestamps(
        final_timestamp - start,
        relative_peaks,
        profile,
    )
    return [round(start + value, 6) for value in relative], peaks


def _build_source_media(
    video_id: str,
    clips: list[dict[str, Any]],
    profile_names: list[str],
    profiles: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    clips = sorted(clips, key=lambda value: int(value["start_frame"]))
    source = Path(clips[0]["source_video_path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    source_digest = sha256_file(source)
    needs_motion = any(
        profiles[name]["kind"] == "adaptive" for name in profile_names
    )
    observations: list[dict[str, float]] = []
    if needs_motion:
        analysis_fps = max(
            float(profiles[name]["analysis_fps"])
            for name in profile_names
            if profiles[name]["kind"] == "adaptive"
        )
        _, observations = detect_event_peaks(
            source,
            analysis_fps=analysis_fps,
            nms_seconds=1,
            max_events=1,
        )

    plans_by_clip: dict[str, list[dict[str, Any]]] = {}
    for clip in clips:
        plans = []
        for profile_name in profile_names:
            profile = profiles[profile_name]
            timestamps, peaks = road_profile_timestamps(
                clip,
                profile,
                motion_observations=observations,
            )
            target = (
                output_dir
                / "derived"
                / profile_name
                / f"{clip['clip_id']}.mp4"
            )
            plans.append(
                {
                    "profile_name": profile_name,
                    "profile": profile,
                    "timestamps": timestamps,
                    "target_frames": [
                        _timestamp_frame(timestamp, float(clip["fps"]))
                        for timestamp in timestamps
                    ],
                    "event_peaks": peaks,
                    "target": target,
                }
            )
        plans_by_clip[clip["clip_id"]] = plans

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open ROAD video: {source}")
    current_frame = -1
    records = []
    try:
        for clip in clips:
            plans = plans_by_clip[clip["clip_id"]]
            requested = sorted(
                {
                    frame_index
                    for plan in plans
                    for frame_index in plan["target_frames"]
                }
            )
            frame_cache = {}
            for target_frame in requested:
                while current_frame < target_frame:
                    ok = capture.grab()
                    if not ok:
                        raise ValueError(
                            f"Cannot decode {source} at frame {target_frame}"
                        )
                    current_frame += 1
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError(
                        f"Cannot retrieve {source} at frame {target_frame}"
                    )
                frame_cache[target_frame] = frame.copy()

            media_cache: dict[tuple[Any, ...], Path] = {}
            for plan in plans:
                profile = plan["profile"]
                burn_in = (
                    profile.get("timestamp_mode") == "text-manifest-burn-in"
                )
                media_key = (
                    tuple(plan["target_frames"]),
                    float(profile["display_fps"]),
                    burn_in,
                    profile.get("frame_transform", "none"),
                )
                target = plan["target"]
                if media_key in media_cache:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.unlink(missing_ok=True)
                    try:
                        os.link(media_cache[media_key], target)
                    except OSError:
                        shutil.copyfile(media_cache[media_key], target)
                else:
                    _encode_media(
                        [
                            frame_cache[frame_index]
                            for frame_index in plan["target_frames"]
                        ],
                        plan["timestamps"],
                        target,
                        output_fps=float(profile["display_fps"]),
                        burn_in=burn_in,
                        frame_transform=str(
                            profile.get("frame_transform", "none")
                        ),
                    )
                    media_cache[media_key] = target
                timestamp_mode = profile.get(
                    "timestamp_mode", "text-manifest"
                )
                base_record = {
                    "video_id": clip["clip_id"],
                    "source_video_id": video_id,
                    "profile": plan["profile_name"],
                    "evaluation_profile": profile.get(
                        "evaluation_profile", plan["profile_name"]
                    ),
                    "kind": profile["kind"],
                    "clip_id": clip["clip_id"],
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
                    "source_sha256": source_digest,
                    "source_duration_seconds": round(
                        float(clip["end_seconds"])
                        - float(clip["start_seconds"]),
                        6,
                    ),
                    "clip_start_seconds": clip["start_seconds"],
                    "clip_end_seconds": clip["end_seconds"],
                    "media_path": str(target.resolve()),
                    "media_sha256": sha256_file(target),
                    "media_io_kwargs": {"video": {"num_frames": -1}},
                    "selected_frame_count": len(plan["timestamps"]),
                    "selected_timestamps_seconds": plan["timestamps"],
                    "timestamp_manifest": (
                        timestamp_manifest(plan["timestamps"])
                        if timestamp_mode != "implicit-control"
                        else None
                    ),
                    "event_peak_timestamps_seconds": plan["event_peaks"],
                    "frame_transform": profile.get(
                        "frame_transform", "none"
                    ),
                }
                for tag_group in _profile_tag_groups(profile):
                    records.append(
                        {**base_record, "tag_group": tag_group}
                    )
    finally:
        capture.release()
    return records


def _clip_motion_peaks(
    observations: list[dict[str, float]],
    *,
    start: float,
    end: float,
    nms_seconds: float,
    max_events: int,
) -> list[float]:
    candidates = [
        item
        for item in observations
        if start <= float(item["timestamp"]) < end
    ]
    peaks: list[float] = []
    for item in sorted(
        candidates,
        key=lambda value: float(value["score"]),
        reverse=True,
    ):
        timestamp = float(item["timestamp"])
        if all(abs(timestamp - peak) >= nms_seconds for peak in peaks):
            peaks.append(timestamp)
        if len(peaks) >= max_events:
            break
    return sorted(peaks)


def _profile_tag_groups(profile: dict[str, Any]) -> list[str]:
    tag_groups = profile.get("tag_groups")
    if tag_groups is None:
        tag_groups = [profile.get("tag_group", "all")]
    values = [str(value) for value in tag_groups]
    if not values or len(values) != len(set(values)):
        raise ValueError("ROAD profile tag_groups must be unique and nonempty")
    return values


def _encode_media(
    frames: list[Any],
    timestamps: list[float],
    target: Path,
    *,
    output_fps: float,
    burn_in: bool,
    frame_transform: str = "none",
) -> None:
    if not frames:
        raise ValueError("Cannot encode ROAD media without frames")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cosmos-road-") as temporary:
        frames_dir = Path(temporary)
        for index, (frame, timestamp) in enumerate(
            zip(frames, timestamps, strict=True), 1
        ):
            output_frame = _transform_frame(frame, frame_transform)
            output_frame = output_frame.copy() if burn_in else output_frame
            if burn_in:
                cv2.rectangle(
                    output_frame, (12, 12), (310, 52), (0, 0, 0), -1
                )
                cv2.putText(
                    output_frame,
                    f"source t={timestamp:.3f}s",
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
                raise OSError(f"Cannot write frame: {destination}")
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


def _transform_frame(frame: Any, transform: str) -> Any:
    if transform == "none":
        return frame
    if transform != "road-multiscale-mosaic":
        raise ValueError(f"Unknown ROAD frame transform: {transform}")
    height, width = frame.shape[:2]
    half_width = width // 2
    half_height = height // 2
    canvas = frame.copy()
    views = [
        frame,
        frame[: max(1, round(height * 0.60)), round(width * 0.18) : round(width * 0.82)],
        frame[round(height * 0.18) :, : round(width * 0.58)],
        frame[round(height * 0.18) :, round(width * 0.42) :],
    ]
    destinations = [
        (0, 0),
        (half_width, 0),
        (0, half_height),
        (half_width, half_height),
    ]
    for view, (x, y) in zip(views, destinations, strict=True):
        canvas[
            y : y + half_height,
            x : x + half_width,
        ] = cv2.resize(
            view,
            (half_width, half_height),
            interpolation=cv2.INTER_AREA,
        )
    return canvas


def _grid_exclusive(start: float, end: float, fps: float) -> list[float]:
    if fps <= 0 or end <= start:
        raise ValueError("ROAD sampling interval and fps must be positive")
    count = max(1, math.ceil((end - start) * fps))
    return [
        round(start + index / fps, 6)
        for index in range(count)
        if start + index / fps < end
    ]


def _cap_timestamps(values: list[float], maximum: int) -> list[float]:
    if maximum < 1:
        raise ValueError("ROAD max_frames must be positive")
    if len(values) <= maximum:
        return values
    if maximum == 1:
        return [values[len(values) // 2]]
    indexes = [
        round(index * (len(values) - 1) / (maximum - 1))
        for index in range(maximum)
    ]
    return [values[index] for index in indexes]


def _gt_oracle_timestamps(
    clip: dict[str, Any], profile: dict[str, Any]
) -> list[float]:
    start = float(clip["start_seconds"])
    end = float(clip["end_seconds"])
    final = end - 1 / float(clip["fps"])
    maximum = int(profile["max_frames"])
    base = _grid_exclusive(start, end, float(profile.get("base_fps", 1)))
    event_points = []
    for tag, label in clip["labels"].items():
        if label != "positive":
            continue
        for event in clip["events"].get(tag, []):
            event_start = max(start, float(event["start_seconds"]))
            event_end = min(final, float(event["end_seconds"]))
            event_points.extend(
                [
                    event_start,
                    (event_start + event_end) / 2,
                    event_end,
                ]
            )
    priority = sorted(
        {
            round(min(final, max(start, value)), 6)
            for value in [start, final, *event_points]
        }
    )
    if len(priority) >= maximum:
        return _cap_timestamps(priority, maximum)
    remaining = maximum - len(priority)
    context = [
        value
        for value in base
        if all(abs(value - chosen) > 1e-6 for chosen in priority)
    ]
    return sorted([*priority, *_cap_timestamps(context, remaining)])


def _timestamp_frame(timestamp: float, fps: float) -> int:
    return int(math.floor(timestamp * fps + 0.5))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_video_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError(f"No ROAD clip IDs in {path}")
    return values
