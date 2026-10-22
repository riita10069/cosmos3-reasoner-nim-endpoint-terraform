from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .util import load_json, run, sha256_file, utc_now, write_json


def build_profile_lock(
    dataset_lock_path: Path,
    profiles_path: Path,
    output_dir: Path,
    *,
    selected_profiles: list[str] | None,
) -> Path:
    dataset = load_json(dataset_lock_path)
    profile_config = load_json(profiles_path)["profiles"]
    profile_names = selected_profiles or list(profile_config)
    unknown = sorted(set(profile_names) - set(profile_config))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")

    records: list[dict[str, Any]] = []
    for video in dataset["videos"]:
        duration = float(video["probe"]["duration_seconds"])
        for profile_name in profile_names:
            profile = profile_config[profile_name]
            print(f"{video['video_id']}: {profile_name}")
            if profile["kind"] == "request_sampling":
                timestamps = request_sampling_timestamps(profile, duration)
                records.append(
                    {
                        "video_id": video["video_id"],
                        "profile": profile_name,
                        "kind": profile["kind"],
                        "source_path": video["path"],
                        "media_path": video["path"],
                        "media_sha256": video["sha256"],
                        "media_io_kwargs": profile["media_io_kwargs"],
                        "selected_frame_count": len(timestamps),
                        "selected_timestamps_seconds": timestamps,
                    }
                )
                continue

            derived_dir = output_dir / "derived" / profile_name
            derived_dir.mkdir(parents=True, exist_ok=True)
            target = derived_dir / video["video_id"]
            record = build_adaptive_video(Path(video["path"]), target, profile)
            records.append(
                {
                    "video_id": video["video_id"],
                    "profile": profile_name,
                    "kind": profile["kind"],
                    "source_path": video["path"],
                    "media_path": str(target.resolve()),
                    "media_sha256": sha256_file(target),
                    "media_io_kwargs": {"video": {"num_frames": -1}},
                    **record,
                }
            )

    lock_path = output_dir / "profile-lock.json"
    write_json(
        lock_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_lock": str(dataset_lock_path.resolve()),
            "profiles_config": str(profiles_path.resolve()),
            "records": records,
        },
    )
    return lock_path


def request_sampling_timestamps(
    profile: dict[str, Any], duration: float
) -> list[float]:
    video = profile["media_io_kwargs"]["video"]
    if "fps" in video:
        return _grid(0, duration, float(video["fps"]))
    count = max(1, int(video["num_frames"]))
    if count == 1:
        return [round(duration / 2, 6)]
    return [round(value, 6) for value in np.linspace(0, duration, count)]


def select_adaptive_timestamps(
    duration: float,
    event_peaks: list[float],
    profile: dict[str, Any],
) -> list[float]:
    priorities: dict[float, int] = {}

    def add_grid(start: float, end: float, fps: float, priority: int) -> None:
        for timestamp in _grid(max(0, start), min(duration, end), fps):
            priorities[timestamp] = max(priority, priorities.get(timestamp, 0))

    add_grid(0, duration, float(profile["base_fps"]), 1)
    for peak in event_peaks:
        add_grid(
            peak - float(profile["shoulder_seconds"]),
            peak + float(profile["shoulder_seconds"]),
            float(profile["shoulder_fps"]),
            2,
        )
        add_grid(
            peak - float(profile["event_seconds"]),
            peak + float(profile["event_seconds"]),
            float(profile["event_fps"]),
            3,
        )
    priorities[0.0] = 4
    priorities[round(duration, 6)] = 4

    ordered = sorted(priorities)
    maximum = int(profile["max_frames"])
    if len(ordered) <= maximum:
        return ordered

    selected: set[float] = set()
    for priority in (4, 3, 2, 1):
        candidates = [value for value in ordered if priorities[value] == priority]
        remaining = maximum - len(selected)
        if remaining <= 0:
            break
        if len(candidates) <= remaining:
            selected.update(candidates)
        else:
            indexes = np.linspace(0, len(candidates) - 1, remaining, dtype=int)
            selected.update(candidates[index] for index in indexes)
    return sorted(selected)


def detect_event_peaks(
    path: Path,
    *,
    analysis_fps: float,
    nms_seconds: float,
    max_events: int,
) -> tuple[list[float], list[dict[str, float]]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise ValueError(f"Invalid source FPS: {path}")
    frame_step = max(1, round(source_fps / analysis_fps))

    observations: list[dict[str, float]] = []
    previous_gray: np.ndarray | None = None
    previous_hist: np.ndarray | None = None
    previous_hash: np.ndarray | None = None
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step:
            frame_index += 1
            continue
        timestamp = frame_index / source_fps
        small = cv2.resize(frame, (320, 80), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([small], [0, 1], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hash_bits = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        hash_bits = hash_bits > hash_bits.mean()

        flow_score = 0.0
        scene_score = 0.0
        duplicate_score = 0.0
        if previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.1, 0
            )
            flow_score = float(np.linalg.norm(flow, axis=2).mean())
            scene_score = float(
                cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            )
            duplicate_score = float(np.not_equal(previous_hash, hash_bits).mean())
        observations.append(
            {
                "timestamp": round(timestamp, 6),
                "flow": flow_score,
                "scene": scene_score,
                "hash_change": duplicate_score,
            }
        )
        previous_gray = gray
        previous_hist = hist
        previous_hash = hash_bits
        frame_index += 1
    capture.release()

    if not observations:
        return [], []
    for field in ("flow", "scene", "hash_change"):
        values = np.array([item[field] for item in observations], dtype=float)
        low, high = np.quantile(values, [0.25, 0.95])
        scale = max(high - low, 1e-9)
        for item in observations:
            item[f"{field}_normalized"] = max(
                0.0, min(2.0, (item[field] - low) / scale)
            )
    for item in observations:
        item["score"] = (
            0.55 * item["flow_normalized"]
            + 0.30 * item["scene_normalized"]
            + 0.15 * item["hash_change_normalized"]
        )

    peaks: list[float] = []
    for item in sorted(observations, key=lambda value: value["score"], reverse=True):
        timestamp = item["timestamp"]
        if all(abs(timestamp - peak) >= nms_seconds for peak in peaks):
            peaks.append(timestamp)
        if len(peaks) >= max_events:
            break
    return sorted(peaks), observations


def build_adaptive_video(
    source: Path, target: Path, profile: dict[str, Any]
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    # Container duration is one frame past the final decodable timestamp.
    duration = max(0.0, (frame_count - 1) / source_fps)
    capture.release()

    peaks, observations = detect_event_peaks(
        source,
        analysis_fps=float(profile["analysis_fps"]),
        nms_seconds=float(profile["nms_seconds"]),
        max_events=int(profile["max_events"]),
    )
    timestamps = select_adaptive_timestamps(duration, peaks, profile)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cosmos-sampler-") as temporary:
        frames_dir = Path(temporary)
        _extract_frames(source, timestamps, frames_dir)
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(profile["event_fps"]),
                "-i",
                str(frames_dir / "frame-%06d.png"),
                "-map_metadata",
                "-1",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ]
        )
    return {
        "selected_frame_count": len(timestamps),
        "selected_timestamps_seconds": timestamps,
        "event_peak_timestamps_seconds": peaks,
        "event_score_summary": [
            {
                "timestamp": item["timestamp"],
                "score": round(item["score"], 6),
            }
            for item in sorted(
                observations, key=lambda value: value["score"], reverse=True
            )[:10]
        ],
    }


def _extract_frames(source: Path, timestamps: list[float], output_dir: Path) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {source}")
    for index, timestamp in enumerate(timestamps, 1):
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode {source} at {timestamp:.3f}s")
        destination = output_dir / f"frame-{index:06d}.png"
        if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"Cannot write frame: {destination}")
    capture.release()


def _grid(start: float, end: float, fps: float) -> list[float]:
    if end < start or fps <= 0:
        return []
    count = math.floor((end - start) * fps) + 1
    return [round(min(end, start + index / fps), 6) for index in range(count)]
