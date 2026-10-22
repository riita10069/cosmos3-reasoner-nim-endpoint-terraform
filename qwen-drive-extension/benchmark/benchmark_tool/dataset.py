from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .util import load_json, run, sha256_file, utc_now, write_json


def probe_video(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration,size,format_name",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    format_data = payload["format"]
    duration = _float(stream.get("duration") or format_data.get("duration"))
    fps = _fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration_seconds": duration,
        "frame_count": _int_or_none(stream.get("nb_frames")),
        "size_bytes": int(format_data["size"]),
        "format": format_data.get("format_name"),
    }


def prepare_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    aws_profile: str | None,
    limit: int | None,
) -> Path:
    manifest = load_json(manifest_path)
    source = manifest["source"]
    videos = manifest["videos"][:limit] if limit else manifest["videos"]
    original_dir = output_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)

    locked_videos: list[dict[str, Any]] = []
    for index, video in enumerate(videos, 1):
        video_id = video["video_id"]
        key = "/".join(
            value
            for value in (source.get("prefix", "").strip("/"), video_id)
            if value
        )
        destination = original_dir / video_id
        uri = f"s3://{source['bucket']}/{key}"
        print(f"[{index}/{len(videos)}] {video_id}")
        head = _head_object(
            source["bucket"], key, source["region"], aws_profile
        )
        if not destination.exists() or destination.stat().st_size != head["ContentLength"]:
            command = ["aws"]
            if aws_profile:
                command.extend(["--profile", aws_profile])
            command.extend(
                [
                    "s3",
                    "cp",
                    uri,
                    str(destination),
                    "--region",
                    source["region"],
                    "--only-show-errors",
                ]
            )
            run(command)

        locked_videos.append(
            {
                **video,
                "s3_uri": uri,
                "s3_etag": str(head.get("ETag", "")).strip('"') or None,
                "s3_version_id": head.get("VersionId"),
                "s3_last_modified": str(head.get("LastModified")),
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
                "probe": probe_video(destination),
            }
        )

    lock = {
        "schema_version": 1,
        "name": manifest["name"],
        "source_manifest": str(manifest_path.resolve()),
        "source": source,
        "prepared_at": utc_now(),
        "videos": locked_videos,
    }
    lock_path = output_dir / "dataset-lock.json"
    write_json(lock_path, lock)
    return lock_path


def prepare_dataset_with_sdk(
    manifest_path: Path,
    output_dir: Path,
    *,
    limit: int | None,
) -> Path:
    import boto3

    manifest = load_json(manifest_path)
    source = manifest["source"]
    videos = manifest["videos"][:limit] if limit else manifest["videos"]
    original_dir = output_dir / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=source["region"])

    locked_videos: list[dict[str, Any]] = []
    excluded_videos: list[dict[str, Any]] = []
    for index, video in enumerate(videos, 1):
        video_id = video["video_id"]
        key = "/".join(
            value
            for value in (source.get("prefix", "").strip("/"), video_id)
            if value
        )
        destination = original_dir / video_id
        uri = f"s3://{source['bucket']}/{key}"
        print(f"[{index}/{len(videos)}] {video_id}", flush=True)
        head = s3.head_object(Bucket=source["bucket"], Key=key)
        expected_size = int(head["ContentLength"])
        downloaded = False
        if not destination.exists() or destination.stat().st_size != expected_size:
            _download_with_sdk(s3, source["bucket"], key, destination)
            downloaded = True

        try:
            probe = probe_video(destination)
        except (KeyError, ValueError, subprocess.CalledProcessError) as first_error:
            if not downloaded:
                _download_with_sdk(s3, source["bucket"], key, destination)
                try:
                    probe = probe_video(destination)
                except (KeyError, ValueError, subprocess.CalledProcessError) as retry_error:
                    excluded_videos.append(
                        {
                            **video,
                            "selection_index": index,
                            "s3_uri": uri,
                            "s3_etag": str(head.get("ETag", "")).strip('"') or None,
                            "s3_version_id": head.get("VersionId"),
                            "s3_last_modified": str(head.get("LastModified")),
                            "content_length": expected_size,
                            "sha256": sha256_file(destination),
                            "reason": "ffprobe_failed_after_redownload",
                            "error": _probe_error(retry_error),
                        }
                    )
                    print(
                        f"[{index}/{len(videos)}] excluded {video_id}: "
                        "ffprobe failed after re-download",
                        flush=True,
                    )
                    continue
            else:
                excluded_videos.append(
                    {
                        **video,
                        "selection_index": index,
                        "s3_uri": uri,
                        "s3_etag": str(head.get("ETag", "")).strip('"') or None,
                        "s3_version_id": head.get("VersionId"),
                        "s3_last_modified": str(head.get("LastModified")),
                        "content_length": expected_size,
                        "sha256": sha256_file(destination),
                        "reason": "ffprobe_failed_after_download",
                        "error": _probe_error(first_error),
                    }
                )
                print(
                    f"[{index}/{len(videos)}] excluded {video_id}: "
                    "ffprobe failed after download",
                    flush=True,
                )
                continue

        locked_videos.append(
            {
                **video,
                "s3_uri": uri,
                "s3_etag": str(head.get("ETag", "")).strip('"') or None,
                "s3_version_id": head.get("VersionId"),
                "s3_last_modified": str(head.get("LastModified")),
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
                "probe": probe,
            }
        )

    lock_path = output_dir / "dataset-lock.json"
    write_json(
        lock_path,
        {
            "schema_version": 1,
            "name": manifest["name"],
            "source_manifest": str(manifest_path.resolve()),
            "source": source,
            "prepared_at": utc_now(),
            "download_backend": "boto3",
            "requested_video_count": len(videos),
            "prepared_video_count": len(locked_videos),
            "excluded_video_count": len(excluded_videos),
            "excluded_videos": excluded_videos,
            "videos": locked_videos,
        },
    )
    return lock_path


def _download_with_sdk(
    s3: Any, bucket: str, key: str, destination: Path
) -> None:
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    temporary.unlink(missing_ok=True)
    try:
        s3.download_file(bucket, key, str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _probe_error(error: Exception) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        stderr = (error.stderr or "").strip()
        return stderr[-1000:] if stderr else str(error)
    return str(error)


def lock_local_dataset(
    manifest_path: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    limit: int | None,
) -> Path:
    manifest = load_json(manifest_path)
    videos = manifest["videos"][:limit] if limit else manifest["videos"]
    locked_videos = []
    for video in videos:
        path = input_dir / video["video_id"]
        if not path.is_file():
            raise FileNotFoundError(path)
        locked_videos.append(
            {
                **video,
                "s3_uri": None,
                "s3_etag": None,
                "s3_version_id": None,
                "s3_last_modified": None,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "probe": probe_video(path),
            }
        )
    lock_path = output_dir / "dataset-lock.json"
    write_json(
        lock_path,
        {
            "schema_version": 1,
            "name": manifest["name"],
            "source_manifest": str(manifest_path.resolve()),
            "source": {"type": "local"},
            "prepared_at": utc_now(),
            "videos": locked_videos,
        },
    )
    return lock_path


def _head_object(
    bucket: str, key: str, region: str, profile: str | None
) -> dict[str, Any]:
    command = ["aws"]
    if profile:
        command.extend(["--profile", profile])
    command.extend(
        [
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--region",
            region,
            "--output",
            "json",
        ]
    )
    return json.loads(run(command).stdout)


def _fraction(value: str | None) -> float | None:
    if not value:
        return None
    numerator, denominator = value.split("/", maxsplit=1)
    return float(numerator) / float(denominator) if float(denominator) else None


def _float(value: Any) -> float | None:
    return float(value) if value not in (None, "N/A") else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if value not in (None, "N/A") else None
