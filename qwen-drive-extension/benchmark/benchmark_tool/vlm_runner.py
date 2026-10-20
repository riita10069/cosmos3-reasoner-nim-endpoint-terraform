from __future__ import annotations

import json
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from .client import (
    GpuTelemetry,
    _kubernetes_metadata,
    _media_url,
    _server_metadata,
    fetch_metrics,
    invoke_video,
    metric_delta,
)
from .util import (
    append_jsonl,
    environment_metadata,
    load_json,
    resolve_media_lock_root,
    sha256_file,
    sha256_json,
    utc_now,
    write_json,
)


def run_vlm(
    experiment_path: Path,
    dataset_lock_path: Path,
    media_lock_path: Path,
    labeling_path: Path,
    run_dir: Path,
    *,
    endpoint: str,
    profiles: list[str] | None,
    limit: int | None,
    concurrency: int,
    request_order: str,
    kube_context: str | None,
    namespace: str,
    deployment: str,
    server_media_root: str | None,
    video_ids_file: Path | None = None,
) -> Path:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if request_order not in {"video-grouped-hot", "shuffled-cold"}:
        raise ValueError(f"Unknown request order: {request_order}")
    experiment = load_json(experiment_path)
    dataset = load_json(dataset_lock_path)
    media_lock = load_json(media_lock_path)
    media_lock_root = resolve_media_lock_root(media_lock, media_lock_path)
    labeling = load_json(labeling_path)
    labeling_digest = sha256_file(labeling_path)
    selected_profiles = profiles or experiment["profiles"]
    selected_video_ids = _load_video_ids(video_ids_file)
    videos = [
        video
        for video in dataset["videos"]
        if selected_video_ids is None
        or video["video_id"] in selected_video_ids
    ]
    if selected_video_ids is not None:
        available_video_ids = {
            video["video_id"] for video in dataset["videos"]
        }
        unknown_video_ids = sorted(selected_video_ids - available_video_ids)
        if unknown_video_ids:
            raise ValueError(
                f"Unknown video IDs in {video_ids_file}: {unknown_video_ids}"
            )
    videos = videos[:limit] if limit else videos
    excluded_videos = (
        []
        if limit or selected_video_ids is not None
        else [
            video["video_id"]
            for video in dataset.get("excluded_videos", [])
        ]
    )
    video_by_id = {video["video_id"]: video for video in videos}
    records = [
        record
        for record in media_lock["records"]
        if record["video_id"] in video_by_id
        and record["profile"] in selected_profiles
    ]
    if not records:
        raise ValueError("No matching VLM media records")

    endpoint = endpoint.rstrip("/")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "results.jsonl"
    completed = _completed_keys(output_path)
    repetitions = int(experiment.get("measured_repetitions", 1))
    tasks = [
        (repetition, record)
        for record in records
        for repetition in range(1, repetitions + 1)
    ]
    if request_order == "shuffled-cold":
        random.Random(int(experiment["seed"])).shuffle(tasks)
    tasks = [
        task
        for task in tasks
        if _task_key(
            experiment, task[1], task[0], labeling_digest
        )
        not in completed
    ]

    with httpx.Client(timeout=httpx.Timeout(3600, connect=30)) as client:
        server = _server_metadata(client, endpoint)
        metrics_before = fetch_metrics(client, endpoint)
    metadata = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "started_at": utc_now(),
        "endpoint": endpoint,
        "experiment": experiment,
        "experiment_sha256": sha256_file(experiment_path),
        "dataset_lock_sha256": sha256_file(dataset_lock_path),
        "media_lock_sha256": sha256_file(media_lock_path),
        "labeling_config_sha256": labeling_digest,
        "server": server,
        "environment": environment_metadata(Path(__file__).resolve().parents[2]),
        "kubernetes": _kubernetes_metadata(
            kube_context, namespace, deployment
        ),
        "execution": {
            "profiles": selected_profiles,
            "video_count": len(videos),
            "requested_video_count": len(videos) + len(excluded_videos),
            "excluded_video_count": len(excluded_videos),
            "excluded_video_ids": excluded_videos,
            "media_record_count": len(records),
            "request_count": len(tasks),
            "repetitions": repetitions,
            "concurrency": concurrency,
            "request_order": request_order,
            "server_media_root": server_media_root,
            "video_ids_file": (
                str(video_ids_file.resolve()) if video_ids_file else None
            ),
            "video_ids_file_sha256": (
                sha256_file(video_ids_file) if video_ids_file else None
            ),
        },
    }
    write_json(run_dir / "experiment-lock.json", experiment)
    write_json(run_dir / "labeling-lock.json", labeling)
    write_json(run_dir / "dataset-lock.json", dataset)
    write_json(run_dir / "media-lock.json", media_lock)
    write_json(run_dir / "run-metadata.json", metadata)

    telemetry = GpuTelemetry(
        run_dir / "gpu-telemetry.jsonl",
        context=kube_context,
        namespace=namespace,
        deployment=deployment,
    )
    started = time.perf_counter()
    telemetry.start()
    try:
        if concurrency == 1:
            for index, task in enumerate(tasks, 1):
                result = _run_task(
                    task,
                    index=index,
                    endpoint=endpoint,
                    experiment=experiment,
                    labeling=labeling,
                    video=video_by_id[task[1]["video_id"]],
                    media_lock_root=media_lock_root,
                    server_media_root=server_media_root,
                    capture_metrics=True,
                    labeling_digest=labeling_digest,
                )
                append_jsonl(output_path, result)
                _progress(result, index, len(tasks))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures: dict[Future[dict[str, Any]], int] = {}
                for index, task in enumerate(tasks, 1):
                    future = executor.submit(
                        _run_task,
                        task,
                        index=index,
                        endpoint=endpoint,
                        experiment=experiment,
                        labeling=labeling,
                        video=video_by_id[task[1]["video_id"]],
                        media_lock_root=media_lock_root,
                        server_media_root=server_media_root,
                        capture_metrics=False,
                        labeling_digest=labeling_digest,
                    )
                    futures[future] = index
                done = 0
                for future in as_completed(futures):
                    result = future.result()
                    append_jsonl(output_path, result)
                    done += 1
                    _progress(result, done, len(tasks))
    finally:
        telemetry.stop()

    wall_seconds = time.perf_counter() - started
    try:
        with httpx.Client(timeout=httpx.Timeout(3600, connect=30)) as client:
            metrics_after = fetch_metrics(client, endpoint)
    except httpx.HTTPError as error:
        metrics_after = {}
        metadata["execution"]["metrics_collection_error"] = (
            f"{type(error).__name__}: {error}"
        )
    metadata["finished_at"] = utc_now()
    metadata["execution"]["wall_seconds"] = round(wall_seconds, 6)
    metadata["execution"]["metrics_delta"] = metric_delta(
        metrics_before, metrics_after
    )
    metadata["execution"]["requests_per_second"] = (
        round(len(tasks) / wall_seconds, 6) if wall_seconds else None
    )
    write_json(run_dir / "run-metadata.json", metadata)
    return output_path


def _run_task(
    task: tuple[int, dict[str, Any]],
    *,
    index: int,
    endpoint: str,
    experiment: dict[str, Any],
    labeling: dict[str, Any],
    video: dict[str, Any],
    media_lock_root: Path,
    server_media_root: str | None,
    capture_metrics: bool,
    labeling_digest: str,
) -> dict[str, Any]:
    repetition, record = task
    groups = labeling.get("tag_groups", {})
    tags = (
        list(labeling["tag_definitions"])
        if record["tag_group"] == "all"
        else list(groups[record["tag_group"]])
    )
    common = {
        "schema_version": 2,
        "recorded_at": utc_now(),
        "task_index": index,
        "task_key": _task_key(
            experiment, record, repetition, labeling_digest
        ),
        "repetition": repetition,
        "model": experiment["model"],
        "video_id": record["video_id"],
        "scenario": video.get("scenario"),
        "expected_positive": video.get("expected_positive"),
        "profile": record.get("evaluation_profile", record["profile"]),
        "media_profile": record["profile"],
        "clip_id": record["clip_id"],
        "tag_group": record["tag_group"],
        "prompt_variant": record.get("prompt_variant", "direct"),
        "media_first_prompt": bool(
            record.get("media_first_prompt", True)
        ),
        "evaluated_tags": tags,
        "timestamp_mode": record["timestamp_mode"],
        "clip_start_seconds": record["clip_start_seconds"],
        "clip_end_seconds": record["clip_end_seconds"],
        "selected_frame_count": record["selected_frame_count"],
        "max_tokens": int(
            experiment.get("max_tokens_override")
            or record.get("max_tokens")
            or experiment["max_tokens"]
        ),
        "temperature": float(
            record.get("temperature")
            if record.get("temperature") is not None
            else experiment["temperature"]
        ),
        "top_p": record.get("top_p"),
        "selected_timestamps_seconds": record[
            "selected_timestamps_seconds"
        ],
        "media_sha256": record["media_sha256"],
    }
    try:
        prompt_variant = record.get("prompt_variant", "direct")
        result = invoke_video(
            endpoint=endpoint,
            model=experiment["model"],
            labeling=labeling,
            video_path=Path(record["media_path"]),
            media_io_kwargs=record["media_io_kwargs"],
            temperature=float(
                record.get("temperature")
                if record.get("temperature") is not None
                else experiment["temperature"]
            ),
            max_tokens=int(
                experiment.get("max_tokens_override")
                or record.get("max_tokens")
                or experiment["max_tokens"]
            ),
            media_url=_media_url(
                Path(record["media_path"]),
                media_lock_root,
                server_media_root,
            ),
            capture_metrics=capture_metrics,
            tags_override=tags,
            timestamp_context=record.get("timestamp_manifest"),
            media_first_prompt=bool(
                record.get("media_first_prompt", True)
            ),
            prompt_variant=prompt_variant,
            tag_group=record["tag_group"],
            seed=int(experiment["seed"]),
            extra_payload=experiment.get("request_extra"),
            top_p=(
                float(record["top_p"])
                if record.get("top_p") is not None
                else None
            ),
        )
        result["attempt_count"] = 1
        variant = labeling.get("prompt_variants", {}).get(
            prompt_variant, {}
        )
        repair_reason = _repair_reason(result, variant)
        repair_variant = (
            variant.get("all_positive_repair_prompt_variant")
            if repair_reason == "all_positive"
            else variant.get("repair_prompt_variant")
        )
        repair_draft = (
            result.get("content")
            if repair_reason == "parse_failure"
            and variant.get("repair_include_primary_draft")
            else None
        )
        repair_include_video = not bool(
            repair_reason == "parse_failure"
            and variant.get("repair_text_only")
        )
        if repair_reason and repair_variant:
            repair_group_names = (
                variant.get("all_positive_repair_tag_groups", [])
                if repair_reason == "all_positive"
                else []
            )
            if repair_group_names:
                repair_groups = _resolve_repair_tag_groups(
                    groups, repair_group_names, tags
                )
                repairs = []
                for repair_group, repair_tags in repair_groups:
                    repair = invoke_video(
                        endpoint=endpoint,
                        model=experiment["model"],
                        labeling=labeling,
                        video_path=Path(record["media_path"]),
                        media_io_kwargs=record["media_io_kwargs"],
                        temperature=float(
                            variant.get("repair_temperature", 0)
                        ),
                        max_tokens=int(
                            variant.get("repair_max_tokens", 64)
                        ),
                        media_url=_media_url(
                            Path(record["media_path"]),
                            media_lock_root,
                            server_media_root,
                        ),
                        capture_metrics=capture_metrics,
                        tags_override=repair_tags,
                        timestamp_context=record.get(
                            "timestamp_manifest"
                        ),
                        media_first_prompt=bool(
                            record.get("media_first_prompt", True)
                        ),
                        prompt_variant=str(repair_variant),
                        tag_group=repair_group,
                        seed=int(experiment["seed"]),
                        extra_payload=experiment.get("request_extra"),
                        top_p=None,
                        draft_context=repair_draft,
                        include_video=repair_include_video,
                    )
                    repairs.append((repair_group, repair))
                result = _merge_grouped_repair_result(
                    result,
                    repairs,
                    expected_tags=tags,
                    repair_variant=str(repair_variant),
                    repair_reason=repair_reason,
                )
            else:
                repair = invoke_video(
                    endpoint=endpoint,
                    model=experiment["model"],
                    labeling=labeling,
                    video_path=Path(record["media_path"]),
                    media_io_kwargs=record["media_io_kwargs"],
                    temperature=float(
                        variant.get("repair_temperature", 0)
                    ),
                    max_tokens=int(
                        variant.get("repair_max_tokens", 64)
                    ),
                    media_url=_media_url(
                        Path(record["media_path"]),
                        media_lock_root,
                        server_media_root,
                    ),
                    capture_metrics=capture_metrics,
                    tags_override=tags,
                    timestamp_context=record.get(
                        "timestamp_manifest"
                    ),
                    media_first_prompt=bool(
                        record.get("media_first_prompt", True)
                    ),
                    prompt_variant=str(repair_variant),
                    tag_group=record["tag_group"],
                    seed=int(experiment["seed"]),
                    extra_payload=experiment.get("request_extra"),
                    top_p=None,
                    draft_context=repair_draft,
                    include_video=repair_include_video,
                )
                result = _merge_repair_result(
                    result,
                    repair,
                    repair_variant=str(repair_variant),
                    repair_reason=repair_reason,
                )
        return {**common, **result}
    except Exception as error:
        return {
            **common,
            "success": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _merge_repair_result(
    primary: dict[str, Any],
    repair: dict[str, Any],
    *,
    repair_variant: str,
    repair_reason: str = "parse_failure",
) -> dict[str, Any]:
    primary_usage = primary.get("usage", {})
    repair_usage = repair.get("usage", {})
    metric_names = set(primary.get("metrics_delta") or {}) | set(
        repair.get("metrics_delta") or {}
    )
    return {
        **primary,
        "content": repair.get("content"),
        "parsed_tags": repair.get("parsed_tags"),
        "parse_error": repair.get("parse_error"),
        "finish_reason": repair.get("finish_reason"),
        "e2e_seconds": round(
            float(primary.get("e2e_seconds", 0))
            + float(repair.get("e2e_seconds", 0)),
            6,
        ),
        "ttft_seconds": round(
            float(primary.get("ttft_seconds") or 0)
            + float(repair.get("ttft_seconds") or 0),
            6,
        ),
        "usage": {
            key: int(primary_usage.get(key, 0))
            + int(repair_usage.get(key, 0))
            for key in {
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            }
        },
        "metrics_delta": {
            name: float((primary.get("metrics_delta") or {}).get(name, 0))
            + float((repair.get("metrics_delta") or {}).get(name, 0))
            for name in metric_names
        },
        "attempt_count": 2,
        "primary_content": primary.get("content"),
        "primary_parse_error": primary.get("parse_error"),
        "primary_finish_reason": primary.get("finish_reason"),
        "primary_usage": primary_usage,
        "primary_e2e_seconds": primary.get("e2e_seconds"),
        "repair_attempted": True,
        "repair_reason": repair_reason,
        "repair_prompt_variant": repair_variant,
        "repair_success": repair.get("parsed_tags") is not None,
        "repair_request_sha256": repair.get("request_sha256"),
        "repair_e2e_seconds": repair.get("e2e_seconds"),
    }


def _merge_grouped_repair_result(
    primary: dict[str, Any],
    repairs: list[tuple[str, dict[str, Any]]],
    *,
    expected_tags: list[str],
    repair_variant: str,
    repair_reason: str,
) -> dict[str, Any]:
    parsed_tags = {
        tag: value
        for _, repair in repairs
        for tag, value in (repair.get("parsed_tags") or {}).items()
    }
    missing_tags = [
        tag for tag in expected_tags if tag not in parsed_tags
    ]
    usage = {
        key: sum(
            int(result.get("usage", {}).get(key, 0))
            for result in [primary, *(repair for _, repair in repairs)]
        )
        for key in {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }
    }
    metric_names = {
        name
        for result in [primary, *(repair for _, repair in repairs)]
        for name in (result.get("metrics_delta") or {})
    }
    metrics_delta = {
        name: round(
            sum(
                float((result.get("metrics_delta") or {}).get(name, 0))
                for result in [
                    primary,
                    *(repair for _, repair in repairs),
                ]
            ),
            6,
        )
        for name in metric_names
    }
    repair_outputs = [
        {
            "tag_group": tag_group,
            "request_sha256": repair.get("request_sha256"),
            "content": repair.get("content"),
            "parsed_tags": repair.get("parsed_tags"),
            "parse_error": repair.get("parse_error"),
            "finish_reason": repair.get("finish_reason"),
            "e2e_seconds": repair.get("e2e_seconds"),
            "ttft_seconds": repair.get("ttft_seconds"),
            "usage": repair.get("usage"),
        }
        for tag_group, repair in repairs
    ]
    repair_finish_reasons = {
        repair.get("finish_reason") for _, repair in repairs
    }
    finish_reason = (
        "length"
        if "length" in repair_finish_reasons
        else next(iter(repair_finish_reasons))
        if len(repair_finish_reasons) == 1
        else "mixed"
    )
    return {
        **primary,
        "content": json.dumps(
            {
                tag_group: repair.get("content")
                for tag_group, repair in repairs
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "parsed_tags": parsed_tags or None,
        "parse_error": (
            None
            if not missing_tags
            else f"Grouped repair missing tags: {missing_tags}"
        ),
        "finish_reason": finish_reason,
        "e2e_seconds": round(
            float(primary.get("e2e_seconds", 0))
            + sum(
                float(repair.get("e2e_seconds", 0))
                for _, repair in repairs
            ),
            6,
        ),
        "ttft_seconds": round(
            float(primary.get("ttft_seconds") or 0)
            + sum(
                float(repair.get("ttft_seconds") or 0)
                for _, repair in repairs
            ),
            6,
        ),
        "usage": usage,
        "metrics_delta": metrics_delta,
        "attempt_count": 1 + len(repairs),
        "primary_content": primary.get("content"),
        "primary_parse_error": primary.get("parse_error"),
        "primary_finish_reason": primary.get("finish_reason"),
        "primary_usage": primary.get("usage", {}),
        "primary_e2e_seconds": primary.get("e2e_seconds"),
        "repair_attempted": True,
        "repair_reason": repair_reason,
        "repair_prompt_variant": repair_variant,
        "repair_success": not missing_tags,
        "repair_request_sha256": sha256_json(
            [
                repair.get("request_sha256")
                for _, repair in repairs
            ]
        ),
        "repair_e2e_seconds": round(
            sum(
                float(repair.get("e2e_seconds", 0))
                for _, repair in repairs
            ),
            6,
        ),
        "repair_group_count": len(repairs),
        "repair_outputs": repair_outputs,
    }


def _resolve_repair_tag_groups(
    groups: dict[str, list[str]],
    group_names: list[str],
    expected_tags: list[str],
) -> list[tuple[str, list[str]]]:
    unknown_groups = [
        name for name in group_names if name not in groups
    ]
    if unknown_groups:
        raise ValueError(
            f"Unknown all-positive repair tag groups: {unknown_groups}"
        )
    resolved = [
        (name, list(groups[name])) for name in group_names
    ]
    flattened = [tag for _, tags in resolved for tag in tags]
    if (
        len(flattened) != len(set(flattened))
        or set(flattened) != set(expected_tags)
    ):
        raise ValueError(
            "All-positive repair tag groups must partition evaluated tags"
        )
    return resolved


def _repair_reason(
    result: dict[str, Any], variant: dict[str, Any]
) -> str | None:
    if not (
        variant.get("repair_prompt_variant")
        or variant.get("all_positive_repair_prompt_variant")
    ):
        return None
    parsed_tags = result.get("parsed_tags")
    if parsed_tags is None:
        return "parse_failure"
    if (
        variant.get("repair_all_positive")
        and len(parsed_tags) > 1
        and all(value == "yes" for value in parsed_tags.values())
    ):
        return "all_positive"
    return None


def _load_video_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    video_ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not video_ids:
        raise ValueError(f"No video IDs in {path}")
    return video_ids


def _task_key(
    experiment: dict[str, Any],
    record: dict[str, Any],
    repetition: int,
    labeling_digest: str,
) -> str:
    return sha256_json(
        {
            "model": experiment["model"],
            "profile": record["profile"],
            "video_id": record["video_id"],
            "clip_id": record["clip_id"],
            "tag_group": record["tag_group"],
            "prompt_variant": record.get("prompt_variant", "direct"),
            "media_first_prompt": bool(
                record.get("media_first_prompt", True)
            ),
            "max_tokens": record.get("max_tokens"),
            "temperature": record.get("temperature"),
            "top_p": record.get("top_p"),
            "media_sha256": record["media_sha256"],
            "repetition": repetition,
            "experiment_sha256": sha256_json(experiment),
            "labeling_sha256": labeling_digest,
        }
    )


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("success") and record.get("task_key"):
                completed.add(record["task_key"])
    return completed


def _progress(record: dict[str, Any], done: int, total: int) -> None:
    status = "ok" if record.get("success") else "ERROR"
    print(
        f"[{done}/{total}] {status} {record['video_id']} "
        f"{record['profile']} {record['clip_id']}"
    )
