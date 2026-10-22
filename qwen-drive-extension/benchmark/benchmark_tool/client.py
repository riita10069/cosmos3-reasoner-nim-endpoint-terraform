from __future__ import annotations

import base64
import json
import random
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from .util import (
    append_jsonl,
    environment_metadata,
    load_json,
    run,
    sha256_file,
    sha256_json,
    utc_now,
    write_json,
)

METRIC_NAMES = {
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:mm_cache_queries_total",
    "vllm:mm_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
}


def run_benchmark(
    experiment_path: Path,
    dataset_lock_path: Path,
    profile_lock_path: Path,
    labeling_path: Path,
    run_dir: Path,
    *,
    endpoint: str,
    profiles: list[str] | None,
    limit: int | None,
    repetitions: int | None,
    warmups: int | None,
    kube_context: str | None,
    namespace: str,
    deployment: str,
    server_media_root: str | None,
    concurrency: int,
    serving_profile: str | None,
) -> Path:
    experiment = load_json(experiment_path)
    dataset = load_json(dataset_lock_path)
    profile_lock = load_json(profile_lock_path)
    labeling = load_json(labeling_path)
    configured_profiles = profiles or experiment["profiles"]
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if concurrency > 1 and len(configured_profiles) != 1:
        raise ValueError(
            "Concurrent throughput runs require exactly one frame profile"
        )
    videos = dataset["videos"][:limit] if limit else dataset["videos"]
    video_by_id = {video["video_id"]: video for video in videos}
    profile_records = {
        (record["video_id"], record["profile"]): record
        for record in profile_lock["records"]
        if record["video_id"] in video_by_id
        and record["profile"] in configured_profiles
    }
    expected = len(videos) * len(configured_profiles)
    if len(profile_records) != expected:
        raise ValueError(
            f"Profile lock has {len(profile_records)} matching records; expected {expected}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "results.jsonl"
    endpoint = endpoint.rstrip("/")
    measured_repetitions = (
        repetitions
        if repetitions is not None
        else int(experiment["measured_repetitions"])
    )
    warmup_repetitions = (
        warmups if warmups is not None else int(experiment["warmup_repetitions"])
    )
    with httpx.Client(timeout=httpx.Timeout(3600, connect=30)) as client:
        server = _server_metadata(client, endpoint)
    repo_root = Path(__file__).resolve().parents[2]
    metadata = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": utc_now(),
        "endpoint": endpoint,
        "experiment": experiment,
        "experiment_sha256": sha256_file(experiment_path),
        "dataset": {
            "name": dataset.get("name"),
            "source": dataset.get("source"),
            "video_count": len(videos),
        },
        "dataset_lock_sha256": sha256_file(dataset_lock_path),
        "profile_lock_sha256": sha256_file(profile_lock_path),
        "labeling_config_sha256": sha256_file(labeling_path),
        "server": server,
        "environment": environment_metadata(repo_root),
        "kubernetes": _kubernetes_metadata(
            kube_context, namespace, deployment
        ),
        "execution": {
            "concurrency": concurrency,
            "warmup_repetitions": warmup_repetitions,
            "measured_repetitions": measured_repetitions,
            "profiles": configured_profiles,
            "serving_profile": serving_profile,
            "media_transport": (
                "node-local-file" if server_media_root else "base64-data-url"
            ),
            "server_media_root": server_media_root,
        },
    }
    write_json(run_dir / "run-metadata.json", metadata)

    telemetry = GpuTelemetry(
        run_dir / "gpu-telemetry.jsonl",
        context=kube_context,
        namespace=namespace,
        deployment=deployment,
    )
    telemetry.start()
    try:
        phase_summaries = _execute_tasks(
            endpoint=endpoint,
            experiment=experiment,
            labeling=labeling,
            videos=videos,
            profile_records=profile_records,
            profiles=configured_profiles,
            output_path=output_path,
            repetitions=measured_repetitions,
            warmups=warmup_repetitions,
            profile_lock_root=profile_lock_path.parent.resolve(),
            server_media_root=server_media_root,
            concurrency=concurrency,
        )
    finally:
        telemetry.stop()

    metadata["execution"]["phase_summaries"] = phase_summaries
    metadata["finished_at"] = utc_now()
    write_json(run_dir / "run-metadata.json", metadata)
    return output_path


def _execute_tasks(
    *,
    endpoint: str,
    experiment: dict[str, Any],
    labeling: dict[str, Any],
    videos: list[dict[str, Any]],
    profile_records: dict[tuple[str, str], dict[str, Any]],
    profiles: list[str],
    output_path: Path,
    repetitions: int,
    warmups: int,
    profile_lock_root: Path,
    server_media_root: str | None,
    concurrency: int,
) -> list[dict[str, Any]]:
    task_groups: list[list[tuple[str, int, dict[str, Any], str]]] = []
    for phase, count in (("warmup", warmups), ("measured", repetitions)):
        tasks: list[tuple[str, int, dict[str, Any], str]] = []
        for repetition in range(count):
            for video in videos:
                for profile in profiles:
                    tasks.append((phase, repetition, video, profile))
        random.Random(int(experiment["seed"]) + len(task_groups)).shuffle(tasks)
        task_groups.append(tasks)

    total = sum(len(tasks) for tasks in task_groups)
    completed = 0
    phase_summaries: list[dict[str, Any]] = []
    for tasks in task_groups:
        if not tasks:
            continue
        phase = tasks[0][0]
        with httpx.Client(timeout=httpx.Timeout(3600, connect=30)) as client:
            metrics_before = fetch_metrics(client, endpoint)
        phase_started_at = utc_now()
        phase_started = time.perf_counter()
        phase_records: list[dict[str, Any]] = []

        if concurrency == 1:
            for task_index, task in enumerate(tasks, 1):
                record = _execute_task(
                    task,
                    task_index=task_index,
                    endpoint=endpoint,
                    experiment=experiment,
                    labeling=labeling,
                    profile_records=profile_records,
                    profile_lock_root=profile_lock_root,
                    server_media_root=server_media_root,
                    concurrency=concurrency,
                    capture_request_metrics=False,
                )
                phase_records.append(record)
                append_jsonl(output_path, record)
                completed += 1
                _print_progress(record, completed, total)
        else:
            with ThreadPoolExecutor(
                max_workers=concurrency,
                thread_name_prefix="cosmos-benchmark",
            ) as executor:
                futures: dict[Future[dict[str, Any]], int] = {}
                for task_index, task in enumerate(tasks, 1):
                    future = executor.submit(
                        _execute_task,
                        task,
                        task_index=task_index,
                        endpoint=endpoint,
                        experiment=experiment,
                        labeling=labeling,
                        profile_records=profile_records,
                        profile_lock_root=profile_lock_root,
                        server_media_root=server_media_root,
                        concurrency=concurrency,
                        capture_request_metrics=False,
                    )
                    futures[future] = task_index
                for future in as_completed(futures):
                    record = future.result()
                    phase_records.append(record)
                    append_jsonl(output_path, record)
                    completed += 1
                    _print_progress(record, completed, total)

        wall_seconds = time.perf_counter() - phase_started
        with httpx.Client(timeout=httpx.Timeout(3600, connect=30)) as client:
            metrics_after = fetch_metrics(client, endpoint)
        successful = [record for record in phase_records if record.get("success")]
        input_video_seconds = sum(
            float(record.get("source_duration_seconds") or 0)
            for record in successful
        )
        selected_frames = sum(
            int(record["selected_frame_count"]) for record in successful
        )
        prompt_tokens = sum(
            int(record.get("usage", {}).get("prompt_tokens", 0))
            for record in successful
        )
        phase_summaries.append(
            {
                "phase": phase,
                "started_at": phase_started_at,
                "finished_at": utc_now(),
                "wall_seconds": round(wall_seconds, 6),
                "requests": len(phase_records),
                "successful_requests": len(successful),
                "failed_requests": len(phase_records) - len(successful),
                "requests_per_second": round(
                    len(successful) / wall_seconds, 6
                ),
                "input_video_seconds_per_wall_second": round(
                    input_video_seconds / wall_seconds, 6
                ),
                "selected_frames_per_second": round(
                    selected_frames / wall_seconds, 6
                ),
                "prompt_tokens_per_second": round(
                    prompt_tokens / wall_seconds, 6
                ),
                "metrics_delta": metric_delta(metrics_before, metrics_after),
            }
        )
    return phase_summaries


def _execute_task(
    task: tuple[str, int, dict[str, Any], str],
    *,
    task_index: int,
    endpoint: str,
    experiment: dict[str, Any],
    labeling: dict[str, Any],
    profile_records: dict[tuple[str, str], dict[str, Any]],
    profile_lock_root: Path,
    server_media_root: str | None,
    concurrency: int,
    capture_request_metrics: bool,
) -> dict[str, Any]:
    phase, repetition, video, profile = task
    profile_record = profile_records[(video["video_id"], profile)]
    common = {
        "schema_version": 1,
        "recorded_at": utc_now(),
        "task_index": task_index,
        "phase": phase,
        "repetition": repetition + 1,
        "client_concurrency": concurrency,
        "video_id": video["video_id"],
        "scenario": video["scenario"],
        "expected_positive": video["expected_positive"],
        "source_duration_seconds": video.get("probe", {}).get(
            "duration_seconds"
        ),
        "profile": profile,
        "selected_frame_count": profile_record["selected_frame_count"],
        "media_sha256": profile_record["media_sha256"],
    }
    try:
        result = invoke_video(
            endpoint=endpoint,
            model=experiment["model"],
            labeling=labeling,
            video_path=Path(profile_record["media_path"]),
            media_io_kwargs=profile_record["media_io_kwargs"],
            temperature=float(experiment["temperature"]),
            max_tokens=int(experiment["max_tokens"]),
            media_url=_media_url(
                Path(profile_record["media_path"]),
                profile_lock_root,
                server_media_root,
            ),
            capture_metrics=capture_request_metrics,
        )
        tags = result.get("parsed_tags") or {}
        return {
            **common,
            **result,
            "expected_positive_retained": (
                tags.get(video["expected_positive"]) == "yes"
            ),
        }
    except Exception as error:
        return {
            **common,
            "error": f"{type(error).__name__}: {error}",
        }


def _print_progress(record: dict[str, Any], completed: int, total: int) -> None:
    print(
        f"[{completed}/{total}] {record['phase']} r{record['repetition']} "
        f"{record['video_id']} {record['profile']}"
    )
    if record.get("error"):
        print(f"  ERROR: {record['error']}")


def invoke_video(
    *,
    endpoint: str,
    model: str,
    labeling: dict[str, Any],
    video_path: Path,
    media_io_kwargs: dict[str, Any],
    temperature: float,
    max_tokens: int,
    media_url: str | None = None,
    capture_metrics: bool = True,
    tags_override: list[str] | None = None,
    timestamp_context: str | None = None,
    media_first_prompt: bool = False,
    prompt_variant: str | None = None,
    tag_group: str = "all",
    seed: int | None = None,
    top_p: float | None = None,
    prefix_context: str | None = None,
    draft_context: str | None = None,
    include_video: bool = True,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tags = tags_override or list(labeling["tag_definitions"])
    variant_name = prompt_variant or labeling.get(
        "default_prompt_variant", "direct"
    )
    variants = labeling.get("prompt_variants", {})
    if variants:
        if variant_name not in variants:
            raise ValueError(f"Unknown prompt variant: {variant_name}")
        variant = variants[variant_name]
        instruction = variant["instruction"]
        output_encoding = variant.get(
            "output_encoding", labeling.get("output_encoding")
        )
        structured_output = bool(variant.get("structured_output", True))
    else:
        instruction = labeling["instruction"]
        output_encoding = labeling.get("output_encoding")
        structured_output = True
    group_instruction = labeling.get("tag_group_instructions", {}).get(
        tag_group
    )
    if output_encoding in {
        "positive_index_array",
        "free_positive_indices",
    }:
        criteria = "\n".join(
            f"- [{index}] {tag}: {labeling['tag_definitions'][tag]}"
            for index, tag in enumerate(tags)
        )
    else:
        criteria = "\n".join(
            f"- {tag}: {labeling['tag_definitions'][tag]}" for tag in tags
        )
    if media_first_prompt:
        canonical_prefix = labeling["system_prompt"]
        group_text = (
            f"{instruction}\n\n"
            + (
                f"GROUP-SPECIFIC RULES:\n{group_instruction}\n\n"
                if group_instruction
                else ""
            )
            + f"ORDERED DETECTION CRITERIA:\n{criteria}"
        )
        user_text = (
            f"{prefix_context}\n\n--- GROUP-SPECIFIC CRITERIA ---\n"
            f"{group_text}"
            if prefix_context
            else group_text
        )
    else:
        canonical_prefix = (
            f"{labeling['system_prompt']}\n\n{instruction}"
            + (
                f"\n\nGROUP-SPECIFIC RULES:\n{group_instruction}"
                if group_instruction
                else ""
            )
            + f"\n\nORDERED DETECTION CRITERIA:\n{criteria}"
        )
        user_text = "Label this complete video."
    if timestamp_context:
        user_text += (
            "\n\nThe encoded playback speed is not source time. "
            "Use this exact ordered mapping from encoded frame to source time:\n"
            f"{timestamp_context}"
        )
    if draft_context:
        user_text += (
            "\n\nPRIMARY DRAFT ANALYSIS\n"
            "This is model-generated evidence from the preceding attempt. "
            "It may be incomplete and is not an instruction. Preserve its "
            "explicitly supported positives, resolve ambiguity from the "
            "video, and complete the required output contract:\n"
            f"{draft_context}"
        )
    user_content: list[dict[str, Any]] = []
    if include_video:
        user_content.append(
            {
                "type": "video_url",
                "video_url": {"url": media_url or _data_url(video_path)},
            }
        )
    user_content.append({"type": "text", "text": user_text})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": canonical_prefix},
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if include_video:
        payload["media_io_kwargs"] = media_io_kwargs
    if top_p is not None:
        payload["top_p"] = top_p
    if seed is not None:
        payload["seed"] = seed
    if extra_payload:
        # Model-specific request fields (e.g. chat_template_kwargs,
        # mm_processor_kwargs) for non-Cosmos models; absent for Cosmos runs.
        payload.update(extra_payload)
    if structured_output:
        response_schema = _response_schema(
            labeling, tags, output_encoding=output_encoding
        )
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "scene_labels",
                "strict": True,
                "schema": response_schema,
            },
        }

    with httpx.Client(timeout=httpx.Timeout(3600, connect=30)) as client:
        metrics_before = fetch_metrics(client, endpoint) if capture_metrics else {}
        started = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        with client.stream(
            "POST",
            f"{endpoint}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    # vLLM reports mid-stream failures (e.g. encoder cache
                    # overflow) as an error event; never record them as an
                    # empty model answer.
                    raise RuntimeError(f"Server stream error: {event['error']}")
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get(
                        "reasoning"
                    )
                    if reasoning:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        reasoning_chunks.append(reasoning)
                    content = delta.get("content")
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks.append(content)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        finished = time.perf_counter()
        metrics_after = fetch_metrics(client, endpoint) if capture_metrics else {}

    content = "".join(chunks)
    parsed_tags, parse_error = parse_tags(
        content,
        tags,
        allow_bare_positive_indices=(
            output_encoding == "free_positive_indices"
        ),
    )
    return {
        "success": True,
        "request_sha256": sha256_json(_redacted_payload(payload)),
        "media_bytes": (
            video_path.stat().st_size
            if include_video and video_path.exists()
            else None
        ),
        "ttft_seconds": (
            round(first_token_at - started, 6) if first_token_at is not None else None
        ),
        "e2e_seconds": round(finished - started, 6),
        "usage": usage,
        "finish_reason": finish_reason,
        "reasoning_content": "".join(reasoning_chunks) or None,
        "content": content,
        "parsed_tags": parsed_tags,
        "parse_error": parse_error,
        "metrics_delta": (
            metric_delta(metrics_before, metrics_after)
            if capture_metrics
            else None
        ),
    }


def parse_tags(
    content: str,
    expected_tags: list[str],
    *,
    allow_bare_positive_indices: bool = False,
) -> tuple[dict[str, str] | None, str | None]:
    marker = re.search(
        r"FINAL_POSITIVE_IND(?:ICES|ICIES)\s*:\s*(\[[^\]]*\])",
        content,
        flags=re.IGNORECASE,
    )
    if marker:
        try:
            indices = json.loads(marker.group(1))
        except json.JSONDecodeError as error:
            return None, f"Invalid final positive indices: {error}"
        return _parse_positive_indices(indices, expected_tags)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None, "No JSON object in response"
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            return None, f"Invalid JSON: {error}"
    if isinstance(parsed, list) and allow_bare_positive_indices:
        return _parse_positive_indices(parsed, expected_tags)
    if isinstance(parsed, dict) and isinstance(parsed.get("labels"), list):
        labels = parsed["labels"]
        if (
            len(labels) != len(expected_tags)
            or any(value not in (0, 1) for value in labels)
        ):
            return None, (
                f"Expected {len(expected_tags)} binary labels; received {labels}"
            )
        return {
            tag: "yes" if value == 1 else "no"
            for tag, value in zip(expected_tags, labels, strict=True)
        }, None
    if isinstance(parsed, dict) and isinstance(
        parsed.get("positive_indices"), list
    ):
        return _parse_positive_indices(
            parsed["positive_indices"], expected_tags
        )
    tags = parsed.get("tags") if isinstance(parsed, dict) else None
    if not isinstance(tags, dict):
        return None, "Response has no tags object"
    invalid = [tag for tag in expected_tags if tags.get(tag) not in ("yes", "no")]
    if invalid:
        return None, f"Missing or invalid tags: {invalid}"
    return {tag: tags[tag] for tag in expected_tags}, None


def _parse_positive_indices(
    indices: Any, expected_tags: list[str]
) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(indices, list) or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= len(expected_tags)
        for value in indices
    ):
        return None, f"Invalid positive tag indices: {indices}"
    selected = set(indices)
    return {
        tag: "yes" if index in selected else "no"
        for index, tag in enumerate(expected_tags)
    }, None


def _response_schema(
    labeling: dict[str, Any],
    tags: list[str],
    *,
    output_encoding: str | None = None,
) -> dict[str, Any]:
    encoding = output_encoding or labeling.get("output_encoding")
    if encoding == "ordered_binary_array":
        return {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {"type": "integer", "enum": [0, 1]},
                    "minItems": len(tags),
                    "maxItems": len(tags),
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        }
    if encoding == "positive_index_array":
        return {
            "type": "object",
            "properties": {
                "positive_indices": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "enum": list(range(len(tags))),
                    },
                    "maxItems": len(tags),
                }
            },
            "required": ["positive_indices"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "tags": {
                "type": "object",
                "properties": {
                    tag: {"type": "string", "enum": ["yes", "no"]} for tag in tags
                },
                "required": tags,
                "additionalProperties": False,
            }
        },
        "required": ["tags"],
        "additionalProperties": False,
    }


def fetch_metrics(client: httpx.Client, endpoint: str) -> dict[str, float]:
    response = client.get(f"{endpoint}/metrics")
    response.raise_for_status()
    totals: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in METRIC_NAMES:
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def metric_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    return {
        key: round(after.get(key, 0.0) - before.get(key, 0.0), 6)
        for key in sorted(set(before) | set(after))
    }


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def _media_url(
    path: Path, local_root: Path, server_media_root: str | None
) -> str | None:
    if not server_media_root:
        return None
    relative = path.resolve().relative_to(local_root)
    server_path = Path(server_media_root) / relative
    return server_path.as_uri()


def _redacted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(payload, ensure_ascii=False))
    for item in redacted["messages"][1]["content"]:
        video_url = item.get("video_url")
        if not video_url:
            continue
        media_url = video_url["url"]
        if media_url.startswith("data:"):
            video_url["url"] = "data:video/mp4;base64,<redacted>"
    return redacted


def _server_metadata(client: httpx.Client, endpoint: str) -> dict[str, Any]:
    version = client.get(f"{endpoint}/version")
    version.raise_for_status()
    models = client.get(f"{endpoint}/v1/models")
    models.raise_for_status()
    return {"version": version.json(), "models": models.json()}


def _kubernetes_metadata(
    context: str | None, namespace: str, deployment: str
) -> dict[str, Any] | None:
    if not context:
        return None
    base = ["kubectl", "--context", context, "-n", namespace]
    deployment_result = run(
        base + ["get", "deployment", deployment, "-o", "json"], check=False
    )
    pods_result = run(
        base
        + [
            "get",
            "pods",
            "-l",
            f"app={deployment}",
            "-o",
            "json",
        ],
        check=False,
    )
    nodes_result = run(
        ["kubectl", "--context", context, "get", "nodes", "-o", "json"],
        check=False,
    )
    if (
        deployment_result.returncode
        or pods_result.returncode
        or nodes_result.returncode
    ):
        return {
            "context": context,
            "namespace": namespace,
            "deployment": deployment,
            "error": (
                deployment_result.stderr
                or pods_result.stderr
                or nodes_result.stderr
            ),
        }
    deployment_json = json.loads(deployment_result.stdout)
    pods_json = json.loads(pods_result.stdout)
    nodes_json = json.loads(nodes_result.stdout)
    running_pods = [
        pod
        for pod in pods_json["items"]
        if pod.get("status", {}).get("phase") == "Running"
    ]
    selected_node_names = {
        pod["spec"].get("nodeName")
        for pod in running_pods
        if pod["spec"].get("nodeName")
    }
    return {
        "context": context,
        "namespace": namespace,
        "deployment": deployment,
        "deployment_generation": deployment_json["metadata"].get("generation"),
        "command": deployment_json["spec"]["template"]["spec"]["containers"][0].get(
            "command"
        ),
        "image": deployment_json["spec"]["template"]["spec"]["containers"][0]["image"],
        "pod_template_labels": deployment_json["spec"]["template"]["metadata"].get(
            "labels", {}
        ),
        "pod_template_annotations": deployment_json["spec"]["template"][
            "metadata"
        ].get("annotations", {}),
        "resources": deployment_json["spec"]["template"]["spec"]["containers"][
            0
        ].get("resources"),
        "pods": [
            {
                "name": pod["metadata"]["name"],
                "node": pod["spec"].get("nodeName"),
                "image_id": (
                    pod.get("status", {})
                    .get("containerStatuses", [{}])[0]
                    .get("imageID")
                ),
            }
            for pod in running_pods
        ],
        "nodes": [
            {
                "name": node["metadata"]["name"],
                "instance_type": node["metadata"]
                .get("labels", {})
                .get("node.kubernetes.io/instance-type"),
                "availability_zone": node["metadata"]
                .get("labels", {})
                .get("topology.kubernetes.io/zone"),
                "provider_id": node["spec"].get("providerID"),
                "allocatable_gpu": node["status"]
                .get("allocatable", {})
                .get("nvidia.com/gpu"),
            }
            for node in nodes_json["items"]
            if node["metadata"]["name"] in selected_node_names
        ],
    }


class GpuTelemetry:
    def __init__(
        self,
        output: Path,
        *,
        context: str | None,
        namespace: str,
        deployment: str,
    ) -> None:
        self.output = output
        self.context = context
        self.namespace = namespace
        self.deployment = deployment
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.context:
            return
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _collect(self) -> None:
        while not self._stop.is_set():
            command = [
                "kubectl",
                "--context",
                self.context or "",
                "-n",
                self.namespace,
                "exec",
                f"deployment/{self.deployment}",
                "--",
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
            result = run(command, check=False)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) != 7:
                        continue
                    append_jsonl(
                        self.output,
                        {
                            "recorded_at": utc_now(),
                            "gpu_timestamp": fields[0],
                            "gpu_index": int(fields[1]),
                            "gpu_utilization_percent": float(fields[2]),
                            "memory_used_mib": float(fields[3]),
                            "memory_total_mib": float(fields[4]),
                            "power_watts": float(fields[5]),
                            "temperature_c": float(fields[6]),
                        },
                    )
            self._stop.wait(1.0)
