from __future__ import annotations

import json
from pathlib import Path

from benchmark_tool import client
from benchmark_tool.client import (
    _execute_tasks,
    _media_url,
    metric_delta,
    parse_tags,
)


def test_parse_tags_accepts_json_object() -> None:
    content = json.dumps({"tags": {"one": "yes", "two": "no"}})
    tags, error = parse_tags(content, ["one", "two"])
    assert error is None
    assert tags == {"one": "yes", "two": "no"}


def test_parse_tags_extracts_json_after_reasoning_text() -> None:
    content = 'Analysis complete.\\n{"tags":{"one":"yes"}}'
    tags, error = parse_tags(content, ["one"])
    assert error is None
    assert tags == {"one": "yes"}


def test_parse_tags_decodes_ordered_binary_array() -> None:
    tags, error = parse_tags('{"labels":[1,0,1]}', ["one", "two", "three"])
    assert error is None
    assert tags == {"one": "yes", "two": "no", "three": "yes"}


def test_parse_tags_decodes_bare_positive_index_array() -> None:
    tags, error = parse_tags(
        "[]", ["one", "two"], allow_bare_positive_indices=True
    )
    assert error is None
    assert tags == {"one": "no", "two": "no"}

    tags, error = parse_tags("[]", ["one", "two"])
    assert tags is None
    assert error == "Response has no tags object"


def test_parse_tags_accepts_common_positive_indices_typo() -> None:
    tags, error = parse_tags(
        "FINAL_POSITIVE_INDICIES: [0, 2]",
        ["one", "two", "three"],
    )
    assert error is None
    assert tags == {"one": "yes", "two": "no", "three": "yes"}


def test_media_url_does_not_require_a_local_file(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    path = media_root / "derived" / "one.mp4"

    assert _media_url(path, media_root, "/benchmark-data") == (
        "file:///benchmark-data/derived/one.mp4"
    )


def test_metric_delta_handles_new_series() -> None:
    assert metric_delta({"a": 2}, {"a": 5, "b": 7}) == {"a": 3, "b": 7}


def test_execute_tasks_writes_each_result_once(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(client, "fetch_metrics", lambda *_: {})
    monkeypatch.setattr(
        client,
        "invoke_video",
        lambda **_: {
            "success": True,
            "e2e_seconds": 1,
            "ttft_seconds": 0.5,
            "usage": {"prompt_tokens": 100},
            "parsed_tags": {"tag": "yes"},
        },
    )
    video = {
        "video_id": "one.mp4",
        "scenario": "temporal",
        "expected_positive": "tag",
        "probe": {"duration_seconds": 10},
    }
    profile_record = {
        "media_path": str(tmp_path / "one.mp4"),
        "media_io_kwargs": {"video": {"fps": 1}},
        "selected_frame_count": 10,
        "media_sha256": "abc",
    }
    output = tmp_path / "results.jsonl"

    summaries = _execute_tasks(
        endpoint="http://unused",
        experiment={
            "seed": 1,
            "model": "model",
            "temperature": 0,
            "max_tokens": 10,
        },
        labeling={},
        videos=[video],
        profile_records={("one.mp4", "uniform-1fps"): profile_record},
        profiles=["uniform-1fps"],
        output_path=output,
        repetitions=2,
        warmups=0,
        profile_lock_root=tmp_path,
        server_media_root=None,
        concurrency=2,
    )

    assert len(output.read_text().splitlines()) == 2
    assert summaries[0]["successful_requests"] == 2
    assert summaries[0]["input_video_seconds_per_wall_second"] > 0
