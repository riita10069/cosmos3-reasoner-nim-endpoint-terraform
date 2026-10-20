from __future__ import annotations

from benchmark_tool.sampler import (
    request_sampling_timestamps,
    select_adaptive_timestamps,
)


def test_uniform_sampling_includes_start_and_stays_in_duration() -> None:
    timestamps = request_sampling_timestamps(
        {"media_io_kwargs": {"video": {"fps": 2}}}, 3.2
    )
    assert timestamps[0] == 0
    assert timestamps[-1] <= 3.2
    assert timestamps == [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def test_fixed_sampling_has_exact_count_and_endpoints() -> None:
    timestamps = request_sampling_timestamps(
        {"media_io_kwargs": {"video": {"num_frames": 32}}}, 10
    )
    assert len(timestamps) == 32
    assert timestamps[0] == 0
    assert timestamps[-1] == 10


def test_adaptive_sampling_prioritizes_event_and_respects_cap() -> None:
    profile = {
        "base_fps": 1,
        "shoulder_fps": 4,
        "event_fps": 8,
        "event_seconds": 1,
        "shoulder_seconds": 3,
        "max_frames": 64,
    }
    timestamps = select_adaptive_timestamps(40, [10, 20, 30], profile)
    assert len(timestamps) == 64
    assert timestamps[0] == 0
    assert timestamps[-1] == 40
    assert 10 in timestamps
    assert 20 in timestamps
    assert 30 in timestamps
    assert 10.125 in timestamps
