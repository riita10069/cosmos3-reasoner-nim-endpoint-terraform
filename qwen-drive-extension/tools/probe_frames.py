"""Probe how many video frames vLLM feeds Qwen-Drive for one ROAD clip.

Sends the exact benchmark request for one clip with several
mm_processor_kwargs settings and prints prompt token counts.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "work/qwen-drive-road/bench/benchmark"))
from benchmark_tool.client import invoke_video  # noqa: E402

endpoint, lock_path = sys.argv[1], Path(sys.argv[2])
labeling = json.load(open(Path.home() / "work/qwen-drive-road/bench/benchmark/configs/road-labeling.json"))
record = json.load(open(lock_path))["records"][0]
variants = {
    "default": {},
    "fps1": {"mm_processor_kwargs": {"fps": 1}},
    "fps4": {"mm_processor_kwargs": {"fps": 4}},
    "nf24": {"mm_processor_kwargs": {"num_frames": 24}},
    "no_sample": {"mm_processor_kwargs": {"do_sample_frames": False}},
    "text_only": None,
}
for name, extra in variants.items():
    result = invoke_video(
        endpoint=endpoint, model="Qwen/Qwen-Drive-1.0-4B", labeling=labeling,
        video_path=Path(record["media_path"]), media_io_kwargs=record["media_io_kwargs"],
        temperature=0.0, max_tokens=64, capture_metrics=False,
        timestamp_context=record["timestamp_manifest"], media_first_prompt=True,
        prompt_variant="road-reasoned", tag_group="all", seed=20260808,
        include_video=extra is not None,
        extra_payload={"chat_template_kwargs": {"enable_thinking": False}, **(extra or {})},
    )
    print(name, result["usage"], repr(result["content"])[:200], result.get("parse_error"))
