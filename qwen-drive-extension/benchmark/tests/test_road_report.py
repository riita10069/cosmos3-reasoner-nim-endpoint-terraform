from __future__ import annotations

from pathlib import Path

from benchmark_tool.road_report import (
    _runtime_row,
    _serving_sweep_profiles,
    generate_road_paper,
)
from benchmark_tool.util import write_json


def test_road_paper_uses_complete_gt_and_writes_reproduction_bundle(
    tmp_path: Path,
) -> None:
    profile = {
        "benchmark_key": "run::road-adaptive-24-reasoned",
        "run_id": "road-serving-road-only-test",
        "model": "nvidia/Cosmos3-Nano",
        "profile": "road-adaptive-24-reasoned",
        "serving_profile": "nano-server-default",
        "concurrency": 1,
        "clip_count": 4,
        "source_video_count": 2,
        "known_pairs": 4,
        "output_pair_coverage": 1.0,
        "output_validity": {"passed": True},
        "micro": {
            "precision": 0.75,
            "strict_recall": 0.75,
            "strict_f1": 0.75,
            "overall_accuracy": 0.75,
            "strict_balanced_accuracy": 0.75,
            "selective_mcc": 0.5,
        },
        "runtime": {
            "request_count": 4,
            "wall_seconds_per_clip": 1.0,
            "request_e2e_seconds_p95": 1.2,
            "prefill_seconds_per_clip": 0.4,
            "decode_seconds_per_clip": 0.3,
            "mm_cache_hit_rate": 0.5,
            "prefix_cache_token_hit_rate": 0.5,
            "preemptions": 0,
            "estimated_usd_per_1000_clips": 1.0,
        },
        "distribution": {
            "prevalence_mae": 0.1,
            "label_distribution_js_divergence": 0.02,
        },
        "rare_scene_retrieval": None,
        "bootstrap_ci95": {"strict_f1": [0.5, 1.0]},
        "tags": [
            {
                "tag": "ego_moving",
                "true_positive": 1,
                "false_negative": 1,
                "abstained_positive": 0,
                "strict_f1": 0.67,
            }
        ],
    }
    summary = tmp_path / "summary.json"
    write_json(summary, {"profiles": [profile]})
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "split_clip_counts": {"development": 2, "evaluation": 4},
            "source_video_count": 2,
            "tags": [{"id": "ego_moving"}],
        },
    )
    audit = tmp_path / "audit.json"
    write_json(
        audit,
        {"annotated_frame_count": 99, "annotation_frame_count": 100},
    )
    outputs = generate_road_paper(
        summary,
        summary,
        summary,
        manifest,
        audit,
        tmp_path / "paper",
        compile_pdf=False,
    )

    tex = outputs[0].read_text(encoding="utf-8")
    assert "完全な正負GT" in tex
    assert "Ryota Yamada" in tex
    assert "Amazon Web Services, Inc." in tex
    assert "nuPlan" not in tex
    assert "旧データ" not in tex
    assert "synthetic" not in tex
    assert "非公開実走行" not in tex
    assert (tmp_path / "paper" / "REPRODUCTION.md").is_file()
    assert (tmp_path / "paper" / "road-paper-artifact-manifest.json").is_file()
    assert not (
        tmp_path / "paper" / "inputs" / "serving-evidence.json"
    ).exists()
    reproduction = (
        tmp_path / "paper" / "REPRODUCTION.md"
    ).read_text(encoding="utf-8")
    assert "inputs/evaluation-summary.json" in reproduction
    assert str(tmp_path.resolve()) not in reproduction


def test_runtime_table_keeps_models_and_concurrency_distinct() -> None:
    runtime = {
        "request_count": 4,
        "wall_seconds_per_clip": 1.0,
        "request_e2e_seconds_p95": 1.2,
        "prefill_seconds_per_clip": 0.4,
        "decode_seconds_per_clip": 0.3,
        "mm_cache_hit_rate": 0.5,
        "prefix_cache_token_hit_rate": 0.5,
        "preemptions": 0,
        "estimated_usd_per_1000_clips": 1.0,
    }
    profiles = [
        {
            "model": model,
            "profile": "road-adaptive-24-reasoned",
            "serving_profile": serving_profile,
            "run_id": run_id,
            "concurrency": concurrency,
            "clip_count": 4,
            "runtime": runtime,
        }
        for model, serving_profile, run_id, concurrency in (
            (
                "nvidia/Cosmos3-Nano",
                "nano-server-default",
                "road-serving-road-only-c1",
                1,
            ),
            (
                "nvidia/Cosmos3-Nano",
                "nano-chunked-2048",
                "road-serving-road-only-chunk",
                4,
            ),
            (
                "nvidia/Cosmos3-Super",
                "super-weights-chunked-8192",
                "road-super-reasoned-c4-smoke-v1",
                4,
            ),
        )
    ]

    selected = _serving_sweep_profiles(profiles)

    assert len(selected) == 3
    rows = "\n".join(_runtime_row(row) for row in selected)
    assert "Cosmos3-Nano / server default (c1)" in rows
    assert "Cosmos3-Nano / Chunked Prefill 2,048 (c4)" in rows
    assert "Cosmos3-Super / TP4, Chunked Prefill 8,192 (c4)" in rows
