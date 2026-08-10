from __future__ import annotations

from pathlib import Path

from benchmark_tool.road_report import (
    _model_comparison_text,
    _runtime_row,
    _serving_sweep_profiles,
    generate_road_paper,
)
from benchmark_tool.util import write_json


def test_model_comparison_does_not_recommend_candidate_when_ci_crosses_zero(
) -> None:
    reference = {
        "benchmark_key": "baseline",
        "model": "nvidia/Cosmos3-Nano",
        "profile": "road-adaptive-24-reasoned",
        "micro": {"strict_f1": 0.689},
        "runtime": {"estimated_usd_per_1000_clips": 0.66},
    }
    candidate = {
        "benchmark_key": "candidate",
        "comparison_reference": "baseline",
        "model": "nvidia/Cosmos3-Nano",
        "profile": "road-uniform-24-reasoned",
        "micro": {"strict_f1": 0.694},
        "paired_delta_ci95": {"strict_f1": [-0.002, 0.012]},
        "runtime": {"estimated_usd_per_1000_clips": 0.69},
    }

    comparison = _model_comparison_text([reference, candidate])

    assert "95\\%信頼区間が0を跨ぐ" in comparison["discussion"]
    assert "二段構成が妥当" not in comparison["discussion"]


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
    motion_prior = tmp_path / "motion-prior.json"
    motion_metrics = {
        "precision": 0.8,
        "strict_recall": 0.8,
        "strict_f1": 0.8,
        "selective_mcc": 0.6,
    }
    write_json(
        motion_prior,
        {
            "baseline": {"metrics": profile["micro"]},
            "fused": {"metrics": motion_metrics},
            "paired_changes": {
                "improved_pairs": 2,
                "regressed_pairs": 0,
            },
            "bootstrap_ci95": {"strict_f1": [0.7, 0.9]},
            "paired_f1_delta": {
                "mean": 0.05,
                "ci95": [0.01, 0.09],
                "samples": 10000,
            },
        },
    )
    composed_motion = tmp_path / "composed-motion.json"
    write_json(
        composed_motion,
        {
            "split": "development",
            "context_tags": [
                "ego_turn_left",
                "ego_turn_right",
                "traffic_light_red",
                "traffic_light_green",
            ],
            "context_runtime": {
                "estimated_usd_per_1000_clips": 12.9,
            },
            "composed": {
                "metrics": {
                    "precision": 0.79,
                    "strict_recall": 0.79,
                    "strict_f1": 0.79,
                    "selective_mcc": 0.59,
                },
                "paired_changes_vs_motion_prior": {
                    "improved_pairs": 3,
                    "regressed_pairs": 7,
                },
            },
        },
    )
    map_context = tmp_path / "map-context.json"
    write_json(
        map_context,
        {
            "osm_snapshot": {
                "sha256": "a" * 64,
                "timestamp_osm_base": "2026-08-09T07:37:40Z",
            }
        },
    )
    prompt_development = tmp_path / "prompt-development.json"
    prompt_selected_profile = {
        **profile,
        "benchmark_key": "prompt::road-adaptive-24-hybrid-core-v1",
        "profile": "road-adaptive-24-hybrid-core-v1",
        "micro": {
            **profile["micro"],
            "precision": 0.8,
            "strict_recall": 0.8,
            "strict_f1": 0.8,
            "selective_mcc": 0.6,
        },
        "runtime": {
            **profile["runtime"],
            "prompt_tokens_mean": 8000,
            "request_e2e_seconds_p95": 20,
        },
    }
    prompt_baseline_profile = {
        **profile,
        "benchmark_key": "prompt::road-adaptive-24-reasoned",
        "profile": "road-adaptive-24-reasoned",
        "runtime": {
            **profile["runtime"],
            "prompt_tokens_mean": 7000,
            "request_e2e_seconds_p95": 15,
        },
    }
    write_json(
        prompt_development,
        {"profiles": [prompt_baseline_profile, prompt_selected_profile]},
    )
    prompt_selection = tmp_path / "prompt-selection.json"
    write_json(
        prompt_selection,
        {
            "selected": {
                "profile": "road-adaptive-24-hybrid-core-v1",
                "detailed_tags": ["ego_turn_left"],
                "prompt_criteria_words": 600,
                "metrics": {
                    "strict_f1": 0.8,
                    "precision": 0.8,
                    "strict_recall": 0.8,
                    "selective_mcc": 0.6,
                },
                "paired_f1_delta_ci95": [0.01, 0.09],
            },
            "parsimony_comparison": {
                "raw_best_minus_reference_f1_ci95": [-0.01, 0.02]
            },
            "selection_rule": {"evaluation_split_used": False},
        },
    )
    prompt_audit = tmp_path / "prompt-audit.json"
    write_json(
        prompt_audit,
        {
            "variants": [
                {
                    "variant": "road-reasoned",
                    "criteria_words": 200,
                    "tags": [{"tag": "ego_moving", "evidence_items": 0}],
                },
                {
                    "variant": "road-reasoned-hybrid-core-v1",
                    "criteria_words": 600,
                    "tags": [{"tag": "ego_turn_left", "evidence_items": 2}],
                },
            ]
        },
    )
    recall_summary = tmp_path / "recall-summary.json"
    recall_profile = {
        "profile": "road-recall-oracle-48-core",
        "original_false_negatives_recovered": 5,
        "original_false_negative_count": 14,
        "original_true_positives_retained": 7,
        "original_true_positive_count": 7,
        "negative_control_false_positives": 4,
        "negative_control_count": 7,
        "mean_selected_frames": 48,
        "mean_e2e_seconds": 20,
        "estimated_usd_per_1000_proxy": 19,
    }
    write_json(
        recall_summary,
        {
            "selection": {
                "target_tags": ["ego_lane_change"],
            },
            "profiles": [recall_profile],
        },
    )
    recall_confidence = tmp_path / "recall-confidence.json"
    write_json(
        recall_confidence,
        {
            "confidence": {
                "thresholds": [
                    {
                        "threshold": threshold,
                        "original_false_negatives_recovered": recovered,
                        "original_true_positives_retained": 7,
                        "negative_control_false_positives": fp,
                        "precision": precision,
                        "recall": recall,
                    }
                    for threshold, recovered, fp, precision, recall in [
                        (0.05, 11, 5, 0.78, 0.86),
                        (0.20, 8, 3, 0.83, 0.71),
                        (0.40, 7, 2, 0.88, 0.67),
                    ]
                ]
            }
        },
    )
    outputs = generate_road_paper(
        summary,
        summary,
        summary,
        manifest,
        audit,
        tmp_path / "paper",
        compile_pdf=False,
        motion_prior_summary_path=motion_prior,
        composed_motion_summary_path=composed_motion,
        map_context_lock_path=map_context,
        prompt_development_summary_path=prompt_development,
        prompt_selection_lock_path=prompt_selection,
        prompt_audit_path=prompt_audit,
        recall_diagnostic_summary_path=recall_summary,
        recall_confidence_baseline_path=recall_confidence,
        recall_confidence_oracle_path=recall_confidence,
    )

    tex = outputs[0].read_text(encoding="utf-8")
    assert "完全な正負GT" in tex
    assert "Ryota Yamada" in tex
    assert "Amazon Web Services, Inc." in tex
    assert "nuPlan" not in tex
    assert "旧データ" not in tex
    assert "synthetic" not in tex
    assert "非公開実走行" not in tex
    assert "公開motion context" in tex
    assert "固定VO motion prior" in tex
    assert "固定OpenStreetMap snapshot" in tex
    assert "3 pairs改善に対して7 pairs悪化" in tex
    assert "研究課題と検証仮説" in tex
    assert "提案手法" in tex
    assert "段階的実験設計と採用基準" in tex
    assert "Developmentで固定したVO motion prior規則" in tex
    assert "品質を支配した要因の階層" in tex
    assert "生成的contextと決定論的fusion" in tex
    assert "否定的結果から得られた設計知見" in tex
    assert "Hybrid Core Reasoning" in tex
    assert "候補駆動48-frame Recall refinement" in tex
    assert "未校正YES score" in tex
    assert "実験結果の証拠水準と許容する主張" in tex
    assert "negative-control FP" in tex
    assert "低Recallタグの診断結果と本番で必要となる追加証拠" in tex
    assert "Evaluationで実証済みの品質優先構成" in tex
    assert "Evaluation性能ではない" in tex
    assert (tmp_path / "paper" / "REPRODUCTION.md").is_file()
    assert (tmp_path / "paper" / "ADVERSARIAL_SELF_REVIEW.md").is_file()
    assert (tmp_path / "paper" / "road-paper-artifact-manifest.json").is_file()
    assert not (
        tmp_path / "paper" / "inputs" / "serving-evidence.json"
    ).exists()
    reproduction = (
        tmp_path / "paper" / "REPRODUCTION.md"
    ).read_text(encoding="utf-8")
    assert "inputs/evaluation-summary.json" in reproduction
    assert "benchmark_tool.road_paper_cli --paper-dir paper" in reproduction
    assert "map-context-lock.json" in reproduction
    assert "prompt-selection-lock.json" in reproduction
    assert "recall-diagnostic-summary.json" in reproduction
    assert str(tmp_path.resolve()) not in reproduction
    paper_dir = tmp_path / "paper"
    artifact_manifest = (
        paper_dir / "road-paper-artifact-manifest.json"
    ).read_text(encoding="utf-8")
    assert "ADVERSARIAL_SELF_REVIEW.md" in artifact_manifest
    assert "source_path" not in artifact_manifest
    assert str(tmp_path.resolve()) not in artifact_manifest

    rerun_outputs = generate_road_paper(
        paper_dir / "inputs" / "evaluation-summary.json",
        paper_dir / "inputs" / "development-summary.json",
        paper_dir / "inputs" / "serving-smoke-summary.json",
        paper_dir / "inputs" / "road-benchmark-manifest.json",
        paper_dir / "inputs" / "road-annotation-audit.json",
        paper_dir,
        compile_pdf=False,
        motion_prior_summary_path=(
            paper_dir / "inputs" / "motion-prior-summary.json"
        ),
        composed_motion_summary_path=(
            paper_dir / "inputs" / "composed-motion-summary.json"
        ),
        map_context_lock_path=(
            paper_dir / "inputs" / "map-context-lock.json"
        ),
        prompt_development_summary_path=(
            paper_dir / "inputs" / "prompt-development-summary.json"
        ),
        prompt_selection_lock_path=(
            paper_dir / "inputs" / "prompt-selection-lock.json"
        ),
        prompt_audit_path=paper_dir / "inputs" / "prompt-audit.json",
        recall_diagnostic_summary_path=(
            paper_dir / "inputs" / "recall-diagnostic-summary.json"
        ),
        recall_confidence_baseline_path=(
            paper_dir / "inputs" / "recall-confidence-baseline.json"
        ),
        recall_confidence_oracle_path=(
            paper_dir / "inputs" / "recall-confidence-oracle.json"
        ),
    )
    assert rerun_outputs[0].is_file()


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
