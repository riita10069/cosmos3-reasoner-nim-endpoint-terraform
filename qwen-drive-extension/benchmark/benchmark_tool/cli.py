from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .vlm_runner import run_vlm
from .road import (
    audit_road_annotations,
    build_road_ground_truth,
    select_road_smoke,
)
from .road_evaluation import evaluate_road
from .road_diagnostics import (
    analyze_road_errors,
    audit_road_sampling,
    evaluate_road_cascades,
    summarize_road_serving_repeats,
)
from .road_media import build_road_media_lock
from .road_report import generate_road_paper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROAD scene-labeling benchmark for Cosmos 3 and Qwen-Drive"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    vlm_run = subparsers.add_parser(
        "vlm-run", help="Run timestamp-aware VLM inference"
    )
    vlm_run.add_argument("--experiment", type=Path, required=True)
    vlm_run.add_argument("--dataset-lock", type=Path, required=True)
    vlm_run.add_argument("--media-lock", type=Path, required=True)
    vlm_run.add_argument("--labeling-config", type=Path, required=True)
    vlm_run.add_argument("--runs-dir", type=Path, required=True)
    vlm_run.add_argument("--run-id")
    vlm_run.add_argument("--endpoint", required=True)
    vlm_run.add_argument("--profiles")
    vlm_run.add_argument("--limit", type=int)
    vlm_run.add_argument("--concurrency", type=int, default=1)
    vlm_run.add_argument(
        "--request-order",
        choices=["video-grouped-hot", "shuffled-cold"],
        default="video-grouped-hot",
    )
    vlm_run.add_argument("--kube-context")
    vlm_run.add_argument("--namespace", default="cosmos")
    vlm_run.add_argument("--deployment", default="cosmos3-nano-vllm")
    vlm_run.add_argument("--server-media-root")
    vlm_run.add_argument(
        "--video-ids-file",
        type=Path,
        help="Run only newline-delimited video IDs from this file",
    )

    road_audit = subparsers.add_parser(
        "road-audit",
        help="Audit ROAD annotation completeness and label frequencies",
    )
    road_audit.add_argument("--annotations", type=Path, required=True)
    road_audit.add_argument("--videos-dir", type=Path, required=True)
    road_audit.add_argument("--output-dir", type=Path, required=True)

    road_gt = subparsers.add_parser(
        "road-build-gt",
        help="Build complete positive/negative/unknown ROAD clip GT",
    )
    road_gt.add_argument("--annotations", type=Path, required=True)
    road_gt.add_argument("--videos-dir", type=Path, required=True)
    road_gt.add_argument("--config", type=Path, required=True)
    road_gt.add_argument("--output-dir", type=Path, required=True)

    road_select = subparsers.add_parser(
        "road-select-smoke",
        help="Select a class-balanced, sampling-sensitive ROAD smoke set",
    )
    road_select.add_argument("--ground-truth", type=Path, required=True)
    road_select.add_argument("--output-dir", type=Path, required=True)
    road_select.add_argument(
        "--split",
        choices=["development", "evaluation"],
        default="development",
    )
    road_select.add_argument("--positives-per-tag", type=int, default=3)
    road_select.add_argument("--negatives-per-tag", type=int, default=3)
    road_select.add_argument("--max-clips", type=int, default=40)
    road_select.add_argument("--seed", type=int, default=20260808)

    road_media = subparsers.add_parser(
        "road-sample",
        help="Build timestamp-preserving ROAD evidence media",
    )
    road_media.add_argument("--ground-truth", type=Path, required=True)
    road_media.add_argument("--profiles-config", type=Path, required=True)
    road_media.add_argument("--output-dir", type=Path, required=True)
    road_media.add_argument("--profiles")
    road_media.add_argument(
        "--split", choices=["development", "evaluation"]
    )
    road_media.add_argument("--workers", type=int, default=1)
    road_media.add_argument("--limit", type=int)
    road_media.add_argument("--video-ids-file", type=Path)

    road_eval = subparsers.add_parser(
        "road-evaluate",
        help="Evaluate ROAD runs against complete positive/negative GT",
    )
    road_eval.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    road_eval.add_argument("--ground-truth", type=Path, required=True)
    road_eval.add_argument("--output-dir", type=Path, required=True)
    road_eval.add_argument(
        "--split", choices=["development", "evaluation"]
    )
    road_eval.add_argument("--bootstrap-samples", type=int, default=10000)
    road_eval.add_argument("--seed", type=int, default=20260808)
    road_eval.add_argument("--reference-profile")
    road_eval.add_argument("--reference-benchmark-key")

    road_errors = subparsers.add_parser(
        "road-diagnose-errors",
        help="Separate ROAD temporal sampling misses from covered misses",
    )
    road_errors.add_argument(
        "--run-dirs", type=Path, nargs="+", required=True
    )
    road_errors.add_argument("--ground-truth", type=Path, required=True)
    road_errors.add_argument("--output-dir", type=Path, required=True)
    road_errors.add_argument(
        "--split", choices=["development", "evaluation"]
    )

    road_sampling_audit = subparsers.add_parser(
        "road-audit-sampling",
        help="Measure ROAD event coverage for one or more media locks",
    )
    road_sampling_audit.add_argument(
        "--media-locks", type=Path, nargs="+", required=True
    )
    road_sampling_audit.add_argument(
        "--ground-truth", type=Path, required=True
    )
    road_sampling_audit.add_argument(
        "--output-dir", type=Path, required=True
    )
    road_sampling_audit.add_argument(
        "--split", choices=["development", "evaluation"]
    )

    road_serving_repeats = subparsers.add_parser(
        "road-summarize-serving-repeats",
        help="Aggregate repeated ROAD serving runs by condition",
    )
    road_serving_repeats.add_argument(
        "--summary", type=Path, required=True
    )
    road_serving_repeats.add_argument(
        "--output-dir", type=Path, required=True
    )

    road_cascade = subparsers.add_parser(
        "road-evaluate-cascades",
        help="Evaluate Nano-first selective Super routing on ROAD",
    )
    road_cascade.add_argument("--nano-run-dir", type=Path, required=True)
    road_cascade.add_argument("--super-run-dir", type=Path, required=True)
    road_cascade.add_argument("--ground-truth", type=Path, required=True)
    road_cascade.add_argument("--output-dir", type=Path, required=True)
    road_cascade.add_argument(
        "--split", choices=["development", "evaluation"]
    )
    road_cascade.add_argument(
        "--nano-usd-per-1000", type=float, required=True
    )
    road_cascade.add_argument(
        "--super-usd-per-1000", type=float, required=True
    )

    road_report = subparsers.add_parser(
        "road-report",
        help="Generate and compile the Japanese ROAD complete-GT paper",
    )
    road_report.add_argument(
        "--evaluation-summary", type=Path, required=True
    )
    road_report.add_argument(
        "--development-summary", type=Path, required=True
    )
    road_report.add_argument("--smoke-summary", type=Path, required=True)
    road_report.add_argument("--manifest", type=Path, required=True)
    road_report.add_argument("--audit", type=Path, required=True)
    road_report.add_argument("--error-analysis", type=Path)
    road_report.add_argument("--sampling-audit", type=Path)
    road_report.add_argument("--serving-repeat-summary", type=Path)
    road_report.add_argument("--cascade-summary", type=Path)
    road_report.add_argument("--output-dir", type=Path, required=True)
    road_report.add_argument("--no-pdf", action="store_true")

    args = parser.parse_args()
    if args.command == "vlm-run":
        run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        print(
            run_vlm(
                args.experiment,
                args.dataset_lock,
                args.media_lock,
                args.labeling_config,
                args.runs_dir / run_id,
                endpoint=args.endpoint,
                profiles=_csv(args.profiles),
                limit=args.limit,
                concurrency=args.concurrency,
                request_order=args.request_order,
                kube_context=args.kube_context,
                namespace=args.namespace,
                deployment=args.deployment,
                server_media_root=args.server_media_root,
                video_ids_file=args.video_ids_file,
            )
        )
    elif args.command == "road-audit":
        for path in audit_road_annotations(
            args.annotations,
            args.videos_dir,
            args.output_dir,
        ):
            print(path)
    elif args.command == "road-build-gt":
        for path in build_road_ground_truth(
            args.annotations,
            args.videos_dir,
            args.config,
            args.output_dir,
        ):
            print(path)
    elif args.command == "road-select-smoke":
        for path in select_road_smoke(
            args.ground_truth,
            args.output_dir,
            split=args.split,
            positives_per_tag=args.positives_per_tag,
            negatives_per_tag=args.negatives_per_tag,
            max_clips=args.max_clips,
            seed=args.seed,
        ):
            print(path)
    elif args.command == "road-sample":
        for path in build_road_media_lock(
            args.ground_truth,
            args.profiles_config,
            args.output_dir,
            selected_profiles=_csv(args.profiles),
            split=args.split,
            workers=args.workers,
            limit=args.limit,
            video_ids_file=args.video_ids_file,
        ):
            print(path)
    elif args.command == "road-evaluate":
        for path in evaluate_road(
            args.run_dirs,
            args.ground_truth,
            args.output_dir,
            split=args.split,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            reference_profile=args.reference_profile,
            reference_benchmark_key=args.reference_benchmark_key,
        ):
            print(path)
    elif args.command == "road-diagnose-errors":
        for path in analyze_road_errors(
            args.run_dirs,
            args.ground_truth,
            args.output_dir,
            split=args.split,
        ):
            print(path)
    elif args.command == "road-audit-sampling":
        for path in audit_road_sampling(
            args.media_locks,
            args.ground_truth,
            args.output_dir,
            split=args.split,
        ):
            print(path)
    elif args.command == "road-summarize-serving-repeats":
        for path in summarize_road_serving_repeats(
            args.summary, args.output_dir
        ):
            print(path)
    elif args.command == "road-evaluate-cascades":
        for path in evaluate_road_cascades(
            args.nano_run_dir,
            args.super_run_dir,
            args.ground_truth,
            args.output_dir,
            split=args.split,
            nano_usd_per_1000=args.nano_usd_per_1000,
            super_usd_per_1000=args.super_usd_per_1000,
        ):
            print(path)
    elif args.command == "road-report":
        for path in generate_road_paper(
            args.evaluation_summary,
            args.development_summary,
            args.smoke_summary,
            args.manifest,
            args.audit,
            args.output_dir,
            compile_pdf=not args.no_pdf,
            error_analysis_path=args.error_analysis,
            sampling_audit_path=args.sampling_audit,
            serving_repeat_path=args.serving_repeat_summary,
            cascade_path=args.cascade_summary,
        ):
            print(path)


def _csv(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


if __name__ == "__main__":
    main()
