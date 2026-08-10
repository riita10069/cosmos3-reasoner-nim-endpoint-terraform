from __future__ import annotations

import argparse
from pathlib import Path

from .road_report import generate_road_paper


INPUT_ARGUMENTS = {
    "evaluation_summary_path": "evaluation-summary.json",
    "development_summary_path": "development-summary.json",
    "smoke_summary_path": "serving-smoke-summary.json",
    "manifest_path": "road-benchmark-manifest.json",
    "audit_path": "road-annotation-audit.json",
    "error_analysis_path": "error-analysis.json",
    "sampling_audit_path": "sampling-audit.json",
    "serving_repeat_path": "serving-repeat-summary.json",
    "cascade_path": "cascade-summary.json",
    "motion_prior_summary_path": "motion-prior-summary.json",
    "composed_motion_summary_path": "composed-motion-summary.json",
    "context_fusion_summary_path": "context-fusion-summary.json",
    "opr_motion_lock_path": "opr-motion-lock.json",
    "visual_motion_lock_path": "visual-motion-lock.json",
    "map_context_lock_path": "map-context-lock.json",
    "prompt_development_summary_path": "prompt-development-summary.json",
    "prompt_selection_lock_path": "prompt-selection-lock.json",
    "prompt_audit_path": "prompt-audit.json",
    "recall_diagnostic_summary_path": "recall-diagnostic-summary.json",
    "recall_confidence_baseline_path": "recall-confidence-baseline.json",
    "recall_confidence_oracle_path": "recall-confidence-oracle.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the ROAD paper from its locked input bundle."
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper"),
        help="Directory containing inputs/ and receiving generated artifacts.",
    )
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    paper_dir = args.paper_dir.expanduser().resolve()
    input_dir = paper_dir / "inputs"
    missing = [
        name for name in INPUT_ARGUMENTS.values()
        if not (input_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing paper inputs: " + ", ".join(sorted(missing))
        )

    paths = {
        argument: input_dir / name
        for argument, name in INPUT_ARGUMENTS.items()
    }
    generated = generate_road_paper(
        paths.pop("evaluation_summary_path"),
        paths.pop("development_summary_path"),
        paths.pop("smoke_summary_path"),
        paths.pop("manifest_path"),
        paths.pop("audit_path"),
        paper_dir,
        compile_pdf=not args.no_pdf,
        **paths,
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
