"""Complete-GT evaluation for ROAD 13-tag scene labeling runs.

Re-implements the metric definitions of the Cosmos 3 ROAD paper:
unknown GT pairs are excluded; a missing or unparsable model output is an
abstention that is neither a positive nor a negative prediction, so it lowers
strict recall and overall accuracy but not precision. Confidence intervals use
a source-video block bootstrap.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

TAGS = [
    "ego_moving", "ego_stopped", "ego_turn_left", "ego_turn_right",
    "ego_lane_change", "car_braking", "car_stopped",
    "pedestrian_waiting_to_cross", "pedestrian_crossing", "cyclist_present",
    "motorcycle_present", "traffic_light_red", "traffic_light_green",
]


def load_gt(path: Path, split: str) -> dict[str, dict]:
    rows = {}
    for line in path.open():
        row = json.loads(line)
        if row["split"] == split:
            rows[row["clip_id"]] = row
    return rows


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    """Return clip_id -> tag -> 'yes' | 'no'. Later successful rows win."""
    predictions: dict[str, dict[str, str]] = {}
    for line in path.open():
        row = json.loads(line)
        if not row.get("success"):
            continue
        clip = row.get("clip_id") or row.get("video_id")
        parsed = row.get("parsed_tags") or {}
        predictions.setdefault(clip, {}).update(
            {t: v for t, v in parsed.items() if v in ("yes", "no")}
        )
    return predictions


def counts(gt, predictions, clips=None, tags=TAGS):
    c = defaultdict(int)
    for clip in clips if clips is not None else gt:
        labels = gt[clip]["labels"]
        pred = predictions.get(clip, {})
        for tag in tags:
            truth = labels[tag]
            if truth == "unknown":
                continue
            p = pred.get(tag)
            c["known"] += 1
            if p is None:
                c["abstain"] += 1
                c["fn" if truth == "positive" else "neg_abstain"] += 1
            elif truth == "positive":
                c["tp" if p == "yes" else "fn"] += 1
            else:
                c["fp" if p == "yes" else "tn"] += 1
    return c


def metrics(c) -> dict[str, float]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    neg = fp + tn + c["neg_abstain"]
    pos = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / pos if pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / c["known"] if c["known"] else 0.0
    specificity = tn / neg if neg else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "balanced_accuracy": (recall + specificity) / 2,
        "mcc": mcc, "coverage": 1 - c["abstain"] / c["known"] if c["known"] else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "abstain": c["abstain"],
        "gt_positive": pos, "known": c["known"],
    }


def by_video(gt):
    groups = defaultdict(list)
    for clip, row in gt.items():
        groups[row["source_video_id"]].append(clip)
    return groups


def bootstrap(gt, runs: dict[str, dict], tags=TAGS, iterations=2000, seed=20260808,
              key="f1"):
    """Block bootstrap over source videos. Returns per-run CI and paired diffs."""
    groups = by_video(gt)
    videos = sorted(groups)
    rng = random.Random(seed)
    samples = defaultdict(list)
    names = list(runs)
    for _ in range(iterations):
        clips = [c for v in (rng.choice(videos) for _ in videos) for c in groups[v]]
        values = {n: metrics(counts(gt, runs[n], clips, tags))[key] for n in names}
        for n in names:
            samples[n].append(values[n])
        for a in names:
            for b in names:
                if a < b:
                    samples[(a, b)].append(values[b] - values[a])

    def ci(xs):
        xs = sorted(xs)
        return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]

    return {str(k) if not isinstance(k, tuple) else f"{k[1]} - {k[0]}": ci(v)
            for k, v in samples.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--run", action="append", required=True,
                    help="name=path/to/results.jsonl; repeatable")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    gt = load_gt(args.gt, args.split)
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        runs[name] = load_predictions(Path(path))

    summary = {"split": args.split, "clips": len(gt), "runs": {}}
    for name, pred in runs.items():
        overall = metrics(counts(gt, pred))
        per_tag = {t: metrics(counts(gt, pred, tags=[t])) for t in TAGS}
        summary["runs"][name] = {
            "clips_with_output": sum(1 for c in gt if c in pred),
            "overall": overall, "per_tag": per_tag,
        }
        print(f"{name}: P {overall['precision']:.3f} R {overall['recall']:.3f} "
              f"F1 {overall['f1']:.3f} Acc {overall['accuracy']:.3f} "
              f"BalAcc {overall['balanced_accuracy']:.3f} MCC {overall['mcc']:.3f} "
              f"cov {overall['coverage']:.4f}")
    if args.bootstrap:
        summary["bootstrap_f1"] = bootstrap(gt, runs, iterations=args.bootstrap)
        for k, v in summary["bootstrap_f1"].items():
            print(f"  F1 95% CI {k}: [{v[0]:+.3f}, {v[1]:+.3f}]")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
