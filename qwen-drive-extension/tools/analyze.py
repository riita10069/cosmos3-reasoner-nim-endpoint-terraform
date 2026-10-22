"""Weak-tag analysis for the Qwen-Drive ROAD extension.

Outputs per-tag metrics for every run, paired source-video block-bootstrap CIs
of per-tag F1 differences against the Cosmos references, predicted-positive
counts, and repair statistics. Single-tag runs are merged over a base run so
only the weak tags change.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import evaluate as ev

WEAK_TAGS = [
    "ego_turn_left", "ego_lane_change", "car_stopped",
    "pedestrian_waiting_to_cross", "pedestrian_crossing",
]


def merged(base: dict, single_path: Path, fallback: bool) -> dict:
    """Replace each single-tag pair with the single-criterion answer.

    An unparsable single-tag answer is an abstention (strict) unless
    fallback is set, in which case the 13-tag answer is kept.
    """
    out = {clip: dict(values) for clip, values in base.items()}
    for line in single_path.open():
        row = json.loads(line)
        if not row.get("success"):
            continue
        clip = row["clip_id"]
        for tag in row["evaluated_tags"]:
            value = (row.get("parsed_tags") or {}).get(tag)
            if value in ("yes", "no"):
                out.setdefault(clip, {})[tag] = value
            elif not fallback:
                out.setdefault(clip, {}).pop(tag, None)
    return out


def tag_bootstrap(gt, runs, pairs, tags, iterations, seed=20260808):
    groups = ev.by_video(gt)
    videos = sorted(groups)
    rng = random.Random(seed)
    deltas = defaultdict(list)
    for _ in range(iterations):
        clips = [c for v in (rng.choice(videos) for _ in videos) for c in groups[v]]
        for tag in tags + ["__all__"]:
            selected = ev.TAGS if tag == "__all__" else [tag]
            for a, b in pairs:
                fa = ev.metrics(ev.counts(gt, runs[a], clips, selected))["f1"]
                fb = ev.metrics(ev.counts(gt, runs[b], clips, selected))["f1"]
                deltas[(tag, a, b)].append(fb - fa)
    result = {}
    for (tag, a, b), xs in deltas.items():
        xs.sort()
        result[f"{tag}|{b} - {a}"] = (
            xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]
        )
    return result


def repair_stats(path: Path) -> dict:
    rows = [json.loads(line) for line in path.open()]
    repaired = [r for r in rows if r.get("attempt_count", 1) > 1]
    return {
        "requests": len(rows),
        "repaired": len(repaired),
        "repaired_clips": sorted({r["clip_id"] for r in repaired}),
        "completion_tokens_mean": sum(
            (r.get("usage") or {}).get("completion_tokens", 0) for r in rows
        ) / max(len(rows), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True,
                    help="name=results.jsonl or name=base_name+single.jsonl")
    ap.add_argument("--pair", action="append", default=[],
                    help="reference_name:candidate_name")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gt = ev.load_gt(args.gt, "evaluation")
    runs, repairs = {}, {}
    for spec in args.run:
        name, source = spec.split("=", 1)
        if "+" in source:
            base, single = source.split("+", 1)
            fallback = single.endswith("@fallback")
            single = single.removesuffix("@fallback")
            runs[name] = merged(runs[base], Path(single), fallback)
            repairs[name] = repair_stats(Path(single))
        else:
            runs[name] = ev.load_predictions(Path(source))
            repairs[name] = repair_stats(Path(source))

    report = {"runs": {}, "repairs": {}, "bootstrap": {}}
    for name, pred in runs.items():
        overall = ev.metrics(ev.counts(gt, pred))
        per_tag = {}
        for tag in ev.TAGS:
            m = ev.metrics(ev.counts(gt, pred, tags=[tag]))
            m["predicted_positive"] = m["tp"] + m["fp"]
            per_tag[tag] = m
        weak = ev.metrics(ev.counts(gt, pred, tags=WEAK_TAGS))
        report["runs"][name] = {"overall": overall, "weak_micro": weak, "per_tag": per_tag}
        stats = dict(repairs[name])
        clips = stats.pop("repaired_clips")
        if clips:
            in_rep = [c for c in gt if c in set(clips)]
            out_rep = [c for c in gt if c not in set(clips)]
            stats["f1_repaired_clips"] = ev.metrics(ev.counts(gt, pred, in_rep))["f1"]
            stats["f1_other_clips"] = ev.metrics(ev.counts(gt, pred, out_rep))["f1"]
        report["repairs"][name] = stats

    pairs = [tuple(p.split(":", 1)) for p in args.pair]
    if pairs and args.bootstrap:
        report["bootstrap"] = tag_bootstrap(gt, runs, pairs, WEAK_TAGS, args.bootstrap)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    for name, r in report["runs"].items():
        o, w = r["overall"], r["weak_micro"]
        print(f"{name:34s} F1 {o['f1']:.3f} MCC {o['mcc']:.3f} | weak5 P {w['precision']:.3f} "
              f"R {w['recall']:.3f} F1 {w['f1']:.3f} | repaired {report['repairs'][name]['repaired']}")
    for key, (lo, hi) in report["bootstrap"].items():
        print(f"  {key}: [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
