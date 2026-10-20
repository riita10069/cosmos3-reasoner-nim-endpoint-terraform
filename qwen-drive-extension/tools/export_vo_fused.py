"""Export per-clip predictions after the frozen VO prior (archive implementation)
as a results.jsonl-like file so it can be merged with single-tag answers."""
import json
import sys
from pathlib import Path

from benchmark_tool.road_evaluation import load_road_ground_truth
from benchmark_tool.road_motion_prior import _apply_rules, _motion_summaries, _profile_predictions

run_dir, profile, motion_lock, gt_path, config_path, out = sys.argv[1:7]
gt, tags = load_road_ground_truth(Path(gt_path), split="evaluation")
baseline = _profile_predictions(Path(run_dir), profile, tags)
motion = _motion_summaries(Path(motion_lock))
rules = json.loads(Path(config_path).read_text())["rules"]
scope = sorted(set(gt) & set(baseline) & set(motion))
fused = _apply_rules(scope, baseline, motion, rules, tags)
with open(out, "w") as f:
    for clip in sorted(baseline):
        values = fused.get(clip, baseline[clip])
        parsed = {t: v for t, v in values.items() if v in ("yes", "no")}
        f.write(json.dumps({"success": True, "clip_id": clip, "parsed_tags": parsed}) + "\n")
print(len(scope), "clips fused of", len(baseline))
