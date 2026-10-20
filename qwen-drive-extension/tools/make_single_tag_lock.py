"""Derive single-criterion request records for the weak tags.

Each media record is duplicated once per weak tag with tag_group set to a
single-tag group, so the runner asks for one criterion at a time while keeping
the media, timestamp manifest, system prompt and road-reasoned contract.
"""
import json
import sys
from pathlib import Path

WEAK_TAGS = {
    # The three low-F1 tags highlighted in the Cosmos 3 article (Super F1
    # 15.0 / 28.3 / 44.9). tag: category instruction reused as
    # GROUP-SPECIFIC RULES.
    "ego_lane_change": "ego_actions",
    "car_stopped": "road_user_actions",
    "pedestrian_waiting_to_cross": "road_user_actions",
}

labeling_in, lock_in, labeling_out, lock_out = map(Path, sys.argv[1:5])
labeling = json.loads(labeling_in.read_text())
for tag, category in WEAK_TAGS.items():
    group = f"single_{tag}"
    labeling["tag_groups"][group] = [tag]
    labeling["tag_group_instructions"][group] = labeling["tag_group_instructions"][category]
labeling_out.write_text(json.dumps(labeling, indent=2) + "\n")

lock = json.loads(lock_in.read_text())
records = []
for record in lock["records"]:
    for tag in WEAK_TAGS:
        records.append({**record, "tag_group": f"single_{tag}"})
lock["records"] = records
lock["derived_from"] = str(lock_in)
lock["derivation"] = "single-tag records for weak tags: " + ", ".join(WEAK_TAGS)
lock_out.write_text(json.dumps(lock, indent=2) + "\n")
print(len(records), "records")
