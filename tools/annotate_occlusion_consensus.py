"""Generate consensus groups from two independent annotation rounds.

Run: python tools/annotate_occlusion_consensus.py <scene>
Read first_round/ and v2/ under local_record/annotation_records/<scene>/.
Write groups to dataset/per_scene/<scene>/occlusion_review/consensus/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.annotate_occlusion_round2 import (
    SPLITS, atomic_write_json, load_json, load_records, read_label_file,
    records_by_split, utc_now,
)


def read_rounds(dataset_dir: Path, record_dir: Path) -> dict:
    """Check both rounds for all splits before any output is written."""
    rounds = {}
    for split, records in records_by_split(load_records(dataset_dir)).items():
        expected = {record["ann_id"] for record in records}
        all_ids = {
            int(ann["id"])
            for ann in load_json(dataset_dir / f"{split}.coco.json")["annotations"]
        }
        first = {}
        for label in ("occluded", "not_occluded"):
            path = record_dir / "first_round" / f"{label}.{split}.json"
            if not path.is_file():
                raise SystemExit(f"first-round record not found: {path}")
            values = load_json(path)
            if not isinstance(values, list):
                raise SystemExit(f"expected an instance ID list: {path}")
            ids = {int(value) for value in values}
            if ids - all_ids:
                raise SystemExit(f"unknown instance IDs in {path}")
            # Older first-round lists also contain other annotated categories.
            first[label] = ids & expected
        overlap = first["occluded"] & first["not_occluded"]
        missing = expected - first["occluded"] - first["not_occluded"]
        if overlap or missing:
            raise SystemExit(
                f"first round/{split}: conflicting={len(overlap)}, unlabeled={len(missing)}"
            )
        labels1 = {ann_id: label for label, ids in first.items() for ann_id in ids}

        path = record_dir / "v2" / f"labels.{split}.json"
        if not path.is_file():
            raise SystemExit(f"second-round record not found: {path}")
        labels2 = read_label_file(path)
        missing, extra = expected - set(labels2), set(labels2) - expected
        if missing or extra:
            raise SystemExit(
                f"second round/{split}: unlabeled={len(missing)}, unknown={len(extra)}"
            )
        rounds[split] = (labels1, labels2)
    return rounds


def build_consensus_labels(scene: str, dataset_dir: Path, record_dir: Path) -> dict:
    rounds = read_rounds(dataset_dir, record_dir)
    output_dir = dataset_dir / "occlusion_review" / "consensus"
    summary = {
        "schema_version": 1,
        "dataset": scene,
        "generated_at": utc_now(),
        "definition": {
            "occluded": "both rounds label the instance as occluded",
            "not_occluded": "both rounds label the instance as not_occluded",
            "ambiguous": "the two rounds disagree",
        },
        "splits": {},
    }
    for split, (first, second) in rounds.items():
        groups = {label: [] for label in ("occluded", "not_occluded", "ambiguous")}
        for ann_id in sorted(first):
            label = first[ann_id] if first[ann_id] == second[ann_id] else "ambiguous"
            groups[label].append(ann_id)
        for label, ids in groups.items():
            atomic_write_json(output_dir / f"{label}.{split}.json", ids)
        summary["splits"][split] = {
            "num_instances": len(first),
            **{label: len(ids) for label, ids in groups.items()},
        }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=["pig", "frame", "stream", "mix"])
    args = parser.parse_args()
    dataset_dir = PROJECT_ROOT / "dataset" / "per_scene" / args.scene
    record_dir = PROJECT_ROOT / "local_record" / "annotation_records" / args.scene
    summary = build_consensus_labels(args.scene, dataset_dir, record_dir)
    print(f"Consensus output: {dataset_dir / 'occlusion_review' / 'consensus'}")
    for split, counts in summary["splits"].items():
        print(
            f"  {split}: occluded={counts['occluded']}, "
            f"not_occluded={counts['not_occluded']}, ambiguous={counts['ambiguous']}"
        )


if __name__ == "__main__":
    main()
