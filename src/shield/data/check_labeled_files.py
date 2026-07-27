from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SPLITS = ("train", "val", "test")
MAX_SHOW = 20
_NAME_RE = re.compile(r"^(frontal|lateral)(_\d+)?\.png$")


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument(
        "--labeled", type=Path, default=Path("dataset/iu-xray/iu_xray_r2gen_labeled")
    )
    args = parser.parse_args(argv)

    base = args.root / args.labeled
    images_dir = base / "images"
    annotation = json.loads((base / "annotation.json").read_text(encoding="utf-8"))
    state = json.loads((base / "labeling_state.json").read_text(encoding="utf-8"))
    done = state.get("done", {})

    ann_by_id = {r["id"]: r for split in SPLITS for r in annotation.get(split, [])}
    disk_dirs = {p.name for p in images_dir.iterdir() if p.is_dir()}

    problems: list[str] = []
    n_ann_imgs = n_disk_imgs = 0

    for sid in sorted(set(ann_by_id) - disk_dirs):
        problems.append(f"5: {sid} in annotation ma senza cartella su disco")
    for sid in sorted(disk_dirs - set(ann_by_id)):
        problems.append(f"5: cartella {sid} senza record in annotation")

    for sid in sorted(set(ann_by_id) & disk_dirs):
        folder = images_dir / sid
        on_disk = sorted(p.name for p in folder.glob("*.png"))
        n_disk_imgs += len(on_disk)
        ann_names = [Path(p).name for p in ann_by_id[sid].get("image_path", [])]
        n_ann_imgs += len(ann_names)
        state_new = {new for _orig, new, _view in done.get(sid, [])}

        for name in on_disk:
            if not _NAME_RE.match(name):
                problems.append(f"6: {sid}/{name} non segue frontal[_N]/lateral[_N].png")

        for name in ann_names:
            if not (folder / name).is_file():
                problems.append(f"1: {sid} annotation punta a {name} assente su disco")

        for name in state_new:
            if not (folder / name).is_file():
                problems.append(f"2: {sid} labeling_state punta a {name} assente su disco")

        for name in on_disk:
            if name not in state_new:
                problems.append(f"3: {sid}/{name} su disco ma non in labeling_state")

        for name in ann_names:
            if name not in state_new:
                problems.append(f"4: {sid} annotation usa {name}, assente da labeling_state")

    print(f"studi: {len(ann_by_id)} in annotation, {len(disk_dirs)} cartelle su disco")
    print(f"immagini: {n_ann_imgs} referenziate in annotation, {n_disk_imgs} file su disco")
    print()
    if not problems:
        print("✓ TUTTO SI TROVA: annotation, labeling_state e file su disco coerenti.")
        return 0
    by_type: dict[str, int] = {}
    for p in problems:
        by_type[p[0]] = by_type.get(p[0], 0) + 1
    print(f"✗ {len(problems)} problemi  (per tipo: "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())) + ")\n")
    for p in problems[:MAX_SHOW]:
        print("  -", p)
    if len(problems) > MAX_SHOW:
        print(f"  … e altri {len(problems) - MAX_SHOW}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
