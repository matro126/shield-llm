from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPLITS = ("train", "val", "test")
MAX_SHOW = 15


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _by_id(annotation) -> dict[str, dict]:
    return {r["id"]: {**r, "_split": split}
            for split in SPLITS for r in annotation.get(split, [])}


def _basenames(record) -> list[str]:
    return [Path(p).name for p in record.get("image_path", [])]


def check(original, labeled, state, images_dir: Path) -> list[str]:
    problems: list[str] = []
    orig = _by_id(original)
    lab = _by_id(labeled)

    only_orig = set(orig) - set(lab)
    only_lab = set(lab) - set(orig)
    if only_orig:
        problems.append(f"A: {len(only_orig)} id nell'originale ma non nel labeled: {sorted(only_orig)[:5]}")
    if only_lab:
        problems.append(f"A: {len(only_lab)} id nel labeled ma non nell'originale: {sorted(only_lab)[:5]}")
    for sid in sorted(set(orig) & set(lab)):
        if orig[sid]["_split"] != lab[sid]["_split"]:
            problems.append(f"A: {sid} split diverso: {orig[sid]['_split']} → {lab[sid]['_split']}")

    done = state.get("done", {})
    for sid in sorted(set(orig) & set(lab)):
        o, la = orig[sid], lab[sid]

        if o.get("report") != la.get("report"):
            problems.append(f"B: {sid} report diverso fra originale e labeled")

        if len(o.get("image_path", [])) != len(la.get("image_path", [])):
            problems.append(
                f"C: {sid} n. immagini {len(o['image_path'])} → {len(la['image_path'])}"
            )

        if set(o) - {"_split"} != set(la) - {"_split"}:
            problems.append(f"F: {sid} campi diversi: {set(o) ^ set(la)}")

        mapping = {new: orig_name for orig_name, new, _view in done.get(sid, [])}
        lab_names = _basenames(la)
        orig_names = _basenames(o)
        if sid not in done:
            problems.append(f"D: {sid} assente da labeling_state.json")
        else:
            unmapped = [n for n in lab_names if n not in mapping]
            if unmapped:
                problems.append(f"D: {sid} nomi labeled non in labeling_state: {unmapped}")
            else:
                remapped = sorted(mapping[n] for n in lab_names)
                if remapped != sorted(orig_names):
                    problems.append(
                        f"D: {sid} coppia diversa dall'originale: "
                        f"labeled→orig {remapped} vs originale {sorted(orig_names)}"
                    )

        missing = [n for n in lab_names if not (images_dir / sid / n).is_file()]
        if missing:
            problems.append(f"E: {sid} file labeled mancanti su disco: {missing}")

    return problems


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("dataset/iu-xray/iu_xray_r2gen/annotation.json"),
    )
    parser.add_argument(
        "--labeled",
        type=Path,
        default=Path("dataset/iu-xray/iu_xray_r2gen_labeled"),
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    labeled_dir = root / args.labeled
    original = _load(root / args.original)
    labeled = _load(labeled_dir / "annotation.json")
    state = _load(labeled_dir / "labeling_state.json")

    n_orig = sum(len(original.get(s, [])) for s in SPLITS)
    n_lab = sum(len(labeled.get(s, [])) for s in SPLITS)
    print(f"studi: originale {n_orig}  labeled {n_lab}")
    print(f"per split: originale {{{', '.join(f'{s}={len(original.get(s,[]))}' for s in SPLITS)}}}  "
          f"labeled {{{', '.join(f'{s}={len(labeled.get(s,[]))}' for s in SPLITS)}}}")

    problems = check(original, labeled, state, labeled_dir / "images")
    print()
    if not problems:
        print("✓ CONFORMI: la labeled e' l'originale con le sole immagini rinominate.")
        return 0
    by_type: dict[str, int] = {}
    for p in problems:
        by_type[p[0]] = by_type.get(p[0], 0) + 1
    print(f"✗ {len(problems)} non conformita'  (per tipo: "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())) + ")\n")
    for p in problems[:MAX_SHOW]:
        print("  -", p)
    if len(problems) > MAX_SHOW:
        print(f"  … e altre {len(problems) - MAX_SHOW}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
