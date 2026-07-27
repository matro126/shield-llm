#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from check_annotation_diff_views import view_of  # noqa: E402

ORDER = {"frontal": 0, "lateral": 1}


def ordered_paths(image_paths: list[str]) -> tuple[list[str], str]:
    views = [view_of(p) for p in image_paths]
    if sorted(v for v in views if v) != ["frontal", "lateral"]:
        return list(image_paths), "NOT_ORDERABLE"
    ordered = sorted(image_paths, key=lambda p: ORDER[view_of(p)])
    return ordered, "ALREADY_OK" if ordered == list(image_paths) else "REORDERED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dataset di input (annotation.json + images/)")
    ap.add_argument("--out", type=Path, required=True,
                    help="dove scrivere il dataset riordinato")
    ap.add_argument("--images", choices=("copy", "symlink", "none"), default="copy",
                    help="come portare le immagini nel dataset in uscita (default: copy)")
    ap.add_argument("--overwrite", action="store_true",
                    help="sovrascrive --out se esiste gia'")
    ap.add_argument("--dry-run", action="store_true",
                    help="riporta soltanto, non scrive nulla")
    ap.add_argument("--show", type=int, default=20,
                    help="quanti sample elencare per categoria (0 = tutti)")
    args = ap.parse_args()

    ann_path = args.dataset / "annotation.json"
    images_dir = args.dataset / "images"
    if not ann_path.is_file():
        sys.exit(f"Percorso non trovato: {ann_path}")
    if args.images != "none" and not images_dir.is_dir():
        sys.exit(f"Cartella immagini non trovata: {images_dir}")

    annotation = json.loads(ann_path.read_text(encoding="utf-8"))

    result: dict[str, list] = {}
    counts: Counter = Counter()
    reordered: list[tuple[str, str, list[str], list[str]]] = []
    not_orderable: list[tuple[str, str, list[str]]] = []

    for split, records in annotation.items():
        result[split] = []
        for record in records:
            sample_id = str(record["id"])
            paths, outcome = ordered_paths(record["image_path"])
            counts[outcome] += 1
            if outcome == "REORDERED":
                reordered.append((split, sample_id, record["image_path"], paths))
            elif outcome == "NOT_ORDERABLE":
                not_orderable.append((split, sample_id, record["image_path"]))
            result[split].append({**record, "image_path": paths})

    total = sum(counts.values())
    print(f"dataset: {args.dataset}\n")
    print(f"  sample:             {total}")
    print(f"  gia' in ordine:     {counts['ALREADY_OK']}")
    print(f"  riordinati:         {counts['REORDERED']}")
    print(f"  non riordinabili:   {counts['NOT_ORDERABLE']}")

    if reordered:
        shown = reordered if not args.show else reordered[: args.show]
        print(f"\n--- riordinati ({len(reordered)}) ---")
        for split, sample_id, before, after in shown:
            names_before = " | ".join(Path(p).name for p in before)
            names_after = " | ".join(Path(p).name for p in after)
            print(f"  {sample_id:<26}{split:<7}{names_before}  →  {names_after}")
        if len(shown) < len(reordered):
            print(f"  … e altri {len(reordered) - len(shown)}  (--show 0 per tutti)")

    if not_orderable:
        shown = not_orderable if not args.show else not_orderable[: args.show]
        print(f"\n--- NON riordinabili, lasciati invariati ({len(not_orderable)}) ---")
        for split, sample_id, paths in shown:
            print(f"  {sample_id:<26}{split:<7}"
                  f"{' | '.join(Path(p).name for p in paths)}")
        if len(shown) < len(not_orderable):
            print(f"  … e altri {len(not_orderable) - len(shown)}  (--show 0 per tutti)")
        print("  (sistemali con scripts/labeling/fix_annotation_views.py)")

    if args.dry_run:
        print("\n[dry-run] nessun file scritto.")
        return 0

    out = args.out
    if out.exists():
        if not args.overwrite:
            sys.exit(f"\n{out} esiste gia': usa --overwrite per sostituirlo.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    (out / "annotation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.images != "none":
        (out / "images").mkdir()
        for records in result.values():
            for record in records:
                sample_id = str(record["id"])
                source = images_dir / sample_id
                if not source.is_dir():
                    sys.exit(f"Cartella immagini mancante per {sample_id}: {source}")
                destination = out / "images" / sample_id
                if args.images == "copy":
                    shutil.copytree(source, destination)
                else:
                    destination.symlink_to(source.resolve(), target_is_directory=True)
        print(f"\nimmagini: {total} cartelle ({args.images})")

    for extra in ("labeling_state.json", "removed_no_impression.txt",
                  "removed_same_view_pairs.txt", "fix_views_state.json"):
        if (args.dataset / extra).is_file():
            shutil.copy2(args.dataset / extra, out / extra)

    print(f"dataset riordinato: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
