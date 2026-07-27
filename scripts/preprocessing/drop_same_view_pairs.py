#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from check_same_view_pairs import IMAGE_SUFFIXES, classify  # noqa: E402

REMOVED_STATES = {"FRONTAL_PAIR", "LATERAL_PAIR"}


def status_of(sample_id: str, images_dir: Path) -> tuple[str, list[str]]:
    folder = images_dir / sample_id
    if not folder.is_dir():
        return "DIR_MISSING", []
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    status, _ = classify(images)
    return status, [p.name for p in images]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dataset da ripulire (contiene annotation.json e images/)")
    ap.add_argument("--out", type=Path, required=True,
                    help="dove scrivere il dataset ripulito")
    ap.add_argument("--images", choices=("copy", "symlink", "none"), default="copy",
                    help="come portare le immagini nel dataset ripulito (default: copy)")
    ap.add_argument("--drop-missing-dir", action="store_true",
                    help="rimuove anche i sample senza cartella immagini "
                         "invece di fermarsi")
    ap.add_argument("--overwrite", action="store_true",
                    help="sovrascrive --out se esiste gia'")
    ap.add_argument("--dry-run", action="store_true",
                    help="riporta soltanto, non scrive nulla")
    ap.add_argument("--show", type=int, default=30,
                    help="quanti sample rimossi elencare (0 = tutti)")
    args = ap.parse_args()

    ann_path = args.dataset / "annotation.json"
    images_dir = args.dataset / "images"
    for path in (ann_path, images_dir):
        if not path.exists():
            sys.exit(f"Percorso non trovato: {path}")

    annotation = json.loads(ann_path.read_text(encoding="utf-8"))

    kept: dict[str, list] = {}
    removed: list[tuple[str, str, str, str]] = []
    missing_dir: list[tuple[str, str]] = []
    counts: Counter = Counter()

    for split, records in annotation.items():
        kept[split] = []
        for record in records:
            sample_id = str(record["id"])
            status, files = status_of(sample_id, images_dir)
            counts[status] += 1
            if status in REMOVED_STATES:
                removed.append((split, sample_id, status, "|".join(files)))
            elif status == "DIR_MISSING":
                missing_dir.append((split, sample_id))
                if args.drop_missing_dir:
                    removed.append((split, sample_id, status, ""))
                else:
                    kept[split].append(record)
            else:
                kept[split].append(record)

    total = sum(counts.values())
    n_kept = sum(len(v) for v in kept.values())
    print(f"dataset: {args.dataset}\n")
    print(f"  {'split':<10}{'prima':>8}{'dopo':>8}{'rimossi':>10}")
    for split in annotation:
        n_removed = sum(1 for s, _, _, _ in removed if s == split)
        print(f"  {split:<10}{len(annotation[split]):>8}{len(kept[split]):>8}{n_removed:>10}")
    print(f"  {'TOTALE':<10}{total:>8}{n_kept:>8}{len(removed):>10}\n")
    print("  cartelle: " + ", ".join(f"{s}={n}" for s, n in counts.most_common()))

    if removed:
        shown = removed if not args.show else removed[: args.show]
        print(f"\n--- rimossi ({len(removed)}) ---")
        for split, sample_id, status, files in shown:
            print(f"  {sample_id:<26}{split:<8}{status:<14}{files}")
        if len(shown) < len(removed):
            print(f"  … e altri {len(removed) - len(shown)}  (--show 0 per tutti)")
    else:
        print("\nNessun sample con la sola coppia mono-vista: niente da rimuovere.")

    if missing_dir and not args.drop_missing_dir:
        print(f"\n{len(missing_dir)} sample senza cartella immagini:")
        for split, sample_id in missing_dir[:20]:
            print(f"  {sample_id:<26}{split}")
        sys.exit("\nInterrotto: non rimuovo sample che non ho potuto verificare. "
                 "Sistema le cartelle, oppure passa --drop-missing-dir.")

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
        json.dumps(kept, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "removed_same_view_pairs.txt").write_text(
        "\n".join(sample_id for _, sample_id, _, _ in removed) + "\n", encoding="utf-8"
    )

    if args.images != "none":
        kept_ids = [str(r["id"]) for records in kept.values() for r in records]
        (out / "images").mkdir()
        for sample_id in kept_ids:
            source = images_dir / sample_id
            destination = out / "images" / sample_id
            if args.images == "copy":
                shutil.copytree(source, destination)
            else:
                destination.symlink_to(source.resolve(), target_is_directory=True)
        print(f"\nimmagini: {len(kept_ids)} cartelle ({args.images})")

    for extra in ("labeling_state.json", "removed_no_impression.txt"):
        if (args.dataset / extra).is_file():
            shutil.copy2(args.dataset / extra, out / extra)

    print(f"dataset ripulito: {out}")
    print(f"id rimossi:       {out / 'removed_same_view_pairs.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
