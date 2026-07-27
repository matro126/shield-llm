#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from check_missing_impression_r2gen_to_original import (  # noqa: E402
    classify,
    extract_uid,
    parse_sections,
)

DEFAULT_REPORTS = ROOT / "dataset" / "iu-xray" / "iu_xray_original" / "reports"

MISSING_STATES = {"MISSING_TAG", "EMPTY", "PLACEHOLDER"}
STRICT_STATES = {"MISSING_TAG", "EMPTY"}
UNVERIFIABLE = {"XML_MISSING", "BAD_ID"}


def status_of(sample_id: str, reports_dir: Path, cache: dict) -> tuple[str, str]:
    uid = extract_uid(sample_id)
    if uid is None:
        return "BAD_ID", ""
    xml_file = reports_dir / f"{uid}.xml"
    if not xml_file.is_file():
        return "XML_MISSING", ""
    if uid not in cache:
        cache[uid] = parse_sections(xml_file)
    return classify(cache[uid])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dataset da ripulire (contiene annotation.json e images/)")
    ap.add_argument("--out", type=Path, required=True,
                    help="dove scrivere il dataset ripulito")
    ap.add_argument("--reports", type=Path, default=DEFAULT_REPORTS,
                    help=f"report XML originali (default: {DEFAULT_REPORTS})")
    ap.add_argument("--images", choices=("copy", "symlink", "none"), default="copy",
                    help="come portare le immagini nel dataset ripulito (default: copy)")
    ap.add_argument("--strict", action="store_true",
                    help="tiene i placeholder tipo 'None.' come impression valide")
    ap.add_argument("--drop-unverifiable", action="store_true",
                    help="rimuove anche i sample con XML mancante o id non valido "
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
    for path in (ann_path, args.reports):
        if not path.exists():
            sys.exit(f"Percorso non trovato: {path}")
    if args.images != "none" and not images_dir.is_dir():
        sys.exit(f"Cartella immagini non trovata: {images_dir}")

    annotation = json.loads(ann_path.read_text(encoding="utf-8"))
    removed_states = STRICT_STATES if args.strict else MISSING_STATES

    cache: dict = {}
    kept: dict[str, list] = {}
    removed: list[tuple[str, str, str]] = []
    unverifiable: list[tuple[str, str, str]] = []
    counts: Counter = Counter()

    for split, records in annotation.items():
        kept[split] = []
        for record in records:
            sample_id = str(record["id"])
            status, _ = status_of(sample_id, args.reports, cache)
            counts[status] += 1
            if status in removed_states:
                removed.append((split, sample_id, status))
            elif status in UNVERIFIABLE:
                unverifiable.append((split, sample_id, status))
                if args.drop_unverifiable:
                    removed.append((split, sample_id, status))
                else:
                    kept[split].append(record)
            else:
                kept[split].append(record)

    total = sum(counts.values())
    n_kept = sum(len(v) for v in kept.values())
    print(f"dataset:  {args.dataset}")
    print(f"report:   {args.reports}")
    print(f"modalita: {'strict (placeholder = valido)' if args.strict else 'placeholder = mancante'}\n")
    print(f"  {'split':<10}{'prima':>8}{'dopo':>8}{'rimossi':>10}")
    for split in annotation:
        n_before = len(annotation[split])
        n_removed = sum(1 for s, _, _ in removed if s == split)
        print(f"  {split:<10}{n_before:>8}{len(kept[split]):>8}{n_removed:>10}")
    print(f"  {'TOTALE':<10}{total:>8}{n_kept:>8}{len(removed):>10}\n")
    print("  esiti: " + ", ".join(f"{s}={n}" for s, n in counts.most_common()))

    if removed:
        shown = removed if not args.show else removed[: args.show]
        print(f"\n--- rimossi ({len(removed)}) ---")
        for split, sample_id, status in shown:
            print(f"  {sample_id:<26}{split:<8}{status}")
        if len(shown) < len(removed):
            print(f"  … e altri {len(removed) - len(shown)}  (--show 0 per tutti)")

    if unverifiable and not args.drop_unverifiable:
        print(f"\n{len(unverifiable)} sample NON verificabili (XML assente o id non valido):")
        for split, sample_id, status in unverifiable[:20]:
            print(f"  {sample_id:<26}{split:<8}{status}")
        sys.exit("\nInterrotto: non rimuovo sample che non ho potuto verificare. "
                 "Sistema i report, oppure passa --drop-unverifiable.")

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
    removed_ids = [sample_id for _, sample_id, _ in removed]
    (out / "removed_no_impression.txt").write_text(
        "\n".join(removed_ids) + "\n", encoding="utf-8"
    )

    if args.images != "none":
        kept_ids = [str(r["id"]) for records in kept.values() for r in records]
        (out / "images").mkdir()
        for sample_id in kept_ids:
            source = images_dir / sample_id
            if not source.is_dir():
                sys.exit(f"Cartella immagini mancante per {sample_id}: {source}")
            destination = out / "images" / sample_id
            if args.images == "copy":
                shutil.copytree(source, destination)
            else:
                destination.symlink_to(source.resolve(), target_is_directory=True)
        print(f"\nimmagini: {len(kept_ids)} cartelle ({args.images})")

    for extra in ("labeling_state.json",):
        if (args.dataset / extra).is_file():
            shutil.copy2(args.dataset / extra, out / extra)

    print(f"dataset ripulito: {out}")
    print(f"id rimossi:       {out / 'removed_no_impression.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
