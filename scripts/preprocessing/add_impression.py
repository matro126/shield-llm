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

DEFAULT_DATASET = ROOT / "dataset" / "iu-xray" / "iu_xray_r2gen_final"
DEFAULT_OUT = ROOT / "dataset" / "iu-xray" / "iu_xray_r2gen_final_impression"
DEFAULT_REPORTS = ROOT / "dataset" / "iu-xray" / "iu_xray_original" / "reports"

BAD_STATES = {"MISSING_TAG", "EMPTY", "PLACEHOLDER", "XML_MISSING", "BAD_ID"}
EXTRA_FILES = (
    "labeling_state.json",
    "removed_no_impression.txt",
    "removed_same_view_pairs.txt",
    "fix_views_state.json",
)


def normalize(text: str) -> str:
    return " ".join(text.split())


def impression_of(
    sample_id: str, reports_dir: Path, cache: dict[str, dict]
) -> tuple[str, str]:
    uid = extract_uid(sample_id)
    if uid is None:
        return "BAD_ID", ""
    xml_file = reports_dir / f"{uid}.xml"
    if not xml_file.is_file():
        return "XML_MISSING", ""
    if uid not in cache:
        cache[uid] = parse_sections(xml_file)
    status, text = classify(cache[uid])
    return status, "" if status in BAD_STATES else normalize(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help=f"dataset di partenza (default: {DEFAULT_DATASET})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"dataset da creare (default: {DEFAULT_OUT})")
    ap.add_argument("--reports", type=Path, default=DEFAULT_REPORTS,
                    help=f"report XML originali (default: {DEFAULT_REPORTS})")
    ap.add_argument("--name", default="annotation_complete.json",
                    help="nome del json nel dataset in uscita "
                         "(default: annotation_complete.json)")
    ap.add_argument("--images", choices=("copy", "symlink", "none"), default="copy",
                    help="come portare le immagini nel dataset in uscita "
                         "(default: copy)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="procede lasciando impression vuota sui sample problematici")
    ap.add_argument("--overwrite", action="store_true",
                    help="sovrascrive --out se esiste gia'")
    ap.add_argument("--dry-run", action="store_true",
                    help="riporta soltanto, non scrive nulla")
    ap.add_argument("--show", type=int, default=30,
                    help="quanti sample problematici elencare (0 = tutti)")
    args = ap.parse_args(argv)

    ann_path = args.dataset / "annotation.json"
    images_dir = args.dataset / "images"
    for path in (ann_path, args.reports):
        if not path.exists():
            sys.exit(f"Percorso non trovato: {path}")
    if args.images != "none" and not images_dir.is_dir():
        sys.exit(f"Cartella immagini non trovata: {images_dir}")

    annotation = json.loads(ann_path.read_text(encoding="utf-8"))

    cache: dict[str, dict] = {}
    counts: Counter = Counter()
    problemi: list[tuple[str, str, str]] = []
    completo: dict[str, list] = {}
    lunghezze: list[int] = []

    for split, records in annotation.items():
        completo[split] = []
        for record in records:
            sample_id = str(record["id"])
            status, impression = impression_of(sample_id, args.reports, cache)
            counts[status] += 1
            if status in BAD_STATES:
                problemi.append((split, sample_id, status))
            else:
                lunghezze.append(len(impression))
            completo[split].append({**record, "impression": impression})

    totale = sum(counts.values())
    print(f"dataset: {args.dataset}")
    print(f"report:  {args.reports}\n")
    print(f"  {'split':<10}{'sample':>8}{'con impression':>16}{'problemi':>10}")
    for split in annotation:
        n_bad = sum(1 for s, _, _ in problemi if s == split)
        n = len(annotation[split])
        print(f"  {split:<10}{n:>8}{n - n_bad:>16}{n_bad:>10}")
    print(f"  {'TOTALE':<10}{totale:>8}{totale - len(problemi):>16}{len(problemi):>10}\n")
    print("  esiti: " + ", ".join(f"{s}={n}" for s, n in counts.most_common()))
    if lunghezze:
        media = sum(lunghezze) / len(lunghezze)
        print(f"  impression: media {media:.0f} caratteri, "
              f"min {min(lunghezze)}, max {max(lunghezze)}")

    if problemi:
        mostrati = problemi if not args.show else problemi[: args.show]
        print(f"\n--- sample senza impression utilizzabile ({len(problemi)}) ---")
        for split, sample_id, status in mostrati:
            print(f"  {sample_id:<26}{split:<8}{status}")
        if len(mostrati) < len(problemi):
            print(f"  … e altri {len(problemi) - len(mostrati)}  (--show 0 per tutti)")
        if not args.allow_missing:
            sys.exit("\nInterrotto: questi sample resterebbero senza impression. "
                     "Controlla i report, oppure passa --allow-missing.")

    if args.dry_run:
        print("\n[dry-run] nessun file scritto.")
        return 0

    out = args.out
    if out.exists():
        if not args.overwrite:
            sys.exit(f"\n{out} esiste gia': usa --overwrite per sostituirlo.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    (out / args.name).write_text(
        json.dumps(completo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.images != "none":
        ids = [str(r["id"]) for records in completo.values() for r in records]
        (out / "images").mkdir()
        for sample_id in ids:
            source = images_dir / sample_id
            if not source.is_dir():
                sys.exit(f"Cartella immagini mancante per {sample_id}: {source}")
            destination = out / "images" / sample_id
            if args.images == "copy":
                shutil.copytree(source, destination)
            else:
                destination.symlink_to(source.resolve(), target_is_directory=True)
        print(f"\nimmagini: {len(ids)} cartelle ({args.images})")

    for extra in EXTRA_FILES:
        if (args.dataset / extra).is_file():
            shutil.copy2(args.dataset / extra, out / extra)

    print(f"dataset creato: {out}")
    print(f"annotation:     {out / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
