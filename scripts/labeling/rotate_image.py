#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

DEFAULT_DATASET = (Path(__file__).resolve().parents[2] / "dataset" / "iu-xray"
                   / "iu_xray_r2gen_labeled")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CCW_TRANSPOSE = {90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}


def load_annotation(dataset: Path) -> dict:
    path = dataset / "annotation.json"
    if not path.is_file():
        sys.exit(f"annotation.json non trovato in {dataset}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def find_sample(annotation: dict, sample_id: str, split: str | None):
    splits = [split] if split else list(annotation)
    for sp in splits:
        for rec in annotation.get(sp, []):
            if rec.get("id") == sample_id:
                return sp, rec
    where = f"split '{split}'" if split else "nessuno split"
    sys.exit(f"sample '{sample_id}' non trovato in {where} di annotation.json")


def select_images(dataset: Path, record: dict, wanted: str):
    paths = list(record.get("image_path", []))
    sample_dir = dataset / "images" / record["id"]
    if sample_dir.is_dir():
        known = {Path(p).name for p in paths}
        for extra in sorted(sample_dir.iterdir()):
            if extra.suffix.lower() in IMAGE_SUFFIXES and extra.name not in known:
                paths.append(f"{record['id']}/{extra.name}")
    if not paths:
        sys.exit(f"il sample '{record['id']}' non ha immagini")

    if wanted == "all":
        return paths
    if wanted.isdigit():
        idx = int(wanted)
        if idx >= len(paths):
            sys.exit(f"indice {idx} fuori range: il sample ha {len(paths)} immagini")
        return [paths[idx]]

    low = wanted.lower()
    exact = [p for p in paths
             if Path(p).name.lower() == low or Path(p).stem.lower() == low]
    if exact:
        return exact

    hits = [p for p in paths if low in Path(p).name.lower()]
    if not hits:
        sys.exit(f"nessuna immagine di '{record['id']}' corrisponde a '{wanted}'.\n"
                 f"Disponibili: {', '.join(paths)}")
    return hits


def rotate(src: Path, dst: Path, angle: int, clockwise: bool, backup: bool,
           dry_run: bool):
    ccw_angle = (360 - angle) % 360 if clockwise else angle
    with Image.open(src) as img:
        size_before = img.size
        rotated = img.transpose(CCW_TRANSPOSE[ccw_angle])
        size_after = rotated.size
        verso = "orario" if clockwise else "antiorario"
        print(f"{src}\n  -> {dst}\n     {angle}° {verso}, "
              f"{size_before[0]}x{size_before[1]} -> {size_after[0]}x{size_after[1]}"
              + ("  [dry-run]" if dry_run else ""))
        if dry_run:
            return
        if backup and dst == src:
            bak = src.with_suffix(src.suffix + ".orig")
            if not bak.exists():
                shutil.copy2(src, bak)
                print(f"     backup: {bak}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        rotated.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True,
                    help="id del sample come in annotation.json (es. CXR2384_IM-0942)")
    ap.add_argument("--image", required=True,
                    help="quale immagine ruotare: nome esatto ('lateral_2', "
                         "'frontal.png'), sottostringa, un indice ('0') o 'all'")
    ap.add_argument("--angle", required=True, type=int, choices=(90, 180, 270),
                    help="gradi di rotazione")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="dataset con annotation.json e images/<ID>/...")
    ap.add_argument("--split", choices=("train", "val", "test"),
                    help="limita la ricerca del sample a uno split")
    ap.add_argument("--counterclockwise", action="store_true",
                    help="ruota in senso antiorario (default: orario)")
    ap.add_argument("--out", type=Path,
                    help="cartella di output (default: sovrascrive gli originali). "
                         "Le immagini vengono scritte in <out>/<ID>/<file>")
    ap.add_argument("--backup", action="store_true",
                    help="quando si sovrascrive, salva prima una copia .orig")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra cosa verrebbe fatto senza scrivere nulla")
    args = ap.parse_args()

    annotation = load_annotation(args.dataset)
    split, record = find_sample(annotation, args.sample, args.split)
    targets = select_images(args.dataset, record, args.image)
    print(f"sample '{args.sample}' (split: {split}) — {len(targets)} immagine/i\n")

    for rel in targets:
        src = args.dataset / "images" / rel
        if not src.is_file():
            sys.exit(f"file non trovato: {src}")
        dst = (args.out / rel) if args.out else src
        rotate(src, dst, args.angle, not args.counterclockwise, args.backup,
               args.dry_run)


if __name__ == "__main__":
    main()
