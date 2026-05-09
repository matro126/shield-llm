"""
sanity_check.py — Verifica integrità dello split e delle immagini
=================================================================

Controlla:
  1. Che i file JSON e il manifest esistano
  2. Che ogni esempio nei JSON abbia findings e impression non vuoti
  3. Che ogni immagine referenziata nei JSON esista sul disco
  4. Che non ci siano uid duplicati tra i split (data leakage)
  5. Che il totale degli uid nei JSON corrisponda al manifest

Uso:
    uv run python scripts/data/sanity_check.py \
        --processed-dir workspace/data/processed \
        --images-dir    workspace/data/images/images_normalized

Dipendenze: pandas (opzionale), nessuna libreria non-standard
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


# ── Colori terminale ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠️  {msg}{RESET}")
def err(msg):   print(f"  {RED}❌ {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{'='*60}\n {msg}\n{'='*60}{RESET}")


# ── Estrai testo assistant da un esempio ──────────────────────────────────────
def get_assistant_text(example: dict) -> str:
    for msg in example.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


# ── Estrai image path da un esempio ──────────────────────────────────────────
def get_image_path(example: dict) -> str | None:
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        return item.get("image")
    return None


# ── Check 1: esistenza file ───────────────────────────────────────────────────
def check_files_exist(processed_dir: Path) -> bool:
    header("CHECK 1 — Esistenza file JSON e manifest")
    all_ok = True
    for name in ["train.json", "val.json", "test.json", "split_manifest.json"]:
        p = processed_dir / name
        if p.exists():
            size_kb = p.stat().st_size / 1024
            ok(f"{name} ({size_kb:.0f} KB)")
        else:
            err(f"{name} NON TROVATO in {processed_dir}")
            all_ok = False
    return all_ok


# ── Check 2: struttura messaggi e testi ──────────────────────────────────────
def check_json_content(processed_dir: Path) -> dict:
    header("CHECK 2 — Contenuto JSON (findings, impression, struttura)")

    results = {}
    splits = ["train", "val", "test"]

    for split in splits:
        path = processed_dir / f"{split}.json"
        if not path.exists():
            err(f"{split}.json non trovato — skip")
            results[split] = None
            continue

        with open(path, encoding="utf-8") as f:
            examples = json.load(f)

        n_total         = len(examples)
        n_no_assistant  = 0
        n_no_findings   = 0
        n_no_impression = 0
        n_no_image      = 0
        n_no_system     = 0

        for i, ex in enumerate(examples):
            messages = ex.get("messages", [])
            roles = [m.get("role") for m in messages]

            # Verifica struttura messaggi
            if "system" not in roles:
                n_no_system += 1
            if "user" not in roles:
                n_no_image += 1
            if "assistant" not in roles:
                n_no_assistant += 1
                continue

            # Verifica testo assistant
            text = get_assistant_text(ex)
            if not text.strip():
                n_no_assistant += 1
                continue

            if "Reperti:" not in text:
                n_no_findings += 1
            if "Impressione:" not in text:
                n_no_impression += 1

            # Verifica image
            if get_image_path(ex) is None:
                n_no_image += 1

        print(f"\n  {BOLD}{split}.json{RESET} — {n_total} esempi")
        if n_no_system     == 0: ok(f"System prompt presente in tutti gli esempi")
        else:                    err(f"System prompt assente in {n_no_system} esempi")

        if n_no_image      == 0: ok(f"Image path presente in tutti gli esempi")
        else:                    err(f"Image path assente in {n_no_image} esempi")

        if n_no_assistant  == 0: ok(f"Testo assistant presente in tutti gli esempi")
        else:                    err(f"Testo assistant assente/vuoto in {n_no_assistant} esempi")

        if n_no_findings   == 0: ok(f"'Reperti:' presente in tutti gli esempi")
        else:                    err(f"'Reperti:' assente in {n_no_findings} esempi")

        if n_no_impression == 0: ok(f"'Impressione:' presente in tutti gli esempi")
        else:                    err(f"'Impressione:' assente in {n_no_impression} esempi")

        results[split] = examples

    return results


# ── Check 3: immagini sul disco ───────────────────────────────────────────────
def check_images(split_data: dict, images_dir: Path) -> None:
    header("CHECK 3 — Immagini sul disco")

    if not images_dir.exists():
        err(f"Cartella immagini non trovata: {images_dir}")
        return

    # Conta quante immagini ci sono in totale nella cartella
    all_pngs = list(images_dir.glob("*.png"))
    print(f"\n  Immagini .png nella cartella: {len(all_pngs)}")

    for split, examples in split_data.items():
        if examples is None:
            continue

        missing  = []
        found    = 0
        no_path  = 0

        for ex in examples:
            img_path_str = get_image_path(ex)
            if img_path_str is None:
                no_path += 1
                continue

            # Prova il path così com'è, poi prova solo il filename nella images_dir
            img_path = Path(img_path_str)
            if img_path.exists():
                found += 1
            else:
                # Fallback: cerca solo per filename nella images_dir
                candidate = images_dir / img_path.name
                if candidate.exists():
                    found += 1
                else:
                    missing.append(img_path.name)

        print(f"\n  {BOLD}{split}.json{RESET}")
        ok(f"Trovate sul disco: {found}/{len(examples)}")
        if no_path:
            warn(f"Esempi senza image path: {no_path}")
        if missing:
            err(f"Immagini MANCANTI: {len(missing)}")
            for fn in missing[:10]:
                print(f"      {RED}→ {fn}{RESET}")
            if len(missing) > 10:
                print(f"      {RED}... e altre {len(missing) - 10}{RESET}")
        else:
            ok(f"Nessuna immagine mancante")


# ── Check 4: data leakage tra split ──────────────────────────────────────────
def check_leakage(processed_dir: Path) -> None:
    header("CHECK 4 — Data leakage tra split")

    manifest_path = processed_dir / "split_manifest.json"
    if not manifest_path.exists():
        err("split_manifest.json non trovato — impossibile verificare leakage")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    train_set = set(manifest.get("train_uids", []))
    val_set   = set(manifest.get("val_uids",   []))
    test_set  = set(manifest.get("test_uids",  []))

    print(f"\n  uid nel manifest: train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    leak_tv = train_set & val_set
    leak_tt = train_set & test_set
    leak_vt = val_set   & test_set

    if leak_tv: err(f"Train ∩ Val:  {len(leak_tv)} uid in comune")
    else:       ok("Train ∩ Val:  nessun uid in comune")

    if leak_tt: err(f"Train ∩ Test: {len(leak_tt)} uid in comune")
    else:       ok("Train ∩ Test: nessun uid in comune")

    if leak_vt: err(f"Val ∩ Test:   {len(leak_vt)} uid in comune")
    else:       ok("Val ∩ Test:   nessun uid in comune")

    # Verifica coerenza totale
    total_uid = len(train_set) + len(val_set) + len(test_set)
    all_uid   = train_set | val_set | test_set
    if total_uid == len(all_uid):
        ok(f"Totale uid unici: {len(all_uid)} (nessuna sovrapposizione)")
    else:
        err(f"Totale uid: {total_uid} ma unici: {len(all_uid)} — ci sono sovrapposizioni")


# ── Check 5: coerenza manifest ↔ JSON ────────────────────────────────────────
def check_manifest_coherence(processed_dir: Path, split_data: dict) -> None:
    header("CHECK 5 — Coerenza manifest ↔ JSON")

    manifest_path = processed_dir / "split_manifest.json"
    if not manifest_path.exists():
        err("split_manifest.json non trovato — skip")
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"\n  Seed nel manifest: {manifest.get('seed')}")
    print(f"  Split: train={manifest.get('train_size')}  val={manifest.get('val_size')}  test={manifest.get('test_size')}")

    for split in ["train", "val", "test"]:
        examples = split_data.get(split)
        if examples is None:
            continue
        manifest_count = len(manifest.get(f"{split}_uids", []))
        json_count     = len(examples)
        if manifest_count == json_count:
            ok(f"{split}: manifest={manifest_count}  json={json_count}  ✓")
        else:
            err(f"{split}: manifest={manifest_count} ma json={json_count} — discrepanza!")


# ── Riepilogo finale ──────────────────────────────────────────────────────────
def print_summary(split_data: dict) -> None:
    header("RIEPILOGO FINALE")
    total = 0
    for split, examples in split_data.items():
        if examples is not None:
            print(f"  {split:6s}: {len(examples):>5} esempi")
            total += len(examples)
    print(f"  {'TOTALE':6s}: {total:>5} esempi")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Sanity check su split JSON e immagini")
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("workspace/data/processed"),
        help="Cartella con train/val/test.json e split_manifest.json"
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=Path("workspace/data/images/images_normalized"),
        help="Cartella con le immagini .png"
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n{BOLD}SANITY CHECK — Split e Immagini{RESET}")
    print(f"  processed-dir: {args.processed_dir}")
    print(f"  images-dir:    {args.images_dir}")

    # Check 1
    files_ok = check_files_exist(args.processed_dir)
    if not files_ok:
        print(f"\n{RED}Interrotto: file mancanti. Rigenera lo split prima di procedere.{RESET}")
        sys.exit(1)

    # Check 2
    split_data = check_json_content(args.processed_dir)

    # Check 3
    check_images(split_data, args.images_dir)

    # Check 4
    check_leakage(args.processed_dir)

    # Check 5
    check_manifest_coherence(args.processed_dir, split_data)

    # Riepilogo
    print_summary(split_data)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"  Sanity check completato.")
    print(f"  Se tutti i check mostrano ✅ puoi procedere con il training.")
    print(f"{BOLD}{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()
