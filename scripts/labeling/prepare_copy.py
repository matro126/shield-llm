#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "iu_xray_r2gen")
DST = os.path.join(ROOT, "iu_xray_r2gen_labeled")
STATE_NAME = "labeling_state.json"


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="ricrea la copia da zero, buttando via le etichette gia' fatte")
    args = ap.parse_args()

    if not os.path.isdir(SRC):
        sys.exit(f"Dataset originale non trovato: {SRC}")

    if os.path.isdir(DST):
        if not args.force:
            print(f"La copia di lavoro esiste gia': {DST}")
            print("Niente da fare. Usa --force per ricrearla da zero.")
            return
        state = os.path.join(DST, STATE_NAME)
        if os.path.exists(state):
            print(f"ATTENZIONE: {state} esiste, contiene le etichette gia' assegnate.")
            if input("Cancellare tutto e ricopiare? [scrivi 'si'] ").strip().lower() != "si":
                sys.exit("Annullato.")
        print("Rimuovo la copia esistente...")
        shutil.rmtree(DST)

    partial = DST + ".partial"
    if os.path.isdir(partial):
        print("Rimuovo una copia parziale precedente...")
        shutil.rmtree(partial)

    src_images = os.path.join(SRC, "images")
    studies = sorted(d for d in os.listdir(src_images)
                     if os.path.isdir(os.path.join(src_images, d)))
    total = len(studies)
    print(f"Copio {total} studi da {SRC} -> {DST}")

    os.makedirs(os.path.join(partial, "images"))
    shutil.copy2(os.path.join(SRC, "annotation.json"),
                 os.path.join(partial, "annotation.json"))

    t0 = time.time()
    nbytes = 0
    for i, study in enumerate(studies, 1):
        s = os.path.join(src_images, study)
        d = os.path.join(partial, "images", study)
        shutil.copytree(s, d)
        nbytes += sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        if i % 100 == 0 or i == total:
            el = time.time() - t0
            print(f"  {i}/{total} studi  {human(nbytes)}  {el:.0f}s", flush=True)

    os.rename(partial, DST)
    print(f"\nFatto in {time.time() - t0:.0f}s. Copia di lavoro pronta: {DST}")
    print("Ora lancia:  python3 labeler.py")


if __name__ == "__main__":
    main()
