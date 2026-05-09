"""
extract_test_set.py
-------------------
Legge test.json (prodotto da split_dataset.py) e:
  1. Crea un CSV con solo le righe del test set (da dataset_filtrato.csv)
  2. Copia le immagini del test set in una cartella dedicata

Uso:
    uv run python scripts/data/extract_test_set.py \
        --test-json  workspace/data/processed/test.json \
        --csv        workspace/data/dataset_filtrato.csv \
        --images-dir workspace/data/images/images_normalized \
        --output-dir workspace/data/test_set_baseline
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Estrai test set (CSV + immagini) da split_dataset output")
    p.add_argument("--test-json",   required=True,  help="Path a test.json prodotto da split_dataset.py")
    p.add_argument("--csv",         required=True,  help="Path al CSV completo (dataset_filtrato.csv)")
    p.add_argument("--images-dir",  required=True,  help="Cartella con tutte le immagini .png")
    p.add_argument("--output-dir",  required=True,  help="Cartella di output (verrà creata se non esiste)")
    return p.parse_args()


def extract_filenames_from_test_json(test_json_path: Path) -> list[str]:
    """
    Estrae i filename delle immagini dal test.json.
    Ogni esempio ha messages[1]["content"] che è una lista;
    cerca il dict con type=="image" e prende il basename del path.
    """
    with open(test_json_path, encoding="utf-8") as f:
        examples = json.load(f)

    filenames = []
    for ex in examples:
        for msg in ex["messages"]:
            if msg["role"] == "user":
                content = msg["content"]
                # content può essere stringa o lista
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            filenames.append(Path(item["image"]).name)
                            break
                elif isinstance(content, str):
                    # fallback: non dovrebbe succedere con questo notebook
                    pass
    return filenames


def main():
    args = parse_args()

    test_json_path = Path(args.test_json)
    csv_path       = Path(args.csv)
    images_dir     = Path(args.images_dir)
    output_dir     = Path(args.output_dir)

    # Verifica input
    for p, label in [(test_json_path, "test.json"), (csv_path, "CSV"), (images_dir, "cartella immagini")]:
        if not p.exists():
            raise FileNotFoundError(f"❌ {label} non trovato: {p}")

    # Cartelle di output
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" ESTRAZIONE TEST SET")
    print("=" * 60)
    print(f"  test.json:    {test_json_path}")
    print(f"  CSV input:    {csv_path}")
    print(f"  Immagini src: {images_dir}")
    print(f"  Output:       {output_dir}")
    print()

    # 1. Leggi i filename dal test.json
    print("📂 Lettura test.json...")
    test_filenames = extract_filenames_from_test_json(test_json_path)
    print(f"   Esempi nel test set: {len(test_filenames)}")

    if not test_filenames:
        raise ValueError("Nessun filename estratto da test.json — verifica il formato del file.")

    # 2. Filtra il CSV
    print("\n📊 Filtraggio CSV...")
    df_full = pd.read_csv(csv_path)
    print(f"   Righe totali nel CSV: {len(df_full)}")

    df_test = df_full[df_full["filename"].isin(test_filenames)].copy()
    print(f"   Righe nel test set:   {len(df_test)}")

    # Segnala eventuali filename non trovati nel CSV
    found_in_csv = set(df_test["filename"].tolist())
    missing_in_csv = [f for f in test_filenames if f not in found_in_csv]
    if missing_in_csv:
        print(f"   ⚠️  {len(missing_in_csv)} filename da test.json non trovati nel CSV:")
        for fn in missing_in_csv[:5]:
            print(f"       {fn}")
        if len(missing_in_csv) > 5:
            print(f"       ... e altri {len(missing_in_csv) - 5}")

    # Salva CSV test set
    out_csv_path = output_dir / "test_set.csv"
    df_test.to_csv(out_csv_path, index=False, encoding="utf-8")
    print(f"\n✅ CSV salvato: {out_csv_path}")

    # 3. Copia immagini
    print("\n🖼️  Copia immagini...")
    copied   = 0
    missing  = []

    for filename in test_filenames:
        src = images_dir / filename
        dst = out_images_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(filename)

    print(f"   Copiate: {copied}")
    if missing:
        print(f"   ⚠️  Non trovate: {len(missing)}")
        for fn in missing[:5]:
            print(f"       {fn}")
        if len(missing) > 5:
            print(f"       ... e altre {len(missing) - 5}")

    # Riepilogo finale
    print()
    print("=" * 60)
    print(" RIEPILOGO OUTPUT")
    print("=" * 60)
    print(f"  {out_csv_path}")
    print(f"    → {len(df_test)} righe, colonne: {list(df_test.columns)}")
    print(f"  {out_images_dir}/")
    print(f"    → {copied} immagini .png")
    print()
    if not missing and not missing_in_csv:
        print("✅ Tutto estratto correttamente. Nessun file mancante.")
    else:
        print("⚠️  Estrazione completata con alcuni file mancanti (vedi sopra).")


if __name__ == "__main__":
    main()
