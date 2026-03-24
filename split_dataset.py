"""
split_dataset.py — Split train/val/test per IU X-Ray (Qwen3-VL fine-tuning)
=============================================================================

Legge il CSV già filtrato prodotto dal preprocessing e genera tre file JSON:
  train.json  (~80% dei pazienti)
  val.json    (~10% dei pazienti)
  test.json   (~10% dei pazienti)

Lo split avviene SEMPRE per uid (paziente), mai per riga:
questo garantisce che le immagini dello stesso paziente non
finiscano in split diversi (data leakage).

PREREQUISITO: esegui prima il preprocessing che produce dataset_filtrato.csv
              e carica le immagini in /workspace/data/images/images_normalized/

Uso su RunPod (default, nessun argomento necessario):
  python split_dataset.py

Uso con percorsi custom:
  python split_dataset.py --csv /path/to/dataset_filtrato.csv
  python split_dataset.py --train 0.8 --val 0.1 --test 0.1

Dipendenze: pandas, scikit-learn, tqdm
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ── Configurazione di default (RunPod) ──────────────────────────────────────
SEED       = 42
TRAIN_SIZE = 0.80
VAL_SIZE   = 0.10
TEST_SIZE  = 0.10

# Colonne del CSV
COL_FINDINGS   = "findings_it"
COL_IMPRESSION = "impression_it"
COL_UID        = "uid"
COL_FILENAME   = "filename"

# Percorsi RunPod
IMAGES_DIR  = Path("/home/maselli/develop/shield/data/images/images_normalized")
CSV_DEFAULT = Path("/home/maselli/develop/shield/data/dataset_filtrato.csv")
OUT_DEFAULT = Path("/home/maselli/develop/shield/data/processed")

# System prompt del radiologo
SYSTEM_PROMPT = (
    "Sei un radiologo esperto. "
    "Ti verranno fornite una o più immagini mediche. "
    "Il tuo compito è: "
    "1) Descrivere in modo conciso i reperti visibili nell'immagine o nelle immagini, "
    "incluse le strutture anatomiche, le anomalie e le osservazioni rilevanti. "
    "2) Fornire un'impressione clinica concisa che riassuma i reperti principali. "
    "Fornisci la risposta in ESATTAMENTE questo formato:\n"
    "Reperti:\n<i tuoi reperti dettagliati qui>\n\n"
    "Impressione:\n<la tua impressione concisa qui>\n\n"
    "NON includere altre sezioni o preamboli."
)


# ── Riproducibilità ──────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ── Formato chat-template con system prompt ───────────────────────────────────
def format_report(findings: str, impression: str) -> str:
    """
    Formatta il testo dell'assistant in modo coerente con il formato
    richiesto dal system prompt.
    """
    return (
        f"Reperti:\n{findings.strip()}\n\n"
        f"Impressione:\n{impression.strip()}"
    )


def build_chat_example(image_path: str, findings: str, impression: str) -> dict:
    """
    Costruisce un esempio nel formato chat-template di Qwen3-VL
    con system prompt, messaggio utente (immagine) e risposta assistant.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)}
                ]
            },
            {
                "role": "assistant",
                "content": format_report(findings, impression)
            }
        ]
    }


# ── Caricamento CSV preprocessato ────────────────────────────────────────────
def load_csv(csv_path: Path, images_dir: Path) -> pd.DataFrame:
    if not csv_path.exists():
        sys.exit(
            f"❌ CSV non trovato: {csv_path}\n"
            f"   Assicurati di aver caricato dataset_filtrato.csv in /workspace/data/"
        )
    if not images_dir.exists():
        sys.exit(
            f"❌ Cartella immagini non trovata: {images_dir}\n"
            f"   Assicurati di aver caricato le immagini in /workspace/data/images/images_normalized/"
        )

    df = pd.read_csv(csv_path)
    print(f"Righe caricate dal CSV: {len(df)}")
    print(f"Colonne: {list(df.columns)}")

    # Verifica colonne obbligatorie
    required = {COL_UID, COL_FILENAME, COL_FINDINGS, COL_IMPRESSION}
    missing_cols = required - set(df.columns)
    if missing_cols:
        sys.exit(
            f"❌ Colonne mancanti nel CSV: {missing_cols}\n"
            f"   Controlla COL_FINDINGS e COL_IMPRESSION in cima allo script."
        )

    # Costruisci image_path assoluto su RunPod
    df = df.copy()
    df["image_path"] = df[COL_FILENAME].apply(lambda f: str(images_dir / f))

    # Verifica che almeno alcune immagini esistano (spot check)
    sample = df["image_path"].head(5)
    missing = [p for p in sample if not Path(p).exists()]
    if len(missing) == len(sample):
        sys.exit(
            f"❌ Nessuna immagine trovata nel campione. Esempio path atteso:\n"
            f"   {sample.iloc[0]}\n"
            f"   Verifica --images-dir."
        )
    elif missing:
        print(f"⚠️  {len(missing)}/5 immagini del campione non trovate — verifica --images-dir")
    else:
        print(f"✅ Immagini trovate in: {images_dir}")

    return df


# ── Diagnostica sbilanciamento ────────────────────────────────────────────────
def print_imbalance_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(" DIAGNOSTICA SBILANCIAMENTO")
    print("=" * 60)

    findings_lens   = df[COL_FINDINGS].str.split().apply(len)
    impression_lens = df[COL_IMPRESSION].str.split().apply(len)

    print(f"\nLunghezza {COL_FINDINGS} (parole):")
    print(f"  min={findings_lens.min()}  max={findings_lens.max()}"
          f"  media={findings_lens.mean():.0f}  mediana={findings_lens.median():.0f}")

    print(f"\nLunghezza {COL_IMPRESSION} (parole):")
    print(f"  min={impression_lens.min()}  max={impression_lens.max()}"
          f"  media={impression_lens.mean():.0f}  mediana={impression_lens.median():.0f}")

    # Pazienti con più righe
    imgs_per_uid = df.groupby(COL_UID).size()
    multi = imgs_per_uid[imgs_per_uid > 1]
    if len(multi) > 0:
        print(f"\n⚠️  Pazienti con più di 1 riga nel CSV: {len(multi)}")
        print(f"   Max righe per paziente: {multi.max()}")
        print(f"   Lo split per uid gestisce correttamente questo caso.")
    else:
        print(f"\n✅ Ogni paziente ha esattamente 1 riga (deduplicazione già applicata).")

    # Quota referti brevi
    short_impression = (impression_lens <= 15).sum()
    pct_short = 100 * short_impression / len(df)
    print(f"\nReferti con impression ≤ 15 parole: {short_impression} ({pct_short:.1f}%)")
    if pct_short > 40:
        print("  ⚠️  Alta quota di referti brevi — tienila a mente se il ROUGE")
        print("     sul test set risulta artificialmente alto.")
    else:
        print("  ✅ Distribuzione impression nella norma per un task generativo.")

    print("\nNOTA: IU X-Ray è un task di generazione, non classificazione.")
    print("Non è necessaria stratificazione per classe.")
    print("Lo split per uid è la sola misura anti-leakage necessaria.")
    print("=" * 60)


# ── Split per uid ─────────────────────────────────────────────────────────────
def split_by_uid(
    df: pd.DataFrame,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> tuple[list, list, list]:
    """
    Split deterministico per uid (paziente).

    Garanzia anti-data-leakage:
      - Lo split avviene sugli uid unici, NON sulle righe.
      - sorted() garantisce determinismo indipendentemente dall'ordine del CSV.
    """
    unique_uids = sorted(df[COL_UID].unique().tolist())
    print(f"\nPazienti unici: {len(unique_uids)}")

    temp_size = val_size + test_size
    train_uids, temp_uids = train_test_split(
        unique_uids, test_size=temp_size, random_state=seed
    )

    relative_test_size = test_size / temp_size
    val_uids, test_uids = train_test_split(
        temp_uids, test_size=relative_test_size, random_state=seed
    )

    return train_uids, val_uids, test_uids


# ── Verifica data leakage ─────────────────────────────────────────────────────
def verify_no_leakage(train_uids: list, val_uids: list, test_uids: list) -> bool:
    train_set = set(train_uids)
    val_set   = set(val_uids)
    test_set  = set(test_uids)

    leak_tv = train_set & val_set
    leak_tt = train_set & test_set
    leak_vt = val_set   & test_set

    if leak_tv or leak_tt or leak_vt:
        print("\n❌ DATA LEAKAGE RILEVATO:")
        if leak_tv: print(f"   Train ∩ Val:  {len(leak_tv)} uid")
        if leak_tt: print(f"   Train ∩ Test: {len(leak_tt)} uid")
        if leak_vt: print(f"   Val ∩ Test:   {len(leak_vt)} uid")
        return False

    print("\n✅ Nessun data leakage — split per paziente corretto.")
    return True


# ── Riepilogo numerico ────────────────────────────────────────────────────────
def print_split_summary(
    df: pd.DataFrame,
    train_uids: list, val_uids: list, test_uids: list,
) -> None:
    n_train = df[COL_UID].isin(train_uids).sum()
    n_val   = df[COL_UID].isin(val_uids).sum()
    n_test  = df[COL_UID].isin(test_uids).sum()
    total   = len(df)

    print("\n" + "=" * 60)
    print(" RIEPILOGO SPLIT")
    print("=" * 60)
    print(f"{'Split':<8} {'Pazienti':>10} {'Esempi':>10} {'%':>8}")
    print("-" * 40)
    print(f"{'Train':<8} {len(train_uids):>10} {n_train:>10} {100*n_train/total:>7.1f}%")
    print(f"{'Val':<8} {len(val_uids):>10}   {n_val:>10} {100*n_val/total:>7.1f}%")
    print(f"{'Test':<8} {len(test_uids):>10}   {n_test:>10} {100*n_test/total:>7.1f}%")
    print(f"{'TOTALE':<8} {len(train_uids)+len(val_uids)+len(test_uids):>10} {total:>10}")
    print("=" * 60)


# ── Salvataggio JSON ──────────────────────────────────────────────────────────
def save_split(df: pd.DataFrame, uids: list, split_name: str, output_dir: Path) -> None:
    split_df = df[df[COL_UID].isin(uids)]
    examples = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"  {split_name}"):
        examples.append(
            build_chat_example(row["image_path"], row[COL_FINDINGS], row[COL_IMPRESSION])
        )
    path = output_dir / f"{split_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"  ✅ {split_name}.json — {len(examples)} esempi → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Split train/val/test per IU X-Ray su RunPod — anti data leakage per uid"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_DEFAULT,
        help=f"CSV filtrato dal preprocessing (default: {CSV_DEFAULT})",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help=f"Cartella immagini su RunPod (default: {IMAGES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DEFAULT,
        help=f"Cartella output per train/val/test.json (default: {OUT_DEFAULT})",
    )
    parser.add_argument("--seed",  type=int,   default=SEED,       help=f"Seed (default: {SEED})")
    parser.add_argument("--train", type=float, default=TRAIN_SIZE, help=f"Proporzione train (default: {TRAIN_SIZE})")
    parser.add_argument("--val",   type=float, default=VAL_SIZE,   help=f"Proporzione val (default: {VAL_SIZE})")
    parser.add_argument("--test",  type=float, default=TEST_SIZE,  help=f"Proporzione test (default: {TEST_SIZE})")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-9:
        sys.exit(
            f"❌ --train + --val + --test deve essere 1.0 "
            f"(ricevuto: {args.train + args.val + args.test:.4f})"
        )

    print("=" * 60)
    print(" SPLIT DATASET — IU X-Ray su RunPod")
    print("=" * 60)
    print(f"  CSV input:   {args.csv}")
    print(f"  Images dir:  {args.images_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Seed:        {args.seed}")
    print(f"  Split:       train={args.train:.0%} / val={args.val:.0%} / test={args.test:.0%}")
    print(f"\n  System prompt: {'sì, incluso in ogni esempio'}")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carica CSV
    df = load_csv(args.csv, args.images_dir)

    # 2. Diagnostica
    print_imbalance_report(df)

    # 3. Split per uid
    train_uids, val_uids, test_uids = split_by_uid(
        df,
        seed=args.seed,
        train_size=args.train,
        val_size=args.val,
        test_size=args.test,
    )

    # 4. Verifica data leakage
    if not verify_no_leakage(train_uids, val_uids, test_uids):
        sys.exit("Interrotto: data leakage rilevato.")

    # 5. Riepilogo
    print_split_summary(df, train_uids, val_uids, test_uids)

    # 6. Salva JSON
    print("\nSalvataggio JSON...")
    save_split(df, train_uids, "train", args.output_dir)
    save_split(df, val_uids,   "val",   args.output_dir)
    save_split(df, test_uids,  "test",  args.output_dir)

    # 7. Manifest
    manifest = {
        "seed":        args.seed,
        "train_size":  args.train,
        "val_size":    args.val,
        "test_size":   args.test,
        "csv_input":   str(args.csv),
        "images_dir":  str(args.images_dir),
        "system_prompt": SYSTEM_PROMPT,
        "train_uids":  sorted(train_uids),
        "val_uids":    sorted(val_uids),
        "test_uids":   sorted(test_uids),
    }
    manifest_path = args.output_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n  📋 Manifest salvato: {manifest_path}")
    print("\n✅ Split completato!")


if __name__ == "__main__":
    main()