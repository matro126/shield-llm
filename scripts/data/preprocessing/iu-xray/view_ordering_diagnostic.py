from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.data.preprocessing import _resolve

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_iu-xray_v1.0.yaml"


def _symmetry(png: Path) -> float:
    import numpy as np
    from PIL import Image

    arr = np.asarray(Image.open(png).convert("L").resize((128, 128)), dtype="float32")
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return float((arr * np.fliplr(arr)).mean())


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description="Diagnostico ordinamento viste R2Gen.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--sample", type=int, default=300, help="numero di studi campionati"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="soglia simmetria frontale/laterale",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = json.loads(
        json.dumps(
            __import__("yaml").safe_load(Path(args.config).read_text(encoding="utf-8"))
        )
    )
    images_dir = _resolve(PROJECT_ROOT, config["split"]["images_dir"])
    annotation_path = _resolve(PROJECT_ROOT, config["split"]["annotation"])
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    studies = [record["id"] for split in annotation.values() for record in split]

    random.seed(args.seed)
    pool = random.sample(studies, min(args.sample, len(studies)))
    sample = [
        s
        for s in pool
        if (images_dir / s / "0.png").exists() and (images_dir / s / "1.png").exists()
    ]
    if not sample:
        print(
            f"[diag] nessuno studio con entrambe le viste sotto {images_dir}",
            file=sys.stderr,
        )
        return 1

    sym0 = np.array([_symmetry(images_dir / s / "0.png") for s in sample])
    sym1 = np.array([_symmetry(images_dir / s / "1.png") for s in sample])
    inverted = float(np.mean(sym0 <= sym1))
    zero_lateral = float(np.mean(sym0 < args.threshold))

    print("── Ordinamento viste R2Gen (0.png = frontale?) ───────────────")
    print(f"  studi campionati (con entrambe le viste) : {len(sample)}")
    print(f"  simmetria media  0.png                   : {sym0.mean():.3f}")
    print(f"  simmetria media  1.png                   : {sym1.mean():.3f}")
    print(f"  ─ INVERSIONE (0.png ≤ simmetria di 1.png) : {inverted:.1%}")
    print(
        f"  0.png con simmetria < {args.threshold} (∼laterale)    : {zero_lateral:.1%}"
    )
    print("──────────────────────────────────────────────────────────────")
    print("Lettura: se l'inversione è ~0 → 0.png è affidabilmente la frontale.")
    print(
        "Se è non trascurabile (%%) → NON fidarsi per le varianti frontal-only: quella"
    )
    print(
        "frazione di studi verrebbe addestrata sulla vista sbagliata (misalignment vista↔testo)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
