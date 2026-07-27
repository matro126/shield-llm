from __future__ import annotations

from pathlib import Path


def normalize_image(src: str | Path, dst: str | Path) -> None:
    from PIL import Image

    destination = Path(dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.convert("RGB").save(destination)


def view_symmetry(png: str | Path) -> float:
    import numpy as np
    from PIL import Image

    with Image.open(png) as image:
        array = np.asarray(image.convert("L").resize((128, 128)), dtype="float32")
    array = (array - array.mean()) / (array.std() + 1e-6)
    return float((array * np.fliplr(array)).mean())
