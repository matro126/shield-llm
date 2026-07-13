from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _counts_table(
    counts: Mapping[str, Any], columns: tuple[str, str] = ("Valore", "N")
) -> str:
    rows = [f"| {columns[0]} | {columns[1]} |", "| --- | --- |"]
    for key, value in sorted(counts.items()):
        rows.append(f"| {key} | {value} |")
    return "\n".join(rows) if len(rows) > 2 else "_n/d_"


def render_dataset_card(manifest: Mapping[str, Any], stats: Mapping[str, Any]) -> str:
    manifest = dict(manifest or {})
    stats = dict(stats or {})
    name = manifest.get("dataset", "-")
    version = manifest.get("version", "-")
    preprocessing = manifest.get("preprocessing", {}) or {}
    split = manifest.get("split", {}) or {}
    source = manifest.get("source", {}) or {}

    lines: list[str] = []
    lines.append(f"# Dataset Card — {name} ({version})")
    lines.append("")
    lines.append(
        "> Schema Hugging Face / Gebru et al. (2021). Sezioni quantitative auto-generate."
    )
    lines.append("")

    lines.append("## Provenienza")
    lines.append("")
    lines.append(f"- Sorgente raw: `{source.get('raw_path', '-')}`")
    lines.append(f"- Hash DVC del raw: `{source.get('raw_dvc_hash', '-')}`")
    lines.append("")

    lines.append("## Struttura")
    lines.append("")
    lines.append(_counts_table(stats.get("n_examples", {}), ("Split", "Esempi")))
    lines.append("")

    lines.append("## Distribuzione (§3.7.6)")
    lines.append("")
    lines.append("**Categoria diagnostica**")
    lines.append("")
    lines.append(_counts_table(stats.get("diagnostic_category", {})))
    lines.append("")
    lines.append("**Proiezione**")
    lines.append("")
    lines.append(_counts_table(stats.get("projection", {})))
    lines.append("")

    lines.append("## Preprocessing (§5.2.1)")
    lines.append("")
    lines.append(f"- Tipo: `{preprocessing.get('type', '-')}`")
    lines.append(f"- Script git sha: `{preprocessing.get('script_git_sha', '-')}`")
    if preprocessing.get("translation_method"):
        lines.append(f"- Traduzione: `{preprocessing['translation_method']}`")
    lines.append("")

    lines.append("## Split")
    lines.append("")
    lines.append(f"- Strategia: `{split.get('strategy', '-')}`")
    if split.get("annotation"):
        lines.append(f"- Annotation: `{split['annotation']}`")
    if split.get("reference"):
        lines.append(f"- Riferimento: {split['reference']}")
    lines.append("")

    lines.append("## Versione DVC")
    lines.append("")
    lines.append(
        f"Versione `{version}`, tracciata su DVC con hash del contenuto. Ogni run MLflow che la usa la referenzia via tag `dataset.version` + `dvc.dataset_hash`."
    )
    lines.append("")

    lines.append("## Licenza, bias e rappresentatività")
    lines.append("")
    lines.append("")

    return "\n".join(lines)


def write_dataset_card(
    manifest: Mapping[str, Any],
    stats: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dataset_card(manifest, stats), encoding="utf-8")
    return path
