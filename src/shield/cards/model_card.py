from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _numeric_table(
    metrics: Mapping[str, Any], columns: tuple[str, str] = ("Metrica", "Valore")
) -> str:
    rows = [f"| {columns[0]} | {columns[1]} |", "| --- | --- |"]
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append(f"| {key} | {_fmt(value)} |")
    return "\n".join(rows) if len(rows) > 2 else "_Nessuna metrica disponibile._"


def _kv_table(
    mapping: Mapping[str, Any], columns: tuple[str, str] = ("Campo", "Contenuto")
) -> str:
    rows = [f"| {columns[0]} | {columns[1]} |", "| --- | --- |"]
    rows.extend(f"| {key} | {value} |" for key, value in mapping.items())
    return "\n".join(rows)


def _comparison_table(comparison: Mapping[str, Any]) -> str:
    if "status" in comparison:
        return f"_{comparison['status']}_"
    rows = ["| Metrica | Baseline | Fine-tuned | Δ |", "| --- | --- | --- | --- |"]
    for key, entry in comparison.items():
        if isinstance(entry, Mapping) and "delta" in entry:
            rows.append(
                f"| {key} | {_fmt(entry['baseline'])} | {_fmt(entry['current'])} | {entry['delta']:+.4f} |"
            )
    return "\n".join(rows) if len(rows) > 2 else "_Confronto non disponibile._"


def _disaggregated_section(disaggregated: Mapping[str, Any]) -> str:
    if not disaggregated:
        return "_Nessuna analisi disaggregata._"
    blocks: list[str] = []
    for factor, groups in disaggregated.items():
        lines = [
            f"**{factor}**",
            "",
            "| Valore | n | BLEU | ROUGE-L |",
            "| --- | --- | --- | --- |",
        ]
        for value, metrics in groups.items():
            if metrics.get("status") == "not_estimable":
                lines.append(f"| {value} | {metrics.get('n', 0)} | _n.e._ | _n.e._ |")
            else:
                lines.append(
                    f"| {value} | {metrics.get('n', 0)} | {_fmt(metrics.get('bleu', 0.0))} | {_fmt(metrics.get('rougeL', 0.0))} |"
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_model_card(
    config: Mapping[str, Any],
    results: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    exp = config["experiment"]
    ft = config["finetuning"]
    ds = config["dataset"]
    model = config["model"]
    ev = config.get("evaluation", {})
    provenance = provenance or {}
    results = results or {}

    details = {
        "Nome esperimento": exp["name"],
        "Modello base": model["base_model"],
        "Metodo": ft["method"],
        "Quantizzazione": ft.get("quantization", "-"),
        "Seed": ft.get("seed", "-"),
        "Dataset": f"{ds['name']} ({ds['version']})",
        "Git commit": provenance.get("git.commit", "-"),
        "DVC dataset hash": provenance.get("dvc.dataset_hash", "-"),
    }
    if config.get("peft"):
        peft = config["peft"]
        details["LoRA r / alpha / dropout"] = (
            f"{peft.get('lora_r')} / {peft.get('lora_alpha')} / {peft.get('lora_dropout')}"
        )

    lines: list[str] = []
    lines.append(f"# Model Card — {exp['name']}")
    lines.append("")
    lines.append(
        "> Schema Mitchell et al. (2019). Sezioni quantitative auto-generate; le altre sono da compilare."
    )
    lines.append("")

    lines.append("## 1. Model details")
    lines.append("")
    lines.append(_kv_table(details))
    lines.append("")

    lines.append("## 2. Intended use")
    lines.append("")
    lines.append("")

    lines.append("## 3. Factors")
    lines.append("")
    factors = ev.get("disaggregate_by", [])
    lines.append(
        "Factor di valutazione: " + (", ".join(factors) if factors else "-") + "."
    )
    lines.append("")

    lines.append("## 4. Metrics")
    lines.append("")
    lines.append("Metriche: " + ", ".join(ev.get("metrics", [])) + ".")
    lines.append("")
    lines.append(
        f"Soglia d'accettazione: miglioramento misurabile sullo zero-shot baseline (`{ev.get('baseline_run', '-')}`)."
    )
    lines.append("")

    lines.append("## 5. Evaluation data")
    lines.append("")
    lines.append(
        f"Dataset di test: **{ds['name']}** versione **{ds['version']}** (DVC hash: `{provenance.get('dvc.dataset_hash', '-')}`). Valutazione su test set separato dal training."
    )
    lines.append("")

    lines.append("## 6. Training data")
    lines.append("")
    lines.append(
        f"Fine-tuning su **{ds['name']}** ({ds['version']}), referti tradotti in italiano, formattati come coppie prompt-risposta. Versione DVC: `{provenance.get('dvc.dataset_hash', '-')}`."
    )
    lines.append("")

    lines.append("## 7. Quantitative analyses")
    lines.append("")
    lines.append("### Risultati aggregati")
    lines.append("")
    lines.append(_numeric_table(results.get("aggregate", {})))
    lines.append("")
    lines.append("### Risultati disaggregati (unitari)")
    lines.append("")
    lines.append(_disaggregated_section(results.get("disaggregated", {})))
    lines.append("")
    lines.append("### Confronto vs baseline")
    lines.append("")
    lines.append(
        _comparison_table(
            results.get("comparison_vs_baseline", {"status": "non valutato"})
        )
    )
    lines.append("")
    if results.get("operational"):
        lines.append("### Metriche operative")
        lines.append("")
        lines.append(_numeric_table(results["operational"]))
        lines.append("")

    lines.append("## 8. Ethical considerations")
    lines.append("")
    lines.append("")

    lines.append("## 9. Caveats and recommendations")
    lines.append("")
    lines.append("")

    return "\n".join(lines)


def write_model_card(
    config: Mapping[str, Any],
    output_path: str | Path,
    results: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_model_card(config, results, provenance), encoding="utf-8")
    return path
