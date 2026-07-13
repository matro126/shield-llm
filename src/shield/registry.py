from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_GATE_KEY = "bertscore_f1"


def promotion_decision(
    metrics: Mapping[str, Any],
    promote_to: str,
    gate_key: str = DEFAULT_GATE_KEY,
) -> tuple[str, str]:
    if promote_to == "none":
        return ("skip", "promote_to=none (nessuna registrazione)")

    comparison = metrics.get("comparison_vs_baseline")
    if not isinstance(comparison, Mapping) or "status" in comparison:
        return (
            "hold",
            "confronto vs baseline non disponibile: registro senza promuovere",
        )

    n_entry = comparison.get("num_examples")
    if isinstance(n_entry, Mapping) and n_entry.get("delta"):
        return (
            "hold",
            f"num_examples diversi dal baseline (Δ {n_entry['delta']:+.0f}): "
            f"test set effettivi non allineati, confronto non valido",
        )

    entry = comparison.get(gate_key)
    if not isinstance(entry, Mapping) or "delta" not in entry:
        return ("hold", f"metrica gate '{gate_key}' assente nel confronto")

    delta = entry["delta"]
    if delta <= 0:
        return ("hold", f"{gate_key} Δ {delta:+.4f} ≤ 0: non supera il baseline")

    significance = metrics.get("significance")
    sig_entry = None
    if isinstance(significance, Mapping) and isinstance(
        significance.get("metrics"), Mapping
    ):
        sig_entry = significance["metrics"].get(gate_key)
    if isinstance(sig_entry, Mapping) and "significant" in sig_entry:
        ci = f"IC95 [{sig_entry['ci95_low']:+.4f}, {sig_entry['ci95_high']:+.4f}], p={sig_entry['p_value']:.4f}"
        if sig_entry["significant"]:
            return (
                "promote",
                f"{gate_key} Δ {delta:+.4f} > 0 e significativo ({ci}): supera il baseline",
            )
        return (
            "hold",
            f"{gate_key} Δ {delta:+.4f} > 0 ma NON significativo ({ci}): "
            f"rumore di campionamento non escluso",
        )
    return (
        "promote",
        f"{gate_key} Δ {delta:+.4f} > 0: supera il baseline "
        f"(⚠️ senza test di significatività: rigenera la valutazione per averlo)",
    )


def production_promotion_decision(
    target_version: str,
    target_tags: Mapping[str, Any],
    production_version: str | None,
    force: bool = False,
) -> tuple[str, str]:
    if production_version is not None and str(production_version) == str(
        target_version
    ):
        return ("skip", f"v{target_version} è già la production corrente")

    gate = target_tags.get("gate")
    if gate != "promote" and not force:
        return (
            "refuse",
            f"v{target_version} ha tag gate={gate!r}: non ha superato il gate "
            f"d'accettazione alla registrazione (usa --force per scavalcare)",
        )

    parts = [f"v{target_version} → production"]
    if gate != "promote":
        parts.append(f"⚠️ FORZATO nonostante gate={gate!r}")
    if production_version is not None:
        parts.append(f"la production corrente v{production_version} sarà archiviata")
    else:
        parts.append("nessuna production precedente da archiviare")
    return ("promote", "; ".join(parts))
