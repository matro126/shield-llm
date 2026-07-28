from __future__ import annotations

import statistics
import time
from typing import Any

from ..data.loaders import extract_image_paths
from .generate import _system_prompt, _user_prompt, generate_reports_batched
from .operational import percentile

DEFAULT_BATCHES = (1, 2, 4, 8, 16)


def _cuda():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


def _sync() -> None:
    if (torch := _cuda()) is not None:
        torch.cuda.synchronize()


def _items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "image_paths": extract_image_paths(r),
            "system": _system_prompt(r),
            "user": _user_prompt(r),
        }
        for r in records
    ]


def _vram() -> dict[str, float]:
    if (torch := _cuda()) is None:
        return {}
    return {
        "allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }


def _reset_vram() -> None:
    if (torch := _cuda()) is not None:
        torch.cuda.reset_peak_memory_stats()


def _distribuzione(valori: list[float], prefisso: str) -> dict[str, float]:
    if not valori:
        return {}
    return {
        f"{prefisso}_p50_s": percentile(valori, 50),
        f"{prefisso}_p95_s": percentile(valori, 95),
        f"{prefisso}_p99_s": percentile(valori, 99),
        f"{prefisso}_mean_s": statistics.fmean(valori),
        f"{prefisso}_min_s": min(valori),
        f"{prefisso}_max_s": max(valori),
        f"{prefisso}_stdev_s": statistics.stdev(valori) if len(valori) > 1 else 0.0,
    }


def _conta_token(processor: Any, testi: list[str]) -> list[int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        return [len(tokenizer.encode(t, add_special_tokens=False)) for t in testi]
    except Exception:  # noqa: BLE001
        return [len(t.split()) for t in testi]


def latenza_singola(
    model: Any,
    processor: Any,
    records: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    repetition_penalty: float,
    progress=None,
) -> dict[str, Any]:
    """Un referto alla volta, cronometrato singolarmente. Batch 1."""
    tempi: list[float] = []
    token: list[int] = []
    _reset_vram()

    for indice, record in enumerate(records):
        items = _items([record])
        _sync()
        t0 = time.perf_counter()
        try:
            testi = generate_reports_batched(
                model,
                processor,
                items,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
        except Exception:
            continue
        _sync()
        tempi.append(time.perf_counter() - t0)
        token += _conta_token(processor, testi)
        if progress:
            progress(indice + 1, len(records))

    out: dict[str, Any] = {
        "n": len(tempi),
        "batch_size": 1,
        **_distribuzione(tempi, "latency"),
    }
    if tempi and token:
        out["tokens_mean"] = statistics.fmean(token)
        out["tokens_per_s"] = sum(token) / sum(tempi)
        out["throughput_req_s"] = len(tempi) / sum(tempi)
    out.update({f"vram_{k}": v for k, v in _vram().items()})
    return out


def curva_carico(
    model: Any,
    processor: Any,
    records: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    repetition_penalty: float,
    batch_sizes=DEFAULT_BATCHES,
    batches_per_size: int = 3,
    progress=None,
) -> list[dict[str, Any]]:
    righe = []
    totale = len(batch_sizes) * batches_per_size
    fatti = 0

    for ampiezza in batch_sizes:
        servono = ampiezza * batches_per_size
        if len(records) < ampiezza:
            continue
        campione = (records * (servono // len(records) + 1))[:servono]
        tempi: list[float] = []
        token = 0
        _reset_vram()
        for k in range(batches_per_size):
            chunk = campione[k * ampiezza : (k + 1) * ampiezza]
            _sync()
            t0 = time.perf_counter()
            try:
                testi = generate_reports_batched(
                    model,
                    processor,
                    _items(chunk),
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                )
            except Exception as exc:
                righe.append({"batch_size": ampiezza, "errore": str(exc)[:120]})
                tempi = []
                break
            _sync()
            tempi.append(time.perf_counter() - t0)
            token += sum(_conta_token(processor, testi))
            fatti += 1
            if progress:
                progress(fatti, totale)
        if not tempi:
            continue
        medio = statistics.fmean(tempi)
        righe.append(
            {
                "batch_size": ampiezza,
                "batches": len(tempi),
                "latency_batch_mean_s": medio,
                "latency_batch_min_s": min(tempi),
                "latency_batch_max_s": max(tempi),
                "throughput_req_s": ampiezza / medio,
                "tokens_per_s": token / sum(tempi),
                "latency_amortized_s": medio / ampiezza,
                **{f"vram_{k}": v for k, v in _vram().items()},
            }
        )
    return righe


def profila(
    model: Any,
    processor: Any,
    records: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    repetition_penalty: float,
    n_singole: int = 60,
    batch_sizes=DEFAULT_BATCHES,
    batches_per_size: int = 3,
    vram_modello_gb: float | None = None,
    progress=None,
) -> dict[str, Any]:
    utili = [r for r in records if extract_image_paths(r)]
    if not utili:
        return {"errore": "nessun record con immagini utilizzabili"}

    try:
        generate_reports_batched(
            model,
            processor,
            _items(utili[:1]),
            max_new_tokens=min(64, max_new_tokens),
            repetition_penalty=repetition_penalty,
        )
        _sync()
    except Exception:
        pass

    if progress:
        progress("latenza a richiesta singola", 0, n_singole)
    singola = latenza_singola(
        model,
        processor,
        utili[:n_singole],
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        progress=(
            (lambda d, t: progress("latenza a richiesta singola", d, t))
            if progress
            else None
        ),
    )

    if progress:
        progress("curva carico/latenza", 0, len(batch_sizes) * batches_per_size)
    carico = curva_carico(
        model,
        processor,
        utili,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        batch_sizes=batch_sizes,
        batches_per_size=batches_per_size,
        progress=(
            (lambda d, t: progress("curva carico/latenza", d, t)) if progress else None
        ),
    )

    picco = max((r.get("vram_reserved_gb", 0) for r in carico), default=0)
    vram = {
        "modello_gb": vram_modello_gb,
        "picco_riservato_gb": max(picco, singola.get("vram_reserved_gb", 0)),
        "picco_allocato_gb": max(
            max((r.get("vram_allocated_gb", 0) for r in carico), default=0),
            singola.get("vram_allocated_gb", 0),
        ),
    }
    if vram_modello_gb:
        vram["generazione_gb"] = vram["picco_allocato_gb"] - vram_modello_gb

    return {
        "schema": "3.8.3",
        "richiesta_singola": singola,
        "carico": carico,
        "vram": vram,
        "robustezza_input_rumorosi": None,
        "nota": (
            "latenza a batch 1 = attesa reale di una richiesta; la curva di carico "
            "assimila B richieste concorrenti a un batch di ampiezza B, che e' come "
            "un motore di inferenza le serve. La robustezza si calcola a parte."
        ),
    }


def riassunto(profilo: dict[str, Any]) -> str:
    """Poche righe da stampare a fine valutazione."""
    if not profilo or profilo.get("errore"):
        return f"  profilazione non riuscita: {profilo.get('errore', '?')}"
    s = profilo["richiesta_singola"]
    righe = [
        f"  latenza per richiesta (batch 1, n={s.get('n', 0)}): "
        f"p50 {s.get('latency_p50_s', 0):.2f}s  "
        f"p95 {s.get('latency_p95_s', 0):.2f}s  "
        f"p99 {s.get('latency_p99_s', 0):.2f}s  "
        f"({s.get('tokens_per_s', 0):.0f} token/s)",
        f"  {'batch':>7}{'latenza':>10}{'throughput':>13}{'token/s':>10}{'VRAM':>10}",
    ]
    for r in profilo.get("carico", []):
        if "errore" in r:
            righe.append(f"  {r['batch_size']:>7}   errore: {r['errore']}")
            continue
        righe.append(
            f"  {r['batch_size']:>7}{r['latency_batch_mean_s']:>9.2f}s"
            f"{r['throughput_req_s']:>11.2f}/s{r['tokens_per_s']:>10.0f}"
            f"{r.get('vram_reserved_gb', 0):>9.1f}G"
        )
    v = profilo.get("vram", {})
    righe.append(
        f"  VRAM: picco allocato {v.get('picco_allocato_gb', 0):.1f} GB, "
        f"riservato {v.get('picco_riservato_gb', 0):.1f} GB"
        + (f", solo pesi {v['modello_gb']:.1f} GB" if v.get("modello_gb") else "")
    )
    return "\n".join(righe)
