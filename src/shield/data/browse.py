from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .analyze import SymmetryCache, caption_views
from .openi import parse_report


def _relative(target: Path, start: Path) -> str:
    return os.path.relpath(target, start).replace(os.sep, "/")


def build_records(
    root: Path,
    original_dir: Path,
    r2gen_dir: Path,
    page_dir: Path,
    cache: SymmetryCache,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reports_dir = original_dir / "reports"
    original_images = original_dir / "images"

    original: list[dict[str, Any]] = []
    by_uid: dict[str, Any] = {}

    files = sorted(
        reports_dir.glob("*.xml"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    for index, xml_path in enumerate(files, start=1):
        study = parse_report(xml_path)
        by_uid[study.uid] = study
        frontal, lateral = caption_views(study.caption)

        images = []
        for name in study.parent_images:
            path = original_images / f"{name}.png"
            if path.is_file():
                images.append(
                    {
                        "src": _relative(path, page_dir),
                        "label": name,
                        "sym": round(cache.score(path, name), 3)
                        if cache.path
                        else None,
                    }
                )

        original.append(
            {
                "id": study.uid,
                "split": "original",
                "file": study.report_file,
                "caption": study.caption,
                "frontal": frontal,
                "lateral": lateral,
                "findings": study.findings,
                "impression": study.impression,
                "n_declared": len(study.parent_images),
                "images": images,
            }
        )
        if index % 500 == 0:
            print(f"  … {index}/{len(files)} referti", file=sys.stderr)

    r2gen: list[dict[str, Any]] = []
    annotation_path = r2gen_dir / "annotation.json"
    if annotation_path.is_file():
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        r2gen_images = r2gen_dir / "images"
        for split in ("train", "val", "test"):
            for record in annotation.get(split, []):
                sample_id = str(record["id"])
                uid = sample_id.split("_", 1)[0]
                study = by_uid.get(uid)
                caption = study.caption if study else ""
                frontal, lateral = caption_views(caption)

                images = []
                for rel in record.get("image_path", []):
                    path = r2gen_images / rel
                    if path.is_file():
                        images.append(
                            {
                                "src": _relative(path, page_dir),
                                "label": rel,
                                "sym": round(cache.score(path, rel), 3)
                                if cache.path
                                else None,
                            }
                        )

                r2gen.append(
                    {
                        "id": sample_id,
                        "uid": uid,
                        "split": split,
                        "caption": caption,
                        "frontal": frontal,
                        "lateral": lateral,
                        "findings": study.findings if study else "",
                        "impression": study.impression if study else "",
                        "report": record.get("report", ""),
                        "n_declared": len(record.get("image_path", [])),
                        "images": images,
                    }
                )

    return original, r2gen


PAGE_TEMPLATE = """<title>IU X-ray — ispezione dataset</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280;
    --card: #f7f7f8; --line: #e2e2e5; --accent: #2563eb;
    --ok: #15803d; --warn: #b45309; --bad: #b91c1c;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #16171a; --fg: #e8e8ea; --muted: #9ca3af;
      --card: #1f2124; --line: #33363b; --accent: #60a5fa;
      --ok: #4ade80; --warn: #fbbf24; --bad: #f87171;
    }
  }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }
  header { position: sticky; top: 0; z-index: 5; background: var(--bg);
           border-bottom: 1px solid var(--line); padding: 12px 16px; }
  h1 { font-size: 16px; margin: 0 0 10px; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  select, input, button { font: inherit; padding: 5px 8px; border-radius: 6px;
           border: 1px solid var(--line); background: var(--card); color: var(--fg); }
  input[type=search] { min-width: 220px; }
  button { cursor: pointer; }
  button:hover { border-color: var(--accent); }
  .count { color: var(--muted); margin-left: auto; }
  .note { color: var(--muted); font-size: 12px; margin: 8px 0 0; max-width: 90ch; }
  main { padding: 16px; display: grid; gap: 14px; }
  .card { border: 1px solid var(--line); border-radius: 10px; background: var(--card);
          padding: 12px; display: grid; grid-template-columns: minmax(0, 300px) 1fr;
          gap: 14px; }
  @media (max-width: 720px) { .card { grid-template-columns: 1fr; } }
  .thumbs { display: flex; flex-wrap: wrap; gap: 6px; align-content: start; }
  .thumb { display: grid; gap: 2px; justify-items: center; }
  .thumb img { width: 140px; height: 140px; object-fit: contain; background: #000;
               border-radius: 6px; cursor: zoom-in; }
  .thumb span { font-size: 11px; color: var(--muted); max-width: 140px;
                overflow-wrap: anywhere; text-align: center; }
  .noimg { color: var(--bad); font-size: 12px; }
  .meta { min-width: 0; }
  .idline { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline;
            margin-bottom: 6px; }
  .idline code { font-size: 15px; font-weight: 600; }
  .badge { font-size: 11px; padding: 1px 7px; border-radius: 999px;
           border: 1px solid currentColor; }
  .b-ok { color: var(--ok); } .b-bad { color: var(--bad); }
  .b-warn { color: var(--warn); } .b-mut { color: var(--muted); }
  dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px;
       margin: 8px 0 0; }
  dt { color: var(--muted); font-size: 12px; }
  dd { margin: 0; overflow-wrap: anywhere; }
  dd.empty::after { content: "— assente —"; color: var(--bad); font-style: italic; }
  .pager { display: flex; gap: 8px; align-items: center; justify-content: center;
           padding: 8px 0 32px; }
  dialog { border: none; background: transparent; max-width: 96vw; max-height: 96vh; }
  dialog::backdrop { background: rgba(0,0,0,.85); }
  dialog img { max-width: 92vw; max-height: 88vh; object-fit: contain; }
  dialog p { color: #fff; text-align: center; font-size: 12px; margin: 6px 0 0; }
</style>

<header>
  <h1>IU X-ray — ispezione manuale</h1>
  <div class="controls">
    <select id="dataset">
      <option value="original">OpenI original (__N_ORIGINAL__)</option>
      <option value="r2gen">R2Gen split (__N_R2GEN__)</option>
    </select>
    <select id="split">
      <option value="">tutti gli split</option>
      <option value="train">train</option>
      <option value="val">val</option>
      <option value="test">test</option>
    </select>
    <select id="issue">
      <option value="">nessun filtro</option>
      <option value="no_findings">senza findings</option>
      <option value="no_impression">senza impression</option>
      <option value="no_frontal">senza frontale (caption)</option>
      <option value="no_lateral">senza laterale (caption)</option>
      <option value="unknown_views">viste indeterminate</option>
      <option value="no_images">senza file immagine</option>
      <option value="multi_images">3+ immagini</option>
      <option value="complete">completi</option>
    </select>
    <input type="search" id="q" placeholder="cerca id o testo…">
    <select id="size">
      <option value="25">25 per pagina</option>
      <option value="50">50</option>
      <option value="100">100</option>
    </select>
    <span class="count" id="count"></span>
  </div>
  <p class="note" id="note"></p>
</header>

<main id="list"></main>
<div class="pager">
  <button id="prev">← precedente</button>
  <span id="page"></span>
  <button id="next">successiva →</button>
</div>

<dialog id="zoom"><img id="zoomimg" alt=""><p id="zoomcap"></p></dialog>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const el = id => document.getElementById(id);
let page = 0;

const has = v => v !== null && v !== undefined && v !== '';

function matches(row, issue) {
  switch (issue) {
    case 'no_findings':   return !has(row.findings);
    case 'no_impression': return !has(row.impression);
    case 'no_frontal':    return row.frontal === false;
    case 'no_lateral':    return row.lateral === false;
    case 'unknown_views': return row.frontal === null;
    case 'no_images':     return row.images.length === 0;
    case 'multi_images':  return row.images.length > 2;
    case 'complete':      return has(row.findings) && has(row.impression)
                                 && row.frontal === true && row.lateral === true;
    default: return true;
  }
}

function filtered() {
  const rows = DATA[el('dataset').value];
  const split = el('split').value;
  const issue = el('issue').value;
  const q = el('q').value.trim().toLowerCase();
  return rows.filter(r =>
    (!split || r.split === split) &&
    matches(r, issue) &&
    (!q || r.id.toLowerCase().includes(q)
        || (r.findings || '').toLowerCase().includes(q)
        || (r.impression || '').toLowerCase().includes(q)
        || (r.report || '').toLowerCase().includes(q))
  );
}

function badge(label, state) { return `<span class="badge b-${state}">${label}</span>`; }

function viewBadges(row) {
  if (row.frontal === null) return badge('viste indeterminate', 'warn');
  return badge(row.frontal ? 'frontale' : 'no frontale', row.frontal ? 'ok' : 'bad')
       + badge(row.lateral ? 'laterale' : 'no laterale', row.lateral ? 'ok' : 'bad');
}

function esc(s) {
  return (s || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

function card(row) {
  const thumbs = row.images.length
    ? row.images.map(img => `<figure class="thumb">
        <img loading="lazy" src="${img.src}" alt="${esc(img.label)}"
             data-cap="${esc(img.label)} · simmetria ${img.sym ?? '—'}">
        <span>${esc(img.label)}<br>sym ${img.sym ?? '—'}</span></figure>`).join('')
    : `<p class="noimg">nessun file immagine trovato (${row.n_declared} dichiarate)</p>`;

  // In R2Gen esiste solo 'report'; findings/impression arrivano dall'XML OpenI via uId.
  const isR2gen = row.report !== undefined;
  const src = isR2gen ? ' <small>(OpenI)</small>' : '';
  const reportRow = isR2gen
    ? `<dt><b>report R2Gen</b></dt><dd>${esc(row.report)}</dd>` : '';

  return `<article class="card">
    <div class="thumbs">${thumbs}</div>
    <div class="meta">
      <div class="idline">
        <code>${esc(row.id)}</code>
        ${row.split !== 'original' ? badge(row.split, 'mut') : ''}
        ${viewBadges(row)}
        ${badge((has(row.findings) ? 'findings' : 'no findings') + (isR2gen ? ' OpenI' : ''),
                has(row.findings) ? 'ok' : 'bad')}
        ${badge((has(row.impression) ? 'impression' : 'no impression') + (isR2gen ? ' OpenI' : ''),
                has(row.impression) ? 'ok' : 'bad')}
        ${badge(row.images.length + '/' + row.n_declared + ' img',
                row.images.length === row.n_declared ? 'mut' : 'warn')}
      </div>
      <dl>
        <dt>caption</dt><dd class="${row.caption ? '' : 'empty'}">${esc(row.caption)}</dd>
        ${reportRow}
        <dt>findings${src}</dt><dd class="${row.findings ? '' : 'empty'}">${esc(row.findings)}</dd>
        <dt>impression${src}</dt><dd class="${row.impression ? '' : 'empty'}">${esc(row.impression)}</dd>
      </dl>
    </div>
  </article>`;
}

const NOTES = {
  original: "Sorgente OpenI: findings e impression sono le sezioni <AbstractText> "
    + "dell'XML. Le viste vengono dal <caption> (livello studio, dice quali proiezioni "
    + "comprende l'esame); la simmetria sotto ogni miniatura e' solo indicativa.",
  r2gen: "annotation.json contiene SOLO il campo 'report' (identico ai FINDINGS OpenI "
    + "in tutti i 2955 casi). findings e impression marcati (OpenI) NON esistono in "
    + "R2Gen: sono recuperati dallo studio di origine col join sull'uId, e servono a "
    + "sapere quali studi sarebbero inutilizzabili per una variante con target impression."
};

function render() {
  el('note').textContent = NOTES[el('dataset').value];
  const rows = filtered();
  const size = +el('size').value;
  const pages = Math.max(1, Math.ceil(rows.length / size));
  page = Math.min(page, pages - 1);
  el('list').innerHTML = rows.slice(page * size, page * size + size).map(card).join('')
    || '<p style="color:var(--muted)">nessuno studio corrisponde ai filtri.</p>';
  el('count').textContent = `${rows.length} studi`;
  el('page').textContent = `pagina ${page + 1} / ${pages}`;
  el('prev').disabled = page === 0;
  el('next').disabled = page >= pages - 1;
  window.scrollTo(0, 0);
}

for (const id of ['dataset', 'split', 'issue', 'size']) {
  el(id).addEventListener('change', () => { page = 0; render(); });
}
el('q').addEventListener('input', () => { page = 0; render(); });
el('prev').addEventListener('click', () => { page--; render(); });
el('next').addEventListener('click', () => { page++; render(); });

el('list').addEventListener('click', event => {
  const img = event.target.closest('img');
  if (!img) return;
  el('zoomimg').src = img.src;
  el('zoomcap').textContent = img.dataset.cap;
  el('zoom').showModal();
});
el('zoom').addEventListener('click', () => el('zoom').close());

render();
</script>
"""


def render_page(
    original: Sequence[dict[str, Any]], r2gen: Sequence[dict[str, Any]]
) -> str:
    payload = json.dumps(
        {"original": list(original), "r2gen": list(r2gen)}, ensure_ascii=False
    ).replace("</", "<\\/")
    return (
        PAGE_TEMPLATE.replace("__N_ORIGINAL__", f"{len(original)} studi")
        .replace("__N_R2GEN__", f"{len(r2gen)} studi")
        .replace("__DATA__", payload)
    )


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument("--original", type=Path, default=Path("dataset/iu-xray/iu_xray_original"))
    parser.add_argument("--r2gen", type=Path, default=Path("dataset/iu-xray/iu_xray_r2gen"))
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/dataset_analysis/browser.html")
    )
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="non calcolare la simmetria (usa solo la cache se presente)",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path = out_path.parent / "symmetry_cache.json"
    cache = SymmetryCache(None if args.no_symmetry else cache_path)

    print("[browse] raccolta studi…", file=sys.stderr)
    original, r2gen = build_records(
        root, root / args.original, root / args.r2gen, out_path.parent, cache
    )
    cache.save()

    out_path.write_text(render_page(original, r2gen), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1_048_576
    print(
        f"[browse] {len(original)} studi OpenI + {len(r2gen)} R2Gen → "
        f"{out_path} ({size_mb:.1f} MB)",
        file=sys.stderr,
    )
    print(f"  apri con:  open {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
