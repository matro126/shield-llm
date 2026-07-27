from pathlib import Path
from typing import Optional


def render(headers, rows, align_right=None) -> str:
    align_right = set(align_right or [])
    data = [[str(c) for c in r] for r in rows]
    widths = [max(len(str(headers[i])), *(len(r[i]) for r in data)) if data
              else len(str(headers[i])) for i in range(len(headers))]

    def line(ch="-", junction="+"):
        return junction + junction.join(ch * (w + 2) for w in widths) + junction

    def row(cells):
        out = []
        for i, c in enumerate(cells):
            out.append(f" {str(c):>{widths[i]}} " if i in align_right
                       else f" {str(c):<{widths[i]}} ")
        return "|" + "|".join(out) + "|"

    parts = [line(), row(headers), line("=")]
    parts += [row(r) for r in data]
    parts.append(line())
    return "\n".join(parts)


def markdown(headers, rows, align_right=None) -> str:
    align_right = set(align_right or [])
    sep = ["---:" if i in align_right else ":---" for i in range(len(headers))]
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def emit(title, tables, md_path: Optional[Path] = None):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))
    for subtitle, headers, rows, ar in tables:
        print(f"\n{subtitle}")
        print(render(headers, rows, ar))

    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = [f"# {title}", ""]
        for subtitle, headers, rows, ar in tables:
            chunks += [f"## {subtitle}", "", markdown(headers, rows, ar), ""]
        md_path.write_text("\n".join(chunks), encoding="utf-8")
        print(f"\nTabella riepilogativa (Markdown): {md_path}")
