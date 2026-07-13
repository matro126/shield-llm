from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.data.preprocessing import (
    _resolve,
    load_r2gen_annotation,
    parse_openi_report,
    r2gen_coverage_diagnostic,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_iu-xray_v1.0.yaml"


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description="Diagnostico copertura OpenI↔R2Gen.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    reports_dir = _resolve(PROJECT_ROOT, config["source"]["reports_dir"])
    annotation_path = _resolve(PROJECT_ROOT, config["split"]["annotation"])
    fields = config.get("fields", {})
    findings_label = fields.get("findings_label", "FINDINGS")
    impression_label = fields.get("impression_label", "IMPRESSION")

    reports = []
    for xml_path in sorted(reports_dir.glob("*.xml")):
        report = parse_openi_report(xml_path, findings_label, impression_label)
        if report is not None:
            reports.append(report)
    if not reports:
        print(f"[diag] nessun XML OpenI in {reports_dir}", file=sys.stderr)
        return 1

    annotation = load_r2gen_annotation(annotation_path)
    diag = r2gen_coverage_diagnostic(reports, annotation)

    print("── Copertura OpenI ↔ split R2Gen ─────────────────────────────")
    print(f"  studi R2Gen (annotation.json)            : {diag['n_r2gen_studies']}")
    print(f"  con FINDINGS OpenI (coperti dal filtro)  : {diag['n_with_findings']}")
    print(
        f"  FINDINGS vuoto, IMPRESSION presente      : {diag['n_empty_findings_with_impression']}"
    )
    print(
        f"  FINDINGS e IMPRESSION entrambi vuoti      : {diag['n_empty_findings_no_impression']}"
    )
    print(f"  nessun report OpenI (divergenza parsing) : {diag['n_no_openi_report']}")
    print(
        f"  ─ GAP (studi che il filtro scarterebbe)  : {diag['n_gap_findings']} ({diag['gap_fraction']:.2%})"
    )
    print("──────────────────────────────────────────────────────────────")
    print("Interpretazione:")
    print(
        "  gap ≈ 0       → il build OpenI-driven copre già i 2955: nessun intervento."
    )
    print(
        "  gap piccolo   → sottoinsieme dichiarato: alza split.coverage_tolerance e noti N in tesi."
    )
    print(
        "  gap grande    → variante standard con target R2Gen uniforme (leggibile → varianti custom)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
