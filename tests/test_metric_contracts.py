from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from shield.data.prompts import split_sections
from shield.training import config
from shield.training.config import BASELINE_MODE, TRAINING_MODES, Identity, build_config
from shield.training import evaluation
from training import check_mlflow
from training.generate import read_overrides
from training import rq_report


ROOT = Path(__file__).resolve().parents[1]
MODES = set(TRAINING_MODES) | {BASELINE_MODE}


def active_identities(lang: str) -> list[Identity]:
    identities = []
    for directory in sorted((ROOT / "training" / lang).glob("*/*/*")):
        if directory.is_dir():
            identity = Identity.from_path(directory)
            if identity.mode in MODES:
                identities.append(identity)
    return identities


def resolved_config(identity: Identity):
    script = ROOT / identity.script_relpath
    return build_config(identity, ROOT, read_overrides(script))


class MetricConfigurationTests(unittest.TestCase):
    def test_all_active_entrypoints_have_valid_metric_configuration(self) -> None:
        failures = []
        identities = active_identities("en") + active_identities("it")
        for identity in identities:
            try:
                resolved_config(identity)
            except ValueError as exc:
                failures.append(f"{identity.name}: {exc}")
        self.assertEqual(40, len(identities))
        self.assertEqual([], failures)

    def test_italian_defaults_are_language_safe(self) -> None:
        identity = active_identities("it")[0]
        try:
            cfg = build_config(identity, ROOT, {})
        except ValueError as exc:
            self.fail(str(exc))
        self.assertEqual(
            ("bleu", "rougeL", "bertscore"),
            cfg.eval_metrics,
        )
        self.assertEqual(cfg.eval_metrics, cfg.test_metrics)
        self.assertEqual("findings.bertscore_f1", cfg.monitor_metric)
        self.assertFalse(cfg.chexbert_translate)

    def test_english_entrypoints_opt_in_to_chexbert(self) -> None:
        identities = active_identities("en")
        self.assertEqual(4, len(identities))
        for identity in identities:
            cfg = resolved_config(identity)
            self.assertIn("chexbert", cfg.test_metrics, identity.name)
            if identity.mode != BASELINE_MODE:
                self.assertIn("chexbert", cfg.eval_metrics, identity.name)
                self.assertEqual(
                    "findings.chexbert_f1_micro_top5",
                    cfg.monitor_metric,
                    identity.name,
                )

    def test_translated_italian_entrypoint_uses_chexbert_in_both_phases(self) -> None:
        identity = next(
            identity
            for identity in active_identities("it")
            if identity.name == "it_2B_lora_FL-FI"
        )
        cfg = resolved_config(identity)
        self.assertTrue(cfg.chexbert_translate)
        self.assertIn("chexbert", cfg.eval_metrics)
        self.assertIn("chexbert", cfg.test_metrics)
        self.assertEqual(
            "findings.chexbert_f1_micro_top5",
            cfg.monitor_metric,
        )

    def test_entrypoint_config_loader_applies_script_overrides(self) -> None:
        builder = getattr(config, "build_entrypoint_config", None)
        self.assertIsNotNone(builder)
        if builder is None:
            return
        english = next(
            identity
            for identity in active_identities("en")
            if identity.name == "en_2B_lora_F-F"
        )
        italian = next(
            identity
            for identity in active_identities("it")
            if identity.name == "it_2B_lora_FL-FI"
        )
        english_cfg = builder(english, ROOT)
        italian_cfg = builder(italian, ROOT)
        self.assertIn("chexbert", english_cfg.test_metrics)
        self.assertIn("chexbert", italian_cfg.test_metrics)
        self.assertTrue(italian_cfg.chexbert_translate)


class TestMetricSelectionTests(unittest.TestCase):
    def test_default_selection_uses_configured_metrics_for_tracking(self) -> None:
        resolver = getattr(config, "resolve_metric_selection", None)
        self.assertIsNotNone(resolver)
        if resolver is None:
            return
        names, tracked = resolver(None, ("bleu", "rougeL", "bertscore"))
        self.assertEqual(["bleu", "rougeL", "bertscore"], names)
        self.assertEqual("bleu,rougeL,bertscore", tracked)

    def test_explicit_selection_uses_requested_metrics_for_tracking(self) -> None:
        resolver = getattr(config, "resolve_metric_selection", None)
        self.assertIsNotNone(resolver)
        if resolver is None:
            return
        names, tracked = resolver(
            ["rougeL", "chexbert"],
            ("bleu", "rougeL", "bertscore"),
        )
        self.assertEqual(["rougeL", "chexbert"], names)
        self.assertEqual("rougeL,chexbert", tracked)


class SectionContractTests(unittest.TestCase):
    def test_findings_impression_references_require_separator(self) -> None:
        validator = getattr(evaluation, "validate_sectioned_references", None)
        self.assertIsNotNone(validator)
        if validator is None:
            return
        records = [{"id": "study-1"}, {"id": "study-2"}]
        references = [
            "Reperti:\nNormale\n<SEP>\nImpressione:\nNormale",
            "Reperti:\nVersamento\nImpressione:\nVersamento",
        ]
        with self.assertRaisesRegex(RuntimeError, "study-2"):
            validator(records, references, "findings_impression")

    def test_findings_references_do_not_require_separator(self) -> None:
        validator = getattr(evaluation, "validate_sectioned_references", None)
        self.assertIsNotNone(validator)
        if validator is None:
            return
        validator(
            [{"id": "study-1"}],
            ["Reperti:\nNormale"],
            "findings",
        )

    def test_header_fallback_separates_noncompliant_output(self) -> None:
        findings, impression = split_sections(
            "Reperti:\nVersamento pleurico\nImpressione:\nVersamento"
        )
        self.assertEqual("Versamento pleurico", findings)
        self.assertEqual("Versamento", impression)

    def test_warning_describes_header_fallback(self) -> None:
        diagnostics = evaluation.format_compliance(
            [{"id": "study-1"}],
            ["Reperti:\nVersamento\nImpressione:\nVersamento"],
            "findings_impression",
        )
        output = io.StringIO()
        with redirect_stdout(output):
            evaluation.stampa_formato(diagnostics, "findings_impression")
        warning = output.getvalue()
        self.assertIn("header Impression", warning)
        self.assertIn("tutto il testo a findings", warning)


class ResearchReportTests(unittest.TestCase):
    def test_findings_impression_report_does_not_invent_section_mean(self) -> None:
        rows = [
            self._row("it_2B_lora_F-F", "R", 0.31, None),
            self._row("it_2B_lora_F-FI", "R+I", 0.36, 0.62),
        ]
        previous = rq_report.HA_GRAFICI
        rq_report.HA_GRAFICI = False
        try:
            with tempfile.TemporaryDirectory() as directory:
                rendered = rq_report.report(
                    rows,
                    {},
                    "bertscore_f1",
                    Path(directory),
                )
        finally:
            rq_report.HA_GRAFICI = previous
        self.assertNotIn("la seconda la media di due", rendered)
        self.assertNotIn("media riportata", rendered)
        self.assertIn("| esperimento | findings | impression |", rendered)
        self.assertIn("Findings e Impression", rendered)

    @staticmethod
    def _row(
        experiment: str,
        target: str,
        findings: float,
        impression: float | None,
    ) -> dict:
        return {
            "esperimento": experiment,
            "modello": "2B",
            "modalita": "lora",
            "dataset": experiment.rsplit("_", 1)[-1],
            "viste": "1",
            "target": target,
            "stato": "completed",
            "early_stopped": False,
            "epoche": 2,
            "max_epoche": 2,
            "valutazioni": 2,
            "best_epoca": 2,
            "best_valore": findings,
            "val_loss": 1.0,
            "metriche": {},
            "findings": {"bertscore_f1": findings},
            "impression": (
                {"bertscore_f1": impression}
                if impression is not None
                else {}
            ),
            "ore": 1.0,
            "vram": 1.0,
            "gpu": "test",
            "curve": {},
        }


class MlflowMetricNamespaceTests(unittest.TestCase):
    def test_namespaces_match_sectioned_validation_metrics(self) -> None:
        resolver = getattr(check_mlflow, "metric_namespaces", None)
        self.assertIsNotNone(resolver)
        if resolver is None:
            return
        self.assertEqual(
            [
                "val.findings.bleu",
                "val.findings.bertscore_f1",
                "val.impression.bleu",
                "val.impression.bertscore_f1",
            ],
            resolver(
                "val",
                ("bleu", "bertscore"),
                "findings_impression",
            ),
        )

    def test_namespaces_use_real_clinical_metric_key(self) -> None:
        resolver = getattr(check_mlflow, "metric_namespaces", None)
        self.assertIsNotNone(resolver)
        if resolver is None:
            return
        self.assertEqual(
            ["test.findings.chexbert_f1_micro_top5"],
            resolver("test", ("chexbert",), "findings"),
        )


if __name__ == "__main__":
    unittest.main()
