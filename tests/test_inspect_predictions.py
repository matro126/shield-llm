from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from training import inspect_predictions


def sample(uid: str, labels: list[str]) -> dict:
    return {
        "id": uid,
        "factors": {"diagnostic_category": labels},
        "reference": "REFERENCE FULL TEXT",
        "prediction": "PREDICTION FULL TEXT",
        "reference_sections": {
            "findings": f"reference findings {uid}",
            "impression": f"reference impression {uid}",
        },
        "prediction_sections": {
            "findings": f"prediction findings {uid}",
            "impression": f"prediction impression {uid}",
        },
    }


class PathResolutionTests(unittest.TestCase):
    def test_resolve_results_accepts_val_predictions_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            predictions = results / "val_predictions"
            predictions.mkdir(parents=True)

            resolved = inspect_predictions.resolve_results(predictions)

        self.assertEqual(results.resolve(), resolved)


class MlflowProvenanceTests(unittest.TestCase):
    def test_mlflow_provenance_reads_results_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "results.json").write_text(
                json.dumps(
                    {
                        "provenance": {
                            "mlflow": {
                                "run_id": "abc123",
                                "tracking_uri": "http://127.0.0.1:5000",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            provenance = inspect_predictions.mlflow_provenance(results)

        self.assertEqual(
            {
                "run_id": "abc123",
                "tracking_uri": "http://127.0.0.1:5000",
            },
            provenance,
        )

    def test_mlflow_provenance_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provenance = inspect_predictions.mlflow_provenance(Path(directory))

        self.assertEqual(
            {"run_id": None, "tracking_uri": None},
            provenance,
        )


class ProbeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [sample(f"healthy-{i}", ["No Finding"]) for i in range(10)]
        self.samples += [
            sample("single-atelectasis", ["Atelectasis"]),
            sample("single-cardiomegaly", ["Cardiomegaly"]),
            sample("single-consolidation", ["Consolidation"]),
            sample("single-edema", ["Edema"]),
            sample("single-effusion", ["Pleural Effusion"]),
            sample("single-fracture", ["Fracture"]),
            sample("single-opacity", ["Lung Opacity"]),
            sample("single-pneumothorax", ["Pneumothorax"]),
            sample("multi-a", ["Atelectasis", "Lung Opacity"]),
            sample("multi-b", ["Cardiomegaly", "Edema"]),
            sample("multi-c", ["Pleural Effusion", "Consolidation"]),
            sample("multi-d", ["Pneumothorax", "Pleural Effusion"]),
            sample("multi-e", ["Fracture", "Lung Opacity"]),
            sample("multi-f", ["Cardiomegaly", "Support Devices"]),
            sample("multi-g", ["Lung Lesion", "Pleural Other"]),
            sample("other", ["Other"]),
        ]

    def test_selection_is_reproducible_balanced_and_capped(self) -> None:
        first = inspect_predictions.select_probe(self.samples, 6, 14, seed=17)
        second = inspect_predictions.select_probe(self.samples, 6, 14, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(6, len(first["healthy"]))
        self.assertEqual(14, len(first["sick"]))
        self.assertLessEqual(len(first["healthy"]) + len(first["sick"]), 30)
        selected = {row["id"]: row for row in self.samples}
        complexities = {
            len(inspect_predictions.pathologies(selected[uid]))
            for uid in first["sick"]
        }
        self.assertIn(1, complexities)
        self.assertTrue(any(value > 1 for value in complexities))
        covered = {
            label
            for uid in first["sick"]
            for label in inspect_predictions.pathologies(selected[uid])
        }
        self.assertTrue(
            {
                "Atelectasis",
                "Cardiomegaly",
                "Consolidation",
                "Edema",
                "Pleural Effusion",
                "Pneumothorax",
            }.issubset(covered)
        )

    def test_selection_rejects_more_than_thirty_requested_cases(self) -> None:
        with self.assertRaisesRegex(ValueError, "30"):
            inspect_predictions.select_probe(self.samples, 15, 16, seed=42)


class ManualReviewTests(unittest.TestCase):
    def test_target_view_includes_findings_and_impression(self) -> None:
        row = sample("case-1", ["Cardiomegaly"])

        text = inspect_predictions.sample_text(
            row,
            "reference",
            target="findings_impression",
            section="target",
        )

        self.assertEqual(
            "Findings:\nreference findings case-1\n\n"
            "Impression:\nreference impression case-1",
            text,
        )
        self.assertEqual(
            "reference impression case-1",
            inspect_predictions.sample_text(
                row,
                "reference",
                target="findings_impression",
                section="impression",
            ),
        )

    def test_review_csv_has_one_full_text_row_per_case_and_epoch(self) -> None:
        steps = []
        for epoch in (1.0, 2.0):
            steps.append(
                {
                    "experiment": "experiment-1",
                    "target": "findings_impression",
                    "epoch": epoch,
                    "step": int(epoch * 10),
                    "samples": [
                        sample("healthy-1", ["No Finding"]),
                        sample("sick-1", ["Cardiomegaly"]),
                    ],
                }
            )
        probe = {
            "healthy": ["healthy-1"],
            "sick": ["sick-1"],
            "labels": {"healthy-1": [], "sick-1": ["Cardiomegaly"]},
        }

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "review.csv"
            inspect_predictions.export_review_csv(
                steps,
                probe,
                destination,
                section="target",
            )
            with destination.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(4, len(rows))
        self.assertEqual(
            [
                ("healthy-1", "1.0"),
                ("healthy-1", "2.0"),
                ("sick-1", "1.0"),
                ("sick-1", "2.0"),
            ],
            [(row["case_id"], row["epoch"]) for row in rows],
        )
        self.assertEqual(
            {
                "false_positive_finding",
                "omitted_finding",
                "incorrect_location",
                "incorrect_severity",
                "spurious_comparison",
                "omitted_comparison",
                "major_errors",
                "minor_errors",
                "notes",
            },
            {
                key
                for key in rows[0]
                if key.endswith("finding")
                or key.startswith("incorrect_")
                or key.endswith("comparison")
                or key.endswith("errors")
                or key == "notes"
            },
        )
        self.assertIn("reference findings healthy-1", rows[0]["reference"])
        self.assertIn("reference impression healthy-1", rows[0]["reference"])


if __name__ == "__main__":
    unittest.main()
