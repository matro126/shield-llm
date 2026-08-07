import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "preprocessing"
    / "translate_diagnostic_labels.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "translate_diagnostic_labels", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_translates_all_supported_labels_and_preserves_their_order():
    module = load_module()
    source_labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
        "No Finding",
        "Other",
    ]
    annotation = {
        "train": [{"id": "CXR1", "diagnostic_label": source_labels}],
        "val": [],
        "test": [],
    }
    translated = module.translate_annotation(annotation)
    assert translated["train"][0]["diagnostic_label"] == [
        "Atelettasia",
        "Cardiomegalia",
        "Consolidamento",
        "Edema",
        "Allargamento cardiomediastinico",
        "Frattura",
        "Lesione polmonare",
        "Opacità polmonare",
        "Versamento pleurico",
        "Altra anomalia pleurica",
        "Polmonite",
        "Pneumotorace",
        "Dispositivi di supporto",
        "Nessun reperto",
        "Altro",
    ]
    assert annotation["train"][0]["diagnostic_label"] == source_labels


def test_preserves_every_field_except_diagnostic_label():
    module = load_module()
    annotation = {
        "train": [
            {
                "id": "CXR2",
                "report": "Mild cardiomegaly.",
                "impression": "Cardiomegaly.",
                "diagnostic_label": ["Cardiomegaly", "Other"],
                "images": ["image.png"],
                "nested": {"value": [1, 2]},
            }
        ],
        "val": [],
        "test": [],
    }
    translated = module.translate_annotation(annotation)
    expected = json.loads(json.dumps(annotation))
    expected["train"][0]["diagnostic_label"] = ["Cardiomegalia", "Altro"]
    assert translated == expected
    assert translated is not annotation
    assert translated["train"][0] is not annotation["train"][0]


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"id": "CXR3"}, "CXR3: missing diagnostic_label"),
        ({"id": "CXR4", "diagnostic_label": []}, "CXR4: diagnostic_label must be a non-empty list"),
        ({"id": "CXR5", "diagnostic_label": "Edema"}, "CXR5: diagnostic_label must be a non-empty list"),
        ({"id": "CXR6", "diagnostic_label": [1]}, "CXR6: diagnostic_label entries must be strings"),
        ({"id": "CXR7", "diagnostic_label": ["Mass"]}, "CXR7: unknown diagnostic label: Mass"),
    ],
)
def test_rejects_missing_malformed_or_unknown_labels_with_sample_id(
    record, message
):
    module = load_module()
    with pytest.raises(ValueError, match=message):
        module.translate_annotation({"train": [record]})


def test_default_output_path_is_derived_from_input_name():
    module = load_module()
    assert module.default_output_path(
        Path("data/annotation_labeled.json")
    ) == Path("data/annotation_labeled_diagnostic_it.json")


def test_cli_writes_complete_json_to_default_output(tmp_path):
    source = tmp_path / "annotation_labeled.json"
    source.write_text(
        json.dumps(
            {
                "train": [
                    {
                        "id": "CXR8",
                        "report": "No acute disease.",
                        "diagnostic_label": ["No Finding"],
                    }
                ],
                "val": [],
                "test": [],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(source)],
        text=True,
        capture_output=True,
        check=False,
    )
    output = tmp_path / "annotation_labeled_diagnostic_it.json"
    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["train"][0] == {
        "id": "CXR8",
        "report": "No acute disease.",
        "diagnostic_label": ["Nessun reperto"],
    }
    assert str(output) in completed.stdout


def test_cli_does_not_write_partial_output_after_validation_error(tmp_path):
    source = tmp_path / "annotation_labeled.json"
    output = tmp_path / "translated.json"
    source.write_text(
        json.dumps(
            {
                "train": [
                    {"id": "CXR9", "diagnostic_label": ["Edema"]},
                    {"id": "CXR10", "diagnostic_label": ["Unknown"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "CXR10: unknown diagnostic label: Unknown" in completed.stderr
    assert not output.exists()
