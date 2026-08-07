import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "preprocessing"
    / "translate_annotation_openrouter.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "translate_annotation_openrouter", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def annotation():
    return {
        "train": [
            {
                "id": "CXR1_IM-0001",
                "report": "Mild cardiomegaly. No pneumothorax.",
                "impression": "Mild cardiomegaly.",
                "diagnostic_label": ["Cardiomegaly"],
                "untouched": {"value": 1},
            }
        ],
        "val": [
            {
                "id": "CXR2_IM-0002",
                "report": "The lungs are clear.",
                "impression": "No acute disease.",
                "diagnostic_label": ["No Finding"],
            }
        ],
        "test": [],
    }


def succeeded(custom_id, report_it, impression_it):
    return {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "report_it": report_it,
                                "impression_it": impression_it,
                            }
                        ),
                    }
                ],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        },
    }


def test_cli_defaults_to_openrouter_sonnet_5_low_and_all_samples():
    module = load_module()
    parser = module.build_parser()
    submit = parser.parse_args(["submit"])
    assert submit.input == Path("annotation_complete.json")
    assert submit.state == Path("annotation_translation_it.batch.json")
    assert submit.model == "claude-sonnet-5"
    assert submit.effort == "low"
    assert submit.max_tokens == 2048
    assert submit.limit is None
    download = parser.parse_args(["download"])
    assert download.output == Path("annotation_translated_it.json")
    assert download.raw_results == Path(
        "annotation_translation_it.batch_results.jsonl"
    )
    assert download.error_report == Path(
        "annotation_translation_it.batch_errors.json"
    )


def test_build_request_uses_sample_id_prompt_and_structured_translation():
    module = load_module()
    sample = module.iter_samples(annotation(), overwrite=False)[0]
    request = module.build_request(
        sample, "anthropic/claude-sonnet-5", "low", 2048
    )
    assert request["custom_id"] == "CXR1_IM-0001"
    assert request["params"]["model"] == "anthropic/claude-sonnet-5"
    assert request["params"]["output_config"]["effort"] == "low"
    assert "English into Italian" in request["params"]["system"]
    assert "Do not add, remove, or interpret information" in request["params"][
        "system"
    ]
    user = json.loads(request["params"]["messages"][0]["content"])
    assert user == {
        "report": "Mild cardiomegaly. No pneumothorax.",
        "impression": "Mild cardiomegaly.",
    }
    schema = request["params"]["output_config"]["format"]["schema"]
    assert schema["required"] == ["report_it", "impression_it"]
    assert set(schema["properties"]) == {"report_it", "impression_it"}
    assert schema["additionalProperties"] is False


def test_select_samples_uses_first_n_and_rejects_invalid_limits():
    module = load_module()
    samples = module.iter_samples(annotation(), overwrite=False)
    assert module.select_samples(samples, None) == samples
    assert [sample.sample_id for sample in module.select_samples(samples, 1)] == [
        "CXR1_IM-0001"
    ]
    with pytest.raises(ValueError, match="greater than zero"):
        module.select_samples(samples, 0)
    with pytest.raises(ValueError, match="exceeds"):
        module.select_samples(samples, 3)


def test_parse_batch_results_uses_ids_not_result_order():
    module = load_module()
    rows = [
        succeeded(
            "CXR2_IM-0002",
            "I polmoni sono liberi.",
            "Non si rilevano alterazioni acute.",
        ),
        succeeded(
            "CXR1_IM-0001",
            "Lieve cardiomegalia. Non evidenza di pneumotorace.",
            "Lieve cardiomegalia.",
        ),
    ]
    parsed, usage = module.parse_batch_results(
        rows, ("CXR1_IM-0001", "CXR2_IM-0002")
    )
    assert parsed == {
        "CXR1_IM-0001": {
            "report_it": "Lieve cardiomegalia. Non evidenza di pneumotorace.",
            "impression_it": "Lieve cardiomegalia.",
        },
        "CXR2_IM-0002": {
            "report_it": "I polmoni sono liberi.",
            "impression_it": "Non si rilevano alterazioni acute.",
        },
    }
    assert usage == {"input_tokens": 200, "output_tokens": 40}
    with pytest.raises(ValueError, match="missing result"):
        module.parse_batch_results(rows[:1], ("CXR1_IM-0001", "CXR2_IM-0002"))
    with pytest.raises(ValueError, match="duplicate result"):
        module.parse_batch_results(
            rows + [rows[0]], ("CXR1_IM-0001", "CXR2_IM-0002")
        )


def test_apply_translations_adds_italian_and_preserves_every_original_field():
    module = load_module()
    source = annotation()
    translated = module.apply_translations(
        source,
        {
            "CXR1_IM-0001": {
                "report_it": "Lieve cardiomegalia. Non evidenza di pneumotorace.",
                "impression_it": "Lieve cardiomegalia.",
            }
        },
        overwrite=False,
    )
    original = source["train"][0]
    output = translated["train"][0]
    assert output["report"] == original["report"]
    assert output["impression"] == original["impression"]
    assert output["diagnostic_label"] == original["diagnostic_label"]
    assert output["untouched"] == original["untouched"]
    assert output["report_it"] == (
        "Lieve cardiomegalia. Non evidenza di pneumotorace."
    )
    assert output["impression_it"] == "Lieve cardiomegalia."
    assert translated["val"][0] == source["val"][0]
    assert "report_it" not in original


def test_apply_translations_requires_nonempty_output_and_explicit_overwrite():
    module = load_module()
    source = annotation()
    with pytest.raises(ValueError, match="report_it is empty"):
        module.apply_translations(
            source,
            {
                "CXR1_IM-0001": {
                    "report_it": "",
                    "impression_it": "Lieve cardiomegalia.",
                }
            },
            False,
        )
    source["train"][0]["report_it"] = "Vecchia traduzione"
    source["train"][0]["impression_it"] = "Vecchia impressione"
    with pytest.raises(ValueError, match="translation already exists"):
        module.apply_translations(
            source,
            {
                "CXR1_IM-0001": {
                    "report_it": "Nuova traduzione",
                    "impression_it": "Nuova impressione",
                }
            },
            False,
        )
    replaced = module.apply_translations(
        source,
        {
            "CXR1_IM-0001": {
                "report_it": "Nuova traduzione",
                "impression_it": "Nuova impressione",
            }
        },
        True,
    )
    assert replaced["train"][0]["report_it"] == "Nuova traduzione"


def test_state_binds_batch_to_exact_input_and_selected_ids(tmp_path):
    module = load_module()
    source = tmp_path / "annotation_complete.json"
    source.write_text(json.dumps(annotation()), encoding="utf-8")
    samples = module.iter_samples(module.load_annotation(source), False)
    state = module.create_state(
        source,
        samples[:1],
        "batch_or_123",
        "anthropic/claude-sonnet-5",
        "low",
        False,
    )
    assert state["sample_ids"] == ["CXR1_IM-0001"]
    assert state["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert state["effort"] == "low"
    module.validate_state_input(state, source)
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input hash differs"):
        module.validate_state_input(state, source)


def test_download_materializes_full_json_by_custom_id(tmp_path):
    module = load_module()
    source = tmp_path / "annotation_complete.json"
    state_path = tmp_path / "translation.batch.json"
    output = tmp_path / "annotation_translated_it.json"
    raw = tmp_path / "translation.results.jsonl"
    errors = tmp_path / "translation.errors.json"
    source.write_text(json.dumps(annotation()), encoding="utf-8")
    samples = module.iter_samples(module.load_annotation(source), False)
    module.atomic_write_json(
        state_path,
        module.create_state(
            source,
            samples,
            "batch_or_123",
            "anthropic/claude-sonnet-5",
            "low",
            False,
        ),
    )

    class Client:
        def get_batch(self, batch_id):
            return {
                "id": batch_id,
                "status": "completed",
                "results": [
                    succeeded(
                        "CXR2_IM-0002",
                        "I polmoni sono liberi.",
                        "Non si rilevano alterazioni acute.",
                    ),
                    succeeded(
                        "CXR1_IM-0001",
                        "Lieve cardiomegalia. Non evidenza di pneumotorace.",
                        "Lieve cardiomegalia.",
                    ),
                ],
            }

        def batch_phase(self, status):
            return "complete"

        def get_results(self, status):
            return status["results"]

    result = module.download_batch(
        source, state_path, output, raw, errors, Client(), replace_output=False
    )
    translated = json.loads(output.read_text(encoding="utf-8"))
    assert translated["train"][0]["report"] == annotation()["train"][0]["report"]
    assert translated["train"][0]["report_it"].startswith("Lieve cardiomegalia")
    assert translated["val"][0]["impression_it"].startswith("Non si rilevano")
    assert result["sample_count"] == 2
    assert result["usage"] == {"input_tokens": 200, "output_tokens": 40}
    assert not errors.exists()


def test_submit_limit_sends_first_sample_with_resolved_openrouter_model(tmp_path):
    module = load_module()
    source = tmp_path / "annotation_complete.json"
    state_path = tmp_path / "translation.batch.json"
    source.write_text(json.dumps(annotation()), encoding="utf-8")
    args = module.build_parser().parse_args(
        [
            "submit",
            "--input",
            str(source),
            "--state",
            str(state_path),
            "--limit",
            "1",
        ]
    )

    class Client:
        def create_batch(self, requests):
            self.requests = requests
            return {"id": "batch_or_123"}

    client = Client()
    state = module.submit_batch(args, client)
    assert [request["custom_id"] for request in client.requests] == [
        "CXR1_IM-0001"
    ]
    assert client.requests[0]["params"]["model"] == "anthropic/claude-sonnet-5"
    assert state["sample_count"] == 1
    assert state["provider"] == "openrouter"
