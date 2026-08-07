import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/labeling/label_annotation_anthropic.py"


def load_module():
    spec = importlib.util.spec_from_file_location("label_annotation_anthropic", SCRIPT)
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
                "diagnosticl_label": ["Cardiomegaly"],
                "untouched": {"value": 1},
            }
        ],
        "val": [
            {
                "id": "CXR2_IM-0002",
                "report": "The lungs are clear.",
                "impression": "No acute disease.",
                "diagnosticl_label": ["No Finding"],
            }
        ],
        "test": [],
    }


def succeeded(custom_id, labels):
    return {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"diagnostic_label": labels}),
                    }
                ],
                "model": "claude-sonnet-5",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        },
    }


def test_iter_samples_rejects_duplicate_ids_across_splits():
    module = load_module()
    payload = annotation()
    payload["test"].append(
        {"id": "CXR1_IM-0001", "report": "Clear.", "impression": "Normal."}
    )
    with pytest.raises(ValueError, match="duplicate sample id"):
        module.iter_samples(payload, overwrite=False)


def test_cli_limit_defaults_to_all_and_accepts_positive_integer():
    module = load_module()
    parser = module.build_parser()
    assert parser.parse_args(["submit"]).limit is None
    assert parser.parse_args(["run", "--limit", "1"]).limit == 1


def test_select_samples_uses_first_n_and_rejects_invalid_limits():
    module = load_module()
    samples = module.iter_samples(annotation(), overwrite=False)
    assert module.select_samples(samples, None) == samples
    assert [sample.sample_id for sample in module.select_samples(samples, 1)] == [
        "CXR1_IM-0001"
    ]
    with pytest.raises(ValueError, match="greater than zero"):
        module.select_samples(samples, 0)
    with pytest.raises(ValueError, match="greater than zero"):
        module.select_samples(samples, -1)
    with pytest.raises(ValueError, match="exceeds"):
        module.select_samples(samples, 3)


def test_build_request_uses_source_id_and_only_report_and_impression():
    module = load_module()
    sample = module.iter_samples(annotation(), overwrite=False)[0]
    request = module.build_request(sample, "claude-sonnet-5", "medium", 256)
    assert request["custom_id"] == "CXR1_IM-0001"
    assert request["params"]["model"] == "claude-sonnet-5"
    assert request["params"]["output_config"]["effort"] == "medium"
    user = json.loads(request["params"]["messages"][0]["content"])
    assert user == {
        "report": "Mild cardiomegaly. No pneumothorax.",
        "impression": "Mild cardiomegaly.",
    }
    item_schema = request["params"]["output_config"]["format"]["schema"][
        "properties"
    ]["diagnostic_label"]
    assert "uniqueItems" not in item_schema


def test_validate_labels_normalizes_duplicates_and_no_finding_conflict():
    module = load_module()
    assert module.validate_labels(["Pleural Effusion", "Cardiomegaly"]) == (
        "Cardiomegaly",
        "Pleural Effusion",
    )
    with pytest.raises(ValueError, match="unknown label"):
        module.validate_labels(["Mass"])
    assert module.validate_labels(["Edema", "Edema"]) == ("Edema",)
    assert module.validate_labels(["No Finding", "Other"]) == ("Other",)
    assert module.validate_labels(["No Finding", "No Finding"]) == (
        "No Finding",
    )


def test_parse_batch_results_requires_exactly_one_success_for_every_source_id():
    module = load_module()
    rows = [
        succeeded("CXR2_IM-0002", ["No Finding"]),
        succeeded("CXR1_IM-0001", ["Cardiomegaly"]),
    ]
    parsed, usage = module.parse_batch_results(rows, ("CXR1_IM-0001", "CXR2_IM-0002"))
    assert parsed == {
        "CXR1_IM-0001": ("Cardiomegaly",),
        "CXR2_IM-0002": ("No Finding",),
    }
    assert usage == {"input_tokens": 200, "output_tokens": 20}
    with pytest.raises(ValueError, match="missing result"):
        module.parse_batch_results(rows[:1], ("CXR1_IM-0001", "CXR2_IM-0002"))
    with pytest.raises(ValueError, match="unknown result id"):
        module.parse_batch_results(
            rows + [succeeded("CXR3_IM-0003", ["No Finding"])],
            ("CXR1_IM-0001", "CXR2_IM-0002"),
        )
    with pytest.raises(ValueError, match="duplicate result id"):
        module.parse_batch_results(rows + [rows[0]], ("CXR1_IM-0001", "CXR2_IM-0002"))


def test_state_binds_batch_to_exact_input_and_order(tmp_path):
    module = load_module()
    source = tmp_path / "annotation_complete.json"
    source.write_text(json.dumps(annotation()), encoding="utf-8")
    samples = module.iter_samples(module.load_annotation(source), overwrite=False)
    state = module.create_state(
        source,
        samples,
        "msgbatch_123",
        "claude-sonnet-5",
        "medium",
        False,
    )
    assert state["batch_id"] == "msgbatch_123"
    assert state["sample_ids"] == ["CXR1_IM-0001", "CXR2_IM-0002"]
    assert state["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    module.validate_state_input(state, source)
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input hash differs"):
        module.validate_state_input(state, source)


def test_apply_labels_only_replaces_typo_key_and_preserves_sample_order():
    module = load_module()
    source = annotation()
    result = module.apply_labels(
        source,
        {
            "CXR1_IM-0001": ("Cardiomegaly",),
            "CXR2_IM-0002": ("No Finding",),
        },
        overwrite=False,
    )
    assert list(result) == ["train", "val", "test"]
    assert [row["id"] for row in result["train"] + result["val"]] == [
        "CXR1_IM-0001",
        "CXR2_IM-0002",
    ]
    assert "diagnosticl_label" not in result["train"][0]
    assert result["train"][0]["diagnostic_label"] == ["Cardiomegaly"]
    assert result["train"][0]["untouched"] == {"value": 1}
    assert "diagnosticl_label" in source["train"][0]


def test_apply_labels_updates_only_selected_samples():
    module = load_module()
    source = annotation()
    result = module.apply_labels(
        source,
        {"CXR1_IM-0001": ("Cardiomegaly",)},
        overwrite=False,
    )
    assert result["train"][0]["diagnostic_label"] == ["Cardiomegaly"]
    assert "diagnosticl_label" not in result["train"][0]
    assert result["val"][0] == source["val"][0]


def test_apply_labels_rejects_empty_unknown_ids_and_respects_overwrite():
    module = load_module()
    source = annotation()
    with pytest.raises(ValueError, match="labels are empty"):
        module.apply_labels(source, {}, False)
    with pytest.raises(ValueError, match="unknown labels id"):
        module.apply_labels(source, {"CXR3_IM-0003": ("No Finding",)}, False)
    source["train"][0]["diagnostic_label"] = ["No Finding"]
    with pytest.raises(ValueError, match="already exists"):
        module.apply_labels(
            source,
            {
                "CXR1_IM-0001": ("Cardiomegaly",),
                "CXR2_IM-0002": ("No Finding",),
            },
            False,
        )
    replaced = module.apply_labels(
        source,
        {
            "CXR1_IM-0001": ("Cardiomegaly",),
            "CXR2_IM-0002": ("No Finding",),
        },
        True,
    )
    assert replaced["train"][0]["diagnostic_label"] == ["Cardiomegaly"]


def test_submit_limit_sends_and_records_only_first_n_samples(tmp_path):
    module = load_module()
    source = tmp_path / "annotation_complete.json"
    state_path = tmp_path / "batch_state.json"
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
        def __init__(self):
            self.requests = None

        def create_batch(self, requests):
            self.requests = requests
            return {"id": "msgbatch_123"}

    client = Client()
    state = module.submit_batch(args, client)
    assert [request["custom_id"] for request in client.requests] == [
        "CXR1_IM-0001"
    ]
    assert state["sample_count"] == 1
    assert state["sample_ids"] == ["CXR1_IM-0001"]


def test_atomic_write_json_produces_valid_complete_file(tmp_path):
    module = load_module()
    output = tmp_path / "result.json"
    module.atomic_write_json(output, annotation())
    assert json.loads(output.read_text(encoding="utf-8")) == annotation()
    assert not list(tmp_path.glob(f".{output.name}.*"))


def test_batch_client_sends_expected_http_envelope():
    module = load_module()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, headers, body, timeout))
        return json.dumps(
            {
                "id": "msgbatch_123",
                "type": "message_batch",
                "processing_status": "in_progress",
                "request_counts": {
                    "processing": 2,
                    "succeeded": 0,
                    "errored": 0,
                    "canceled": 0,
                    "expired": 0,
                },
                "ended_at": None,
                "created_at": "2026-08-07T12:00:00Z",
                "expires_at": "2026-08-08T12:00:00Z",
                "cancel_initiated_at": None,
                "results_url": None,
            }
        ).encode()

    client = module.AnthropicBatchClient(
        "secret", "https://api.anthropic.test", "2023-06-01", transport=transport
    )
    requests = [
        module.build_request(sample, "claude-sonnet-5", "medium", 256)
        for sample in module.iter_samples(annotation(), False)
    ]
    response = client.create_batch(requests)
    assert response["id"] == "msgbatch_123"
    method, url, headers, body, timeout = calls[0]
    assert method == "POST"
    assert url == "https://api.anthropic.test/v1/messages/batches"
    assert headers["x-api-key"] == "secret"
    assert headers["anthropic-version"] == "2023-06-01"
    assert json.loads(body) == {"requests": requests}
    assert timeout == 120.0


def test_download_materializes_labels_by_custom_id_not_result_order(tmp_path):
    module = load_module()
    source = tmp_path / "annotation_complete.json"
    state_path = tmp_path / "batch_state.json"
    output = tmp_path / "annotation_complete_labeled.json"
    raw = tmp_path / "batch_results.jsonl"
    errors = tmp_path / "batch_errors.json"
    source.write_text(json.dumps(annotation()), encoding="utf-8")
    samples = module.iter_samples(module.load_annotation(source), False)
    state = module.create_state(
        source, samples, "msgbatch_123", "claude-sonnet-5", "medium", False
    )
    module.atomic_write_json(state_path, state)

    class Client:
        def get_batch(self, batch_id):
            return {
                "id": batch_id,
                "processing_status": "ended",
                "results_url": "https://api.anthropic.test/results",
            }

        def download_results(self, results_url):
            return [
                succeeded("CXR2_IM-0002", ["No Finding"]),
                succeeded("CXR1_IM-0001", ["Cardiomegaly"]),
            ]

    result = module.download_batch(
        source, state_path, output, raw, errors, Client(), replace_output=False
    )
    labeled = json.loads(output.read_text(encoding="utf-8"))
    assert labeled["train"][0]["diagnostic_label"] == ["Cardiomegaly"]
    assert labeled["val"][0]["diagnostic_label"] == ["No Finding"]
    assert result["sample_count"] == 2
    assert result["usage"] == {"input_tokens": 200, "output_tokens": 20}
    assert [json.loads(line)["custom_id"] for line in raw.read_text().splitlines()] == [
        "CXR2_IM-0002",
        "CXR1_IM-0001",
    ]
    assert not errors.exists()


def test_download_writes_error_report_and_never_partial_output(tmp_path):
    module = load_module()
    source = tmp_path / "annotation_complete.json"
    state_path = tmp_path / "batch_state.json"
    output = tmp_path / "annotation_complete_labeled.json"
    raw = tmp_path / "batch_results.jsonl"
    errors = tmp_path / "batch_errors.json"
    source.write_text(json.dumps(annotation()), encoding="utf-8")
    samples = module.iter_samples(module.load_annotation(source), False)
    module.atomic_write_json(
        state_path,
        module.create_state(
            source, samples, "msgbatch_123", "claude-sonnet-5", "medium", False
        ),
    )

    class Client:
        def get_batch(self, batch_id):
            return {
                "id": batch_id,
                "processing_status": "ended",
                "results_url": "https://api.anthropic.test/results",
            }

        def download_results(self, results_url):
            return [succeeded("CXR1_IM-0001", ["Cardiomegaly"])]

    with pytest.raises(ValueError, match="missing result"):
        module.download_batch(
            source, state_path, output, raw, errors, Client(), replace_output=False
        )
    assert not output.exists()
    report = json.loads(errors.read_text(encoding="utf-8"))
    assert report["batch_id"] == "msgbatch_123"
    assert "missing result id: CXR2_IM-0002" in report["error"]


def test_cli_defaults_to_sonnet_5_medium_and_safe_output_names():
    module = load_module()
    parser = module.build_parser()
    args = parser.parse_args(["submit"])
    assert args.input == Path("annotation_complete.json")
    assert args.state == Path("annotation_complete.batch.json")
    assert args.model == "claude-sonnet-5"
    assert args.effort == "medium"
    assert args.max_tokens == 256
    args = parser.parse_args(["download"])
    assert args.output == Path("annotation_complete_labeled.json")
    assert args.raw_results == Path("annotation_complete.batch_results.jsonl")
    assert args.error_report == Path("annotation_complete.batch_errors.json")


def test_openrouter_client_uses_batch_endpoint_bearer_auth_and_messages_skin():
    module = load_module()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, headers, body, timeout))
        return json.dumps(
            {"id": "batch_or_123", "status": "validating", "results": None}
        ).encode()

    client = module.OpenRouterBatchClient(
        "or-secret", "https://openrouter.ai", transport=transport
    )
    samples = module.iter_samples(annotation(), False)
    requests = [
        module.build_request(sample, "anthropic/claude-sonnet-5", "medium", 256)
        for sample in samples
    ]
    response = client.create_batch(requests)
    assert response["id"] == "batch_or_123"
    method, url, headers, body, timeout = calls[0]
    assert method == "POST"
    assert url == "https://openrouter.ai/api/beta/batches"
    assert headers["Authorization"] == "Bearer or-secret"
    assert "x-api-key" not in headers
    payload = json.loads(body)
    assert payload["endpoint"] == "/v1/messages"
    assert payload["model"] == "anthropic/claude-sonnet-5"
    assert payload["requests"][0]["custom_id"] == "CXR1_IM-0001"
    assert payload["requests"][0]["body"] == requests[0]["params"]


def test_openrouter_normalizes_inlined_messages_results():
    module = load_module()
    client = module.OpenRouterBatchClient(
        "or-secret", "https://openrouter.ai", transport=lambda *args: b"{}"
    )
    status = {
        "id": "batch_or_123",
        "status": "completed",
        "results": [
            {
                "custom_id": "CXR2_IM-0002",
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "msg_2",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"diagnostic_label":["No Finding"]}',
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 3},
                    },
                },
                "error": None,
            },
            {
                "custom_id": "CXR1_IM-0001",
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"diagnostic_label":["Cardiomegaly"]}',
                            }
                        ],
                        "usage": {"input_tokens": 11, "output_tokens": 4},
                    },
                },
                "error": None,
            },
        ],
    }
    rows = client.get_results(status)
    labels, usage = module.parse_batch_results(
        rows, ("CXR1_IM-0001", "CXR2_IM-0002")
    )
    assert labels == {
        "CXR1_IM-0001": ("Cardiomegaly",),
        "CXR2_IM-0002": ("No Finding",),
    }
    assert usage == {"input_tokens": 21, "output_tokens": 7}


def test_provider_defaults_select_matching_key_url_and_model(monkeypatch):
    module = load_module()
    parser = module.build_parser()
    anthropic = parser.parse_args(["submit"])
    assert anthropic.provider is None
    assert module.resolved_provider(anthropic) == "anthropic"
    assert module.resolved_model(module.resolved_provider(anthropic), anthropic.model) == "claude-sonnet-5"
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    openrouter = parser.parse_args(["submit", "--provider", "openrouter"])
    assert module.resolved_model(openrouter.provider, openrouter.model) == "anthropic/claude-sonnet-5"
    client = module.client_from_args(openrouter)
    assert isinstance(client, module.OpenRouterBatchClient)
    assert client.base_url == "https://openrouter.ai"


def test_status_and_download_infer_openrouter_from_state(monkeypatch):
    module = load_module()
    parser = module.build_parser()
    args = parser.parse_args(["status"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    state = {"schema_version": 1, "provider": "openrouter"}
    assert module.resolved_provider(args, state) == "openrouter"
    client = module.client_from_args(args, state)
    assert isinstance(client, module.OpenRouterBatchClient)


def test_wait_for_openrouter_batch_shows_progress_and_retries_not_found(capsys):
    module = load_module()
    client = module.OpenRouterBatchClient(
        "or-secret", "https://openrouter.ai", transport=lambda *args: b"{}"
    )
    responses = [
        RuntimeError('Anthropic HTTP 404: {"error":{"message":"not found"}}'),
        {
            "id": "batch_or_123",
            "status": "completed",
            "request_counts": {"total": 10, "completed": 10, "failed": 0},
            "results": [{}],
        },
    ]

    def get_batch(batch_id):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    client.get_batch = get_batch
    status = module.wait_for_batch(
        client, "batch_or_123", 0.001, 1.0, sample_count=10
    )
    assert status["status"] == "completed"
    assert responses == []
    progress = capsys.readouterr().err
    assert "propagazione" in progress
    assert "completed" in progress
    assert "10/10" in progress
