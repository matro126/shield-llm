from types import SimpleNamespace

import shield.training.runner as training_runner


def test_prepare_clinical_validation_records_preserves_dense_target_format():
    prepare = getattr(
        training_runner, "prepare_clinical_validation_records", None
    )
    assert prepare is not None
    record = {
        "id": "normal",
        "messages": [
            {"role": "system", "content": "Report system"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "normal/frontal.png"},
                    {"type": "image", "image": "normal/lateral.png"},
                    {"type": "text", "text": "Describe the radiograph"},
                ],
            },
            {"role": "assistant", "content": "Findings:\nNormal"},
        ],
        "images": ["normal/frontal.png", "normal/lateral.png"],
        "factors": {
            "diagnostic_category": ["No Finding"],
            "projection": "frontal+lateral",
            "views": ["frontal", "lateral"],
            "task_type": "report_generation",
        },
    }
    cfg = SimpleNamespace(
        views="frontal_lateral",
        clinical_target_format="dense_binary",
    )
    clinical, stats = prepare([record], cfg)
    assert stats["target_format"] == "dense_binary"
    assert "Pneumothorax: Absent" in clinical[0]["messages"][-1]["content"]
    assert "Support Devices: Absent" in clinical[0]["messages"][-1]["content"]
    assert "No Finding:" not in clinical[0]["messages"][-1]["content"]
