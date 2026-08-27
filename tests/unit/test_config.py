import pytest
from pydantic import ValidationError

from adaptiveguard.io._config_loader import expand_environment
from adaptiveguard.io.config import RunSettings, load_config


def test_default_yaml_is_valid() -> None:
    config = load_config("configs/default.yaml")
    assert config["scb"]["theta_low"] == 0.2
    assert config["pft"]["budget_max"] == 10


def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RunSettings.model_validate({"scb": {"theta_low": 0.8, "theta_high": 0.7}})


def test_environment_expansion_recurses_through_mappings_and_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ADAPTIVEGUARD_TEST_ROOT", "workspace")

    expanded = expand_environment(
        {
            "root": "${ADAPTIVEGUARD_TEST_ROOT}/data",
            "items": ["${ADAPTIVEGUARD_TEST_ROOT}/a", {"name": "${MISSING_TEST_VALUE}"}],
        }
    )

    assert expanded == {
        "root": "workspace/data",
        "items": ["workspace/a", {"name": "${MISSING_TEST_VALUE}"}],
    }


def test_non_mapping_yaml_root_is_rejected(tmp_path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="configuration root must be a mapping"):
        load_config(path)
