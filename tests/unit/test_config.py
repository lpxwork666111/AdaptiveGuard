import pytest
from pydantic import ValidationError

from adaptiveguard.io.config import RunSettings, load_config


def test_default_yaml_is_valid() -> None:
    config = load_config("configs/default.yaml")
    assert config["scb"]["theta_low"] == 0.2
    assert config["pft"]["budget_max"] == 10


def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RunSettings.model_validate({"scb": {"theta_low": 0.8, "theta_high": 0.7}})
