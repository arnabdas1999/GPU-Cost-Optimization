import pytest

from env_config import apply_env_overrides


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on", "  true  "])
def test_truthy_strings(raw):
    result = apply_env_overrides({"debug": False}, {"APP_DEBUG": raw})
    assert result["debug"] is True


@pytest.mark.parametrize("raw", ["0", "false", "False", "NO", "off", " off\n"])
def test_falsy_strings(raw):
    result = apply_env_overrides({"debug": True}, {"APP_DEBUG": raw})
    assert result["debug"] is False


@pytest.mark.parametrize("raw", ["", "maybe", "2", "enabled"])
def test_invalid_bool_raises(raw):
    with pytest.raises(ValueError):
        apply_env_overrides({"debug": True}, {"APP_DEBUG": raw})


def test_int_parsing_and_invalid_int():
    assert apply_env_overrides({"workers": 1}, {"APP_WORKERS": "16"})["workers"] == 16
    with pytest.raises(ValueError):
        apply_env_overrides({"workers": 1}, {"APP_WORKERS": "many"})


def test_int_result_is_not_bool():
    result = apply_env_overrides({"workers": 1}, {"APP_WORKERS": "1"})
    assert type(result["workers"]) is int


def test_float_and_string_values():
    config = {"ratio": 0.5, "name": "svc"}
    environ = {"APP_RATIO": "0.25", "APP_NAME": "worker"}
    assert apply_env_overrides(config, environ) == {"ratio": 0.25, "name": "worker"}


def test_missing_and_unknown_variables():
    config = {"debug": False, "workers": 2}
    assert apply_env_overrides(config, {"APP_OTHER": "1", "DEBUG": "true"}) == config


def test_custom_prefix():
    assert apply_env_overrides({"debug": False}, {"SVC_DEBUG": "yes"}, prefix="SVC_") == {"debug": True}


def test_config_is_not_modified():
    config = {"debug": True, "workers": 2}
    apply_env_overrides(config, {"APP_DEBUG": "off", "APP_WORKERS": "3"})
    assert config == {"debug": True, "workers": 2}
