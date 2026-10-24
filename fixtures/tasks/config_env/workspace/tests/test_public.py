from env_config import apply_env_overrides


def test_false_string_disables_flag():
    assert apply_env_overrides({"debug": True}, {"APP_DEBUG": "false"}) == {"debug": False}


def test_int_override():
    assert apply_env_overrides({"workers": 2}, {"APP_WORKERS": "8"}) == {"workers": 8}
