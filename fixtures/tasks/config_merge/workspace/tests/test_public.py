from config_merge import merge_config


def test_nested_override_keeps_sibling_defaults():
    defaults = {"db": {"host": "localhost", "port": 5432}}
    merged = merge_config(defaults, {"db": {"port": 6543}})
    assert merged == {"db": {"host": "localhost", "port": 6543}}


def test_explicit_false_overrides_true():
    assert merge_config({"debug": True}, {"debug": False}) == {"debug": False}
