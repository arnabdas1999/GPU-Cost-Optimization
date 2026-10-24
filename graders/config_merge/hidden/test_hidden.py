import copy

import pytest

from config_merge import merge_config


@pytest.mark.parametrize("falsy", [False, 0, 0.0, "", [], None])
def test_falsy_override_wins(falsy):
    assert merge_config({"value": "default"}, {"value": falsy}) == {"value": falsy}


def test_nested_false_override():
    defaults = {"cache": {"enabled": True, "ttl": 60}}
    assert merge_config(defaults, {"cache": {"enabled": False}}) == {"cache": {"enabled": False, "ttl": 60}}


def test_nested_zero_override_three_levels_deep():
    defaults = {"a": {"b": {"c": 5, "d": 1}}}
    assert merge_config(defaults, {"a": {"b": {"c": 0}}}) == {"a": {"b": {"c": 0, "d": 1}}}


def test_missing_keys_keep_defaults():
    defaults = {"x": 1, "y": {"z": 2}}
    assert merge_config(defaults, {}) == defaults


def test_new_keys_are_added():
    assert merge_config({"x": 1}, {"y": 0}) == {"x": 1, "y": 0}


def test_scalar_override_replaces_dict_default():
    assert merge_config({"x": {"nested": 1}}, {"x": 0}) == {"x": 0}


def test_dict_override_replaces_scalar_default():
    assert merge_config({"x": 5}, {"x": {"nested": False}}) == {"x": {"nested": False}}


def test_inputs_are_not_modified():
    defaults = {"a": {"b": 1, "c": True}, "d": [1, 2]}
    overrides = {"a": {"c": False}, "d": []}
    defaults_before, overrides_before = copy.deepcopy(defaults), copy.deepcopy(overrides)
    merge_config(defaults, overrides)
    assert defaults == defaults_before
    assert overrides == overrides_before
