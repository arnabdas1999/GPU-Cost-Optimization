"""The JSON action protocol shared by every configuration in a comparison.

Parsing rule (fixed for all runs): the reply must contain a JSON object. The
first decodable object is used, starting from the first "{"; surrounding text
is ignored but counted. A reply cut off by the output-token cap is rejected
without parsing, even if a prefix happens to decode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "json-actions-v1"

MAX_PATH_CHARS = 256

ACTION_FIELDS: dict[str, dict[str, type]] = {
    "read_file": {"path": str},
    "edit_file": {"path": str, "old": str, "new": str},
    "write_file": {"path": str, "content": str},
    "run_tests": {},
    "finish": {},
}

SYSTEM_PROMPT = """\
You are a software engineer fixing a bug in a small Python repository.
You act by replying with exactly one JSON object per reply, and nothing else.

Actions:
{{"action": "read_file", "path": "src/example.py"}}
{{"action": "edit_file", "path": "src/example.py", "old": "exact existing text", "new": "replacement text"}}
    Replaces exactly one occurrence of "old" literally. No indentation is added.
{{"action": "write_file", "path": "src/example.py", "content": "complete new file contents"}}
{{"action": "run_tests"}}
    Runs the public tests and shows their output.
{{"action": "finish"}}
    Submits your work. Nothing can be changed afterwards.

Rules:
- Paths are relative to the repository root.
- The task message already contains the initial source files and public tests.
- Implement the complete behavior in the task and docstrings. Public tests are
  examples, not the complete specification.
- For a short file that fits the output limit, write_file can replace the whole
  file with correctly indented Python. Do not include Markdown in file contents.
- For edit_file, include exact leading whitespace in both old and new. Every
  line after a JSON newline needs its own indentation; it is not inherited from
  the first line. Replace a complete block when changing control flow.
- If an edit fails or tests report a syntax error, read the current file before
  repairing it. Do not repeat an unchanged edit or the same missing old text.
- Run the public tests after your fix. Repair failures before finish, and check
  the full docstring even when public tests pass.
- You have at most {max_steps} replies in total, including the finish reply.
- Each reply is limited to {max_output_tokens} tokens. A reply that is cut off
  is rejected, so keep edits small when a complete file would not fit.
- Hidden tests check the behavior described in the task and docstrings, not
  only the public tests.
"""


class ProtocolError(ValueError):
    """A reply that does not form a valid action. Reported back to the model."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Action:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    extra_text_chars: int = 0


def system_prompt(max_steps: int, max_output_tokens: int) -> str:
    return SYSTEM_PROMPT.format(max_steps=max_steps, max_output_tokens=max_output_tokens)


def parse_action(text: str, finish_reason: str | None, max_payload_bytes: int) -> Action:
    if finish_reason == "length":
        raise ProtocolError(
            "truncated",
            "Your reply was cut off at the output-token limit and was not executed. "
            "Reply with a smaller action; use edit_file for targeted changes.",
        )
    start = text.find("{")
    if start < 0:
        raise ProtocolError("no_json", "Your reply did not contain a JSON object. Reply with exactly one JSON action.")
    try:
        obj, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid_json", f"Your reply was not valid JSON ({exc.msg}). Reply with exactly one JSON action.") from None
    if not isinstance(obj, dict):
        raise ProtocolError("invalid_json", "The JSON value must be an object with an \"action\" field.")
    name = obj.get("action")
    if name not in ACTION_FIELDS:
        raise ProtocolError("unknown_action", f"Unknown action {name!r}. Valid actions: {', '.join(ACTION_FIELDS)}.")
    args: dict[str, Any] = {}
    for key, expected in ACTION_FIELDS[name].items():
        value = obj.get(key)
        if not isinstance(value, expected):
            raise ProtocolError("bad_arguments", f"Action {name} requires a string field {key!r}.")
        args[key] = value
    if "path" in args and len(args["path"]) > MAX_PATH_CHARS:
        raise ProtocolError("bad_arguments", f"Paths are limited to {MAX_PATH_CHARS} characters.")
    payload = sum(len(v.encode("utf-8")) for k, v in args.items() if k != "path")
    if payload > max_payload_bytes:
        raise ProtocolError("payload_too_large", f"Action payload exceeds {max_payload_bytes} bytes.")
    extra = len(text[:start].strip()) + len(text[end:].strip())
    return Action(name=name, args=args, extra_text_chars=extra)
