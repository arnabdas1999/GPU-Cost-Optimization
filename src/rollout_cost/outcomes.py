"""Episode outcome fields and the report categories derived from them.

Three independent fields are recorded for every episode:

* ``stop_reason``: why the agent loop ended.
* ``infrastructure_status``: whether the model server, tool executor or harness
  failed during the agent loop.
* ``grader_outcome``: what the independent grader concluded about the final
  workspace.

Report categories are derived from these, never recorded directly, so an
episode cannot be counted as both "successful" and "budget-exhausted".
"""

from __future__ import annotations

STOP_REASONS = (
    "finished",  # the model issued the finish action
    "max_steps",  # the step budget was used without finish
    "context_exhausted",  # the next prompt would not fit the context window
    "wall_timeout",  # the wall-time backstop expired (scheduling-sensitive)
    "infra_error",  # the loop was stopped by an infrastructure failure
)
BUDGET_STOPS = frozenset({"max_steps", "context_exhausted", "wall_timeout"})

INFRA_STATUSES = (
    "ok",
    "model_server_error",
    "model_timeout",
    "tool_executor_error",
    "harness_error",
)

GRADER_OUTCOMES = (
    "pass",
    "fail",
    "grader_error",  # the grader itself could not run: not a model-quality signal
    "not_graded",
)

CATEGORIES = (
    "success",
    "failed_after_finish",
    "failed_budget_exhausted",
    "infra_error",
)


def derive_category(stop_reason: str, infrastructure_status: str, grader_outcome: str) -> str:
    """Map the three recorded fields to exactly one report category.

    * Any infrastructure failure, or a grader that could not assess the final
      state, makes the episode invalid (``infra_error``) regardless of grade.
    * Otherwise the grade decides success. A correct patch that ran out of
      budget is a success; the stop-reason cross-tab shows how many did.
    * Unsuccessful valid episodes are split by whether the model finished.
    """
    if infrastructure_status != "ok" or grader_outcome in ("grader_error", "not_graded"):
        return "infra_error"
    if grader_outcome == "pass":
        return "success"
    if stop_reason in BUDGET_STOPS:
        return "failed_budget_exhausted"
    return "failed_after_finish"


def is_valid(category: str) -> bool:
    return category != "infra_error"
