from typing import Dict, Any


GENPLAN_ERROR_TYPES = [
    "timeout",
    "python-exception",
    "output-not-plan",

    "operator-syntax-invalid",      # still used?
    "operator-semantics-invalid",

    "type-mistake",
    "action-formatting",
    "wrong-number-parameters",
    "wrong-action",
    "undefined-objects",
    "unsat-preconditions",
    "not-reached-goal",
    "function-not-generated",
    "unsafe-code"
]


def initialize_task_metrics() -> dict:

    metrics = {err: 0 for err in GENPLAN_ERROR_TYPES}

    return metrics

