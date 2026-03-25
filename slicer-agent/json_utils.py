from __future__ import annotations

import json
import re
from typing import Any, Dict


def extract_json_object(text: str) -> str:
    """
    Try to extract the JSON object part from an LLM output.
    Rule: take the substring from the first '{' to the last '}'.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


def safe_json_loads(text: str) -> Dict[str, Any]:
    obj_text = extract_json_object(text)
    return json.loads(obj_text)


def normalize_actions(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compatibility normalization for the `actions[]` array:
    - prefer `action` + `params` (Orca ActionExecutor format)
    - if older keys `op` + `args` are present, map them over
    - ensure `params` is always an object
    """
    actions = plan.get("actions")
    if not isinstance(actions, list):
        return plan

    for a in actions:
        if not isinstance(a, dict):
            continue

        if "action" not in a and "op" in a:
            a["action"] = a.pop("op")

        if "params" not in a and "args" in a:
            a["params"] = a.pop("args")

        params = a.get("params")
        if params is None:
            a["params"] = {}
    return plan

