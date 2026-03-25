from __future__ import annotations

from typing import List

from dsl import validate_action_params
from schemas import ActionValidationError, ActionValidationResult, OrcaAction, PlanResponse


def validate_plan(plan: PlanResponse) -> ActionValidationResult:
    """
    For OrcaSlicer to perform a second validation step:
    - `action` must be in the whitelist
    - `params` must satisfy the corresponding field constraints
    """
    validated_actions: List[OrcaAction] = []
    errors: List[ActionValidationError] = []

    for a in plan.actions:
        try:
            validate_action_params(a.action, a.params)
            validated_actions.append(a)
        except Exception as e:  # noqa: BLE001
            errors.append(
                ActionValidationError(action=a.action, message=str(e))
            )

    ok = len(errors) == 0
    return ActionValidationResult(ok=ok, errors=errors, validated_actions=validated_actions)

