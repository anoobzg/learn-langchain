from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Field, ValidationError


RiskLevel = Literal["low", "medium", "high"]


class _BaseArgs(BaseModel):
    """
    Common args model behavior:
    - allow unknown keys inside params (OrcaSlicer ignores unknown keys)
    """

    model_config = {"extra": "ignore"}


class LoadArgs(_BaseArgs):
    file_path: str = Field(..., description="Path to a local input file")


class LoadFromNetArgs(_BaseArgs):
    url: str = Field(..., description="HTTP(S) URL to fetch the input file")
    output_filename: Optional[str] = Field(
        None, description="Optional override filename after downloading"
    )


class SliceCurrentArgs(_BaseArgs):
    # No action-specific parameters for `slice_current`.
    pass


class SliceAllArgs(_BaseArgs):
    # No action-specific parameters for `slice_all`.
    pass


ArgsModel = Type[BaseModel]


@dataclass(frozen=True)
class ActionSpec:
    action: str
    args_model: ArgsModel
    risk_level: RiskLevel
    description: str

    def validate_params(self, params: Dict[str, Any]) -> BaseModel:
        try:
            return self.args_model.model_validate(params)
        except ValidationError as e:
            raise ValueError(str(e)) from e


ACTION_REGISTRY: Dict[str, ActionSpec] = {
    "load": ActionSpec(
        action="load",
        args_model=LoadArgs,
        risk_level="medium",
        description="Load local input file",
    ),
    "load_from_net": ActionSpec(
        action="load_from_net",
        args_model=LoadFromNetArgs,
        risk_level="high",
        description="Load input file from HTTP(S) URL",
    ),
    "slice_current": ActionSpec(
        action="slice_current",
        args_model=SliceCurrentArgs,
        risk_level="low",
        description="Slice current loaded model(s)",
    ),
    "slice_all": ActionSpec(
        action="slice_all",
        args_model=SliceAllArgs,
        risk_level="low",
        description="Slice all available models",
    ),
}


def validate_action_params(action: str, params: Dict[str, Any]) -> BaseModel:
    spec = ACTION_REGISTRY.get(action)
    if not spec:
        raise ValueError(f"Unsupported action: {action}")
    return spec.validate_params(params)


def list_supported_ops() -> List[str]:
    return sorted(ACTION_REGISTRY.keys())

