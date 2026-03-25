from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["low", "medium", "high"]


class PlanRequest(BaseModel):
    request_id: str = Field(..., description="Client request ID, used for traceability")
    user_text: str = Field(..., description="User-provided natural language")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Minimal context provided by the client (printer/material/current parameter snapshot, etc.)",
    )
    client_version: Optional[str] = Field(None, description="Client version (for compatibility and debugging)")


class OrcaAction(BaseModel):
    action: str = Field(..., description="Whitelisted DSL action name (matches OrcaSlicer ActionExecutor)")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action parameters (matches OrcaSlicer ActionExecutor params)",
    )


class PlanWarning(BaseModel):
    code: str
    message: str


class PlanError(BaseModel):
    code: str
    message: str


class PlanResponse(BaseModel):
    schema_version: str = Field("1.0", description="Protocol schema version")
    request_id: str
    plan_id: str
    requires_confirmation: bool = Field(
        True,
        description="Whether user confirmation is required before execution. MVP suggests always true to avoid silent mistakes.",
    )
    actions: List[OrcaAction] = Field(default_factory=list)
    risk_level: RiskLevel = Field("low", description="Overall risk level (agent estimate, for UI display)")
    summary: str = Field("", description="Brief summary of the planned actions")
    warnings: List[PlanWarning] = Field(default_factory=list)
    errors: List[PlanError] = Field(default_factory=list)


class ActionValidationError(BaseModel):
    action: str
    message: str


class ActionValidationResult(BaseModel):
    ok: bool
    errors: List[ActionValidationError] = Field(default_factory=list)
    validated_actions: List[OrcaAction] = Field(default_factory=list)

