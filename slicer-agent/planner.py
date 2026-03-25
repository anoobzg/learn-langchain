from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.chat_models import MiniMaxChat
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from dsl import ACTION_REGISTRY, validate_action_params
from json_utils import normalize_actions, safe_json_loads
from schemas import OrcaAction, PlanRequest, PlanResponse


def _create_llm():
    """
    Uses the same environment variable conventions as `weather-reporter`:
    - MODEL_PROVIDER=deepseek|minimax
    - deepseek: DEEPSEEK_API_KEY + optional DEEPSEEK_MODEL/DEEPSEEK_BASE_URL
    - minimax: MINIMAX_API_KEY + MINIMAX_GROUP_ID + optional MINIMAX_MODEL/MINIMAX_BASE_URL
    """
    provider = os.getenv("MODEL_PROVIDER", "deepseek").lower()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Please set DEEPSEEK_API_KEY before running slicer-agent.")
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=0,
        )

    if provider == "minimax":
        minimax_api_key = os.getenv("MINIMAX_API_KEY")
        minimax_group_id = os.getenv("MINIMAX_GROUP_ID")
        if not minimax_api_key or not minimax_group_id:
            raise RuntimeError(
                "Please set MINIMAX_API_KEY and MINIMAX_GROUP_ID before running slicer-agent."
            )
        use_model = os.getenv("MINIMAX_MODEL", "abab6.5-chat")
        return MiniMaxChat(
            minimax_api_key=minimax_api_key,
            minimax_group_id=minimax_group_id,
            model=use_model,
            base_url=os.getenv(
                "MINIMAX_BASE_URL",
                "https://api.minimax.chat/v1/text/chatcompletion_v2",
            ),
            temperature=0,
        )

    raise RuntimeError("Unsupported MODEL_PROVIDER. Use 'deepseek' or 'minimax'.")


def _build_system_prompt() -> str:
    # Enforce the closed DSL:
    # - `action` must be from ACTION_REGISTRY
    # - `params` must match the corresponding fields
    supported_actions = ", ".join(sorted(ACTION_REGISTRY.keys()))
    return (
        "You are an OrcaSlicer intent planner."
        " Your only job is to convert the user's natural-language request into a JSON list of actions in a closed DSL."
        " Do not output any extra text; output only a JSON object.\n"
        f" Whitelisted actions (action) can only be: {supported_actions}."
        " Each action must include `action` and `params` (params must be an object)."
        " Only use parameters relevant to the selected action (e.g. `file_path`, `url`, `output_filename`)."
        " If the request is outside the supported whitelist, return `actions` as an empty array and explain in errors/warnings."
        " requires_confirmation must be true."
    )


def _build_user_prompt(req: PlanRequest) -> str:
    ctx = req.context or {}
    return (
        f"request_id: {req.request_id}\n"
        f"user_text: {req.user_text}\n"
        f"context: {ctx}\n\n"
        "Output JSON based on user_text and context:"
        "{schema_version:'1.0', request_id, plan_id, requires_confirmation:true, actions:[{action,params}], risk_level, summary, warnings, errors}."
    )


class MockPlanner:
    """
    Integration/no-key mode: use rules to generate stable JSON.
    """

    def plan(self, req: PlanRequest) -> PlanResponse:
        user_text = (req.user_text or "").strip()
        actions: List[OrcaAction] = []

        lowered = user_text.lower()

        # Load local file: support patterns like:
        # - "load file_path=/path/to/model.stl"
        # - "path=/path/to/model.3mf"
        m = re.search(
            r"(?:file_path|path)\s*[:=]\s*(\S+\.(?:stl|3mf|obj|step|stp|ply|amf))",
            user_text,
            re.IGNORECASE,
        )
        if m:
            actions.append(OrcaAction(action="load", params={"file_path": m.group(1)}))

        # Load from URL: support patterns like:
        # - "load_from_net url=https://.../model.3mf output_filename=out.3mf"
        m = re.search(r"url\s*[:=]\s*(\S+)", user_text, re.IGNORECASE)
        if m and ("http://" in lowered or "https://" in lowered):
            url = m.group(1)
            out_m = re.search(
                r"output_filename\s*[:=]\s*(\S+)", user_text, re.IGNORECASE
            )
            args: Dict[str, Any] = {"url": url}
            if out_m:
                args["output_filename"] = out_m.group(1)
            actions.append(OrcaAction(action="load_from_net", params=args))

        # Slice ops
        if re.search(r"\bslice\s+all\b|all\s+models\s+slice\b", lowered):
            actions.append(OrcaAction(action="slice_all", params={}))
        elif re.search(r"\bslice\s+(current|this)\b", lowered) or re.search(
            r"\bslice\s+now\b", lowered
        ):
            actions.append(OrcaAction(action="slice_current", params={}))
        else:
            # Default to slice_current if we detected a load but not a slice instruction.
            if actions and not any(a.action.startswith("slice_") for a in actions):
                actions.append(OrcaAction(action="slice_current", params={}))

        # If no actions were recognized, return empty actions and mark a warning.
        warnings: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        risk_level: RiskLevel = "low"
        summary = "Generated an action plan"
        if not actions:
            warnings.append(
                {
                    "code": "no_match",
                    "message": "No executable actions were recognized from the text.",
                }
            )
            summary = "Could not parse the request into executable actions"
            risk_level = "medium"

        # Risk level adjustment
        if any(a.action in {"load", "load_from_net"} for a in actions):
            risk_level = (
                "medium" if any(a.action == "load" for a in actions) else "high"
            )

        # Validate action/params against the whitelist before returning (prevents mock bugs).
        validated_actions: List[OrcaAction] = []
        for a in actions:
            validate_action_params(a.action, a.params)
            validated_actions.append(a)

        return PlanResponse(
            request_id=req.request_id,
            plan_id=str(uuid.uuid4()),
            requires_confirmation=True,
            actions=validated_actions,
            risk_level=risk_level,  # type: ignore[arg-type]
            summary=summary,
            warnings=warnings,  # type: ignore[arg-type]
            errors=errors,  # type: ignore[arg-type]
        )


class LLMPlanner:
    def __init__(self):
        load_dotenv()
        self.llm = _create_llm()

    def plan(self, req: PlanRequest) -> PlanResponse:
        # Ask the model to output JSON only, then validate with Pydantic + whitelist checks.
        system = _build_system_prompt()
        user = _build_user_prompt(req)

        raw = self.llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        text = getattr(raw, "content", None) or str(raw)

        # Allow models to output extra text: extract JSON when possible,
        # otherwise return a structured error response.
        try:
            plan_dict = safe_json_loads(text)
            plan_dict = normalize_actions(plan_dict)
            resp = PlanResponse.model_validate(plan_dict)
        except Exception as e:  # noqa: BLE001
            return PlanResponse(
                request_id=req.request_id,
                plan_id=str(uuid.uuid4()),
                requires_confirmation=True,
                actions=[],
                risk_level="high",  # type: ignore[arg-type]
                summary="Agent output could not be parsed as valid JSON/Schema",
                warnings=[],
                errors=[{"code": "model_output_invalid", "message": str(e)}],  # type: ignore[arg-type]
            )

        # Whitelist-validate action/params again to ensure closed DSL compliance.
        validated_actions: List[OrcaAction] = []
        errors: List[Dict[str, Any]] = []
        for a in resp.actions:
            try:
                validate_action_params(a.action, a.params)
                validated_actions.append(a)
            except Exception as e:  # noqa: BLE001
                errors.append(
                    {"code": "invalid_params", "message": f"{a.action}: {e}"}
                )

        if errors:
            # If validation failed, clear actions to avoid ambiguous execution on Orca.
            return PlanResponse(
                request_id=req.request_id,
                plan_id=resp.plan_id,
                requires_confirmation=True,
                actions=[],
                risk_level="high",  # type: ignore[arg-type]
                summary="Generated action args did not pass whitelist validation",
                warnings=resp.warnings,
                errors=errors,  # type: ignore[arg-type]
            )

        return PlanResponse(
            request_id=req.request_id,
            plan_id=resp.plan_id,
            requires_confirmation=True,
            actions=validated_actions,
            risk_level=resp.risk_level,
            summary=resp.summary,
            warnings=resp.warnings,
            errors=resp.errors,
        )


def plan_for_request(req: PlanRequest) -> PlanResponse:
    """
    Plan with deterministic fallback:
    - Always try the lightweight heuristic parser first.
    - If it produces any actions, return them immediately.
    - Otherwise, if SLICER_AGENT_MOCK=1, return the empty heuristic result.
    - Otherwise, call the LLM planner.
    """
    heuristic_resp = MockPlanner().plan(req)
    if heuristic_resp.actions:
        return heuristic_resp

    if os.getenv("SLICER_AGENT_MOCK", "").strip() == "1":
        return heuristic_resp

    return LLMPlanner().plan(req)

