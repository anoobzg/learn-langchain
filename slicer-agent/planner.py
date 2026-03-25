from __future__ import annotations

import logging
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
from schemas import OrcaAction, PlanRequest, PlanResponse, RiskLevel

logger = logging.getLogger("slicer-agent")


KNOWN_MODEL_EXTS = (".stl", ".3mf", ".obj", ".step", ".stp", ".ply", ".amf")


def _normalize_input_text(text: str) -> tuple[str, str]:
    """
    Returns:
    - raw: normalized punctuation only (keeps original values)
    - norm: lowercase + collapsed whitespace (for keyword matching)
    """
    raw = text.strip().replace("：", ":").replace("＝", "=").replace("／", "/")
    norm = re.sub(r"\s+", " ", raw).lower()
    return raw, norm


def _strip_trailing_punct(token: str) -> str:
    return token.rstrip(".,;:!?)]}\"'`")


def _extract_quoted_or_unquoted_kv_value(
    text: str,
    key: str,
) -> Optional[str]:
    """
    Extract value for patterns like:
    - key="..."
    - key='...'
    - key=`...`
    - key=token
    """
    # Order matters: quoted first, then unquoted token.
    pattern = rf"(?i)\b{re.escape(key)}\s*[:=]\s*(?:\"(?P<dq>[^\"]+)\"|'(?P<sq>[^']+)'|`(?P<bq>[^`]+)`|(?P<u>\S+))"
    m = re.search(pattern, text)
    if not m:
        return None
    val = m.group("dq") or m.group("sq") or m.group("bq") or m.group("u")
    if not val:
        return None
    return _strip_trailing_punct(val.strip())


def _extract_load_slots(user_text: str) -> tuple[Optional[Dict[str, Any]], float, List[Dict[str, Any]]]:
    """
    Returns:
    - best_load_action_params or None
    - confidence for the extracted load (0..1)
    - warnings
    """
    raw, norm = _normalize_input_text(user_text)
    warnings: List[Dict[str, Any]] = []

    file_val = _extract_quoted_or_unquoted_kv_value(raw, "file_path") or _extract_quoted_or_unquoted_kv_value(
        raw, "path"
    )
    url_val = _extract_quoted_or_unquoted_kv_value(raw, "url")
    output_filename = _extract_quoted_or_unquoted_kv_value(raw, "output_filename")

    # Fallback: support "load <path>" without explicit key.
    if not file_val and re.search(r"\b(load|open)\b", norm):
        m = re.search(
            r"(?i)\b(?:load|open)\s*(?:\"([^\"]+)\"|'([^']+)'|`([^`]+)`|(\S+))",
            raw,
        )
        if m:
            candidate = m.group(1) or m.group(2) or m.group(3) or m.group(4)
            candidate = _strip_trailing_punct(candidate.strip()) if candidate else None
            if candidate:
                looks_like_path = bool(
                    re.search(r"^[A-Za-z]:\\", candidate)
                    or "\\" in candidate
                    or "/" in candidate
                    or any(candidate.lower().endswith(ext) for ext in KNOWN_MODEL_EXTS)
                )
                if looks_like_path:
                    file_val = candidate
                    warnings.append(
                        {
                            "code": "load_path_without_key",
                            "message": "Loaded file path inferred without file_path/path key.",
                        }
                    )

    load_action_params: Optional[Dict[str, Any]] = None
    confidence = 0.0

    if url_val and url_val.lower().startswith(("http://", "https://")):
        load_action_params = {"url": url_val}
        if output_filename:
            load_action_params["output_filename"] = output_filename
        confidence = 0.95
    elif file_val:
        load_action_params = {"file_path": file_val}
        # If user provided a known extension, increase confidence.
        if any(file_val.lower().endswith(ext) for ext in KNOWN_MODEL_EXTS):
            confidence = 0.9
        else:
            confidence = 0.75
    else:
        return None, 0.0, warnings

    return load_action_params, confidence, warnings


def _extract_slice_intent(user_text: str) -> tuple[float, float, List[Dict[str, Any]]]:
    """
    Returns:
    - slice_current_confidence
    - slice_all_confidence
    - warnings
    """
    raw, norm = _normalize_input_text(user_text)
    warnings: List[Dict[str, Any]] = []

    slice_current_score = 0.0
    slice_all_score = 0.0

    # slice_all patterns
    if re.search(r"\b(slice\s*)?(all|every)\s*(models?|model)?\b", norm):
        slice_all_score = 0.9
    if re.search(r"\b(slice\s+all)\b", norm):
        slice_all_score = max(slice_all_score, 0.98)
    if re.search(r"\b(all\s+models)\b", norm):
        slice_all_score = max(slice_all_score, 0.95)

    # slice_current patterns
    if re.search(r"\b(slice\s*)?(current|this|active|selected)\b", norm):
        slice_current_score = 0.85
    if re.search(r"\bslice\s+(current|this|active|selected)\b", norm):
        slice_current_score = max(slice_current_score, 0.97)
    if re.search(r"\bslice\s+now\b", norm) or re.search(r"\bstart\s+slicing\b", norm):
        slice_current_score = max(slice_current_score, 0.78)

    # Ambiguity checks will be handled later by caller.
    if slice_all_score > 0 and slice_current_score > 0:
        warnings.append(
            {
                "code": "ambiguous_slice_target",
                "message": "Both 'current' and 'all' slice targets were detected; please specify which one you want.",
            }
        )

    return slice_current_score, slice_all_score, warnings


def _context_indicates_loaded_models(context: Dict[str, Any]) -> bool:
    if not context:
        return False

    candidates_int_keys = [
        "loaded_model_count",
        "active_model_count",
        "model_count",
        "models_loaded_count",
    ]
    for k in candidates_int_keys:
        v = context.get(k)
        if isinstance(v, int) and v > 0:
            return True
        # allow numeric strings
        if isinstance(v, str) and v.isdigit() and int(v) > 0:
            return True

    candidates_bool_keys = [
        "has_active_selection",
        "has_loaded_models",
        "models_loaded",
        "has_models",
    ]
    for k in candidates_bool_keys:
        v = context.get(k)
        if isinstance(v, bool) and v:
            return True
        if isinstance(v, str) and v.lower() in {"true", "1", "yes"}:
            return True

    return False


def _build_actions_from_slots(
    load_params: Optional[Dict[str, Any]],
    load_confidence: float,
    load_warnings: List[Dict[str, Any]],
    slice_current_score: float,
    slice_all_score: float,
    context: Dict[str, Any],
) -> tuple[List[OrcaAction], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns: actions, warnings, errors
    """
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    actions: List[OrcaAction] = []

    # Load action (non-destructive): include it if we parsed it.
    if load_params:
        if "url" in load_params:
            actions.append(OrcaAction(action="load_from_net", params=load_params))
        else:
            actions.append(OrcaAction(action="load", params=load_params))
        warnings.extend(load_warnings)

    # Slice decision (potentially destructive / user-visible): be conservative.
    SLICE_THRESHOLD = 0.75
    AMBIGUOUS_THRESHOLD = 0.85

    slice_current_hit = slice_current_score >= SLICE_THRESHOLD
    slice_all_hit = slice_all_score >= SLICE_THRESHOLD

    if slice_current_hit and slice_all_hit and slice_current_score >= AMBIGUOUS_THRESHOLD and slice_all_score >= AMBIGUOUS_THRESHOLD:
        errors.append(
            {
                "code": "ambiguous_slice_target",
                "message": "Both 'current' and 'all' slice targets were detected. Please specify exactly which one to run.",
            }
        )
        # Do not add any slice action under ambiguity.
        return actions, warnings, errors

    if slice_all_hit:
        actions.append(OrcaAction(action="slice_all", params={}))
        return actions, warnings, errors

    if slice_current_hit:
        actions.append(OrcaAction(action="slice_current", params={}))
        return actions, warnings, errors

    # No explicit slice instruction detected.
    if actions:
        # If we can't infer that slicing is safe, don't guess.
        if _context_indicates_loaded_models(context):
            # Loaded models exist: it's safe to default to slicing the current selection,
            # but we must warn the user that a default was applied.
            warnings.append(
                {
                    "code": "slice_defaulted_to_current",
                    "message": "Load detected, but slice target was not specified. Defaulting to 'slice current' because loaded-model context was provided. Please specify 'slice all' if you want all models instead.",
                }
            )
            actions.append(OrcaAction(action="slice_current", params={}))
        else:
            warnings.append(
                {
                    "code": "slice_not_specified_after_load",
                    "message": "Load detected, but slice target was not specified (and no loaded-model context was provided). Please specify 'slice current' or 'slice all'.",
                }
            )
        return actions, warnings, errors

    # Neither load nor slice intent found.
    warnings.append(
        {
            "code": "no_match",
            "message": "No executable actions were recognized from the text.",
        }
    )
    return actions, warnings, errors


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

        logger.info(
            "Heuristic parsing started request_id=%s user_text=%r",
            req.request_id,
            user_text[:200],
        )

        load_params, load_confidence, load_warnings = _extract_load_slots(user_text)
        slice_current_score, slice_all_score, slice_warnings = _extract_slice_intent(user_text)

        # Merge warnings from both extraction steps.
        combined_warnings: List[Dict[str, Any]] = []
        combined_warnings.extend(load_warnings)
        combined_warnings.extend(slice_warnings)

        actions, warnings, errors = _build_actions_from_slots(
            load_params=load_params,
            load_confidence=load_confidence,
            load_warnings=combined_warnings,
            slice_current_score=slice_current_score,
            slice_all_score=slice_all_score,
            context=req.context or {},
        )

        logger.info(
            "Heuristic slots request_id=%s load_conf=%.2f slice_current_conf=%.2f slice_all_conf=%.2f actions=%s warnings=%s errors=%s",
            req.request_id,
            load_confidence,
            slice_current_score,
            slice_all_score,
            [a.action for a in actions],
            warnings,
            errors,
        )

        risk_level: RiskLevel = "low"
        if any(a.action == "load_from_net" for a in actions):
            risk_level = "high"
        elif any(a.action == "load" for a in actions):
            risk_level = "medium"

        # If we have any explicit slice action, increase risk a bit.
        if any(a.action in {"slice_current", "slice_all"} for a in actions):
            risk_level = "high"

        summary = "Generated an action plan"
        if not actions and warnings:
            summary = "Could not parse the request into executable actions"

        # Validate action/params against the whitelist before returning.
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
        logger.info("LLMPlanner invoked request_id=%s", req.request_id)
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
            logger.warning(
                "LLMPlanner JSON/Schema parsing failed request_id=%s error=%r",
                req.request_id,
                str(e),
            )
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
        logger.info(
            "Plan selected heuristic-only request_id=%s actions=%s",
            req.request_id,
            [a.action for a in heuristic_resp.actions],
        )
        return heuristic_resp

    if os.getenv("SLICER_AGENT_MOCK", "").strip() == "1":
        logger.info(
            "Plan selected heuristic result (empty) due to mock mode request_id=%s",
            req.request_id,
        )
        return heuristic_resp

    logger.info(
        "Heuristic returned no actions; falling back to LLMPlanner request_id=%s",
        req.request_id,
    )
    return LLMPlanner().plan(req)

