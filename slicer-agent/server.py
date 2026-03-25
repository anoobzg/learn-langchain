from __future__ import annotations

import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

from schemas import PlanRequest, PlanResponse
from planner import plan_for_request


app = FastAPI(title="slicer-agent", version="1.0.0")

_LOG_LEVEL = os.getenv("SLICER_AGENT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("slicer-agent")

_UI_PATH = Path(__file__).resolve().parent / "ui.html"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # Serve the minimal test UI for OrcaSlicer integration debugging.
    html = _UI_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/v1/plan", response_model=PlanResponse)
def create_plan(req: PlanRequest) -> PlanResponse:
    try:
        logger.info(
            "Received plan request request_id=%s user_text=%r context_keys=%s",
            req.request_id,
            req.user_text[:200],
            list((req.context or {}).keys()),
        )
        resp = plan_for_request(req)
    except Exception as e:  # noqa: BLE001
        logger.exception("Unhandled exception while planning request_id=%s", req.request_id)
        raise HTTPException(status_code=500, detail=str(e)) from e

    logger.info(
        "Plan response request_id=%s plan_id=%s used_actions=%d requires_confirmation=%s risk_level=%s summary=%r",
        resp.request_id,
        resp.plan_id,
        len(resp.actions),
        resp.requires_confirmation,
        resp.risk_level,
        resp.summary[:200],
    )
    if resp.warnings:
        logger.warning("Plan warnings request_id=%s warnings=%s", resp.request_id, resp.warnings)
    if resp.errors:
        logger.error("Plan errors request_id=%s errors=%s", resp.request_id, resp.errors)
    return resp

