from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

from schemas import PlanRequest, PlanResponse
from planner import plan_for_request


app = FastAPI(title="slicer-agent", version="1.0.0")


_UI_PATH = Path(__file__).resolve().parent / "ui.html"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # Serve the minimal test UI for OrcaSlicer integration debugging.
    html = _UI_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/v1/plan", response_model=PlanResponse)
def create_plan(req: PlanRequest) -> PlanResponse:
    try:
        resp = plan_for_request(req)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return resp

