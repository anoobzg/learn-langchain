from __future__ import annotations

from fastapi import FastAPI, HTTPException

from schemas import PlanRequest, PlanResponse
from planner import plan_for_request


app = FastAPI(title="slicer-agent", version="1.0.0")


@app.post("/v1/plan", response_model=PlanResponse)
def create_plan(req: PlanRequest) -> PlanResponse:
    try:
        resp = plan_for_request(req)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return resp

