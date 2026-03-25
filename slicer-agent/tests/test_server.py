import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SLICER_AGENT_DIR = REPO_ROOT / "slicer-agent"
sys.path.insert(0, str(SLICER_AGENT_DIR))

os.environ["SLICER_AGENT_MOCK"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from server import app  # noqa: E402
from schemas import PlanRequest  # noqa: E402


def test_plan_endpoint_returns_valid_response() -> None:
    client = TestClient(app)

    req = PlanRequest(
        request_id="r1",
        user_text="load file_path=C:\\models\\test.stl slice current",
        context={},
        client_version="test",
    )

    res = client.post("/v1/plan", json=req.model_dump())
    assert res.status_code == 200, res.text

    data = res.json()
    assert data["request_id"] == "r1"
    assert data["requires_confirmation"] is True
    assert isinstance(data["actions"], list)

