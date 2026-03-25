import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SLICER_AGENT_DIR = REPO_ROOT / "slicer-agent"
sys.path.insert(0, str(SLICER_AGENT_DIR))

os.environ["SLICER_AGENT_MOCK"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from server import app  # noqa: E402


def test_homepage_serves_ui() -> None:
    client = TestClient(app)

    res = client.get("/")
    assert res.status_code == 200
    html = res.text

    assert "slicer-agent test web" in html
    assert "btnPlan" in html
    assert "/v1/plan" in html

