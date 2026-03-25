import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SLICER_AGENT_DIR = REPO_ROOT / "slicer-agent"
sys.path.insert(0, str(SLICER_AGENT_DIR))

from schemas import PlanRequest  # noqa: E402
from planner import plan_for_request  # noqa: E402


def test_mock_planner_basic_parse() -> None:
    os.environ["SLICER_AGENT_MOCK"] = "1"

    req = PlanRequest(
        request_id="r1",
        user_text="load file_path=C:\\models\\test.stl slice current",
        context={},
        client_version="test",
    )
    resp = plan_for_request(req)
    ops = [a.action for a in resp.actions]

    assert ops == ["load", "slice_current"]
    assert resp.requires_confirmation is True


def test_mock_planner_load_from_net_and_slice_all() -> None:
    os.environ["SLICER_AGENT_MOCK"] = "1"

    req = PlanRequest(
        request_id="r1",
        user_text="load_from_net url=https://example.com/model.3mf output_filename=out.3mf slice all",
        context={},
        client_version="test",
    )
    resp = plan_for_request(req)
    ops = [a.action for a in resp.actions]

    assert ops == ["load_from_net", "slice_all"]
    assert resp.requires_confirmation is True


def test_dsl_validation_rejects_unknown_action() -> None:
    from dsl import validate_action_params

    try:
        validate_action_params("unknown_action", {})
    except ValueError as e:
        assert "Unsupported action" in str(e)
    else:
        raise AssertionError("Expected validation error")

