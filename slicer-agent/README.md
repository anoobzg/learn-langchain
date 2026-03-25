# slicer-agent

This is an independent HTTP AI Agent service that converts natural-language requests
from OrcaSlicer (or its AI dialog) into a JSON action plan in a **closed DSL**
("operations list + parameters").

## Goals
- The agent only does "intent understanding/planning" and does not manipulate OrcaSlicer internal state.
- OrcaSlicer is responsible for: validation (whitelist + parameter ranges + state preconditions),
  preview/confirmation, execution, and rollback.

## HTTP API (MVP)
- `POST /v1/plan`
  - Request: `PlanRequest` (`user_text` + `context`)
  - Response: `PlanResponse` (`actions[]`, each action is a whitelisted `action` + `params`)

## Quick Run
Configure LLM credentials (e.g. `.env` or environment variables) and provide:
- `MODEL_PROVIDER`: `deepseek` or `minimax`
- API keys: `DEEPSEEK_API_KEY` or `MINIMAX_API_KEY` + `MINIMAX_GROUP_ID`

You can also use Mock mode (no LLM call) for integration testing:
- `SLICER_AGENT_MOCK=1`

Then start:
- Entry is `server.py` (recommended: run inside `slicer-agent/`: `uvicorn server:app --reload`)

Test web UI:
- Visit `GET /` in your browser (use `SLICER_AGENT_MOCK=1` for predictable behavior)

## JSON Contract
The schema models are in `schemas.py`, and the whitelist DSL is defined in `dsl.py`.

