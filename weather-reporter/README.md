# Weather Reporter

Minimal runnable LangChain weather agent based on the official quickstart.

## Prerequisites

- Python 3.12+
- `uv` installed
- Anthropic API key

## Setup

From the repository root:

```bash
uv sync
```

Set your API key:

```powershell
$env:ANTHROPIC_API_KEY="your_api_key_here"
```

## Run

From the repository root:

```bash
uv run python weather-reporter/agent.py
```

The script creates an agent, registers a weather tool, and runs one sample user query.
