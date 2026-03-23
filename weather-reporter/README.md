# Weather Reporter

Minimal runnable LangChain weather agent supporting DeepSeek and MiniMax.

## Prerequisites

- Python 3.12+
- `uv` installed
- API key for your selected provider

## Setup

From the repository root:

```bash
uv sync
```

Select provider and set environment variables.

### Option A: DeepSeek

```powershell
$env:MODEL_PROVIDER="deepseek"
$env:DEEPSEEK_API_KEY="your_api_key_here"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

### Option B: MiniMax

```powershell
$env:MODEL_PROVIDER="minimax"
$env:MINIMAX_API_KEY="your_api_key_here"
$env:MINIMAX_GROUP_ID="your_group_id_here"
$env:MINIMAX_BASE_URL="https://api.minimax.chat/v1/text/chatcompletion_v2"
$env:MINIMAX_MODEL="abab6.5-chat"
```

## Run

From the repository root:

```bash
uv run python weather-reporter/agent.py
```

The script creates an agent, registers a weather tool, and runs one sample user query.
