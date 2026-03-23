import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_community.chat_models import MiniMaxChat
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def create_model():
    provider = os.getenv("MODEL_PROVIDER", "deepseek").lower()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Please set DEEPSEEK_API_KEY before running this script.")

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
                "Please set MINIMAX_API_KEY and MINIMAX_GROUP_ID before running this script."
            )

        return MiniMaxChat(
            minimax_api_key=minimax_api_key,
            minimax_group_id=minimax_group_id,
            model=os.getenv("MINIMAX_MODEL", "abab6.5-chat"),
            base_url=os.getenv(
                "MINIMAX_BASE_URL",
                "https://api.minimax.chat/v1/text/chatcompletion_v2",
            ),
            temperature=0,
        )

    raise RuntimeError("Unsupported MODEL_PROVIDER. Use 'deepseek' or 'minimax'.")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    model = create_model()

    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful weather assistant.",
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    print(response)


if __name__ == "__main__":
    main()
