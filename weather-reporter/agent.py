import os

from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Please set ANTHROPIC_API_KEY before running this script.")

    agent = create_agent(
        model="claude-sonnet-4-5-20250929",
        tools=[get_weather],
        system_prompt="You are a helpful weather assistant.",
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    print(response)


if __name__ == "__main__":
    main()
