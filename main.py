"""Standard Deep Agent: an orchestrator that runs specialist subagents in parallel.

Based on https://docs.langchain.com/oss/python/deepagents/overview
Run: .\\.venv\\Scripts\\python.exe main.py "your question"
"""

import os
import sys
from pathlib import Path

from deepagents import (
    GeneralPurposeSubagentProfile, HarnessProfile,
    create_deep_agent, register_harness_profile,
)
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))
MODEL = f"openai:{os.getenv('OPENAI_MODEL', 'gpt-5.5')}"


# A custom tool, as in the quickstart. The docstring tells the model what it does.
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# Each subagent gets its own prompt and context. The main agent reaches them
# through the built-in `task` tool, using `description` to pick one.
subagents = [
    {
        "name": "researcher",
        "description": "Delegate one focused research task: gather key facts on a single topic. "
                       "Use for parallel coverage of distinct subjects.",
        "system_prompt": "You are a research specialist. Complete your assigned topic only. "
                         "Return 3-5 bullet points with the most important facts. "
                         "No preamble, no follow-up questions, and do not delegate further work.",
        "tools": [get_weather],
    },
    {
        "name": "analyzer",
        "description": "Delegate one focused analysis task: pros, cons, and trade-offs for a single "
                       "option or approach. Use when comparing alternatives in parallel.",
        "system_prompt": "You are an analysis specialist. Complete your assigned option or angle only. "
                         "Return 3-5 bullet points covering strengths, weaknesses, and key trade-offs. "
                         "No preamble, no follow-up questions, and do not delegate further work.",
    },
    {
        "name": "writer",
        "description": "Delegate one focused summary task: turn findings on a single subtopic into "
                       "clear prose. Use when each parallel subagent should produce readable copy.",
        "system_prompt": "You are a writing specialist. Complete your assigned subtopic only. "
                         "Return a short paragraph or 3-5 bullets with the key takeaways. "
                         "No preamble, no follow-up questions, and do not delegate further work.",
    },
]

SYSTEM_PROMPT = """You are an orchestrator demo agent. Your job is to showcase parallel
subagents: spin up a small set of specialists at once, let each do a short piece of work,
then synthesize and finish.

For every user request:
1. Write one short sentence explaining which specialists you will run in parallel.
2. In the very next tool-calling turn, spawn exactly 2-3 subagents using multiple task()
   calls in that single turn. Each task must be independent so they can run in parallel.
3. Give each subagent one narrow assignment - cover a single topic, option, or angle only.
4. Never ask the user questions. Never spawn more than 3 subagents for one request.
   Do not delegate in multiple rounds unless a subagent failed.
5. After all subagents return, write one concise synthesis and stop. No follow-up questions.

Do not do the specialist work yourself. Delegate all research and analysis via task(),
then synthesize the results."""

# Besides your tools and subagents, the agent also gets the built-in write_todos,
# filesystem tools (ls, read_file, write_file, ...) and a catch-all "general-purpose"
# subagent. Turn that one off so the orchestrator must use the three specialists.
register_harness_profile(
    MODEL,
    HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)

# Model strings use "provider:model".
agent = create_deep_agent(
    model=MODEL,
    tools=[],  # Only the researcher has get_weather, so the orchestrator must delegate.
    system_prompt=SYSTEM_PROMPT,
    subagents=subagents,
)


def main():
    question = " ".join(sys.argv[1:]) or "research deepagent of langchain, and langgraph and summarize the key points, then analyze the pros and cons of using it."
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Show the delegation so you can see the parallel task() calls.
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            args = call["args"]
            target = args.get("subagent_type", "")
            detail = args.get("description", args)
            print(f"-> {call['name']} {target}: {str(detail)[:100]}")

    print("\n" + result["messages"][-1].text)


if __name__ == "__main__":
    main()
