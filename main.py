"""Standard Deep Agent: an orchestrator that runs specialist subagents in parallel.

Based on https://docs.langchain.com/oss/python/deepagents/overview
Run: .\\.venv\\Scripts\\python.exe main.py "your question"
"""

import os
import sys
import time
from pathlib import Path
from typing import Literal

from deepagents import (
    GeneralPurposeSubagentProfile, HarnessProfile,
    create_deep_agent, register_harness_profile,
)
from deepagents.backends import CompositeBackend, FilesystemBackend
from dotenv import load_dotenv
from langchain.agents.middleware import TodoListMiddleware
from tavily import TavilyClient

load_dotenv(Path(__file__).with_name(".env"))
MODEL = f"openai:{os.getenv('OPENAI_MODEL', 'gpt-5.5')}"
if not os.getenv("TAVILY_API_KEY"):
    raise SystemExit("Add TAVILY_API_KEY to .env (get one at https://tavily.com).")
tavily_client = TavilyClient()  # reads TAVILY_API_KEY

# Each run gets its own folder under workspace/. The agent sees it as "/", so its
# "/findings/1.md" is saved on disk as workspace/<run>/findings/1.md.
RUN_DIR = Path(__file__).with_name("workspace") / time.strftime("%Y%m%d-%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Long-term memory lives in one fixed folder shared by every run, so what the agent
# learns in one run is loaded into its prompt in the next.
MEMORY_DIR = Path(__file__).with_name("memories")
MEMORY_FILE = MEMORY_DIR / "AGENTS.md"
if not MEMORY_FILE.exists():
    MEMORY_DIR.mkdir(exist_ok=True)
    MEMORY_FILE.write_text("# Agent memory\n\n## User preferences\n- (none yet)\n", encoding="utf-8")


# A custom tool, as in the quickstart. The docstring tells the model what it does.
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# Web search via Tavily, as in the deepagents docs' research example.
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    #include_raw_content: bool = False,
) -> dict:
    """Search the web and return result titles, URLs and content snippets."""
    return tavily_client.search(
        query,
        max_results=max_results,
        topic=topic,
        include_raw_content=False,  # snippets only; not the model's choice
    )


# Every subagent saves its full working notes to a file; the short answer it
# returns is all the orchestrator sees.
SAVE_FINDINGS = ("Before you return, save your full findings with sources to /findings/<n>.md "
                 "using write_file, where <n> is the number in brackets at the start of your "
                 "assignment. ")

# Each subagent gets its own prompt and context. The main agent reaches them
# through the built-in `task` tool, using `description` to pick one.
subagents = [
    {
        "name": "researcher",
        "description": "Delegate one focused research task: gather key facts on a single topic. "
                       "Use for parallel coverage of distinct subjects.",
        "system_prompt": "You are a research specialist. Complete your assigned topic only. "
                         "Use internet_search to find current facts; do not rely on memory. "
                         + SAVE_FINDINGS +
                         "Return 3-5 bullet points with the most important facts, each with its source URL. "
                         "No preamble, no follow-up questions, and do not delegate further work.",
        "tools": [get_weather, internet_search],
    },
    {
        "name": "analyzer",
        "description": "Delegate one focused analysis task: pros, cons, and trade-offs for a single "
                       "option or approach. Use when comparing alternatives in parallel.",
        "system_prompt": "You are an analysis specialist. Complete your assigned option or angle only. "
                         "Use internet_search to ground your analysis in current sources. "
                         + SAVE_FINDINGS +
                         "Return 3-5 bullet points covering strengths, weaknesses, and key trade-offs. "
                         "No preamble, no follow-up questions, and do not delegate further work.",
        "tools": [internet_search],
    },
    {
        "name": "writer",
        "description": "Delegate one focused summary task: turn findings on a single subtopic into "
                       "clear prose. Use when each parallel subagent should produce readable copy.",
        "system_prompt": "You are a writing specialist. Complete your assigned subtopic only. "
                         + SAVE_FINDINGS +
                         "Return a short paragraph or 3-5 bullets with the key takeaways. "
                         "No preamble, no follow-up questions, and do not delegate further work.",
    },
]

SYSTEM_PROMPT = """You are an orchestrator demo agent. Your job is to showcase parallel
subagents: spin up a small set of specialists at once, let each do a short piece of work,
then synthesize and finish.

For every user request:
0. If the user states a lasting preference or instruction (e.g. "from now on...",
   "always...", "I prefer..."), first save it under "User preferences" in
   /memories/AGENTS.md: read_file it, then edit_file. Follow saved preferences in every
   answer, and tell each subagent about any that affect its assignment.
1. First call write_todos with one todo per subagent assignment, all pending. Do not add
   a todo for the final synthesis.
2. In the very next tool-calling turn, spawn exactly 2-3 subagents using multiple task()
   calls in that single turn. Each task must be independent so they can run in parallel.
   Start each task description with its todo's position in brackets, e.g. "[1] Find ...".
3. Give each subagent one narrow assignment - cover a single topic, option, or angle only.
4. Never ask the user questions. Never spawn more than 3 subagents for one request.
   Do not delegate in multiple rounds unless a subagent failed.
5. After all subagents return, call write_todos to mark their todos completed. Then save
   one concise synthesis to /report.md with write_file, give that same synthesis as your
   final answer, and stop. No follow-up questions.

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
    middleware=[TodoListMiddleware()],  # Adds the write_todos planning tool.
    # Real files instead of the default in-memory StateBackend. virtual_mode keeps
    # every path inside its root_dir, so the agent can't touch anything else on disk.
    # Paths under /memories/ go to MEMORY_DIR; everything else to this run's folder.
    backend=CompositeBackend(
        default=FilesystemBackend(root_dir=RUN_DIR, virtual_mode=True),
        routes={"/memories/": FilesystemBackend(root_dir=MEMORY_DIR, virtual_mode=True)},
    ),
    # Loaded into the orchestrator's system prompt at the start of every run.
    memory=["/memories/AGENTS.md"],
)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # so the Windows console prints °, —, etc.
    question = " ".join(sys.argv[1:]) or "what is the weather in my location? Always Remember I am in Mesa. ALSO DO research deepagent of langchain, and langgraph and summarize the key points, then analyze the pros and cons of using it."
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # Each turn of the orchestrator's conversation and the tools it called.
    for message in result["messages"]:
        if message.type == "tool":
            continue
        if message.text:
            print(f"\n[{message.type}] {message.text}")
        for call in getattr(message, "tool_calls", None) or []:
            print(f"  -> {call['name']} {call['args']}")


if __name__ == "__main__":
    main()
