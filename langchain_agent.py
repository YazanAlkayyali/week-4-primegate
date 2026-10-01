from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.mcp import MCPAdapter

from src.week_4_primegate.ai_config import API_KEY as GROQ_API_KEY
from src.week_4_primegate.retrieval import search_chapter


@tool
def retrieve_chapter_context(query: str) -> str:
    """Search "The Twenty-First-Century Entrepreneur" Chapter 1 for
    relevant passages. Use this for unstructured "what does the
    chapter say about X" questions -- concepts, definitions,
    explanations, advice given in the text. Do NOT use this for
    exact structured facts like an employee's leave balance; use
    lookup_leave_balance for that instead.

    Args:
        query: The average question to search the chapter for.
    """
    chunks = search_chapter(query, limit=5)

    if not chunks:
        return "No relevant passages found in Chapter 1 for that query."

    return "\n\n---\n\n".join(chunks)


SYSTEM_PROMPT = """You are a course assistant with two tools:

1. retrieve_chapter_context, for unstructured questions about what
   the entrepreneurship textbook chapter says (concepts, advice,
   explanations). Use this whenever the answer depends on the
   chapter's actual content.

2. lookup_leave_balance, for exact, structured facts tied to an
   employee ID (e.g. leave balance, status). Use this only when the
   user gives or asks about a specific employee ID.

Decision rules:
- If the question is about chapter content -> call retrieve_chapter_context.
- If the question is a structured lookup by ID -> call lookup_leave_balance.
- If the question is general knowledge unrelated to the chapter or any
  employee record (e.g. "what's 12 * 4") -> answer directly, no tool call.
- If the question is too ambiguous to route confidently (e.g. it's
  unclear whether they mean the chapter's content or an employee record,
  or an ID is missing) -> ask ONE short clarifying question instead of
  guessing or calling a tool.

IMPORTANT: For any question about the textbook chapter, you must rely
ONLY on what retrieve_chapter_context returns. If it returns "No
relevant passages found," tell the user that directly -- do not answer
from your own general knowledge about business or entrepreneurship
textbooks, even if you think you know the answer. Only Chapter 1 has
been ingested; you have no information about any other chapter.
"""


# Path to our local MCP server
MCP_SERVER_PATH = Path(__file__).resolve().parent / "mcp_server.py"


_agent = None


async def get_agent():
    global _agent

    if _agent is None:

        async with MCPAdapter(MCP_SERVER_PATH) as adapter:

            mcp_tools = await adapter.list_tools()

            print(
                "MCP tools loaded:",
                [tool.name for tool in mcp_tools]
            )

            _agent = create_agent(
                model=ChatGroq(
                    model="openai/gpt-oss-20b",
                    api_key=GROQ_API_KEY
                ),
                tools=[
                    retrieve_chapter_context,
                    *mcp_tools
                ],
                system_prompt=SYSTEM_PROMPT,
            )

    return _agent


async def ask(user_input: str) -> str:
    """Run one turn through the agent and return the final text reply."""

    agent = await get_agent()

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        }
    )

    return result["messages"][-1].content