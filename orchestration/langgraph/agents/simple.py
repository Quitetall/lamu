"""Minimal single-node LangGraph chat agent.

Usage:
    python agents/simple.py "your prompt here"
"""

import sys
from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from base import get_config, langfuse_handler, llm


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def chat_node(state: ChatState) -> ChatState:
    response = llm.invoke(state["messages"], config=get_config())
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()


app = build_graph()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello!"
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})
    print(result["messages"][-1].content)
