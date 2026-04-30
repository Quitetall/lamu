"""
Chainlit frontend — local-llm stack.
Agentic: tool-calling loop with think display, streaming final answer.
Persistent: SQLite chat history shown in sidebar.
"""
import asyncio
import json
import os

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────

BIFROST_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1")
BIFROST_KEY = os.getenv("OPENAI_API_KEY", "sk-local")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen/qwen3.6-27b-uncensored")

# Models that support tool calling → get the agentic loop
AGENTIC_MODELS = {"dflash/luce-dflash", "qwen/qwen3.6-27b-uncensored"}
AVAILABLE_MODELS = [
    "qwen/qwen3.6-27b-uncensored",
    "dflash/luce-dflash",
    "gpt2/shitty-best2021",
    "gpt2/shitty-inferkit",
    "gpt2/shitty-coherent",
    "gpt2/shitty-repetitive",
    "gpt2/shitty-terrible",
    "gpt2/shitty-incoherent",
    "gpt2/shitty-bloatware",
    "gpt2/shitty-default",
]

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. "
    "When you need current information, use the available tools. "
    "Be concise and direct."
)

MAX_AGENT_ITERS = 8

# ── Data layer ───────────────────────────────────────────────────────────────

from data_layer import SQLiteDataLayer

cl.data_layer = SQLiteDataLayer()

# ── Tools ────────────────────────────────────────────────────────────────────

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

_ddg = DuckDuckGoSearchRun(name="web_search")
_wiki = WikipediaQueryRun(
    name="wikipedia",
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000),
)


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return stdout/result. Use for calculations, data processing, etc."""
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<repl>", "exec"), {})  # noqa: S102
        return buf.getvalue() or "(no output)"
    except Exception as e:
        return f"Error: {e}"


TOOLS = [_ddg, _wiki, python_repl]
TOOLS_DICT = {t.name: t for t in TOOLS}

# ── Helpers ──────────────────────────────────────────────────────────────────


def extract_think(text: str) -> tuple[str, str]:
    """Split '<think>…</think>reply' → (think_text, reply_text).
    The chat template pre-injects <think>, so it never appears in the stream.
    """
    if "</think>" in text:
        think, _, reply = text.partition("</think>")
        return think.strip(), reply.strip()
    return "", text.strip()


def build_llm(model: str, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_base=BIFROST_BASE,
        openai_api_key=BIFROST_KEY,
        streaming=streaming,
        temperature=0.7,
        max_tokens=2048,
    )


# ── Agent loop ───────────────────────────────────────────────────────────────


async def run_agent(
    user_input: str,
    history: list,
    model: str,
    response_msg: cl.Message,
) -> tuple[str, list]:
    """
    Agentic loop for dflash models:
      - Shows thinking as a collapsible cl.Step
      - Shows each tool call as a cl.Step
      - Streams the final answer into response_msg
    Returns (final_reply_text, updated_messages_list).
    """
    llm = build_llm(model, streaming=False).bind_tools(TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=user_input)]

    for _ in range(MAX_AGENT_ITERS):
        response = await llm.ainvoke(messages)

        content = response.content or ""
        think_text, clean_content = extract_think(content)

        # Show think block
        if think_text:
            tok_count = len(think_text.split())
            async with cl.Step(name=f"Thinking ({tok_count} tokens)", type="run") as step:
                step.output = think_text

        # Tool calls → execute each, loop
        if response.tool_calls:
            messages.append(response)
            for tc in response.tool_calls:
                name = tc["name"]
                args = tc["args"]
                tool_fn = TOOLS_DICT.get(name)
                async with cl.Step(name=name, type="tool") as step:
                    step.input = json.dumps(args, indent=2)
                    if tool_fn:
                        try:
                            result = await asyncio.to_thread(
                                tool_fn.run, args if isinstance(args, str) else json.dumps(args)
                            )
                        except Exception as e:
                            result = f"Error running {name}: {e}"
                    else:
                        result = f"Unknown tool: {name}"
                    step.output = str(result)[:3000]
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            continue

        # Final answer — stream it
        llm_stream = build_llm(model, streaming=True)
        # Build messages with the clean final response guidance
        stream_messages = messages + [HumanMessage(content="[continue]")] if False else messages

        full_reply = ""
        async for chunk in llm_stream.astream(messages):
            token = chunk.content or ""
            if token:
                # Strip any leftover think prefix on first token
                if not full_reply and "</think>" in token:
                    _, _, token = token.partition("</think>")
                    token = token.lstrip("\n")
                full_reply += token
                await response_msg.stream_token(token)

        # In case think content leaked into the stream
        _, final_reply = extract_think(full_reply)
        if final_reply != full_reply:
            # Think leaked — update message with clean version
            response_msg.content = final_reply
            await response_msg.update()
            full_reply = final_reply

        return full_reply, messages

    final = "I reached the maximum reasoning steps without a conclusive answer."
    await response_msg.stream_token(final)
    return final, messages


async def run_simple(
    user_input: str,
    history: list,
    model: str,
    response_msg: cl.Message,
) -> tuple[str, list]:
    """Simple (non-agentic) chat for gpt2 models."""
    llm = build_llm(model, streaming=True)
    messages = [SystemMessage(content="You are a helpful assistant.")] + history + [HumanMessage(content=user_input)]
    full_reply = ""
    async for chunk in llm.astream(messages):
        token = chunk.content or ""
        if token:
            full_reply += token
            await response_msg.stream_token(token)
    return full_reply, messages


# ── Chainlit callbacks ────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("model", DEFAULT_MODEL)
    cl.user_session.set("history", [])
    await cl.Message(
        content=f"Model: **{DEFAULT_MODEL}**  |  type `/model <name>` to switch\n\n"
        f"Tools available: {', '.join(TOOLS_DICT.keys())}",
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: cl.types.ThreadDict):
    """Restore history when resuming a saved conversation."""
    history = []
    for step in thread.get("steps", []):
        if step.get("type") == "user_message":
            history.append(HumanMessage(content=step.get("output", "")))
        elif step.get("type") == "assistant_message":
            history.append(AIMessage(content=step.get("output", "")))
    cl.user_session.set("history", history)
    cl.user_session.set("model", DEFAULT_MODEL)


@cl.on_message
async def on_message(message: cl.Message):
    content = message.content.strip()

    # /model command
    if content.startswith("/model"):
        parts = content.split(maxsplit=1)
        if len(parts) < 2:
            await cl.Message(
                content=f"Usage: `/model <name>`\nAvailable: {', '.join(AVAILABLE_MODELS)}"
            ).send()
            return
        requested = parts[1].strip()
        if requested not in AVAILABLE_MODELS:
            await cl.Message(
                content=f"Unknown model `{requested}`."
            ).send()
            return
        cl.user_session.set("model", requested)
        await cl.Message(content=f"Switched to **{requested}**.").send()
        return

    model: str = cl.user_session.get("model") or DEFAULT_MODEL
    history: list = cl.user_session.get("history") or []

    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        if model in AGENTIC_MODELS:
            reply, _ = await run_agent(content, history, model, response_msg)
        else:
            reply, _ = await run_simple(content, history, model, response_msg)
    except Exception as e:
        await response_msg.stream_token(f"\n\n*Error: {e}*")
        return

    await response_msg.update()

    history = history + [HumanMessage(content=content), AIMessage(content=reply)]
    if len(history) > 40:
        history = history[-40:]
    cl.user_session.set("history", history)
