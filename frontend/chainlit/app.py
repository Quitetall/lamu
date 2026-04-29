"""
Chainlit frontend for local-llm stack.
  Bifrost (OpenAI-compat) → ChatOpenAI → Chainlit streaming
  Langfuse tracing, model switching via /model command or chat-start selector.
"""

import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()

BIFROST_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8080/v1")
BIFROST_KEY = os.getenv("OPENAI_API_KEY", "sk-local")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3.5-27b")
AVAILABLE_MODELS = ["qwen3.5-27b", "gpt2-xl"]

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")


def get_langfuse_handler():
    """Return a Langfuse CallbackHandler, or None if keys are not configured."""
    if not LANGFUSE_PUBLIC_KEY or LANGFUSE_PUBLIC_KEY.startswith("pk-lf-..."):
        return None
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    except Exception as e:
        print(f"[langfuse] degraded — {e}")
        return None


def build_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_base=BIFROST_BASE,
        openai_api_key=BIFROST_KEY,
        streaming=True,
        temperature=0.7,
    )


@cl.on_chat_start
async def on_chat_start():
    # Model selector action
    actions = [
        cl.Action(
            name="select_model",
            value=m,
            label=m,
            description=f"Switch to {m}",
        )
        for m in AVAILABLE_MODELS
    ]

    cl.user_session.set("model", DEFAULT_MODEL)
    cl.user_session.set("history", [])

    await cl.Message(
        content=f"Model: **{DEFAULT_MODEL}**. Type `/model <name>` or click to switch.",
        actions=actions,
    ).send()


@cl.action_callback("select_model")
async def on_model_select(action: cl.Action):
    model = action.value
    cl.user_session.set("model", model)
    await cl.Message(content=f"Switched to **{model}**.").send()
    await action.remove()


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
                content=f"Unknown model `{requested}`. Available: {', '.join(AVAILABLE_MODELS)}"
            ).send()
            return
        cl.user_session.set("model", requested)
        await cl.Message(content=f"Switched to **{requested}**.").send()
        return

    model = cl.user_session.get("model") or DEFAULT_MODEL
    history: list = cl.user_session.get("history") or []

    # Build message list for the LLM
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        *history,
        HumanMessage(content=content),
    ]

    llm = build_llm(model)
    callbacks = []
    lf_handler = get_langfuse_handler()
    if lf_handler:
        callbacks.append(lf_handler)

    # Stream response
    response_msg = cl.Message(content="")
    await response_msg.send()

    full_response = ""
    token_count = 0

    async for chunk in llm.astream(messages, config={"callbacks": callbacks} if callbacks else {}):
        token = chunk.content
        if token:
            full_response += token
            token_count += 1
            await response_msg.stream_token(token)

    # Update footer with token count (approximate — chunk count, not real tokens)
    response_msg.content = full_response
    await response_msg.update()

    # Persist to history (keep last 20 turns to avoid context blowout)
    from langchain.schema import AIMessage
    history.append(HumanMessage(content=content))
    history.append(AIMessage(content=full_response))
    if len(history) > 40:
        history = history[-40:]
    cl.user_session.set("history", history)
