import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the agents directory
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    warnings.warn(
        f".env not found at {_env_path}. "
        "Copy .env.example to .env and fill in your keys.",
        stacklevel=1,
    )

from langchain_openai import ChatOpenAI  # noqa: E402

_api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:8080/v1")
_api_key = os.environ.get("OPENAI_API_KEY", "sk-local")
_model = os.environ.get("DEFAULT_MODEL", "qwen3.5-27b")

llm = ChatOpenAI(
    base_url=_api_base,
    api_key=_api_key,
    model=_model,
)

# Langfuse tracing — optional; degrades gracefully if keys are missing
try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

    langfuse_handler = LangfuseCallbackHandler(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
    )
except Exception as exc:  # noqa: BLE001
    warnings.warn(f"Langfuse handler not initialised: {exc}", stacklevel=1)
    langfuse_handler = None


def get_config() -> dict:
    """Return a LangChain RunnableConfig dict with active callbacks."""
    callbacks = [langfuse_handler] if langfuse_handler is not None else []
    return {"callbacks": callbacks}
