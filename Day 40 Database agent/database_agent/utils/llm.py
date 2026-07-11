"""LLM clients for the agent.

All model/endpoint settings come from `utils.config` (env-driven). The
primary model is Mistral; `custom_llm` (Ollama) is optional and only built
when explicitly requested, so importing this module never requires a running
Ollama server.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_mistralai import ChatMistralAI

from utils.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatMistralAI:
    """Return the shared, timeout-bounded Mistral client."""
    s = get_settings()
    return ChatMistralAI(
        model=s.mistral_model,
        temperature=0,
        max_retries=s.llm_max_retries,
        timeout=s.llm_timeout_s,
    )


@lru_cache(maxsize=1)
def get_custom_llm():
    """Optional local Ollama client (built lazily on first use)."""
    from langchain_ollama import ChatOllama

    s = get_settings()
    return ChatOllama(
        base_url=s.ollama_base_url,
        model=s.ollama_model,
        temperature=0,
    )


# Backwards-compatible module-level handle used across the nodes.
llm = get_llm()


if __name__ == "__main__":
    response = llm.invoke("What are the updates in the world?")
    print(response.content)
