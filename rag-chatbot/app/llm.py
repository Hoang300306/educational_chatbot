"""
Milestone 5 — LLM
Question + Retrieved Context  ->  LLM  ->  Answer

Supports OpenAI and Anthropic, selected via LLM_PROVIDER in .env, so you
only need to install/configure the provider you actually want to use.
"""
from __future__ import annotations

from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .logger import get_logger
from .retrieval import RetrievedChunk

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided context excerpts from the user's own documents.\n"
    "Rules:\n"
    "1. Only use information found in the context below. If the answer isn't "
    "in the context, say plainly that you don't have enough information — "
    "never make facts up.\n"
    "2. When you use a piece of context, reference it inline with its [n] marker.\n"
    "3. Answer in the same language the question was asked in.\n"
    "4. Be concise and direct."
)


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, start=1):
        location = f"page {rc.chunk.page}" if rc.chunk.page else "n/a"
        lines.append(f"[{i}] (source: {rc.chunk.source}, {location})\n{rc.chunk.text}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    context_block = build_context_block(chunks) if chunks else "(no relevant context found)"
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and cite sources inline like [1], [2]."
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _call_openai(user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _call_anthropic(user_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    """Milestone 5 entry point: question + retrieved chunks -> answer string."""
    user_prompt = build_user_prompt(question, chunks)
    provider = settings.llm_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set — add it to your .env file")
        answer = _call_openai(user_prompt)
    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set — add it to your .env file")
        answer = _call_anthropic(user_prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'openai' or 'anthropic'.")

    logger.info(f"Generated answer ({len(answer)} chars) using provider='{provider}'")
    return answer
