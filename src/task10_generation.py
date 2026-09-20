"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation [Document n]. Nếu thiếu evidence, hãy từ chối xác minh."""

# Opencode Zen (OpenAI-compatible Responses API).
# User cho: base url https://opencode.ai/zen/go/v1/responses + model muse-spark-1.3-contributor.
# SDK base_url phải bỏ đuôi /responses (SDK tự append).
OPENCODE_DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
OPENCODE_DEFAULT_MODEL = "muse-spark-1.3-contributor"


def _normalize_base_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    for suffix in ("/responses", "/chat/completions", "/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    return url


def _opencode_config() -> tuple[str, str, str]:
    key = os.getenv("OPENCODE_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    base_url = _normalize_base_url(
        os.getenv("OPENCODE_BASE_URL", "") or OPENCODE_DEFAULT_BASE_URL
    )
    model = os.getenv("LLM_MODEL", "") or OPENCODE_DEFAULT_MODEL
    return key, base_url, model


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[Document {index} | Title: {metadata['title']} | "
            f"Source: {metadata['source']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_responses_text(resp) -> str:
    text = getattr(resp, "output_text", "")
    if text and text.strip():
        return text.strip()
    texts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            t = getattr(block, "text", "")
            if t:
                texts.append(t)
    return "".join(texts).strip()


def _call_opencode(system_prompt: str, user_message: str) -> str:
    """Gọi Opencode Zen qua python-sdk OpenAI-compatible (Responses API)."""
    from openai import OpenAI

    api_key, base_url, model = _opencode_config()
    if not api_key:
        raise RuntimeError("Missing OPENCODE_API_KEY (hoặc OPENAI_API_KEY)")
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_message,
            temperature=TEMPERATURE,
        )
    except TypeError:
        resp = client.responses.create(
            model=model, instructions=system_prompt, input=user_message
        )
    text = _extract_responses_text(resp)
    if not text.strip():
        raise RuntimeError("Empty response from opencode")
    return text


def _call_openai(system_prompt: str, user_message: str) -> str:
    from openai import OpenAI

    model = os.getenv("LLM_MODEL", "") or "gpt-4o-mini"
    client = OpenAI()  # OPENAI_API_KEY, OPENAI_BASE_URL nếu có
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return (resp.choices[0].message.content or "").strip()


def _call_gemini(system_prompt: str, user_message: str) -> str:
    from google import genai

    model = os.getenv("LLM_MODEL", "") or "gemini-2.0-flash"
    client = genai.Client()
    resp = client.models.generate_content(
        model=model, contents=f"{system_prompt}\n\n{user_message}"
    )
    return (resp.text or "").strip()


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    from anthropic import Anthropic

    model = os.getenv("LLM_MODEL", "") or "claude-3-5-haiku-latest"
    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    texts = [
        block.text for block in resp.content if getattr(block, "text", "")
    ]
    return "".join(texts).strip()


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini, Anthropic hoặc Opencode theo cấu hình."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    # Alias: zen cũng là opencode.
    if provider in ("opencode", "zen"):
        return _call_opencode(system_prompt, user_message)
    if provider == "openai":
        # Cho phép dùng key opencode qua base_url custom mà vẫn để provider=openai.
        if os.getenv("OPENCODE_BASE_URL", "") or (
            os.getenv("OPENAI_BASE_URL", "").rstrip("/").endswith("/go/v1")
            or "opencode.ai" in os.getenv("OPENAI_BASE_URL", "")
        ):
            return _call_opencode(system_prompt, user_message)
        return _call_openai(system_prompt, user_message)
    if provider == "gemini":
        return _call_gemini(system_prompt, user_message)
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_message)
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
        if not answer.strip():
            raise RuntimeError("Empty LLM answer")
    except Exception:
        return {
            "answer": SAFE_REFUSAL,
            "sources": [],
            "retrieval_source": "none",
        }
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0]["retrieval_method"],
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
