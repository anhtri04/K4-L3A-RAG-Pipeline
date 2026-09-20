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

SYSTEM_PROMPT = """Bạn là chatbot RAG về bóng đá. Trả lời CHỈ từ context được cung cấp.
Quy tắc:
- Mỗi khẳng định phải kèm citation dạng [1], [2] tương ứng số Document trong context.
- Citation phải map 1-1 với sources trả về.
- Nếu context thiếu evidence, trả lời đúng câu: Tôi không thể xác minh thông tin này từ nguồn hiện có.
- Trả lời ngắn gọn, đúng trọng tâm, ưu tiên luật (IFAB Laws) cho câu hỏi về luật."""


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


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi LLM qua OpenAI-compatible API (LM Studio local / OpenAI / Gemini / Anthropic)."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "")
    if not model:
        raise RuntimeError("LLM_MODEL is not set in .env")

    if provider in ("openai", "lmstudio", "lm_studio", "openai_compatible"):
        from openai import OpenAI

        client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
        )
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

    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        resp = client.models.generate_content(
            model=model,
            contents=f"{system_prompt}\n\n{user_message}",
        )
        return (resp.text or "").strip()

    if provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()

    raise RuntimeError(f"Unsupported LLM_PROVIDER={provider}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception as error:
        print(f"LLM error (safe refusal): {error}")
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }
    if not answer.strip():
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }
    method = chunks[0].get("retrieval_method", "hybrid")
    retrieval_source = method if method in ("hybrid", "pageindex") else "hybrid"
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("test query"))
