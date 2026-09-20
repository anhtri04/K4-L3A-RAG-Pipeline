"""Task 10 — grounded answer generation with verifiable citations."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from dotenv import load_dotenv

from .contracts import validate_search_results
from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
MAX_TOKENS = 1024

SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# The extractive provider is deliberately small and transparent.  It is an
# offline fallback, not a language model: it selects verbatim evidence
# sentences whose normalized concepts overlap with the question.
_CONTEXT_SEPARATOR = "\n\n---\n\n"
_GENERIC_QUERY_TERMS = {
    "answer",
    "bong",
    "context",
    "document",
    "evidence",
    "explain",
    "football",
    "information",
    "say",
    "soccer",
    "source",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "should",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    # Vietnamese function/question words after diacritic removal.
    "ban",
    "bi",
    "cho",
    "co",
    "cua",
    "da",
    "de",
    "den",
    "duoc",
    "gi",
    "hay",
    "la",
    "mot",
    "nao",
    "nhu",
    "nhung",
    "o",
    "ra",
    "tai",
    "the",
    "thi",
    "tu",
    "va",
    "ve",
}
_VIETNAMESE_PHRASES = {
    "ban thang ky vong": "expected goal",
    "bao ve": "protect",
    "cau lac bo": "club",
    "chan dong": "concussion",
    "chan thuong": "injury",
    "chu so huu": "owner",
    "co dong vien": "supporter",
    "co quan quan ly": "regulator",
    "du doan": "forecast",
    "gan ket": "engage",
    "kiem tra": "test",
    "luat bong da": "football rule",
    "nguoi ham mo": "supporter",
    "phu nu": "women",
    "quan tri": "governance",
    "tai chinh": "finance",
    "tai nang": "talent",
    "tham gia": "engage",
    "trong tai": "referee",
    "viet vi": "offside",
}
_TERM_ALIASES = {
    "clubs": "club",
    "concussed": "concussion",
    "engaged": "engage",
    "engagement": "engage",
    "fans": "supporter",
    "female": "women",
    "finances": "finance",
    "financial": "finance",
    "forecasting": "forecast",
    "governing": "governance",
    "injuries": "injury",
    "laws": "rule",
    "owners": "owner",
    "ownership": "owner",
    "duties": "objective",
    "duty": "objective",
    "goals": "objective",
    "objectives": "objective",
    "protected": "protect",
    "protecting": "protect",
    "protection": "protect",
    "protects": "protect",
    "regulations": "rule",
    "regulator": "regulator",
    "regulators": "regulator",
    "rules": "rule",
    "safeguard": "protect",
    "safeguards": "protect",
    "spectator": "supporter",
    "spectators": "supporter",
    "supporters": "supporter",
    "testing": "test",
    "tests": "test",
}

SYSTEM_PROMPT = f"""Bạn là chatbot RAG về bóng đá. Trả lời CHỈ từ context được cung cấp.
Quy tắc:
- Mỗi khẳng định thực tế phải kèm citation dạng [1], [2] tương ứng đúng số nguồn trong context.
- Không dùng kiến thức bên ngoài context và không tự tạo citation.
- Nếu context không đủ bằng chứng, chỉ trả lời đúng câu: {SAFE_REFUSAL}
- Trả lời ngắn gọn, đúng trọng tâm và bằng cùng ngôn ngữ với câu hỏi."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Move high-ranked chunks to the context edges without mutating input."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Format context while preserving any preassigned citation numbers."""
    parts: list[str] = []
    for position, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        citation_index = chunk.get("_citation_index", position)
        title = metadata.get("title") or "Untitled"
        source = metadata.get("source") or "Unknown source"
        chunk_index = metadata.get("chunk_index", "-")
        url = metadata.get("url")
        header = (
            f"[{citation_index}] Title: {title} | Source: {source} | "
            f"Chunk: {chunk_index}"
        )
        if isinstance(url, str) and url.strip():
            header += f" | URL: {url.strip()}"
        parts.append(f"{header}\n{str(chunk.get('content', '')).strip()}")
    return "\n\n---\n\n".join(parts)


def _ascii_lower(text: str) -> str:
    """Return a lower-case, accent-free form for deterministic matching."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )


def _normalized_terms(text: str, *, translate_vietnamese: bool = False) -> set[str]:
    normalized = _ascii_lower(text)
    if translate_vietnamese:
        # Longest phrases are replaced first so "luat bong da" is not partly
        # consumed by a shorter expression.  The mapping is intentionally
        # finite and documented; it is not presented as machine translation.
        for phrase, replacement in sorted(
            _VIETNAMESE_PHRASES.items(), key=lambda item: len(item[0]), reverse=True
        ):
            normalized = re.sub(
                rf"\b{re.escape(phrase)}\b", replacement, normalized
            )

    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", normalized):
        if token in _STOPWORDS or len(token) < 2:
            continue
        canonical = _TERM_ALIASES.get(token, token)
        terms.add(canonical)
    return terms


def _looks_vietnamese(text: str) -> bool:
    lowered = text.lower()
    vietnamese_characters = (
        "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
        "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    )
    if any(character in lowered for character in vietnamese_characters):
        return True
    accent_free = f" {_ascii_lower(text)} "
    return any(
        marker in accent_free
        for marker in (" nguoi ", " khong ", " nhu the nao ", " tai sao ", " bong da ")
    )


def _parse_extractive_message(user_message: str) -> tuple[str, list[tuple[int, str]]]:
    """Parse the private context/question envelope used by this module."""
    if not isinstance(user_message, str):
        return "", []
    context_prefix, separator, question = user_message.rpartition("\n\nQuestion:")
    if not separator or not context_prefix.startswith("Context:\n"):
        return "", []

    evidence: list[tuple[int, str]] = []
    context = context_prefix[len("Context:\n") :]
    for block in context.split(_CONTEXT_SEPARATOR):
        header, newline, content = block.partition("\n")
        match = re.match(r"^\[(\d+)\]\s+Title:", header)
        if not newline or not match or not content.strip():
            continue
        citation = int(match.group(1))
        if citation > 0:
            evidence.append((citation, content.strip()))
    return question.strip(), evidence


def _evidence_sentences(content: str) -> list[str]:
    """Split prose/Markdown into bounded, citation-free extractive units."""
    sentences: list[str] = []
    for raw_sentence in re.split(r"(?<=[.!?])\s+|\n+", content):
        sentence = re.sub(r"^\s*(?:#{1,6}|[-*+]\s+|\d+[.)]\s+)", "", raw_sentence)
        sentence = CITATION_PATTERN.sub("", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip(" -\t")
        if len(sentence) < 20:
            continue
        if len(sentence) > 420:
            sentence = sentence[:417].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
        sentences.append(sentence)
    return sentences


def _call_extractive(system_prompt: str, user_message: str) -> str:
    """Select relevant source sentences for a deterministic offline answer.

    ``system_prompt`` is accepted to keep the provider interface uniform, but
    it is not interpreted.  This function never synthesizes facts: every
    factual clause in its output is copied from a supplied evidence block.
    """
    del system_prompt
    question, evidence = _parse_extractive_message(user_message)
    if not question or not evidence:
        return SAFE_REFUSAL

    query_terms = _normalized_terms(question, translate_vietnamese=True)
    specific_terms = query_terms - _GENERIC_QUERY_TERMS
    if not specific_terms:
        return SAFE_REFUSAL

    # Two matching concepts are required for multi-concept questions.  This
    # keeps incidental words such as "rule" from making an unrelated question
    # look grounded, while still supporting focused queries such as "offside?".
    minimum_matches = 1 if len(specific_terms) == 1 else 2
    candidates: list[tuple[float, int, int, str]] = []
    for evidence_position, (citation, content) in enumerate(evidence):
        for sentence_position, sentence in enumerate(_evidence_sentences(content)):
            sentence_terms = _normalized_terms(sentence)
            matched = specific_terms & sentence_terms
            if len(matched) < minimum_matches:
                continue
            coverage = len(matched) / len(specific_terms)
            density = len(matched) / max(len(sentence_terms), 1)
            # Coverage matters more than density: otherwise a short heading
            # such as "The Independent Football Regulator" can outrank the
            # longer sentence that actually lists the regulator's objectives.
            fragment_penalty = (
                0.20
                if len(sentence) < 55 and not re.search(r"[.!?]$", sentence)
                else 0.0
            )
            score = coverage + 0.20 * density - fragment_penalty
            candidates.append(
                (score, -evidence_position, -sentence_position, f"{sentence} [{citation}]")
            )

    if not candidates:
        return SAFE_REFUSAL

    candidates.sort(reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    best_score = candidates[0][0]
    for score, _, _, sentence in candidates:
        normalized_sentence = _ascii_lower(CITATION_PATTERN.sub("", sentence)).strip()
        if normalized_sentence in seen or score < best_score * 0.65:
            continue
        selected.append(sentence)
        seen.add(normalized_sentence)
        if len(selected) == 2:
            break

    prefix = (
        "Tài liệu nêu: "
        if _looks_vietnamese(question)
        else "The indexed evidence states: "
    )
    return prefix + " ".join(selected) if selected else SAFE_REFUSAL


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


def _openai_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _call_openai(system_prompt: str, user_message: str, model: str) -> str:
    from openai import OpenAI

    kwargs: dict[str, str] = {"api_key": _required_env("OPENAI_API_KEY")}
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )
    return _openai_text(response.choices[0].message.content)


def _call_openai_compatible(system_prompt: str, user_message: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1").strip(),
        api_key=os.getenv("OPENAI_API_KEY", "lm-studio").strip() or "lm-studio",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )
    return _openai_text(response.choices[0].message.content)


def _call_gemini(system_prompt: str, user_message: str, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_required_env("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_TOKENS,
        ),
    )
    return str(response.text or "").strip()


def _call_anthropic(system_prompt: str, user_message: str, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=_required_env("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and isinstance(block.text, str)
    ).strip()


def call_llm(
    system_prompt: str,
    user_message: str,
    provider_override: str | None = None,
) -> str:
    """Dispatch to the explicitly configured generation provider."""
    configured_provider = provider_override or os.getenv("LLM_PROVIDER", "openai")
    provider = configured_provider.strip().lower().replace("-", "_")
    if provider in {"extractive", "offline", "offline_extractive"}:
        return _call_extractive(system_prompt, user_message)

    # Hosted and OpenAI-compatible providers retain their existing model
    # requirement.  The deterministic extractive fallback has no model.
    model = _required_env("LLM_MODEL")

    if provider == "openai":
        return _call_openai(system_prompt, user_message, model)
    if provider in {"lmstudio", "lm_studio", "openai_compatible"}:
        return _call_openai_compatible(system_prompt, user_message, model)
    if provider in {"gemini", "google", "google_genai"}:
        return _call_gemini(system_prompt, user_message, model)
    if provider in {"anthropic", "claude"}:
        return _call_anthropic(system_prompt, user_message, model)
    raise RuntimeError(f"Unsupported LLM_PROVIDER={provider!r}")


def _refusal_result() -> dict:
    return {
        "answer": SAFE_REFUSAL,
        "sources": [],
        "retrieval_source": "none",
    }


def _citations_are_valid(answer: str, source_count: int) -> bool:
    citations = [int(value) for value in CITATION_PATTERN.findall(answer)]
    return bool(citations) and all(1 <= citation <= source_count for citation in citations)


def generate_from_chunks(
    query: str,
    chunks: list[dict],
    *,
    provider_override: str | None = None,
) -> dict:
    """Generate from a supplied ranking using the same Task 10 safeguards.

    This seam keeps evaluation honest: dense-only and hybrid experiments can
    supply their own retrieved contexts while exercising the exact generator,
    context reordering, citation mapping, and refusal logic used by the app.
    """
    if not isinstance(query, str) or not query.strip() or not chunks:
        return _refusal_result()
    try:
        validate_search_results(chunks, top_k=len(chunks))
    except (TypeError, ValueError):
        return _refusal_result()

    # Sources stay score-sorted to satisfy the public SearchResult contract.
    # Private citation labels retain those source indexes even after the
    # context is reordered to reduce lost-in-the-middle effects.
    numbered_chunks = [
        {**chunk, "_citation_index": index}
        for index, chunk in enumerate(chunks, 1)
    ]
    context = format_context(reorder_for_llm(numbered_chunks))
    user_message = f"Context:\n{context}\n\nQuestion: {query.strip()}"
    try:
        if provider_override is not None:
            raw_answer = call_llm(
                SYSTEM_PROMPT,
                user_message,
                provider_override=provider_override,
            )
        else:
            raw_answer = call_llm(SYSTEM_PROMPT, user_message)
        answer = raw_answer.strip()
    except Exception as error:
        print(f"LLM error (safe refusal): {error}")
        return _refusal_result()

    if answer == SAFE_REFUSAL:
        return _refusal_result()
    if not answer or not _citations_are_valid(answer, len(chunks)):
        return _refusal_result()

    method = chunks[0].get("retrieval_method")
    retrieval_source = "pageindex" if method == "pageindex" else "hybrid"
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Retrieve evidence, generate an answer, and enforce citation integrity."""
    if (
        not isinstance(query, str)
        or not query.strip()
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        return _refusal_result()

    try:
        chunks = retrieve(query.strip(), top_k=top_k)
        validate_search_results(chunks, top_k=top_k)
    except Exception as error:
        print(f"Retrieval error (safe refusal): {error}")
        return _refusal_result()
    if not chunks:
        return _refusal_result()
    return generate_from_chunks(query.strip(), chunks)


if __name__ == "__main__":
    print(generate_with_citation("Football governance rules protect supporters how?"))
