import pytest

from src.contracts import validate_generation_result


def _message(question: str, content: str, citation: int = 1) -> str:
    return (
        "Context:\n"
        f"[{citation}] Title: Football governance | Source: policy.md | Chunk: 0\n"
        f"{content}\n\n"
        f"Question: {question}"
    )


def _result(content: str) -> dict:
    return {
        "id": "policy-0",
        "content": content,
        "score": 0.91,
        "metadata": {
            "source": "policy.md",
            "title": "Football governance",
            "doc_type": "legal",
            "url": "https://example.test/policy",
            "chunk_index": 0,
        },
        "retrieval_method": "hybrid",
    }


def test_extractive_provider_dispatches_without_model_or_api_key(monkeypatch):
    import src.task10_generation as generation

    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    content = "The regulator protects supporters through mandatory club engagement."

    answer = generation.call_llm(
        "ignored by deterministic provider",
        _message("How does the regulator protect supporters?", content),
    )

    assert answer == f"The indexed evidence states: {content} [1]"


def test_hosted_provider_still_requires_explicit_model(monkeypatch):
    import src.task10_generation as generation

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="LLM_MODEL is not configured"):
        generation.call_llm("system", "user")


def test_extractive_provider_matches_documented_vietnamese_concepts(monkeypatch):
    import src.task10_generation as generation

    monkeypatch.setenv("LLM_PROVIDER", "offline-extractive")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    content = "The regulator protects supporters through mandatory club engagement."

    answer = generation.call_llm(
        "unused",
        _message("Người hâm mộ được bảo vệ như thế nào?", content),
    )

    assert answer.startswith("Tài liệu nêu: ")
    assert content in answer
    assert answer.endswith("[1]")


@pytest.mark.parametrize(
    "user_message",
    [
        _message(
            "Thời tiết trên sao Hỏa hôm nay thế nào?",
            "The regulator protects supporters through mandatory club engagement.",
        ),
        "A message without the private context envelope",
    ],
)
def test_extractive_provider_refuses_unrelated_or_malformed_input(
    monkeypatch, user_message
):
    import src.task10_generation as generation

    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert generation.call_llm("unused", user_message) == generation.SAFE_REFUSAL


def test_generate_with_citation_runs_end_to_end_offline(monkeypatch):
    import src.task10_generation as generation

    content = "The regulator protects supporters through mandatory club engagement."
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(
        generation,
        "retrieve",
        lambda query, top_k: [_result(content)],
    )

    output = generation.generate_with_citation(
        "How does the regulator protect supporters?", top_k=1
    )

    validate_generation_result(output)
    assert output["answer"].endswith("[1]")
    assert output["sources"][0]["content"] == content
    assert output["retrieval_source"] == "hybrid"


def test_generate_with_citation_hides_sources_when_extractive_evidence_is_weak(
    monkeypatch,
):
    import src.task10_generation as generation

    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    monkeypatch.setattr(
        generation,
        "retrieve",
        lambda query, top_k: [
            _result("The regulator protects supporters through club engagement.")
        ],
    )

    output = generation.generate_with_citation(
        "How should I bake a chocolate cake?", top_k=1
    )

    assert output == {
        "answer": generation.SAFE_REFUSAL,
        "sources": [],
        "retrieval_source": "none",
    }
