import json

import pytest

from src.contracts import validate_generation_result, validate_search_results


def _metadata(index: int = 0) -> dict:
    return {
        "source": "football-policy.md",
        "title": "Football policy",
        "doc_type": "legal",
        "url": "https://example.test/policy",
        "chunk_index": index,
    }


def _result(index: int, score: float, method: str = "hybrid") -> dict:
    return {
        "id": f"chunk-{index}",
        "content": f"Grounded evidence {index}",
        "score": score,
        "metadata": _metadata(index),
        "retrieval_method": method,
    }


def test_pageindex_upload_caches_digest_and_avoids_duplicate_upload(
    monkeypatch, tmp_path
):
    import src.task8_pageindex_vectorless as pageindex

    pdf_dir = tmp_path / "legal"
    pdf_dir.mkdir()
    (pdf_dir / "policy.pdf").write_bytes(b"%PDF-1.7\nfootball policy")
    cache_path = tmp_path / "pageindex_doc_ids.json"

    class FakeClient:
        def __init__(self):
            self.uploads = []

        def submit_document(self, path):
            self.uploads.append(path)
            return {"doc_id": "doc-123"}

    client = FakeClient()
    monkeypatch.setenv("PAGEINDEX_API_KEY", "test-key")
    monkeypatch.setattr(pageindex, "LEGAL_PDF_DIR", pdf_dir)
    monkeypatch.setattr(pageindex, "PAGEINDEX_CACHE_PATH", cache_path)
    monkeypatch.setattr(pageindex, "_make_client", lambda key: client)

    pageindex.upload_documents()
    pageindex.upload_documents()

    assert len(client.uploads) == 1
    cache = json.loads(cache_path.read_text(encoding="utf-8"))["documents"]
    entry = next(iter(cache.values()))
    assert entry["doc_id"] == "doc-123"
    assert len(entry["sha256"]) == 64


def test_pageindex_search_parses_nested_contents_offline(monkeypatch):
    import src.task8_pageindex_vectorless as pageindex

    document = {
        "doc_id": "doc-123",
        "source": "policy.pdf",
        "title": "Football policy",
        "doc_type": "legal",
        "url": None,
    }

    class FakeClient:
        def is_retrieval_ready(self, doc_id):
            return doc_id == "doc-123"

        def submit_query(self, doc_id, query, thinking):
            assert query == "supporters"
            assert thinking is True
            return {"retrieval_id": "retrieval-1"}

        def get_retrieval(self, retrieval_id):
            return {
                "status": "completed",
                "retrieved_nodes": [
                    {
                        "node_id": "node-7",
                        "title": "Supporter engagement",
                        "relevant_contents": [
                            [
                                {
                                    "page_index": 7,
                                    "relevant_content": "Clubs must engage supporters.",
                                },
                                {
                                    "page_index": 8,
                                    "relevant_content": "The regulator protects heritage.",
                                },
                            ]
                        ],
                    }
                ],
            }

    monkeypatch.setenv("PAGEINDEX_API_KEY", "test-key")
    monkeypatch.setattr(pageindex, "_load_cache", lambda: {"policy.pdf": document})
    monkeypatch.setattr(pageindex, "_make_client", lambda key: FakeClient())

    output = pageindex.pageindex_search("supporters", top_k=2)

    validate_search_results(output, top_k=2, expected_method="pageindex")
    assert [item["metadata"]["chunk_index"] for item in output] == [7, 8]
    assert output[0]["score"] > output[1]["score"]


def test_retrieve_degrades_when_dense_and_pageindex_fail(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    sparse = [_result(1, 4.0, "bm25")]

    def broken_dense(query, top_k):
        raise RuntimeError("vector database unavailable")

    def broken_pageindex(query, top_k):
        raise RuntimeError("remote provider unavailable")

    monkeypatch.setattr(pipeline, "semantic_search", broken_dense)
    monkeypatch.setattr(pipeline, "lexical_search", lambda query, top_k: sparse)
    monkeypatch.setattr(pipeline, "pageindex_search", broken_pageindex)

    output = pipeline.retrieve("football", top_k=1, score_threshold=0.5)

    validate_search_results(output, top_k=1, expected_method="hybrid")
    assert output[0]["id"] == "chunk-1"


def test_generation_keeps_source_ranking_and_citation_mapping(monkeypatch):
    import src.task10_generation as generation

    chunks = [_result(index, 1.0 - index / 10) for index in range(5)]
    captured = {}
    monkeypatch.setattr(generation, "retrieve", lambda query, top_k: chunks)

    def fake_llm(system_prompt, user_message):
        captured["prompt"] = user_message
        return "Nguồn thứ hai cung cấp bằng chứng này [2]."

    monkeypatch.setattr(generation, "call_llm", fake_llm)

    output = generation.generate_with_citation("Bằng chứng là gì?", top_k=5)

    validate_generation_result(output)
    assert [item["id"] for item in output["sources"]] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
        "chunk-3",
        "chunk-4",
    ]
    prompt = captured["prompt"]
    assert prompt.index("[1] Title") < prompt.index("[3] Title")
    assert prompt.index("[3] Title") < prompt.index("[5] Title")
    assert prompt.index("[5] Title") < prompt.index("[4] Title")
    assert prompt.index("[4] Title") < prompt.index("[2] Title")


@pytest.mark.parametrize("answer", ["Không có citation.", "Sai nguồn [9].", "Nguồn [0]."])
def test_generation_refuses_answers_with_unverifiable_citations(
    monkeypatch, answer
):
    import src.task10_generation as generation

    monkeypatch.setattr(
        generation,
        "retrieve",
        lambda query, top_k: [_result(0, 0.9)],
    )
    monkeypatch.setattr(generation, "call_llm", lambda *args: answer)

    output = generation.generate_with_citation("Question", top_k=1)

    assert output == {
        "answer": generation.SAFE_REFUSAL,
        "sources": [],
        "retrieval_source": "none",
    }


@pytest.mark.parametrize(
    ("configured", "helper_name"),
    [
        ("openai", "_call_openai"),
        ("lm-studio", "_call_openai_compatible"),
        ("gemini", "_call_gemini"),
        ("claude", "_call_anthropic"),
    ],
)
def test_llm_provider_dispatch_is_mockable(monkeypatch, configured, helper_name):
    import src.task10_generation as generation

    calls = []

    def fake_provider(system_prompt, user_message, model):
        calls.append((system_prompt, user_message, model))
        return "answer"

    monkeypatch.setenv("LLM_PROVIDER", configured)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(generation, helper_name, fake_provider)

    assert generation.call_llm("system", "user") == "answer"
    assert calls == [("system", "user", "test-model")]
