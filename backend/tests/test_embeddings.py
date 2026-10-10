"""The embeddings client (#450): the OpenAI `/v1/embeddings` wire, the task
prefixes EmbeddingGemma needs, batching, and every failure turning into
`EmbeddingsUnavailable` so the caller can fall back to keywords. The server
is an `httpx.MockTransport`; no live service is ever required."""

import json

import httpx
import pytest

from chessapp import embeddings


def _embedder(handler) -> embeddings.Embedder:
    client = httpx.Client(
        base_url="http://embeddings.test/v1", transport=httpx.MockTransport(handler)
    )
    return embeddings.create_embedder("unused", client=client)


def _answer(inputs: list[str], model: str = "gemma.gguf", reverse: bool = False):
    data = [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(inputs))]
    if reverse:
        data.reverse()
    return httpx.Response(200, json={"model": model, "data": data})


def test_a_query_is_sent_with_the_search_prefix():
    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content)))
        return _answer(seen[-1][1]["input"])

    result = _embedder(handler).embed_query("what's a fork")
    assert seen == [
        ("/v1/embeddings", {"input": ["task: search result | query: what's a fork"]})
    ]
    assert result == embeddings.Embedded("gemma.gguf", [[0.0, 1.0]])


def test_vectors_come_back_in_input_order():
    def handler(request):
        return _answer(json.loads(request.content)["input"], reverse=True)

    result = _embedder(handler).embed(["a", "b", "c"])
    assert [vector[0] for vector in result.vectors] == [0.0, 1.0, 2.0]


def test_documents_are_embedded_in_batches():
    sizes = []

    def handler(request):
        inputs = json.loads(request.content)["input"]
        sizes.append(len(inputs))
        return _answer(inputs)

    documents = [embeddings.document_text(f"t{i}", "x") for i in range(40)]
    result = _embedder(handler).embed_documents(documents)
    assert sizes == [16, 16, 8]
    assert len(result.vectors) == 40


def test_the_document_format():
    assert embeddings.document_text("Fork", "Two at once.") == (
        "title: Fork | text: Two at once."
    )


def _refuse(request):
    raise httpx.ConnectError("connection refused", request=request)


def _timeout(request):
    raise httpx.ReadTimeout("slow", request=request)


@pytest.mark.parametrize(
    "handler",
    [
        _refuse,
        _timeout,
        lambda request: httpx.Response(500, text="boom"),
        lambda request: httpx.Response(200, text="not json"),
        lambda request: httpx.Response(200, json={"data": "nope"}),
        lambda request: httpx.Response(200, json={"model": "m", "data": []}),
        lambda request: httpx.Response(
            200, json={"model": "m", "data": [{"index": 0, "embedding": []}]}
        ),
    ],
    ids=["refused", "timeout", "500", "junk", "shape", "too-few", "empty-vector"],
)
def test_every_failure_is_unavailable(handler):
    with pytest.raises(embeddings.EmbeddingsUnavailable):
        _embedder(handler).embed_query("anything")


def test_a_model_change_mid_batch_is_unavailable():
    models = iter(["a", "b"])

    def handler(request):
        return _answer(json.loads(request.content)["input"], model=next(models))

    documents = ["x"] * 20
    with pytest.raises(embeddings.EmbeddingsUnavailable):
        _embedder(handler).embed_documents(documents)
