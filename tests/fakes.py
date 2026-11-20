import json

import httpx

from smriti.embed import EMBED_DIMS, VOYAGE_BASE_URL


def constant_vector(text: str) -> list[float]:
    return [0.1] * EMBED_DIMS


def fake_embedder(status=200, calls=None, vector_for=constant_vector, fail_on=None):
    def handler(request):
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body)
        if status != 200 and fail_on in (None, body["input_type"]):
            return httpx.Response(status)
        data = [
            {"index": i, "embedding": vector_for(text)}
            for i, text in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": data})

    return httpx.AsyncClient(
        base_url=VOYAGE_BASE_URL, transport=httpx.MockTransport(handler)
    )
