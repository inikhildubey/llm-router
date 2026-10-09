import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def make_litellm_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_route_success():
    # app.py and router.py both `import litellm`, so they share the same module object -
    # a single mock with ordered side_effects stands in for the triage call then the execution call.
    triage_response = make_litellm_response(json.dumps({"complexity_score": 2, "reasoning": "simple"}))
    exec_response = make_litellm_response("Hi there!")
    with patch("litellm.acompletion", new=AsyncMock(side_effect=[triage_response, exec_response])):
        response = client.post("/route", json={"prompt": "Hello!"})
    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 2
    assert body["response"] == "Hi there!"
    assert "model" in body


def test_route_rejects_empty_prompt():
    response = client.post("/route", json={"prompt": "   "})
    assert response.status_code == 422


def test_route_rejects_oversized_prompt():
    response = client.post("/route", json={"prompt": "x" * (app_module.config.max_prompt_length + 1)})
    assert response.status_code == 422


def test_route_execution_failure_returns_502():
    triage_response = make_litellm_response(json.dumps({"complexity_score": 2, "reasoning": "simple"}))
    with patch("router.litellm.acompletion", new=AsyncMock(return_value=triage_response)), \
         patch("app.litellm.acompletion", new=AsyncMock(side_effect=Exception("model unavailable"))):
        response = client.post("/route", json={"prompt": "Hello!"})
    assert response.status_code == 502


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "report" in response.json()
