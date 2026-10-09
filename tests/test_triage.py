import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from router import Triage, RouterConfig


def make_litellm_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


@pytest.fixture
def triage():
    config = RouterConfig(low_threshold=4, high_threshold=8, model_mapping={
        "low": "model-low", "medium": "model-medium", "high": "model-high"
    })
    return Triage(config)


@pytest.mark.asyncio
async def test_get_complexity_valid_json(triage):
    content = json.dumps({"complexity_score": 6, "reasoning": "multi-step task"})
    with patch("router.litellm.acompletion", new=AsyncMock(return_value=make_litellm_response(content))):
        result = await triage.get_complexity("Write a function to sort a list.")
    assert result.complexity_score == 6
    assert result.reasoning == "multi-step task"


@pytest.mark.asyncio
async def test_get_complexity_malformed_json_falls_back(triage):
    with patch("router.litellm.acompletion", new=AsyncMock(return_value=make_litellm_response("not json"))):
        result = await triage.get_complexity("Hello")
    assert result.complexity_score == 5
    assert result.reasoning == "Fallback due to system error"


@pytest.mark.asyncio
async def test_get_complexity_missing_fields_falls_back(triage):
    content = json.dumps({"complexity_score": 11})  # out of 0-10 range, missing reasoning
    with patch("router.litellm.acompletion", new=AsyncMock(return_value=make_litellm_response(content))):
        result = await triage.get_complexity("Hello")
    assert result.complexity_score == 5
    assert result.reasoning == "Fallback due to system error"


@pytest.mark.asyncio
async def test_get_complexity_exception_falls_back(triage):
    with patch("router.litellm.acompletion", new=AsyncMock(side_effect=Exception("API down"))):
        result = await triage.get_complexity("Hello")
    assert result.complexity_score == 5
    assert result.reasoning == "Fallback due to system error"
