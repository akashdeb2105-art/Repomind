"""Structured-output handling: extraction, validation, retry."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from repomind.agent.llm import StructuredOutputError, extract_json, structured_call
from repomind.agent.providers import AllProvidersFailed, LLMResponse


class Answer(BaseModel):
    value: str
    count: int


def reply(text: str) -> LLMResponse:
    return LLMResponse(text=text, provider="fake", model="m", prompt_tokens=1, completion_tokens=1)


class ScriptedRouter:
    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.calls = 0

    def complete(self, messages, **kwargs) -> LLMResponse:
        self.calls += 1
        return reply(self.texts[min(self.calls - 1, len(self.texts) - 1)])


# --------------------------------------------------------------------------- #
# extract_json
# --------------------------------------------------------------------------- #


def test_plain_json_passes_through():
    assert json.loads(extract_json('{"value": "x", "count": 1}'))["value"] == "x"


def test_json_wrapped_in_a_fence_is_unwrapped():
    assert json.loads(extract_json('```json\n{"value": "x", "count": 1}\n```'))["count"] == 1


def test_json_containing_a_mermaid_fence_survives():
    """The bug the pipeline tests caught: a fence *inside* the JSON string."""
    payload = {"value": "```mermaid\nflowchart TD\n  A --> B\n```", "count": 2}
    raw = json.dumps(payload)

    assert json.loads(extract_json(raw)) == payload


def test_json_with_surrounding_prose_is_recovered():
    text = 'Sure! Here is the result:\n{"value": "x", "count": 3}\nHope that helps.'

    assert json.loads(extract_json(text))["count"] == 3


# --------------------------------------------------------------------------- #
# structured_call
# --------------------------------------------------------------------------- #


def test_valid_first_response_needs_one_call():
    router = ScriptedRouter('{"value": "ok", "count": 1}')

    result = structured_call(router, [{"role": "user", "content": "go"}], Answer)  # type: ignore[arg-type]

    assert result.value == "ok"
    assert router.calls == 1


def test_malformed_output_is_retried_with_the_error_fed_back():
    router = ScriptedRouter("not json at all", '{"value": "recovered", "count": 2}')

    result = structured_call(router, [{"role": "user", "content": "go"}], Answer)  # type: ignore[arg-type]

    assert result.value == "recovered"
    assert router.calls == 2


def test_persistent_garbage_raises_rather_than_returning_junk():
    """A node that proceeds on unparsed output writes a confident, wrong document."""
    router = ScriptedRouter("still not json")

    with pytest.raises(StructuredOutputError, match="Answer"):
        structured_call(router, [{"role": "user", "content": "go"}], Answer, max_attempts=2)  # type: ignore[arg-type]

    assert router.calls == 2


def test_schema_violations_count_as_failures():
    """Valid JSON with the wrong shape must not slip through."""
    router = ScriptedRouter('{"value": "x"}', '{"value": "x", "count": 7}')

    result = structured_call(router, [{"role": "user", "content": "go"}], Answer)  # type: ignore[arg-type]

    assert result.count == 7


def test_provider_exhaustion_is_reported_clearly():
    class DeadRouter:
        def complete(self, messages, **kwargs):
            raise AllProvidersFailed({"groq": "429", "gemini": "503"})

    with pytest.raises(StructuredOutputError, match="no provider could answer"):
        structured_call(DeadRouter(), [{"role": "user", "content": "go"}], Answer)  # type: ignore[arg-type]


def test_usage_is_collected_across_retries():
    router = ScriptedRouter("bad", '{"value": "x", "count": 1}')
    sink: list[LLMResponse] = []

    structured_call(router, [{"role": "user", "content": "go"}], Answer, usage_sink=sink)  # type: ignore[arg-type]

    assert len(sink) == 2, "a retried call still spent tokens and must be counted"
