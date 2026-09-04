"""Structured LLM calls: ask for JSON, validate against Pydantic, retry on failure.

Free-tier models do not reliably support native JSON modes, so RepoMind asks
for JSON in the prompt and validates what comes back. That makes malformed
output a certainty rather than a possibility, which is exactly why the retry
loop feeds the validation error back to the model instead of just trying again:
a model told *what* was wrong usually fixes it on the next attempt.

After the retries are exhausted the call raises rather than returning a
half-parsed object. A node that silently proceeds on garbage produces a
confident, wrong document — the failure mode this whole project exists to avoid.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from repomind.agent.providers import AllProvidersFailed, LLMResponse, LLMRouter

logger = logging.getLogger("repomind.llm")

ModelT = TypeVar("ModelT", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(RuntimeError):
    """The model never produced output matching the requested schema."""


def _is_json(candidate: str) -> bool:
    try:
        json.loads(candidate)
    except (ValueError, TypeError):
        return False
    return True


def extract_json(text: str) -> str:
    """Pull a JSON object out of a reply that may be wrapped in prose or fences.

    Order matters. An earlier version searched for a ``` fence first, which
    broke the moment the Synthesizer returned valid JSON *containing* a Mermaid
    diagram — the regex matched the ```mermaid fence inside the JSON string and
    handed a flowchart to the parser. So: try the whole reply first, then only
    accept a fenced block that actually looks like JSON.
    """
    stripped = text.strip()
    if _is_json(stripped):
        return stripped

    for fenced in _FENCE.finditer(text):
        candidate = fenced.group(1).strip()
        if candidate.startswith(("{", "[")) and _is_json(candidate):
            return candidate

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def structured_call(
    router: LLMRouter,
    messages: Sequence[dict[str, Any]],
    schema: type[ModelT],
    *,
    max_attempts: int = 3,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    usage_sink: list[LLMResponse] | None = None,
) -> ModelT:
    """Call the LLM and return a validated `schema` instance."""
    conversation = list(messages)
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            response = router.complete(conversation, max_tokens=max_tokens, temperature=temperature)
        except AllProvidersFailed as exc:
            raise StructuredOutputError(f"no provider could answer: {exc}") from exc

        if usage_sink is not None:
            usage_sink.append(response)

        raw = extract_json(response.text)
        try:
            return schema.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)[:600]
            logger.warning(
                "schema validation failed (attempt %d/%d): %s", attempt, max_attempts, last_error
            )
            if attempt == max_attempts:
                break
            # Feed the error back: the model usually corrects itself when told
            # precisely what was wrong with its previous answer.
            conversation = [
                *messages,
                {"role": "assistant", "content": response.text[:2000]},
                {
                    "role": "user",
                    "content": (
                        "That did not match the required schema.\n"
                        f"Validation error:\n{last_error}\n\n"
                        "Reply with corrected JSON only — no prose, no code fences."
                    ),
                },
            ]

    raise StructuredOutputError(
        f"{schema.__name__} validation failed after {max_attempts} attempts: {last_error}"
    )


def schema_hint(schema: type[BaseModel]) -> str:
    """A compact JSON-schema description to paste into a prompt."""
    return json.dumps(schema.model_json_schema(), indent=2)
