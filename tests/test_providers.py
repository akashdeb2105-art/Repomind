"""Unit tests for the provider fallback layer.

Every test here is offline: HTTP is mocked with respx, so CI never needs an API
key and never touches a free-tier quota.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from repomind.agent.providers import (
    DEFAULT_ORDER,
    PROVIDER_CONFIGS,
    AllProvidersFailed,
    LLMRouter,
    Provider,
    ProviderError,
)

MESSAGES = [{"role": "user", "content": "ping"}]


def _chat_payload(text: str, model: str = "test-model") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4},
    }


def _route(provider: str) -> str:
    return f"{PROVIDER_CONFIGS[provider].base_url}/chat/completions"


@pytest.fixture
def all_keys(monkeypatch):
    for config in PROVIDER_CONFIGS.values():
        monkeypatch.setenv(config.api_key_env, "test-key")


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


def test_provider_is_unavailable_without_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert Provider(PROVIDER_CONFIGS["groq"]).available is False


def test_provider_parses_a_successful_response(all_keys):
    with respx.mock:
        respx.post(_route("groq")).mock(
            return_value=httpx.Response(200, json=_chat_payload("pong"))
        )
        reply = Provider(PROVIDER_CONFIGS["groq"]).complete(MESSAGES)

    assert reply.text == "pong"
    assert reply.provider == "groq"
    assert reply.total_tokens == 15
    assert reply.latency_s >= 0


def test_rate_limit_is_retryable(all_keys):
    with respx.mock:
        respx.post(_route("groq")).mock(return_value=httpx.Response(429, text="slow down"))
        with pytest.raises(ProviderError) as exc:
            Provider(PROVIDER_CONFIGS["groq"]).complete(MESSAGES)

    assert exc.value.retryable is True
    assert exc.value.status_code == 429


def test_bad_credentials_are_not_retryable(all_keys):
    with respx.mock:
        respx.post(_route("groq")).mock(return_value=httpx.Response(401, text="bad key"))
        with pytest.raises(ProviderError) as exc:
            Provider(PROVIDER_CONFIGS["groq"]).complete(MESSAGES)

    assert exc.value.retryable is False


def test_malformed_body_is_reported_as_a_provider_error(all_keys):
    with respx.mock:
        respx.post(_route("groq")).mock(return_value=httpx.Response(200, json={"nope": True}))
        with pytest.raises(ProviderError, match="unexpected response shape"):
            Provider(PROVIDER_CONFIGS["groq"]).complete(MESSAGES)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


def test_router_prefers_the_first_provider(all_keys):
    with respx.mock:
        respx.post(_route("groq")).mock(
            return_value=httpx.Response(200, json=_chat_payload("from groq"))
        )
        reply = LLMRouter(base_backoff=0).complete(MESSAGES)

    assert reply.provider == "groq"


def test_router_falls_back_when_the_primary_rate_limits(all_keys):
    """The behaviour the brief cares about: the primary 429s, the next one answers."""
    primary, secondary = DEFAULT_ORDER[0], DEFAULT_ORDER[1]

    with respx.mock:
        first = respx.post(_route(primary)).mock(return_value=httpx.Response(429, text="limit"))
        respx.post(_route(secondary)).mock(
            return_value=httpx.Response(200, json=_chat_payload("from the fallback"))
        )
        reply = LLMRouter(max_attempts_per_provider=2, base_backoff=0).complete(MESSAGES)

    assert reply.provider == secondary
    assert reply.text == "from the fallback"
    assert first.call_count == 2, "should exhaust its retries before failing over"


def test_router_retries_then_succeeds_on_the_same_provider(all_keys):
    with respx.mock:
        respx.post(_route("groq")).mock(
            side_effect=[
                httpx.Response(503, text="unavailable"),
                httpx.Response(200, json=_chat_payload("recovered")),
            ]
        )
        reply = LLMRouter(base_backoff=0).complete(MESSAGES)

    assert reply.provider == "groq"
    assert reply.attempts == 2


def test_router_skips_providers_with_no_key(monkeypatch):
    for config in PROVIDER_CONFIGS.values():
        monkeypatch.delenv(config.api_key_env, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    with respx.mock:
        groq = respx.post(_route("groq")).mock(
            return_value=httpx.Response(200, json=_chat_payload("x"))
        )
        respx.post(_route("gemini")).mock(
            return_value=httpx.Response(200, json=_chat_payload("from gemini"))
        )
        reply = LLMRouter(base_backoff=0).complete(MESSAGES)

    assert reply.provider == "gemini"
    assert groq.call_count == 0, "a keyless provider must not be called at all"


def test_force_fail_proves_the_fallback_drill(all_keys):
    with respx.mock:
        groq = respx.post(_route("groq")).mock(
            return_value=httpx.Response(200, json=_chat_payload("x"))
        )
        respx.post(_route(DEFAULT_ORDER[1])).mock(
            return_value=httpx.Response(200, json=_chat_payload("from the fallback"))
        )
        reply = LLMRouter(base_backoff=0, force_fail={"groq"}).complete(MESSAGES)

    assert reply.provider == DEFAULT_ORDER[1], "the drill lands on whoever is next in the chain"
    assert groq.call_count == 0


def test_router_raises_when_everything_fails(all_keys):
    with respx.mock:
        for name in PROVIDER_CONFIGS:
            respx.post(_route(name)).mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(AllProvidersFailed) as exc:
            LLMRouter(max_attempts_per_provider=1, base_backoff=0).complete(MESSAGES)

    assert set(exc.value.failures) == set(PROVIDER_CONFIGS)


def test_unknown_provider_name_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        LLMRouter(order=["groq", "not-a-provider"])


def test_provider_order_can_be_set_by_env(monkeypatch, all_keys):
    monkeypatch.setenv("REPOMIND_PROVIDER_ORDER", "openrouter,groq")
    assert [p.name for p in LLMRouter().providers] == ["openrouter", "groq"]


# --------------------------------------------------------------------------- #
# Retry-After: providers tell us how long to wait, so wait that long
# --------------------------------------------------------------------------- #


def test_retry_after_is_read_from_the_header(all_keys):
    with respx.mock:
        respx.post(_route("groq")).mock(
            return_value=httpx.Response(429, text="slow down", headers={"retry-after": "4"})
        )
        with pytest.raises(ProviderError) as exc:
            Provider(PROVIDER_CONFIGS["groq"]).complete(MESSAGES)

    assert exc.value.retry_after_s == 4.0


def test_retry_after_is_parsed_from_groq_error_text(all_keys):
    """Groq's free tier states the wait in the body, not a header."""
    body = (
        '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` on tokens '
        "per minute (TPM): Limit 8000, Used 6578, Requested 2211. Please try again in 5.9175s."
        '","code":"rate_limit_exceeded"}}'
    )
    with respx.mock:
        respx.post(_route("groq")).mock(return_value=httpx.Response(429, text=body))
        with pytest.raises(ProviderError) as exc:
            Provider(PROVIDER_CONFIGS["groq"]).complete(MESSAGES)

    assert exc.value.retry_after_s == pytest.approx(5.9175)


def test_a_short_rate_limit_is_waited_out_on_the_same_provider(all_keys, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("repomind.agent.providers.time.sleep", slept.append)

    with respx.mock:
        respx.post(_route("groq")).mock(
            side_effect=[
                httpx.Response(429, text="Please try again in 2.5s"),
                httpx.Response(200, json=_chat_payload("recovered")),
            ]
        )
        reply = LLMRouter().complete(MESSAGES)

    assert reply.provider == "groq", "a 2.5s wait is cheaper than failing over"
    assert slept and 2.0 < slept[0] < 4.0, f"should wait about as long as asked, got {slept}"


def test_a_long_rate_limit_fails_over_immediately(all_keys, monkeypatch):
    """Waiting 40 seconds on the primary is worse than using the secondary now."""
    slept: list[float] = []
    monkeypatch.setattr("repomind.agent.providers.time.sleep", slept.append)

    with respx.mock:
        groq = respx.post(_route("groq")).mock(
            return_value=httpx.Response(429, text="Please try again in 39.9s")
        )
        respx.post(_route(DEFAULT_ORDER[1])).mock(
            return_value=httpx.Response(200, json=_chat_payload("from the fallback"))
        )
        reply = LLMRouter().complete(MESSAGES)

    assert reply.provider == DEFAULT_ORDER[1]
    assert groq.call_count == 1, "no point retrying a limit we refuse to wait out"
    assert slept == [], "and no sleeping before failing over"


def test_adding_providers_needs_no_code_beyond_their_config(all_keys):
    """The README claims adding a provider is a config entry. This asserts it."""
    from repomind.agent.providers import DEFAULT_ORDER

    assert {"nararouter", "nvidia"} <= set(PROVIDER_CONFIGS)
    assert DEFAULT_ORDER == ("groq", "gemini", "nvidia", "nararouter", "openrouter")

    target = "nararouter"
    with respx.mock:
        for name in DEFAULT_ORDER[: DEFAULT_ORDER.index(target)]:
            respx.post(_route(name)).mock(return_value=httpx.Response(429, text="limit"))
        respx.post(_route(target)).mock(
            return_value=httpx.Response(200, json=_chat_payload("from nararouter"))
        )
        reply = LLMRouter(max_attempts_per_provider=1, base_backoff=0).complete(MESSAGES)

    assert reply.provider == "nararouter", "reachable wherever it sits in the chain"


def test_the_chain_still_reaches_the_end_when_three_providers_fail(all_keys):
    with respx.mock:
        for name in DEFAULT_ORDER[:-1]:
            respx.post(_route(name)).mock(return_value=httpx.Response(503, text="down"))
        respx.post(_route("openrouter")).mock(
            return_value=httpx.Response(200, json=_chat_payload("last resort"))
        )
        reply = LLMRouter(max_attempts_per_provider=1, base_backoff=0).complete(MESSAGES)

    assert reply.provider == "openrouter"
