"""Free-tier LLM provider abstraction with ordered fallback.

RepoMind never depends on a single LLM vendor. Every free tier has rate limits,
so the router below tries providers in order (Groq -> Gemini -> OpenRouter),
retrying transient failures within a provider before moving to the next one.

All three vendors expose an OpenAI-compatible ``/chat/completions`` endpoint,
so a single request/response shape covers them. That is a deliberate design
choice: one code path, no per-vendor SDK, and adding a fourth provider is a
five-line config entry.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("repomind.providers")

Message = dict[str, Any]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ProviderError(RuntimeError):
    """A single provider failed to answer.

    ``retryable`` distinguishes "wait and try again" (rate limits, 5xx, network
    blips) from "this will never work" (bad key, unknown model), which should
    fail over to the next provider immediately instead of burning backoff time.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class AllProvidersFailed(RuntimeError):
    """Every configured provider was exhausted."""

    def __init__(self, failures: dict[str, str]) -> None:
        detail = "; ".join(f"{name}: {err}" for name, err in failures.items())
        super().__init__(f"all providers failed -> {detail}")
        self.failures = failures


# --------------------------------------------------------------------------- #
# Response model
# --------------------------------------------------------------------------- #


class LLMResponse(BaseModel):
    """Normalised result of one successful completion.

    Token counts and latency are recorded on every call because the benchmark
    table in the README is built from exactly these numbers.
    """

    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# --------------------------------------------------------------------------- #
# Provider configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    model_env: str
    docs_url: str
    extra_headers: dict[str, str] = field(default_factory=dict)


PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        model_env="REPOMIND_GROQ_MODEL",
        docs_url="https://console.groq.com",
    ),
    "gemini": ProviderConfig(
        name="gemini",
        # Google ships an OpenAI-compatible shim, which lets us reuse one client.
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-3.5-flash-lite",
        model_env="REPOMIND_GEMINI_MODEL",
        docs_url="https://aistudio.google.com",
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="z-ai/glm-5.2:free",
        model_env="REPOMIND_OPENROUTER_MODEL",
        docs_url="https://openrouter.ai",
        extra_headers={
            "HTTP-Referer": "https://github.com/akashdeb2105-art/Repomind",
            "X-Title": "RepoMind",
        },
    ),
}

DEFAULT_ORDER: tuple[str, ...] = ("groq", "gemini", "openrouter")


# --------------------------------------------------------------------------- #
# One provider
# --------------------------------------------------------------------------- #


class Provider:
    """A single OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        model: str | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.model = model or os.getenv(config.model_env) or config.default_model
        self.timeout = timeout
        self._client = client

    # -- introspection ------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def api_key(self) -> str | None:
        key = os.getenv(self.config.api_key_env)
        return key.strip() if key and key.strip() else None

    @property
    def available(self) -> bool:
        """True when an API key is present. Missing keys are skipped, not errors."""
        return self.api_key is not None

    # -- the call ----------------------------------------------------------- #

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self.available:
            raise ProviderError(
                self.name,
                f"no API key: set {self.config.api_key_env} (get one at {self.config.docs_url})",
                retryable=False,
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }

        started = time.perf_counter()
        client = self._client or httpx.Client(timeout=self.timeout)
        try:
            response = client.post(
                f"{self.config.base_url}/chat/completions", json=payload, headers=headers
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(
                self.name, f"timeout after {self.timeout}s", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"network error: {exc}", retryable=True) from exc
        finally:
            if self._client is None:
                client.close()

        latency = time.perf_counter() - started
        self._raise_for_status(response)
        return self._parse(response, latency)

    # -- helpers ------------------------------------------------------------ #

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        body = response.text[:400]
        if response.status_code == 429:
            raise ProviderError(
                self.name, f"rate limited (429): {body}", retryable=True, status_code=429
            )
        if response.status_code >= 500:
            raise ProviderError(
                self.name,
                f"server error ({response.status_code}): {body}",
                retryable=True,
                status_code=response.status_code,
            )
        # 401 / 403 / 404 / 422 -> configuration problem. Retrying cannot help.
        raise ProviderError(
            self.name,
            f"request rejected ({response.status_code}): {body}",
            retryable=False,
            status_code=response.status_code,
        )

    def _parse(self, response: httpx.Response, latency: float) -> LLMResponse:
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(self.name, "response was not JSON", retryable=True) from exc

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                self.name, f"unexpected response shape: {str(data)[:300]}", retryable=True
            ) from exc

        usage = data.get("usage") or {}
        return LLMResponse(
            text=(message.get("content") or "").strip(),
            provider=self.name,
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=round(latency, 3),
            tool_calls=message.get("tool_calls") or [],
            raw=data,
        )


# --------------------------------------------------------------------------- #
# The router
# --------------------------------------------------------------------------- #


class LLMRouter:
    """Calls providers in order, retrying transient failures, then failing over.

    Example::

        router = LLMRouter()
        reply = router.complete([{"role": "user", "content": "hi"}])
        print(reply.provider, reply.text)
    """

    def __init__(
        self,
        order: Sequence[str] | None = None,
        *,
        max_attempts_per_provider: int = 3,
        base_backoff: float = 1.0,
        max_backoff: float = 20.0,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        force_fail: frozenset[str] | set[str] = frozenset(),
    ) -> None:
        names = list(order or self._order_from_env())
        unknown = [n for n in names if n not in PROVIDER_CONFIGS]
        if unknown:
            raise ValueError(f"unknown provider(s): {unknown}")

        self.providers = [
            Provider(PROVIDER_CONFIGS[n], timeout=timeout, client=client) for n in names
        ]
        self.max_attempts_per_provider = max_attempts_per_provider
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        # Test/demo hook: pretend these providers are broken, to prove fallback works.
        self.force_fail = set(force_fail)

    @staticmethod
    def _order_from_env() -> tuple[str, ...]:
        raw = os.getenv("REPOMIND_PROVIDER_ORDER")
        if not raw:
            return DEFAULT_ORDER
        return tuple(part.strip() for part in raw.split(",") if part.strip())

    @property
    def available_providers(self) -> list[str]:
        return [p.name for p in self.providers if p.available]

    def complete(self, messages: Sequence[Message], **kwargs: Any) -> LLMResponse:
        failures: dict[str, str] = {}
        total_attempts = 0

        for provider in self.providers:
            if provider.name in self.force_fail:
                failures[provider.name] = "forced failure (fallback drill)"
                logger.warning("provider %s: forced failure, falling through", provider.name)
                continue
            if not provider.available:
                failures[provider.name] = f"no {provider.config.api_key_env} set"
                logger.info("provider %s: no API key, skipping", provider.name)
                continue

            for attempt in range(1, self.max_attempts_per_provider + 1):
                total_attempts += 1
                try:
                    result = provider.complete(messages, **kwargs)
                except ProviderError as exc:
                    failures[provider.name] = str(exc)
                    if not exc.retryable or attempt == self.max_attempts_per_provider:
                        logger.warning(
                            "provider %s failed (%s), falling through", provider.name, exc
                        )
                        break
                    delay = self._backoff(attempt)
                    logger.info(
                        "provider %s attempt %d/%d failed (%s); retrying in %.1fs",
                        provider.name,
                        attempt,
                        self.max_attempts_per_provider,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    result.attempts = total_attempts
                    failures.pop(provider.name, None)
                    return result

        raise AllProvidersFailed(failures)

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, so parallel runs don't retry in lockstep."""
        delay = min(self.base_backoff * (2 ** (attempt - 1)), self.max_backoff)
        return delay * (0.5 + random.random() / 2)
