"""Phase 0 acceptance check.

Proves two things the brief requires:
  1. every configured free-tier provider can actually answer a trivial prompt;
  2. the router really falls through to the next provider when one fails.

Run:  python scripts/check_providers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repomind.agent.providers import (  # noqa: E402
    PROVIDER_CONFIGS,
    AllProvidersFailed,
    LLMRouter,
    Provider,
    ProviderError,
)

# Reasoning models (gpt-oss, thinking-enabled Gemini) burn tokens on internal
# reasoning before emitting any content. Too small a budget returns an empty
# string with a healthy 200, which is worse than an error — it looks like success.
REPLY_BUDGET = 512

PROMPT = [
    {"role": "system", "content": "Answer with a single word, no punctuation."},
    {"role": "user", "content": "What is the capital of France?"},
]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def check_each_provider() -> dict[str, bool]:
    print("\n=== 1. Each provider answers on its own ===\n")
    results: dict[str, bool] = {}

    for name, config in PROVIDER_CONFIGS.items():
        provider = Provider(config)
        label = f"{name:<11} ({provider.model})"

        if not provider.available:
            print(f"  {YELLOW}SKIP{RESET}  {label}")
            hint = f"{config.api_key_env} not set — free key at {config.docs_url}"
            print(f"        {DIM}{hint}{RESET}")
            results[name] = False
            continue

        try:
            reply = provider.complete(PROMPT, max_tokens=REPLY_BUDGET)
        except ProviderError as exc:
            print(f"  {RED}FAIL{RESET}  {label}")
            print(f"        {DIM}{exc}{RESET}")
            results[name] = False
        else:
            if not reply.text:
                print(f"  {RED}FAIL{RESET}  {label}")
                print(
                    f"        {DIM}HTTP 200 but empty content ({reply.total_tokens} tokens "
                    f"spent) — raise max_tokens or pick a non-reasoning model{RESET}"
                )
                results[name] = False
                continue
            print(f"  {GREEN}OK{RESET}    {label}")
            print(
                f"        {DIM}reply={reply.text!r}  {reply.latency_s}s  "
                f"{reply.total_tokens} tokens{RESET}"
            )
            results[name] = True

    return results


def check_fallback(healthy: dict[str, bool]) -> bool:
    print("\n=== 2. Fallback triggers on failure ===\n")

    working = [name for name, ok in healthy.items() if ok]
    if len(working) < 2:
        print(f"  {YELLOW}SKIP{RESET}  need at least 2 working providers to prove fallover")
        print(f"        {DIM}currently working: {working or 'none'}{RESET}")
        return False

    primary, expected = working[0], working[1]
    print(f"  Forcing {primary!r} to fail; expecting the router to land on {expected!r}.")

    router = LLMRouter(order=working, force_fail={primary})
    try:
        reply = router.complete(PROMPT, max_tokens=REPLY_BUDGET)
    except AllProvidersFailed as exc:
        print(f"  {RED}FAIL{RESET}  router gave up: {exc}")
        return False

    if reply.provider == expected:
        print(f"  {GREEN}OK{RESET}    fell through to {reply.provider!r} -> {reply.text!r}")
        return True

    print(f"  {RED}FAIL{RESET}  expected {expected!r}, got {reply.provider!r}")
    return False


def main() -> int:
    load_dotenv()
    print("RepoMind — Phase 0 provider check")

    healthy = check_each_provider()
    fallback_ok = check_fallback(healthy)

    print("\n=== Summary ===\n")

    def status(ok: bool, yes: str, no: str) -> str:
        return f"{GREEN}{yes}{RESET}" if ok else f"{RED}{no}{RESET}"

    for name, ok in healthy.items():
        print(f"  {name:<11} {status(ok, 'working', 'not working')}")
    print(f"  {'fallback':<11} {status(fallback_ok, 'proven', 'not proven')}")

    if any(healthy.values()) and fallback_ok:
        print(f"\n{GREEN}Phase 0 acceptance: PASS{RESET}\n")
        return 0
    print(f"\n{RED}Phase 0 acceptance: INCOMPLETE{RESET} — see messages above.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
