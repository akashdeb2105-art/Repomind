"""List the models each provider actually offers today.

Model catalogues churn — names get deprecated, free tiers get reshuffled.
Rather than hardcoding a guess, ask each provider and pick from the answer.

Run:  python scripts/list_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repomind.agent.providers import PROVIDER_CONFIGS  # noqa: E402

# Models we would never want for code reasoning, whatever the provider calls them.
NOISE = ("whisper", "tts", "embed", "guard", "moderation", "image", "vision", "audio", "video")

# Cheap heuristic for "probably good at code, probably fast enough for an agent loop".
PREFERRED = ("instant", "flash", "lite", "8b", "9b", "12b", "oss", "coder", "mini", "small")


def fetch(name: str) -> list[dict]:
    config = PROVIDER_CONFIGS[name]
    key = os.getenv(config.api_key_env)
    if not key:
        print(f"\n### {name}: no {config.api_key_env} set — skipping")
        return []

    try:
        response = httpx.get(
            f"{config.base_url}/models",
            headers={"Authorization": f"Bearer {key}", **config.extra_headers},
            timeout=30,
        )
    except httpx.HTTPError as exc:
        print(f"\n### {name}: request failed — {exc}")
        return []

    if response.status_code >= 400:
        print(f"\n### {name}: HTTP {response.status_code} — {response.text[:200]}")
        return []

    return response.json().get("data", [])


def main() -> int:
    load_dotenv()

    for name in PROVIDER_CONFIGS:
        models = fetch(name)
        if not models:
            continue

        ids = sorted(m.get("id", "") for m in models)
        ids = [i for i in ids if i and not any(n in i.lower() for n in NOISE)]

        if name == "openrouter":
            # Only ':free' variants cost nothing, which is the whole point here.
            ids = [i for i in ids if i.endswith(":free")]

        picks = [i for i in ids if any(p in i.lower() for p in PREFERRED)]

        print(f"\n### {name}  ({len(ids)} usable models)")
        print("\n--- likely good defaults ---")
        for i in picks[:15]:
            print(f"  {i}")
        print("\n--- everything else ---")
        for i in ids:
            if i not in picks:
                print(f"  {i}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
