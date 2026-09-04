# RepoMind

> An MCP server + LangGraph agent that makes any GitHub repository self-explaining —
> it generates an onboarding guide, an architecture diagram, and answers questions
> about the codebase. Runs entirely on free-tier LLM APIs, at zero cost.

**Status: in development.** Phase 0 of 6 complete (scaffold + provider fallback).
The full README — architecture diagram, benchmark table, and design-decisions
section — lands in Phase 6.

## The problem

Every unfamiliar repository costs a developer hours before they write a line of
code: where is the entry point, what talks to what, how do I run the tests.
Existing "AI explains your code" tools tend to hallucinate file paths that don't
exist, which makes their output worse than nothing. RepoMind's answer is a
verification pass — a Critic node that checks every factual claim in the
generated docs against real tool-call results and strips anything ungrounded.

## Development

```bash
git clone https://github.com/akashdeb2105-art/Repomind.git
cd Repomind
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -e ".[dev]"

cp .env.example .env      # then paste your free API keys into .env
pytest                    # offline test suite, no keys needed
python scripts/check_providers.py   # live check against the free tiers
```

## Providers

RepoMind never depends on one vendor. Requests go to Groq first, and fall
through to Gemini, then OpenRouter, on rate limits or errors. All three are
free tiers requiring no credit card.

| Order | Provider | Default model | Free-tier key |
|---|---|---|---|
| 1 | Groq | `llama-3.3-70b-versatile` | [console.groq.com](https://console.groq.com) |
| 2 | Google Gemini | `gemini-2.0-flash-lite` | [aistudio.google.com](https://aistudio.google.com) |
| 3 | OpenRouter | `meta-llama/llama-3.3-70b-instruct:free` | [openrouter.ai](https://openrouter.ai) |

## License

MIT — see [LICENSE](LICENSE).
