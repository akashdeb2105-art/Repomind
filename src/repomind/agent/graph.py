"""LangGraph wiring: Explorer → Deep-Dive → Synthesizer → Critic.

The graph is deliberately thin. All the behaviour lives in `nodes.py` as plain
functions, so the pipeline can be tested — and reasoned about — without a graph
framework in the way. LangGraph contributes the state merging, the execution
order, and a structure that stays readable when Phase 3 adds a second entry
point for Q&A.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from repomind.agent.nodes import (
    make_critic,
    make_deep_dive,
    make_explorer,
    make_synthesizer,
)
from repomind.agent.providers import LLMRouter
from repomind.agent.state import AgentState, Evidence
from repomind.models import RunUsage
from repomind.tools import RepoContext

logger = logging.getLogger("repomind.graph")


def build_graph(repo: RepoContext, router: LLMRouter, use_llm_critic: bool = True):
    """Compile the documentation pipeline. Requires the `agent` extra."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install extras
        raise RuntimeError(
            "LangGraph is not installed. Install the agent extra:  pip install -e '.[agent]'"
        ) from exc

    graph = StateGraph(AgentState)
    graph.add_node("explorer", make_explorer(repo, router))
    graph.add_node("deep_dive", make_deep_dive(repo, router))
    graph.add_node("synthesizer", make_synthesizer(router))
    graph.add_node("critic", make_critic(router, use_llm=use_llm_critic))

    graph.add_edge(START, "explorer")
    graph.add_edge("explorer", "deep_dive")
    graph.add_edge("deep_dive", "synthesizer")
    graph.add_edge("synthesizer", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


def initial_state(repo: RepoContext) -> AgentState:
    return {
        "repo_path": str(repo.root),
        "evidence": Evidence(),
        "repo_map": None,
        "file_notes": [],
        "draft": None,
        "verified": None,
        "critic_report": None,
        "claims": [],
        "usage": RunUsage(),
        "errors": [],
        "llm_responses": [],
    }


def run_pipeline(
    repo: RepoContext,
    router: LLMRouter,
    use_llm_critic: bool = True,
    use_langgraph: bool = True,
) -> AgentState:
    """Run the four nodes in order and return the final state.

    Falls back to calling the nodes directly when LangGraph is unavailable, so
    the pipeline stays usable (and testable) without the optional extra.
    """
    started = time.perf_counter()
    state = initial_state(repo)

    if not use_langgraph:
        final = _run_without_langgraph(repo, router, state, use_llm_critic)
        final["usage"] = _summarise_usage(final.get("llm_responses", []), started)
        return final

    try:
        compiled = build_graph(repo, router, use_llm_critic)
    except RuntimeError:
        logger.warning("LangGraph unavailable — running nodes directly")
        final = _run_without_langgraph(repo, router, state, use_llm_critic)
    else:
        final = dict(compiled.invoke(state))
        # LangGraph copies state between nodes; keep the mutable ledger and the
        # usage log that the nodes appended to on the way through.
        final.setdefault("evidence", state["evidence"])
        final["llm_responses"] = state["llm_responses"]

    final["usage"] = _summarise_usage(final.get("llm_responses", []), started)
    return final  # type: ignore[return-value]


def _run_without_langgraph(
    repo: RepoContext, router: LLMRouter, state: AgentState, use_llm_critic: bool
) -> AgentState:
    for node in (
        make_explorer(repo, router),
        make_deep_dive(repo, router),
        make_synthesizer(router),
        make_critic(router, use_llm=use_llm_critic),
    ):
        state.update(node(state))  # type: ignore[typeddict-item]
    return state


def _summarise_usage(responses: list, started: float) -> RunUsage:
    return RunUsage(
        provider_calls=[r.provider for r in responses],
        prompt_tokens=sum(r.prompt_tokens for r in responses),
        completion_tokens=sum(r.completion_tokens for r in responses),
        wall_clock_s=round(time.perf_counter() - started, 2),
    )


def write_documents(state: AgentState, output_dir: str | Path) -> list[Path]:
    """Write the verified documents to disk. Returns the paths written."""
    draft = state.get("verified") or state.get("draft")
    if draft is None:
        return []

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    for name, content in (
        ("ONBOARDING.md", draft.onboarding_md),
        ("ARCHITECTURE.md", draft.architecture_md),
    ):
        path = directory / name
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        written.append(path)

    return written
