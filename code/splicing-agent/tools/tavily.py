from dataclasses import dataclass
import os
from langsmith import traceable
from splicing_agent.state import SplicingAgentState
from splicing_agent.helpers import _trace, _tool_call, _error
from splicing_agent.failure_taxonomy_local import _add_failure
from splicing_agent.config import HAVE_TAVILY

from typing import List, Dict, Any
from dataclasses import dataclass
import os

from splicing_agent.state import SplicingAgentState
from splicing_agent.helpers import _trace, _tool_call, _error
from splicing_agent.failure_taxonomy_local import _add_failure
from splicing_agent.config import HAVE_TAVILY

@dataclass
class tavily_tool:
    max_results: int = 5
    search_depth: str = "basic"

    def run(self, state: SplicingAgentState) -> SplicingAgentState:

        _tool_call(state, "TAVILY")

        _trace(
            state,
            f"TAVILY tool invoked | HAVE_TAVILY={HAVE_TAVILY} "
            f"| API_KEY_SET={bool(os.getenv('TAVILY_API_KEY','').strip())}"
        )

        # ============================================================
        # API key check (deterministic failure)
        # ============================================================

        api_key = os.getenv("TAVILY_API_KEY", "").strip()

        if not api_key:
            state["literature_notes"] = []

            _add_failure(
                state,
                "INPUT_MISSING",
                "TAVILY_API_KEY missing; cannot run Tavily search.",
            )

            _trace(state, "tavily_tool: done (missing key)")
            return state

        # ============================================================
        # External search execution
        # ============================================================

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)

            gene = (state.get("gene_symbol_hint") or "").strip()
            tx = (state.get("transcript_id") or "").strip()

            query = (
                f"{gene} aberrant splicing NMD premature termination codon "
                f"transcript {tx}"
            ).strip()

            res = client.search(
                query=query,
                max_results=int(self.max_results),
                search_depth=self.search_depth,
                include_answer=False,
                include_raw_content=False,
            )

            results = res.get("results") or []

            notes: List[Dict[str, Any]] = []

            for r in results:
                notes.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content"),
                    "score": r.get("score"),
                })

            state["literature_notes"] = notes

            if not notes:
                _add_failure(
                    state,
                    "NO_LITERATURE_FOUND",
                    "Tavily search returned 0 results for the query.",
                )

            _trace(state, f"tavily_tool: done | n_results={len(notes)}")
            return state

        # ============================================================
        # Error handling (external failure)
        # ============================================================

        except Exception as e:

            state["literature_notes"] = []

            _error(
                state,
                f"tavily search failed: {type(e).__name__}: {e}",
            )

            _add_failure(
                state,
                "NO_LITERATURE_FOUND",
                "Tavily search errored; see errors for details.",
            )

            _trace(state, "tavily_tool: done (error)")
            return state