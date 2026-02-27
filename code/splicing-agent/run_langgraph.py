"""
Splicing-Agent (LangGraph)

Enforced pipeline:
CDS → AGENT → NMD (forced first) → (MOTIF → TAVILY) → TABLES
→ FAILURE_COMPILER → JUDGE (optional LLM) → FINAL
"""

from __future__ import annotations

from typing import Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langsmith import traceable

from splicing_agent.state import SplicingAgentState
from splicing_agent.tools.cds import cds_tool
from splicing_agent.tools.nmd import nmd_tool
from splicing_agent.tools.motif import motif_tool
from splicing_agent.tools.tavily import tavily_tool
from splicing_agent.reporting import node_tables_report, node_final_report
from splicing_agent.failure_compiler import node_failure_compiler
from splicing_agent.helpers import mark_tool_done, node_llm_judge
from splicing_agent.routing import node_agent_router


# ============================================================
# ROUTING LOGIC (FORCED NMD)
# ============================================================

def route_from_agent(state: SplicingAgentState) -> str:
    """
    Enforces:
      1. NMD must run immediately after CDS
      2. MOTIF optional
      3. TAVILY optional
      4. TABLES only after NMD completed
    """

    done = state.get("plan_done", []) or []

    # HARD CONSTRAINT: NMD must run after CDS
    if "CDS" in done and "NMD" not in done:
        return "NMD"

    # After NMD, allow optional tools
    if "NMD" in done and "MOTIF" not in done:
        return "MOTIF"

    if "NMD" in done and "TAVILY" not in done:
        return "TAVILY"

    return "TABLES"


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph(
    cds: cds_tool,
    nmd: nmd_tool,
    motif: motif_tool,
    tavily: tavily_tool,
    llm: Optional[Any],
) -> Any:

    g = StateGraph(SplicingAgentState)

    # ---------------------------
    # TOOL WRAPPERS
    # ---------------------------

    @traceable(name="CDS", run_type="tool")
    def _cds(state: SplicingAgentState) -> SplicingAgentState:
        state = cds.run(state)
        return mark_tool_done(state, "CDS")

    @traceable(name="NMD", run_type="tool")
    def _nmd(state: SplicingAgentState) -> SplicingAgentState:
        state = nmd.run(state)
        return mark_tool_done(state, "NMD")

    @traceable(name="MOTIF", run_type="tool")
    def _motif(state: SplicingAgentState) -> SplicingAgentState:
        state = motif.run(state)
        return mark_tool_done(state, "MOTIF")

    @traceable(name="TAVILY", run_type="tool")
    def _tavily(state: SplicingAgentState) -> SplicingAgentState:
        state = tavily.run(state)
        return mark_tool_done(state, "TAVILY")

    @traceable(name="TABLES", run_type="chain")
    def _tables(state: SplicingAgentState) -> SplicingAgentState:
        state = node_tables_report(state)
        return mark_tool_done(state, "TABLES")

    @traceable(name="FAILURE_COMPILER", run_type="chain")
    def _failure_compiler(state: SplicingAgentState) -> SplicingAgentState:
        return node_failure_compiler(state)

    @traceable(name="JUDGE", run_type="chain")
    def _judge(state: SplicingAgentState) -> SplicingAgentState:
        return node_llm_judge(state, llm=llm)

    @traceable(name="FINAL", run_type="chain")
    def _final(state: SplicingAgentState) -> SplicingAgentState:
        return node_final_report(state)

    # ---------------------------
    # ADD NODES
    # ---------------------------

    g.add_node("CDS", _cds)
    g.add_node("AGENT", node_agent_router)
    g.add_node("NMD", _nmd)
    g.add_node("MOTIF", _motif)
    g.add_node("TAVILY", _tavily)
    g.add_node("TABLES", _tables)
    g.add_node("FAILURE_COMPILER", _failure_compiler)
    g.add_node("JUDGE", _judge)
    g.add_node("FINAL", _final)

    # ---------------------------
    # EDGES
    # ---------------------------

    g.set_entry_point("CDS")

    g.add_edge("CDS", "AGENT")

    g.add_conditional_edges(
        "AGENT",
        route_from_agent,
        {
            "NMD": "NMD",
            "MOTIF": "MOTIF",
            "TAVILY": "TAVILY",
            "TABLES": "TABLES",
        },
    )

    g.add_edge("NMD", "AGENT")
    g.add_edge("MOTIF", "AGENT")
    g.add_edge("TAVILY", "AGENT")

    g.add_edge("TABLES", "FAILURE_COMPILER")
    g.add_edge("FAILURE_COMPILER", "JUDGE")
    g.add_edge("JUDGE", "FINAL")
    g.add_edge("FINAL", END)

    return g.compile(checkpointer=MemorySaver())