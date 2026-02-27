"""
Splicing-Agent LangSmith Runtime
Matches test.py (test case), but with LangSmith tracing and MemorySaver checkpointer.
"""

from __future__ import annotations
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable

from splicing_agent.state import SplicingAgentState
from splicing_agent.routing import node_agent_router, route_from_agent
from splicing_agent.reporting import node_tables_report, node_final_report
from splicing_agent.failure_compiler import node_failure_compiler
from splicing_agent.tools.cds import cds_tool
from splicing_agent.tools.nmd import nmd_tool
from splicing_agent.tools.motif import motif_tool
from splicing_agent.tools.tavily import tavily_tool
from splicing_agent.helpers import mark_tool_done, node_llm_judge, make_row_id
from splicing_agent.failure_taxonomy_local import _add_failure


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

    def _cds(state: SplicingAgentState) -> SplicingAgentState:
        state = cds.run(state)
        return mark_tool_done(state, "CDS")

    def _nmd(state: SplicingAgentState) -> SplicingAgentState:
        state = nmd.run(state)
        return mark_tool_done(state, "NMD")

    def _motif(state: SplicingAgentState) -> SplicingAgentState:
        state = motif.run(state)
        return mark_tool_done(state, "MOTIF")

    def _tavily(state: SplicingAgentState) -> SplicingAgentState:
        state = tavily.run(state)
        return mark_tool_done(state, "TAVILY")

    def _tables(state: SplicingAgentState) -> SplicingAgentState:
        state = node_tables_report(state)
        return mark_tool_done(state, "TABLES")

    def _judge(state: SplicingAgentState) -> SplicingAgentState:
        return node_llm_judge(state, llm=llm)

    # ---------------------------
    # REGISTER NODES
    # ---------------------------

    g.add_node("CDS", _cds)
    g.add_node("AGENT", node_agent_router)
    g.add_node("NMD", _nmd)
    g.add_node("MOTIF", _motif)
    g.add_node("TAVILY", _tavily)
    g.add_node("TABLES", _tables)
    g.add_node("FAILURE_COMPILER", node_failure_compiler)
    g.add_node("JUDGE", _judge)
    g.add_node("FINAL", node_final_report)

    # ---------------------------
    # EDGES (IDENTICAL TO TEST CASE)
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