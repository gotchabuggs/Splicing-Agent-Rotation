from langsmith import traceable
from splicing_agent.state import SplicingAgentState
from splicing_agent.helpers import _trace, _add_failure, _error, _extract_first_json_object
from splicing_agent.failure_taxonomy_local import _add_failure
import json

def node_agent_router(state: SplicingAgentState, planner_llm=None) -> SplicingAgentState:
    """
    LLM-based planner.
    
    - Uses system_prompt (strict/minimal/loose) to guide behavior.
    - Chooses next tool.
    - Hard constraint: NMD must run before TABLES.
    """

    _trace(state, "AGENT (LLM planner): start")

    done = state.get("plan_done", []) or []
    llm = planner_llm

    _trace(state, f"DEBUG planner_llm is None? {llm is None}")

    # Hard enforcement: NMD must happen at least once before TABLES
    if "NMD" not in done:
        state["next_tool"] = "NMD"
        state["router_reason"] = "Hard constraint: NMD must run before any finalization."
        _trace(state, "AGENT forced NMD (hard constraint)")
        return state

    # If LLM not available → fallback deterministic
    if llm is None:
        raise RuntimeError("Planner LLM is required but not initialized.")

    # updating build reasoning context for LLM
    available = ["MOTIF", "TAVILY", "TABLES"]
    # Only offer tools that aren't already done (except TABLES, which we allow once)
    available = [t for t in available if t not in done]
    # Always allow TABLES as the terminal step
    if "TABLES" not in available:
        available.append("TABLES")

    planner_input = {
        "plan_done": done,
        "ptc_predicted": state.get("ptc_predicted"),
        "nmd": state.get("nmd"),
        "failure_modes": state.get("failure_modes", []),
        "available_tools": available,
    }

    prompt = f"""
{state.get("system_prompt")}

You are planning tool usage in a splicing analysis pipeline.

Available tools:
{chr(10).join([f"- {t}" for t in planner_input["available_tools"]])}

You must select exactly one next tool from available_tools.

Constraints:
- NMD must run before TABLES.
- Do not repeat tools already completed.
- Use ONLY the evidence below.

Evidence:
{json.dumps(planner_input, indent=2)}

Return JSON:
{{
  "next_tool": "tool_name",
  "reason": "short explanation"
}}
""".strip()

    try:
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", "").strip()
        parsed = _extract_first_json_object(content)
        if parsed is None:
            raise json.JSONDecodeError("Planner JSON parse failed", content, 0)

        nxt = str(parsed.get("next_tool", "TABLES")).strip().upper()
        reason = str(parsed.get("reason", "")).strip()

        allowed = set(t.upper() for t in planner_input["available_tools"])
        if nxt not in allowed:
            _add_failure(state, "ROUTER_CONFLICT", f"Planner chose invalid tool: {nxt}")
            nxt = "TABLES"
            reason = f"Overridden: invalid tool returned by planner."

        # do not repeat completed tools (except TABLES)
        if nxt in (t.upper() for t in done) and nxt != "TABLES":
            nxt = "TABLES"
            reason = f"Overridden: planner attempted to repeat {nxt}."

        state["next_tool"] = nxt
        state["router_reason"] = reason

        _trace(state, f"AGENT LLM chose → {nxt} | reason={reason}")
        return state

    except Exception as e:
        _error(state, f"LLM planner failed: {e}")
        state["next_tool"] = "TABLES"
        state["router_reason"] = "Planner failed; defaulting to TABLES."
        return state


def route_from_agent(state: SplicingAgentState) -> str:
    nxt = state.get("next_tool", "TABLES")
    if nxt in ("NMD", "MOTIF", "TAVILY", "TABLES"):
        return nxt
    _add_failure(state, "ROUTER_CONFLICT", f"Unknown router target: {nxt}")
    return "TABLES"