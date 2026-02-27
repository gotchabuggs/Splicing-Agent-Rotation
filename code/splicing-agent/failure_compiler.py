from langsmith import traceable
from splicing_agent.state import SplicingAgentState
from splicing_agent.helpers import _trace, _add_failure, _error, compute_predicted_label, infer_expected_label_from_state, normalize_label_bucket, compute_task_completed, compute_tool_usage_accuracy, compute_has_critical_failure, compute_success_and_error_flag
from splicing_agent.failure_taxonomy_local import FAILURE_TIER, CRITICAL_TIERS
try:
    from splicing_agent.failure_packet import build_failure_packet, HAVE_FAILURE_PACKET # type: ignore
except ImportError:
    HAVE_FAILURE_PACKET = False


@traceable(name="FAILURE_COMPILER", run_type="chain")
def node_failure_compiler(state: SplicingAgentState) -> SplicingAgentState:
    """
    Single source of truth for:
      - failure tiers + criticality
      - predicted_label
      - task_completed/tool_usage_accuracy/success/error_rate_flag
      - optional descriptive failure_packet (if failure_taxonomy.py is importable)
    """
    _trace(state, "failure_compiler: start")

    # normalize failures
    modes = sorted(set(state.get("failure_modes", []) or []))
    state["failure_modes"] = modes
    state["failure_tiers"] = {m: int(FAILURE_TIER.get(m, 0)) for m in modes}
    state["has_critical_failure"] = any(state["failure_tiers"].get(m) in CRITICAL_TIERS for m in modes)

    # systemic: required tool calls
    done = state.get("plan_done", []) or []
    if "NMD" not in done:
        _add_failure(state, "TOOL_NOT_RUN", "NMD was not executed but is required.")
        state["failure_modes"] = sorted(set(state.get("failure_modes", []) or []))
        state["failure_tiers"] = {m: int(FAILURE_TIER.get(m, 0)) for m in state["failure_modes"]}
        # Treat as critical because hard constraint violated
        state["has_critical_failure"] = True

    # basic state consistency check
    try:
        tx_len = state.get("tx_len")
        stop_end = state.get("stop_end_tx")
        if tx_len is not None and stop_end is not None and int(stop_end) > int(tx_len):
            _add_failure(state, "STATE_INCONSISTENT", "stop_end_tx exceeds tx_len.")
    except Exception:
        pass

    # derived labels (detailed + bucketed)
    state["predicted_label"] = compute_predicted_label(state)
    state["expected_label"] = infer_expected_label_from_state(state)

    state["predicted_label_bucket"] = normalize_label_bucket(state.get("predicted_label", ""))
    state["expected_label_bucket"] = normalize_label_bucket(state.get("expected_label", ""))
    # Deterministic metric flags (authoritative)
    state["task_completed"] = compute_task_completed(state)
    state["tool_usage_accuracy"] = compute_tool_usage_accuracy(state)

    # Compute critical failure with your custom rule (skip CDS_MISSING special-case)
    state["has_critical_failure"] = compute_has_critical_failure(state)

    success, error_flag = compute_success_and_error_flag(state)
    state["success"] = bool(success)
    state["error_rate_flag"] = bool(error_flag)


    # Optional: build descriptive failure packet (enriched taxonomy + next steps)
    if HAVE_FAILURE_PACKET:
        try:
            state["failure_packet"] = build_failure_packet(dict(state))  # type: ignore
        except Exception as e:
            _error(state, f"failure_packet build failed: {type(e).__name__}: {e}")
            state["failure_packet"] = {}
    else:
        state["failure_packet"] = {}

    # default hallucination (only judge can set it True)
    state["hallucination"] = bool(state.get("hallucination", False))

    _trace(
        state,
        "failure_compiler: done | "
        f"predicted_label={state.get('predicted_label')} "
        f"task_completed={state.get('task_completed')} "
        f"tool_usage_accuracy={state.get('tool_usage_accuracy')} "
        f"success={state.get('success')} error_flag={state.get('error_rate_flag')}"
    )
    return state
