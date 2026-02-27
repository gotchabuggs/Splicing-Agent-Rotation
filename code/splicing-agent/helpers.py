from typing import Any, Dict, List, Tuple, Optional
import hashlib
import os
import re
import pandas as pd

from splicing_agent.state import SplicingAgentState
from splicing_agent.failure_taxonomy_local import CRITICAL_TIERS


# ============================================================
# Constants
# ============================================================

STOP_CODONS = {"TAA", "TAG", "TGA"}

# ============================================================
# Core Trace + Error Utilities
# ============================================================

def _trace(state: SplicingAgentState, msg: str) -> None:
    state.setdefault("trace", []).append(msg)


def _error(state: SplicingAgentState, msg: str) -> None:
    state.setdefault("errors", []).append(msg)


def _tool_call(state: SplicingAgentState, tool_name: str) -> None:
    state.setdefault("tool_calls", []).append(tool_name)


# ============================================================
# Sequence Utilities
# ============================================================

def _chunk(seq: str, n: int) -> List[str]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]


def _scan_stops_all_inframe(cds_seq: str) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for i, codon in enumerate(_chunk(cds_seq, 3)):
        if len(codon) < 3:
            break
        if codon in STOP_CODONS:
            hits.append({
                "codon": codon,
                "cds_nt_offset": i * 3,
                "cds_aa_index": i
            })
    return hits


def _split_internal_vs_terminal_stop(
    stops_all: List[Dict[str, Any]],
    cds_seq: str
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:

    if not stops_all:
        return [], None

    terminal_triplet = cds_seq[-3:] if len(cds_seq) >= 3 else ""

    if terminal_triplet in STOP_CODONS:
        terminal_offset = len(cds_seq) - 3
        terminal_hit = None
        internal: List[Dict[str, Any]] = []

        for h in stops_all:
            if h["cds_nt_offset"] == terminal_offset:
                terminal_hit = h
            else:
                internal.append(h)

        return internal, terminal_hit

    return list(stops_all), None


def _map_cds_offset_to_tx_coord(
    cds_segments: List[Tuple[int, int]],
    cds_offset_0based: int
) -> int:

    remaining = int(cds_offset_0based)

    for a, b in cds_segments:
        seg_len = int(b) - int(a) + 1
        if remaining < seg_len:
            return int(a) + remaining
        remaining -= seg_len

    raise ValueError("CDS offset exceeds CDS length")


def _stop_candidate_end_tx(
    cds_segments: List[Tuple[int, int]],
    cds_nt_offset: int
) -> int:
    return _map_cds_offset_to_tx_coord(
        cds_segments,
        int(cds_nt_offset) + 2
    )


# ============================================================
# DataFrame / Parsing Utilities
# ============================================================

def _df_to_md_table(df: pd.DataFrame) -> str:
    if df.shape[0] == 0:
        return "_(none)_"
    return df.to_markdown(index=False)


def _safe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes"}:
        return True
    if s in {"false", "f", "0", "no"}:
        return False
    return None


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    return str(x).strip() == ""


# ============================================================
# Tool Completion Tracking
# ============================================================

def mark_tool_done(state: SplicingAgentState, tool_name: str) -> SplicingAgentState:
    done = state.get("plan_done", []) or []

    if tool_name not in done:
        done.append(tool_name)

    state["plan_done"] = done

    state.setdefault("node_events", [])
    state["node_events"].append({
        "name": tool_name,
        "status": "ok"
    })

    _trace(state, f"DONE {tool_name} | plan_done={done}")

    return state


# ============================================================
# Label Logic
# ============================================================

def compute_predicted_label(state: SplicingAgentState) -> str:

    if _is_missing(state.get("stop_end_tx")):
        return "Ambiguous (CDS missing)"

    ptc = _safe_bool(state.get("ptc_predicted"))
    nmdv = _safe_bool(state.get("nmd"))

    if nmdv is True:
        return "NMD+"

    if ptc is True and nmdv is False:
        return "PTC+/NMD-"

    if ptc is False and nmdv is False:
        return "PTC-/NMD-"

    return "Ambiguous"


# ------------------------------------------------------------
# Label normalization buckets
# ------------------------------------------------------------

LABEL_BUCKET_MAP: Dict[str, str] = {
    "NMD+": "nmd",
    "PTC+/NMD-": "protein_coding",
    "PTC-/NMD-": "protein_coding",
    "Ambiguous (CDS missing)": "cds_not_defined",
    "Ambiguous": "ambiguous",

    "nmd": "nmd",
    "protein_coding": "protein_coding",
    "cds_not_defined": "cds_not_defined",
    "ambiguous": "ambiguous",

    "NMD": "nmd",
    "NMD-": "protein_coding",
}


def normalize_label_bucket(label: str) -> str:
    s = (label or "").strip()
    if not s:
        return "ambiguous"
    return LABEL_BUCKET_MAP.get(s, "ambiguous")


def infer_expected_label_from_state(state: SplicingAgentState) -> str:

    exp = (state.get("expected_label") or "").strip()

    if exp:
        state["expected_label_provided"] = True
        state["expected_label_inferred"] = False
        return exp

    state["expected_label_provided"] = False
    state["expected_label_inferred"] = True

    return compute_predicted_label(state)


# ============================================================
# Evaluation Metrics
# ============================================================

def compute_task_completed(state: SplicingAgentState) -> bool:

    tool_calls = state.get("tool_calls", [])
    plan_done = state.get("plan_done", [])

    tools_ok = all(t in plan_done for t in tool_calls) and ("TABLES" in plan_done)
    has_tables = bool(state.get("report_tables_md", ""))
    has_report = bool(state.get("report_text", ""))

    return bool(tools_ok and has_tables and has_report)


def compute_tool_usage_accuracy(state: SplicingAgentState) -> bool:

    calls = state.get("tool_calls", []) or []
    done = state.get("plan_done", []) or []

    if "CDS" not in calls or "NMD" not in calls:
        return False

    if "TABLES" not in done:
        return False

    try:
        i_cds = done.index("CDS")
        i_nmd = done.index("NMD")
        i_tables = done.index("TABLES")
        return (i_cds < i_nmd < i_tables)
    except Exception:
        return False


def compute_has_critical_failure(state: SplicingAgentState) -> bool:

    modes = state.get("failure_modes", []) or []
    tiers = state.get("failure_tiers", {}) or {}

    for code in modes:
        t = int(tiers.get(code, 0))
        if t in CRITICAL_TIERS:
            if code == "CDS_MISSING":
                continue
            return True

    return False


def compute_success_and_error_flag(state: SplicingAgentState):

    pred_det = (state.get("predicted_label") or "").strip()
    exp_raw = (state.get("expected_label") or "").strip()

    pred_bucket = normalize_label_bucket(pred_det)
    exp_bucket = normalize_label_bucket(exp_raw)

    is_scorable = bool(state.get("expected_label_provided", False))
    mismatch = (pred_bucket != exp_bucket) if is_scorable else False

    errs = state.get("errors", []) or []
    has_runtime_errors = len(errs) > 0

    has_critical = bool(state.get("has_critical_failure", False))
    task_completed = bool(state.get("task_completed", False))
    tool_ok = bool(state.get("tool_usage_accuracy", False))

    success = bool(
        task_completed
        and tool_ok
        and (not has_runtime_errors)
        and (not has_critical)
        and (not mismatch)
    )

    error_flag = bool(has_runtime_errors or has_critical or mismatch)

    state["predicted_label_bucket"] = pred_bucket
    state["expected_label_bucket"] = exp_bucket
    state["label_mismatch_bucketed"] = bool(mismatch)

    return success, error_flag


# ============================================================
# Logging Helpers
# ============================================================

def _fingerprint_file(path: str) -> Dict[str, Any]:
    p = os.path.abspath(os.path.normpath(path))
    size = None
    md5_64k = None
    try:
        size = os.path.getsize(p)
        h = hashlib.md5()
        with open(p, "rb") as f:
            h.update(f.read(64 * 1024))
        md5_64k = h.hexdigest()
    except Exception:
        pass
    return {"size_bytes": size, "md5_first64kb": md5_64k}


def dataset_fingerprint_for_log(path: str) -> Dict[str, Any]:
    return {
        "dataset_path_abs": os.path.abspath(os.path.normpath(path)),
        "fingerprint": _fingerprint_file(path),
    }


def _slug(s: str, maxlen: int = 80) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s[:maxlen] if len(s) > maxlen else s


def make_row_id(i: int, transcript_id: str = "", gene: str = "") -> str:
    base = f"row_{i+1:04d}"
    t = _slug(transcript_id)
    g = _slug(gene)

    if t and g:
        return f"{base}__{t}__{g}"
    if t:
        return f"{base}__{t}"
    return base