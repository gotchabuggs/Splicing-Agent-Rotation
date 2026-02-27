"""
Splicing-Agent — End-to-End Benchmark Test Harness

This file runs the full Splicing-Agent pipeline on one or more benchmark TSVs.
It is designed for reproducible evaluation, debugging, and benchmarking.

It is NOT a minimal library module.
It is a complete test runner that:
    • Loads BioMart-style transcript TSV data
    • Executes the LangGraph agent pipeline
    • Enforces hard execution constraints
    • Computes deterministic evaluation metrics
    • Writes per-case and run-level artifacts
    • Optionally uses an LLM for planning and narrative judgment
    • Never allows the LLM to override deterministic outcomes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIPELINE (STRICTLY ENFORCED ORDER)

CDS
→ AGENT (LLM planner or deterministic fallback)
→ NMD  (MANDATORY immediately after CDS)
→ Optional tools (MOTIF and/or TAVILY; planner-selected)
→ TABLES
→ FAILURE_COMPILER (authoritative)
→ JUDGE (optional LLM; narrative only)
→ FINAL

HARD CONSTRAINT:
- NMD MUST execute immediately after CDS.
- The router cannot skip or reorder NMD.
- If violated, this is a critical failure.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS FILE CONTAINS

1) BioMart TSV Loaders
   - Exon loader (constructs transcript coordinate system)
   - CDS segment loader
   - Sequence loader (cDNA)

2) Tool Implementations
   - CDS tool (builds transcript + CDS state)
   - NMD tool (PTC + EJC distance rule)
   - MOTIF tool (RNA structure heuristics)
   - TAVILY tool (optional literature search)

3) Agentic Routing Logic
   - LLM-based planner (optional)
   - Deterministic hard constraints
   - Explicit tool ordering enforcement

4) Deterministic Failure Compiler
   - Assigns failure tiers
   - Determines critical failures
   - Computes ALL evaluation metrics
   - Authoritative scoring layer

5) Optional LLM Judge
   - Generates narrative summary only
   - Cannot override labels, metrics, or failure status

6) Reporting + Artifact Writing
   - Per-case JSON + Markdown
   - Run-level CSV + JSON + Markdown summaries

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAILURE TAXONOMY

Each tool may append failure codes to:
    state["failure_modes"]

The FAILURE_COMPILER:
    • Deduplicates failure codes
    • Assigns tier levels
    • Flags critical failures (tiers 1–2)
    • Computes success/error flags
    • Optionally builds enriched failure packets

Failure tiers:
    Tier 1: Input / annotation errors
    Tier 2: CDS / translation errors
    Tier 3: NMD ambiguity
    Tier 4: Splicing confounds
    Tier 5: Knowledge gaps
    Tier 6: Agent/system errors

Critical failures = Tier 1 or Tier 2 (except special cases).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION METRICS (DETERMINISTIC + EXPLICIT)

task_completed:
    True iff:
        - TABLES produced report_tables_md
        - FINAL produced report_text
    False if:
        - graph.invoke crashed
        - TABLES never ran
        - report_text missing

tool_usage_accuracy:
    True iff:
        - CDS ran
        - NMD ran
        - NMD ran before TABLES
    False if:
        - NMD missing
        - TABLES before NMD
        - Required ordering violated

success:
    True iff:
        - task_completed == True
        - tool_usage_accuracy == True
        - No critical failures
        - No runtime errors
        - AND (if expected_label provided)
              predicted_label matches expected_label (bucketed)

error_rate_flag:
    True iff:
        - Any runtime error
        - Any critical failure
        - OR (expected_label provided AND mismatch)

This is a review flag, not a nuanced score.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LABEL LOGIC

Predicted label is fully deterministic:
    - NMD+ 
    - PTC+/NMD-
    - PTC-/NMD-
    - Ambiguous (CDS missing)

If expected_label is not provided in benchmark:
    → It is inferred deterministically from state.

Label comparison is bucket-normalized
(e.g., PTC+/NMD- → protein_coding).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARTIFACTS WRITTEN

Per-case:
    pretty_json/<row_id>.json
    reports_md/<row_id>.md
    reports_md/<row_id>__consequence.md
    reports_md/<row_id>__metrics.md

Run-level:
    test_case_runs.jsonl
    test_case_runs.json
    test_case_summary.csv
    failures_only.csv
    benchmark_summary.csv
    benchmark_summary.md
    run_manifest.json
    graph.mmd
    graph.png (if available)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LLM USAGE POLICY

Planner LLM:
    - Chooses optional tools
    - Must respect hard constraints
    - Cannot skip NMD

Judge LLM:
    - Produces structured JSON summary
    - May flag hallucination risk
    - Cannot modify:
        predicted_label
        success
        tool_usage_accuracy
        error_rate_flag
        failure tiers

All scoring is deterministic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTENDED USE

This file is for:
    • Benchmark experiments
    • Reproducibility testing
    • Failure-mode analysis
    • Agent routing evaluation
    • Prompt comparison (strict/minimal/loose)

It is intentionally verbose, explicit, and self-contained
to support scientific auditability and debugging.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPENDENCIES

- pandas
- langgraph
- tabulate (required for DataFrame.to_markdown)
    pip install tabulate
    or
    conda install -c conda-forge tabulate

Optional:
- openai / langchain_openai (for planner + judge)
- tavily (for literature search)
- python-dotenv
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Callable

import os
import json
import time
import hashlib
import traceback
import re
import argparse
import sys

import pandas as pd
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Optional import: descriptive failure packet builder
# (If failure_taxonomy.py is present in your repo, this will enrich outputs.)
try:
    from failure_taxonomy import make_failure_packet as build_failure_packet  # type: ignore
    HAVE_FAILURE_PACKET = True
except Exception:
    HAVE_FAILURE_PACKET = False

# Optional LLM layer (for end-of-graph JUDGE + optional run-level summary narration)
try:
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    HAVE_LLM = True
except Exception:
    HAVE_LLM = False
# Optional Tavily layer (web search tool)
try:
    from tavily import TavilyClient  # type: ignore
    HAVE_TAVILY = True
except Exception as e:
    HAVE_TAVILY = False
    TAVILY_IMPORT_ERROR = str(e)

PROMPT_PATHS = {
    "loose": r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\agent_prompts\loose_query",
    "minimal": r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\agent_prompts\minimal_constraint_query",
    "strict": r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\agent_prompts\strict_query",
}

DATASET_PATHS = {
    # examples — replace with your real TSV paths
    "test_case": r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\test_case_benchmark.tsv",
    "no_sequence": r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\no_sequence.tsv",
    "no_genomic_coords": r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\no_genomic_coords.tsv",
    "sequence_only": r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\sequence_only.tsv",
}


def win_to_wsl(path_str: str) -> str:
    """
    Convert C:\\Users\\... to /mnt/c/Users/... if running in WSL.
    If already POSIX, returns as-is.
    """
    p = path_str.strip().strip('"')
    if p.startswith("/"):
        return p
    # e.g. C:\Users\justi\... -> /mnt/c/Users/justi/...
    if len(p) >= 3 and p[1:3] == ":\\":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return p

def is_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME"))

def norm_path(path_str: str) -> str:
    """
    Normalize user/TSV-provided paths so they work on the current OS.
    - In WSL: convert Windows paths to /mnt/<drive>/...
    - Else: return original
    """
    if not path_str:
        return path_str
    return win_to_wsl(path_str) if is_wsl() else path_str


def load_prompt_from_key(prompt_key: str) -> tuple[str, str]:
    """
    Returns (prompt_text, prompt_name).
    """
    if prompt_key not in PROMPT_PATHS:
        raise ValueError(f"Unknown prompt_key={prompt_key}. Choose from: {list(PROMPT_PATHS)}")

    raw = PROMPT_PATHS[prompt_key]
    path = win_to_wsl(raw) if "WSL" in os.environ.get("WSL_DISTRO_NAME", "") else raw
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found at: {p}")

    return p.read_text(encoding="utf-8").strip(), p.stem

# ============================================================
# Paths (DEFAULTS)
# ============================================================

REPO_ROOT = r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation"
DEFAULT_BENCHMARK = r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\test_case_benchmark.tsv"
TEST_RUNS_ROOT = os.path.join(REPO_ROOT, "data", "test_runs")


def make_test_run_dir(prefix: str = "test_run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(TEST_RUNS_ROOT, f"{prefix}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _assert_file_exists(path: str, label: str = "file") -> None:
    if not path or not isinstance(path, str):
        raise FileNotFoundError(f"{label} path was empty or invalid.")
    path = norm_path(path)
    abs_path = os.path.abspath(os.path.normpath(path))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"{label} not found:\n  {abs_path}")
    if os.path.getsize(abs_path) == 0:
        raise FileNotFoundError(f"{label} exists but is 0 bytes:\n  {abs_path}")

# ============================================================
# Logging helpers
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


def _ensure_run_subdirs(run_dir: str) -> Dict[str, str]:
    pretty = os.path.join(run_dir, "pretty_json")
    reports = os.path.join(run_dir, "reports_md")
    os.makedirs(pretty, exist_ok=True)
    os.makedirs(reports, exist_ok=True)
    return {"pretty": pretty, "reports": reports}

def write_case_artifacts(
    run_dir: str,
    row_id: str,
    record: Dict[str, Any],
    report_text: str = "",
    consequence_md: str = "",
    metrics_md: str = "",
) -> None:
    """
    Writes:
      - pretty_json/<row_id>.json
      - reports_md/<row_id>.md                 (back-compat: main report_text)
      - reports_md/<row_id>__consequence.md    (optional)
      - reports_md/<row_id>__metrics.md        (optional)
    """
    sub = _ensure_run_subdirs(run_dir)

    with open(os.path.join(sub["pretty"], f"{row_id}.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    # Back-compat single report
    with open(os.path.join(sub["reports"], f"{row_id}.md"), "w", encoding="utf-8") as f:
        f.write(report_text or "")

    # Optional split artifacts
    if consequence_md:
        with open(os.path.join(sub["reports"], f"{row_id}__consequence.md"), "w", encoding="utf-8") as f:
            f.write(consequence_md)

    if metrics_md:
        with open(os.path.join(sub["reports"], f"{row_id}__metrics.md"), "w", encoding="utf-8") as f:
            f.write(metrics_md)



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

# ============================================================
# Failure taxonomy (v1) + tiers
# ============================================================

# Tier meanings:
# 1: Input/annotation failures (dataset/fields/canonical missing)
# 2: CDS/translation failures (cds missing, stop absent, frameshift)
# 3: NMD ambiguity (threshold edge cases, rule not applicable)
# 4: Splicing confounds (RI proxy, novel junctions)
# 5: External knowledge gaps (lit/motif db missing)
# 6: Agentic/systemic failures (tool not run, router conflict, LLM hallucination risk)

FAILURE_TIER: Dict[str, int] = {
    # Tier 1
    "INPUT_MISSING": 1,
    "ANNOTATION_INCONSISTENT": 1,
    "GENOME_BUILD_MISMATCH": 1,
    "CANONICAL_UNDEFINED": 1,

    # Tier 2
    "CDS_MISSING": 2,
    "START_CODON_ABSENT": 2,
    "STOP_CODON_ABSENT": 2,
    "FRAME_INCONSISTENT": 2,
    "CDS_TRUNCATED": 2,

    # Tier 3
    "NMD_RULE_INAPPLICABLE": 3,
    "NMD_DISTANCE_AMBIGUOUS": 3,
    "NMD_DEPENDS_ON_CANONICAL": 3,
    "PTC_IN_LAST_EXON": 3,

    # Tier 4
    "RETAINED_INTRON_PROXY": 4,
    "NOVEL_JUNCTIONS_PRESENT": 4,
    "ISOFORM_REDUNDANT": 4,

    # Tier 5
    "NO_LITERATURE_FOUND": 5,
    "MOTIF_DB_MISSING": 5,
    "GENE_UNCHARACTERIZED": 5,

    # Tier 6
    "TOOL_NOT_RUN": 6,
    "TOOL_OUTPUT_UNUSED": 6,
    "STATE_INCONSISTENT": 6,
    "ROUTER_CONFLICT": 6,
    "LLM_HALLUCINATION_RISK": 6,
}

CRITICAL_TIERS = {1, 2}


def _add_failure(state: "SplicingAgentState", code: str, note: str = "") -> None:
    """
    Add a failure code once into state["failure_modes"] and store/overwrite a short note in state["failure_notes"][code].
    """
    state.setdefault("failure_modes", [])
    if code not in state["failure_modes"]:
        state["failure_modes"].append(code)
    state.setdefault("failure_notes", {})
    if note:
        state["failure_notes"][code] = str(note)


# ============================================================
# State
# ============================================================

class SplicingAgentState(TypedDict, total=False):
    # ============================================================
    # Per-row bookkeeping
    # ============================================================
    row_id: str
    run_dir: str
    timestamp: str

    # ============================================================
    # Prompt + Dataset Metadata
    # ============================================================
    system_prompt: str
    prompt_key: str
    prompt_name: str
    dataset_key: str
    benchmark_tsv: str

    dataset_path: str
    canonical_dataset_path: str

    transcript_id: str
    canonical_transcript_id: str
    gene_symbol_hint: str
    chromosome_hint: str

    # ============================================================
    # Benchmark labels (optional)
    # ============================================================
    expected_label: str
    expected_label_provided: bool
    expected_label_inferred: bool
    expected_label_bucket: str

    # ============================================================
    # Logged fingerprints
    # ============================================================
    dataset_fingerprint: Dict[str, Any]
    canonical_dataset_fingerprint: Dict[str, Any]

    # ============================================================
    # TOOL 1 — CDS outputs
    # ============================================================
    exon_table: List[Dict[str, Any]]
    tx_len: Optional[int]
    strand: Optional[int]

    cds_segments: List[Tuple[int, int]]
    cds_start_tx: Optional[int]
    cds_end_tx: Optional[int]

    cdna_seq: Optional[str]
    cds_seq: Optional[str]

    # ============================================================
    # Canonical baseline reference
    # ============================================================
    canonical_stop_end_tx: Optional[int]
    canonical_tx_len: Optional[int]
    canonical_lastJ: Optional[int]
    margin_nt: int

    # ============================================================
    # TOOL 2 — NMD outputs
    # ============================================================
    lastJ: Optional[int]
    stop_end_tx: Optional[int]
    dist_lastJ_minus_stopEnd: Optional[int]

    ptc_predicted: Optional[bool]
    ptc_reason: str

    nmd: Optional[bool]
    nmd_reason: str

    # ============================================================
    # Stop scanning diagnostics
    # ============================================================
    stop_codons_all: List[Dict[str, Any]]
    stop_codons_internal: List[Dict[str, Any]]
    stop_codon_terminal: Optional[str]
    stop_codon_triplet: Optional[str]

    # ============================================================
    # Internal PTC candidates
    # ============================================================
    ptc_candidates: List[Dict[str, Any]]
    ptc_selected: Optional[Dict[str, Any]]

    # ============================================================
    # TOOL 3 — Motif outputs
    # ============================================================
    motif_hits: List[Dict[str, Any]]
    motif_db_results: List[Dict[str, Any]]

    # ============================================================
    # TOOL 4 — Tavily outputs
    # ============================================================
    literature_notes: List[Dict[str, Any]]

    # ============================================================
    # Agent routing + execution tracking
    # ============================================================
    plan_done: List[str]
    next_tool: str
    router_reason: str
    tool_calls: List[str]
    tool_calls_count: int
    node_events: List[Dict[str, Any]]

    # ============================================================
    # Failure taxonomy outputs
    # ============================================================
    failure_modes: List[str]
    failure_tiers: Dict[str, int]
    failure_notes: Dict[str, str]
    has_critical_failure: bool
    failure_packet: Dict[str, Any]  # optional enriched packet

    # ============================================================
    # Derived label + evaluation metrics
    # ============================================================
    predicted_label: str
    predicted_label_bucket: str
    label_mismatch_bucketed: bool

    task_completed: bool
    tool_usage_accuracy: bool
    success: bool
    error_rate_flag: bool

    hallucination: bool  # only meaningful if LLM runs
    tokens_total: Optional[int]
    latency_s: float

    # ============================================================
    # Reporting artifacts
    # ============================================================
    report_tables_md: str
    report_consequence_md: str
    report_metrics_md: str
    report_text: str

    # ============================================================
    # LLM Judge outputs
    # ============================================================
    judge_summary: str
    recommended_next_debug_step: str
    judge_confidence: str
    judge_error: str
    judge_errors: List[str]

    # ============================================================
    # Tracing + runtime diagnostics
    # ============================================================
    trace: List[str]
    errors: List[str]

# ============================================================
# Helpers
# ============================================================

STOP_CODONS = {"TAA", "TAG", "TGA"}


def _trace(state: SplicingAgentState, msg: str) -> None:
    state.setdefault("trace", []).append(msg)


def _error(state: SplicingAgentState, msg: str) -> None:
    state.setdefault("errors", []).append(msg)


def _tool_call(state: SplicingAgentState, tool_name: str) -> None:
    state.setdefault("tool_calls", []).append(tool_name)


def _chunk(seq: str, n: int) -> List[str]:
    return [seq[i:i+n] for i in range(0, len(seq), n)]


def _scan_stops_all_inframe(cds_seq: str) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for i, codon in enumerate(_chunk(cds_seq, 3)):
        if len(codon) < 3:
            break
        if codon in STOP_CODONS:
            hits.append({"codon": codon, "cds_nt_offset": i * 3, "cds_aa_index": i})
    return hits


def _split_internal_vs_terminal_stop(
    stops_all: List[Dict[str, Any]], cds_seq: str
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


def _map_cds_offset_to_tx_coord(cds_segments: List[Tuple[int, int]], cds_offset_0based: int) -> int:
    remaining = int(cds_offset_0based)
    for a, b in cds_segments:
        seg_len = int(b) - int(a) + 1
        if remaining < seg_len:
            return int(a) + remaining
        remaining -= seg_len
    raise ValueError("CDS offset exceeds CDS length")


def _stop_candidate_end_tx(cds_segments: List[Tuple[int, int]], cds_nt_offset: int) -> int:
    # +2 because offset points to codon start, and stop_end is end coordinate of triplet
    return _map_cds_offset_to_tx_coord(cds_segments, int(cds_nt_offset) + 2)


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


def mark_tool_done(state: SplicingAgentState, tool_name: str) -> SplicingAgentState:
    """
    Append tool_name to plan_done exactly once (preserves ordering).
    plan_done is used for ordering checks (e.g., NMD must occur before TABLES).

    Also writes a minimal node_events record for compatibility with external
    logging/metrics helpers (e.g., failure_logger.py / metrics.py).
    """
    done = state.get("plan_done", []) or []
    if tool_name not in done:
        done.append(tool_name)
    state["plan_done"] = done

    # Bridge for external scripts that expect event-style logs
    state.setdefault("node_events", [])
    try:
        state["node_events"].append({"name": tool_name, "status": "ok"})
    except Exception:
        pass

    _trace(state, f"DONE {tool_name} | plan_done={done}")
    return state


def compute_predicted_label(state: SplicingAgentState) -> str:
    """
    Deterministic label used for evaluation + summaries.

    Rules:
      - If stop_end_tx missing => Ambiguous (CDS missing)
      - Else if nmd True => NMD+
      - Else if ptc True and nmd False => PTC+/NMD-
      - Else if ptc False and nmd False => PTC-/NMD-
      - Else Ambiguous
    """
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

# -------------------------------
# Label normalization (authoritative)
# -------------------------------
# We keep detailed biology-forward labels for readability (e.g., "PTC+/NMD-"),
# but also compute a canonical bucket label so benchmark-provided labels like
# "protein_coding" still compare cleanly during scoring.

LABEL_BUCKET_MAP: Dict[str, str] = {
    # Detailed labels produced by compute_predicted_label()
    "NMD+": "nmd",
    "PTC+/NMD-": "protein_coding",
    "PTC-/NMD-": "protein_coding",
    "Ambiguous (CDS missing)": "cds_not_defined",
    "Ambiguous": "ambiguous",

    # Common benchmark-style labels (pass-through to buckets)
    "nmd": "nmd",
    "protein_coding": "protein_coding",
    "cds_not_defined": "cds_not_defined",
    "ambiguous": "ambiguous",

    # Extra variants you might see
    "NMD": "nmd",
    "NMD-": "protein_coding",
}

def normalize_label_bucket(label: str) -> str:
    """Map any label string to a stable comparison bucket."""
    s = (label or "").strip()
    if not s:
        return "ambiguous"
    return LABEL_BUCKET_MAP.get(s, "ambiguous")


def infer_expected_label_from_state(state: SplicingAgentState) -> str:
    """
    Auto-fill expected_label ONLY if missing.

    Also records provenance flags:
      - expected_label_provided: benchmark had a label
      - expected_label_inferred: we inferred it deterministically
    """
    exp = (state.get("expected_label") or "").strip()
    if exp:
        state["expected_label_provided"] = True
        state["expected_label_inferred"] = False
        return exp  # respect benchmark-provided labels

    state["expected_label_provided"] = False
    state["expected_label_inferred"] = True
    return compute_predicted_label(state)



def compute_task_completed(state: SplicingAgentState) -> bool:
    tool_calls = state.get("tool_calls", [])
    plan_done = state.get("plan_done", [])

    tools_ok = all(t in plan_done for t in tool_calls) and ("TABLES" in plan_done)
    has_tables = bool(state.get("report_tables_md", ""))
    has_report = bool(state.get("report_text", ""))

    return bool(tools_ok and has_tables and has_report)

def compute_tool_usage_accuracy(state: SplicingAgentState) -> bool:
    """
    True iff:
      - CDS ran
      - NMD ran
      - TABLES ran
      - NMD happened before TABLES (checked via plan_done ordering)
    """
    calls = state.get("tool_calls", []) or []
    done = state.get("plan_done", []) or []

    if "CDS" not in calls or "NMD" not in calls:
        return False
    if "TABLES" not in done:
        return False
    if "NMD" not in done or "CDS" not in done:
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
    # Compare using bucketed labels so benchmark strings like "protein_coding"
    # still match detailed labels like "PTC+/NMD-".
    pred_det = (state.get("predicted_label") or "").strip()
    exp_raw = (state.get("expected_label") or "").strip()

    pred_bucket = normalize_label_bucket(pred_det)
    exp_bucket = normalize_label_bucket(exp_raw)

    is_scorable = bool(state.get("expected_label_provided", False))
    mismatch = (pred_bucket != exp_bucket) if is_scorable else False

    # runtime errors only (not judge)
    errs = state.get("errors", []) or []
    has_runtime_errors = len(errs) > 0

    has_critical = bool(state.get("has_critical_failure", False))
    task_completed = bool(state.get("task_completed", False))
    tool_ok = bool(state.get("tool_usage_accuracy", False))

    success = bool(task_completed and tool_ok and (not has_runtime_errors) and (not has_critical) and (not mismatch))
    error_flag = bool(has_runtime_errors or has_critical or mismatch)

    # Store for downstream summaries if desired
    state["predicted_label_bucket"] = pred_bucket
    state["expected_label_bucket"] = exp_bucket
    state["label_mismatch_bucketed"] = bool(mismatch)

    return success, error_flag

# ============================================================
# TOOL 1: cds_tool (flags failures)
# ============================================================

@dataclass
class cds_tool:
    exon_loader: Callable[..., Dict[str, Any]]
    cds_segment_loader: Callable[..., List[Tuple[int, int]]]
    seq_loader: Callable[..., str]

    def run(self, state: SplicingAgentState) -> SplicingAgentState:
        _tool_call(state, "CDS")
        _trace(state, "cds_tool: start")

        # Fingerprints early
        try:
            state["dataset_fingerprint"] = dataset_fingerprint_for_log(state["dataset_path"])
            if state.get("canonical_dataset_path"):
                state["canonical_dataset_fingerprint"] = dataset_fingerprint_for_log(state["canonical_dataset_path"])
        except Exception as e:
            _error(state, f"fingerprint failed: {e}")

        # Exons + tx_len + strand
        try:
            payload = self.exon_loader(dataset_path=state["dataset_path"], transcript_id=state["transcript_id"])
            state["exon_table"] = payload["exon_table"]
            state["tx_len"] = int(payload["tx_len"])
            state["strand"] = int(payload.get("strand", 1))
        except Exception as e:
            _error(state, f"exon_loader failed: {type(e).__name__}: {e}")
            _add_failure(state, "INPUT_MISSING", "Exon table could not be loaded for transcript_id.")
            _trace(state, "cds_tool: done (exon_loader failed)")
            return state

        # CDS segments
        try:
            cds_segments = self.cds_segment_loader(dataset_path=state["dataset_path"], transcript_id=state["transcript_id"])
            state["cds_segments"] = cds_segments
            state["cds_start_tx"] = min((s for s, _ in cds_segments), default=None)
            state["cds_end_tx"] = max((e for _, e in cds_segments), default=None)
            if not cds_segments:
                _add_failure(state, "CDS_MISSING", "No CDS segments found (cds_defined=False).")
        except Exception as e:
            _error(state, f"cds_segment_loader failed: {type(e).__name__}: {e}")
            state["cds_segments"] = []
            state["cds_start_tx"] = None
            state["cds_end_tx"] = None
            _add_failure(state, "CDS_MISSING", "CDS segment loader failed; treating CDS as missing.")

        # cDNA + CDS seq
        try:
            cdna = self.seq_loader(dataset_path=state["dataset_path"], transcript_id=state["transcript_id"])
            state["cdna_seq"] = cdna
            if state.get("cds_segments") and cdna:
                state["cds_seq"] = "".join(cdna[int(s)-1:int(e)] for s, e in state["cds_segments"])
            else:
                state["cds_seq"] = None
                if not state.get("cds_segments"):
                    _add_failure(state, "CDS_MISSING", "CDS sequence could not be constructed because CDS segments are empty.")
        except Exception as e:
            _error(state, f"seq_loader failed: {type(e).__name__}: {e}")
            state["cdna_seq"] = None
            state["cds_seq"] = None
            _add_failure(state, "INPUT_MISSING", "cDNA sequence could not be loaded.")

        # Basic consistency check
        try:
            tx_len = state.get("tx_len")
            cds_end = state.get("cds_end_tx")
            if tx_len is not None and cds_end is not None and int(cds_end) > int(tx_len):
                _add_failure(state, "ANNOTATION_INCONSISTENT", "cds_end_tx exceeds transcript length; check BioMart export.")
        except Exception:
            pass

        _trace(state, "cds_tool: done")
        return state


# ============================================================
# TOOL 2: nmd_tool (flags failures)
# ============================================================

@dataclass
class nmd_tool:
    ejc_threshold_nt: int = 55
    require_ptc_for_nmd: bool = True
    margin_nt: int = 55  # PTC buffer vs reference stop

    def run(self, state: SplicingAgentState) -> SplicingAgentState:
        _tool_call(state, "NMD")
        _trace(state, "nmd_tool: start")

        # lastJ = end of penultimate exon in transcript coords
        exons = state.get("exon_table", []) or []
        if len(exons) >= 2:
            penult = exons[-2]
            state["lastJ"] = penult.get("end_tx") or penult.get("end") or penult.get("exon_end_tx")
        else:
            state["lastJ"] = None
            _add_failure(state, "NMD_RULE_INAPPLICABLE", "Transcript has <2 exons; no exon–exon junction available for EJC rule.")

        # stop_end_tx = CDS end in transcript coords
        state["stop_end_tx"] = state.get("cds_end_tx")

        cds_seq = state.get("cds_seq") or ""
        cds_segments = state.get("cds_segments") or []

        # STOP SCAN (ALL + internal + terminal)
        if cds_seq:
            all_hits = _scan_stops_all_inframe(cds_seq)
            internal_hits, terminal_hit = _split_internal_vs_terminal_stop(all_hits, cds_seq)
            state["stop_codons_all"] = all_hits
            state["stop_codons_internal"] = internal_hits
            state["stop_codon_terminal"] = terminal_hit["codon"] if terminal_hit else None
            state["stop_codon_triplet"] = cds_seq[-3:] if (len(cds_seq) >= 3 and cds_seq[-3:] in STOP_CODONS) else None
            if not state.get("stop_codon_terminal"):
                _add_failure(state, "STOP_CODON_ABSENT", "No terminal in-frame stop codon found in CDS sequence.")
            if len(cds_seq) % 3 != 0:
                _add_failure(state, "FRAME_INCONSISTENT", "CDS length not divisible by 3; potential frameshift/annotation issue.")
        else:
            state["stop_codons_all"] = []
            state["stop_codons_internal"] = []
            state["stop_codon_terminal"] = None
            state["stop_codon_triplet"] = None
            _add_failure(state, "CDS_MISSING", "Cannot scan stop codons; CDS sequence missing.")

        # Internal PTC candidates (map to tx coords)
        lastJ = state.get("lastJ")
        ptc_candidates: List[Dict[str, Any]] = []
        for h in (state.get("stop_codons_internal") or []):
            cand = dict(h)
            try:
                cand_end_tx = _stop_candidate_end_tx(cds_segments, cand["cds_nt_offset"])
            except Exception as e:
                cand_end_tx = None
                _error(state, f"map stop->tx failed for offset={cand.get('cds_nt_offset')}: {type(e).__name__}: {e}")

            cand["stop_end_tx_candidate"] = cand_end_tx
            if (lastJ is not None) and (cand_end_tx is not None):
                cand_dist = int(lastJ) - int(cand_end_tx)
                cand["dist_lastJ_minus_stopEnd_candidate"] = cand_dist
                cand["ejc_nmd_candidate"] = bool(cand_dist >= int(self.ejc_threshold_nt))
            else:
                cand["dist_lastJ_minus_stopEnd_candidate"] = None
                cand["ejc_nmd_candidate"] = None
            ptc_candidates.append(cand)

        state["ptc_candidates"] = ptc_candidates
        state["ptc_selected"] = ptc_candidates[0] if ptc_candidates else None

        # Reference baseline PTC-by-position (your test-case logic)
        state["margin_nt"] = int(self.margin_nt)
        can_stop = state.get("canonical_stop_end_tx")
        obs_stop = state.get("stop_end_tx")

        if obs_stop is None:
            state["ptc_predicted"] = False
            state["ptc_reason"] = "Early stop = unknown -> observed stop position missing (CDS missing/undefined)"
            _add_failure(state, "CDS_MISSING", "Observed stop_end_tx is missing; PTC inference not reliable.")
        elif can_stop is None:
            state["ptc_predicted"] = False
            state["ptc_reason"] = "Early stop = unknown -> reference stop position missing"
            _add_failure(state, "CANONICAL_UNDEFINED", "Canonical stop position missing; PTC-by-delta logic not applicable.")
        else:
            obs = int(obs_stop)
            can = int(can_stop)
            state["ptc_predicted"] = bool(obs <= (can - int(self.margin_nt)))
            state["ptc_reason"] = (
                f"Early stop predicted={state['ptc_predicted']} by reference comparison: "
                f"observed_stop_end_tx={obs} <= reference_stop_end_tx={can} - margin={int(self.margin_nt)}"
            )

        # EJC rule ONLY when ptc_predicted=True
        if (lastJ is None) or (obs_stop is None):
            state["dist_lastJ_minus_stopEnd"] = None
            state["nmd"] = False
            state["nmd_reason"] = "NMD=unknown -> set False (missing CDS or lastJ/stop_end_tx)"
            _trace(state, "nmd_tool: done (unknown -> False)")
            return state

        dist = int(lastJ) - int(obs_stop)
        state["dist_lastJ_minus_stopEnd"] = dist

        # Last-exon signature
        if dist < 0:
            _add_failure(state, "PTC_IN_LAST_EXON", f"dist={dist} (stop is downstream of lastJ).")

        # Ambiguity near threshold (buffer window)
        if abs(dist - int(self.ejc_threshold_nt)) <= 5:
            _add_failure(state, "NMD_DISTANCE_AMBIGUOUS", f"Distance={dist} is within ±5 nt of threshold={int(self.ejc_threshold_nt)}.")

        if self.require_ptc_for_nmd and not state.get("ptc_predicted"):
            state["nmd"] = False
            state["nmd_reason"] = "Skipped distance rule (early stop predicted=False) -> NMD=False"
            _trace(state, "nmd_tool: done (ptc_predicted=False)")
            return state

        state["nmd"] = bool(dist >= int(self.ejc_threshold_nt))
        state["nmd_reason"] = (
            f"Distance rule applied (early stop predicted=True): distance={dist} "
            f"{'>=' if state['nmd'] else '<'} {int(self.ejc_threshold_nt)} -> NMD={state['nmd']}"
        )
        _trace(state, "nmd_tool: done")
        return state


# ============================================================
# TOOL 3 & 4 (stubs) — selectable by agent
# ============================================================

@dataclass
class motif_tool:
    """
    Deterministic RNA secondary structure motif detector
    + structured database-style enrichment (CoSSMos-inspired).

    Does NOT modify predicted_label.
    Only adds structural evidence.
    """

    def run(self, state: SplicingAgentState) -> SplicingAgentState:
        _tool_call(state, "MOTIF")
        _trace(state, "motif_tool: start")

        seq = (state.get("cds_seq") or "").upper().replace("T", "U")

        if not seq:
            state["motif_hits"] = []
            state["motif_db_results"] = []
            _add_failure(state, "MOTIF_DB_MISSING", "No CDS sequence available.")
            _trace(state, "motif_tool: done (no sequence)")
            return state

        hits: List[Dict[str, Any]] = []

        def revcomp(s: str) -> str:
            return s[::-1].translate(str.maketrans("AUGC", "UACG"))

        # ---------------------------------------------------
        # 1️⃣ Hairpin detection (stem-loop)
        # ---------------------------------------------------
        for i in range(len(seq) - 12):
            stem = seq[i:i+6]
            rc = revcomp(stem)
            search_region = seq[i+6:i+40]
            if rc in search_region:
                j = search_region.index(rc)
                hits.append({
                    "motif_name": "hairpin_loop",
                    "start": i + 1,
                    "end": i + 6,
                    "sequence": stem,
                    "loop_distance": j,
                    "evidence": "inverted_repeat_detected"
                })
                break  # one strong hit is enough for report

        # ---------------------------------------------------
        # 2️⃣ Simple internal loop heuristic
        # ---------------------------------------------------
        for i in range(len(seq) - 14):
            left = seq[i:i+4]
            right = seq[i+8:i+12]
            if revcomp(left) == right:
                mismatch = seq[i+4:i+8]
                hits.append({
                    "motif_name": "internal_loop",
                    "start": i + 1,
                    "end": i + 12,
                    "sequence": seq[i:i+12],
                    "mismatch_region": mismatch,
                    "evidence": "paired_flanks_with_internal_mismatch"
                })
                break

        # ---------------------------------------------------
        # 3️⃣ Single-nt bulge heuristic
        # ---------------------------------------------------
        for i in range(len(seq) - 10):
            left = seq[i:i+4]
            bulge = seq[i+4]
            right = seq[i+5:i+9]
            if revcomp(left) == right:
                hits.append({
                    "motif_name": "single_nt_bulge",
                    "start": i + 1,
                    "end": i + 9,
                    "sequence": seq[i:i+9],
                    "bulge_nt": bulge,
                    "evidence": "single_unpaired_nucleotide"
                })
                break

        state["motif_hits"] = hits

        # ---------------------------------------------------
        # 4️⃣ Database enrichment (CoSSMos-style)
        # ---------------------------------------------------
        db_results: List[Dict[str, Any]] = []

        for hit in hits:
            mtype = hit.get("motif_name")

            if mtype == "hairpin_loop":
                db_results.append({
                    "motif_type": "hairpin",
                    "database_supported": True,
                    "known_sizes_nt": "3–7",
                    "structural_features": [
                        "stacking_interactions",
                        "base_pairing_edges",
                        "sugar_pucker_variability"
                    ],
                    "reference_database": "RNA CoSSMos"
                })

            elif mtype == "internal_loop":
                db_results.append({
                    "motif_type": "internal_loop",
                    "database_supported": True,
                    "common_sizes": "1x1, 2x2, 3x3",
                    "structural_features": [
                        "noncanonical_base_pairs",
                        "Hoogsteen_interactions",
                        "Watson_Crick_edges"
                    ],
                    "reference_database": "RNA CoSSMos"
                })

            elif mtype == "single_nt_bulge":
                db_results.append({
                    "motif_type": "bulge_loop",
                    "database_supported": True,
                    "common_sizes": "1 nt",
                    "structural_features": [
                        "outward_stack",
                        "backbone_distortion"
                    ],
                    "reference_database": "RNA CoSSMos"
                })

        state["motif_db_results"] = db_results

        if hits and not db_results:
            _add_failure(state, "NO_LITERATURE_FOUND", "Motif detected but no structural annotation available.")

        _trace(
            state,
            f"motif_tool: done | hits={len(hits)} | db_results={len(db_results)}"
        )

        return state

@dataclass
class tavily_tool:
    max_results: int = 5
    search_depth: str = "basic"

    def run(self, state: SplicingAgentState) -> SplicingAgentState:
        _tool_call(state, "TAVILY")
        _trace(state, f"TAVILY tool invoked | HAVE_TAVILY={HAVE_TAVILY} | API_KEY_SET={bool(os.getenv('TAVILY_API_KEY','').strip())}")

        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            # deterministically fail: tool ran, but cannot execute
            state["literature_notes"] = []
            _add_failure(state, "INPUT_MISSING", "TAVILY_API_KEY missing; cannot run Tavily search.")
            _trace(state, "tavily_tool: done (missing key)")
            return state
        try:
            # Official Tavily Python SDK :contentReference[oaicite:2]{index=2}
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)

            gene = (state.get("gene_symbol_hint") or "").strip()
            tx = (state.get("transcript_id") or "").strip()
            query = f"{gene} aberrant splicing NMD premature termination codon transcript {tx}".strip()

            res = client.search(
                query=query,
                max_results=int(self.max_results),
                search_depth=self.search_depth,
                include_answer=False,
                include_raw_content=False,
            )

            # Tavily returns a dict with "results" typically containing title/url/content
            results = res.get("results") or []
            notes = []
            for r in results:
                notes.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content"),
                    "score": r.get("score"),
                })

            state["literature_notes"] = notes

            if not notes:
                _add_failure(state, "NO_LITERATURE_FOUND", "Tavily search returned 0 results for the query.")

            _trace(state, f"tavily_tool: done | n_results={len(notes)}")
            return state

        except Exception as e:
            state["literature_notes"] = []
            _error(state, f"tavily search failed: {type(e).__name__}: {e}")
            _add_failure(state, "NO_LITERATURE_FOUND", "Tavily search errored; see errors for details.")
            _trace(state, "tavily_tool: done (error)")
            return state


# ============================================================
# Dynamic routing: CDS -> AGENT -> NMD (forced) -> (MOTIF, TAVILY) -> TABLES ...
# ============================================================

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


# ============================================================
# TABLES (compiler / aggregator; NOT a tool)
# ============================================================

def node_tables_report(state: SplicingAgentState) -> SplicingAgentState:
    _trace(state, "tables_report: start")

    cds_present = bool(state.get("cds_seq"))
    cds_missing_note = "" if cds_present else " (CDS missing/undefined)"

    # adding motif hits for demonstration; replace with real motif logic as needed
    motif_hits = state.get("motif_hits") or []
    motif_df = (
        pd.DataFrame(motif_hits[:10])
        if motif_hits
        else pd.DataFrame(columns=["motif_name", "start", "end", "sequence"])
    )

    if not motif_df.empty:
        motif_df = motif_df.rename(columns={
            "motif_name": "Motif",
            "start": "Start (nt)",
            "end": "End (nt)",
            "sequence": "Matched sequence",
        })

    motif_count = len(motif_hits)

    # literature notes from Tavily
    literature_notes = state.get("literature_notes") or []
    literature_df = (
        pd.DataFrame(literature_notes[:5])
        if literature_notes
        else pd.DataFrame(columns=["title", "url", "score"])
    )

    if not literature_df.empty:
        literature_df = literature_df.rename(columns={
            "title": "Title",
            "url": "URL",
            "score": "Relevance score",
        })

    literature_count = len(literature_notes)

    summary_df = pd.DataFrame([{
        "Case ID": state.get("row_id"),
        "Transcript ID": state.get("transcript_id"),
        "Gene Symbol": state.get("gene_symbol_hint"),
        "Transcript length (nt)": state.get("tx_len"),
        "Last exon–exon junction (nt)": state.get("lastJ"),
        "Observed stop codon end (nt)" + cds_missing_note: state.get("stop_end_tx"),
        "Distance from stop to last junction (nt)": state.get("dist_lastJ_minus_stopEnd"),
        "Reference transcript (canonical)": state.get("canonical_transcript_id"),
        "Reference stop codon end (nt)": state.get("canonical_stop_end_tx"),
        "Early-stop margin (nt)": state.get("margin_nt"),
        "Early stop predicted (PTC?)": state.get("ptc_predicted"),
        "NMD predicted?": state.get("nmd"),

        # new agentic evidence summary
        "Motif Scanner + Detection (MOTIF tool)": motif_count,
        "Literature Search + Validation (TAVILY tool)": literature_count,
        "Tools executed (ordered)": ", ".join(state.get("plan_done", [])),
    }])

    stops_all = state.get("stop_codons_all") or []
    stops_df = (
        pd.DataFrame(stops_all[:10])
        if stops_all
        else pd.DataFrame(columns=["codon", "cds_nt_offset", "cds_aa_index"])
    )
    if not stops_df.empty:
        stops_df = stops_df.rename(columns={
            "codon": "Stop codon",
            "cds_nt_offset": "Position in CDS (nt)",
            "cds_aa_index": "Amino acid position",
        })

    ptcs = state.get("ptc_candidates") or []
    ptc_df = (
        pd.DataFrame(ptcs[:10])
        if ptcs
        else pd.DataFrame(columns=[
            "codon", "cds_nt_offset", "cds_aa_index",
            "stop_end_tx_candidate", "dist_lastJ_minus_stopEnd_candidate", "ejc_nmd_candidate"
        ])
    )
    if not ptc_df.empty:
        ptc_df = ptc_df.rename(columns={
            "codon": "Stop codon",
            "cds_nt_offset": "Position in CDS (nt)",
            "cds_aa_index": "Amino acid position",
            "stop_end_tx_candidate": "Stop codon end in transcript (nt)",
            "dist_lastJ_minus_stopEnd_candidate": "Distance from stop to last junction (nt)",
            "ejc_nmd_candidate": "NMD predicted from distance rule?",
        })
    

    md_parts: List[str] = []

    md_parts.append("## Summary")
    md_parts.append(_df_to_md_table(summary_df))
    md_parts.append("")

    md_parts.append("## Reference comparison (early stop + NMD logic)")
    md_parts.append(f"- **Early stop (PTC) reason:** {state.get('ptc_reason','')}")
    md_parts.append(f"- **NMD reason:** {state.get('nmd_reason','')}")
    md_parts.append("")

    # ------------------------------------------------------------
    # STOP CODONS
    # ------------------------------------------------------------
    md_parts.append("## Stop codons found in the CDS (in-frame; up to 10)")
    md_parts.append(_df_to_md_table(stops_df))
    md_parts.append("")

    md_parts.append("## Internal stop codons (possible early stops; up to 10)")
    md_parts.append(_df_to_md_table(ptc_df))
    md_parts.append("")

    md_parts.append("## Terminal stop")
    md_parts.append(f"- terminal_stop_present: **{'Yes' if state.get('stop_codon_terminal') else 'No'}**")
    md_parts.append(f"- terminal_stop_codon: **{state.get('stop_codon_terminal')}**")
    md_parts.append(f"- cds_terminal_triplet: **{state.get('stop_codon_triplet')}**")
    md_parts.append("")

    # ------------------------------------------------------------
    # MOTIF SECTION
    # ------------------------------------------------------------
    md_parts.append("## RNA-binding motif evidence (MOTIF tool)")
    md_parts.append(f"- motif_hits_detected: **{motif_count}**")
    if motif_count > 0:
        md_parts.append(_df_to_md_table(motif_df))
    else:
        md_parts.append("_No motif evidence detected or tool not invoked._")
    md_parts.append("")

    # ------------------------------------------------------------
    # MOTIF DATABASE ENRICHMENT
    # ------------------------------------------------------------
    md_parts.append("## Motif structural database enrichment")
    db_results = state.get("motif_db_results") or []
    if db_results:
        md_parts.append("```json")
        md_parts.append(json.dumps(db_results, indent=2))
        md_parts.append("```")
    else:
        md_parts.append("_No structural enrichment available._")
    md_parts.append("")

    # ------------------------------------------------------------
    # LITERATURE SECTION
    # ------------------------------------------------------------
    md_parts.append("## Literature evidence (TAVILY tool)")
    md_parts.append(f"- literature_notes_retrieved: **{literature_count}**")
    if literature_count > 0:
        md_parts.append(_df_to_md_table(literature_df))
    else:
        md_parts.append("_No literature retrieved or tool not invoked._")

    state["report_tables_md"] = "\n".join(md_parts)

    # Record TABLES as reached for ordering checks
    state = mark_tool_done(state, "TABLES")

    # Provide a minimal report_text early so FAILURE_COMPILER can score task_completed deterministically.
    # FINAL will overwrite this with the full consequence report.
    if not state.get("report_text"):
        state["report_text"] = state["report_tables_md"]

    _trace(state, "tables_report: done")
    return state

# ============================================================
# Deterministic failure + metrics compiler (authoritative)
# ============================================================

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

# ============================================================
# JUDGE (LLM) — optional: narrative + hallucination risk
# ============================================================

def _try_get_tokens(resp: Any) -> Optional[int]:
    try:
        md = getattr(resp, "response_metadata", None) or {}
        usage = md.get("token_usage") or md.get("usage") or {}
        for k in ("total_tokens", "total", "totalTokens"):
            if k in usage and usage[k] is not None:
                return int(usage[k])
    except Exception:
        pass
    return None

def _extract_first_json_object(text_blob: Any) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON extractor for LLM outputs.

    The judge prompt requests STRICT JSON, but in practice the model may include
    extra prose or markdown fences. We keep this deterministic and minimal:
      1) try json.loads(text_blob)
      2) strip ```json fences if present
      3) locate the first {...} block and json.loads(that)

    Returns dict on success, otherwise None.
    """
    if text_blob is None:
        return None
    raw = str(text_blob).strip()
    if not raw:
        return None

    # 1) direct parse
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) strip common fences
    raw2 = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw2 = re.sub(r"\s*```$", "", raw2).strip()

    # 3) first JSON object block
    m = re.search(r"\{.*\}", raw2, flags=re.S)
    if not m:
        return None
    candidate = m.group(0).strip()
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def node_llm_judge(state: SplicingAgentState, llm: Optional[Any]) -> SplicingAgentState:
    _trace(state, "judge: start")

    evidence = {
        "row_id": state.get("row_id"),
        "transcript_id": state.get("transcript_id"),
        "gene_symbol_hint": state.get("gene_symbol_hint"),
        "expected_label": (state.get("expected_label") or "").strip(),
        "predicted_label": (state.get("predicted_label") or "").strip(),
        "ptc_predicted": state.get("ptc_predicted"),
        "nmd": state.get("nmd"),
        "ptc_reason": state.get("ptc_reason"),
        "nmd_reason": state.get("nmd_reason"),
        "lastJ": state.get("lastJ"),
        "stop_end_tx": state.get("stop_end_tx"),
        "dist_lastJ_minus_stopEnd": state.get("dist_lastJ_minus_stopEnd"),
        "canonical_transcript_id": state.get("canonical_transcript_id"),
        "canonical_stop_end_tx": state.get("canonical_stop_end_tx"),
        "failure_modes": state.get("failure_modes", []),
        "failure_tiers": state.get("failure_tiers", {}),
        "failure_notes": state.get("failure_notes", {}),
        "failure_packet": state.get("failure_packet", {}),
        "metrics_flags": {
            "task_completed": state.get("task_completed"),
            "tool_usage_accuracy": state.get("tool_usage_accuracy"),
            "success": state.get("success"),
            "error_rate_flag": state.get("error_rate_flag"),
        },
        "tool_calls": state.get("tool_calls", []),
        "plan_done": state.get("plan_done", []),
        "errors": state.get("errors", []),
    }

    if llm is None:
        state["judge_summary"] = "LLM judge disabled (missing deps or OpenAI key). Deterministic failure_compiler metrics are authoritative."
        state["judge_confidence"] = "low"
        state["recommended_next_debug_step"] = "Use failure_modes + failure_packet (if present) + errors + report tables to debug."
        _trace(state, "judge: done (disabled)")
        return state

    prompt = f"""
Return ONLY a single JSON object (no markdown, no code fences, no trailing text).
You MUST use ONLY the Evidence JSON. Do not invent facts.
Do not add new failure codes beyond those already present.

JSON schema:
{{
  "judge_summary": "string (2-6 sentences)",
  "confidence": "low|medium|high",
  "recommended_next_debug_step": "string (one concrete action)",
  "hallucination": "boolean (true only if your summary would require facts not in Evidence)"
}}

Evidence JSON:
{json.dumps(evidence, indent=2)}
""".strip()

    txt = ""
    try:
        resp = llm.invoke(prompt)
        txt = getattr(resp, "content", str(resp)).strip()
        
        tok = _try_get_tokens(resp)
        if tok is not None:
            state["tokens_total"] = int(tok)

        parsed = _extract_first_json_object(txt)
        if parsed is None:
            raise json.JSONDecodeError("Could not parse judge JSON", txt, 0)

        state["judge_summary"] = str(parsed.get("judge_summary", "")).strip()
        state["judge_confidence"] = str(parsed.get("confidence", "")).strip()
        state["recommended_next_debug_step"] = str(parsed.get("recommended_next_debug_step", "")).strip()
        state["hallucination"] = bool(parsed.get("hallucination", False))

        if state["hallucination"]:
            _add_failure(state, "LLM_HALLUCINATION_RISK", "Judge self-reported hallucination risk.")
            state["failure_modes"] = sorted(set(state.get("failure_modes", []) or []))
            state["failure_tiers"] = {m: int(FAILURE_TIER.get(m, 0)) for m in state["failure_modes"]}

        _trace(state, "judge: done")
        return state
    
    except Exception as e:
    # Judge is optional; do not treat parse/LLM issues as a run error.
        state["judge_error"] = f"{type(e).__name__}: {e}"
        _trace(state, f"judge failed: {type(e).__name__}: {e}")
        
        state.setdefault("judge_errors", []).append(f"judge failed: {type(e).__name__}: {e}")
        try:
            state["judge_errors"].append(f"judge raw output (first 500 chars): {txt[:500]}")
        except Exception:
            pass
        state["judge_summary"] = "LLM judge failed; see errors."
        state["judge_confidence"] = "low"
        state["recommended_next_debug_step"] = "Inspect errors; rerun with judge disabled."
        state["hallucination"] = False
        _trace(state, "judge: done (error)")
        return state

# ============================================================
# 3) FINAL report node (UPDATED: build 2 markdown artifacts)
# ============================================================

def node_final_report(state: SplicingAgentState) -> SplicingAgentState:
    _trace(state, "final_report: start")

    # -----------------------------
    # A) Functional consequence report
    # -----------------------------
    consequence: List[str] = []
    consequence.append("# FUNCTIONAL CONSEQUENCE REPORT")
    consequence.append("")
    consequence.append(f"**ROW:** {state.get('row_id','')}")
    consequence.append(f"**TRANSCRIPT:** {state.get('transcript_id','')}  |  **GENE:** {state.get('gene_symbol_hint','')}")
    consequence.append(f"**DATASET:** `{state.get('dataset_path','')}`")
    consequence.append("")

    consequence.append("## Call")
    consequence.append(f"- **predicted_label:** **{state.get('predicted_label','')}**")
    consequence.append(f"- **PTC predicted:** **{state.get('ptc_predicted')}**")
    consequence.append(f"- **NMD predicted:** **{state.get('nmd')}**")
    consequence.append("")

    consequence.append("## Rationale")
    consequence.append(f"- **PTC reason:** {state.get('ptc_reason','')}")
    consequence.append(f"- **NMD reason:** {state.get('nmd_reason','')}")
    consequence.append("")

    consequence.append("## Evidence tables")
    consequence.append(state.get("report_tables_md", "_(no tables produced)_"))
    consequence.append("")

    state["report_consequence_md"] = "\n".join(consequence)

    # IMPORTANT: keep task_completed definition aligned:
    # compute_task_completed() expects report_tables_md + report_text
    state["report_text"] = state["report_consequence_md"]

    # -----------------------------
    # B) Recompute deterministic flags AFTER report_text exists
    # -----------------------------
    state["task_completed"] = compute_task_completed(state)
    state["tool_usage_accuracy"] = compute_tool_usage_accuracy(state)

    success, error_flag = compute_success_and_error_flag(state)
    state["success"] = bool(success)
    state["error_rate_flag"] = bool(error_flag)

    # -----------------------------
    # C) Metrics / QA report
    # -----------------------------
    metrics: List[str] = []
    metrics.append("# METRICS / QA REPORT")
    metrics.append("")
    metrics.append(f"**ROW:** {state.get('row_id','')}")
    metrics.append(f"**TRANSCRIPT:** {state.get('transcript_id','')}  |  **GENE:** {state.get('gene_symbol_hint','')}")
    metrics.append(f"**DATASET:** `{state.get('dataset_path','')}`")
    metrics.append("")

    metrics.append("## Tool calls")
    metrics.append(f"- tool_calls: `{state.get('tool_calls', [])}`")
    metrics.append(f"- plan_done: `{state.get('plan_done', [])}`")
    metrics.append("")

    metrics.append("## Failure modes (taxonomy)")
    fm = state.get("failure_modes", []) or []
    if fm:
        metrics.append("```json")
        metrics.append(json.dumps({
            "failure_modes": fm,
            "failure_tiers": state.get("failure_tiers", {}),
            "failure_notes": state.get("failure_notes", {}),
            "has_critical_failure": state.get("has_critical_failure", False),
        }, indent=2))
        metrics.append("```")
    else:
        metrics.append("_(none flagged)_")
    metrics.append("")

    if state.get("failure_packet"):
        metrics.append("## Failure packet (descriptive; from failure_taxonomy.py)")
        metrics.append("```json")
        metrics.append(json.dumps(state.get("failure_packet", {}), indent=2, default=str))
        metrics.append("```")
        metrics.append("")

    metrics.append("## Metrics (tight definitions; deterministic flags)")
    metrics.append("```json")
    metrics.append(json.dumps({
        "task_completed": state.get("task_completed", False),
        "tool_usage_accuracy": state.get("tool_usage_accuracy", False),
        "success": state.get("success", False),
        "error_rate_flag": state.get("error_rate_flag", False),
        "hallucination_metric": state.get("hallucination", False),
        "predicted_label": state.get("predicted_label", ""),
        "expected_label": (state.get("expected_label") or "").strip(),
        "token_cost_total_tokens": state.get("tokens_total", None),
        "latency_s": state.get("latency_s", None),
    }, indent=2))
    metrics.append("```")
    metrics.append("")

    metrics.append("## JUDGE (LLM; optional)")
    metrics.append(f"- confidence: **{state.get('judge_confidence','')}**")
    metrics.append("")
    metrics.append(state.get("judge_summary", ""))
    metrics.append("")
    metrics.append("### Recommended next debug step")
    metrics.append(state.get("recommended_next_debug_step", ""))
    metrics.append("")

    if state.get("errors"):
        metrics.append("## Errors")
        metrics.append("```")
        metrics.append("\n".join(state.get("errors") or []))
        metrics.append("```")
        metrics.append("")

    state["report_metrics_md"] = "\n".join(metrics)

    _trace(
        state,
        f"final_report: done | task_completed={state.get('task_completed')} "
        f"tool_usage_accuracy={state.get('tool_usage_accuracy')} "
        f"success={state.get('success')} error_flag={state.get('error_rate_flag')}"
    )
    return state

# ============================================================
# Graph builder (dynamic agentic pipeline)
# ============================================================

def build_graph(
    cds: cds_tool,
    nmd: nmd_tool,
    motif: motif_tool,
    tavily: tavily_tool,
    planner_llm: Optional[Any],
    judge_llm: Optional[Any],
) -> Any:
    g = StateGraph(SplicingAgentState)

    # Wrap CDS so plan_done records ordering for tool_usage_accuracy
    def _cds(state: SplicingAgentState) -> SplicingAgentState:
        state = cds.run(state)
        return mark_tool_done(state, "CDS")

    g.add_node("CDS", _cds)
    def _agent(state: SplicingAgentState) -> SplicingAgentState:
        return node_agent_router(state, planner_llm)
    g.add_node("AGENT", _agent)


    def _nmd(state: SplicingAgentState) -> SplicingAgentState:
        state = nmd.run(state)
        return mark_tool_done(state, "NMD")

    def _motif(state: SplicingAgentState) -> SplicingAgentState:
        state = motif.run(state)
        return mark_tool_done(state, "MOTIF")

    def _tavily(state: SplicingAgentState) -> SplicingAgentState:
        state = tavily.run(state)
        return mark_tool_done(state, "TAVILY")

    g.add_node("NMD", _nmd)
    g.add_node("MOTIF", _motif)
    g.add_node("TAVILY", _tavily)

    g.add_node("TABLES", node_tables_report)
    g.add_node("FAILURE_COMPILER", node_failure_compiler)

    def _judge(state: SplicingAgentState) -> SplicingAgentState:
        return node_llm_judge(state, llm=judge_llm)

    g.add_node("JUDGE", _judge)
    g.add_node("FINAL", node_final_report)

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


# ============================================================
# Run-level summaries
# ============================================================

FRIENDLY_SUMMARY_COLS = {
    "prompt_key": "Prompt key",
    "prompt_name": "Prompt name",
    "dataset_key": "Dataset key",
    "benchmark_tsv": "Benchmark TSV",
    "row_id": "Case ID",
    "transcript_id": "Transcript ID",
    "gene_symbol_hint": "Gene Symbol",
    "tx_len": "Transcript length (nt)",
    "lastJ": "Last exon–exon junction (nt)",
    "stop_end_tx": "Observed stop codon end (nt)",
    "dist": "Distance from stop to last junction (nt)",
    "canonical_transcript_id": "Reference transcript (canonical)",
    "canonical_stop_end_tx": "Reference stop codon end (nt)",
    "margin_nt": "Early-stop margin (nt)",
    "ptc_predicted": "Early stop predicted (PTC?)",
    "nmd": "NMD predicted?",
    "motif_hits_count": "Motif Scanner + Detection (MOTIF tool)",
    "literature_notes_count": "Literature Search + Validation (TAVILY tool)",
    "predicted_label": "Predicted label",
    "expected_label":"Expected label",
    "task_completed": "Task completed",
    "tool_usage_accuracy": "Tool usage accuracy",
    "success": "Success",
    "error_rate_flag": "Error rate flag",
    "hallucination": "Hallucination metric",
    "tokens_total": "Token cost (total tokens)",
    "latency_s": "Latency (s)",
    "status": "Status",
}


def write_benchmark_summaries(
    run_dir: str,
    bench_path: str,
    summary_rows: List[Dict[str, Any]],
    canonicals: Dict[str, str],
    llm: Optional[Any] = None,
) -> Dict[str, str]:
    out_csv = os.path.join(run_dir, "benchmark_summary.csv")
    out_md = os.path.join(run_dir, "benchmark_summary.md")

    df_all = pd.DataFrame(summary_rows).copy()

    friendly = df_all.copy()
    for k in FRIENDLY_SUMMARY_COLS.keys():
        if k not in friendly.columns:
            friendly[k] = ""
    friendly = friendly[list(FRIENDLY_SUMMARY_COLS.keys())].rename(columns=FRIENDLY_SUMMARY_COLS)
    friendly.to_csv(out_csv, index=False)

    n_total = int(df_all.shape[0])
    n_errors = int((df_all["status"] == "error").sum()) if "status" in df_all.columns else 0
    n_success = int(df_all["success"].fillna(False).astype(bool).sum()) if "success" in df_all.columns else 0
    n_flag = int(df_all["error_rate_flag"].fillna(False).astype(bool).sum()) if "error_rate_flag" in df_all.columns else 0
    n_hall = int(df_all["hallucination"].fillna(False).astype(bool).sum()) if "hallucination" in df_all.columns else 0
    avg_latency = float(df_all["latency_s"].astype(float).mean()) if "latency_s" in df_all.columns and df_all.shape[0] else 0.0

    counts_df = pd.DataFrame([
        {"Metric": "Total cases", "Count": n_total},
        {"Metric": "Success", "Count": n_success},
        {"Metric": "Error flag (review needed)", "Count": n_flag},
        {"Metric": "Hallucination flagged", "Count": n_hall},
        {"Metric": "Errors (runtime)", "Count": n_errors},
        {"Metric": "Avg latency (s)", "Count": round(avg_latency, 3)},
    ])

    by_gene = pd.DataFrame()
    if "gene_symbol_hint" in df_all.columns:
        by_gene = (
            df_all.groupby("gene_symbol_hint", dropna=False)
            .agg(
                Cases=("row_id", "count"),
                Success=("success", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
                ErrorFlag=("error_rate_flag", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
                RuntimeErrors=("status", lambda x: int((pd.Series(x) == "error").sum())),
                Avg_Latency_s=("latency_s", lambda x: float(pd.Series(x).astype(float).mean()) if len(x) else 0.0),
            )
            .reset_index()
            .rename(columns={"gene_symbol_hint": "Gene Symbol"})
        )

    exec_summary_md = "_(LLM executive summary disabled.)_"
    if llm is not None:
        payload = {
            "n_total": n_total,
            "n_success": n_success,
            "n_error_flag": n_flag,
            "n_runtime_errors": n_errors,
            "n_hallucination": n_hall,
            "avg_latency_s": avg_latency,
            "by_gene": by_gene.to_dict(orient="records") if by_gene.shape[0] else [],
            "benchmark_path": os.path.abspath(os.path.normpath(bench_path)),
        }
        prompt = f"""
You are summarizing a benchmark run. Use ONLY this JSON. No invention.
Write 3-6 bullets: outcome distribution, where error flags cluster, and next actions.

JSON:
{json.dumps(payload, indent=2)}
""".strip()
        try:
            resp = llm.invoke(prompt)
            txt = getattr(resp, "content", "") if resp is not None else ""
            exec_summary_md = (txt.strip() or "_(no executive summary returned)_")
        except Exception as e:
            exec_summary_md = f"_(LLM executive summary failed: {type(e).__name__}: {e})_"

    md_lines: List[str] = []
    md_lines.append("# BENCHMARK SUMMARY")
    md_lines.append("")
    md_lines.append(f"**benchmark.tsv:** `{os.path.abspath(os.path.normpath(bench_path))}`")
    md_lines.append(f"**run_dir:** `{os.path.abspath(os.path.normpath(run_dir))}`")
    md_lines.append(f"**created_at:** `{datetime.now().isoformat(timespec='seconds')}`")
    md_lines.append("")
    md_lines.append("## Canonicals used")
    md_lines.append("```json")
    md_lines.append(json.dumps(canonicals, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Overall counts")
    md_lines.append(_df_to_md_table(counts_df))
    md_lines.append("")
    md_lines.append("## By gene")
    md_lines.append(_df_to_md_table(by_gene) if by_gene.shape[0] else "_(no gene column)_")
    md_lines.append("")
    md_lines.append("## Executive summary")
    md_lines.append(exec_summary_md)
    md_lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return {"benchmark_summary_csv": out_csv, "benchmark_summary_md": out_md}


# ============================================================
# MAIN — CLI benchmark path(s) + per-row run (no interactive input)
# ============================================================

if __name__ == "__main__":
    import argparse
    import sys

    os.makedirs(TEST_RUNS_ROOT, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run Splicing-Agent on one or more benchmark TSVs.")
    parser.add_argument(
        "--bench",
        nargs="+",
        default=[DEFAULT_BENCHMARK],
        help="One or more benchmark TSV paths (space-separated).",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="strict",
        choices=["strict", "minimal", "loose"],
        help="Which system prompt to use"
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        default="test_case",
        choices=list(DATASET_PATHS.keys()),
        help="Which benchmark dataset TSV to run"
    )
    args = parser.parse_args()
    
    system_prompt, prompt_name = load_prompt_from_key(args.prompt_key)
    
    # If user passed --bench, respect it. Otherwise default to dataset_key path.
    if args.bench and args.bench != [DEFAULT_BENCHMARK]:
        bench_paths = [norm_path(p) for p in args.bench]
    else:
        dataset_path_raw = DATASET_PATHS[args.dataset_key]
        bench_paths = [norm_path(dataset_path_raw)]

    CANONICALS = {
        "BRCA1": "ENST00000357654",   # BRCA1-203
        "VEGFA": "ENST00000372055",   # VEGFA-004
    }

    # Optional LLM init (JUDGE + optional run-level exec summary)
    planner_llm = None
    judge_llm = None

    if HAVE_LLM:
        try:
            load_dotenv()
            key = os.getenv("OPENAI_API_KEY")
            if key:
                os.environ["OPENAI_API_KEY"] = key
                base_llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0,
                    api_key=key,
                )

                # Force JSON object output for judge
                planner_llm = base_llm.bind(response_format={"type": "json_object"})  # planning model
                judge_llm = base_llm.bind(response_format={"type": "json_object"})
                llm = judge_llm
                print("✅ Planner + Judge LLM initialized.")
            else:
                print("⚠️  No OPENAI_API_KEY found; planner + judge will be disabled.")
        except Exception as e:
            print(f"⚠️  LLM init failed ({type(e).__name__}: {e}); disabled.")
            llm = None
    else:
        print("⚠️ LLM dependencies missing; planner + judge disabled.")

    # -------------------------
    # BioMart TSV loaders
    # -------------------------

    def _load_df_cached(path: str, _cache: Dict[str, pd.DataFrame] = {}) -> pd.DataFrame:
        ap = os.path.abspath(os.path.normpath(path))
        if ap not in _cache:
            _cache[ap] = pd.read_csv(ap, sep="\t", dtype=str, low_memory=False).fillna("")
        return _cache[ap]

    def _parse_semicolon_ints(x: str) -> List[int]:
        if x is None:
            return []
        s = str(x).strip()
        if s == "" or s.lower() == "na":
            return []
        out: List[int] = []
        for v in s.split(";"):
            v = v.strip()
            if v == "" or v.lower() == "na":
                continue
            out.append(int(v))
        return out

    def exon_loader(dataset_path: str, transcript_id: str) -> Dict[str, Any]:
        df = _load_df_cached(dataset_path)
        sub = df[df["Transcript stable ID"] == transcript_id]
        if sub.shape[0] == 0:
            raise ValueError(f"Transcript stable ID not found: {transcript_id}")

        row = sub.iloc[0]
        starts = _parse_semicolon_ints(row.get("Exon region start (bp)", ""))
        ends = _parse_semicolon_ints(row.get("Exon region end (bp)", ""))
        ranks = _parse_semicolon_ints(row.get("Exon rank in transcript", ""))

        if not (len(starts) == len(ends) == len(ranks)):
            raise ValueError(f"Exon starts/ends/ranks mismatch for {transcript_id}")

        exons = list(zip(ranks, starts, ends))
        exons.sort(key=lambda t: t[0])

        exon_table: List[Dict[str, Any]] = []
        tx_cursor = 1
        for r, s, e in exons:
            exon_len = int(e) - int(s) + 1
            exon_table.append({
                "rank": int(r),
                "start_genome": int(s),
                "end_genome": int(e),
                "start_tx": int(tx_cursor),
                "end_tx": int(tx_cursor + exon_len - 1),
                "len": int(exon_len),
            })
            tx_cursor += exon_len

        tx_len = tx_cursor - 1

        strand_raw = row.get("Strand", "")
        try:
            strand = int(strand_raw) if str(strand_raw).strip() != "" else 1
        except Exception:
            strand = 1

        return {"exon_table": exon_table, "tx_len": int(tx_len), "strand": int(strand)}

    def cds_segment_loader(dataset_path: str, transcript_id: str) -> List[Tuple[int, int]]:
        df = _load_df_cached(dataset_path)
        sub = df[df["Transcript stable ID"] == transcript_id]
        if sub.shape[0] == 0:
            raise ValueError(f"Transcript stable ID not found: {transcript_id}")
        row = sub.iloc[0]
        cds_starts = _parse_semicolon_ints(row.get("cDNA coding start", ""))
        cds_ends = _parse_semicolon_ints(row.get("cDNA coding end", ""))
        if len(cds_starts) != len(cds_ends):
            raise ValueError(f"CDS start/end mismatch for {transcript_id}")
        segs = sorted(list(zip(cds_starts, cds_ends)), key=lambda t: t[0])
        return [(int(a), int(b)) for a, b in segs if int(a) > 0 and int(b) > 0]

    def seq_loader(dataset_path: str, transcript_id: str) -> str:
        df = _load_df_cached(dataset_path)
        sub = df[df["Transcript stable ID"] == transcript_id]
        if sub.shape[0] == 0:
            raise ValueError(f"Transcript stable ID not found: {transcript_id}")
        row = sub.iloc[0]
        seq = (row.get("cDNA sequences", "") or "").upper()
        seq = "".join([c for c in seq if c in "ACGTN"])
        return seq

    # Tools + graph (reused across datasets)
    cds = cds_tool(exon_loader=exon_loader, cds_segment_loader=cds_segment_loader, seq_loader=seq_loader)
    nmd = nmd_tool(ejc_threshold_nt=55, require_ptc_for_nmd=True, margin_nt=55)
    motif = motif_tool()
    tavily = tavily_tool()

    graph = build_graph(cds, nmd, motif, tavily, planner_llm=planner_llm, judge_llm=judge_llm)

    # ============================================================
    # LOOP OVER DATASETSF
    # ============================================================
    for bench_path in bench_paths:
        bench_path = norm_path(bench_path) 
        _assert_file_exists(bench_path, "benchmark.tsv")

        # run_dir per dataset
        base = os.path.splitext(os.path.basename(bench_path))[0]
        safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)[:60]
        run_dir = make_test_run_dir(prefix=f"test_run_{safe_base}")

        print("\n================= Splicing-Agent =================")
        print(f"Python: {sys.executable}")
        print(f"Benchmark TSV: {bench_path}")
        print(f"Run folder: {run_dir}\n")

        # --- Graph visualization (Mermaid) per dataset ---
        try:
            mermaid = graph.get_graph().draw_mermaid()
            with open(os.path.join(run_dir, "graph.mmd"), "w", encoding="utf-8") as f:
                f.write(mermaid)
            try:
                png_bytes = graph.get_graph().draw_mermaid_png()
                with open(os.path.join(run_dir, "graph.png"), "wb") as f:
                    f.write(png_bytes)
            except Exception as e:
                print(f"⚠️  PNG export not available: {e}")
        except Exception as e:
            print(f"⚠️  Could not render Mermaid graph: {e}")

        # Read benchmark TSV
        df = pd.read_csv(bench_path, sep="\t", dtype=str).fillna("")

        # Outputs (per dataset)
        jsonl_path = os.path.join(run_dir, "test_case_runs.jsonl")
        runs_json_path = os.path.join(run_dir, "test_case_runs.json")
        summary_path = os.path.join(run_dir, "test_case_summary.csv")
        failures_only_path = os.path.join(run_dir, "failures_only.csv")

        summary_rows: List[Dict[str, Any]] = []
        all_runs: List[Dict[str, Any]] = []

        with open(jsonl_path, "w", encoding="utf-8") as fjsonl:
            for i, (_, row) in enumerate(df.iterrows()):
                dataset_path = norm_path(
                    (row.get("dataset_path", "") or "").strip() or bench_path
                )

                canonical_dataset_path = norm_path(
                    (row.get("canonical_dataset_path", "") or "").strip() or bench_path
                )

                transcript_id = (row.get("transcript_id", "") or "").strip() or (row.get("Transcript stable ID", "") or "").strip()
                gene_symbol_hint = (row.get("gene_symbol_hint", "") or "").strip() or (row.get("Gene name", "") or "").strip()
                chromosome_hint = (row.get("chromosome_hint", "") or "").strip() or (row.get("Chromosome/scaffold name", "") or "").strip()

                canonical_transcript_id = (row.get("canonical_transcript_id", "") or "").strip() or CANONICALS.get(gene_symbol_hint, "")
                expected_label = (row.get("expected_label", "") or "").strip() or (row.get("expected", "") or "").strip()

                row_id = make_row_id(i, transcript_id=transcript_id, gene=gene_symbol_hint)

                # Compute canonical stop_end_tx deterministically
                canonical_stop_end_tx = None
                canonical_tx_len = None
                canonical_lastJ = None
                can_err = None
                try:
                    if canonical_transcript_id:
                        can_exon_payload = exon_loader(dataset_path=canonical_dataset_path, transcript_id=canonical_transcript_id)
                        can_cds_segments = cds_segment_loader(dataset_path=canonical_dataset_path, transcript_id=canonical_transcript_id)

                        can_exons = can_exon_payload["exon_table"]
                        if len(can_exons) >= 2:
                            canonical_lastJ = int(can_exons[-2]["end_tx"])
                        canonical_tx_len = int(can_exon_payload["tx_len"])
                        canonical_stop_end_tx = max((e for _, e in can_cds_segments), default=None)
                    else:
                        can_err = "canonical_transcript_id missing"
                except Exception as e:
                    can_err = f"canonical compute failed: {type(e).__name__}: {e}"
                    canonical_stop_end_tx = None

                state: SplicingAgentState = {
                    "row_id": row_id,
                    "run_dir": run_dir,
                    "dataset_path": dataset_path,
                    "system_prompt": system_prompt,
                    "prompt_name": prompt_name,
                    "prompt_key": args.prompt_key,
                    "dataset_key": args.dataset_key,
                    "benchmark_tsv": bench_path,
                    "canonical_dataset_path": canonical_dataset_path,
                    "transcript_id": transcript_id,
                    "canonical_transcript_id": canonical_transcript_id,
                    "gene_symbol_hint": gene_symbol_hint,
                    "chromosome_hint": chromosome_hint,
                    "expected_label": expected_label,

                    "canonical_stop_end_tx": canonical_stop_end_tx,
                    "canonical_tx_len": canonical_tx_len,
                    "canonical_lastJ": canonical_lastJ,

                    "plan_done": [],
                    "tool_calls": [],
                    "failure_modes": [],
                    "failure_notes": {},
                    "trace": [],
                    "errors": [],
                    "tokens_total": None,
                    "hallucination": False,

                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }

                if can_err:
                    state["errors"].append(f"canonical note: {can_err}")
                    _add_failure(state, "CANONICAL_UNDEFINED", "Canonical transcript/stop could not be computed.")

                t0 = time.perf_counter()
                try:
                    config = {"configurable": {"thread_id": f"{safe_base}::{row_id}"}}
                    out = graph.invoke(state, config=config)
                except Exception as e:
                    out = dict(state)
                    out.setdefault("errors", []).append(f"graph.invoke failed: {type(e).__name__}: {e}")
                    out.setdefault("errors", []).append(traceback.format_exc())
                    out.setdefault("report_tables_md", "")
                    out.setdefault("report_text", "")

                out["latency_s"] = time.perf_counter() - t0
                out["tool_calls_count"] = len(out.get("tool_calls", []) or [])

                # Write per-case artifacts
                try:
                    write_case_artifacts(
                        run_dir,
                        row_id=row_id,
                        record=dict(out),
                        report_text=out.get("report_text", ""),
                        consequence_md=out.get("report_consequence_md", ""),
                        metrics_md=out.get("report_metrics_md", ""),
                    )
                except Exception as e:
                    out.setdefault("errors", []).append(f"write_case_artifacts failed: {e}")

                # JSONL
                fjsonl.write(json.dumps(dict(out), ensure_ascii=False, default=str) + "\n")
                all_runs.append(dict(out))

                # Summary CSV row
                fp = (out.get("dataset_fingerprint") or {}).get("fingerprint", {})
                summary_rows.append({
                    "row_id": row_id,
                    "prompt_key": out.get("prompt_key", args.prompt_key),
                    "prompt_name": out.get("prompt_name", prompt_name),
                    "dataset_key": out.get("dataset_key", args.dataset_key),
                    "benchmark_tsv": bench_path,
                    "dataset_path": out.get("dataset_path"),
                    "size_bytes": fp.get("size_bytes"),
                    "md5_first64kb": fp.get("md5_first64kb"),
                    "transcript_id": out.get("transcript_id"),
                    "gene_symbol_hint": out.get("gene_symbol_hint"),
                    "tx_len": out.get("tx_len"),
                    "lastJ": out.get("lastJ"),
                    "stop_end_tx": out.get("stop_end_tx"),
                    "dist": out.get("dist_lastJ_minus_stopEnd"),
                    "canonical_transcript_id": out.get("canonical_transcript_id"),
                    "canonical_stop_end_tx": out.get("canonical_stop_end_tx"),
                    "margin_nt": out.get("margin_nt"),
                    "ptc_predicted": out.get("ptc_predicted"),
                    "nmd": out.get("nmd"),
                    "motif_hits_count": len(out.get("motif_hits", []) or []),
                    "literature_notes_count": len(out.get("literature_notes", []) or []),
                    "predicted_label": out.get("predicted_label", ""),
                    "expected_label": out.get("expected_label", ""),
                    "failure_modes": ",".join(out.get("failure_modes", []) or []),
                    "has_critical_failure": out.get("has_critical_failure", False),
                    "task_completed": out.get("task_completed", False),
                    "tool_usage_accuracy": out.get("tool_usage_accuracy", False),
                    "success": out.get("success", False),
                    "error_rate_flag": out.get("error_rate_flag", False),
                    "hallucination": out.get("hallucination", False),
                    "tokens_total": out.get("tokens_total", None),
                    "latency_s": out.get("latency_s", None),
                    "status": "error" if (out.get("errors") or []) else "ok",
                })

        # Write test_case_runs.json (single JSON array)
        with open(runs_json_path, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2, ensure_ascii=False, default=str)

        # Write test_case_summary.csv
        df_sum = pd.DataFrame(summary_rows)
        df_sum.to_csv(summary_path, index=False)

        # Failures-only CSV (review triage)
        try:
            def _has_any_failure_modes(x) -> bool:
                # if stored as list -> list length
                if isinstance(x, list):
                    return len(x) > 0
                # if stored as stringified list, comma-joined, or "" -> cheap check
                s = str(x or "").strip()
                return s not in ("", "[]", "None")

            fail_mask = (
                (df_sum["error_rate_flag"].fillna(False).astype(bool))
                | (df_sum["hallucination"].fillna(False).astype(bool))
                | (df_sum["has_critical_failure"].fillna(False).astype(bool))
                | (df_sum["status"] == "error")
                | (df_sum["failure_modes"].apply(_has_any_failure_modes))
            )

            df_fail = df_sum.loc[fail_mask].copy()
            df_fail.to_csv(failures_only_path, index=False)
        except Exception as e:
            print(f"⚠️  Could not write failures_only.csv: {e}")

        # write benchmark_summary.csv + benchmark_summary.md
        bench_out_paths = write_benchmark_summaries(
            run_dir=run_dir,
            bench_path=bench_path,
            summary_rows=summary_rows,
            canonicals=CANONICALS,
            llm=llm,
        )

        # Manifest (per dataset)
        manifest = {
            "run_dir": run_dir,
            "benchmark_tsv": os.path.abspath(os.path.normpath(bench_path)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_rows": int(df.shape[0]),
            "jsonl_path": jsonl_path,
            "runs_json_path": runs_json_path,
            "summary_csv_path": summary_path,
            "failures_only_csv_path": failures_only_path,
            "benchmark_summary_csv": bench_out_paths.get("benchmark_summary_csv"),
            "benchmark_summary_md": bench_out_paths.get("benchmark_summary_md"),
            "pretty_json_dir": os.path.join(run_dir, "pretty_json"),
            "reports_md_dir": os.path.join(run_dir, "reports_md"),
            "canonicals": CANONICALS,
        }
        with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\n✅ Dataset run complete")
        print(f"- Run folder: {run_dir}")
        print(f"- JSONL: {jsonl_path}")
        print(f"- Runs JSON: {runs_json_path}")
        print(f"- Summary CSV: {summary_path}")
        print(f"- Failures-only CSV: {failures_only_path}")
        print(f"- Benchmark summary CSV: {manifest['benchmark_summary_csv']}")
        print(f"- Benchmark summary MD: {manifest['benchmark_summary_md']}")
        print(f"- Canonicals: {CANONICALS}")

    print("\n✅ Complete")