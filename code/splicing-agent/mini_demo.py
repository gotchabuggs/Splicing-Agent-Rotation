"""
mini_demo.py

Interactive terminal-based demonstration script for the Splicing-Agent pipeline.

Purpose
-------
This script runs a single transcript case through the LangGraph-based
Splicing-Agent workflow and prints a human-readable, audience-friendly
summary in the terminal. It is designed for:

• Rotation talks
• Live demos
• Quick sanity checks of a single transcript
• Lab walkthroughs without LangSmith

Pipeline Executed
-----------------
CDS → AGENT → NMD → MOTIF → TAVILY → TABLES → FAILURE_COMPILER → FINAL

Important Notes
---------------
- This script imports `splicing_agent_test_case.py` as a module.
- LangSmith is NOT required.
- LLM judge is disabled by default for live-demo stability.
- Assumes BioMart-style TSV input schema.

Outputs
-------
Writes two files to a timestamped run directory:
    • demo_report.md      → human-readable consequence report
    • demo_state.json     → full agent state after execution

Optional Dependencies
---------------------
For improved terminal rendering:
    pip install rich tabulate
"""

from __future__ import annotations

import os
import sys
import time
import json
import traceback
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

# ------------------------------------------------------------
# Optional: Pretty Terminal endering (Rich)
# ------------------------------------------------------------
HAVE_RICH = False
Console = None
Table = None
Panel = None

try:
    from rich.console import Console as _Console
    from rich.table import Table as _Table
    from rich.panel import Panel as _Panel

    HAVE_RICH = True
    Console = _Console
    Table = _Table
    Panel = _Panel
except Exception:
    HAVE_RICH = False

# ------------------------------------------------------------
# Importing Splicing-Agent Module
# ------------------------------------------------------------

try:
    import splicing_agent_test_case as SA
except Exception as e:
    print("\nERROR: Could not import splicing_agent_test_case.py")
    print("Make sure mini_demo.py is in the same folder as splicing_agent_test_case.py.")
    print(f"Import error: {type(e).__name__}: {e}\n")
    sys.exit(1)

# ------------------------------------------------------------
# Utility Helper Functions
# ------------------------------------------------------------
def _now() -> str:
    """
    Return current timestamp as formatted string.
    
    Used for:
    - Banner display
    - Run directory naming
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _exists_nonempty(path: str) -> bool:
    """
    Check whether a file exists at the given path and is non-empty.
    
    Parameters:
    - path: str → file path to check

    Returns:
    - bool → True if file exists and has size > 0, else False
    """
    ap = os.path.abspath(os.path.normpath(path))
    return os.path.exists(ap) and os.path.getsize(ap) > 0


def _prompt(default: str, msg: str) -> str:
    """
    Prompt the user for input with a default value.

    Parameters:
    - default: str → default value to use if user input is empty
    - msg: str → message to display in the prompt

    Retunrns:
    - str → user input or default if input is empty
    """
    s = input(f"{msg} [{default}]: ").strip()
    return s or default


def _say(console: Optional[Any], who: str, text: str) -> None:
    """
    Print a message to the console, using rich formatting if available.

    Parameters:
    - console: Optional[Any] → rich Console object if available, else None
    - who: str → speaker or source of the message
    - text: str → message content to display

    Returns:
    - None → prints output to console
    """
    if HAVE_RICH and console is not None:
        console.print(f"[bold]{who}:[/bold] {text}")
    else:
        print(f"{who}: {text}")


def _banner(console: Optional[Any]) -> None:
    """
    Display a banner with the demo title, subtitle, and current timestamp.

    Parameters:
    - console: Optional[Any] → rich Console object if available, else None
    
    Returns:
    - None → prints banner to console
    """
    title = "🧬🍳 Splicing-Agent Mini Demo"
    subtitle = "CDS → AGENT → NMD (forced) → MOTIF → TAVILY → TABLES → FAILURE_COMPILER → FINAL"
    if HAVE_RICH and console is not None:
        console.print(Panel.fit(f"[bold]{title}[/bold]\n{subtitle}\n({_now()})"))
    else:
        print("=" * 90)
        print(title)
        print(subtitle)
        print(f"({_now()})")
        print("=" * 90)


def _load_df_cached(path: str, _cache: Dict[str, pd.DataFrame] = {}) -> pd.DataFrame:
    """
    Load a TSV file into a pandas DataFrame with caching to avoid redundant reads during demo execution.

    Parameters:
    - path: str → file path to the TSV dataset
    - _cache: Dict[str, pd.DataFrame] → internal cache dictionary (default empty
        and persists across calls)

    Assumptions:
    - The TSV file has a header row and is tab-delimited.
    - The file is expected to be reasonably sized for in-memory loading (typical for BioMart exports).
    
    Returns:
    - pd.DataFrame → loaded DataFrame from the TSV file
    - Raises ValueError if the file cannot be loaded or is empty.
    - Caches DataFrames by absolute path to optimize repeated loads of the same file.
    - Uses pandas to read TSV with dtype=str and fills NA values with empty strings.
    """
    ap = os.path.abspath(os.path.normpath(path))
    if ap not in _cache:
        _cache[ap] = pd.read_csv(ap, sep="\t", dtype=str, low_memory=False).fillna("")
    return _cache[ap]


def _parse_semicolon_ints(x: str) -> List[int]:
    """
    Parse a semicolon-separated string of integers into a list of ints, handling edge cases.
    
    Parameters:
    - x: str → input string, e.g. "100; 200; 300"
    Returns:
    - List[int] → list of integers parsed from the string, e.g. [100, 200, 300]
    
    Edge Cases:
    - If x is None, empty, or "NA" (case-insensitive), returns
        an empty list.
    - Ignores empty or "NA" values between semicolons, e.g. "100; NA; 200;;300" → [100, 200, 300]
    - Strips whitespace around values before parsing.
    - If any value cannot be parsed as an int, it will raise a ValueError.
    - Assumes that valid integers are non-negative and that the input format is consistent with BioMart exports.
    - Designed for robustness in parsing exon start/end positions and ranks from BioMart TSV fields.
    """
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


def _safe_join_path(*parts: str) -> str:
    """
    Join multiple path components into a single absolute path, normalizing it to prevent issues with redundant separators or relative references.

    Parameters:
    - *parts: str → variable number of path components to join, e.g. ("
        data", "test_runs", "demo_20240601_123000")

    Returns:
    - str → an absolute, normalized path resulting from joining the input components, e.g.
        "C:/Users/justi/OneDrive/Desktop/CU-Anschutz/repos/davidsonlab/Splicing-Agent-Rotation/code/data/test_runs/demo_20240601_123000"

    """
    return os.path.abspath(os.path.normpath(os.path.join(*parts)))

# ------------------------------------------------------------
# Minimal BioMart TSV loaders
# ------------------------------------------------------------
def exon_loader(dataset_path: str, transcript_id: str) -> Dict[str, Any]:
    """
    Load exon information for a given transcript ID from a BioMart-style TSV dataset.

    Converts exon start/end positions and ranks into a structured exon table with genomic and transcript coordinates.

    Designed for use in the Splicing-Agent demo to provide exon structure information for a given transcript ID.

    Requires the TSV to have columns:
    - "Transcript stable ID"
    - "Exon region start (bp)"
    - "Exon region end (bp)"
    - "Exon rank in transcript"
    - "Strand" (optional, defaults to 1 if missing or invalid)

    Parameters:
    - dataset_path: str → file path to the TSV dataset
    - transcript_id: str → the transcript stable ID to look up, e.g. "
        ENST00000357654"
    
    Returns:
    - Dict[str, Any] with keys:
        - "exon_table": List[Dict[str, Any]] → list of exon dictionaries with keys:
            - "rank": int → exon rank in the transcript
            - "start_genome": int → exon start position in genomic coordinates
            - "end_genome": int → exon end position in genomic coordinates
            - "start_tx": int → exon start position in transcript coordinates (1-based)
            - "end_tx": int → exon end position in transcript coordinates (1-based)
            - "len": int → length of the exon in nucleotides
        - "tx_len": int → total length of the transcript in nucleotides (sum of exon lengths)
        - "strand": int → strand of the transcript (1 for forward, -1 for reverse; defaults to 1 if missing or invalid)
    - Raises ValueError if the transcript ID is not found in the dataset or if exon information cannot be parsed.
    - Caches the loaded DataFrame to optimize repeated calls with the same dataset path.
    
    """
    df = _load_df_cached(dataset_path)
    sub = df[df["Transcript stable ID"] == transcript_id]
    if sub.shape[0] == 0:
        raise ValueError(f"Transcript stable ID not found: {transcript_id}")

    row = sub.iloc[0]
    starts = _parse_semicolon_ints(row.get("Exon region start (bp)", ""))
    ends = _parse_semicolon_ints(row.get("Exon region end (bp)", ""))
    ranks = _parse_semicolon_ints(row.get("Exon rank in transcript", ""))

    exons = list(zip(ranks, starts, ends))
    exons.sort(key=lambda t: t[0])

    exon_table: List[Dict[str, Any]] = []
    tx_cursor = 1
    for r, s, e in exons:
        exon_len = int(e) - int(s) + 1
        exon_table.append(
            {
                "rank": int(r),
                "start_genome": int(s),
                "end_genome": int(e),
                "start_tx": int(tx_cursor),
                "end_tx": int(tx_cursor + exon_len - 1),
                "len": int(exon_len),
            }
        )
        tx_cursor += exon_len

    strand_raw = row.get("Strand", "")
    try:
        strand = int(strand_raw) if str(strand_raw).strip() != "" else 1
    except Exception:
        strand = 1

    return {"exon_table": exon_table, "tx_len": tx_cursor - 1, "strand": strand}


def cds_segment_loader(dataset_path: str, transcript_id: str) -> List[Tuple[int, int]]:
    """
    Load CDS segment information for a given transcript ID from a BioMart-style TSV dataset.

    Designed for use in the Splicing-Agent demo to provide CDS segment information for a given transcript ID.

    Requires the TSV to have columns:
    - "Transcript stable ID"
    - "cDNA coding start"   
    - "cDNA coding end"

    Parameters:
    - dataset_path: str → file path to the TSV dataset
    - transcript_id: str → the transcript stable ID to look up, e.g. "
        ENST00000357654"
    Returns:
    - List[Tuple[int, int]] → list of (start, end) tuples representing CDS segments in transcript coordinates (1-based)
    - Only includes segments where both start and end are > 0, as per BioMart conventions (0 or negative values indicate non-coding regions).
    - Raises ValueError if the transcript ID is not found in the dataset or if CDS information cannot be parsed.
    - Caches the loaded DataFrame to optimize repeated calls with the same dataset path.
    """
    df = _load_df_cached(dataset_path)
    sub = df[df["Transcript stable ID"] == transcript_id]
    row = sub.iloc[0]
    cds_starts = _parse_semicolon_ints(row.get("cDNA coding start", ""))
    cds_ends = _parse_semicolon_ints(row.get("cDNA coding end", ""))
    segs = sorted(list(zip(cds_starts, cds_ends)), key=lambda t: t[0])
    return [(int(a), int(b)) for a, b in segs if int(a) > 0 and int(b) > 0]


def seq_loader(dataset_path: str, transcript_id: str) -> str:
    """
    Load the cDNA sequence for a given transcript ID from a BioMart-style TSV dataset.
    Designed for use in the Splicing-Agent demo to provide the cDNA sequence for a given transcript ID.

    Requires the TSV to have columns:
    - "Transcript stable ID"
    - "cDNA sequences"
    
    Parameters:
    - dataset_path: str → file path to the TSV dataset
    - transcript_id: str → the transcript stable ID to look up, e.g. "
        ENST00000357654"
    
    Returns:
    - str → the cDNA sequence for the specified transcript ID, with all characters converted to
        uppercase and non-ACGTN characters removed.
    - If the "cDNA sequences" field is missing, empty, or null for the specified transcript, returns an empty string.
    - Raises ValueError if the transcript ID is not found in the dataset.       
    - Caches the loaded DataFrame to optimize repeated calls with the same dataset path.
    """
    df = _load_df_cached(dataset_path)
    sub = df[df["Transcript stable ID"] == transcript_id]
    row = sub.iloc[0]
    seq = (row.get("cDNA sequences", "") or "").upper()
    return "".join([c for c in seq if c in "ACGTN"])

# ------------------------------------------------------------
# Human-friendly summary “chart”
# ------------------------------------------------------------
FRIENDLY_ROWS = [
    # Identity
    ("Case", "row_id", "Unique identifier for this demo run", "Identity"),
    ("Transcript", "transcript_id", "The isoform being interpreted", "Identity"),
    ("Gene", "gene_symbol_hint", "Biological context for the transcript", "Identity"),

    # Core biology
    ("Transcript length (nt)", "tx_len", "Total transcript length", "Core biology"),
    ("Last exon–exon junction (nt)", "lastJ", "Used for the NMD 55-nt rule", "Core biology"),
    ("Stop codon position (nt)", "stop_end_tx", "Where translation stops", "Core biology"),
    ("Distance (junction − stop)", "dist_lastJ_minus_stopEnd", "Supports NMD if ≥55 nt", "Core biology"),

    # Reference
    ("Canonical transcript", "canonical_transcript_id", "Reference isoform", "Reference"),
    ("Canonical stop position", "canonical_stop_end_tx", "Baseline stop position", "Reference"),

    # Calls
    ("Early stop (PTC)?", "ptc_predicted", "Premature stop truncates protein", "Calls"),
    ("NMD predicted?", "nmd", "Transcript predicted to be degraded", "Calls"),
    ("Final consequence", "predicted_label", "Human-readable outcome label", "Calls"),

    # Quality
    ("Report produced?", "task_completed", "Pipeline produced a report", "Quality"),
    ("Workflow valid?", "tool_usage_accuracy", "Hard constraints followed", "Quality"),
    ("Overall success?", "success", "No critical failures", "Quality"),
    ("Needs review?", "error_rate_flag", "Flags problems for inspection", "Quality"),
]

BOOL_KEYS = {
    "ptc_predicted", "nmd", "task_completed",
    "tool_usage_accuracy", "success", "error_rate_flag"
}

def _status(val: Any, key: str) -> str:
    if key == "error_rate_flag":
        return "⚠️  REVIEW" if val else "✅  CLEAR"
    return "✅  YES" if val else "—   NO"

def _headline(out: Dict[str, Any]) -> str:
    gene = out.get("gene_symbol_hint", "")
    tx = out.get("transcript_id", "")
    label = out.get("predicted_label", "")
    if out.get("nmd"):
        hint = "🧹 NMD predicted"
    elif out.get("ptc_predicted"):
        hint = "✂️ Early stop, truncated protein"
    else:
        hint = "🧬 Likely protein-coding"
    return f"{gene} {tx} → {label}   {hint}"

def _print_summary_table(console: Optional[Any], out: Dict[str, Any]) -> None:
    groups: Dict[str, List[Tuple[str, str, str]]] = {}
    for label, key, why, grp in FRIENDLY_ROWS:
        groups.setdefault(grp, []).append((label, key, why))

    if HAVE_RICH and console is not None:
        console.print(Panel.fit(f"[bold]Outcome[/bold]\n{_headline(out)}"))

        t = Table(title="Splicing-Agent Demo — Human-Friendly Summary", show_lines=True)
        t.add_column("What")
        t.add_column("Value / Status")
        t.add_column("Why it matters")

        for grp, rows in groups.items():
            t.add_row(f"[bold]{grp}[/bold]", "", "")
            for label, key, why in rows:
                val = out.get(key)
                val_str = _status(val, key) if key in BOOL_KEYS else str(val)
                t.add_row(label, val_str, why)

        console.print(t)
        console.print(f"[bold]Runtime:[/bold] {out.get('latency_s',0):.3f}s")

    else:
        print("\n" + "=" * 90)
        print("Splicing-Agent Demo — Summary")
        print(_headline(out))
        print("=" * 90)
        for grp, rows in groups.items():
            print(f"\n[{grp}]")
            for label, key, why in rows:
                val = out.get(key)
                val_str = _status(val, key) if key in BOOL_KEYS else str(val)
                print(f"- {label}: {val_str}")
                print(f"  ↳ {why}")
        print(f"\nRuntime: {out.get('latency_s',0):.3f}s")
        print("=" * 90)

# ------------------------------------------------------------
# Write outputs
# ------------------------------------------------------------
def _write_demo_outputs(run_dir: str, out: Dict[str, Any]) -> None:
    """
    Write the demo outputs to files in the specified run directory.

    The run directory is created if it does not already exist. If the directory cannot be created, an exception will be raised.

    Parameters:
    - run_dir: str → the directory where output files will be saved
    - out: Dict[str, Any] → the output dictionary containing results and reports from the Splicing-Agent execution

    Returns:
    - None → writes files to disk
    Files Written:
    - demo_report.md → a markdown file containing the human-readable consequence report, extracted from the output
        dictionary (keys: "report_consequence_md" or "report_text"). If neither key is present, writes "_(no report)_".
    - demo_state.json → a JSON file containing the full output dictionary, pretty-printed with indentation for readability. All values are converted to strings if they are not JSON-serializable.
    """
    os.makedirs(run_dir, exist_ok=True)
    md = out.get("report_consequence_md") or out.get("report_text") or "_(no report)_"
    with open(os.path.join(run_dir, "demo_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(run_dir, "demo_state.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

# ------------------------------------------------------------
# Main demo
# ------------------------------------------------------------
def main() -> None:
    """
    Main function to run the Splicing-Agent mini demo.
    This function orchestrates the interactive terminal-based demonstration of the Splicing-Agent pipeline. 
    It performs the following steps:
    1. Displays a banner with the demo title and timestamp.
    2. Prompts the user for the path to a TSV dataset and a transcript ID to interpret.
    3. Loads necessary tools and builds the LangGraph workflow.         
    4. Initializes the agent state with the provided inputs and metadata.
    5. Invokes the workflow and measures execution time.
    6. Prints a human-friendly summary table of the results to the console.
    7. Writes the detailed outputs (report and state) to a timestamped run directory for later inspection.
        The function is designed for use in interactive settings such as rotation talks, live demos, and lab walkthroughs, and does not require LangSmith or an LLM judge for stability.
    8. Handles errors gracefully by printing informative messages and exiting if critical issues arise (e.g., missing TSV file, import errors).

    Parameters:
    - None → all inputs are gathered interactively from the user.
    
    Returns:
    - None → outputs are printed to the console and written to files, but not returned from
        the function.
    """
    console = Console() if HAVE_RICH and Console is not None else None
    _banner(console)

    default_bench = getattr(SA, "DEFAULT_BENCHMARK", "")
    repo_root = getattr(SA, "REPO_ROOT", os.getcwd())

    CANONICALS = {
        "BRCA1": "ENST00000357654",
        "VEGFA": "ENST00000372055",
    }

    _say(console, "Splicing-Agent", "Hi! I can run one splicing case.")
    bench_path = _prompt(default_bench or "path/to/your.tsv", "Path to TSV")
    bench_path = os.path.abspath(os.path.normpath(bench_path))
    if not _exists_nonempty(bench_path):
        print("TSV not found or empty.")
        sys.exit(1)

    transcript_id = _prompt("ENST00000357654", "Transcript_ID to interpret")
    gene_hint = input("You (optional gene symbol; Enter to skip): ").strip() or "UNKNOWN"

    canonical_transcript_id = CANONICALS.get(gene_hint, "")
    override = input("You (optional canonical override; Enter to skip): ").strip()
    if override:
        canonical_transcript_id = override

    cds = SA.cds_tool(exon_loader, cds_segment_loader, seq_loader)
    nmd = SA.nmd_tool()
    motif = SA.motif_tool()
    tavily = SA.tavily_tool()

    graph = SA.build_graph(cds, nmd, motif, tavily, llm=None)

    run_dir = _safe_join_path(repo_root, "data", "test_runs", f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    state: SA.SplicingAgentState = {
        "row_id": f"demo__{transcript_id}__{gene_hint}",
        "run_dir": run_dir,
        "dataset_path": bench_path,
        "canonical_dataset_path": bench_path,
        "transcript_id": transcript_id,
        "canonical_transcript_id": canonical_transcript_id,
        "gene_symbol_hint": gene_hint,
        "plan_done": [],
        "tool_calls": [],
        "failure_modes": [],
        "failure_notes": {},
        "trace": [],
        "errors": [],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    t0 = time.perf_counter()
    config = {"configurable": {"thread_id": f"demo::{state['row_id']}"}}
    out = graph.invoke(state, config=config)
    out["latency_s"] = time.perf_counter() - t0


    _print_summary_table(console, out)

    _say(console, "Splicing-Agent", "Evidence tables:\n")
    print(out.get("report_tables_md", ""))

    _write_demo_outputs(run_dir, out)
    print(f"\nOutputs saved to: {run_dir}")

if __name__ == "__main__":
    main()