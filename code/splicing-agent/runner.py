import os
import re
import time
import json
import traceback
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from splicing_agent.state import SplicingAgentState

from splicing_agent.tools.cds import cds_tool
from splicing_agent.tools.nmd import nmd_tool
from splicing_agent.tools.motif import motif_tool
from splicing_agent.tools.tavily import tavily_tool
from splicing_agent.graph import build_graph
from splicing_agent.reporting import write_case_artifacts, write_benchmark_summaries
from splicing_agent.helpers import _add_failure, make_test_run_dir, make_row_id, _assert_file_exists, tracing_context
from splicing_agent.config import HAVE_LLM, DEFAULT_BENCHMARK, TEST_RUNS_ROOT


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
        "--trace_bulk",
        action="store_true",
        help="Enable LangSmith tracing for the bulk pass (NOT recommended). Default: OFF.",
    )
    parser.add_argument(
        "--rerun_failures_with_trace",
        action="store_true",
        help="After bulk run, rerun only failures_only.csv cases with tracing ON into traced_failures/.",
    )
    args = parser.parse_args()

    bench_paths = [b.strip() for b in (args.bench or []) if b and b.strip()]
    if not bench_paths:
        bench_paths = [DEFAULT_BENCHMARK]

    CANONICALS = {
        "BRCA1": "ENST00000357654",
        "VEGFA": "ENST00000372055",
    }

    llm = None
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
                judge_llm = base_llm.bind(response_format={"type": "json_object"})
                llm = judge_llm
                print("✅ LLM initialized for JUDGE.")
            else:
                print("⚠️  No OPENAI_API_KEY found; JUDGE will be disabled.")
        except Exception as e:
            print(f"⚠️  LLM init failed ({type(e).__name__}: {e}); JUDGE disabled.")
            llm = None
    else:
        print("⚠️  langchain_openai/dotenv not available; JUDGE disabled.")

    # ---------------------------------------------------------
    # TSV Loaders
    # ---------------------------------------------------------

    def _load_df_cached(path: str, _cache: Dict[str, pd.DataFrame] = {}) -> pd.DataFrame:
        ap = os.path.abspath(os.path.normpath(path))
        if ap not in _cache:
            _cache[ap] = pd.read_csv(ap, sep="\t", dtype=str, low_memory=False).fillna("")
        return _cache[ap]

    def _parse_semicolon_ints(x: str) -> List[int]:
        if not x:
            return []
        out = []
        for v in str(x).split(";"):
            v = v.strip()
            if v and v.lower() != "na":
                out.append(int(v))
        return out

    def exon_loader(dataset_path: str, transcript_id: str) -> Dict[str, Any]:
        df = _load_df_cached(dataset_path)
        sub = df[df["Transcript stable ID"] == transcript_id]
        if sub.empty:
            raise ValueError(f"Transcript stable ID not found: {transcript_id}")

        row = sub.iloc[0]
        starts = _parse_semicolon_ints(row.get("Exon region start (bp)", ""))
        ends = _parse_semicolon_ints(row.get("Exon region end (bp)", ""))
        ranks = _parse_semicolon_ints(row.get("Exon rank in transcript", ""))

        exons = sorted(zip(ranks, starts, ends), key=lambda t: t[0])

        exon_table = []
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

        strand_raw = row.get("Strand", "")
        strand = int(strand_raw) if str(strand_raw).strip() else 1

        return {"exon_table": exon_table, "tx_len": tx_cursor - 1, "strand": strand}

    def cds_segment_loader(dataset_path: str, transcript_id: str) -> List[Tuple[int, int]]:
        df = _load_df_cached(dataset_path)
        sub = df[df["Transcript stable ID"] == transcript_id]
        if sub.empty:
            raise ValueError(f"Transcript stable ID not found: {transcript_id}")
        row = sub.iloc[0]
        starts = _parse_semicolon_ints(row.get("cDNA coding start", ""))
        ends = _parse_semicolon_ints(row.get("cDNA coding end", ""))
        return sorted([(int(a), int(b)) for a, b in zip(starts, ends) if int(a) > 0 and int(b) > 0])

    def seq_loader(dataset_path: str, transcript_id: str) -> str:
        df = _load_df_cached(dataset_path)
        sub = df[df["Transcript stable ID"] == transcript_id]
        if sub.empty:
            raise ValueError(f"Transcript stable ID not found: {transcript_id}")
        seq = (sub.iloc[0].get("cDNA sequences", "") or "").upper()
        return "".join([c for c in seq if c in "ACGTN"])

    cds = cds_tool(exon_loader=exon_loader, cds_segment_loader=cds_segment_loader, seq_loader=seq_loader)
    nmd = nmd_tool(ejc_threshold_nt=55, require_ptc_for_nmd=True, margin_nt=55)
    motif = motif_tool()
    tavily = tavily_tool()

    graph = build_graph(cds, nmd, motif, tavily, llm=llm)

    # ==========================================================
    # LOOP DATASETS
    # ==========================================================

    for bench_path in bench_paths:
        _assert_file_exists(bench_path, "benchmark.tsv")

        base = os.path.splitext(os.path.basename(bench_path))[0]
        safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)[:60]
        run_dir = make_test_run_dir(prefix=f"test_run_{safe_base}")

        df = pd.read_csv(bench_path, sep="\t", dtype=str).fillna("")

        jsonl_path = os.path.join(run_dir, "test_case_runs.jsonl")
        runs_json_path = os.path.join(run_dir, "test_case_runs.json")
        summary_path = os.path.join(run_dir, "test_case_summary.csv")
        failures_only_path = os.path.join(run_dir, "failures_only.csv")

        summary_rows = []
        all_runs = []

        def _invoke_graph_with_trace(state: SplicingAgentState, trace_enabled: bool):
            config = {"configurable": {"thread_id": f"{safe_base}::{state.get('row_id','')}"}} 
            with tracing_context(enabled=bool(trace_enabled)):
                return dict(graph.invoke(state, config=config))

        with open(jsonl_path, "w", encoding="utf-8") as fjsonl:
            for i, (_, row) in enumerate(df.iterrows()):
                transcript_id = (row.get("transcript_id", "") or "").strip()
                gene = (row.get("gene_symbol_hint", "") or "").strip()
                row_id = make_row_id(i, transcript_id=transcript_id, gene=gene)

                state: SplicingAgentState = {
                    "row_id": row_id,
                    "run_dir": run_dir,
                    "dataset_path": bench_path,
                    "canonical_dataset_path": bench_path,
                    "transcript_id": transcript_id,
                    "gene_symbol_hint": gene,
                    "plan_done": [],
                    "tool_calls": [],
                    "failure_modes": [],
                    "failure_notes": {},
                    "trace": [],
                    "errors": [],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }

                t0 = time.perf_counter()
                out = _invoke_graph_with_trace(state, trace_enabled=bool(args.trace_bulk))
                out["latency_s"] = time.perf_counter() - t0
                out["tool_calls_count"] = len(out.get("tool_calls", []) or [])

                write_case_artifacts(
                    run_dir,
                    row_id=row_id,
                    record=dict(out),
                    report_text=out.get("report_text", ""),
                    consequence_md=out.get("report_consequence_md", ""),
                    metrics_md=out.get("report_metrics_md", ""),
                )

                fjsonl.write(json.dumps(dict(out), ensure_ascii=False, default=str) + "\n")
                all_runs.append(dict(out))

                summary_rows.append({
                    "row_id": row_id,
                    "predicted_label": out.get("predicted_label", ""),
                    "expected_label": out.get("expected_label", ""),
                    "failure_modes": ",".join(out.get("failure_modes", []) or []),
                    "has_critical_failure": out.get("has_critical_failure", False),
                    "task_completed": out.get("task_completed", False),
                    "tool_usage_accuracy": out.get("tool_usage_accuracy", False),
                    "success": out.get("success", False),
                    "error_rate_flag": out.get("error_rate_flag", False),
                    "hallucination": out.get("hallucination", False),
                    "latency_s": out.get("latency_s", None),
                    "status": "error" if (out.get("errors") or []) else "ok",
                })

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

        # FIXED failures-only logic
        try:
            df_sum = pd.DataFrame(summary_rows)

            fail_mask = (
                (df_sum["error_rate_flag"].fillna(False).astype(bool))
                | (df_sum["hallucination"].fillna(False).astype(bool))
                | (df_sum["has_critical_failure"].fillna(False).astype(bool))
                | (df_sum["status"] == "error")
            )

            df_sum.loc[fail_mask].to_csv(failures_only_path, index=False)
        except Exception as e:
            print(f"⚠️  Could not write failures_only.csv: {e}")

        with open(runs_json_path, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2, ensure_ascii=False, default=str)

        print("✅ Dataset run complete")

    print("\n✅ Complete")