import json
import os
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List, Optional
from splicing_agent.helpers import _df_to_md_table


# =============================================================
# Reporting Helpers
# =============================================================

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

# =============================================================
# Report generation + Benchmark Summary
# =============================================================

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
    "expected_label": "Expected label",
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