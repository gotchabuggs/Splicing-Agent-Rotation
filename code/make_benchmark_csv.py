from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, List

# ============================================================
# BRCA1 Benchmark
# ------------------------------------------------------------
# These Ensembl transcript IDs are a small BRCA1 validation set.
# Source:
#   Curated from Ensembl/BioMart BRCA1 transcript tables + manual review
#   (selected to span protein_coding, NMD, retained_intron, cds_not_defined).
#
# Why this exists:
#   Keeps benchmark definition centralized so we don't have to list
#   transcript IDs / case IDs / labels in multiple places.
# ============================================================

# -----------------------------
# Paths
# -----------------------------
DATASET_PATH = Path(r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\code\queries\BRCA1_benchmark.tsv")
OUT_CSV = Path(r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\benchmark_cases.csv")
CANONICAL_DATASET_PATH = Path(r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-203.txt")


# -----------------------------
# Canonical baseline for BRCA1
# -----------------------------
CANONICAL_CASE_ID = "BRCA1-203"
CANONICAL_TRANSCRIPT_ID = "ENST00000357654"

BRCA1_CASES: Dict[str, Dict[str, str]] = {
    "BRCA1-203": {"transcript_id": "ENST00000357654", "expected_label": "protein_coding"},
    "BRCA1-202": {"transcript_id": "ENST00000354071", "expected_label": "retained_intron"},
    "BRCA1-204": {"transcript_id": "ENST00000461221", "expected_label": "nmd"},
    "BRCA1-206": {"transcript_id": "ENST00000461798", "expected_label": "nmd"},
    "BRCA1-208": {"transcript_id": "ENST00000470026", "expected_label": "protein_coding"},
    "BRCA1-218": {"transcript_id": "ENST00000492859", "expected_label": "nmd"},
    "BRCA1-221": {"transcript_id": "ENST00000494123", "expected_label": "protein_coding"},
    "BRCA1-227": {"transcript_id": "ENST00000621897", "expected_label": "cds_not_defined"},
    "BRCA1-225": {"transcript_id": "ENST00000591849", "expected_label": "protein_coding"},  # 5-exon protein
    "BRCA1-237": {"transcript_id": "ENST00000700183", "expected_label": "nmd"},
}

# Mappings 
EXPECTED_TRANSCRIPTS: List[str] = [v["transcript_id"] for v in BRCA1_CASES.values()]
CASE_ID_BY_TX: Dict[str, str] = {v["transcript_id"]: case_id for case_id, v in BRCA1_CASES.items()}
EXPECTED_LABEL_BY_CASE: Dict[str, str] = {case_id: v["expected_label"] for case_id, v in BRCA1_CASES.items()}

# ============================================================
# Helper Functions
# ============================================================

def pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first matching column name from candidates; raise if none exist."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing expected columns. Tried: {candidates}\nFound: {list(df.columns)}")

def safe_first(series: pd.Series) -> Optional[str]:
    """Return the first non-null value in a Series as a string, else None."""
    s = series.dropna()
    if len(s) == 0:
        return None
    v = s.iloc[0]
    if pd.isna(v):
        return None
    return str(v)

def split_semicol(x: Optional[str]) -> List[str]:
    """Split a semicolon-delimited BioMart field into a list of tokens."""
    if not x:
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

def ensure_expected_present(df: pd.DataFrame, tx_col: str) -> None:
    """Verify all curated benchmark transcript IDs exist in the input TSV."""
    present = set(df[tx_col].dropna().astype(str).unique().tolist())
    missing = sorted(list(set(EXPECTED_TRANSCRIPTS) - present))
    if missing:
        raise ValueError(f"Missing expected transcript IDs in TSV: {missing}")

# ============================================================
# Main
# ============================================================
def main() -> None:
    print("🚀 Building benchmark_cases.csv")
    print(f"   Input TSV: {DATASET_PATH}")
    print(f"   Output CSV: {OUT_CSV}")
    print(f"   Canonical transcript: {CANONICAL_TRANSCRIPT_ID} ({CANONICAL_CASE_ID})")
    print(f"   Expected transcripts: {len(EXPECTED_TRANSCRIPTS)}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"benchmark TSV not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, sep="\t", dtype=str, low_memory=False)
    print(f"✅ Loaded TSV: {len(df):,} rows | {len(df.columns)} columns")
    
    # BioMart column names (match your TSV header)
    tx_col = pick_col(df, ["Transcript stable ID"])
    gene_symbol_col = pick_col(df, ["Gene name"])
    chrom_col = pick_col(df, ["Chromosome/scaffold name"])
    gene_start_col = pick_col(df, ["Gene start (bp)"])
    gene_end_col = pick_col(df, ["Gene end (bp)"])
    gene_type_col = pick_col(df, ["Gene type"])

    exon_id_col = "Exon stable ID" if "Exon stable ID" in df.columns else None
    cds_len_col = "CDS Length" if "CDS Length" in df.columns else None

    ensure_expected_present(df, tx_col)
    print("✅ All expected transcripts are present in the TSV.")

    # canonical path fallback
    canonical_path = CANONICAL_DATASET_PATH if CANONICAL_DATASET_PATH.exists() else DATASET_PATH
    if canonical_path == DATASET_PATH:
        print("ℹ️ Canonical dataset file not found; using merged TSV as canonical_dataset_path fallback.")

    # Filter to only the 10 transcripts (deterministic ordering = EXPECTED_TRANSCRIPTS)
    df_locked = df[df[tx_col].astype(str).isin(EXPECTED_TRANSCRIPTS)].copy()
    print(f"✅ Filtered to benchmark rows: {len(df_locked):,}")

    rows: List[Dict[str, Any]] = []
    for i, tx in enumerate(EXPECTED_TRANSCRIPTS, start=1):
        print(f"   [{i}/{len(EXPECTED_TRANSCRIPTS)}] Processing {tx}")

        if tx not in CASE_ID_BY_TX:
            raise KeyError(f"Transcript {tx} not found in CASE_ID_BY_TX mapping. Check BRCA1_CASES registry.")

        sub = df_locked[df_locked[tx_col].astype(str) == tx].copy()
        if sub.empty:
            raise ValueError(f"Transcript {tx} expected but has 0 rows after filtering.")

        case_id = CASE_ID_BY_TX[tx]
        expected_label = EXPECTED_LABEL_BY_CASE.get(case_id, "unknown")

        gene_symbol = safe_first(sub[gene_symbol_col]) or "BRCA1"
        chrom = safe_first(sub[chrom_col])
        gene_start = safe_first(sub[gene_start_col])
        gene_end = safe_first(sub[gene_end_col])
        biotype = safe_first(sub[gene_type_col])

        cds_len = safe_first(sub[cds_len_col]) if cds_len_col else None

        # exon_count:
        # - If BioMart provides an exon stable ID field, count exon IDs from semicolon list.
        # - Otherwise, fall back to the number of rows for that transcript (approx. one row per exon in this export).
        if exon_id_col:
            exon_ids_raw = safe_first(sub[exon_id_col])
            exon_count = len(split_semicol(exon_ids_raw))
        else:
            exon_count = int(len(sub))

        raw_input = (
            f"Assess aberrant splicing / exon structure and NMD for BRCA1 transcript {tx} "
            f"using the merged BioMart TSV + canonical baseline."
        )

        rows.append(
            {
                "case_id": case_id,
                "dataset_path": str(DATASET_PATH),
                "transcript_id": tx,
                "gene_symbol": gene_symbol,
                "raw_input": raw_input,
                "canonical_dataset_path": str(canonical_path),
                "canonical_transcript_id": CANONICAL_TRANSCRIPT_ID,
                "expected_label": expected_label,
                # minimal helpful metadata
                "biotype": biotype,
                "chrom": chrom,
                "gene_start": gene_start,
                "gene_end": gene_end,
                "cds_length": cds_len,
                "exon_count": exon_count,
            }
        )

    out_df = pd.DataFrame(rows)

    # canonical first, then case_id
    out_df["__is_canonical"] = out_df["transcript_id"].eq(CANONICAL_TRANSCRIPT_ID)
    out_df = out_df.sort_values(["__is_canonical", "case_id"], ascending=[False, True]).drop(columns="__is_canonical")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"✅ Wrote: {OUT_CSV}")
    print(out_df[["case_id", "transcript_id", "expected_label", "exon_count", "cds_length"]].to_string(index=False))


if __name__ == "__main__":
    main()