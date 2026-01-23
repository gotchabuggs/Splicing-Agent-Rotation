from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, List

# ============================================================
# INPUT: single merged TSV / OUTPUT: benchmark_cases.csv
# ============================================================

DATASET_PATH = r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\code\queries\BRCA1_benchmark_10.tsv"
OUT_CSV = r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\benchmark_cases.csv"

# ============================================================
# Benchmark constants
# ============================================================
CANONICAL_TRANSCRIPT_ID = "ENST00000357654"
CANONICAL_DATASET_PATH = DATASET_PATH

EXPECTED_TRANSCRIPTS: List[str] = [
    "ENST00000357654",
    "ENST00000494123",
    "ENST00000470026",
    "ENST00000621897",
    "ENST00000354071",
    "ENST00000461798",
    "ENST00000700081",
    "ENST00000492859",
    "ENST00000461221",
    "ENST00000700183",
]

CASE_ID_BY_TX: Dict[str, str] = {
    "ENST00000357654": "BRCA1-203",
    "ENST00000494123": "BRCA1-221",
    "ENST00000470026": "BRCA1-208",
    "ENST00000621897": "BRCA1-227",
    "ENST00000354071": "BRCA1-202",
    "ENST00000461798": "BRCA1-206",
    "ENST00000700081": "BRCA1-233",
    "ENST00000492859": "BRCA1-218",
    "ENST00000461221": "BRCA1-204",
    "ENST00000700183": "BRCA1-237",
}

EXPECTED_LABEL_BY_CASE: Dict[str, str] = {
    "BRCA1-203": "protein_coding",
    "BRCA1-221": "protein_coding",
    "BRCA1-208": "protein_coding",
    "BRCA1-227": "cds_not_defined",
    "BRCA1-202": "retained_intron",
    "BRCA1-206": "nmd",
    "BRCA1-233": "retained_intron",
    "BRCA1-218": "nmd",
    "BRCA1-204": "nmd",
    "BRCA1-237": "nmd",
}

# ============================================================
# Helper Functions
# ============================================================

def pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing expected columns. Tried: {candidates}\nFound: {list(df.columns)}")

def safe_first(series: pd.Series) -> Optional[str]:
    s = series.dropna()
    if len(s) == 0:
        return None
    v = s.iloc[0]
    if pd.isna(v):
        return None
    return str(v)

def split_semicol(x: Optional[str]) -> List[str]:
    if not x:
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

def ensure_expected_present(df: pd.DataFrame, tx_col: str) -> None:
    present = set(df[tx_col].dropna().astype(str).unique().tolist())
    missing = sorted(list(set(EXPECTED_TRANSCRIPTS) - present))
    if missing:
        raise ValueError(f"Missing expected transcript IDs in TSV: {missing}")

def main() -> None:
    tsv_path = Path(DATASET_PATH)
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t", dtype=str, low_memory=False)

    tx_col = pick_col(df, ["Transcript stable ID"])
    gene_symbol_col = pick_col(df, ["Gene name"])
    chrom_col = pick_col(df, ["Chromosome/scaffold name"])
    gene_start_col = pick_col(df, ["Gene start (bp)"])
    gene_end_col = pick_col(df, ["Gene end (bp)"])
    gene_type_col = pick_col(df, ["Gene type"])

    # Optional list-like fields
    exon_id_col = "Exon stable ID" if "Exon stable ID" in df.columns else None
    cdna_col = "cDNA sequences" if "cDNA sequences" in df.columns else None
    cds_len_col = "CDS Length" if "CDS Length" in df.columns else None

    ensure_expected_present(df, tx_col)

    df_locked = df[df[tx_col].astype(str).isin(EXPECTED_TRANSCRIPTS)].copy()

    rows: List[Dict[str, Any]] = []
    for tx in EXPECTED_TRANSCRIPTS:
        sub = df_locked[df_locked[tx_col].astype(str) == tx].copy()
        if sub.empty:
            raise ValueError(f"Transcript {tx} expected but has 0 rows after filtering.")

        case_id = CASE_ID_BY_TX[tx]
        expected_label = EXPECTED_LABEL_BY_CASE.get(case_id, "unknown")

        gene_symbol = safe_first(sub[gene_symbol_col]) or "BRCA1"
        chrom = safe_first(sub[chrom_col])
        biotype = safe_first(sub[gene_type_col])
        gene_start = safe_first(sub[gene_start_col])
        gene_end = safe_first(sub[gene_end_col])

        # exon_count: if exon IDs are semicolon-delimited in a single row, split them
        exon_count = None
        if exon_id_col:
            exon_ids_raw = safe_first(sub[exon_id_col])
            exon_count = len(split_semicol(exon_ids_raw))
        else:
            # fallback: number of rows for that transcript
            exon_count = int(len(sub))

        cds_len = safe_first(sub[cds_len_col]) if cds_len_col else None

        # raw_input: DO NOT dump entire cDNA sequence into CSV unless you really want it.
        # Keep it short + deterministic.
        raw_input = f"Assess aberrant splicing / exon structure and NMD for BRCA1 transcript {tx} using the merged TSV + canonical baseline."

        row = {
            "case_id": case_id,
            "dataset_path": DATASET_PATH,
            "transcript_id": tx,
            "gene_symbol": gene_symbol,
            "raw_input": raw_input,
            "canonical_dataset_path": CANONICAL_DATASET_PATH,
            "canonical_transcript_id": CANONICAL_TRANSCRIPT_ID,
            "expected_label": expected_label,

            # helpful metadata
            "biotype": biotype,
            "chrom": chrom,
            "gene_start": gene_start,
            "gene_end": gene_end,
            "cds_length": cds_len,
            "exon_count": exon_count,
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)

    # canonical first, then case_id
    out_df["__is_canonical"] = out_df["transcript_id"].eq(CANONICAL_TRANSCRIPT_ID)
    out_df = out_df.sort_values(["__is_canonical", "case_id"], ascending=[False, True]).drop(columns="__is_canonical")

    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"✅ Wrote: {out_path}")
    print(out_df[["case_id", "transcript_id", "dataset_path", "expected_label", "exon_count"]].to_string(index=False))

if __name__ == "__main__":
    main()