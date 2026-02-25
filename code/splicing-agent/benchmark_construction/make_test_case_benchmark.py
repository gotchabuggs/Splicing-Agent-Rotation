from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, List

# -----------------------------
# Output (new mini benchmark)
# -----------------------------
OUT_CSV = Path(r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\test_case_benchmark_cases.csv")

# -----------------------------
# Paths
# -----------------------------
MERGED_TSV: Optional[Path] = None

BRCA1_TSV = Path(
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\code\queries\BRCA1_benchmark.tsv"
)
VEGFA_TSV = Path(
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\code\queries\VEGFA_benchmark.tsv"
)

# -----------------------------
# Canonical baseline (only needed for BRCA1 comparisons)
# -----------------------------
BRCA1_CANONICAL_CASE_ID = "BRCA1-203"
BRCA1_CANONICAL_TRANSCRIPT_ID = "ENST00000357654"
BRCA1_CANONICAL_DATASET_PATH = Path(
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-203.txt"
)

# -----------------------------
# The ONLY transcripts in your mini benchmark
# -----------------------------
EXPECTED_TRANSCRIPTS: List[str] = [
    "ENST00000461798",  # BRCA1-206
    "ENST00000591849",  # BRCA1-225
    "ENST00000518824",  # VEGFA-219 (forward strand)
]

CASE_ID_BY_TX: Dict[str, str] = {
    "ENST00000461798": "BRCA1-206",
    "ENST00000591849": "BRCA1-225",
    "ENST00000518824": "VEGFA-219",
}

EXPECTED_LABEL_BY_CASE: Dict[str, str] = {
    "BRCA1-206": "nmd",
    "BRCA1-225": "protein_coding",
    "VEGFA-219": "protein_coding",
}

# -----------------------------
# Helpers
# -----------------------------
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

def load_df_for_tx(tx: str) -> pd.DataFrame:
    """
    Loads the right TSV depending on whether you're using one merged file
    or per-gene files.
    """
    if MERGED_TSV is not None:
        if not MERGED_TSV.exists():
            raise FileNotFoundError(f"MERGED_TSV not found: {MERGED_TSV}")
        return pd.read_csv(MERGED_TSV, sep="\t", dtype=str, low_memory=False)

    # per-gene TSV routing (based on case_id prefix)
    case_id = CASE_ID_BY_TX[tx]
    if case_id.startswith("BRCA1"):
        if not BRCA1_TSV.exists():
            raise FileNotFoundError(f"BRCA1_TSV not found: {BRCA1_TSV}")
        return pd.read_csv(BRCA1_TSV, sep="\t", dtype=str, low_memory=False)

    if case_id.startswith("VEGFA"):
        if not VEGFA_TSV.exists():
            raise FileNotFoundError(f"VEGFA_TSV not found: {VEGFA_TSV}")
        return pd.read_csv(VEGFA_TSV, sep="\t", dtype=str, low_memory=False)

    raise ValueError(f"Don't know which TSV to use for {case_id} / {tx}")

def canonical_for_case(case_id: str) -> Dict[str, str]:
    """
    Only BRCA1 cases use BRCA1-203 as canonical baseline.
    VEGFA row will not include canonical baseline (empty strings).
    """
    if case_id.startswith("BRCA1"):
        canonical_path = BRCA1_CANONICAL_DATASET_PATH if BRCA1_CANONICAL_DATASET_PATH.exists() else (
            str(BRCA1_TSV if MERGED_TSV is None else MERGED_TSV)
        )
        return {
            "canonical_dataset_path": str(canonical_path),
            "canonical_transcript_id": BRCA1_CANONICAL_TRANSCRIPT_ID,
        }

    # VEGFA: leave blank (your agent can treat missing canonical as “no PTC baseline”)
    return {"canonical_dataset_path": "", "canonical_transcript_id": ""}

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    rows: List[Dict[str, Any]] = []

    for tx in EXPECTED_TRANSCRIPTS:
        df = load_df_for_tx(tx)

        tx_col = pick_col(df, ["Transcript stable ID"])
        gene_symbol_col = pick_col(df, ["Gene name"])
        chrom_col = pick_col(df, ["Chromosome/scaffold name"])
        gene_start_col = pick_col(df, ["Gene start (bp)"])
        gene_end_col = pick_col(df, ["Gene end (bp)"])
        gene_type_col = pick_col(df, ["Gene type"])

        exon_id_col = "Exon stable ID" if "Exon stable ID" in df.columns else None
        cds_len_col = "CDS Length" if "CDS Length" in df.columns else None

        sub = df[df[tx_col].astype(str) == tx].copy()
        if sub.empty:
            raise ValueError(f"Transcript {tx} not found in selected TSV.")

        case_id = CASE_ID_BY_TX[tx]
        expected_label = EXPECTED_LABEL_BY_CASE.get(case_id, "unknown")

        gene_symbol = safe_first(sub[gene_symbol_col]) or ""
        chrom = safe_first(sub[chrom_col])
        gene_start = safe_first(sub[gene_start_col])
        gene_end = safe_first(sub[gene_end_col])
        biotype = safe_first(sub[gene_type_col])

        cds_len = safe_first(sub[cds_len_col]) if cds_len_col else None

        if exon_id_col:
            exon_ids_raw = safe_first(sub[exon_id_col])
            exon_count = len(split_semicol(exon_ids_raw))
        else:
            exon_count = int(len(sub))

        # pick dataset_path string for this tx
        dataset_path = str(MERGED_TSV) if MERGED_TSV is not None else (
            str(BRCA1_TSV) if case_id.startswith("BRCA1") else str(VEGFA_TSV)
        )

        raw_input = (
            f"Assess exon structure and NMD-relevant consequences for {gene_symbol} transcript {tx}. "
            f"Use TSV features; if canonical baseline is provided, compare against it."
        )

        canon = canonical_for_case(case_id)

        rows.append(
            {
                "case_id": case_id,
                "dataset_path": dataset_path,
                "transcript_id": tx,
                "gene_symbol": gene_symbol,
                "raw_input": raw_input,
                "canonical_dataset_path": canon["canonical_dataset_path"],
                "canonical_transcript_id": canon["canonical_transcript_id"],
                "expected_label": expected_label,
                "biotype": biotype,
                "chrom": chrom,
                "gene_start": gene_start,
                "gene_end": gene_end,
                "cds_length": cds_len,
                "exon_count": exon_count,
            }
        )

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"✅ Wrote: {OUT_CSV}")
    print(out_df[["case_id", "transcript_id", "gene_symbol", "expected_label", "exon_count", "cds_length"]].to_string(index=False))

if __name__ == "__main__":
    main()