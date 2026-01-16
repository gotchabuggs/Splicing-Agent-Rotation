import os
import re
import pandas as pd
from typing import Dict, Any, Optional

FILE_PATHS = [
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-202.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-203.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-204.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-206 (2).txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-210.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-226.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-227.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-230.txt",
    # VEGFA omitted until files confirmed non-empty
]

OUT_CSV = r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\benchmark_cases.csv"

# ---- Canonical baseline for BRCA1 ----
CANONICAL_CASE_ID = "BRCA1-203"
CANONICAL_DATASET_PATH = r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-203.txt"
CANONICAL_TRANSCRIPT_ID = "ENST00000357654"

# ---- Your curated expected labels ----
EXPECTED_LABELS = {
    "BRCA1-202": "retained_intron",
    "BRCA1-203": "protein_coding",
    "BRCA1-204": "nmd",
    "BRCA1-206": "nmd",
    "BRCA1-210": "retained_intron",
    "BRCA1-226": "protein_coding",
    "BRCA1-227": "cds_not_defined",
    "BRCA1-230": "cds_not_defined",
}

def infer_gene_from_filename(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"([A-Za-z0-9]+)-\d+", base)
    return (m.group(1).upper() if m else "UNKNOWN")

def infer_case_id(path: str) -> str:
    """
    Use filename stem, drop trailing " (2)", and normalize known quirks.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    stem = re.sub(r"\s*\(\d+\)$", "", stem).strip()

    # Normalize "BRCA-227" -> "BRCA1-227" if it ever occurs
    stem = re.sub(r"^BRCA-(\d+)$", r"BRCA1-\1", stem, flags=re.IGNORECASE)

    # Keep exact "BRCA1-206" format (no underscores)
    return stem

def read_first_header_line(path: str) -> str:
    """
    Return the first FASTA-like header line (starts with '>') if present, else "".
    Handles files where header may be preceded by whitespace.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                return s
            # If the file is weird like "BRCA1-206:>ENSG..." on one line (no space),
            # still catch the ">" portion
            if ">" in s:
                idx = s.find(">")
                return s[idx:].strip()
            # stop early if it doesn't look like FASTA at all
            break
    return ""

def parse_benchmark_header(header: str) -> Dict[str, Any]:
    """
    Parse your pipe-delimited header into structured columns.

    Example header (yours):
    >ENSG...|ENST...|BRCA1|17|43044292|43170245|protein_coding|...

    We only extract the fields that are stable/consistent.
    """
    out: Dict[str, Any] = {
        "header": header,
        "ensembl_gene_id": "",
        "ensembl_transcript_id": "",
        "header_gene_symbol": "",
        "chrom": "",
        "gene_start": "",
        "gene_end": "",
        "transcript_biotype": "",
        "cds_start": "",
        "cds_end": "",
        "exon_count": "",
    }

    if not header or not header.startswith(">"):
        return out

    # split by '|'
    parts = header[1:].split("|")  # drop leading '>'
    # Guard: some headers can be shorter; only fill what exists.
    if len(parts) >= 1:
        out["ensembl_gene_id"] = parts[0].strip()
    if len(parts) >= 2:
        out["ensembl_transcript_id"] = parts[1].strip()
    if len(parts) >= 3:
        out["header_gene_symbol"] = parts[2].strip()
    if len(parts) >= 4:
        out["chrom"] = parts[3].strip()
    if len(parts) >= 5:
        out["gene_start"] = parts[4].strip()
    if len(parts) >= 6:
        out["gene_end"] = parts[5].strip()
    if len(parts) >= 7:
        out["transcript_biotype"] = parts[6].strip()

    # In your headers, these *often* appear next as CDS start/end,
    # but some cases can have different counts/extra fields.
    # We'll attempt to parse parts[7] and parts[8] as ints if they look numeric.
    def looks_int(x: str) -> bool:
        return bool(re.fullmatch(r"-?\d+", x.strip()))

    if len(parts) >= 8 and looks_int(parts[7]):
        out["cds_start"] = parts[7].strip()
    if len(parts) >= 9 and looks_int(parts[8]):
        out["cds_end"] = parts[8].strip()

    # Exon count heuristic:
    # Many of your headers contain exon start positions as a ';' separated list.
    # We scan all remaining fields for a plausible exon-start list (lots of ';' and ints),
    # and take the max length list we find.
    best_n = 0
    for field in parts[9:]:
        s = field.strip()
        if ";" in s and all(re.fullmatch(r"-?\d+", tok.strip()) for tok in s.split(";") if tok.strip()):
            n = len([tok for tok in s.split(";") if tok.strip()])
            best_n = max(best_n, n)
    out["exon_count"] = str(best_n) if best_n > 0 else ""

    return out

def infer_transcript_id(path: str) -> str:
    """
    Prefer transcript ID from the header (ENST...), otherwise fall back to scanning.
    """
    header = read_first_header_line(path)
    if header:
        m = re.search(r"(ENST\d{11})", header)
        if m:
            return m.group(1)

    # fallback: scan first ~200 non-empty lines
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        seen = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            seen += 1
            m = re.search(r"(ENST\d{11})", s)
            if m:
                return m.group(1)
            if seen >= 200:
                break
    return ""

rows = []
for p in FILE_PATHS:
    case_id = infer_case_id(p)
    gene_from_filename = infer_gene_from_filename(p)

    header = read_first_header_line(p)
    header_info = parse_benchmark_header(header)

    tx = infer_transcript_id(p)
    if not tx:
        raise ValueError(f"Could not infer transcript_id from file: {p}")

    # If filename gene is ambiguous/odd, prefer the header gene symbol when present
    gene = (header_info.get("header_gene_symbol") or gene_from_filename or "UNKNOWN").upper()

    # Use canonical baseline for all BRCA1 rows (including BRCA1-203 itself)
    canonical_dataset_path = CANONICAL_DATASET_PATH if gene == "BRCA1" else ""
    canonical_transcript_id = CANONICAL_TRANSCRIPT_ID if gene == "BRCA1" else ""

    expected_label = EXPECTED_LABELS.get(case_id, "")

    raw_input = (
        f"Assess aberrant splicing / exon structure and PTC/NMD for {gene}. "
        f"Use transcript {tx}. "
        f"Use canonical baseline {CANONICAL_TRANSCRIPT_ID} when provided."
    )

    # ✅ New columns generated from your file header:
    # - ensembl_gene_id
    # - transcript_biotype
    # - chrom / gene_start / gene_end
    # - cds_start / cds_end
    # - exon_count
    rows.append(
        {
            "case_id": case_id,
            "dataset_path": p,
            "transcript_id": tx,
            "gene_symbol": gene,
            "raw_input": raw_input,

            # v4 baseline columns:
            "canonical_dataset_path": canonical_dataset_path,
            "canonical_transcript_id": canonical_transcript_id,

            # curated truth:
            "expected_label": expected_label,

            # ---- NEW: parsed-from-header columns ----
            "fasta_header": header_info.get("header", ""),
            "ensembl_gene_id": header_info.get("ensembl_gene_id", ""),
            "ensembl_transcript_id_header": header_info.get("ensembl_transcript_id", ""),
            "transcript_biotype": header_info.get("transcript_biotype", ""),
            "chrom": header_info.get("chrom", ""),
            "gene_start": header_info.get("gene_start", ""),
            "gene_end": header_info.get("gene_end", ""),
            "cds_start": header_info.get("cds_start", ""),
            "cds_end": header_info.get("cds_end", ""),
            "exon_count": header_info.get("exon_count", ""),
        }
    )

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(f"✅ Wrote benchmark CSV:\n{OUT_CSV}\n")
print("Preview:\n", df)