import os
import re
import pandas as pd

FILE_PATHS = [
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-202.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-203.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-204.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-206 (2).txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-210.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-226.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-227.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\BRCA1-230.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\VEGFA-001.txt",
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\VEGFA-004.txt",
]

OUT_CSV = r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\benchmark_cases.csv"


def infer_gene_from_filename(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"([A-Za-z0-9]+)-\d+", base)
    return (m.group(1).upper() if m else "UNKNOWN")


def infer_case_id(path: str) -> str:
    base = os.path.basename(path)
    base = os.path.splitext(base)[0]
    base = re.sub(r"\s*\(\d+\)$", "", base)  # drop trailing " (2)" etc
    return base.replace(" ", "_")


def infer_transcript_id_from_tsv(path: str) -> str:
    """
    Heuristic: read first ~200 lines and look for an ENST... token in the transcript column.
    Works for Biomart TSV exports.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                m = re.search(r"(ENST\d{11})", line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return ""


rows = []
for p in FILE_PATHS:
    case_id = infer_case_id(p)
    gene = infer_gene_from_filename(p)
    tx = infer_transcript_id_from_tsv(p)

    raw_input = (
        f"Assess aberrant splicing / exon structure and NMD for {gene}. "
        f"Use transcript {tx} if available; otherwise infer from dataset."
    )

    rows.append(
        {
            "case_id": case_id,
            "dataset_path": p,
            "transcript_id": tx,        # may be blank; your runner can handle it
            "gene_symbol": gene,
            "raw_input": raw_input,
        }
    )

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"✅ Wrote benchmark CSV:\n{OUT_CSV}\n\nPreview:\n{df}")