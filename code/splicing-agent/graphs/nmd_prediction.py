"""
single_nmd_prediction_comparison.py

Compare NMD prediction distributions across:
    • Ground Truth benchmark
    • BIOMNI (v1)
    • BIOMNI Lab (v2)
    • Splicing-Agent

Purpose
-------
This script generates a stacked bar chart showing how each model
distributes transcript predictions across canonical outcome buckets.

This is NOT a per-transcript accuracy plot.
It visualizes prediction behavior differences.

Key Question
------------
Do different systems classify transcripts into similar biological
outcome categories, or do they systematically overuse certain buckets
(e.g., cds_not_defined)?

Output
------
Saves:
    figure1_nmd_prediction_comparison.png
to the graphs directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# OUTPUT DIRECTORY
# ============================================================
# Directory to save generated figures; will be created if it doesn't exist
# Note: This is a local path specific to the user's environment and should be updated accordingly if run elsewhere.

OUTPUT_DIR = r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# INPUT FILE PATHS
# ============================================================
"""
Input file paths for the Splicing-Agent predictions, BIOMNI Lab v2 predictions, and BIOMNI v1 predictions. 
These paths point to local TSV files containing the respective model outputs, which will be loaded and processed 
to compare NMD prediction distributions across the different systems. Ensure that these paths are correct and that 
the TSV files are properly formatted for successful execution of the script.

Note: The Splicing-Agent local path points to a CSV file containing the predicted labels, while the BIOMNI paths point 
to TSV files with predicted label buckets. The script will handle loading and processing these files accordingly.
"""
SPLICING_AGENT_LOCAL = (
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab"
    r"\Splicing-Agent-Rotation\data\test_runs"
    r"\test_run_test_case_benchmark_20260206_144457"
    r"\test_case_summary.csv"
)

BIOMNI_V2_TSV = (
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab"
    r"\Splicing-Agent-Rotation\apps\biomni\biomni_lab"
    r"\run_3_020426\strict_query"
    r"\nmd_predictions_test_case_benchmark.tsv"
)

BIOMNI_V1_TSV = (
    r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab"
    r"\Splicing-Agent-Rotation\apps\biomni\biomni_v1"
    r"\run_2_02082026\strict_query"
    r"\nmd_predictions_test_case_benchmark.tsv"
)

# ============================================================
# BUCKET DEFINITIONS (CANONICAL)
# ============================================================
"""
All predicted labels will be mapped into one of the following canonical buckets for comparison:
- "nmd": Transcripts predicted to be subject to nonsense-mediated decay, indicating a likely loss of function.
- "protein_coding": Transcripts predicted to produce a functional protein, indicating a likely retainedfunction.
- "ambiguous": Transcripts for which the prediction is intentionally ambiguous, reflecting uncertainty in the
    classification.
- "cds_not_defined": Transcripts for which the coding sequence (CDS) is not defined, often due to missing
    information, leading to an inability to confidently classify the transcript.
- "other": Transcripts that do not fit into the above categories, including those with missing predictions or
    those that fall outside the defined classification scheme.

This allows qualitative comparison of how different models distribute their predictions across biologically 
meaningful categories, even if the specific predicted labels differ in format or terminology across models.
"""
BUCKET_ORDER = [
    "nmd",
    "protein_coding",
    "ambiguous",
    "cds_not_defined",
    "other",
]

BUCKET_COLORS = {
    "protein_coding": "#6A5ACD",   # slate purple
    "nmd": "#D4AF37",              # gold
    "ambiguous": "#C7C7C7",        # light grey
    "cds_not_defined": "#7A7A7A",  # dark grey
    "other": "#B39DDB",            # soft lavender
}

# ============================================================
# BUCKET MAPPING FUNCTIONS
# ============================================================

def bucket_from_label(label):
    """
    Derive canonical bucket from Splicing-Agent predicted label.

    Used for both Splicing-Agent and Ground Truth benchmark, which share the same label format.

    Logic:
    - If label contains "AMBIG", classify as "ambiguous".
    - If label contains "NMD+", classify as "nmd".
    - If label contains "NMD-" or "PTC", classify as "protein_coding".
    - If label contains "CDS", classify as "cds_not_defined".
    - Otherwise, classify as "other".

    Parameters:
    - label: The predicted label from the Splicing-Agent output, which may contain various terms indicating the predicted outcome for a transcript.
    
    Returns:
    - A string representing the canonical bucket classification derived from the input label, categorized as "nmd", "protein_coding", "ambiguous", 
    "cds_not_defined", or "other" based on the presence of specific keywords in the label.

    The function checks for the presence of certain keywords in the label to determine the appropriate bucket classification. If the label is missing or does not contain any of the expected keywords, it defaults to "other".
    """
    if pd.isna(label):
        return "other"
    s = str(label).upper()

    if "AMBIG" in s:
        return "ambiguous"
    if "NMD+" in s:
        return "nmd"
    if "NMD-" in s or "PTC" in s:
        return "protein_coding"
    if "CDS" in s:
        return "cds_not_defined"
    return "other"


def normalize_bucket(bucket):
    """
    Normalize bucket names from BIOMNI predictions to match canonical buckets.

    BIOMNI predictions may use different terminology or formatting for their predicted label buckets. 
    This function maps those various bucket names into the canonical set of buckets defined for comparison.

    Logic:
    - If bucket is "nmd" or "nmd+", classify as "nmd".
    - If bucket is "protein_coding" or "protein", classify as "protein_coding
    - If bucket is "ambiguous", classify as "ambiguous".
    - If bucket is "cds_not_defined" or "cds_missing", classify as "cds_not_defined".
    - Otherwise, classify as "other".
    
    Parameters:
    - bucket: The predicted label bucket from the BIOMNI output, which may use different
        terminology or formatting compared to the Splicing-Agent labels.
    
    Returns:
    - A string representing the normalized canonical bucket classification derived from the input bucket, categorized as "nmd", "protein_coding", "ambiguous", "cds_not_defined", or "other" based on the presence of specific keywords in the bucket name.
    
    The function checks for the presence of certain keywords in the bucket name to determine the appropriate canonical bucket classification. If the bucket is missing or does not contain any of the expected keywords, it defaults to "other".
    """
    if pd.isna(bucket):
        return "other"
    b = str(bucket).lower().replace("-", "_").replace(" ", "_")

    if b in {"nmd", "nmd+"}:
        return "nmd"
    if b in {"protein_coding", "protein"}:
        return "protein_coding"
    if b == "ambiguous":
        return "ambiguous"
    if b in {"cds_not_defined", "cds_missing"}:
        return "cds_not_defined"
    return "other"


# ============================================================
# LOAD DATA
# ============================================================
"""
Load prediction data from Splicing-Agent, Ground Truth benchmark, BIOMNI Lab v2, and BIOMNI v1. 
Each dataset is processed to create a DataFrame with two columns: "model" (indicating the source of the predictions) 
and "bucket" (the canonical bucket classification derived from the original predicted labels).
- Splicing-Agent predictions are loaded from a CSV file and mapped to buckets using the bucket_from
    function.
- Ground Truth benchmark predictions are loaded from the same CSV file as Splicing-Agent (since they share the same label format) and also mapped to buckets using the bucket_from_label function.
- BIOMNI Lab v2 predictions are loaded from a TSV file and normalized to buckets using the normalize_bucket function.
- BIOMNI v1 predictions are loaded from a TSV file and normalized to buckets using the normalize_bucket function.

The resulting DataFrames for each model are then concatenated into a single DataFrame (all_df) for subsequent analysis and visualization.
"""
# Splicing-Agent (ground truth benchmark; bucket derived from label)
sa = pd.read_csv(SPLICING_AGENT_LOCAL)
sa_df = pd.DataFrame({
    "model": "Splicing-Agent",
    "bucket": sa["predicted_label"].map(bucket_from_label)
})

# Ground Truth
gt = pd.read_csv(SPLICING_AGENT_LOCAL)

gt_df = pd.DataFrame({
    "model": "Ground Truth",
    "bucket": gt["predicted_label"].map(bucket_from_label)
})


# BIOMNI Lab v2 (reported bucket)
b2 = pd.read_csv(BIOMNI_V2_TSV, sep="\t")
b2_df = pd.DataFrame({
    "model": "BIOMNI Lab (v2)",
    "bucket": b2["predicted_label_bucket"].map(normalize_bucket)
})

# BIOMNI v1 (reported bucket)
b1 = pd.read_csv(BIOMNI_V1_TSV, sep="\t")
b1_df = pd.DataFrame({
    "model": "BIOMNI (v1)",
    "bucket": b1["predicted_label_bucket"].map(normalize_bucket)
})

all_df = pd.concat([gt_df, sa_df, b2_df, b1_df], ignore_index=True)


# ============================================================
# COUNT CASES PER MODEL × BUCKET
# ============================================================
"""
Count the number of transcripts classified into each bucket for each model. 
This involves grouping the combined DataFrame (all_df) by the "model" and "bucket" columns, 
counting the occurrences of each combination, and then unstacking the result to create a matrix 
where rows correspond to models and columns correspond to buckets.

To ensure a consistent order of buckets across the models, the code checks for the presence of each 
bucket in the columns of the resulting counts DataFrame and adds any missing buckets with a count of zero. 
The columns are then reordered according to the predefined BUCKET_ORDER. Finally, the rows are reordered according 
to the predefined MODEL_ORDER to ensure that the models are displayed in a specific sequence in the final visualization.
"""
counts = (
    all_df.groupby(["model", "bucket"])
          .size()
          .unstack(fill_value=0)
)

# Ensure consistent bucket order
for b in BUCKET_ORDER:
    if b not in counts.columns:
        counts[b] = 0
counts = counts[BUCKET_ORDER]

MODEL_ORDER = [
    "Ground Truth",
    "BIOMNI (v1)",
    "BIOMNI Lab (v2)",
    "Splicing-Agent",
]

counts = counts.reindex(MODEL_ORDER)


print("\n=== Figure 1 counts (model × outcome bucket) ===")
print(counts)

# ============================================================
# PLOT — SINGLE FIGURE
# ============================================================
"""
Generate a stacked bar chart comparing the distribution of NMD predictions across the Ground Truth benchmark, BIOMNI Lab v2, 
BIOMNI v1, and Splicing-Agent.

Each bar represents a model, and the segments within each bar represent the count of transcripts classified into each canonical bucket (nmd, protein_coding, ambiguous, cds_not_defined, other).
    - The colors of the segments correspond to the predefined BUCKET_COLORS, and a legend is included to explain the meaning of each color.
    - The x-axis is labeled with the model names, and the y-axis represents the number of transcripts. The plot is saved as "figure1_nmd_prediction_comparison.png" in the specified output directory.
"""
fig, ax = plt.subplots(figsize=(12, 6))  # <-- use fig, ax explicitly

counts.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=[BUCKET_COLORS[b] for b in counts.columns]
)

ax.set_xlabel("Model")
ax.set_ylabel("Number of Transcripts")
ax.set_title("NMD Prediction Comparison against Ground Truth")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# --- Build a legend OUTSIDE the axes, on the right
handles, labels = ax.get_legend_handles_labels()

# Remove the axes legend (we'll use the figure legend)
ax.legend_.remove()

# Reserve space on the right for legend
fig.subplots_adjust(right=0.80)

# Place legend in the reserved space (right side, centered)
fig.legend(
    handles,
    labels,
    title="Model Prediction Bucket",
    loc="center left",
    bbox_to_anchor=(0.82, 0.5),
    frameon=False
)

outpath = os.path.join(OUTPUT_DIR, "figure1_nmd_prediction_comparison.png")
fig.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {outpath}")
