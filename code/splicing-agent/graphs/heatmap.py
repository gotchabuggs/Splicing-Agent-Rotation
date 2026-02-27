"""
heatmap.py

Visual specification of expected Splicing-Agent prediction behavior across
prompt conditions and input ablations.

Purpose
-------
This script generates a heatmap describing the *expected behavioral
patterns* of the Splicing-Agent under different:

    • Prompt constraints (Strict / Minimal / Loose)
    • Input ablations (missing biological information)

This heatmap does NOT show empirical results.
It encodes the evaluation hypothesis used in the ablation study.

Biological & Evaluation Context
-------------------------------
The Splicing-Agent is evaluated under different prompt strictness levels.
We expect prompt sensitivity to influence:

    • Confidence calibration
    • Ambiguity handling
    • Refusal behavior
    • Hallucination risk (unsupported confident claims)

The heatmap visually communicates the intended evaluation framework.

Outputs
-------
A PNG file saved to:
    data/graphs/expected_outcomes_heatmap_prompt_sensitive.png

This figure is designed for:
    • Rotation talks
    • Evaluation strategy slides
    • Methods explanation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import os

# ============================================================
# OUTPUT DIRECTORY
# ============================================================
"""Directory to save the generated heatmap. Adjust this path as needed."""
OUTPUT_DIR = r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# COLORS (YOUR PALETTE, UPDATED)
# - remove dark gray entirely
# - use GOLD for one of the eval buckets (requested)
# ============================================================
"""
Each bucket is assigned a distinct color to visually differentiate expected outcomes:
- "conf_supported": A strong, positive color (e.g., slate purple) indicating confident answers with good evidence.
- "ambiguous": A neutral color (e.g., light grey) representing intentionally ambiguous answers
- "conf_unsupported": A cautionary color (e.g., GOLD) signaling confident answers with limited evidence, highlighting potential hallucination risk.
- "refuse": A softer color (e.g., soft lavender) indicating refusal to answer due to insufficient information.

Color Palette Notes:
- Uses project palette colors where possible for consistency (Purple / Gold / Lavender / Grey).
- Avoids dark gray to ensure all buckets are visually distinct and easily interpretable.
"""
BUCKET_COLORS = {
    "conf_supported": "#D4AF37",   # slate purple (good/valid confident call)
    "ambiguous": "#C7C7C7",        # light grey (ambiguous)
    "conf_unsupported": "#6A5ACD", # GOLD (confident but unsupported / hallucination-risk)
    "refuse": "#B39DDB",           # soft lavender (refuse / not computable)
}

# ============================================================
# EVAL BUCKET ORDER (controls legend + colormap)
# ============================================================
"""
Defines the order of evaluation buckets for consistent mapping to colors and legend entries.
- "conf_supported": Expected confident answers with strong evidence (best case).
- "ambiguous": Expected ambiguous answers returned intentionally (neutral case).
- "conf_unsupported": Expected confident answers with limited evidence, indicating potential hallucination risk (
cautionary case).
- "refuse": Expected refusal to answer due to insufficient information (worst case).

This order is used to create the colormap and legend, ensuring that the visual representation aligns with the intended interpretation of each bucket.
"""

BUCKET_ORDER = ["conf_supported", "ambiguous", "conf_unsupported", "refuse"]
BUCKET_TO_INT = {b: i for i, b in enumerate(BUCKET_ORDER)}

cmap = ListedColormap([BUCKET_COLORS[b] for b in BUCKET_ORDER])
norm = BoundaryNorm(
    boundaries=np.arange(-0.5, len(BUCKET_ORDER) + 0.5, 1),
    ncolors=len(BUCKET_ORDER),
)

# ============================================================
# PROMPT CONDITIONS (order matters for interpretation)
# ============================================================
"""
Prompt conditions represent different levels of constraint applied to the Splicing-Agent's responses:
- "Strict": The agent is expected to be conservative, preferring ambiguity or refusal when evidence is missing.
- "Minimal constraint": The agent is expected to be balanced, attempting to provide answers more often
    than strict but still cautious when evidence is limited.
- "Loose": The agent is expected to be completion-biased, more likely to provide confident
    answers even when data is missing, which may lead to unsupported claims (hallucinations).

These conditions are used as column labels in the heatmap and influence the expected outcomes for each input ablation scenario.
"""
PROMPTS = ["Strict", "Minimal constraint", "Loose"]

# ============================================================
# INPUT ABLATIONS
# ============================================================
"""
Input ablations represent different scenarios of missing biological information that the Splicing-Agent must contend with:
    1. "Complete information (reference case)": The agent has access to all relevant data, allowing for confident and well-supported answers.
    2. "No sequence information (CDS unknown)": The agent lacks sequence data, which should
    lead to more ambiguous or cautious responses, especially under stricter prompts.
    3. "Sequence only (no exon structure)": The agent has sequence data but lacks exon structure
    information, which may allow for some confident answers but with increased risk of unsupported claims, particularly under looser prompts.
    4. "No genomic coordinates (no splice junctions)": The agent lacks genomic coordinate data,
    which should lead to refusal to answer under strict and minimal prompts, while the loose prompt may still yield unsupported confident answers.
"""
DATASETS = [
    "Complete information\n(Reference)",
    "No sequence information\n(CDS Unknown)",
    "Sequence only\n(No Exon Structure)",
    "No genomic coordinates\n(No Splice Junctions)",
]

# ============================================================
# EXPECTED OUTCOMES GRID (PROMPT-SENSITIVE)
# rows = DATASETS, cols = PROMPTS
#
# Mapping to your narrative:
# - Strict: conservative; prefers ambiguity/refusal when evidence is missing
# - Minimal: balanced; attempts more than strict but still cautious
# - Loose: completion-biased; more confident-but-unsupported outputs when data missing
# ============================================================
"""
This grid defines the expected behavior of the Splicing-Agent across different combinations of input ablations and prompt conditions. Each cell in the grid corresponds to a specific scenario, with the value indicating the anticipated outcome:
    - "conf_supported": The agent is expected to return a confident answer that is well-supported by    
        evidence, indicating a successful classification.
    - "ambiguous": The agent is expected to return an intentionally ambiguous answer, reflecting uncertainty.           
    - "conf_unsupported": The agent is expected to return a confident answer that lacks strong evidence, indicating a potential hallucination risk.
    - "refuse": The agent is expected to refuse to answer due to insufficient information.

The grid is structured with rows representing different input ablation scenarios (from complete information to various levels of missing data) and columns representing the prompt conditions (from strict to loose). This layout allows for a clear visualization of how the expected outcomes change based on both the available information and the constraints imposed by the prompts.
"""
EXPECTED_GRID = [
    # Full input: evidence supports confident classifications under all prompts
    ["conf_supported", "conf_supported", "conf_supported"],

    # No sequence: strict/minimal should avoid confident calls; loose tends to "guess"
    ["ambiguous", "ambiguous", "conf_unsupported"],

    # Sequence only: strict is conservative but can still succeed; minimal typically best; loose overconfident
    ["conf_supported", "conf_supported", "conf_unsupported"],

    # No genomic coords: strict/minimal should refuse (not computable); loose may still make unsupported claims
    ["refuse", "refuse", "conf_unsupported"],
]

# ============================================================
# CELL TEXT (explicitly describe behavior per bucket)
# ============================================================
"""
Text annotations for each cell in the heatmap, providing a clear and concise description of the expected agent behavior corresponding to each evaluation bucket:
- "conf_supported": Indicates that the agent is expected to return a confident answer that is well-supported by evidence, representing the ideal outcome.
- "ambiguous": Indicates that the agent is expected to return an intentionally ambiguous answer, reflecting
    uncertainty in the absence of sufficient evidence.
- "conf_unsupported": Indicates that the agent is expected to return a confident answer that lacks
    strong evidence, highlighting a potential risk of hallucination or unsupported claims.
- "refuse": Indicates that the agent is expected to refuse to answer due to insufficient information
    or computational limitations, representing a cautious and responsible response in the face of uncertainty.
"""
# CELL_TEXT = {
#    "conf_supported": "AGENT\nRETURNS\nCONFIDENT\nANSWER\n(SUPPORTED)",
#    "ambiguous": "AGENT\nRETURNS\nAMBIGUOUS\nANSWER",
#    "conf_unsupported": "AGENT\nRETURNS\nCONFIDENT\nANSWER\n(WEAK SUPPORT)",
#    "refuse": "AGENT\nRETURNS\nNO ANSWER\n(INSUFFICIENT\nINFORMATION)",
# }

# ============================================================
# LEGEND LABELS (science-communication friendly)
# ============================================================
"""
Labels for the legend entries, designed to be clear and accessible for a broad audience, including those without a technical background in AI or biology:
- "conf_supported": Describes the ideal scenario where the agent provides a confident answer that is well-supported by evidence, indicating a successful and reliable response.
- "ambiguous": Describes the scenario where the agent intentionally returns an ambiguous answer, reflecting
    uncertainty due to insufficient evidence, which is a responsible behavior in such cases.
- "conf_unsupported": Describes the scenario where the agent returns a confident answer that lacks
    strong evidence, highlighting the potential risk of hallucination or unsupported claims, which is a critical consideration in evaluating AI behavior.
- "refuse": Describes the scenario where the agent refuses to answer due to insufficient information, representing a cautious and responsible response that avoids making unsupported claims.
"""
LEGEND_LABELS = {
    "conf_supported": (
        "Confident answer with\nstrong evidence (ideal case)"
    ),
    "ambiguous": (
        "Ambiguous answer returned\nintentionally (reflecting uncertainty)"
    ),
    "conf_unsupported": (
        "Confident answer with\ninsufficient evidence (hallucination risk)"
    ),
    "refuse": (
        "Refusal to answer due\nto insufficient information"
    ),
}

# ============================================================
# PLOTTING
# ============================================================
def plot_expected_outcomes_heatmap(
    grid,
    figsize=(12, 5.2),
    output_filename="expected_outcomes_heatmap_prompt_sensitive.png",
):
    """
    Generates and saves a heatmap visualizing the expected outcomes of the Splicing-Agent across different prompt conditions and input ablations.

    Parameters:
    - grid: A 2D list where each element corresponds to an evaluation bucket (eg., "conf_supported", "ambiguous", "conf_unsupported", "refuse") for the respective dataset and prompt condition.
    - figsize: Tuple specifying the size of the figure (width, height) in inches.
    - output_filename: The name of the output PNG file to save the heatmap.

    The function creates a heatmap where each cell's color corresponds to the expected outcome based on the provided grid. It also includes axes labels, gridlines, cell annotations describing the expected behavior, and a legend explaining the meaning of each color. Finally, it saves the generated heatmap to the specified output directory.
    """
    # Convert grid to int matrix
    mat = np.zeros((len(grid), len(grid[0])), dtype=int)
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            mat[r, c] = BUCKET_TO_INT[grid[r][c]]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    # Axes labels
    ax.set_xticks(range(len(PROMPTS)))
    ax.set_xticklabels(PROMPTS, fontsize=12)
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS, fontsize=12)

    # Gridlines
    ax.set_xticks(np.arange(-0.5, len(PROMPTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.invert_yaxis()

    # Cell annotations
    # for r in range(mat.shape[0]):
        # for c in range(mat.shape[1]):
            # bucket = grid[r][c]
            # ax.text(
                # c, r,
                # CELL_TEXT[bucket],
                # ha="center", va="center",
                # fontsize=9.5, fontweight="bold",
                # linespacing=1.05
            # )

    # Title
    ax.set_title(
        "Expected Agent Behavior Across Prompt Conditions\nEvaluation Strategy for Ablation Study",
        fontsize=15,
        pad=12
    )

    # Legend (RIGHT, guaranteed space)
    legend_handles = [
        Patch(
            facecolor=BUCKET_COLORS[b],
            edgecolor="black",
            label=LEGEND_LABELS[b]
        )
        for b in BUCKET_ORDER
    ]

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=13,
        handlelength=2.2,
        labelspacing=1.6,
    )

    # Make room on the right for the legend
    plt.subplots_adjust(right=0.67)

    # Save
    outpath = os.path.join(OUTPUT_DIR, output_filename)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved: {outpath}")

    plt.show()


# ============================================================
# RUN
# ============================================================
plot_expected_outcomes_heatmap(EXPECTED_GRID)
