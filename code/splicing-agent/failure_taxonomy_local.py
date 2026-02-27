from typing import Dict
from splicing_agent.state import SplicingAgentState


# ============================================================
# Failure taxonomy (v1) + tiers
# ============================================================

# Tier meanings:
# 1: Input/annotation failures (dataset/fields/canonical missing)
# 2: CDS/translation failures (cds missing, stop absent, frameshift)
# 3: NMD ambiguity (threshold edge cases, rule not applicable)
# 4: Splicing confounds (RI proxy, novel junctions)
# 5: External knowledge gaps (lit/motif db missing)
# 6: Agentic/systemic failures (tool not run, router conflict, LLM hallucination risk)

FAILURE_TIER: Dict[str, int] = {
    # -------------------------------
    # Tier 1 — Input / Annotation
    # -------------------------------
    "INPUT_MISSING": 1,
    "ANNOTATION_INCONSISTENT": 1,
    "GENOME_BUILD_MISMATCH": 1,
    "CANONICAL_UNDEFINED": 1,

    # -------------------------------
    # Tier 2 — CDS / Translation
    # -------------------------------
    "CDS_MISSING": 2,
    "START_CODON_ABSENT": 2,
    "STOP_CODON_ABSENT": 2,
    "FRAME_INCONSISTENT": 2,
    "CDS_TRUNCATED": 2,

    # -------------------------------
    # Tier 3 — NMD ambiguity
    # -------------------------------
    "NMD_RULE_INAPPLICABLE": 3,
    "NMD_DISTANCE_AMBIGUOUS": 3,
    "NMD_DEPENDS_ON_CANONICAL": 3,
    "PTC_IN_LAST_EXON": 3,

    # -------------------------------
    # Tier 4 — Splicing confounds
    # -------------------------------
    "RETAINED_INTRON_PROXY": 4,
    "NOVEL_JUNCTIONS_PRESENT": 4,
    "ISOFORM_REDUNDANT": 4,

    # -------------------------------
    # Tier 5 — External knowledge
    # -------------------------------
    "NO_LITERATURE_FOUND": 5,
    "MOTIF_DB_MISSING": 5,
    "GENE_UNCHARACTERIZED": 5,

    # -------------------------------
    # Tier 6 — Agentic / systemic
    # -------------------------------
    "TOOL_NOT_RUN": 6,
    "TOOL_OUTPUT_UNUSED": 6,
    "STATE_INCONSISTENT": 6,
    "ROUTER_CONFLICT": 6,
    "LLM_HALLUCINATION_RISK": 6,
}

# Critical failures are only Tier 1–2
CRITICAL_TIERS = {1, 2}


# ============================================================
# Failure registration helper
# ============================================================

def _add_failure(state: SplicingAgentState, code: str, note: str = "") -> None:
    """
    Add a failure code exactly once into state["failure_modes"].
    Also:
      - Register its tier in state["failure_tiers"]
      - Store/overwrite short note in state["failure_notes"][code]
    """

    # -------------------------------
    # failure_modes (list)
    # -------------------------------
    state.setdefault("failure_modes", [])
    if code not in state["failure_modes"]:
        state["failure_modes"].append(code)

    # -------------------------------
    # failure_tiers (dict)
    # -------------------------------
    state.setdefault("failure_tiers", {})
    tier = FAILURE_TIER.get(code, 0)
    state["failure_tiers"][code] = tier

    # -------------------------------
    # failure_notes (dict)
    # -------------------------------
    state.setdefault("failure_notes", {})
    if note:
        state["failure_notes"][code] = str(note)