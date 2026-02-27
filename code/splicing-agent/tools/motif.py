from dataclasses import dataclass
from langsmith import traceable
from splicing_agent.state import SplicingAgentState
from splicing_agent.helpers import _trace, _tool_call

from typing import List, Dict, Any
from dataclasses import dataclass

from splicing_agent.state import SplicingAgentState
from splicing_agent.helpers import _trace, _tool_call
from splicing_agent.failure_taxonomy_local import _add_failure


@dataclass
class motif_tool:
    """
    Deterministic RNA secondary structure motif detector
    + structured database-style enrichment (CoSSMos-inspired).

    Does NOT modify predicted_label.
    Only adds structural evidence.
    """

    def run(self, state: SplicingAgentState) -> SplicingAgentState:

        _tool_call(state, "MOTIF")
        _trace(state, "motif_tool: start")

        # Normalize to RNA alphabet
        seq = (state.get("cds_seq") or "").upper().replace("T", "U")

        if not seq:
            state["motif_hits"] = []
            state["motif_db_results"] = []
            _add_failure(
                state,
                "MOTIF_DB_MISSING",
                "No CDS sequence available."
            )
            _trace(state, "motif_tool: done (no sequence)")
            return state

        hits: List[Dict[str, Any]] = []

        # Reverse complement (RNA)
        def revcomp(s: str) -> str:
            return s[::-1].translate(str.maketrans("AUGC", "UACG"))

        # ============================================================
        # 1️⃣ Hairpin detection (stem-loop heuristic)
        # ============================================================

        for i in range(len(seq) - 12):
            stem = seq[i:i+6]
            rc = revcomp(stem)
            search_region = seq[i+6:i+40]

            if rc in search_region:
                j = search_region.index(rc)

                hits.append({
                    "motif_name": "hairpin_loop",
                    "start": i + 1,
                    "end": i + 6,
                    "sequence": stem,
                    "loop_distance": j,
                    "evidence": "inverted_repeat_detected",
                })

                break  # one strong hit is enough

        # ============================================================
        # 2️⃣ Internal loop heuristic
        # ============================================================

        for i in range(len(seq) - 14):
            left = seq[i:i+4]
            right = seq[i+8:i+12]

            if revcomp(left) == right:
                mismatch = seq[i+4:i+8]

                hits.append({
                    "motif_name": "internal_loop",
                    "start": i + 1,
                    "end": i + 12,
                    "sequence": seq[i:i+12],
                    "mismatch_region": mismatch,
                    "evidence": "paired_flanks_with_internal_mismatch",
                })

                break

        # ============================================================
        # 3️⃣ Single-nucleotide bulge heuristic
        # ============================================================

        for i in range(len(seq) - 10):
            left = seq[i:i+4]
            bulge = seq[i+4]
            right = seq[i+5:i+9]

            if revcomp(left) == right:
                hits.append({
                    "motif_name": "single_nt_bulge",
                    "start": i + 1,
                    "end": i + 9,
                    "sequence": seq[i:i+9],
                    "bulge_nt": bulge,
                    "evidence": "single_unpaired_nucleotide",
                })

                break

        state["motif_hits"] = hits

        # ============================================================
        # 4️⃣ Structured database enrichment (CoSSMos-style)
        # ============================================================

        db_results: List[Dict[str, Any]] = []

        for hit in hits:
            mtype = hit.get("motif_name")

            if mtype == "hairpin_loop":
                db_results.append({
                    "motif_type": "hairpin",
                    "database_supported": True,
                    "known_sizes_nt": "3–7",
                    "structural_features": [
                        "stacking_interactions",
                        "base_pairing_edges",
                        "sugar_pucker_variability",
                    ],
                    "reference_database": "RNA CoSSMos",
                })

            elif mtype == "internal_loop":
                db_results.append({
                    "motif_type": "internal_loop",
                    "database_supported": True,
                    "common_sizes": "1x1, 2x2, 3x3",
                    "structural_features": [
                        "noncanonical_base_pairs",
                        "Hoogsteen_interactions",
                        "Watson_Crick_edges",
                    ],
                    "reference_database": "RNA CoSSMos",
                })

            elif mtype == "single_nt_bulge":
                db_results.append({
                    "motif_type": "bulge_loop",
                    "database_supported": True,
                    "common_sizes": "1 nt",
                    "structural_features": [
                        "outward_stack",
                        "backbone_distortion",
                    ],
                    "reference_database": "RNA CoSSMos",
                })

        state["motif_db_results"] = db_results

        # ============================================================
        # Failure condition (detected motif but no DB enrichment)
        # ============================================================

        if hits and not db_results:
            _add_failure(
                state,
                "NO_LITERATURE_FOUND",
                "Motif detected but no structural annotation available."
            )

        _trace(
            state,
            f"motif_tool: done | hits={len(hits)} | db_results={len(db_results)}"
        )

        return state