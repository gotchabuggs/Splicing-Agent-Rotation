from typing import Dict, List, Any
from dataclasses import dataclass
from langsmith import traceable

from splicing_agent.state import SplicingAgentState
from splicing_agent.helpers import (
    _trace,
    _error,
    _tool_call,
    STOP_CODONS,
    _scan_stops_all_inframe,
    _stop_candidate_end_tx,
    _split_internal_vs_terminal_stop,
)
from splicing_agent.failure_taxonomy_local import _add_failure


@dataclass
class nmd_tool:
    ejc_threshold_nt: int = 55
    require_ptc_for_nmd: bool = True
    margin_nt: int = 55  # PTC buffer vs reference stop

    @traceable(name="NMD", run_type="tool")
    def run(self, state: SplicingAgentState) -> SplicingAgentState:

        _tool_call(state, "NMD")
        _trace(state, "nmd_tool: start")

        # ============================================================
        # Compute lastJ (penultimate exon end in transcript coords)
        # ============================================================

        exons = state.get("exon_table", []) or []

        if len(exons) >= 2:
            penult = exons[-2]
            state["lastJ"] = (
                penult.get("end_tx")
                or penult.get("end")
                or penult.get("exon_end_tx")
            )
        else:
            state["lastJ"] = None
            _add_failure(
                state,
                "NMD_RULE_INAPPLICABLE",
                "Transcript has <2 exons; no exon–exon junction available for EJC rule.",
            )

        # Observed stop position (CDS end in transcript coords)
        state["stop_end_tx"] = state.get("cds_end_tx")

        cds_seq = state.get("cds_seq") or ""
        cds_segments = state.get("cds_segments") or []

        # ============================================================
        # STOP SCANNING (all + internal + terminal)
        # ============================================================

        if cds_seq:

            all_hits = _scan_stops_all_inframe(cds_seq)
            internal_hits, terminal_hit = _split_internal_vs_terminal_stop(
                all_hits, cds_seq
            )

            state["stop_codons_all"] = all_hits
            state["stop_codons_internal"] = internal_hits
            state["stop_codon_terminal"] = (
                terminal_hit["codon"] if terminal_hit else None
            )

            state["stop_codon_triplet"] = (
                cds_seq[-3:]
                if (len(cds_seq) >= 3 and cds_seq[-3:] in STOP_CODONS)
                else None
            )

            if not state.get("stop_codon_terminal"):
                _add_failure(
                    state,
                    "STOP_CODON_ABSENT",
                    "No terminal in-frame stop codon found in CDS sequence.",
                )

            if len(cds_seq) % 3 != 0:
                _add_failure(
                    state,
                    "FRAME_INCONSISTENT",
                    "CDS length not divisible by 3; potential frameshift/annotation issue.",
                )

        else:
            state["stop_codons_all"] = []
            state["stop_codons_internal"] = []
            state["stop_codon_terminal"] = None
            state["stop_codon_triplet"] = None

            _add_failure(
                state,
                "CDS_MISSING",
                "Cannot scan stop codons; CDS sequence missing.",
            )

        # ============================================================
        # Internal PTC candidates mapped to transcript coordinates
        # ============================================================

        lastJ = state.get("lastJ")
        ptc_candidates: List[Dict[str, Any]] = []

        for h in (state.get("stop_codons_internal") or []):

            cand = dict(h)

            try:
                cand_end_tx = _stop_candidate_end_tx(
                    cds_segments,
                    cand["cds_nt_offset"],
                )
            except Exception as e:
                cand_end_tx = None
                _error(
                    state,
                    f"map stop->tx failed for offset={cand.get('cds_nt_offset')}: "
                    f"{type(e).__name__}: {e}",
                )

            cand["stop_end_tx_candidate"] = cand_end_tx

            if (lastJ is not None) and (cand_end_tx is not None):
                cand_dist = int(lastJ) - int(cand_end_tx)
                cand["dist_lastJ_minus_stopEnd_candidate"] = cand_dist
                cand["ejc_nmd_candidate"] = bool(
                    cand_dist >= int(self.ejc_threshold_nt)
                )
            else:
                cand["dist_lastJ_minus_stopEnd_candidate"] = None
                cand["ejc_nmd_candidate"] = None

            ptc_candidates.append(cand)

        state["ptc_candidates"] = ptc_candidates
        state["ptc_selected"] = ptc_candidates[0] if ptc_candidates else None

        # ============================================================
        # Canonical reference PTC-by-position logic
        # ============================================================

        state["margin_nt"] = int(self.margin_nt)

        can_stop = state.get("canonical_stop_end_tx")
        obs_stop = state.get("stop_end_tx")

        if obs_stop is None:

            state["ptc_predicted"] = False
            state["ptc_reason"] = (
                "Early stop = unknown -> observed stop position missing "
                "(CDS missing/undefined)"
            )

            _add_failure(
                state,
                "CDS_MISSING",
                "Observed stop_end_tx is missing; PTC inference not reliable.",
            )

        elif can_stop is None:

            state["ptc_predicted"] = False
            state["ptc_reason"] = (
                "Early stop = unknown -> reference stop position missing"
            )

            _add_failure(
                state,
                "CANONICAL_UNDEFINED",
                "Canonical stop position missing; PTC-by-delta logic not applicable.",
            )

        else:

            obs = int(obs_stop)
            can = int(can_stop)

            state["ptc_predicted"] = bool(
                obs <= (can - int(self.margin_nt))
            )

            state["ptc_reason"] = (
                f"Early stop predicted={state['ptc_predicted']} "
                f"by reference comparison: "
                f"observed_stop_end_tx={obs} "
                f"<= reference_stop_end_tx={can} - margin={int(self.margin_nt)}"
            )

        # ============================================================
        # EJC DISTANCE RULE (only if PTC predicted True)
        # ============================================================

        if (lastJ is None) or (obs_stop is None):

            state["dist_lastJ_minus_stopEnd"] = None
            state["nmd"] = False
            state["nmd_reason"] = (
                "NMD=unknown -> set False "
                "(missing CDS or lastJ/stop_end_tx)"
            )

            _trace(state, "nmd_tool: done (unknown -> False)")
            return state

        dist = int(lastJ) - int(obs_stop)
        state["dist_lastJ_minus_stopEnd"] = dist

        # Last exon signature
        if dist < 0:
            _add_failure(
                state,
                "PTC_IN_LAST_EXON",
                f"dist={dist} (stop is downstream of lastJ).",
            )

        # Ambiguity window (±5 nt around threshold)
        if abs(dist - int(self.ejc_threshold_nt)) <= 5:
            _add_failure(
                state,
                "NMD_DISTANCE_AMBIGUOUS",
                f"Distance={dist} is within ±5 nt of threshold={int(self.ejc_threshold_nt)}.",
            )

        # Skip EJC if PTC not predicted
        if self.require_ptc_for_nmd and not state.get("ptc_predicted"):

            state["nmd"] = False
            state["nmd_reason"] = (
                "Skipped distance rule (early stop predicted=False) -> NMD=False"
            )

            _trace(state, "nmd_tool: done (ptc_predicted=False)")
            return state

        # Apply EJC threshold
        state["nmd"] = bool(dist >= int(self.ejc_threshold_nt))
        state["nmd_reason"] = (
            f"Distance rule applied (early stop predicted=True): "
            f"distance={dist} "
            f"{'>=' if state['nmd'] else '<'} "
            f"{int(self.ejc_threshold_nt)} "
            f"-> NMD={state['nmd']}"
        )

        _trace(state, "nmd_tool: done")
        return state