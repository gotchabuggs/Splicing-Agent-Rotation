from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable
from langsmith import traceable

from splicing_agent.state import SplicingAgentState
from splicing_agent.helpers import (
    _trace,
    _error,
    _tool_call,
    dataset_fingerprint_for_log,
)
from splicing_agent.failure_taxonomy_local import _add_failure


@dataclass
class cds_tool:
    exon_loader: Callable[..., Dict[str, Any]]
    cds_segment_loader: Callable[..., List[Tuple[int, int]]]
    seq_loader: Callable[..., str]

    @traceable(name="CDS", run_type="tool")
    def run(self, state: SplicingAgentState) -> SplicingAgentState:

        _tool_call(state, "CDS")
        _trace(state, "cds_tool: start")

        # ============================================================
        # Dataset fingerprints (early logging)
        # ============================================================

        try:
            state["dataset_fingerprint"] = dataset_fingerprint_for_log(
                state["dataset_path"]
            )

            if state.get("canonical_dataset_path"):
                state["canonical_dataset_fingerprint"] = dataset_fingerprint_for_log(
                    state["canonical_dataset_path"]
                )

        except Exception as e:
            _error(state, f"fingerprint failed: {e}")

        # ============================================================
        # Exon loading + transcript length + strand
        # ============================================================

        try:
            payload = self.exon_loader(
                dataset_path=state["dataset_path"],
                transcript_id=state["transcript_id"],
            )

            state["exon_table"] = payload["exon_table"]
            state["tx_len"] = int(payload["tx_len"])
            state["strand"] = int(payload.get("strand", 1))

        except Exception as e:
            _error(state, f"exon_loader failed: {type(e).__name__}: {e}")
            _add_failure(
                state,
                "INPUT_MISSING",
                "Exon table could not be loaded for transcript_id.",
            )
            _trace(state, "cds_tool: done (exon_loader failed)")
            return state

        # ============================================================
        # CDS segment loading
        # ============================================================

        try:
            cds_segments = self.cds_segment_loader(
                dataset_path=state["dataset_path"],
                transcript_id=state["transcript_id"],
            )

            state["cds_segments"] = cds_segments
            state["cds_start_tx"] = min((s for s, _ in cds_segments), default=None)
            state["cds_end_tx"] = max((e for _, e in cds_segments), default=None)

            if not cds_segments:
                _add_failure(
                    state,
                    "CDS_MISSING",
                    "No CDS segments found (cds_defined=False).",
                )

        except Exception as e:
            _error(state, f"cds_segment_loader failed: {type(e).__name__}: {e}")

            state["cds_segments"] = []
            state["cds_start_tx"] = None
            state["cds_end_tx"] = None

            _add_failure(
                state,
                "CDS_MISSING",
                "CDS segment loader failed; treating CDS as missing.",
            )

        # ============================================================
        # cDNA loading + CDS sequence construction
        # ============================================================

        try:
            cdna = self.seq_loader(
                dataset_path=state["dataset_path"],
                transcript_id=state["transcript_id"],
            )

            state["cdna_seq"] = cdna

            if state.get("cds_segments") and cdna:
                state["cds_seq"] = "".join(
                    cdna[int(s) - 1 : int(e)] for s, e in state["cds_segments"]
                )
            else:
                state["cds_seq"] = None

                if not state.get("cds_segments"):
                    _add_failure(
                        state,
                        "CDS_MISSING",
                        "CDS sequence could not be constructed because CDS segments are empty.",
                    )

        except Exception as e:
            _error(state, f"seq_loader failed: {type(e).__name__}: {e}")

            state["cdna_seq"] = None
            state["cds_seq"] = None

            _add_failure(
                state,
                "INPUT_MISSING",
                "cDNA sequence could not be loaded.",
            )

        # ============================================================
        # Basic annotation consistency check
        # ============================================================

        try:
            tx_len = state.get("tx_len")
            cds_end = state.get("cds_end_tx")

            if (
                tx_len is not None
                and cds_end is not None
                and int(cds_end) > int(tx_len)
            ):
                _add_failure(
                    state,
                    "ANNOTATION_INCONSISTENT",
                    "cds_end_tx exceeds transcript length; check BioMart export.",
                )

        except Exception:
            # Do not crash tool for secondary consistency check
            pass

        _trace(state, "cds_tool: done")
        return state