from typing import Any, Dict, List, Optional, Tuple, TypedDict

class SplicingAgentState(TypedDict, total=False):

    # ============================================================
    # Per-row bookkeeping
    # ============================================================

    row_id: str
    run_dir: str
    timestamp: str

    # ============================================================
    # Inputs / Metadata
    # ============================================================

    dataset_path: str
    canonical_dataset_path: str

    transcript_id: str
    canonical_transcript_id: str
    gene_symbol_hint: str
    chromosome_hint: str

    # Benchmark labels (optional)
    expected_label: str
    expected_label_provided: bool
    expected_label_inferred: bool
    expected_label_bucket: str

    # Logged fingerprints
    dataset_fingerprint: Dict[str, Any]
    canonical_dataset_fingerprint: Dict[str, Any]

    # ============================================================
    # TOOL 1 — CDS outputs
    # ============================================================

    exon_table: List[Dict[str, Any]]
    tx_len: Optional[int]
    strand: Optional[int]

    cds_segments: List[Tuple[int, int]]
    cds_start_tx: Optional[int]
    cds_end_tx: Optional[int]

    cdna_seq: Optional[str]
    cds_seq: Optional[str]

    # ============================================================
    # Canonical baseline reference
    # ============================================================

    canonical_stop_end_tx: Optional[int]
    canonical_tx_len: Optional[int]
    canonical_lastJ: Optional[int]
    margin_nt: int

    # ============================================================
    # TOOL 2 — NMD outputs
    # ============================================================

    lastJ: Optional[int]
    stop_end_tx: Optional[int]
    dist_lastJ_minus_stopEnd: Optional[int]

    ptc_predicted: Optional[bool]
    ptc_reason: str

    nmd: Optional[bool]
    nmd_reason: str

    # ============================================================
    # Stop scanning diagnostics
    # ============================================================

    stop_codons_all: List[Dict[str, Any]]
    stop_codons_internal: List[Dict[str, Any]]
    stop_codon_terminal: Optional[str]
    stop_codon_triplet: Optional[str]

    # Internal PTC candidates
    ptc_candidates: List[Dict[str, Any]]
    ptc_selected: Optional[Dict[str, Any]]

    # ============================================================
    # TOOL 3 — Motif
    # ============================================================

    motif_hits: List[Dict[str, Any]]
    motif_db_results: List[Dict[str, Any]]  # ← missing in your version

    # ============================================================
    # TOOL 4 — Tavily
    # ============================================================

    literature_notes: List[Dict[str, Any]]

    # ============================================================
    # Agent routing + execution tracking
    # ============================================================

    plan_done: List[str]
    next_tool: str
    router_reason: str

    tool_calls: List[str]
    tool_calls_count: int
    node_events: List[Dict[str, Any]]

    # ============================================================
    # Failure taxonomy outputs
    # ============================================================

    failure_modes: List[str]
    failure_tiers: Dict[str, int]
    failure_notes: Dict[str, str]
    has_critical_failure: bool
    failure_packet: Dict[str, Any]  # optional enriched packet

    # ============================================================
    # Derived label + evaluation metrics
    # ============================================================

    predicted_label: str
    predicted_label_bucket: str
    label_mismatch_bucketed: bool

    task_completed: bool
    tool_usage_accuracy: bool
    success: bool
    error_rate_flag: bool

    hallucination: bool
    tokens_total: Optional[int]
    latency_s: float

    # ============================================================
    # Reporting artifacts
    # ============================================================

    report_tables_md: str
    report_consequence_md: str
    report_metrics_md: str
    report_text: str

    # ============================================================
    # LLM Judge outputs
    # ============================================================

    judge_summary: str
    recommended_next_debug_step: str
    judge_confidence: str
    judge_error: str
    judge_errors: List[str]

    # ============================================================
    # Tracing + runtime diagnostics
    # ============================================================

    trace: List[str]
    errors: List[str]