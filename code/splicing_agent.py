from typing import Optional, TypedDict, Literal
import json
import csv
import io
import os

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# =========================
# Load environment
# =========================

# Loads OPEN_API_KEY (not OPENAI_API_Key.. misspelled) from splicing-agent env
load_dotenv()
api_key = os.getenv("OPEN_API_KEY")
if not os.getenv("OPEN_API_KEY"): 
    raise RuntimeError(
        "OPENAI_API_KEY not set. Create a .env file with OPENAI_API_KEY=... "
        "or export it in your shell."
    )

# =========================
# Canonical schemas
# =========================

class SplicingEvent(BaseModel):
    """
    Canonical representation of an aberrant splicing event.

    Extend later with more features (ex: domain, motif info).
    """
    sample_id: Optional[str] = Field(
        default=None,
        description="Sample or patient identifier, if provided."
    )
    gene_symbol: Optional[str] = Field(
        default=None,
        description="HGNC gene symbol, e.g. 'BRCA1', 'BRCA2', 'TP53', if provided."
    )
    transcript_id: Optional[str] = Field(
        default=None,
        description="Transcript ID (e.g. Ensembl or RefSeq) if available."
    )
    event_description: str = Field(
        description=(
            "Natural language description of the aberrant splicing event, "
            "including exon/intron information, junctions, and any frameshift/PTC info."
        )
    )


class NMDResult(BaseModel):
    """
    Output of the NMD classification tool.
    """
    nmd_likelihood: Literal["likely_triggering", "likely_escaping", "uncertain"] = Field(
        description="High-level classification of whether the transcript is likely NMD-triggering."
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="How confident the tool is in this assessment."
    )
    rationale: str = Field(
        description="Short explanation of why this NMD label was assigned."
    )

# =========================
# Agent state
# =========================

class SplicingAgentState(TypedDict, total=False):
    raw_input: str
    input_format: str              # "json", "csv", or "text"
    event: SplicingEvent
    nmd_result: NMDResult


# =========================
# 3. Nodes
# =========================

def detect_format(state: SplicingAgentState) -> SplicingAgentState:
    """Heuristically detect whether the input looks like JSON, CSV, or free text."""
    raw = state["raw_input"].strip()

    if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
        input_format = "json"
    elif "," in raw and "\n" not in raw:
        input_format = "csv"
    else:
        input_format = "text"

    return {"input_format": input_format}


def normalize_event(state: SplicingAgentState) -> SplicingAgentState:
    """
    Normalize JSON / CSV / free-text into a canonical SplicingEvent
    using an LLM with structured output.
    """
    raw = state["raw_input"]
    input_format = state["input_format"]

    # Pre-parse based on format to give the model structured context
    if input_format == "json":
        try:
            parsed_repr = json.loads(raw)
        except json.JSONDecodeError:
            parsed_repr = {"raw": raw}
    elif input_format == "csv":
        f = io.StringIO(raw)
        reader = csv.reader(f)
        row = next(reader)
        parsed_repr = {"csv_row": row}
    else:
        parsed_repr = {"free_text": raw}

    system_prompt = """
You are an RNA splicing assistant working with BRCA1, BRCA2, and other cancer-relevant genes.

Your job is to take a description of an aberrant or alternative splicing event in ANY format
(JSON, CSV row, or free text) and map it into a canonical schema called SplicingEvent.

- If sample_id, gene_symbol, or transcript_id are present, fill them in.
- If they are missing, leave them as null.
- ALWAYS fill event_description with a clear, concise natural language description
  of the splicing event suitable for downstream NMD interpretation.
"""

    structured_model = model.with_structured_output(SplicingEvent)

    prompt = (
        system_prompt
        + "\n\nHere is the raw input describing the event:\n"
        + json.dumps(parsed_repr, indent=2)
    )

    event = structured_model.invoke(prompt)

    return {"event": event}


def run_nmd_classifier(state: SplicingAgentState) -> SplicingAgentState:
    """
    Call a single 'biological tool': an NMD classifier implemented as an LLM.

    Later, you can:
    - Replace this with a Python function that calls a real model or script.
    - Or wrap external command-line tools / APIs.
    """
    event = state["event"]

    system_prompt = """
You are an expert in RNA splicing and nonsense-mediated decay (NMD),
with particular familiarity with BRCA1 and BRCA2 isoforms in ovarian cancer.

Given an aberrant or alternative splicing event, decide whether the resulting transcript is:

- likely_triggering: very likely to be targeted by NMD
- likely_escaping: likely to escape NMD and produce a stable transcript
- uncertain: insufficient information or genuinely ambiguous

Use only the information in the event description.

Return:
- nmd_likelihood: "likely_triggering", "likely_escaping", or "uncertain"
- confidence: "low", "medium", or "high"
- rationale: 3–6 sentences explaining your reasoning in clear biological language.
"""

    structured_model = model.with_structured_output(NMDResult)

    prompt = (
        system_prompt
        + "\n\nHere is the splicing event:\n"
        + event.model_dump_json(indent=2)
    )

    nmd_result = structured_model.invoke(prompt)

    return {"nmd_result": nmd_result}

# =========================
# LLM setup
# =========================

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

# =========================
# Nodes
# =========================

def detect_format(state: SplicingAgentState) -> SplicingAgentState:
    """Heuristically detect whether the input looks like JSON, CSV, or free text."""
    raw = state["raw_input"].strip()

    # Very simple heuristics; good enough for Aim 1
    if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
        input_format = "json"
    elif "," in raw and "\n" not in raw:
        input_format = "csv"
    else:
        input_format = "text"

    return {"input_format": input_format}


def normalize_event(state: SplicingAgentState) -> SplicingAgentState:
    """
    Normalize JSON / CSV / free-text into a canonical SplicingEvent
    using an LLM with structured output.
    """
    raw = state["raw_input"]
    input_format = state["input_format"]

    # Pre-parse based on format to give the model structured context
    if input_format == "json":
        try:
            parsed_repr = json.loads(raw)
        except json.JSONDecodeError:
            parsed_repr = {"raw": raw}
    elif input_format == "csv":
        f = io.StringIO(raw)
        reader = csv.reader(f)
        row = next(reader)
        parsed_repr = {"csv_row": row}
    else:
        parsed_repr = {"free_text": raw}

    system_prompt = """
You are an RNA splicing assistant working with BRCA1, BRCA2, and other cancer-relevant genes.

Your job is to take a description of an aberrant or alternative splicing event in ANY format
(JSON, CSV row, or free text) and map it into a canonical schema called SplicingEvent.

- If sample_id, gene_symbol, or transcript_id are present, fill them in.
- If they are missing, leave them as null.
- ALWAYS fill event_description with a clear, concise natural language description
  of the splicing event suitable for downstream NMD interpretation.
"""

    structured_model = model.with_structured_output(SplicingEvent)

    prompt = (
        system_prompt
        + "\n\nHere is the raw input describing the event:\n"
        + json.dumps(parsed_repr, indent=2)
    )

    # NOTE: passing a str, not a dict
    event = structured_model.invoke(prompt)

    return {"event": event}



def run_nmd_classifier(state: SplicingAgentState) -> SplicingAgentState:
    """
    Call a single 'biological tool': an NMD classifier implemented as an LLM.
    """
    event = state["event"]

    system_prompt = """
You are an expert in RNA splicing and nonsense-mediated decay (NMD),
especially for BRCA1/BRCA2 isoforms in ovarian cancer.

Given an aberrant or alternative splicing event, decide whether the resulting transcript is:

- likely_triggering: very likely to be targeted by NMD
- likely_escaping: likely to escape NMD and produce a stable transcript
- uncertain: insufficient information or genuinely ambiguous

Use only the information in the event description (do not invent specific exons or domains).

Return:
- nmd_likelihood: "likely_triggering", "likely_escaping", or "uncertain"
- confidence: "low", "medium", or "high"
- rationale: 3–6 sentences explaining your reasoning.
"""

    structured_model = model.with_structured_output(NMDResult)

    prompt = (
        system_prompt
        + "\n\nHere is the splicing event:\n"
        + event.model_dump_json(indent=2)
    )

    # Again: pass a string, not a dict
    nmd_result = structured_model.invoke(prompt)

    return {"nmd_result": nmd_result}

# =========================
# Build the graph
# =========================

builder = StateGraph(SplicingAgentState)

builder.add_node("detect_format", detect_format)
builder.add_node("normalize_event", normalize_event)
builder.add_node("run_nmd_classifier", run_nmd_classifier)

builder.add_edge(START, "detect_format")
builder.add_edge("detect_format", "normalize_event")
builder.add_edge("normalize_event", "run_nmd_classifier")
builder.add_edge("run_nmd_classifier", END)

graph = builder.compile()