# Splicing-Agent-Rotation

**Splicing-Agent** is a research rotation project focused on designing, implementing, and rigorously evaluating an **agentic AI workflow for interpreting aberrant RNA splicing events**, with emphasis on predicting biologically interpretable fucntional consequences such as **nonsense-mediated decay (NMD)**.

This project serves as a **controlled, auditable testbed** for understanding how agentic systems behave on agentic, structured biological reasoning tasks.

---

# Scientific Motivation

Aberrant mRNA splicing is a defining hallmark of **high-grade serous carcinoma (HGSC)**, which exhibits one of the highest burdens of non-canonical exon–exon junctions across all tumor types in The Cancer Genome Atlas (TCGA).

This transcriptomic heterogeneity generates hundreds of tumor-enriched isoforms per patient, yet the functional consequences of most splicing events remain unknown.

Aberrant isoforms may:

- Introduce **premature termination codons (PTCs)**
- Trigger or escape **nonsense-mediated decay (NMD)**
- Disrupt protein domains
- Alter RNA-binding protein motifs
- Modify cis-regulatory elements

Existing computational tools analyze these features independently. Real-world interpretation workflows are therefore:

- Fragmented  
- Manual  
- Difficult to scale  
- Hard to audit  

This project explores whether a **stateful, agentic workflow** can provide a more structured and interpretable paradigm for splicing interpretation.

---

# System Architecture

Splicing-Agent is implemented using **LangGraph**, enabling:

- Explicit state tracking
- Deterministic tool execution
- Hard workflow constraints
- Reproducible execution graphs
- Structured evaluation via LangSmith

The agent consists of:

## Deterministic Biological Tools

- **CDS tool** – reconstructs exon table and CDS boundaries from Ensembl BioMart-style exports  
- **NMD tool** – applies the 50–55 nt exon junction complex (EJC) rule  
- **Motif tool** – RNA secondary structure detection  
- **Tavily tool** – optional literature retrieval 

## Agent Router

- Determines tool execution order  
- Enforces required steps  
- Prevents invalid workflows  

## Failure Compiler

Centralizes failure taxonomy and computes:

- `predicted_label`
- `task_completed`
- `tool_usage_accuracy`
- `has_critical_failure`
- `error_rate_flag`

## LLM Judge (Optional)

- Post-graph validation  
- Hallucination detection  
- Structured JSON output  

---

# Evaluation Framework

Evaluation is performed using **LangSmith** with:

- Reproducible benchmark datasets  
- Automated experiment submission  
- Prompt ablation support  
- Dataset ablation support  
- Custom evaluators  

Evaluation dimensions include:

| Category | Metric |
|----------|--------|
| Biological correctness | Label bucket match |
| Workflow validity | Hard order constraint |
| Tool discipline | Tool usage accuracy |
| Runtime stability | No runtime error |
| Failure taxonomy | No critical failure |
| Hallucination | Hallucination-free |
| Efficiency | Latency + token cost |

This separation ensures the system is evaluated on **interpretability and discipline**, not just prediction accuracy.

---

# Benchmark Philosophy

The benchmark is intentionally small and controlled.

It includes:

- True NMD-triggering transcripts  
- Protein-coding controls  
- Canonical transcript baselines  
- Missing-data ablations  
- Sequence-only ablations  
- No-genomic-coordinate ablations  

Ground-truth labels are **kept outside the agent state** to prevent label leakage.

---

# Current Biological Scope

- **Organism:** Human  
- **Genome:** GRCh38  
- **Initial gene:** BRCA1  
- **Canonical transcript:** ENST00000357654 (BRCA1-203)  

The BRCA1 test case validates:

- CDS truncation logic  
- PTC prediction  
- NMD rule implementation  
- Canonical baseline comparison  

---

# LangSmith Integration

The project includes:

## 1. Dataset uploader

```bash
python langsmith_make_benchmark.py
```

## 2. Experiment runner

```bash
python langsmith.py
```

Optional tracing:

```bash
python langsmith.py --trace
```

This enables:

- Compare table analysis  
- Cross-prompt experiments  
- Cross-dataset experiments  
- Automated metric aggregation  

---

### Key Biological Findings

- **Coding sequence (CDS) truncation detected** relative to the canonical transcript  
- **Premature termination codon (PTC) predicted** based on upstream stop codon position  
- **Exon junction complex (EJC) rule applied**, with the stop codon located 227 nt upstream of the final exon–exon junction  
- **Nonsense-mediated decay (NMD) predicted**, indicating that productive protein translation is unlikely  


## Benchmarking Philosophy
# Design Principles

This project prioritizes:

- Deterministic biological rules  
- Explicit failure classification  
- Hard workflow enforcement  
- Auditability  
- Interpretability  
- Controlled evaluation over scale  

The objective is not maximal accuracy, but **mechanistic understanding of agentic reasoning behavior**.

---

### Example Output of Summary Report
In addition to line-by-line diagnostic logs shown above, the agent produces a **structured splicing consequence summary** that consolidates key biological findings into an interpretable report. This summary is the primary artifact used for benchmarking, evaluation, and comparison across models.

```text
Transcript ID: ENST00000461798
Gene: BRCA1
Chromosome: 17
Strand: -1

Transcript Length: 582 bp
Canonical Transcript Length: 7088 bp

CDS Length: 192 bp
Protein Length (including stop): 64 aa
Protein Ends with Stop: Yes

PTC Predicted: Yes
NMD Predicted: Yes
NMD Reason: EJC rule applied (ptc_predicted=True); distance from last junction to stop end = 227 bp (≥ 55 bp threshold)```

---


## Quick Start

## 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 2️⃣ Set environment variables

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
LANGSMITH_API_KEY=...
```

## 3️⃣ Upload benchmark to Langsmith

```bash
python langsmith_make_benchmark.py
```

## 4️⃣ Run experiment + evaluation on Langsmith

```bash
python langsmith_run_experiment.py
```

---

# Project Status

Current version includes:

- Deterministic NMD classifier  
- Canonical transcript comparison  
- Structured failure taxonomy  
- Hallucination flagging  
- LangSmith experiment automation  
- Prompt ablation infrastructure  

Future directions + possible extensions:

- Domain truncation logic  
- RBP motif enrichment  
- Multi-gene scaling  
- HGSC-wide evaluation  