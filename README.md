# Splicing-Agent-Rotation

**Splicing-Agent** is a research rotation project focused on designing, implementing, and evaluating an **agentic AI workflow for interpreting aberrant RNA splicing events**, with an emphasis on predicting biologically interpretable consequences such as nonsense-mediated decay (NMD).

This research project is developed as part of a rotation and serves as a **controlled, auditable testbed** for understanding how agentic systems behave on specialized biological reasoning tasks.

---

## Scientific Motivation

Aberrant mRNA splicing is a defining hallmark of **high-grade serous carcinoma (HGSC)**, which exhibits one of the highest burdens of non-canonical exon–exon junctions across all tumor types in The Cancer Genome Atlas (TCGA). This extreme transcriptomic heterogeneity generates hundreds of tumor-enriched isoforms per patient, yet the functional consequences of most splicing events remain unknown.

Some aberrant splicing events introduce **premature termination codons (PTCs)** that may trigger or escape **nonsense-mediated decay (NMD)**, while others disrupt RNA-binding protein (RBP) motifs, truncate essential protein domains, or alter cis-regulatory elements. Although many computational tools exist to analyze individual aspects of splicing—such as PSI, splice site strength, motif disruption, or NMD susceptibility—these analyses are typically performed in isolation.

As a result, current splicing interpretation workflows are:
- Highly fragmented  
- Largely manual  
- Difficult to scale to the combinatorial complexity observed in HGSC  

This project explores whether **agentic AI systems** can offer a more flexible, interpretable, and scalable paradigm for navigating this complexity.

---

## Agentic AI Framework

Agentic AI systems are large language models (LLMs) augmented with:
- Tool calling  
- Retrieval  
- Explicit state management  
- Multi-step reasoning  

Unlike traditional machine learning models that rely on fixed feature matrices, agentic systems can flexibly:
- Interpret heterogeneous input formats (JSON, CSV, free text)
- Automatically call external bioinformatics tools
- Retrieve relevant annotations
- Synthesize evidence into structured biological interpretations

This project leverages **LangGraph**, a framework for building small, stateful, testable agents with reproducible execution graphs—an essential requirement when working with heterogeneous biomedical data.

---

## Central Hypothesis

The central hypothesis of this rotation is that:

> **A narrowly scoped, task-specific agentic workflow can be rigorously evaluated using a small, controlled benchmark, enabling identification of both strengths and failure modes in splicing-focused biological reasoning.**

Rather than scaling immediately to large multimodal splicing models, this project prioritizes **interpretability, correctness, and failure analysis**.

---

## Specific Aims

### Aim 1: Define and implement an agentic workflow for interpreting aberrant splicing events

A LangGraph-based agent is designed to:
- Accept splicing events in multiple input formats (JSON, CSV, free text)
- Normalize and reinterpret inputs when necessary
- Call a single biological analysis tool (e.g., NMD classification)
- Return structured, interpretable output

A key focus of this aim is explicitly testing whether the agent produces **consistent biological interpretations when the same information is presented in different formats**.

**Expected Outcome:**  
A functional, testable LangGraph prototype that performs one complete biological task end-to-end (e.g., NMD classification) and serves as the backbone for subsequent evaluation.

---

### Aim 2: Develop a benchmark dataset for evaluating agent performance

To evaluate agent behavior in a controlled setting, a small benchmark dataset (~5–10 cases) is constructed to represent a spectrum of splicing outcomes:

- **True positives:**  
  - Frameshifts generating PTCs  
  - Protein domain truncation  
  - Loss of RBP binding motifs  

- **True negatives:**  
  - Protein-coding isoforms with no predicted functional impact  

- **Compound events:**  
  - Events combining domain disruption and NMD triggering, testing whether the agent correctly deprioritizes isoforms unlikely to be translated  

Each case includes ground-truth labels relevant to the evaluation task (e.g., NMD-triggering vs non-NMD, domain preserved vs disrupted).

Evaluation metrics include:
- Tool Usage Accuracy  
- Hallucination  
- Consistency under input-format perturbation  
- Success rate (Task Completion)
- Token Cost
- Latency

**Expected Outcome:**  
A reproducible benchmark dataset enabling rigorous evaluation of agentic reasoning and downstream comparison with Biomni.

---

### Aim 3: Evaluate agentic reasoning and compare against Biomni

Using the benchmark dataset, the agent is evaluated for:
- Correctness of tool use  
- Stability under input variation  
- Alignment with ground-truth labels  
- Interpretability of intermediate reasoning steps  

In parallel, the same splicing tasks are queried using **Biomni**, a large biomedical foundation model. This comparison highlights:
- Where specialized, task-specific agents outperform general-purpose models  
- Where LLM-based systems exhibit blind spots or reasoning failures  

**Expected Outcome:**  
A quantitative and qualitative assessment of agent performance, including identified failure modes and a direct comparison between specialized agentic pipelines and general-purpose foundation models.

---

## Biological Scope (Current)

- **Organism:** Human  
- **Reference Genome:** GRCh38  
- **Initial Benchmark Gene:** BRCA1  
- **Canonical Transcript:** BRCA1-203 (ENST00000357654)  

The BRCA1 benchmark provides a controlled setting for validating logic related to CDS truncation, PTC prediction, and NMD rules before extending to HGSC-wide analyses.

---

## Implemented Methodology (Current)

The current implementation supports:

- Flexible parsing of BioMart-style transcript exports
- Robust exon table reconstruction
- CDS start/end aggregation
- Observed vs canonical stop codon comparison
- **PTC prediction** based on CDS truncation
- **NMD prediction using the 50–55 nucleotide exon junction complex (EJC) rule**
  - Applied only when a PTC is predicted
- Optional **retained intron proxy** using junction mismatch

Ground-truth labels are intentionally **kept outside the agent** to prevent label leakage during evaluation.

---

## Benchmarking Philosophy

This project emphasizes:
- Deterministic, rule-based biological logic
- Explicit biological assumptions
- Auditable intermediate states
- Failure-mode discovery over raw performance

The goal is not to maximize accuracy alone, but to understand **why and how agentic systems succeed or fail** in splicing interpretation.

---

## Quick Start

### Requirements
- Python 3.10+ (recommended)
- Install dependencies:
```bash
pip install -r requirements.txt
```
