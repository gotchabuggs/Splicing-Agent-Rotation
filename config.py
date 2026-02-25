import os
from pathlib import Path

# ============================================================
# Optional: descriptive failure packet builder
# ============================================================

try:
    from failure_taxonomy import make_failure_packet as build_failure_packet  # type: ignore
    HAVE_FAILURE_PACKET = True
except Exception:
    HAVE_FAILURE_PACKET = False


# ============================================================
# Optional LLM layer (planner + judge)
# ============================================================

try:
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    HAVE_LLM = True
except Exception:
    HAVE_LLM = False


# ============================================================
# Optional Tavily layer (web search tool)
# ============================================================

try:
    from tavily import TavilyClient  # type: ignore
    HAVE_TAVILY = True
except Exception as e:
    HAVE_TAVILY = False
    TAVILY_IMPORT_ERROR = str(e)


# ============================================================
# Repository + Dataset Paths
# ============================================================

REPO_ROOT = r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation"

DATA_DIR = os.path.join(REPO_ROOT, "data")
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmark")
TEST_RUNS_ROOT = os.path.join(DATA_DIR, "test_runs")

DEFAULT_BENCHMARK = os.path.join(BENCHMARK_DIR, "test_case_benchmark.tsv")


# ============================================================
# Prompt paths
# ============================================================

PROMPT_PATHS = {
    "loose": os.path.join(BENCHMARK_DIR, "agent_prompts", "loose_query"),
    "minimal": os.path.join(BENCHMARK_DIR, "agent_prompts", "minimal_constraint_query"),
    "strict": os.path.join(BENCHMARK_DIR, "agent_prompts", "strict_query"),
}


# ============================================================
# Dataset registry (for CLI selection)
# ============================================================

DATASET_PATHS = {
    "test_case": os.path.join(BENCHMARK_DIR, "test_case_benchmark.tsv"),
    "no_sequence": os.path.join(BENCHMARK_DIR, "no_sequence.tsv"),
    "no_genomic_coords": os.path.join(BENCHMARK_DIR, "no_genomic_coords.tsv"),
    "sequence_only": os.path.join(BENCHMARK_DIR, "sequence_only.tsv"),
}


# ============================================================
# WSL Path Normalization
# ============================================================

def win_to_wsl(path_str: str) -> str:
    """
    Convert Windows path to WSL format if needed.
    """
    if not path_str:
        return path_str

    p = path_str.strip().strip('"')

    if p.startswith("/"):
        return p

    if len(p) >= 3 and p[1:3] == ":\\":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"

    return p


def is_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME"))


def norm_path(path_str: str) -> str:
    if not path_str:
        return path_str
    return win_to_wsl(path_str) if is_wsl() else path_str


# ============================================================
# LLM Initialization Helper
# ============================================================

def initialize_llms():
    """
    Returns (planner_llm, judge_llm)
    Both are JSON-forced if available.
    """
    if not HAVE_LLM:
        return None, None

    try:
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None, None

        base_llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            api_key=key,
        )

        planner_llm = base_llm.bind(response_format={"type": "json_object"})
        judge_llm = base_llm.bind(response_format={"type": "json_object"})

        return planner_llm, judge_llm

    except Exception:
        return None, None