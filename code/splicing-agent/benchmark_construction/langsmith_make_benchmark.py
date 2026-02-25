"""
langsmith_make_benchmark.py

Uploads local Splicing-Agent benchmark TSV files to LangSmith
as structured evaluation datasets.

Purpose
-------
This script converts locally stored benchmark TSV files into
LangSmith datasets, enabling:

    • Reproducible evaluation runs
    • Prompt ablation experiments
    • Cross-run performance comparisons
    • Automated metric tracking

Each TSV file becomes a separate LangSmith dataset.

Expected TSV Structure
----------------------
Each TSV should contain:

    - One row per transcript case
    - All necessary biological metadata for the Splicing-Agent
    - Optional column: "expected_label"

If "expected_label" exists, it will be stored as reference output
for evaluation comparisons.

Dataset Naming Convention
-------------------------
Dataset names are prefixed with:

    splicing_agent__

This ensures clear grouping in LangSmith.

Overwrite Behavior
------------------
If OVERWRITE = True:
    Existing datasets with the same name are deleted and recreated.

If OVERWRITE = False:
    A version suffix (e.g., "__v2") will be appended.

This prevents accidental duplication during iterative testing.

Usage
-----
Run directly:

    python langsmith_make_benchmark.py

Requirements
------------
- Valid LangSmith API key configured in environment.
- `langsmith` Python package installed.
- TSV files must exist at specified paths.
"""

import pandas as pd
from langsmith import Client

# ============================================================
# TSV FILES TO UPLOAD
# ============================================================

"""
Mapping of dataset names to local TSV file paths. Each entry corresponds to a specific benchmark scenario for the Splicing-Agent, with the dataset name serving as a unique identifier in LangSmith and the path pointing to the local TSV file containing the benchmark data. Ensure that the paths are correct and that the TSV files are properly formatted according to the expected structure for successful upload and evaluation.

Benchmarks included:
- "splicing_agent__test_case_benchmark": A comprehensive benchmark with a variety of test
    cases covering different splicing scenarios.
- "splicing_agent__no_sequence": A benchmark where sequence information is intentionally
    omitted to evaluate the agent's performance under this specific ablation.
- "splicing_agent__sequence_only": A benchmark that includes only sequence information, without
    exon structure or genomic coordinates, to assess the agent's reliance on sequence data.
- "splicing_agent__no_genomic_coords": A benchmark where genomic coordinate information is
    missing, testing the agent's ability to function without this critical data.
"""
TSVS = {
    "splicing_agent__test_case_benchmark": r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\test_case_benchmark.tsv",
    "splicing_agent__no_sequence":        r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\no_sequence.tsv",
    "splicing_agent__sequence_only":      r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\sequence_only.tsv",
    "splicing_agent__no_genomic_coords":  r"C:\Users\justi\OneDrive\Desktop\CU-Anschutz\repos\davidsonlab\Splicing-Agent-Rotation\data\benchmark\no_genomic_coords.tsv",
}

# ============================================================
# OVERWRITE SETTING
# ============================================================

OVERWRITE = True  # set False if you prefer versioned dataset names instead

# ============================================================
# TSV LOADER
# ============================================================

def load_tsv(path: str) -> pd.DataFrame:
    """
    Loads a TSV file from the specified path into a pandas DataFrame. The function reads the TSV file, ensuring that all data is treated 
    as strings and that any missing values are filled with empty strings. If the resulting DataFrame is empty, a ValueError is raised to 
    alert the user that the TSV file may be incorrectly formatted or missing data.

    Parameters:
    - path: The file path to the TSV file to be loaded.

    Returns:
    - A pandas DataFrame containing the data from the TSV file, with all values as strings and missing values filled with empty strings.
    - Raises a ValueError if the TSV file is empty after loading.
    
    Example usage:
    df = load_tsv("path/to/benchmark.tsv")
    """
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if df.empty:
        raise ValueError(f"TSV is empty: {path}")
    return df

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """
    Main function to upload benchmark TSV files to LangSmith as structured datasets. For each dataset defined in the TSVS mapping, 
    the function loads the corresponding TSV file, checks for existing datasets in LangSmith (and handles overwriting or versioning 
    based on the OVERWRITE setting), creates a new dataset with a descriptive name and description, and populates it with examples 
    derived from the rows of the TSV file. 
    
    Each example includes the input data (bench_path, row data, and row index) and an optional reference output if the "expected_label" 
    column is present in the TSV. Finally, it prints a confirmation message for each created dataset, including the number of examples 
    uploaded and the source TSV path.

    The function relies on the LangSmith Client for dataset management and assumes that the user has a valid API key configured in their 
    environment. It also uses the load_tsv helper function to read and process the TSV files before uploading them to LangSmith.

    Workflow:
        1. Initialize the LangSmith Client.
        2. Iterate over each dataset name and corresponding TSV path in the TSVS mapping.
        3. Load the TSV file into a DataFrame using the load_tsv function.
        4. Handle dataset naming and overwriting based on the OVERWRITE setting.
        5. Create a new dataset in LangSmith with a descriptive name and description.
        6. Prepare examples from the DataFrame rows, including inputs and optional reference outputs.
        7. Upload the examples to the created dataset in LangSmith.
        8. Print a confirmation message for each dataset created.

    Example usage:
    Each time this script is run, it will process the defined TSV files and create/update datasets in LangSmith accordingly. 
    Ensure that the TSV files are correctly formatted and that the LangSmith API key is set up before running the script.
        Input: python langsmith_make_benchmark.py
        Output: Confirmation messages in the console for each dataset created, indicating the dataset name, number of examples uploaded, and source TSV path.   
    """
    client = Client()

    for dataset_name, path in TSVS.items():
        df = load_tsv(path)

        # --- Option 1 (recommended): overwrite by deleting the dataset first ---
        if OVERWRITE:
            if client.has_dataset(dataset_name=dataset_name):
                client.delete_dataset(dataset_name=dataset_name)
        else:
            # --- Option 2: version suffix instead of overwrite ---
            # e.g., splicing_agent__no_sequence__v2
            # (edit however you want)
            dataset_name = dataset_name + "__v2"

        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=f"Splicing-Agent benchmark rows uploaded from: {path}",
        )

        examples = []
        for i, row in df.iterrows():
            row_dict = row.to_dict()

            inputs = {
                "bench_path": path,
                "row": row_dict,
                "row_index": int(i),
            }

            reference = {}
            if "expected_label" in row_dict and row_dict["expected_label"].strip():
                reference["expected_label"] = row_dict["expected_label"].strip()

            examples.append({"inputs": inputs, "outputs": reference})

        client.create_examples(dataset_id=dataset.id, examples=examples)
        print(f"✅ Created dataset: {dataset_name} | n={len(examples)} | from {path}")

if __name__ == "__main__":
    main()