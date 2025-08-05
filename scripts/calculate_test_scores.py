import os
import json
import re
from collections import defaultdict
from pathlib import Path
from tabulate import tabulate


def parse_f_score(details_list):
    """
    Parses the F-score float value from a list of strings.
    """
    if type(details_list) == list:
        for item in details_list:
            match = re.search(r"F-score:\s*([\d\.]+)", item, re.IGNORECASE)
            if match:
                return float(match.group(1))
    else:
        match = re.search(r"F-score:\s*([\d\.]+)", details_list, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None


def main():
    """
    Walks sub‑dirs under `search_root`, gathers F1 metrics from
    numbered test‑run folders, and prints a model‑/task‑centric table.
    """
    search_root = "/Users/s0g0a87/studies/coref-llms/results/heb"
    results = defaultdict(lambda: defaultdict(list))

    # -------- collect metrics --------
    for root, _, files in os.walk(search_root):
        if "overall_F1.json" not in files:
            continue

        file_path = Path(root) / "overall_F1.json"

        # keep only test‑run folders whose parent ends with _1…_5
        if "test" not in set(file_path.parts):
            continue
        parent_ok = any(re.search(r"_[1-5]$", p) for p in file_path.parts)
        if not parent_ok:
            continue

        experiment_path = file_path.parent.parent        # …/test/<run_N>
        rel_parts = Path(experiment_path).relative_to(search_root).parts
        model = rel_parts[0].replace("_", "-")           # gpt_4o → gpt-4o
        task  = rel_parts[-1]                            # tokenized_text

        try:
            data = json.load(open(file_path))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Could not process {file_path}: {e}")
            continue

        bucket = results[(model, task)]                  # key = (model, task)
        if (v := data.get("CoNLL_F1")):
            bucket["CoNLL F1"].append(v * 100)
        if (det := data.get("Detailed_F1")):
            if (v := parse_f_score(det.get("muc", []))):     bucket["MUC F1"].append(v)
            if (v := parse_f_score(det.get("b_cubed", []))): bucket["B-Cubed F1"].append(v)
            if (v := parse_f_score(det.get("ceafe", []))):   bucket["CEAFE F1"].append(v)

    # -------- build table --------
    if not results:
        print("\nNo 'overall_F1.json' files found.")
        return

    headers = ["Model", "Task", "Runs",
               "MUC F1", "B‑Cubed F1", "CEAFE F1", "CoNLL F1"]
    table = []

    for (model, task), metrics in sorted(results.items()):
        runs = len(metrics["CoNLL F1"]) or len(next(iter(metrics.values()), []))
        avg = lambda k: (sum(metrics[k]) / len(metrics[k])) if metrics[k] else "–"
        table.append([model, task, runs,
                      avg("MUC F1"), avg("B-Cubed F1"), avg("CEAFE F1"), avg("CoNLL F1")])

    print("\n--- F1 Score Averages Across Test Runs ---")
    print(tabulate(table, headers=headers, tablefmt="grid", floatfmt=".2f"))

if __name__ == "__main__":
    main()