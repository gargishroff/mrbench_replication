from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix


VALID_LABELS = ["Yes", "To some extent", "No"]
LABEL_TO_BINARY = {"Yes": "Pos", "To some extent": "Pos", "No": "Neg"}
DIMS = ["MI", "ML", "PG", "AC"]


# Top published scores on the BEA 2025 leaderboards (exact F1 / lenient F1).
# Used as the comparison target.
BEA_2025_TOP = {
    "MI": {"exact_f1": 0.7181, "lenient_f1": 0.8957, "team": "BJTU"},
    "ML": {"exact_f1": 0.5983, "lenient_f1": 0.8386, "team": "BLCU-ICALL"},
    "PG": {"exact_f1": 0.5834, "lenient_f1": 0.7798, "team": "MSA"},
    "AC": {"exact_f1": 0.7085, "lenient_f1": 0.8527, "team": "bea-jh"},
}


def load_predictions(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame) -> dict:
    results = {}
    n = len(df)
    n_parse_ok = df["pred"].notna().sum()

    for dim in DIMS:
        gold = df["gold"].apply(lambda d: d.get(dim)).tolist()
        pred = df["pred"].apply(lambda d: (d or {}).get(dim, "MISSING")).tolist()

        # Replace any prediction not in valid labels with "No" as a fallback
        # (BEA protocol: parse-fail = penalty).
        pred_clean = [p if p in VALID_LABELS else "No" for p in pred]

        # Exact F1 (3-class macro)
        exact_f1 = f1_score(gold, pred_clean, labels=VALID_LABELS, average="macro", zero_division=0)
        exact_acc = accuracy_score(gold, pred_clean)

        # Lenient F1 (binary: Yes/ToSomeExtent vs No)
        gold_bin = [LABEL_TO_BINARY[g] for g in gold]
        pred_bin = [LABEL_TO_BINARY[p] for p in pred_clean]
        lenient_f1 = f1_score(gold_bin, pred_bin, labels=["Pos", "Neg"], average="macro", zero_division=0)
        lenient_acc = accuracy_score(gold_bin, pred_bin)

        results[dim] = {
            "exact_f1": exact_f1,
            "exact_acc": exact_acc,
            "lenient_f1": lenient_f1,
            "lenient_acc": lenient_acc,
            "classification_report": classification_report(
                gold, pred_clean, labels=VALID_LABELS, zero_division=0, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(
                gold, pred_clean, labels=VALID_LABELS
            ).tolist(),
        }

    results["meta"] = {"n": n, "n_parse_ok": int(n_parse_ok)}
    return results


def render_comparison_table(results: dict, model_name: str = "this model") -> str:
    lines = [
        f"\n{'='*78}",
        f"  MRBench replication results - {model_name}",
        f"  n={results['meta']['n']} responses, {results['meta']['n_parse_ok']} parsed OK",
        f"{'='*78}\n",
    ]
    header = f"  {'Track':<26} {'Ex F1':>8} {'BEA top':>8} {'Gap':>8} | {'Len F1':>8} {'BEA top':>8} {'Gap':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    track_names = {
        "MI": "1. Mistake Identification",
        "ML": "2. Mistake Location",
        "PG": "3. Providing Guidance",
        "AC": "4. Actionability",
    }
    for dim in DIMS:
        r = results[dim]
        top = BEA_2025_TOP[dim]
        exact_gap = r["exact_f1"] - top["exact_f1"]
        lenient_gap = r["lenient_f1"] - top["lenient_f1"]
        lines.append(
            f"  {track_names[dim]:<26} {r['exact_f1']:>8.4f} {top['exact_f1']:>8.4f} {exact_gap:>+8.4f} "
            f"| {r['lenient_f1']:>8.4f} {top['lenient_f1']:>8.4f} {lenient_gap:>+8.4f}"
        )
    lines.append("\n  Top BEA 2025 systems were typically FINE-TUNED MPNet/LLM ensembles,")
    lines.append("  not zero-shot judges. The gap reflects that, not just model quality.")
    return "\n".join(lines)


def render_confusion_matrices(results: dict) -> str:
    lines = ["\nConfusion matrices (rows=gold, cols=pred; order: Yes / To some extent / No)\n"]
    for dim in DIMS:
        cm = results[dim]["confusion_matrix"]
        lines.append(f"  {dim}:")
        for label, row in zip(VALID_LABELS, cm):
            lines.append(f"    {label:<16} {row}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <predictions.jsonl> [model_name]")
        sys.exit(1)
    path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else Path(path).stem
    df = load_predictions(path)
    res = evaluate(df)
    print(render_comparison_table(res, name))
    print(render_confusion_matrices(res))

    out_json = Path(path).with_suffix(".eval.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSummary written to {out_json}")
