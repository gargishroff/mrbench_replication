from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

LABEL_FIELDS = {
    "MI": "Mistake_Identification",
    "ML": "Mistake_Location",
    "PG": "Providing_Guidance",
    "AC": "Actionability",
}

VALID_LABELS = {"Yes", "To some extent", "No"}


def load_mrbench(path: str | Path, drop_unlabeled: bool = True) -> pd.DataFrame:
    path = Path(path)
    with open(path) as f:
        items = json.load(f)

    # Some MRBench files use 'anno_llm_responses' (V1/V2), v3 uses 'tutor_responses'
    rows = []
    for item in items:
        responses = item.get("tutor_responses") or item.get("anno_llm_responses") or {}
        for tutor, payload in responses.items():
            row = {
                "conversation_id": item["conversation_id"],
                "tutor": tutor,
                "history": item["conversation_history"],
                "response": payload.get("response", ""),
                "source": item.get("Data", _infer_source(item["conversation_id"])),
            }
            ann = payload.get("annotation", {})
            for short, full in LABEL_FIELDS.items():
                row[f"label_{short}"] = ann.get(full, None)
            rows.append(row)

    df = pd.DataFrame(rows)
    if drop_unlabeled:
        for short in LABEL_FIELDS:
            df = df[df[f"label_{short}"].isin(VALID_LABELS)]
    return df.reset_index(drop=True)


def _infer_source(conv_id: str) -> str:
    return "Bridge" if conv_id and conv_id.startswith(("0-", "1-", "2-", "3-", "4-")) and len(conv_id.split("-")[0]) <= 2 else "Unknown"


def describe(df: pd.DataFrame) -> dict:
    out = {
        "n_responses": len(df),
        "n_conversations": df["conversation_id"].nunique(),
        "tutors": df["tutor"].value_counts().to_dict(),
    }
    for short in LABEL_FIELDS:
        out[f"label_dist_{short}"] = df[f"label_{short}"].value_counts().to_dict()
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../mrbench/BEA_Shared_Task_2025_Datasets/mrbench_v3_testset.json"
    df = load_mrbench(path)
    print(f"Loaded {len(df)} responses from {df['conversation_id'].nunique()} conversations")
    print(df.head(3).to_string())
    print()
    import json
    print(json.dumps(describe(df), indent=2))
