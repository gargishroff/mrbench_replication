from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
import data_loader


def main(path: str = "../mrbench/BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json"):
    df = data_loader.load_mrbench(path)
    print(f"# MRBench v3 dev set: distribution analysis\n")
    print(f"Loaded {len(df)} labelled responses across {df['conversation_id'].nunique()} conversations.\n")

    print("## Tutor representation\n")
    tc = df["tutor"].value_counts()
    for tutor, n in tc.items():
        print(f"  {tutor:<14} {n}")
    print()

    # Labels per dimension (overall imbalance)
    print("## Label distribution per dimension\n")
    print(f"  {'Dim':<4} {'Yes':>6} {'ToSomeExtent':>14} {'No':>6} {'majority-frac':>14}")
    print("  " + "-" * 50)
    for short in ("MI", "ML", "PG", "AC"):
        c = df[f"label_{short}"].value_counts()
        n_yes = c.get("Yes", 0)
        n_tse = c.get("To some extent", 0)
        n_no = c.get("No", 0)
        total = n_yes + n_tse + n_no
        maj_frac = max(n_yes, n_tse, n_no) / total if total else 0
        print(f"  {short:<4} {n_yes:>6} {n_tse:>14} {n_no:>6} {maj_frac:>14.3f}")
    print()

    # Bridge vs MathDial: do the two sources have different label profiles?
    if "source" in df.columns and df["source"].nunique() > 1:
        print("## Label distribution by source\n")
        for source in df["source"].unique():
            sub = df[df["source"] == source]
            print(f"  {source} (n={len(sub)})")
            for short in ("MI", "ML", "PG", "AC"):
                c = sub[f"label_{short}"].value_counts(normalize=True)
                print(f"    {short:<4} Yes={c.get('Yes',0):.2f} ToSomeExtent={c.get('To some extent',0):.2f} No={c.get('No',0):.2f}")
        print()

    # Per-tutor breakdown: do Expert and Novice (the human tutors) get different
    # label rates than the LLMs? 
    print("## Per-tutor 'Yes' rate (proxy for pedagogical-quality ranking)\n")
    rows = []
    for tutor, sub in df.groupby("tutor"):
        row = {"tutor": tutor, "n": len(sub)}
        for short in ("MI", "ML", "PG", "AC"):
            row[short] = (sub[f"label_{short}"] == "Yes").mean()
        row["avg_Yes"] = sum(row[s] for s in ("MI", "ML", "PG", "AC")) / 4
        rows.append(row)
    rows.sort(key=lambda r: -r["avg_Yes"])
    print(f"  {'tutor':<14} {'n':>5} {'MI':>6} {'ML':>6} {'PG':>6} {'AC':>6} {'avg-Yes':>8}")
    print("  " + "-" * 58)
    for r in rows:
        print(f"  {r['tutor']:<14} {r['n']:>5} "
              f"{r['MI']:>6.2f} {r['ML']:>6.2f} {r['PG']:>6.2f} {r['AC']:>6.2f} {r['avg_Yes']:>8.3f}")
    print()

if __name__ == "__main__":
    main()
