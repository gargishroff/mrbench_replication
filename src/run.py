from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data_loader
import prompts
import inference
import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../mrbench/BEA_Shared_Task_2025_Datasets/mrbench_v3_devset.json",
                   help="Path to MRBench dev/test JSON")
    p.add_argument("--backend", choices=["hf", "vllm", "anthropic", "mock"], default="mock")
    p.add_argument("--model", default=None, help="Model name (required for hf/vllm/anthropic)")
    p.add_argument("--mock", choices=["always_yes", "always_no", "random", "oracle"], default="always_yes",
                   help="Mock-backend mode (only used if --backend mock)")
    p.add_argument("--n", type=int, default=None, help="Subsample to N responses (stratified by tutor + source)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="../results")
    args = p.parse_args()

    print(f"Loading data from {args.data}")
    df = data_loader.load_mrbench(args.data)
    print(f"  loaded {len(df)} labelled responses from {df['conversation_id'].nunique()} conversations")

    if args.n and args.n < len(df):
        n_per_tutor = max(1, args.n // df["tutor"].nunique())
        parts = []
        for tutor, g in df.groupby("tutor"):
            parts.append(g.sample(min(len(g), n_per_tutor), random_state=args.seed))
        df = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        print(f"  subsampled to {len(df)} (stratified by tutor)")

    if args.backend == "hf":
        assert args.model, "--model required for hf backend"
        backend = inference.HFBackend(model_name=args.model)
    elif args.backend == "vllm":
        assert args.model, "--model required for vllm backend"
        backend = inference.VLLMBackend(model_name=args.model)
    else:
        raise ValueError(args.backend)

    print(f"Backend: {backend.name}")

    # Run
    safe_name = backend.name.replace("/", "_")
    pred_path = Path(args.output_dir) / f"preds_{safe_name}_n{len(df)}.jsonl"
    inference.run_inference(df, backend, prompts, str(pred_path))

    # Evaluate
    pred_df = evaluate.load_predictions(str(pred_path))
    results = evaluate.evaluate(pred_df)
    print(evaluate.render_comparison_table(results, backend.name))
    print(evaluate.render_confusion_matrices(results))

    import json
    out_json = pred_path.with_suffix(".eval.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation summary -> {out_json}")


if __name__ == "__main__":
    main()
