from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clinical_static_baseline_benchmark.metrics import evaluate_reconstruction, save_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted metabolomics against ground truth.")
    parser.add_argument("--truth", required=True, help="CSV file containing ground-truth metabolomics.")
    parser.add_argument("--prediction", required=True, help="CSV file containing predicted metabolomics.")
    parser.add_argument("--output", required=True, help="Path to metrics.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = pd.read_csv(args.truth, index_col=0).to_numpy(dtype="float32")
    pred = pd.read_csv(args.prediction, index_col=0).to_numpy(dtype="float32")
    metrics = evaluate_reconstruction(truth, pred)
    save_metrics(metrics, Path(args.output))
    print(metrics)


if __name__ == "__main__":
    main()
