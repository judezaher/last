"""Offline test script for the doctor's CICIDS2017 model.

Use this before live IDS integration.
It proves that:
1. the .h5 model loads correctly,
2. the 72 CSV features match the model input,
3. the class mapping works,
4. model accuracy can be checked on the uploaded test CSV.

Example:
python offline_model_test.py \
  --model "models/CICIDS_baseline (2).h5" \
  --csv models/sample_test_CICIDS2017_1000.csv \
  --features models/cicids_feature_columns.json \
  --mapping models/Mapping \
  --rows 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cicids_model import CICIDSPureNumpyModel, load_cicids_csv_features, load_feature_columns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline evaluator for CICIDS2017 H5 model")
    parser.add_argument("--model", required=True, help="Path to CICIDS .h5 model")
    parser.add_argument("--csv", required=True, help="Path to normalized CICIDS CSV")
    parser.add_argument("--features", required=True, help="Feature JSON or training CSV to define feature order")
    parser.add_argument("--mapping", default=None, help="Path to Mapping file")
    parser.add_argument("--rows", type=int, default=1000, help="How many rows to test. Use 0 for all rows.")
    parser.add_argument("--show", type=int, default=10, help="Show first N predictions")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = None if args.rows == 0 else args.rows

    feature_columns = load_feature_columns(args.features)
    print(f"Loaded {len(feature_columns)} feature columns")
    if len(feature_columns) != 72:
        raise SystemExit(f"Expected 72 feature columns, got {len(feature_columns)}")

    model = CICIDSPureNumpyModel(args.model, args.mapping)
    x, y, df = load_cicids_csv_features(args.csv, feature_columns, rows=rows)
    print(f"Loaded CSV rows={x.shape[0]} features={x.shape[1]}")

    proba = model.predict_proba(x)
    pred = np.argmax(proba, axis=1)
    conf = np.max(proba, axis=1)

    if y is not None:
        accuracy = float(np.mean(pred == y))
        print(f"Accuracy: {accuracy:.4f}")
        labels = sorted(set(y.tolist()) | set(pred.tolist()))
        confusion = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred"), dropna=False)
        confusion = confusion.reindex(index=labels, columns=labels, fill_value=0)
        print("\nConfusion matrix by class id:")
        print(confusion.to_string())

    print("\nSample predictions:")
    id_to_label = model.id_to_label
    for i in range(min(args.show, len(pred))):
        true_text = ""
        if y is not None:
            true_text = f" true={int(y[i])}:{id_to_label.get(int(y[i]), 'unknown')}"
        print(
            f"row={i} pred={int(pred[i])}:{id_to_label.get(int(pred[i]), 'unknown')} "
            f"confidence={float(conf[i]):.4f}{true_text}"
        )


if __name__ == "__main__":
    main()
