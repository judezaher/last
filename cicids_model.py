"""CICIDS2017 Keras/H5 model helper for Phase 4.

This file is made for the doctor's uploaded model:
- input size: 72 numeric CICIDS2017 flow features
- output size: 9 classes
- architecture: Dense(128, relu) -> BatchNorm -> Dropout(no-op at inference)
  -> Dense(32, relu) -> Dense(9, softmax)

Why pure NumPy?
The .h5 file is a Keras model, but installing TensorFlow on Kali can be heavy.
For this exact architecture, we can read the weights with h5py and run inference
with NumPy. This is easier for a student lab and still uses the real model
weights.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


DEFAULT_CLASS_MAP = {
    "BENIGN": 0,
    "DoS Hulk": 1,
    "PortScan": 2,
    "DDoS": 3,
    "DoS GoldenEye": 4,
    "FTP-Patator": 5,
    "SSH-Patator": 6,
    "DoS slowloris": 7,
    "DoS Slowhttptest": 8,
}


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def load_class_mapping(mapping_path: str | Path | None) -> dict[int, str]:
    """Load class mapping from the provided Mapping text file.

    The Mapping file contains something like:
        di= {'BENIGN': 0, 'DoS Hulk': 1, ...}

    Return value maps numeric id -> readable label.
    """
    if mapping_path is None:
        return {v: k for k, v in DEFAULT_CLASS_MAP.items()}

    path = Path(mapping_path)
    if not path.exists():
        return {v: k for k, v in DEFAULT_CLASS_MAP.items()}

    text = path.read_text(errors="replace")
    match = re.search(r"di\s*=\s*(\{.*?\})", text, flags=re.DOTALL)
    if not match:
        return {v: k for k, v in DEFAULT_CLASS_MAP.items()}

    parsed = ast.literal_eval(match.group(1))
    return {int(v): str(k) for k, v in parsed.items()}


@dataclass
class PredictionResult:
    predicted_id: int
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


class CICIDSPureNumpyModel:
    """Run the uploaded Keras .h5 model using NumPy only."""

    def __init__(self, model_path: str | Path, mapping_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.id_to_label = load_class_mapping(mapping_path)
        self.input_features = 72
        self.output_classes = 9
        self._load_weights()

    def _load_weights(self) -> None:
        with h5py.File(self.model_path, "r") as f:
            cfg = json.loads(f.attrs["model_config"])
            self._validate_architecture(cfg)

            weights = f["model_weights"]
            self.dense0_kernel = np.asarray(weights["dense"]["dense"]["kernel:0"], dtype=np.float32)
            self.dense0_bias = np.asarray(weights["dense"]["dense"]["bias:0"], dtype=np.float32)

            bn = weights["batch_normalization"]["batch_normalization"]
            self.bn_beta = np.asarray(bn["beta:0"], dtype=np.float32)
            self.bn_gamma = np.asarray(bn["gamma:0"], dtype=np.float32)
            self.bn_mean = np.asarray(bn["moving_mean:0"], dtype=np.float32)
            self.bn_var = np.asarray(bn["moving_variance:0"], dtype=np.float32)
            self.bn_epsilon = self._batch_norm_epsilon(cfg)

            self.dense1_kernel = np.asarray(weights["dense_1"]["dense_1"]["kernel:0"], dtype=np.float32)
            self.dense1_bias = np.asarray(weights["dense_1"]["dense_1"]["bias:0"], dtype=np.float32)

            self.dense2_kernel = np.asarray(weights["dense_2"]["dense_2"]["kernel:0"], dtype=np.float32)
            self.dense2_bias = np.asarray(weights["dense_2"]["dense_2"]["bias:0"], dtype=np.float32)

    @staticmethod
    def _batch_norm_epsilon(config: dict) -> float:
        for layer in config["config"]["layers"]:
            if layer.get("class_name") == "BatchNormalization":
                return float(layer["config"].get("epsilon", 0.001))
        return 0.001

    @staticmethod
    def _validate_architecture(config: dict) -> None:
        layers = config["config"]["layers"]
        input_layers = [layer for layer in layers if layer.get("class_name") == "InputLayer"]
        if not input_layers:
            raise ValueError("Unsupported model: no InputLayer found")
        input_shape = input_layers[0]["config"].get("batch_input_shape")
        if input_shape != [None, 72]:
            raise ValueError(f"Unsupported input shape {input_shape}; expected [None, 72]")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for x shaped (n_rows, 72)."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.input_features:
            raise ValueError(f"Expected 72 features, got shape {x.shape}")

        z = relu(x @ self.dense0_kernel + self.dense0_bias)
        z = self.bn_gamma * ((z - self.bn_mean) / np.sqrt(self.bn_var + self.bn_epsilon)) + self.bn_beta
        # Dropout is not used during inference.
        z = relu(z @ self.dense1_kernel + self.dense1_bias)
        logits = z @ self.dense2_kernel + self.dense2_bias
        return softmax(logits)

    def predict(self, x: np.ndarray) -> list[PredictionResult]:
        proba = self.predict_proba(x)
        results: list[PredictionResult] = []
        for row in proba:
            predicted_id = int(np.argmax(row))
            label = self.id_to_label.get(predicted_id, f"class_{predicted_id}")
            probabilities = {
                self.id_to_label.get(i, f"class_{i}"): float(value)
                for i, value in enumerate(row)
            }
            results.append(
                PredictionResult(
                    predicted_id=predicted_id,
                    predicted_label=label,
                    confidence=float(row[predicted_id]),
                    probabilities=probabilities,
                )
            )
        return results


def load_feature_columns(path: str | Path) -> list[str]:
    """Load the 72 feature column names from a JSON file or a training CSV."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return list(json.loads(p.read_text()))

    cols = list(pd.read_csv(p, nrows=0).columns)
    if "Classification" in cols:
        cols.remove("Classification")
    return cols


def load_cicids_csv_features(csv_path: str | Path, feature_columns: Iterable[str], rows: int | None = None) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """Load model-ready CICIDS CSV features.

    This expects the CSV to already be preprocessed/normalized like the uploaded
    train/test NumericFS files.
    """
    df = pd.read_csv(csv_path, nrows=rows)
    feature_columns = list(feature_columns)
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing[:10]}")

    x = df[feature_columns].astype("float32").to_numpy()
    y = df["Classification"].astype(int).to_numpy() if "Classification" in df.columns else None
    return x, y, df
