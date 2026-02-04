
#! wap in python to implement multi-liner regrateion

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Metrics:
    mae: float
    mse: float
    rmse: float
    r2: float


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_b = add_bias(X)
    # Closed-form: theta = (X^T X)^-1 X^T y
    theta = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y)
    return theta


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    X_b = add_bias(X)
    return X_b @ theta


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return Metrics(mae=mae, mse=mse, rmse=rmse, r2=r2)


def load_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    # Encode categorical column
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )

    feature_cols = [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ]
    target_col = "Performance Index"

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def print_metrics(title: str, m: Metrics) -> None:
    print(f"\n{title}")
    print(f"MAE : {m.mae:.4f}")
    print(f"MSE : {m.mse:.4f}")
    print(f"RMSE: {m.rmse:.4f}")
    print(f"R2  : {m.r2:.4f}")


def main() -> None:
    csv_path = "/home/tushardevx01/Documents/Machine learning/Day 4/Deta/Student_Performance.csv"
    X, y = load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    theta = fit_ols(X_train, y_train)
    y_pred = predict(X_test, theta)

    metrics = evaluate_metrics(y_test, y_pred)
    print_metrics("Multi-Linear Regression (OLS) Metrics", metrics)

    # Plot: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual Performance Index")
    plt.ylabel("Predicted Performance Index")
    plt.tight_layout()
    output_path = "/home/tushardevx01/Documents/Machine learning/Day 4/actual_vs_predicted.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()





