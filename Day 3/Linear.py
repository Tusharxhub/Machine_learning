# !Write python program to implement Simple Linear Regression from scratch using two 
#! different optimization methods—Ordinary Least Squares (Closed-form) and Gradient Descent 
#! (Iterative)—and evaluate their performance using five key regression metrics.  
#! Part 1: Data Generation  
#! Generate a synthetic dataset with a linear relationship and some added noise:  
#! ▪ x values: 100 random points between 0 and 50.  
#! ▪ y values: y = 3x + 7 + noise (where noise is a random Gaussian distribution).  
#! Part 2: Implementation (No Libraries)  
#! You are prohibited from using scikit-learn for the model or metrics. You may use NumPy for 
#! array operations and Matplotlib for plotting.  
#! ▪ Function 1: fit_ols(x, y) – Returns m and c using the closed-form equations. 
#! ▪ Function 2: fit_gd(x, y, learning_rate, epochs) – Returns m and c after n iterations.  
#! ▪ Function 3: evaluate_metrics(y_true, y_pred, n_features) – Returns a dictionary of the 
#! five metrics.  
#! Part 3: The Metrics Suite Calculate the following for both models: MAE, MSE, RMSE, R2, 
#! Adjusted R2 score.



from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def generate_data(
	n_points: int = 100,
	x_min: float = 0.0,
	x_max: float = 50.0,
	m_true: float = 3.0,
	c_true: float = 7.0,
	noise_std: float = 5.0,
	seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
	rng = np.random.default_rng(seed)
	x = rng.uniform(x_min, x_max, size=n_points)
	noise = rng.normal(0.0, noise_std, size=n_points)
	y = m_true * x + c_true + noise
	return x, y


def fit_ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
	x_mean = np.mean(x)
	y_mean = np.mean(y)
	numerator = np.sum((x - x_mean) * (y - y_mean))
	denominator = np.sum((x - x_mean) ** 2)
	if denominator == 0:
		raise ValueError("Cannot fit OLS when variance of x is zero.")
	m = numerator / denominator
	c = y_mean - m * x_mean
	return m, c


def fit_gd(
	x: np.ndarray,
	y: np.ndarray,
	learning_rate: float = 0.0005,
	epochs: int = 5000,
) -> Tuple[float, float]:
	n = len(x)
	m = 0.0
	c = 0.0
	for _ in range(epochs):
		y_pred = m * x + c
		dm = (-2 / n) * np.sum(x * (y - y_pred))
		dc = (-2 / n) * np.sum(y - y_pred)
		m -= learning_rate * dm
		c -= learning_rate * dc
	return m, c


def evaluate_metrics(
	y_true: np.ndarray, y_pred: np.ndarray, n_features: int
) -> Dict[str, float]:
	n = len(y_true)
	if n == 0:
		raise ValueError("y_true is empty.")

	errors = y_true - y_pred
	mae = np.mean(np.abs(errors))
	mse = np.mean(errors**2)
	rmse = np.sqrt(mse)
	ss_res = np.sum(errors**2)
	ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
	r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
	adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

	return {
		"MAE": float(mae),
		"MSE": float(mse),
		"RMSE": float(rmse),
		"R2": float(r2),
		"Adjusted R2": float(adjusted_r2),
	}


def predict(x: np.ndarray, m: float, c: float) -> np.ndarray:
	return m * x + c


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
	print(f"\n{title}")
	for key, value in metrics.items():
		print(f"{key}: {value:.4f}")


def main() -> None:
	x, y = generate_data()

	m_ols, c_ols = fit_ols(x, y)
	y_pred_ols = predict(x, m_ols, c_ols)
	metrics_ols = evaluate_metrics(y, y_pred_ols, n_features=1)

	m_gd, c_gd = fit_gd(x, y)
	y_pred_gd = predict(x, m_gd, c_gd)
	metrics_gd = evaluate_metrics(y, y_pred_gd, n_features=1)

	print_metrics("OLS Metrics", metrics_ols)
	print_metrics("Gradient Descent Metrics", metrics_gd)

	plt.figure(figsize=(8, 5))
	plt.scatter(x, y, label="Data", alpha=0.6)
	x_line = np.linspace(np.min(x), np.max(x), 200)
	plt.plot(x_line, predict(x_line, m_ols, c_ols), label="OLS", linewidth=2)
	plt.plot(x_line, predict(x_line, m_gd, c_gd), label="Gradient Descent", linewidth=2)
	plt.title("Simple Linear Regression: OLS vs Gradient Descent")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend()
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()

