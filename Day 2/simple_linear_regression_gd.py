# implement simple linear regression using gradient decent to estimate the slope and intercept .use a learning rate of 0.01 and run for 1000 iteration.

import numpy as np


def gradient_descent(x, y, learning_rate=0.01, iterations=1000):
	m = 0.0
	b = 0.0
	n = len(x)

	for _ in range(iterations):
		y_pred = m * x + b
		dm = (-2 / n) * np.sum(x * (y - y_pred))
		db = (-2 / n) * np.sum(y - y_pred)
		m -= learning_rate * dm
		b -= learning_rate * db

	return m, b


if __name__ == "__main__":
	x = np.array([1, 2, 3, 4, 5], dtype=float)
	y = np.array([3, 5, 7, 9, 11], dtype=float)

	slope, intercept = gradient_descent(x, y, learning_rate=0.01, iterations=1000)
	print(f"Estimated slope: {slope:.4f}")
	print(f"Estimated intercept: {intercept:.4f}")




