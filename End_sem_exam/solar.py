
# !A solar panel company tracks sunlight hours vs energy generated.

#! Train a gradient descent based linear regression model (without using any library function).

#! You need to:

#! Experiment with 3 learning rates:
#! 0.001
#! 0.01
#! 0.1
#! Plot and analyze convergence behavior


#! Dataset:


#! | Sunlight Hours | Energy (kWh) |
#! | -------------- | ------------ |
#! | 2              | 1.5          |
#! | 3              | 2.0          |
#! | 4              | 2.8          |
#! | 5              | 3.3          |
#! | 6              | 4.0          |
#! | 7              | 4.6          |


import matplotlib.pyplot as plt

X = [2, 3, 4, 5, 6, 7]
y = [1.5, 2.0, 2.8, 3.3, 4.0, 4.6]

def gd(x, y, lr, iters=1000):
    n = len(x)
    mu = sum(x) / n
    std = (sum((i - mu) ** 2 for i in x) / n) ** 0.5
    xs = [(i - mu) / std for i in x]

    m = b = 0.0
    cost = []
    for _ in range(iters):
        p = [m * i + b for i in xs]
        e = [p[i] - y[i] for i in range(n)]
        cost.append(sum(v * v for v in e) / n)
        m -= lr * (2 / n) * sum(e[i] * xs[i] for i in range(n))
        b -= lr * (2 / n) * sum(e)

    return m / std, b - m * mu / std, cost

rates = [0.001, 0.01, 0.1]
res = {lr: gd(X, y, lr) for lr in rates}

for lr in rates:
    m, b, c = res[lr]
    print(f"LR={lr}  m={m:.4f}  b={b:.4f}  final MSE={c[-1]:.6f}")

plt.figure(figsize=(11, 4))

plt.subplot(1, 2, 1)
for lr in rates:
    plt.plot(res[lr][2], label=f"LR={lr}")
plt.title("Convergence")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X, y, c="black", label="Data")
for lr in rates:
    m, b, _ = res[lr]
    plt.plot(X, [m * i + b for i in X], label=f"LR={lr}")
plt.title("Fitted Lines")
plt.xlabel("Sunlight Hours")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()