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
Y = [1.5, 2.0, 2.8, 3.3, 4.0, 4.6]

def fit(lr, it=1000):
    n = len(X); mu = sum(X)/n
    sd = (sum((x-mu)**2 for x in X)/n) ** 0.5
    Z = [(x-mu)/sd for x in X]
    m = b = 0.0; c = []
    for _ in range(it):
        e = [m*z + b - y for z, y in zip(Z, Y)]
        c.append(sum(v*v for v in e)/n)
        m -= lr * (2/n) * sum(v*z for v, z in zip(e, Z))
        b -= lr * (2/n) * sum(e)
    return m/sd, b - (m*mu/sd), c

rates = [0.001, 0.01, 0.1]
out = {r: fit(r) for r in rates}

for r in rates:
    m, b, c = out[r]
    print(f"LR={r}  m={m:.4f}  b={b:.4f}  MSE={c[-1]:.6f}")

for r in rates:
    plt.plot(out[r][2], label=f"LR={r}")
plt.title("Convergence")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

if "agg" in plt.get_backend().lower():
    plt.savefig("solar_convergence.png", dpi=300, bbox_inches="tight")
    print("Saved: solar_convergence.png")
else:
    plt.show()
