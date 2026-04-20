
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

sunlight_hours = [2, 3, 4, 5, 6, 7]
energy_kwh = [1.5, 2.0, 2.8, 3.3, 4.0, 4.6]

n = len(sunlight_hours)

def gradient_descent(X, y, learning_rate, iterations=1000):
    """
    Implement gradient descent from scratch
    y = mx + b
    """
    x_mean = sum(X) / n
    x_variance = sum((x - x_mean) ** 2 for x in X) / n
    x_std = x_variance ** 0.5
    X_scaled = [(x - x_mean) / x_std for x in X]

    m = 0.0
    b = 0.0
    cost_history = []
    
    for iteration in range(iterations):
        y_pred = [m * x_scaled + b for x_scaled in X_scaled]
        
        mse = sum((pred - actual) ** 2 for pred, actual in zip(y_pred, y)) / n
        cost_history.append(mse)
        
        dm = sum(2 * (y_pred[i] - y[i]) * X_scaled[i] for i in range(n)) / n
        db = sum(2 * (y_pred[i] - y[i]) for i in range(n)) / n
        
        m = m - learning_rate * dm
        b = b - learning_rate * db
    
    m_original = m / x_std
    b_original = b - (m * x_mean / x_std)

    return m_original, b_original, cost_history

learning_rates = [0.001, 0.01, 0.1]
results = {}

print("=" * 60)
print("GRADIENT DESCENT RESULTS WITH DIFFERENT LEARNING RATES")
print("=" * 60)

for lr in learning_rates:
    m, b, cost_history = gradient_descent(sunlight_hours, energy_kwh, lr, iterations=1000)
    results[lr] = {'m': m, 'b': b, 'cost_history': cost_history}
    print(f"\nLearning Rate: {lr}")
    print(f"Slope (m): {m:.6f}")
    print(f"Intercept (b): {b:.6f}")
    print(f"Final MSE: {cost_history[-1]:.6f}")
    print(f"Regression Equation: y = {m:.6f}x + {b:.6f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for lr in learning_rates:
    plt.plot(results[lr]['cost_history'], label=f'LR = {lr}', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Convergence Behavior for Different Learning Rates')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
for lr in learning_rates:
    plt.plot(results[lr]['cost_history'][:100], label=f'LR = {lr}', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Initial Convergence (First 100 Iterations)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
colors = ['blue', 'green', 'red']
for lr, color in zip(learning_rates, colors):
    m, b = results[lr]['m'], results[lr]['b']
    y_pred = [m * x + b for x in sunlight_hours]
    plt.plot(sunlight_hours, y_pred, color=color, label=f'LR = {lr}', linewidth=2)

plt.scatter(sunlight_hours, energy_kwh, color='black', s=100, label='Actual Data', zorder=5)
plt.xlabel('Sunlight Hours')
plt.ylabel('Energy (kWh)')
plt.title('Regression Lines for Different Learning Rates')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("CONVERGENCE ANALYSIS")
print("=" * 60)
print("\nLearning Rate 0.001:")
print("  - Slowest convergence (more iterations needed)")
print("  - Very stable, no oscillation")
print("  - Best for careful training")

print("\nLearning Rate 0.01:")
print("  - Moderate convergence speed")
print("  - Good balance between speed and stability")
print("  - Generally a good default")

print("\nLearning Rate 0.1:")
print("  - Fastest convergence initially")
print("  - May oscillate or diverge if too high")
print("  - Risk of overshooting the minimum")