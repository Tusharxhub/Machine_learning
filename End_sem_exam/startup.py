#
# ! A startup wants to predict the monthly electricity consumption of its office based on total working hours logged by employees.

# !Write a Python program (without using any inbuilt ML functions) to construct an OLS-based simple linear regression model.

#! You need to:

#! Compute slope
#! Compute intercept
#! Visually plot the best-fit line over the dataset



#! Dataset:

#! | Working Hours | Electricity Consumption (kWh) |
#! | ------------- | ----------------------------- |
#! | 5             | 12                            |
#! | 6             | 15                            |
#! | 8             | 20                            |
#! | 10            | 25                            |
#! | 12            | 30                            |
#! | 15            | 38                            |



import matplotlib.pyplot as plt
import matplotlib

x = [5, 6, 8, 10, 12, 15]
y = [12, 15, 20, 25, 30, 38]
n = len(x)

sx, sy = sum(x), sum(y)
sxy = sum(a * b for a, b in zip(x, y))
sx2 = sum(a * a for a in x)

m = (n * sxy - sx * sy) / (n * sx2 - sx * sx)   
b = (sy - m * sx) / n                            

print(f"Slope: {m:.4f}")
print(f"Intercept: {b:.4f}")
print(f"Equation: y = {m:.4f}x + {b:.4f}")

plt.scatter(x, y, label="Data")
plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], "r", label="Best-fit line")
plt.xlabel("Working Hours")
plt.ylabel("Electricity Consumption (kWh)")
plt.legend()
plt.grid(alpha=0.3)

if "agg" in matplotlib.get_backend().lower():
    plt.savefig("startup_best_fit.png", dpi=300, bbox_inches="tight")
    print("Plot saved as startup_best_fit.png")
else:
    plt.show()