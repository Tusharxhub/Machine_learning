
#! A startup wants to predict the monthly electricity consumption of its office based on total working hours logged by employees.

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



import math

working_hours = [5, 6, 8, 10, 12, 15]
electricity_consumption = [12, 15, 20, 25, 30, 38]

n = len(working_hours)
sum_x = sum(working_hours)
sum_y = sum(electricity_consumption)
sum_xy = sum(x * y for x, y in zip(working_hours, electricity_consumption))
sum_x_squared = sum(x ** 2 for x in working_hours)



slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
intercept = (sum_y - slope * sum_x) / n

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"\nLinear Regression Equation: y = {slope:.4f}x + {intercept:.4f}")

import matplotlib.pyplot as plt

y_predicted = [slope * x + intercept for x in working_hours]

plt.figure(figsize=(10, 6))
plt.scatter(working_hours, electricity_consumption, color='blue', label='Actual Data', s=100)
plt.plot(working_hours, y_predicted, color='red', label='Best-Fit Line', linewidth=2)
plt.xlabel('Working Hours')
plt.ylabel('Electricity Consumption (kWh)')
plt.title('Linear Regression: Electricity Consumption vs Working Hours')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
