# implement gradient decent algo for simple linear regression to find optimal m and c this involves i)initialise m and c ii) iteratively cost function with respect to m and c and specify learning rate iii)running the iteration for a fix number of epocs .Kindly note the avobe program need to be implemented without any inbuild function









# Gradient Descent for Simple Linear Regression
# Model: y = m*x + c
# No inbuilt ML libraries used

# Sample dataset
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Number of data points
n = len(x)

# Step 1: Initialize m and c
m = 0.0
c = 0.0

# Step 2: Learning rate and epochs
learning_rate = 0.01
epochs = 1000

# Step 3: Gradient Descent loop
for epoch in range(epochs):

    dm = 0.0
    dc = 0.0
    cost = 0.0

    for i in range(n):
        # Predicted value
        y_pred = m * x[i] + c


        # Error
        error = y[i] - y_pred

        # Gradients
        dm += -2 * x[i] * error
        dc += -2 * error

        # Cost
        cost += error * error

    # Average gradients and cost
    dm = dm / n
    dc = dc / n
    cost = cost / n

    # Update parameters
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    # Print every 100 epochs
    if epoch % 100 == 0:
        print("Epoch:", epoch, "Cost:", cost, "m:", m, "c:", c)

# Final output
print("\nFinal Values:")
print("Slope (m):", m)
print("Intercept (c):", c)
