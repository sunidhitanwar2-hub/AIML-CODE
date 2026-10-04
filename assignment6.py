# Assignment 6
# Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes = load_diabetes()

# Use BMI as independent variable
X = diabetes.data[:, [2]]

# Target variable
y = diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Linear Regression")
print("------------------")

print("\nCoefficient:")
print(model.coef_[0])

print("\nIntercept:")
print(model.intercept_)

print("\nMean Squared Error:")
print(mse)

print("\nR2 Score:")
print(r2)

# Predict new value
new_bmi = [[0.05]]

prediction = model.predict(new_bmi)

print("\nPredicted Disease Progression:")
print(prediction[0])

# Visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    X_test,
    y_test,
    label="Actual"
)

plt.plot(
    X_test,
    y_pred,
    color="red",
    label="Regression Line"
)

plt.xlabel("BMI")
plt.ylabel("Disease Progression")

plt.title("Linear Regression")

plt.legend()

plt.grid(True)

plt.show()