# Assignment 7
# Support Vector Regression (SVR)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes = load_diabetes()

# Use BMI as independent variable
X = diabetes.data[:, [2]]

y = diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Create SVR model
model = SVR(
    kernel="rbf",
    C=100,
    gamma="scale",
    epsilon=0.1
)

# Train
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("Support Vector Regression")
print("-------------------------")

print("\nMean Squared Error:")
print(mse)

print("\nR2 Score:")
print(r2)

# Predict new BMI
new_bmi = [[0.05]]

new_bmi_scaled = scaler_X.transform(new_bmi)

prediction = model.predict(new_bmi_scaled)

print("\nPredicted value:")
print(prediction[0])

# Visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    X_test,
    y_test,
    label="Actual"
)

# Sort values for proper line
sort_index = np.argsort(X_test[:, 0])

plt.plot(
    X_test[sort_index],
    y_pred[sort_index],
    color="red",
    label="SVR Prediction"
)

plt.xlabel("BMI")
plt.ylabel("Disease Progression")

plt.title("Support Vector Regression")

plt.legend()

plt.grid(True)

plt.show()