# Assignment 8
# Regularization to Avoid Overfitting

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Standardize data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)

lasso_model.fit(X_train, y_train)

lasso_pred = lasso_model.predict(X_test)

# Calculate scores
linear_mse = mean_squared_error(y_test, linear_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)

linear_r2 = r2_score(y_test, linear_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print("Regularization")
print("==============")

print("\nLinear Regression")
print("MSE:", linear_mse)
print("R2:", linear_r2)

print("\nRidge Regression")
print("MSE:", ridge_mse)
print("R2:", ridge_r2)

print("\nLasso Regression")
print("MSE:", lasso_mse)
print("R2:", lasso_r2)

# Compare coefficients
comparison = pd.DataFrame({
    "Feature": diabetes.feature_names,
    "Linear": linear_model.coef_,
    "Ridge": ridge_model.coef_,
    "Lasso": lasso_model.coef_
})

print("\nCoefficient Comparison:")
print(comparison)

# Plot coefficients
comparison.set_index("Feature").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Comparison of Model Coefficients")

plt.xlabel("Features")

plt.ylabel("Coefficient Value")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.show()