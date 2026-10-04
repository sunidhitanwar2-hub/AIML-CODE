# Assignment 1
# Identify the association between dependent and independent variables

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes

# Load dataset
diabetes = load_diabetes()

# Convert dataset into DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Add target variable
df["target"] = diabetes.target

print("First five records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# Independent variable
X = df["bmi"]

# Dependent variable
Y = df["target"]

# Calculate correlation
correlation = X.corr(Y)

print("\nCorrelation between BMI and Disease Progression:")
print(correlation)

# Interpretation
if correlation > 0:
    print("There is a positive association between BMI and disease progression.")
elif correlation < 0:
    print("There is a negative association between BMI and disease progression.")
else:
    print("There is no linear association.")

# Scatter plot
plt.figure(figsize=(8, 5))

sns.scatterplot(x=X, y=Y)

plt.title("Association Between BMI and Disease Progression")
plt.xlabel("BMI - Independent Variable")
plt.ylabel("Disease Progression - Dependent Variable")

plt.grid(True)
plt.show()

# Heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.show()