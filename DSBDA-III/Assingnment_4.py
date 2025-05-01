# Data Analytics I
# Create a Linear Regression Model using Python/R to predict home prices using Boston Housing
# Dataset (https://www.kaggle.com/c/boston-housing). The Boston Housing dataset contains
# information about various houses in Boston through different parameters. There are 506 samples
# and 14 feature variables in this dataset.
# The objective is to predict the value of prices of the house using the given features.

# step 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# step 2
data = pd.read_csv("BostonHousing.csv")
print(data)

# step 3
print(data.isnull().sum())
print(data.describe())
print(data.shape)
print(data.dtypes)

# step 4 :- Correlation Heatmap (to check which features affect price)
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# step 5 :- Prepare Data
# Features and target
X = data.drop("medv", axis=1)   # 'medv' is the median value of owner-occupied homes (target)
y = data["medv"]

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# step 6 :- Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# step 7 :-
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# step 8 :- Visualize Actual vs Predicted Prices
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red', linewidth=2)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()