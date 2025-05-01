import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load dataset from CSV file
data = pd.read_csv('HousingData.csv')  # Replace with your CSV file path

# Step 2: Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Handle missing values by filling with the mean (or median, or other methods)
data.fillna(data.mean(), inplace=True)

# Check column names
print("Columns in the dataset:")
print(data.columns)

# Step 3: Separate features (X) and target (y)
# Assuming target column is 'MEDV' (replace with the actual target column name)
x = data.drop(['MEDV'], axis=1)  # Features
y = data['MEDV']  # Target variable

# Step 4: Split data into training and testing sets (80-20 split)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

# Step 5: Initialize and train the linear regression model
lm = LinearRegression()
model = lm.fit(xtrain, ytrain)

# Step 6: Predict the values for both training and test data
ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)

# Step 7: Evaluate the performance of the model using Mean Squared Error (MSE)
train_mse = mean_squared_error(ytrain, ytrain_pred)
test_mse = mean_squared_error(ytest, ytest_pred)

print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")

# Step 8: Plotting the results
plt.scatter(ytrain, ytrain_pred, c='blue', marker='o', label='Training data')
plt.scatter(ytest, ytest_pred, c='lightgreen', marker='s', label='Test data')

plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title("True values vs Predicted values")
plt.legend(loc='upper left')

plt.show()
