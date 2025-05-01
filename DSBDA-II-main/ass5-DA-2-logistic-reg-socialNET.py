# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Step 2: Load the dataset
data = pd.read_csv('Social_Network_Ads.csv')

# Preview the data
print(data.head())

# Step 3: Data Preprocessing
# Ploting Scatterplot
sns.scatterplot(x='Age', y='EstimatedSalary', data=data, palette='coolwarm', hue='Age')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.title('Purchased details')
plt.show()
# Let's assume the dataset has columns 'Age', 'EstimatedSalary', and 'Purchased' as the target
# We'll split the data into features and target

X = data[['Age', 'EstimatedSalary']]  # Features
y = data['Purchased']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Implement Logistic Regression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Step 5: Predict on the test set
y_pred = classifier.predict(X_test)

# Printing Intercept and Coefficient
classifier.intercept_, classifier.coef_
classifier.score(X_train, y_train)

# Step 6: Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Extracting the values from the confusion matrix
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

# Step 7: Compute Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Display the results
print("Confusion Matrix:")
print(cm)
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Accuracy: {accuracy}")
print(f"Error Rate: {error_rate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
