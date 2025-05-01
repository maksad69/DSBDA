# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Step 2: Load the dataset
data = pd.read_csv('Iris.csv')

# Step 3: Check column names to identify any issues
print(data.columns)  # This will print all column names

# Remove leading/trailing whitespaces from column names
data.columns = data.columns.str.strip()

# Step 4: Data Preprocessing
# The last column is the target, 'species'
X = data.drop('species', axis=1)  # Features
y = data['species']  # Target

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Implement Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Step 7: Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Step 8: Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Extracting values from the confusion matrix
TP = cm[1, 1]  # True Positive
TN = cm[0, 0]  # True Negative
FP = cm[0, 1]  # False Positive
FN = cm[1, 0]  # False Negative

# Step 9: Compute Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' for multiclass classification
recall = recall_score(y_test, y_pred, average='weighted')  # 'weighted' for multiclass classification

# Step 10: Display the results
print("Confusion Matrix:")
print(cm)
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Accuracy: {accuracy}")
print(f"Error Rate: {error_rate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
