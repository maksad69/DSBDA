import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv('Social_Network_Ads (1).csv')
print(df.head())

# Step 2: Encode Gender (categorical to numeric)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male:1, Female:0
print("Gender encoding:", df['Gender'].unique())

# Step 3: Split features and label
x = df.drop(columns=['User ID', 'Purchased'])  # Features
y = df['Purchased']                           # Target

# Step 4: Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)

# Step 5: Feature Scaling using MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  # Don't fit again on test!

# Step 6: Train logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Step 7: Predict
y_pred = model.predict(x_test)

# Optional: Scatter plot (less useful for classification but okay to visualize)
plt.scatter(y_test, y_pred)
plt.title("y_test vs y_pred")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Extract TP, TN, FP, FN
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print(f"TP: {TP}")
print(f"TN: {TN}")
print(f"FP: {FP}")
print(f"FN: {FN}")

# Plot confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix")
plt.show()

# Step 9: Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print(f"Accuracy: {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

#Accuracy measures how many predictions are correct in classification,
# while R² measures how well a regression model fits the actual data.
