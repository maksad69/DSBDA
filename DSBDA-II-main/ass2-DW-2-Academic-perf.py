import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Reading the dataset
df = pd.read_csv("StudentsPerformance.csv")

# Replacing the null values
print("----------Replacing the null values--------")
print(df.isnull().sum())  # Check the count of null values

# Replace missing values in 'math score' with the mean
df['math score'].fillna(df['math score'].mean(), inplace=True)
print(df['math score'])

# Replace missing values in 'reading score' with the mean
df['reading score'].fillna(df['reading score'].mean(), inplace=True)
print(df['reading score'])

# Replace missing values in 'writing score' with the mean
df['writing score'].fillna(df['writing score'].mean(), inplace=True)
print(df['writing score'])

# Verifying if any null values remain
print(df.isnull().sum())  # Check again after replacement

# Visualization - BoxPlot for Outlier Detection
print("-----------Identification and Handling of Outliers----------")
plt.figure(figsize=(10,5))
sns.boxplot(data=df[['math score', 'reading score', 'writing score']])
plt.title("Student Performance")
plt.show()

# Scatterplot between math score and reading score
fig, ax = plt.subplots(figsize=(18,10))
ax.scatter(df['math score'], df['reading score'])
ax.set_xlabel('Math Score')
ax.set_ylabel('Reading Score')
plt.show()

# Z-Score Method for Outliers Detection (Math Score)
df['math score'] = pd.to_numeric(df['math score'], errors='coerce')  # Ensure numeric
z_scores = np.abs(stats.zscore(df['math score']))
print(f"Z-Scores:\n{z_scores}")

# IQR Method for Outlier Detection (Math Score)
Q1 = np.percentile(df['math score'], 25)
Q3 = np.percentile(df['math score'], 75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# Identify outliers
outliers_upper = df['math score'] > upper_bound
outliers_lower = df['math score'] < lower_bound

print("Outliers above the upper bound:", np.where(outliers_upper))
print("Outliers below the lower bound:", np.where(outliers_lower))

# Optionally, remove outliers (if required)
df_no_outliers = df[~(outliers_upper | outliers_lower)]
print(f"Data shape without outliers: {df_no_outliers.shape}")
