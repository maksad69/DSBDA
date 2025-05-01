# Data Visualization II
# 1. Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for distribution
# of age with respect to each gender along with the information about whether they survived
# or not. (Column names : 'sex' and 'age')
# 2. Write observations on the inference from the above statistics.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datas = sns.load_dataset("titanic")
print(datas.head())
print(datas.info)
print(datas.isnull().sum())

# Drop rows with missing age values first
# datas = datas.dropna(subset=['age'])
datas['age'] = datas['age'].fillna(datas['age'].mean())
print(datas.isnull().sum())

# Calculate Q1 and Q3
Q1 = datas['age'].quantile(0.25)
Q3 = datas['age'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset to remove outliers
filtered_data = datas[(datas['age'] >= lower_bound) & (datas['age'] <= upper_bound)]

# boxplot to show wheather people are survived or not
plt.figure(figsize=(10,6))
sns.boxplot(data=filtered_data, x='sex',y='age',hue='survived')

# Add title and labels
plt.title('Age Distribution by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()