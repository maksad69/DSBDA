import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# ----------------------------------------
# Distribution Plots
# ----------------------------------------

# Displot - Distribution of Age
plt.figure(figsize=(15,5))
sns.displot(df['Age'], bins=10)
plt.title("Distribution of Age")
plt.show()

# Jointplot - Relationship between Age and Sex
sns.jointplot(x=df['Age'], y=df['Sex'], kind='scatter')
plt.suptitle("Joint Plot: Age vs Sex", y=1.02)
plt.show()

# ----------------------------------------
# Categorical Plots
# ----------------------------------------

# Barplot - Average Age by Sex
plt.figure(figsize=(7,5))
sns.barplot(x=df['Sex'], y=df['Age'])
plt.title("Average Age by Sex")
plt.show()

# Countplot - Number of Male and Female Passengers
plt.figure(figsize=(7,5))
sns.countplot(x=df['Sex'])
plt.title("Number of Male and Female Passengers")
plt.show()

# Boxplot - Distribution of Age by Sex
plt.figure(figsize=(7,5))
sns.boxplot(x=df['Sex'], y=df['Age'])
plt.title("Age Distribution by Sex (Boxplot)")
plt.show()

# Violinplot - Distribution of Age by Sex
plt.figure(figsize=(7,5))
sns.violinplot(x=df['Sex'], y=df['Age'])
plt.title("Age Distribution by Sex (Violinplot)")
plt.show()

# ----------------------------------------
# Matrix Plot
# ----------------------------------------

# Heatmap - Correlation Matrix
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# ----------------------------------------
# Advanced Plots
# ----------------------------------------

# Stripplot - Distribution of Age by Sex
plt.figure(figsize=(7,5))
sns.stripplot(x=df['Sex'], y=df['Age'])
plt.title("Strip Plot: Age Distribution by Sex")
plt.show()

# Swarmplot - Distribution of Age by Sex (non-overlapping)
plt.figure(figsize=(7,5))
sns.swarmplot(x=df['Sex'], y=df['Age'])
plt.title("Swarm Plot: Age Distribution by Sex")
plt.show()

# ----------------------------------------
# Second Question: Fare Distribution
# ----------------------------------------

# Histogram + KDE for Fare
plt.figure(figsize=(7,5))
sns.histplot(df['Fare'], kde=True, bins=30, color='blue')
plt.title("Distribution of Ticket Fare")
plt.ylabel('Number of Passengers')
plt.show()
