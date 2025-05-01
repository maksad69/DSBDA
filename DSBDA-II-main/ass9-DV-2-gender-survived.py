import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Titanic-Dataset.csv')


# Custom color palette for Survived (0 = No, 1 = Yes)
custom_palette = {0: "red", 1: "blue"}  # No = Red, Yes = Blue

#Plotting the boxplot distribution
plt.figure(figsize=(8, 5))
sns.boxplot(x='Sex', y='Age', hue='Survived', data=df, palette='Set2')

plt.xlabel("Gender")
plt.ylabel("Age")
plt.title("Age Distribution by Gender and Survival Status")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()
