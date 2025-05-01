# Data Visualization I
# 1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information
# about the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to
# see if we can find any patterns in the data.
# 2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger
# is distributed by plotting a histogram.

# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset("titanic")
print(data)

print(data.info)
print(data.dtypes)

# histogram
plt.figure(figsize=(10,6))
sns.histplot(data=data,x='fare', kde=True, bins=30, color='skyblue')

plt.title('Distribution of Ticket Fares')
plt.xlabel('Fare')
plt.ylabel('Number of Passengers')
plt.show()

# Plot survival count based on gender
sns.countplot(data=data, x='sex', hue='survived')
plt.title('Survival Count by Gender')
plt.show()

# Boxplot of fare by class
sns.boxplot(data=data, x='class', y='fare')
plt.title('Fare Distribution by Class')
plt.show()