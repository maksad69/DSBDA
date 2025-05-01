import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('Titanic-Dataset.csv')
#Displaying the price of the ticket (column name: 'Fare') for each passenger is distributed by plotting a histogram.
sns.histplot(df['Fare'], bins=10, kde=True)
plt.xlabel('Fare')
plt.ylabel('Count')
plt.title('Histogram of Fare')
plt.show()
