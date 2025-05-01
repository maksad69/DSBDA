import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Iris.csv')
#Getting first values
print(df.head())
#Getting last values
print(df.tail())
#Describing the dataset
print(df.describe())

#Assigning the column values
print("-----------Statistical Operations on iris dataset-----------")
# col_names = ['SepalWidth', 'PetalLength', 'PetalWidth','Species']
irisSet = (df['Species']== 'Iris-setosa')
print('Iris-setosa')
print(df[irisSet].describe())

#Display the statistical details
irisVer = (df['Species']== 'Iris-versicolor')
print('Iris-versicolor')
print(df[irisVer].describe())

#Standard Deviation
print(df['SepalLength'].std())
