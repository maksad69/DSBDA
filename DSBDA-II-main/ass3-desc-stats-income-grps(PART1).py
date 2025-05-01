import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("supermarket_sales.csv")
print("---------Basic Statistics------------")
#Printing the first rows
print(df.head())
#Printing the last rows
print(df.tail())
#Printing no of rows and column
print(df.shape)
#Checking null values if present
print(df.isnull().sum())

#Counting the no of rows from each column
print(df.count())

#Getting sum of the entries of each column
print(df.sum())

#Sum of total sales and gross income
print(df[['Total','gross income']].sum())

#Getting the mean of the Rating column
print(df['Rating'].mean())

#Getting the Standard Deviation of the Rating column
print(df['Rating'].std())

#Getting the mode of the Payment column
print('The Mode of : ',df['Payment'].mode())

#Printing the minimum and maximum quantity sold
print(df['Quantity'].min())
print(df['Quantity'].max())

#Median
median = df['Quantity'].median()
print(median)

#Varience
var = df['Quantity'].var()
print(var)

#Group by on City column
print("----------Group by function-------------")
groupby_sum = df.groupby(['City']).sum()
print(groupby_sum)
groupby_count = df.groupby(['City']).count()
print(groupby_count)

#Group by on Payment column
groupby_sum1 = df.groupby(['Payment']).sum()
print(groupby_sum)
groupby_count1 = df.groupby(['Payment']).count()
print(groupby_count1)


