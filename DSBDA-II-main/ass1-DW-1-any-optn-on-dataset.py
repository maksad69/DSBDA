import pandas as pd
import numpy as np

df = pd.read_csv("Titanic-Dataset.csv")

#Various function of the pandas to print the data in different form
print("--------Various function of the pandas to print the data in different form--------")
print(df.isnull().sum())
print(df.info())
print(df.describe())
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)


#converting a categorical variable (string-based) into a quantitative variable (integer-based)
print("---------categorical variable (string-based) into a quantitative variable (integer-based)-------")
#method 1
print(df['Sex'].replace(['male','female'],[0,1]))
print(df['Embarked'].replace(['S','C','Q'],[0,1,2]))

#method 2
df['Emb_cat']=df['Embarked'].astype('category')
print(df.head())
print(df['Emb_cat'].cat.codes)

#Replacing the null values
#Method 1- Replacing the null values by their respective values
print("--------Replacing the null values-------")
print(df.isnull().sum())
print(df.isnull())

#For the Age column
df['Age'].fillna('NA', inplace=True)
print(df["Age"])

#For the Cabin column
df['Cabin'].fillna('NA', inplace=True)
print(df["Cabin"])

#For the Embarked Column
df['Embarked'].fillna('NA', inplace=True)
print(df["Embarked"])

#Method 2-Deleting the rows containing null values
print(df.isnull().sum())
print(df.isnull())
print(df.dropna())
print(df.isnull().sum())

