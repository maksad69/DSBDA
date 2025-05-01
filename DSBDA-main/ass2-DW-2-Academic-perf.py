import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from  sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('StudentsPerformance.csv')
print(df)
print(df.isnull().sum())

#outliers
plt.figure(figsize=(12,6))
sns.boxplot(x="math score",data=df)
plt.xticks(np.arange(0,200,20))
plt.show()
#ouliers detection and removal
q1 = df['math score'].quantile(0.25)
q2 = df['math score'].quantile(0.75)
iqr = q2 - q1
lowerlimit = q1 - 1.5*iqr
upperlimit = q2 + 1.5*iqr
newdf = df[(df['math score'] > lowerlimit) &  (df['math score'] < upperlimit)]
print(newdf['math score'])
sns.boxplot(x='math score',data=newdf)
plt.show()


#data transformation (only scaling)
scaler = MinMaxScaler()
df['math score']=scaler.fit_transform(df[['math score']])
print(df['math score'])
