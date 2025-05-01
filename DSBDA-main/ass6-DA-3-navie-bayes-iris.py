
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import  GaussianNB


df = pd.read_csv('iris.csv')
print(df)
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])  # Male:1, Female:0
print("Gender encoding:", df['species'].unique())

print(df['species'].unique())

x = df.drop('species',axis=1)
print(x)
y= df['species']
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

me = MinMaxScaler()
x_train = me.fit_transform(x_train)
x_test = me.fit_transform(x_test)

g = GaussianNB()
g.fit(x_train,y_train)

y_pred = g.predict(x_test)

plt.scatter(y_test,y_pred)
plt.show()

#Accuracy measures how many predictions are correct in classification,
# while R² measures how well a regression model fits the actual data.
