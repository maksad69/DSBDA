import matplotlib.pyplot as plt
import seaborn as sns

df=sns.load_dataset('titanic')
plt.figure(figsize=(15,5))
sns.boxplot(x='sex',y='age',hue='survived',data=df)
plt.show()
