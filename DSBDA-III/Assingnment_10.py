# Data Visualization III
# Download the Iris flower dataset or any other dataset into a DataFrame. (e.g.,
# https://archive.ics.uci.edu/ml/datasets/Iris ). Scan the dataset and give the inference as:
# 1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
# 2. Create a histogram for each feature in the dataset to illustrate the feature distributions.
# 3. Create a boxplot for each feature in the dataset.
# 4. Compare distributions and identify outliers.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datas = sns.load_dataset("iris")
print(datas)
print(datas.shape)
print(datas.describe())
print(datas.isnull().sum())

# 1] list down the features and their types (e.g., numeric, nominal) available in the dataset.
print(datas.dtypes)
datas['species'] = datas['species'].astype('category')
print(datas.dtypes)

# 2] Create a histogram for each feature in the dataset to illustrate the feature distributions.
datas.hist(figsize=(10, 8), bins=15, edgecolor='black', color='skyblue')
plt.suptitle('Histograms of Iris Features')
plt.tight_layout()
plt.show()
#Feature	           Groups?	                    What to say
# Sepal Length    	No clear groups	            Values overlap
# Sepal Width	    No clear groups	            Values overlap
# Petal Length	    3 clear groups           	Good separation between species
# Petal Width	    3 clear groups	            Good separation between species

# Boxplot
datas.plot(kind='box', subplots=True, layout=(2, 2), figsize=(10, 8), sharex=False, sharey=False)
plt.suptitle('Boxplots of Iris Features')
plt.tight_layout()
plt.show()

#When we say "box is nicely spread", we mean:
# The box is wide (not too narrow or too squeezed).
# Values are well distributed — not all stuck at one small range.
# It covers different groups or classes clearly.


# 4] Compare Distributions & Identify Outliers
# 🟩 Observations from Histograms:
# Petal length and width clearly show three separate groups (good for classification).
# Sepal features have more overlapping distributions.
# Data is fairly well spread for all features, but petal length has more variation.

# 🟨 Observations from Boxplots:
# Petal width and petal length show clear separation between species.
# Sepal width has some outliers, especially in Setosa.
# No extreme outliers in petal features.
# Some mild outliers in sepal measurements.