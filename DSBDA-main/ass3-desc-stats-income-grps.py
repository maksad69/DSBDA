import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('iris.csv')

# Check for missing values
print(df.isnull().sum())

# Encode categorical variable
# le = LabelEncoder()
# df['species'] = le.fit_transform(df['species'])
# print("Encoded species labels:", df['species'].unique())

# Group by species and print each group (optional for checking)
g = df.groupby('species')
for species_label, species_df in g:
    print(f"\nSpecies label: {species_label}")
    print(species_df)

# Summary statistics for all numeric columns grouped by species
summary = df.groupby('species').agg(['mean', 'median', 'min', 'max', 'std'])
print("\nSummary Statistics (Grouped by species):")
print(summary)

# List of sepal_length values by species
value_list = df.groupby('species')['sepal_length'].apply(list).reset_index()
print("\nList of Sepal Length values by species:")
print(value_list)


#remaining 2nd
setosa = df[df['species'] == 'setosa']
versicolor = df[df['species'] == 'versicolor']
virginica = df[df['species'] == 'virginica']

# Function to display statistics
def display_statistics(species_name, data):
    print(f"\nStatistical details for {species_name}:")
    print(data.describe())

# Display statistics for each species
display_statistics('setosa', setosa)
display_statistics('versicolor', versicolor)
display_statistics('virginica', virginica)
