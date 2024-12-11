#Aquesta és la versió original del codi amb petites modificacions per a que funcioni correctament

# Cell starts here
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pandas as pd

# Ruta del fitxer al teu ordinador
# file_path = r"C:\Users\Omar\ACproject-05-grup\dataset.csv" #Afegir la ruta del fitxer a carregar!!!
file_path = "dataset.csv" #Afegir la ruta del fitxer a carregar!!!

# Càrrega del dataset en un DataFrame de Pandas
dataset = pd.read_csv(file_path)

# Mostra les primeres files per confirmar que s'ha carregat correctament
print(dataset.head())


# Cell starts here
# Check the first 10 rows of the data
df = dataset
df.head(10)
# Cell ends here

# Cell starts here
# what columns do we have?
print(df.columns)
# what are the data types of the columns?
print(df.shape)
# Cell ends here

# Cell starts here
# drop column 'id' as it is not relevant to the research question
df = df.drop('id', axis=1)
# Cell ends here

# Cell starts here
# check for missing values and remove them if necessary (NaN)
df.isnull().sum()
# Cell ends here

# Cell starts here
# check for duplicates and remove them if necessary
df.duplicated().sum()
# Cell ends here

# Cell starts here
# Separate the data into two groups: categorical and numerical
df_cat = df[['sex', 'year', 'glang', 'part', 'job', 'stud_h', 'health', 'psyt']]
df_num = df[['age', 'jspe', 'qcae_cog', 'qcae_aff', 'amsp', 'erec_mean', 'cesd', 'stai_t', 'mbi_ex', 'mbi_cy', 'mbi_ea']]
# Cell ends here

# Cell starts here
df_cat.head(10)
# Cell ends here

# Cell starts here
# Naive description of the categorical data
df_cat.describe(include='all')
# Cell ends here

# Cell starts here
# Count and percentage of each category for each feature
for col in df_cat.columns:
    print(col) # print the name of the column
    print(pd.crosstab(index=df_cat[col], columns='count')) # print the count of each category
    print(pd.crosstab(index=df_cat[col], columns='percentage', normalize=True)) # print the percentage of each category
    print('-----------------')
# Cell ends here

# Cell starts here
# Plotting the count of each category for each feature using Seaborn

# Set the figure size
plt.figure(figsize=(20, 20))

# Plot the count of each category for each feature
for i, col in enumerate(df_cat.columns):
    plt.subplot(3, 3, i+1)
    sns.countplot(x=col, data=df_cat)
    plt.xlabel(col)
    plt.ylabel('Count')
# Cell ends here

# Cell starts here
# Chi-Square Test of Independence for each pair of categorical variables in a new dataframe
from scipy.stats import chi2_contingency
chi2_table = []
for i, col1 in enumerate(df_cat.columns):
    for j, col2 in enumerate(df_cat.columns):
        if i < j:
            chi2, p, _, _ = chi2_contingency(pd.crosstab(df_cat[col1], df_cat[col2]))
            chi2_table.append({'Variable 1': col1, 'Variable 2': col2, 'Chi-Square': chi2, 'p-value': p})

chi2_table = pd.DataFrame(chi2_table)
# Cell ends here

chi2_table = chi2_table[chi2_table['p-value'] < 0.05].sort_values(by='p-value')
print("Chi-Square results:\n", chi2_table)
# Cell ends here

# Cell starts here
# Keep only the pairs of variables that are related (p-value < 0.05) and sort them by p-value in ascending order (the smaller the p-value, the stronger the relationship)
# Cell ends here

# Cell starts here
# Percentage of each category for each pair of variables in chi2_table?
for i, row in chi2_table.iterrows():
    var1 = row['Variable 1']
    var2 = row['Variable 2']
    print(f"\n{var1} and {var2}")
    crosstab = pd.crosstab(df[var1], df[var2], normalize=True)
    print(crosstab)
    print('-----------------')
# Cell ends here

# Cell starts here
# Plotting relationships between psyt and sex
var1 = "psyt"
var2 = "sex"

# Create a cross-tabulation table
ctab = pd.crosstab(df[var1], df[var2])

# Plot the cross-tabulation table
ctab.plot(kind='bar', stacked=True)
plt.title("Relationship between " + var1 + " and " + var2)
plt.xlabel(var1)
plt.ylabel(var2)
plt.show()
# Cell ends here

# Cell starts here
# Create the new data frame
df_psyt = pd.DataFrame(data={'psyt': df_cat['psyt'], 'sex': df_cat['sex']})

# Group the data by psychotherapy and sex
df_psyt = df_psyt.groupby(['psyt', 'sex']).size().reset_index(name='count')

# Calculate the percentage of male, female, and non binary in each health category
total = df_psyt['count'].sum()
df_psyt['percentage'] = df_psyt['count'] / total * 100
df_psyt['sex'] = df_psyt['sex'].map({1: 'Male', 2: 'Female', 3: 'Non Binary'})
# Cell ends here

# Cell starts here
df_psyt
# Cell ends here

# Cell starts here
# Do it for every pair of variables in chi2_table
for i, row in chi2_table.iterrows():
    var1 = row['Variable 1']
    var2 = row['Variable 2']
    df_temp = pd.DataFrame(data={var1: df[var1], var2: df[var2]})
    df_temp = df_temp.groupby([var1, var2]).size().reset_index(name='count')
    total = df_temp['count'].sum()
    df_temp['percentage'] = df_temp['count'] / total * 100

    # Sumar els valors totals i afegir-los com una nova fila
    total_row = pd.DataFrame(df_temp[['count', 'percentage']].sum()).transpose()
    total_row[var1] = 'Total'
    total_row[var2] = 'Total'
    df_temp = pd.concat([df_temp, total_row], ignore_index=True)

    print(df_temp)
    print('-----------------')

# Cell ends here

# Cell starts here
df_num.head(10)
# Cell ends here

# Cell starts here
# Plotting the distribution of each numerical variable using Seaborn
# Set the figure size
plt.figure(figsize=(20, 20))

for i, col in enumerate(df_num.columns):
    plt.subplot(4, 3, i+1)
    sns.histplot(df_num[col])
    plt.xlabel(col)
    plt.ylabel('Density')

plt.show()
# Cell ends here

# Cell starts here
# Correlation matrix using Pearson's correlation coefficient (r) and p-value (p) to determine the statistical significance
corr_matrix = df_num.corr(method='pearson')
corr_matrix
# Cell ends here

# Cell starts here
# Heatmap of the correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
# Cell ends here

# Cell starts here
# Get a dataframe of the most to least correlated variables
corr_matrix = corr_matrix.unstack().reset_index()
corr_matrix.columns = ['Variable 1', 'Variable 2', 'Correlation']
corr_matrix = corr_matrix[corr_matrix['Variable 1'] != corr_matrix['Variable 2']]
corr_matrix = corr_matrix.sort_values(by='Correlation', ascending=False)
corr_matrix
# Cell ends here

# Cell starts here
# Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Cell ends here

# Cell starts here
# Standardize the data
scaler = StandardScaler()
df_num_scaled = scaler.fit_transform(df_num)

# Create a PCA instance: pca
pca = PCA(n_components=0.95)

# Fit the PCA instance to the scaled samples
pca.fit(df_num_scaled)

# Transform the scaled samples: pca_features
pca_features = pca.transform(df_num_scaled)

# Print the shape of pca_features
print("The shape of the pca is : ", pca_features.shape)
# Cell ends here

# Cell starts here
# Plot the cumulative sum of the explained variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()
# Cell ends here

# Cell starts here
# Create a dataframe of the loadings of the first four principal components and the variable names (columns) of the original dataset (index)
loadings = pd.DataFrame({'Feature': df_num.columns, 'PC1': pca.components_[0], 'PC2': pca.components_[1], 'PC3': pca.components_[2], 'PC4': pca.components_[3]})
loadings

# Sort the loadings of each principal component by their absolute value in descending order and print the first 5 rows
loadings = loadings.sort_values(by=['PC1', 'PC2', 'PC3', 'PC4'], ascending=False)
loadings.head(20)
# Cell ends here

# Cell starts here
# Imports
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering 
# Cell ends here

# Cell starts here
# How can I determine the optimal number of clusters?
# Create a list of inertia values for different k values
inertia = []
for k in range(1, 10):
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(pca_features)

    # Append the inertia to the list of inertias
    inertia.append(model.inertia_)

# Plot ks vs inertias
plt.plot(range(1, 10), inertia, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(range(1, 10))
plt.show()

# Cell ends here
##################################################################################################################################
# Cell starts here
# Create a KMeans instance with 3 clusters: model (k=3) and fit it to the data (pca_features) using the fit() method
model = KMeans(n_clusters=3)
clusters = model.fit_predict(pca_features)
# Cell ends here

# Cell starts here
# Create a scatter plot of the first two principal components
plt.scatter(pca_features[:,0], pca_features[:,1], c=clusters, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
# Cell ends here

# Cell starts here
# Extracting the cluster labels
cluster_labels = model.labels_
# Cell ends here

# Cell starts here
# Adding the cluster labels to the original data frame
df_clustered = df_num.copy() # Create a copy of the original data frame
df_clustered['Cluster'] = cluster_labels # Add the cluster labels to the copy of the original data frame
# Cell ends here

# Cell starts here
df_clustered.head(5) # Print the first 5 rows of the data frame
# Cell ends here

# Cell starts here
# Grouping the data frame by cluster to get the properties of each cluster
cluster_grouped = df_clustered.groupby('Cluster')
cluster_properties = cluster_grouped.mean()

# Printing the properties of each cluster (mean values of the variables)
print(cluster_properties)
# Cell ends here

# Cell starts here
# Getting the number of patients in each cluster
cluster_grouped.size()
# Cell ends here

# Cell starts here
# Plotting the properties of each cluster
cluster_properties.plot(kind='bar', figsize=(15, 10))
plt.show()
# Cell ends here

# Cell starts here
# Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Cell ends here

# Cell starts here
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(pca_features, cluster_labels, test_size=0.2, random_state=42) # 80% training and 20% test data sets

# Create a logistic regression classifier
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))
# Cell ends here

# Cell starts here
# Import the necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Cell ends here

# Cell starts here
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Cell ends here