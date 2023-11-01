# %%
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, complete, single, fcluster  
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# %%
# Load Data
# Read the data from the csv file
df = pd.read_csv('cereals.csv')

df.head()

# %%
# Task 0. We are going to develop clustering algorithms using the following variables: 'Calories', 'Protein',
# 'Fat', 'Fiber', 'Carbo', 'Sodium', 'Sugars', 'Potass', and 'Vitamins'. Drop any observations that have one or
# more missing value for these variables.

# Drop any observations that have one or more missing value for these variables.
df = df.dropna(subset=['Calories', 'Protein', 'Fat', 'Fiber', 'Carbo', 'Sodium', 'Sugars', 'Potass', 'Vitamins'])
df['Name'] = df['Name'].str.strip() 




# %%
# Task 1. First, perform agglomerative Clustering with complete linkage- Report the number of cereals in
# each cluster when the number of cluster is 2.

cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='complete')
cluster_labels = cluster.fit_predict(df[['Calories', 'Protein', 'Fat', 'Fiber', 'Carbo', 'Sodium', 'Sugars', 'Potass', 'Vitamins']])

unique, counts = np.unique(cluster_labels, return_counts=True) # return the number of cereals in each cluster

for cluster_num, counts in zip(unique, counts):
    print(f"Number of cereals in Cluster {cluster_num}: {counts}")





# %%
# Task 2. Then, use the same set of variables above to perform K-Mean Clustering with k=2. Report the
# number of cereals in each cluster.

cluster2 = KMeans(n_clusters=2, random_state=0)
cluster2_labels = cluster2.fit_predict(df[['Calories', 'Protein', 'Fat', 'Fiber', 'Carbo', 'Sodium', 'Sugars', 'Potass', 'Vitamins']])
unique, counts = np.unique(cluster2_labels, return_counts=True) # return the number of cereals in each cluster

for cluster_num, counts in zip(unique, counts):
    print(f"Number of cereals in Cluster {cluster_num}: {counts}")

#Describe cluster 0 and cluster 1

# Cluster 0

df[cluster2_labels == 0].describe()



# %%
# Cluster 1
df[cluster2_labels == 1].describe()

# %%
# visualize the clusters together with the original data
plt.figure(figsize=(10, 7))
plt.scatter(df['Calories'], df['Protein'], c=cluster2_labels, cmap='rainbow')
plt.xlabel('Calories')
plt.ylabel('Protein')
plt.title('K-Means Clustering with k=2')
plt.show()

    

# %%



