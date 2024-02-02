# Cereal Clustering Project

This repository contains Python code for clustering cereals based on their nutritional content. The project utilizes hierarchical agglomerative clustering with complete linkage and K-means clustering algorithms. The clustering is performed using the following variables: 'Calories', 'Protein', 'Fat', 'Fiber', 'Carbo', 'Sodium', 'Sugars', 'Potass', and 'Vitamins'.

## Table of Contents
- [Import Libraries](#import-libraries)
- [Load Data](#load-data)
- [Data Preprocessing](#data-preprocessing)
- [Agglomerative Clustering](#agglomerative-clustering)
- [K-Means Clustering](#k-means-clustering)
- [Cluster Descriptions](#cluster-descriptions)
- [Visualizing Clusters](#visualizing-clusters)

## Import Libraries <a name="import-libraries"></a>

```python
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, complete, single, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
```

## Load Data <a name="load-data"></a>

```python
# Load Data
# Read the data from the csv file
df = pd.read_csv('cereals.csv')
df.head()
```

## Data Preprocessing <a name="data-preprocessing"></a>

```python
# Task 0. Drop any observations that have one or more missing values for specific variables.
df = df.dropna(subset=['Calories', 'Protein', 'Fat', 'Fiber', 'Carbo', 'Sodium', 'Sugars', 'Potass', 'Vitamins'])
df['Name'] = df['Name'].str.strip()
```

## Agglomerative Clustering <a name="agglomerative-clustering"></a>

```python
# Task 1. Perform agglomerative Clustering with complete linkage and report the number of cereals in each cluster when the number of clusters is 2.
cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='complete')
cluster_labels = cluster.fit_predict(df[['Calories', 'Protein', 'Fat', 'Fiber', 'Carbo', 'Sodium', 'Sugars', 'Potass', 'Vitamins']])

unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster_num, counts in zip(unique, counts):
    print(f"Number of cereals in Cluster {cluster_num}: {counts}")
```

## K-Means Clustering <a name="k-means-clustering"></a>

```python
# Task 2. Perform K-Mean Clustering with k=2 and report the number of cereals in each cluster.
cluster2 = KMeans(n_clusters=2, random_state=0)
cluster2_labels = cluster2.fit_predict(df[['Calories', 'Protein', 'Fat', 'Fiber', 'Carbo', 'Sodium', 'Sugars', 'Potass', 'Vitamins']])

unique, counts = np.unique(cluster2_labels, return_counts=True)
for cluster_num, counts in zip(unique, counts):
    print(f"Number of cereals in Cluster {cluster_num}: {counts}")
```

## Cluster Descriptions <a name="cluster-descriptions"></a>

```python
# Describe cluster 0
df[cluster2_labels == 0].describe()

# Describe cluster 1
df[cluster2_labels == 1].describe()
```

## Visualizing Clusters <a name="visualizing-clusters"></a>

```python
# Visualize the clusters together with the original data
plt.figure(figsize=(10, 7))
plt.scatter(df['Calories'], df['Protein'], c=cluster2_labels, cmap='rainbow')
plt.xlabel('Calories')
plt.ylabel('Protein')
plt.title('K-Means Clustering with k=2')
plt.show()
```

Feel free to explore and modify the code to adapt it to your specific needs!
