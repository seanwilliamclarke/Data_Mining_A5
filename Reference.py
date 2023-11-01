# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:18:46 2023

@author: ehan1
"""

## Example 1: Replicating the example from the slides
X = [[2],[5],[9],[15],[16],[18],[25],[33],[33],[45]]

# Using scipy
from scipy.cluster.hierarchy import dendrogram, linkage, complete, single, fcluster  
from scipy.spatial.distance import pdist

linked1 = linkage(X, method='single', metric='euclidean')
dendrogram(linked1, labels=X)

Z1 = single(pdist(X))
fcluster(Z1, t=6, criterion='distance')

linked2 = linkage(X, method='complete', metric='euclidean')
dendrogram(linked2, labels=X)

#Adding the cutoff line
from matplotlib import pyplot as plt
fig = plt.figure()
dendrogram(linked2, labels=X)
plt.axhline(y = 8, color='r')
plt.show()

Z2 = complete(pdist(X))
fcluster(Z2, t=6, criterion='distance')

# Using sklearn
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='euclidean')  
cluster.fit_predict(X) 
print(cluster.labels_)  


## Example 2: Hierarchical using Utilities.csv (data on electricity power plants)
import pandas as pd, numpy as np
utilities_df = pd.read_csv("Utilities.csv")
X = utilities_df[['Sales','Fuel_Cost']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Using scipy
from scipy.cluster.hierarchy import dendrogram, linkage  
linked = linkage(X_std, method='complete', metric='euclidean')
dendrogram(linked, labels=utilities_df['Company'].tolist())

# Using sklearn
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='euclidean')  
cluster.fit_predict(X_std) 
print(cluster.labels_)

pd.DataFrame(list(zip(utilities_df['Company'],np.transpose(cluster.labels_))), columns = ['Company','Cluster label'])

# Plot cluster membership
from matplotlib import pyplot
plot = pyplot.scatter(utilities_df['Sales'], utilities_df['Fuel_Cost'], c=cluster.labels_, cmap='rainbow') 
pyplot.legend(*plot.legend_elements(), title='clusters')
pyplot.show()


## Example 3: K-Means using Utilities.csv
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
model = kmeans.fit(X_std)
labels = model.predict(X_std)

pd.DataFrame(list(zip(utilities_df['Company'],np.transpose(labels))), columns = ['Company','Cluster label'])

# Plot cluster membership
from matplotlib import pyplot
pyplot.scatter(utilities_df['Sales'], utilities_df['Fuel_Cost'], c=labels, cmap='rainbow') 
plot = pyplot.scatter(utilities_df['Sales'], utilities_df['Fuel_Cost'], c=labels, cmap='rainbow') 
pyplot.legend(*plot.legend_elements(), title='clusters')
pyplot.show()