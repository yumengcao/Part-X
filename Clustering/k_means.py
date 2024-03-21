
from sklearn.cluster import KMeans
import numpy as np

def kmeans(x):
    km = KMeans(n_clusters=5)
    km.fit(x)
    return km.cluster_centers_
