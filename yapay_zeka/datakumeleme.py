from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yükleme
veriler = pd.read_csv('veri_seti/data.csv')

X = veriler.iloc[:, 3:].values

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=123)
kmeans.fit(X)

sonuclar = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)


kmeans = KMeans(n_clusters=4, init='k-means++', random_state=123)
Y_tahmin = kmeans.fit_predict(X)
print(Y_tahmin)

#%%hiyeraşik
ac = AgglomerativeClustering(n_clusters=4)
Y_tahmin1 = ac.fit_predict(X)
print(Y_tahmin1)


import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()