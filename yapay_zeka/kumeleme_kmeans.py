from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning


# %%create dataset
x1 = np.random.normal(25, 5, 1000)
y1 = np.random.normal(25, 5, 1000)

# class 2
x2 = np.random.normal(55, 5, 1000)
y2 = np.random.normal(60, 5, 1000)

# class 3
x3 = np.random.normal(55, 5, 1000)
y3 = np.random.normal(15, 5, 1000)

x = np.concatenate((x1, x2, x3))
y = np.concatenate((x1, x2, x3))

dictionary = {"x": x, "y": y}

data = pd.DataFrame(dictionary)

# %%k means

wcss = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(range(1, 15), wcss)
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()

# %%k=3 için model

kmeans2 = KMeans(n_clusters=3)
cluster = kmeans2.fit_predict(data)

data["label"] = clusters
plt.figure(figsize=(6, 6))
plt.scatter(data.x[data.label == 0], data.y[data.label == 0])
plt.scatter(data.x[data.label == 1], data.y[data.label == 1])
plt.scatter(data.x[data.label == 2], data.y[data.label == 2])
plt.scatter(data.x[data.label == 3], data.y[data.label == 3])

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_)
plt.show()