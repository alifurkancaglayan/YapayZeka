import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
data = pd.read_csv("data_diagnosis.csv")

# %%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.tail()
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

# %%
# M = data[data.diagnosis == "M"]
# B = data[data.diagnosis == "B"]
# # scatter plot
# plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.3)
# plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.3)
# plt.xlabel("radius_mean")
# plt.ylabel("texture_mean")
# plt.legend()
# plt.show()

# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

# %%
# normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#%%
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
# %%
# knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =1) # n_neighbors = k

from sklearn.model_selection import cross_val_score
basari = cross_val_score(estimator = knn, X=x, y=y , cv = 10)
print(basari.mean()*100)
print(basari.std()*100)

#%% print hyperparameter KNN algoritmasindaki K degeri
from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,100)}
knn= KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10)  # GridSearchCV
knn_cv.fit(x,y)

print("tuned hyperparameter K: ",knn_cv.best_params_)
print("tuned parametreye gore en iyi accuracy (best score): ",knn_cv.best_score_*100)