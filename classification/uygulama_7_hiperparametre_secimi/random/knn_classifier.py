import pandas as pd
import numpy as np
# %%
data = pd.read_csv("data_diagnosis.csv")

# %%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.tail()

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
basari = cross_val_score(estimator = knn, X=x, y=y , cv = 4)
print(basari.mean()*100)
print(basari.std()*100)

#%% print hyperparameter KNN algoritmasindaki K degeri
from sklearn.model_selection import RandomizedSearchCV

rf_params = {"n_neighbors":np.arange(1,100)}
knn= KNeighborsClassifier()

knn_cv = RandomizedSearchCV(knn, rf_params,n_iter=100, cv = 10) 
knn_cv.fit(x,y)

print("tuned hyperparameter K: ",knn_cv.best_params_)
print("tuned parametreye gore en iyi accuracy (best score): ",
      knn_cv.best_score_*100)




