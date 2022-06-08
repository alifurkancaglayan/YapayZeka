from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import random_projection
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import numpy as np
import pandas as pd

data = pd.read_csv("veri_seti/data_kalite.csv")
#data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
x = data.iloc[:, 0:12].values
y = data.iloc[:, 12].values

LE_X = LabelEncoder()
x[:, 0] = LE_X.fit_transform(x[:, 0])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# # plot data
# data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]
# y = data.diagnosis.values
# x_data = data.drop(["diagnosis"], axis=1)

# # %% normalizasyon
# x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

# # %% train test
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

# %%prediction
prediction = dt.predict(x_test)
print("score: ", dt.score(x_test, y_test)*100)

# %%randomforest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
print("random forest result: ", rf.score(x_test, y_test)*100)

# %%Knn
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
print("knn score: ", knn.score(x_test, y_test)*100)

# %%svm
svm = SVC()
svm.fit(x_train, y_train)
print("svm score: ", svm.score(x_test, y_test)*100)


# %% hypermeter KNN algoritmai k degeri
grid = {"n_neighbors": np.arange(1, 100)}
kn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=10)
knn_cv.fit(x, y)

print("tuned hypermeter K : ", knn_cv.best_params_)
print("tuned parametreye göre en iyi accuracy (best score) : ", knn_cv.best_score_)

# %%
grid = {'C': [0.1, 1, 5, 9, 10, 100, 1000],
        'gamma': [2, 1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'linear']}
svm = SVC()
svm_cv = GridSearchCV(svm, grid, cv=4)
svm_cv.fit(x, y)
print("tuned hypermeter K : ", svm_cv.best_params_)
print("tuned parametreye göre en iyi accuracy (best score) : ", svm_cv.best_score_)

# %% k-katlamalı çarpraz dogrulama
basari = cross_val_score(estimator=svm_cv, X=x, y=y, cv=10)
print("k-katlamalı: ", basari.mean()*100)
print("std: ", basari.std()*100)
