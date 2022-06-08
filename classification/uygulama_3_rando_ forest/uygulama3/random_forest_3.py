import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

dataset = pd.read_csv('data_satınalma.csv')

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Bağımsız değişkenlerden yaş ile tahmini gelir aynı birimde olmadığı için feature scaling uygulayacağız.

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

print("decision tree score: ", dt.score(X_test,y_test)*100)

#%%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10,random_state = 0)
rf.fit(X_train,y_train)
print("random forest result: ",rf.score(X_test,y_test)*100)