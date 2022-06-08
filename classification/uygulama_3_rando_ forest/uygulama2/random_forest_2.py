import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data_kalite.csv')
Data_pd = pd.DataFrame(data=dataset)

X = Data_pd.iloc[:, 0:12].values
Y = Data_pd.iloc[:, 12].values

"""  red ve white 0/1 olarak etiketleme işlemi """ 
from sklearn.preprocessing import LabelEncoder
LE_X = LabelEncoder()
X[:, 0] = LE_X.fit_transform(X[:, 0])

""" özellik normalize etme/dönüştürme işlemi (Standardizasyon=(X - ort) / var) """ 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#%%
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("decision tree score: ", dt.score(X_test,y_test)*100)

#%%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100,random_state = 1)
rf.fit(X_train,y_train)
print("random forest result: ",rf.score(X_test,y_test)*100)