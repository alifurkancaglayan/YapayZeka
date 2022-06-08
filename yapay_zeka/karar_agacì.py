from sklearn.ensemble import RandomForestClassifier
from sklearn import random_projection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data = pd.read_csv("veri_seti/data_diagnosis.csv")
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# plot data
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)


# %% normalizasyon
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %% train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

# %%prediction
prediction = dt.predict(x_test)
print("score: ", dt.score(x_test, y_test)*100)

# %%randomforest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
print("random forest result: ", rf.score(x_test, y_test)*100)
