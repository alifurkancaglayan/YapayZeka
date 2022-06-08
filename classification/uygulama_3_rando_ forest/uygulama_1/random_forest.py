import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%%  import data

data = pd.read_csv("data_diagnosis.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace = True)

# %%
data.diagnosis = [ 1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)
#%% normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

#%%
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("decision tree score: ", dt.score(x_test,y_test)*100)

#%%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
print("random forest result: ",rf.score(x_test,y_test)*100)
#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =3) # n_neighbors = k
knn.fit(x_train,y_train)
print("knn result: ",knn.score(x_test,y_test)*100)
#%%
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train,y_train) # Eğitim işlemi gerçekleşir
print("print accuracy of svm algo: ",svm.score(x_test,y_test)*100)

# find k value
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()