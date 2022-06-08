import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
data = pd.read_csv("data_diagnosis.csv")

# %%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# malignant = M  kotu huylu tumor
# benign = B     iyi huylu tumor

# %%
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# scatter plot
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha= 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha= 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

# %%
# normalization 
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %% SVM
from sklearn.svm import SVC
svm = SVC()

 # %%k-katlamali capraz dogrulama 
from sklearn.model_selection import cross_val_score
basari = cross_val_score(estimator = svm, X=x, y=y , cv = 10)
print(basari.mean()*100)
print(basari.std()*100)

#%% print hyperparameter SVM algoritması
from sklearn.model_selection import GridSearchCV

grid = {'C': [0.1, 1,5,9, 10, 100, 1000],
              'gamma': [2,1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','linear']}

svm = SVC()

svm_cv = GridSearchCV(svm, grid, cv = 10)  # GridSearchCV
svm_cv.fit(x,y)

print("tuned hyperparameter SVM: ",svm_cv.best_params_)
print("tuned parametreye gore en iyi accuracy (best score): ",svm_cv.best_score_*100)

























