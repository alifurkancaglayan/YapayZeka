import pandas as pd
import numpy as np
# %%
data = pd.read_csv("data_diagnosis.csv")

# %%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)


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
basari = cross_val_score(estimator = svm, X=x, y=y , cv = 4)
print(basari.mean()*100)
print(basari.std()*100)


#%% print hyperparameter KNN algoritmasindaki K degeri
from sklearn.model_selection import RandomizedSearchCV

rf_params = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','linear']}
svm = SVC()

svm_cv = RandomizedSearchCV(svm, rf_params,n_iter=50, cv = 4, scoring='accuracy', n_jobs=-1, verbose=2)
svm_cv.fit(x,y)

print("tuned hyperparameter SVM: ",svm_cv.best_params_)
print("tuned parametreye gore en iyi accuracy (best score): ",svm_cv.best_score_*100)























