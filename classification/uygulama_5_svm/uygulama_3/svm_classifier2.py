import pandas as pd

dataset = pd.read_csv('data_satınalma.csv')

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Bağımsız değişkenlerden yaş ile tahmini gelir aynı birimde olmadığı için feature scaling uygulayacağız.

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#%%
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train) # Eğitim işlemi gerçekleşir
print("accuracy of svm algo: ",svm.score(X_test,y_test)*100)