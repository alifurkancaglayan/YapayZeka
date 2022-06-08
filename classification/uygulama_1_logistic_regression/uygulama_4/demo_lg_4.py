#kutuphaneler
import pandas as pd

#veri yukleme
veriler = pd.read_csv('data_cinsiyet.csv')
print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

from sklearn.preprocessing import LabelEncoder
LE_X = LabelEncoder()
y = LE_X.fit_transform(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)

print(" logistic score:",logr.score(X_test,y_test)*100)
print((y_test == y_pred).mean()*100)

