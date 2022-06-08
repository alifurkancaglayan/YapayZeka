import pandas as pd
import sklearn

dataset=pd.read_csv('data_satınalma.csv')

X= dataset.iloc[:, [2,3]].values
y= dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train) #eğitim işlemi
print("test accuracy {}".format(classifier.score(X_test,y_test)*100))

y_pred = classifier.predict(X_test)
print((y_test == y_pred).mean())

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("")




dataset=pd.read_csv('data_kalite.csv')

X= dataset.iloc[:, 0:12].values
y= dataset.iloc[:, 12].values

#red ve white 0/1 olarak etiketleme işlemi
from sklearn.preprocessing import LabelEncoder
LE_X=LabelEncoder()
X[:,0] = LE_X.fit_transform(X[:,0])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train) #eğitim işlemi
print("test accuracy {}".format(classifier.score(X_test,y_test)*100))

y_pred = classifier.predict(X_test)
print((y_test == y_pred).mean())

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

print("")




dataset=pd.read_csv('data_cinsiyet.csv')

X= dataset.iloc[:, [1,3]].values
y= dataset.iloc[:, 4].values

#red ve white 0/1 olarak etiketleme işlemi
from sklearn.preprocessing import LabelEncoder
LE_X=LabelEncoder()
X[:,0] = LE_X.fit_transform(X[:,0])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train) #eğitim işlemi
print("test accuracy {}".format(classifier.score(X_test,y_test)*100))

y_pred = classifier.predict(X_test)
print((y_test == y_pred).mean())

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

print("")




dataset=pd.read_csv('data_diagnosis.csv')

X= dataset.iloc[:, 2:31].values
y= dataset.iloc[:, 1].values

#red ve white 0/1 olarak etiketleme işlemi
from sklearn.preprocessing import LabelEncoder
LE_X=LabelEncoder()
X[:,1] = LE_X.fit_transform(X[:,1])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train) #eğitim işlemi
print("test accuracy {}".format(classifier.score(X_test,y_test)*100))

y_pred = classifier.predict(X_test)
print((y_test == y_pred).mean())

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)