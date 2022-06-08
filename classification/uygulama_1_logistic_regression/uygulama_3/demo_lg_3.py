import pandas as pd

dataset = pd.read_csv('classification/uygulama_1_logistic_regression/uygulama_3/data_kalite.csv')
Data_pd = pd.DataFrame(data=dataset)

X = Data_pd.iloc[:, 0:12].values
Y = Data_pd.iloc[:, 12].values

"""  red ve white 0/1 olarak etiketleme işlemi """ 
from sklearn.preprocessing import LabelEncoder
LE_X = LabelEncoder()
X[:, 0] = LE_X.fit_transform(X[:, 0])

""" özellik normalize etme/dönüştürme işlemi (Standardizasyon=(X - ort) / var) """ 
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X = sc.fit_transform(X)

""" Özellik Ölçeklendirme/Normalizasyon (MinMax Scaling=(X-Xmin / Xmax-Xmin)"""
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X = mm.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print((y_test == y_pred).mean()*100)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm)