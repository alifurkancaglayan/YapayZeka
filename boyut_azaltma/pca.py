from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# veri kümesi
veriler = pd.read_csv('boyut_azaltma/data.csv')
X = veriler.iloc[:, 0:13].values
y = veriler.iloc[:, 13].values

# eğitim ve test kümelerinin bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Ölçekleme
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA
pca = PCA(n_components=5)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# pca dönüşümünden önce gelen LR
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# pca dönüşümünden sonra gelen LR
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

# tahminler
y_pred = classifier.predict(X_test)

y_pred2 = classifier2.predict(X_test2)

# actual / PCA olmadan çıkan sonuç
print('gercek / PCAsiz')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# actual / PCA sonrası çıkan sonuç
print("gercek / pca ile")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
