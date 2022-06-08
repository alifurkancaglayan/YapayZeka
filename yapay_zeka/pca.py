from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# veri kümesi
dataset = pd.read_csv('veri_seti/data_kalite.csv')

X = dataset.iloc[:, 0:12].values
y = dataset.iloc[:, 12].values

from sklearn.preprocessing import LabelEncoder
LE_X=LabelEncoder()
X[:,0] = LE_X.fit_transform(X[:,0]) 

# eğitim ve test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# pca
pca = PCA(n_components=3)
X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# pca dönüşümünden önce gelen LR
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train2, y_train)  # eğitim işlemi
print("test accuracy {}".format(classifier.score(X_test2, y_test)*100)),
