import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('satislar.csv')
aylar=veriler[['Aylar']]
print(aylar)
satislar=veriler[['Satislar']]
print(satislar)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(aylar,satislar,test_size=0.3)

from sklearn.linear_model import LinearRegrasyon
lr=LinearRegrasyon()

lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)