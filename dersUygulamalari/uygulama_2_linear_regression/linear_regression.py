import pandas as pd

veriler = pd.read_csv('satislar.csv')

aylar=veriler[['Aylar']]
print(aylar)

satislar=veriler[['Satislar']]
print(satislar)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(aylar,satislar,test_size=0.3)


from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)

#%% skor tahmini

y_head = lr.predict(x_test)
from sklearn.metrics import r2_score
print("r_score: ", r2_score(y_test,y_head))


