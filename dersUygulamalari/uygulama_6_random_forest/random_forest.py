#1.kutuphaneler
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values



from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y)

plt.scatter(X,Y, color='red')
plt.plot(X,rf_reg.predict(X), color='blue')


# print(rf_reg.predict([[11]]))
print(rf_reg.predict([[6.6]]))





















