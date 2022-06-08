
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('dersUygulamalari/uygulama_4_polinom_regression/veri.csv')

#data frame
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

# numpy array (dizi) dönüşümü
X = x.values
Y = y.values


#linear regression (Doğrusal Model oluşturma)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)



#polynomial regression (Doğrusal olmayan Model oluşturma)
# 2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)
x_poly2 = poly_reg2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly2,y)
# print(x_poly2)

# 4.dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)
# print(x_poly3)


# Görselleştirme
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg2.fit_transform(X)), color = 'green')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'black')
plt.show()


#tahminler

# print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg2.fit_transform([[6.6]])))
# print(lin_reg2.predict(poly_reg2.fit_transform([[11]])))

print(lin_reg3.predict(poly_reg3.fit_transform([[6.6]])))
# print(lin_reg3.predict(poly_reg3.fit_transform([[11]])))








