from statistics import linear_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#veri yükleme
veriler=pd.read_csv('veriler.csv')

#data frame
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#numpy array dönüşümü
X=x.values
Y=y.values

#linear regrasyon
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
linear_reg.fit(X,Y)

#polinom regrasyon
#2.dereceden denklem
from sklearn.preprocessing import PolynomialFeatures
poly_reg2=PolynomialFeatures(degree=2)
x_poly2=poly_reg2.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly2,y)
print(x_poly2)

#4.dereceden
poly_reg3=PolynomialFeatures(degree=4)
x_poly3=poly_reg3.fit_transform(X)
lin_reg3=LinearRegression()
lin_reg3.fit(x_poly3,y)
print(x_poly3)

#tahminler
print(linear_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg2.fit_transform([[6.6]])))
print(lin_reg3.predict(poly_reg3.fit_transform([[6.6]])))