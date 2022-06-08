import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("linear_regression_dataset.csv",sep=";")

#plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%%linear regrasyon
# sklearn library

from sklearn.linear_model import LinearRegression
x=df.deneyim.values.resahpe(-1,1)
y=df.maas.values.resahpe(-1,1)
#linear regrssion model
linear_reg=LinearRegression()
linear_reg.fit(x,y)
#%%prediction
import numpy as np
print(linear_reg.predict([[13]]))
