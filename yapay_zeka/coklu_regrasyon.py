from optparse import Values
import pandas as pd
import numpy as np

df=pd.read_csv("multi_veri.csv",sep=";")
x=df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)


from sklearn.linear_model import LinearRegression
multiple_linear_regrasyon=LinearRegression()
multiple_linear_regrasyon.fit(x,y)

print("b0: ", multiple_linear_regrasyon.intercept_)
print("b1,b2:", multiple_linear_regrasyon.coef_)

#pridict
multiple_linear_regrasyon.predict(np.array([[10,35]]))