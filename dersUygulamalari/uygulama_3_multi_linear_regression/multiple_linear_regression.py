import pandas as pd
import numpy as np

df = pd.read_csv("veri.csv",sep = ";")

x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

# %% fitting data
from sklearn.linear_model import LinearRegression
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2: ",multiple_linear_regression.coef_)

# predict
# print(multiple_linear_regression.predict(np.array([[10,35]])))