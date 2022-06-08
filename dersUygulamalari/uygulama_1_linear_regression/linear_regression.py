# import library
import pandas as pd
import matplotlib.pyplot as plt
# import data
df = pd.read_csv("dersUygulamalari/uygulama_1_linear_regression/linear_regression_dataset.csv",sep = ";")
# plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%% linear regression
# sklearn library
from sklearn.linear_model import LinearRegression
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)
# linear regression model
linear_reg = LinearRegression()
linear_reg.fit(x,y)


#%% prediction
import numpy as np
print(linear_reg.predict([[13]]))






# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape(-1,1)  # deneyim


plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array)  # maas

plt.plot(array, y_head,color = "red")












