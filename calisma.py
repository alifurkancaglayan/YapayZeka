# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('dersUygulamalari/uygulama_6_random_forest/veriler.csv', sep=",")
print(data)
# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor

# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)

x = data.iloc[:,1].values
y = data.iloc[:,2].values

# fit the regressor with x and y data
regressor.fit(x, y)
Y_pred = regressor.predict(np.array([]).reshape(-1, 1)) # test the output by changing values
# Visualising the Random Forest Regression results

# arrange for creating a range of values
# from min value of x to max
# value of x with a difference of 0.01
# between two consecutive values
X_grid = np.arrange(min(x), max(x), 0.01)

# reshape for reshaping the data into a len(X_grid)*1 array,
# i.e. to make a column out of the X_grid value				
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot for original data
plt.scatter(x, y, color = 'blue')

# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid),
		color = 'green')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
