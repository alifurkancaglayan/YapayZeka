import tensorflow 
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization
from keras.layers import Dense
from keras.models import Sequential
from keras import layers
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import keras


data = pd.read_csv("yapay_zeka_ali/veri_seti/data_diagnosis.csv")
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# plot data
data.diagnosis = [1 if each == 'M' else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

x_data.astype("uint8")
# %% standart scaler
scaler = StandardScaler()
x = scaler.fit_transform(x_data)

# %%
y = to_categorical(y)
# %% train test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# %%
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 1. kütüphane için
# 2. kütüphane için

verbose, epochs, batch_size = 0, 10, 8
n_features, n_outputs = x_train.shape[1], y_train.shape[1]

model = Sequential()
input_shape = (x_train.shape[1], 1)

# 1 kütüphane için
model.add(Conv1D(filters=8, kernel_size=5,
          activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3))
model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.summary()
print("başladı")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size ,verbose=1)

accuracy = model.evaluate(x_test, y_test, verbose=0)

print(accuracy)