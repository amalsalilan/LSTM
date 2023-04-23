import numpy as np
from sklearn import preprocessing
import pandas as pd

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras.callbacks import EarlyStopping

from keras import optimizers

np.random.seed(4)
from tensorflow.random import set_seed
set_seed(4)

history_points = 50
data=pd.read_csv('TSLA_daily.csv')

data=data.iloc[::-1]
data.reset_index(drop=True,inplace=True)
data=data.drop('date',axis=1)





data_normaliser = preprocessing.MinMaxScaler()
data_normalised = data_normaliser.fit_transform(data)
ohlcv_histories_normalised = np.array([data_normalised[i  : i + history_points].copy() for i in range(len(data_normalised) - history_points)])
next_day_close_values_normalised = np.array([data_normalised[:,3][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
next_day_close_values_normalised = np.expand_dims(next_day_close_values_normalised, -1)
next_day_close_values = np.array([data.to_numpy()[:,3][i + history_points].copy() for i in range(len(data) - history_points)])
next_day_close_values = next_day_close_values.reshape(next_day_close_values.shape[0], 1)


y_normaliser = preprocessing.MinMaxScaler()
y_normaliser.fit(next_day_close_values)



test_split = 0.9 # the percent of data to be used for training
n = int(ohlcv_histories_normalised.shape[0] * test_split)


x_train = ohlcv_histories_normalised[:n]
y_train = next_day_close_values_normalised[:n]



x_test = ohlcv_histories_normalised[n:]
y_test = next_day_close_values_normalised[n:]



unscaled_y_train = next_day_close_values[:n]
unscaled_y_test = next_day_close_values[n:]



technical_indicators = []

for his in ohlcv_histories_normalised:
  # since we are using his[3] we are taking the SMA of the closing price
  sma = np.mean(his[:,3])
  technical_indicators.append(np.array([sma]))

technical_indicators = np.array(technical_indicators)

tech_ind_scaler = preprocessing.MinMaxScaler()
technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)




tech_ind_train = technical_indicators_normalised[:n]
tech_ind_test = technical_indicators_normalised[n:]

# define two sets of inputs
lstm_input = Input(shape=(history_points, 5), name='lstm_input')
dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

# the first branch operates on the first input
x = LSTM(32, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
lstm_branch = Model(inputs=lstm_input, outputs=x)

# the second branch opreates on the second input
y = Dense(20, name='tech_dense_0')(dense_input)
y = Activation("relu", name='tech_relu_0')(y)
y = Dropout(0.2, name='tech_dropout_0')(y)
technical_indicators_branch = Model(inputs=dense_input, outputs=y)

# combine the output of the two branches
combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
z = Dense(1, activation="linear", name='dense_out')(z)

# our model will accept the inputs of the two branches and then output a single value
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)

adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam,
              loss='mse')

from keras.utils import plot_model

# plot_model(model, to_file='model.png', show_shapes=True)


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
history = model.fit(x=[x_train, tech_ind_train], y=y_train, batch_size=32, epochs=300, shuffle=True, validation_split=0.2, callbacks=[es])



evaluation = model.evaluate([x_test, tech_ind_test], y_test)
print(evaluation)


y_predicted_train = model.predict([x_train, tech_ind_train])
y_predicted_train = y_normaliser.inverse_transform(y_predicted_train)

real_mse_train = np.mean(np.square(unscaled_y_train - y_predicted_train))
print("Train RMSE = {}".format(real_mse_train))



y_test_predicted = model.predict([x_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

real_mse_test = np.mean(np.square(unscaled_y_test - y_test_predicted))
print("Test RMSE = {}".format(real_mse_test))


from matplotlib import pyplot

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()

import matplotlib.pyplot as plt
plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

