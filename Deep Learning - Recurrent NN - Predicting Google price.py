# Recurrent Neural Network
#Prediction of Google's future stock price

#Part 1 - Data preprocessing
#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating the data structure with 60 timestamps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping (adding a new dimension to predict better). In this case we are creating a Keras "3D tensor with shape". In this case X_train = np.reshape(batch_size, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Part 2 - Building the RNN

#Importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding a fourth LSTM layer and Dropour regularisation
regressor.add(LSTM(units = 50)) #return sequence is not needed in this layer
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss= 'mean_squared_error')

#Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size= 32) #100 epochs is recommended value to begin with WARNING: will take time to process


#Part 3 - Making the predictions and visualizing the results

#Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

#Getting the predicted stock price of January 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values #Getting the 60 previous stock prices of the 60 previous days
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

#Getting the predicted stock price of 2017
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Measuring RMSE
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print('RMSE is', rmse)

#Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#IMPROVING THIS RNN MODEL
'''
    Getting more training data: we trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.
    Increasing the number of timesteps: the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).
    Adding some other indicators: if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.
    Adding more LSTM layers: we built a RNN with four LSTM layers but you could try with even more.
    Adding more neurones in the LSTM layers: we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.'''


#===============================================================END===============================
#===============================================================END===============================
#===============================================================END===============================
#===============================================================END===============================


##VARIATION ON THE PROGRAM ABOVE. HOW TO PREDICT SEVERAL DAYS IN ADVANCE
##5 VARIABLES ARE USED INSTED OF 3

# # Part 1 - Data Preprocessing
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#
# # Importing Training Set
# dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#
# cols = list(dataset_train)[1:5]
#
# # Preprocess data for training by removing all commas
#
# dataset_train = dataset_train[cols].astype(str)
# for i in cols:
#     for j in range(0, len(dataset_train)):
#         dataset_train[i][j] = dataset_train[i][j].replace(",", "")
#
# dataset_train = dataset_train.astype(float)
#
# training_set = dataset_train.as_matrix()  # Using multiple predictors.
#
# # Feature Scaling
# from sklearn.preprocessing import MinMaxScaler
#
# sc = MinMaxScaler(feature_range=(0, 1))
# training_set_scaled = sc.fit_transform(training_set)
#
# sc_predict = MinMaxScaler(feature_range=(0, 1))
#
# sc_predict.fit_transform(training_set[:, 0:1])
#
# # Creating a data structure with 60 timesteps and 1 output
# X_train = []
# y_train = []
#
# n_future = 20  # Number of days you want to predict into the future
# n_past = 60  # Number of past days you want to use to predict the future
#
# for i in range(n_past, len(training_set_scaled) - n_future + 1):
#     X_train.append(training_set_scaled[i - n_past:i, 0:5])
#     y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])
#
# X_train, y_train = np.array(X_train), np.array(y_train)
#
# # Part 2 - Building the RNN
#
# # Import Libraries and packages from Keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
#
# # Initializing the RNN
# regressor = Sequential()
#
# # Adding fist LSTM layer and Drop out Regularization
# regressor.add(LSTM(units=50, return_sequences=True, input_shape=(n_past, 4)))
# regressor.add(Dropout(.2))
#
# # Part 3 - Adding more layers
#
# # Adding 2nd layer with some drop out regularization
# regressor.add(LSTM(units=50, return_sequences=True))
# regressor.add(Dropout(.2))
#
# # Adding 3rd layer with some drop out regularization
# regressor.add(LSTM(units=50, return_sequences=True))
# regressor.add(Dropout(.2))
#
# # Adding 4th layer with some drop out regularization
# regressor.add(LSTM(units=50, return_sequences=False))
# regressor.add(Dropout(.2))
#
# # Output layer
# regressor.add(Dense(units=1, activation='sigmoid'))
#
# # Compiling the RNN
# regressor.compile(optimizer='adam', loss="binary_crossentropy")  # Can change loss to mean-squared-error if you require.
#
# # Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.
#
# es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
# rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
# mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
# tb = TensorBoard('logs')
#
# history = regressor.fit(X_train, y_train, shuffle=True, epochs=100,
#                         callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=64)
#
# # Predicting on the training set  and plotting the values. the test csv only has 20 values and
# # "ideally" cannot be used since we use 60 timesteps here.
#
# predictions = regressor.predict(X_train)
#
# # predictions[0] is supposed to predict y_train[19] and so on.
# predictions_plot = sc_predict.inverse_transform(predictions[0:-20])
# actual_plot = sc_predict.inverse_transform(y_train[19:-1])
#
# hfm, = plt.plot(predictions_plot, 'r', label='predicted_stock_price')
# hfm2, = plt.plot(actual_plot, 'b', label='actual_stock_price')
#
# plt.legend(handles=[hfm, hfm2])
# plt.title('Predictions vs Actual Price')
# plt.xlabel('Sample index')
# plt.ylabel('Stock Price')
# plt.savefig('graph.png', bbox_inches='tight')
# plt.show()
# plt.close()
#
# # For generating new predictions, create an X_test dataset just like the X_train data of (at-least) 80 days previous data
# # Format it for RNN input and use regressor.predict(new_X_test) to get predictions of the new_x_test
# # starting with day 81 to day 100