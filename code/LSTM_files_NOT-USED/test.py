#############################################################
####### THIS CODE IS NOT USED IN THE FINAL PROJECT ##########
####### This is just for illustration of the LSTM network  ##########

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from train import preprocessData, loadData
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


actions = np.array(['up', 'down', 'left', 'right', 'jump'])


### Load weights #######
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('../weights/action.h5')

### Load data #######
data, labels = loadData('dataPoints', 'labels')
x_train, x_test, y_train, y_test = preprocessData(data, labels)

#### Test ########
yhat = model.predict(x_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print("yhat:", yhat)
print("ytrue:", ytrue)

print("confusion_matrix:", multilabel_confusion_matrix(ytrue, yhat))



