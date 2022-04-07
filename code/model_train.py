import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from train import preprocessData, loadData


actions = np.array(['up', 'down', 'left', 'right', 'jump'])


###Build Model#######
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

###Load data#####
data, labels = loadData('dataPoints', 'labels')
x_train, x_test, y_train, y_test = preprocessData(data, labels)

##### Train and save weights ########
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=800)
model.summary()
model.save('../weights/action.h5')
