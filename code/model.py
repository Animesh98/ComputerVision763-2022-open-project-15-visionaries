import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from train import preprocessData, loadData

#log_dir = os.path.join('Logs')
#tb_callback = TensorBoard(log_dir=log_dir)

actions = np.array(['up', 'down', 'left', 'right', 'jump'])
#tf.debugging.set_log_device_placement(True)


###Build and train LSTM#######
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(20,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


data, labels = loadData('dataPoints', 'labels')
x_train, x_test, y_train, y_test = preprocessData(data, labels)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=2000)

model.summary()