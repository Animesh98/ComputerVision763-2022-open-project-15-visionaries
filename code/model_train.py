import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from train import preprocessData, loadData
from mymodel import *


#actions = np.array(['up', 'down', 'left', 'right', 'jump', 'stop'])


###Build and train model#######

model = MyModel()

data, labels = loadData('dataPointsMotion_1', 'labelsMotion_1')
x_train, x_test, y_train, y_test = preprocessData(data, labels)


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=400)
model.summary()
model.save('../weights/actionMotion_1')
