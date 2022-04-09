import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from train import preprocessData, loadData
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from mymodel import *


actions = np.array(['up', 'down', 'left', 'right', 'jump', 'stop'])


###LSTM model#######

model = MyModel()
model.load_weights('../weights/actionMotion_1') ##load weights##

data, labels = loadData('dataPointsMotion_1', 'labelsMotion_1')
x_train, x_test, y_train, y_test = preprocessData(data, labels) ## get the test data ## 


yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print("yhat:", yhat)
print("ytrue:", ytrue)

print("confusion_matrix:", multilabel_confusion_matrix(ytrue, yhat))



