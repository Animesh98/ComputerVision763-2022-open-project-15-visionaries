import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from train import preprocessData, loadData


actions = np.array(['up', 'down', 'left', 'right', 'jump', 'stop'])

class MyModel(Model):
  def __init__(self):
    super(MyModel,self).__init__()
    self.lstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(20, 105))
    self.lstm2 = layers.Bidirectional(layers.LSTM(128, return_sequences=True, activation='relu'))
    self.lstm3 = layers.Bidirectional(layers.LSTM(64, return_sequences=False, activation='relu'))
    self.fc1 = layers.Dense(64, activation = 'relu')
    self.fc2 = layers.Dense(32, activation = 'relu')
    self.fc3 = layers.Dense(32, activation = 'relu')
    self.fc4 = layers.Dense(actions.shape[0], activation='softmax')

  def call(self,x):
    x = self.lstm1(x)
    x = self.lstm2(x)
    x = self.lstm3(x)

    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.fc4(x)
    return x
