import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import argparse

def loadParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Name of data file', type=str, default='dataPoints')
    parser.add_argument('--labels', help='Name of labels file', type=str, default='labels')
    args = parser.parse_args()
    return args.data, args.labels

def loadData(dataFile, labelsFile):
    path = '../data/'
    data = np.load(path+dataFile+'.npy')
    labels = np.load(path+labelsFile+'.npy')
    return data, labels

def preprocessData(data, labels):
    X = data
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    dataFile, labelsFile = loadParameters()
    data, labels = loadData(dataFile, labelsFile)
    _,_,_,_ = preprocessData(data, labels)
    print(data.shape)
    print(labels.shape)
    print(temp.shape)