import numpy as np
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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
    X = []
    for x in data:
        X.append(np.array(x).flatten().tolist())

    X = np.array(X)

    # y = to_categorical(labels).astype(int)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    return X_train, X_test, y_train, y_test

def trainModel(X_train, y_train):
    # training a DescisionTreeClassifier
    dtree_model = DecisionTreeClassifier(max_depth = 10).fit(X_train, y_train)
    return dtree_model

if __name__ == "__main__":
    dataFile, labelsFile = loadParameters()
    data, labels = loadData(dataFile, labelsFile)
    X_train,X_test,y_train,y_test = preprocessData(data, labels)
 
    # training a DescisionTreeClassifier
    dtree_model = trainModel(X_train, y_train)
    dtree_predictions = dtree_model.predict(X_test)
#     print(dtree_model)
    print(dtree_predictions)
    print(y_test)
    score = accuracy_score(y_test, dtree_predictions)
    print(score)
#     # print(X_train.shape)