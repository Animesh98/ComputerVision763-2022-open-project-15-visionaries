#############################################################
####### CODE FOR CONTROLLING THE GAME #######################

from matplotlib import use
import cv2
import mediapipe as mp
import numpy as np
import argparse

# importing the ML model code
from trainMy import *


# Library for emulating key-presses
from pynput.keyboard import Key, Controller


# Hyperparameter for detecting motion
threshold = 8

# Defining some global objects
keyboard = Controller()
magArray=np.zeros(10)
userData = np.zeros((20,105))


# Mediapipe hands objects
mpHands = mp.solutions.hands
Hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Defining the Hand gestures capturing window
cv2.namedWindow('Get User Input',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Get User Input',450 ,300)

# Funtion for loading the command line arguments
def loadParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Name of data file', type=str, default='dataPointsMotion')
    parser.add_argument('--labels', help='Name of labels file', type=str, default='labelsMotion')
    args = parser.parse_args()
    return args.data, args.labels

# Function for loading the trained DECISION TREE model 
def getModel():
    dataFile, labelsFile = loadParameters()
    data, labels = loadData(dataFile, labelsFile)
    X_train,X_test,y_train,y_test = preprocessData(data, labels)
 
    # training a DescisionTreeClassifier
    dtree_model = trainModel(X_train, y_train)
    
    data = {}
    data['X_train'] = X_train
    data['X_test'] = X_test
    data['y_train'] = y_train
    data['y_test'] = y_test

    return dtree_model, data

# Loading the ML model
model,_ = getModel()

# Function for obtaining the gesture predictions based on user input
def getPrediction(model=model):
    global userData
    gestures = {0:'Forward',1:'Backward', 2:'Left', 3:'Right'}
    userData = np.array(userData).flatten()
    userData = userData.reshape((1,-1))
    # userData = userData.T
    # print(userData.shape)
    prediction = model.predict(userData)
    return gestures[prediction[0]]

# Function to calculate the optical flow in the camera input and obtaining the user input and returning the magnitude of the motion
def everyFrame(oldgray,newgray, hand):
    global userData
    global magArray
    h, w= newgray.shape
    if hand.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = np.array([[l.x,l.y,l.z] for l in handLms.landmark]).flatten()
            lm = lm.tolist()
            # storing all the 21 landmarks img coordiantes in a 2D numpy array
            p0 = np.array([[l.x * w,l.y * h] for l in handLms.landmark]).reshape(-1,1,2)
            p0 = np.float32(p0)
            # obtaining the flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(oldgray, newgray, p0, None)
            good_new = p1[st==1]
            good_old = p0[st==1]
            flow = good_new-good_old
            mag, ang = cv2.cartToPolar(flow[:,0], flow[:,1],angleInDegrees=True)
            mag = mag.flatten()
            ang = ang.flatten()
            mag = np.float32(mag)
            ang = np.float32(ang)

            # Getting the amount of motion between the consecutive frames  
            magArray= np.delete(magArray,0)
            magArray = np.append(magArray,np.average(mag))
            
            lm.extend(mag)
            lm.extend(ang)
            if len(lm)==105:
                userData = np.append(userData,np.array(lm))
                userData = userData[105:]
    else: 
        magArray= np.delete(magArray,0)
        magArray = np.append(magArray,0)
        userData = np.append(userData,np.zeros(105))
        userData = userData[105:]
    
    return np.average(magArray)


if __name__ == "__main__":
    keys = {'Forward':'w', 'Backward':'s', 'Left':'a', 'Right':'d'}
    keyPressed = False
    key = 'w'
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) # If you're on windows
    # cap = cv2.VideoCapture(0) #If you're using linux
    _, oldimg = cap.read()
    oldgray = cv2.cvtColor(oldimg, cv2.COLOR_BGR2GRAY)
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = Hands.process(imgRGB)
        showImg = cv2.flip(img,1)
        x= everyFrame(oldgray,imggray, results)
        if x>threshold:
            prediction = getPrediction()
            print(prediction)
            newkey = keys[prediction]
            if key!=newkey:
                keyboard.release(key)
                key=newkey
                keyboard.press(key)
                keyPressed=True
        else :
            print(0)
            if (keyPressed):
                keyboard.release(key)
                keyPressed=False

        cv2.imshow('Get User Input',showImg)
        oldimg=img
        oldgray=imggray
        k = cv2.waitKey(1)
        if k==ord('q'):
            break
    cv2.destroyAllWindows()