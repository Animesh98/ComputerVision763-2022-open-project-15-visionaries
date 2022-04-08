from matplotlib import use
import cv2
import mediapipe as mp
import numpy as np
from trainMy import *
import argparse
import os

from pynput.keyboard import Key as kk
from pynput.keyboard import Controller as kc
from pynput.mouse import Button as mb
from pynput.mouse import Controller as mc
import time

# os.system(r'"C:/Users/91916/Desktop/CV Project/open-project-15-visionaries-main/Game/CV.exe"')

mouse = mc()
keyboard = kc()

def altTab(k=keyboard):
    k.press(kk.alt)
    k.press(kk.tab)
    k.release(kk.alt)
    k.release(kk.tab)

# number of frames per video
numFrames = 20

# Mediapipe hands objects
mpHands = mp.solutions.hands
Hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils


cv2.namedWindow('Get User Input',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Get User Input',1080,720)

def loadParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='Name of data file', type=str, default='dataPoints')
    parser.add_argument('--labels', help='Name of labels file', type=str, default='labels')
    args = parser.parse_args()
    return args.data, args.labels

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

model, globalData = getModel()

def getAccuracy(model, y_test):
    dtree_predictions = model.predict(globalData['X_test'])
    score = accuracy_score(y_test, dtree_predictions)
    return score

def getPrediction(model=model):
    gestures = {0:'Forward',1:'Backward', 2:'Left', 3:'Right', 4:'Jump'}
    userData = np.array(getUserInput()).flatten()
    userData = userData.reshape((-1,1))
    userData = userData.T
    prediction = model.predict(userData)
    predictedGesture = gestures[prediction[0]]
    return predictedGesture


def getUserInput():
    userData = []
    i=0
    # webcam capture object
    cap = cv2.VideoCapture(0)

    # capturing initial frame
    _, oldimg = cap.read()
    oldgray = cv2.cvtColor(oldimg, cv2.COLOR_BGR2GRAY)
    
    # print('Get Input')
    # time.sleep(2)

    while i<numFrames:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = Hands.process(imgRGB)

        # getting height, width of the frame
        h, w, c = img.shape

        # if hand is found in the frame
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # storing all the 21 landmarks x,y,z values in a flat numpy array
                lm = np.array([[l.x,l.y,l.z] for l in handLms.landmark]).flatten()

                # storing all the 21 landmarks img coordiantes in a 2D numpy array
                p0 = np.array([[l.x * w,l.y * h] for l in handLms.landmark]).reshape(-1,1,2)
                p0 = np.float32(p0)

                # getting the list back from the landmarks nparraay
                lm = lm.tolist()

                # obtaining the flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(oldgray, imggray, p0, None)

                good_new = p1[st==1]
                good_old = p0[st==1]
                flow = good_new-good_old

                mag, ang = cv2.cartToPolar(flow[:,0], flow[:,1],angleInDegrees=True)
                mag = mag.flatten()
                ang = ang.flatten()
                mag = np.float32(mag)
                ang = np.float32(ang)

                # adding the flow info to frame features
                lm.extend(mag)
                lm.extend(ang)

                # Move to next frame
                oldgray = imggray

                # appending to userData if 105 features are obtained successfully
                if len(lm)==105:
                    userData.append(lm)

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
            
            # cv2.namedWindow('Get User Input',cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Get User Input',1080,720)
            showImg = cv2.flip(img,1)
            cv2.putText(showImg,'Hand detected, make gestures',(0,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
            cv2.imshow('Get User Input',showImg)
            
            if len(lm)==105:
                i = i+1

            k = cv2.waitKey(1)
            if k==ord('q'):
                exit()
        else:
            # cv2.namedWindow('Get User Input',cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Get User Input',1080,720)
            showImg = cv2.flip(img,1)
            cv2.putText(showImg,'No hand detected!',(0,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1)
            cv2.imshow('Get User Input',showImg)
            k = cv2.waitKey(1)
            if k==ord('q'):
                exit()
    # print('Capture Complete')
    # time.sleep(1)
    return userData


if __name__ == "__main__":
    keys = {'Forward':'w', 'Backward':'s', 'Left':'a', 'Right':'d', 'Jump':'j'}

    # altTab()
    # time.sleep(2)
    # mouse.click(mb.left)
    while True:
        # keyboard.press(keys[getPrediction()])
        # keyboard.release(keys[getPrediction()])
        print(getPrediction())
    # altTab()