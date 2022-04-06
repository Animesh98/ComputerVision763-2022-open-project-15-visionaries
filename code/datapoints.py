import time
import cv2
import mediapipe as mp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('numGestures', help='Number of gestures to train', type=int)
parser.add_argument('gestures', nargs='+', help='Space separated names for the gestures', type=str)
args = parser.parse_args()

actions = args.gestures

if args.numGestures != len(actions):
    print('Number of gestures entered is not correct, exiting')
    exit()



# path for data to store
dataPointsPath = '../data/dataPoints'

# path for labels to store
labelsPath = '../data/labels'

# number of videos per action
numVideos = 30
# number of frames per video
numFrames = 20

# Mediapipe hands objects
mpHands = mp.solutions.hands
Hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# webcam capture object
cap = cv2.VideoCapture(0)

# list for storing labels per gesture per video, thus final shape is (numGestures*numVideos,1)
labels = []

# iterating for every gesture
for actionID,action in enumerate(actions):
    
    # giving 3 seconds of buffer time to user to make the gewsture properly
    for i in [3,2,1]:
        print(f'Training starts in {i}s')
        time.sleep(1)
    print(f'Training started for {action} action, make the gesture')

    trainMsg = f'Training for {action} action'

    # np array for storing the keypoints for every frame in every video and for each gesture, thus its final shape would be (numGestures*numVideos, numFrames, 63), as there are total of 63 keypoints for every hand (21*3)
    vid = []

    # iterating through every video sample
    for video in range(numVideos):
        # np array for storing keypoints for every video, thus its final shape would be (numFrames, 63)
        res = []
        # frame counter
        i=0
        # iterating through every frame 
        while i<numFrames:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = Hands.process(imgRGB)
            # if hand is found in the frame
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    lm = np.array([[l.x,l.y,l.z] for l in handLms.landmark]).flatten()
                mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
                res.append(lm)

            # if hand is found
            if len(res)!=0:
                cv2.namedWindow('Training',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Training',1080,720)
                showImg = cv2.flip(img,1)
                cv2.putText(showImg,trainMsg,(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
                cv2.putText(showImg,'Hand detected, training',(0,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)
                cv2.imshow('Training',showImg)
                k = cv2.waitKey(1)
                if k==ord('q'):
                    print('Training Aborted, exiting')
                    exit()
                # Goto next frame
                i = i+1

            # if hand is not found in the frame, don't change the frame
            else:
                cv2.namedWindow('Training',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Training',1080,720)
                showImg = cv2.flip(img,1)
                cv2.putText(showImg,trainMsg,(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
                cv2.putText(showImg,'No hand detected! Training stopped',(0,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),1)
                cv2.imshow('Training',showImg)
                k = cv2.waitKey(1)
                if k==ord('q'):
                    print('Training Aborted, exiting')
                    exit()

        res = np.array(res)
        vid.append(res)

        # appending the gesture ID to the labels list
        labels.append(actionID)

        print(f'Training for video {video+1} in {action} action')

# converting datapoints and labels list to np-array
labels = np.array(labels)
vid = np.array(vid)

# saving the final np arrays to the disk
np.save(dataPointsPath,vid)
np.save(labelsPath,labels)

print("Datpoints successfully saved now train the network by running trainNetwork.py file")
cap.release()
cv2.destroyAllWindows()