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
dataPointsPath = '../data/dataPointsMotion'

# path for labels to store
labelsPath = '../data/labelsMotion'

# number of videos per action
numVideos = 50
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

# np array for storing the keypoints for every frame in every video and for each gesture, thus its final shape would be (numGestures*numVideos, numFrames, 105), as there are total of 63 keypoints for every hand (21*3) + 42 features having flow information (21*2 [magnitude and angle for every point])
data = []

# iterating for every gesture
for actionID,action in enumerate(actions):
    
    # giving 3 seconds of buffer time to user to make the gewsture properly
    for i in [3,2,1]:
        print(f'Training starts in {i}s')
        time.sleep(1)
    print(f'Training started for {action} action, make the gesture')

    trainMsg = f'Training for {action} action'

    # iterating through every video sample
    for video in range(numVideos):
        # np array for storing keypoints for every video, thus its final shape would be (numFrames, 105)
        res = []
        # frame counter
        i=0
        
        if video==0:
            cv2.waitKey(2000)
        
        # capturing initial frame
        _, oldimg = cap.read()
        oldgray = cv2.cvtColor(oldimg, cv2.COLOR_BGR2GRAY)

        # iterating through every frame 
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
                    lm = np.float32(lm)
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

                mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

            # if hand is found
            if results.multi_hand_landmarks and len(lm)==105:
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

                # appending frame features
                res.append(lm)

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
        # appending the video extracted features to vid list
        data.append(res)
        # appending the gesture ID to the labels list
        labels.append(actionID)

        print(f'Training for video {video+1} in {action} action')

# converting datapoints and labels list to np-array
labels = np.array(labels)
data = np.array(data)

# saving the final np arrays to the disk
np.save(dataPointsPath,data)
np.save(labelsPath,labels)

print("Datpoints successfully saved now train the network by running trainNetwork.py file")
cap.release()
cv2.destroyAllWindows()