# Video hand gestures to play game with hand motion capture

## Dependencies
`mediapipe, opencv, python, numpy, tensorflow`
## Project directory structure
* **code**
    * **datapoints.py** : For storing gesture datapoints and gesture labels in data directory.
        * Command to run : python3 datapoints.py numGestures [gesture1, gesture2, ...]\
        example : `python3 datapoints.py 2 forward backward`\
        *Note : numGesture and length of gestures list must be same*
    * **datapointsMotion.py** : For storing gesture datapoints and gesture labels in data directory with motion capturing.
        * Command to run : python3 datapointsMotion.py numGestures [gesture1, gesture2, ...]\
        example : `python3 datapointsMotion.py 5 forward backward left right jump`
    * **trainMy.py** : File having functions to train a simple *decision tree classifier* with `max_depth=10`.
        * Command to run : python3 trainMy.py dataPointsFileName labelsFileName\
        example : `python3 trainmy.py dataPoinits labels`
    * **inference.py** : File having functionality to infer from the trained model. User can make gesture in the webcam video and see the predicted gestures.
        * Command to run : python3 inference.py dataPointsFileName labelsFileName\
        example : `python3 inference.py dataPoinitsMotion labelsMotion`
        
* **data**
    * **dataPoints.npy** : numpy array file having gestures keypoints data
    * **labels.npy** : numpy array file having gestures label data for dataPoints
    * **dataPointsMotion.npy** : numpy array file having gestures keypoints data with motion features too
    * **labelsMotion.npy** : numpy array file having gestures label data for dataPointsMotion 
* **readme.md**
    
