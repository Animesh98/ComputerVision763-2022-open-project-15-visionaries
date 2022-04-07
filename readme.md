# Video hand gestures to play game

## Dependencies
`mediapipe, opencv, python, numpy, tensorflow`
## Project directory structure
* **code**
    * **datapoints.py** : For storing gesture datapoints and gesture labels in data directory.
        * Command to run : python3 datapoints.py [numGestures] [gesture1, gesture2, ...]\
        example : `python3 datapoints.py 2 forward backward`\
        *Note : numGesture and length of gestures list must be same*
        
* **data**
    * **dataPoints.npy** : numpy array file having gestures keypoints data
    * **labels.npy** : numpy array file having gestures label data 
* **readme.md**
    
