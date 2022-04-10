# Video hand gestures to play game with hand motion capture
## Quickstart
Create a new virtual environment and install all the dependencies using the `requirements.txt` file.
## Dependencies
`mediapipe, opencv, python, numpy, sklearn`
## Project directory structure
* **code**
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
    * **dataPointsMotion.npy** : numpy array file having gestures keypoints data with motion features too
    * **labelsMotion.npy** : numpy array file having gestures label data for dataPointsMotion 
* **readme.md**
* **requirements.txt**
    
