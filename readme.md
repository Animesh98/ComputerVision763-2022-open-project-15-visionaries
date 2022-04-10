# Video hand gestures to play game with hand motion capture
## Please Run the game as explained in the demo, and use Windows to evaluate the project.
(As unity game will not run on linux as expected)
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

* **gestures**\
    Folder containing gestures as `.gif` files. User can look at these for reference while controlling the game character.

* **Game**\
    Folder containing a simple unity based game. User can run the game bu double clicking the `CV.exe` file.\
    **Note** : This file runs only in windows so use windows based system for evaluation.

* **reflectionEssay.pdf**
* **Presentation.mp4**
    Explanatory video about the whole project
* **Demo.mp4**
    A demo video for the project in action
* **readme.md**
* **requirements.txt**


    
