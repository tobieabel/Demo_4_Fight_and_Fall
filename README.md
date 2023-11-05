# Demo_4_Fight_and_Fall
Computer Vision Demo 4 - Detecting Fighting and Falling with YOLOv8 object detection and classification
Python files and YOLO models used for demo 4.  The highest level script file is Incident_Detection.py.
This uses Streamlit for all front end interfaces.  Python Interpreter version is 3.11

You will need to download the following python packages before running this script - see requirements.txt for version details:
cv2, Numpy, Pandas, Ultralytics, Streamlit and Supervision.  If possible install these into a virtual environment on an IDE such as Pycharm

Use the following command in your terminal from the root directory, 
replacing *YourDiretory* with the correct file directory to reach the Incident_Detection.py file

streamlit run *YourDiretory*/Incident_Detection.py

Analysis by ChatGPT:
I've reviewed the code you provided. It appears to be a Python script for incident detection using YOLO object detection models and Streamlit for visualization. Here are some key points about the code:

1. It uses several libraries, including OpenCV (for image processing and video capture), NumPy (for numerical operations), the "supervision" library (for drawing bounding boxes and annotations), Ultralytics YOLO (for object detection), Streamlit (for the user interface), and Pandas (for creating data tables).

2. The code defines two main classes, "Rules_Engine" and "Main_Work_Flow," and several functions within these classes to manage the incident detection workflow.

3. The "Rules_Engine" class is designed to keep track of detections and evaluate whether incidents have occurred based on predefined rules. It uses a sliding window approach to track detections over time.

4. The "Main_Work_Flow" class is responsible for capturing frames from video sources, running object and classification models, and managing the incident detection process. It uses Ultralytics YOLO for object detection and custom classification models.

5. The code maintains global state using Streamlit's `st.session_state` for variables like the current frame, a pause flag, an incident log, and a timer reset.

6. It creates a video for incidents and adds entries to the incident log when an incident is detected, and the timer reset is triggered.

7. The "start" function initiates the main workflow, and the "pause" function allows you to pause the processing.

8. Streamlit is used to create a user interface with options to start and pause the processing, display live video streams, and show an incident log.

The code is well-structured and appears to be designed for real-time incident detection from video sources. 
