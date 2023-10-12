import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import streamlit as st
from threading import Thread
import time
import datetime
import pandas as pd

#global variables
IDX = 0  #declare this globally so all functions are using the same frame index
COLORS = sv.ColorPalette.default()
Base_output_video_path = '/Users/tobieabel/PycharmProjects/Demo_4_Fight_and_Fall/'#for creating video alerts
Incident_log = [] #list to hold the info for displaying in Stremlit incident log table
Timer_Reset = 0 #timer to control when an incident has been declared, to avoid duplicate alerts
class Rules_Engine:
    def __init__(self, maximum: int, threshold: int):#when creating the instance, pass in the how many frames are needed(threshold) our of how many frames(Maximum)
        self.maximum = maximum
        self.threshold = threshold
        self.score = {}#a dictionary to contain a key of the IDX number, and the value of the 0 or 1 for whether people/incidents detected
        for i in range(self.maximum):
                self.score[i-self.maximum] = 0 #prepopulate the score dict with 0's, with the keys of negative numbers so they don't clash with th real IDX frames

    def Count_frames_rule(self,det:int): #can use same count rule for both object detection and classification rules
        self.det = det #whether the model recognised a person/incident (1) or not (0)
        #add detection to the list of detections
        self.score[IDX] = det
        #remove the earliest IDX
        self.score.pop(min(self.score))
        if sum(self.score.values()) >= self.threshold:#if the values are more than the threshold (5/10 for obj detection, 4/5 for incident)
            return True
        else:
            return False



class Main_Work_Flow:
    def __init__(self, vid_or_cam: list[str] = ["/Users/tobieabel/Desktop/Fighting and Falling 1.mov"], models: list[str] = ['yolov8s_class_custom.pt','yolov8s.pt'],
                 zones: list[np.ndarray] = [np.array([[640, 154],[0, 242],[0, 360],[640, 360]])]):
        self.vid_or_cam = vid_or_cam
        self.cam_or_vid_dict = {}  # Create a dictionary to hold each camera or video source
        for i in self.vid_or_cam:
            self.cam_or_vid_dict[i] = cv2.VideoCapture(i) # create VideoCapture object for each video or camera in the list passed into the Main_Work_Flow class
            amount_of_frames = self.cam_or_vid_dict[i].get(cv2.CAP_PROP_FRAME_COUNT) #get the number of frames for videos so you know how many IDX will go up to
        time.sleep(1)  # allow cameras to initialise

        self.models = models
        self.class_model = YOLO(self.models[0])
        self.object_det_model = YOLO(self.models[1])
        self.zones = zones
        self.frame_dict = {} #dictionary to hold buffer of frames which get turned into video when incident detected
        self.incident_flag = False #flag to capture if there is currently an incident in progress so we do not duplicate alerts
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100,thickness=2)

        self.obj_det_rule = Rules_Engine(maximum = 10, threshold = 5)
        self.class_rule = Rules_Engine(maximum=5, threshold=4)
    def Create_Frames(self) -> np.ndarray:#need to put this in seperate threads and use video generator to speed things up
        #if there are multiple videos or cameras, are you capturing frames from each and sending each to the same model?
        global IDX
        for vid in self.cam_or_vid_dict.values():#loop through the videocapture objects
            vid.set(1,IDX)
            ret, frame  = vid.read()#read the next frame
            self.frame_dict[IDX] = frame #add frame to the dict of frames to use for creating videos of incidents, with IDX as the key
        if len(self.frame_dict) > 200: #if frame_dict dict has more than 200 frames, remove the first one
            self.frame_dict.pop(min(self.frame_dict))

        return frame

    def Run_Models(self,frame:np.ndarray) ->sv.detection:
        obj_det_result = self.object_det_model.predict(frame, device = "mps", verbose = False, conf=0.4,iou=0.7)[0]#send every frame to YOLO model
        obj_detections = sv.Detections.from_ultralytics(obj_det_result)  # pass the results from the model to the sv libary as they are easier to manipulate
        obj_detections= obj_detections[obj_detections.class_id == 0]  # filter the list of detections so it only shows category '0' which is people
        obj_detections = self.tracker.update_with_detections(obj_detections)  # pass the detections through the tracker to add tracker ID as additional field to detections object

        if obj_detections:
            det = 1

        else:
            det = 0

        #send result to first rule to update obj detection score and decide if we need to run classification model
        Obj_Det_Score = self.Call_Rules_Engine(det=det, rule_type="obj_det")

        if Obj_Det_Score == True and Timer_Reset == 0: #if we have passed the obj det threshold and there is not a current Incident, then run same frame through classification model
            if IDX % 10 == 0:#run every 10th frame, not every frame, to speed things up
                class_results = self.class_model(frame,device='mps')
                for i in class_results:
                    classification = (i.probs.top1)
                    if classification == 1:#unfortunately the incident label is actually 0, but I need it to be one for my threshold rules engine to calculate properly
                        classification = 0
                    else:
                        classification = 1
                        print("Incident detected")
            else:
                classification = 2 #2 means I don't want to run the classification rules engine on this loop as its not the 10th frame
                class_results = None

        else: #No people detected or already reporting an Incident
            classification = 0 #not running the classification model so assume no incident on this frame
            class_results = None

        # send class model result to Incident rule everytime, or every 10th time if people are detected, to allow clas score to go back to 0
        if classification < 2:#don't run rules engine where classification is 2 as this means object detected but not 10th frame, so don't want any score recoreded
            Incident = self.Call_Rules_Engine(det=classification, rule_type="classification")
        else:
            Incident = False #not running incident rules engine so assume no incident.

        self.Reset_Timer(Incident)

        return obj_detections, class_results, Incident

    def Call_Rules_Engine(self, det: int = 0, rule_type:str = "obj_det") -> bool:#not sure I need this, but will keep it for now in case calling rules gets more complex
        if rule_type == "obj_det":
            score = self.obj_det_rule.Count_frames_rule(det)
            return score
        elif rule_type == "classification":
            score = self.class_rule.Count_frames_rule(det)
            return score
        #score in this function is a true or false flag for whether we have crossed the threshold of an incident

    def Reset_Timer(self,Incident: bool):#function to update Timer_Reset which controls when we record videos, display red border and send frames to class model
        global Timer_Reset
        if Incident == True and Timer_Reset == 0:
            Timer_Reset = 200

        elif Timer_Reset > 0:
            Timer_Reset -= 1

    def Create_Outputs(self, frame: np.ndarray, result: sv.Detections, class_results: sv.Detections, Incident:bool)-> tuple:
        annotated_frame = frame.copy()
        #if class_results: #write classification results on frame# commenting this out as its not useful info
        #    annotated_frame = class_results[0].plot() #need to plot classification results ahead of the obj detection results for some reason, otherwise annotations don't render correctly
        if result: #draw boxes and tracker on frame
            labels = [f"#{tracker_id}" for tracker_id in result.tracker_id]
            annotated_frame = self.box_annotator.annotate(annotated_frame, result,labels=labels)
            annotated_frame = self.trace_annotator.annotate(annotated_frame, result)

        if Timer_Reset > 0: #if there is an incident currently being reported
            annotated_frame = cv2.copyMakeBorder(annotated_frame, top=15, bottom=15, left=15, right=15,
                                             borderType=cv2.BORDER_CONSTANT, value=[0, 0, 255])
            annotated_frame = sv.draw_text(scene=annotated_frame, text=("Incident Detected"),
                                           text_anchor=sv.Point(x=100, y=100), text_scale=1, text_thickness=2,
                                           background_color=COLORS.colors[0], text_padding=40)

        if Timer_Reset == 200:#if this is a new incident then write frames stored in frame_dict to mp4 video and store on streamlit
            height, width, _ = annotated_frame.shape #get dimensions of the frames - in this case the shape was an odd 988 x 1740 which cv2.Videowriter doesn't seem to handle
            #so I resize the image below to 1920, 1080 which is the closest common resolution for mp4v, and hard coded that into video_writer variable
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')# Define the codec and create the video writer
            fps = 20  # Adjust as needed
            output_video_path = (Base_output_video_path + str(IDX) + ".mp4")
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))
            # Loop through the image files and write them to the video
            for k,v in self.frame_dict.items():
                image = cv2.resize(v,(1920,1080))#Need to resize the image to fit the resolution you have put into video_writer variable
                video = video_writer.write(image)# Write the image to the video writer
            Incident_log.append((IDX, datetime.datetime.now(), output_video_path))
            # Release the video writer
            video_writer.release()

        annotated_frame = sv.draw_text(scene=annotated_frame, text=("Frame " + str(IDX)),text_anchor=sv.Point(x=100, y=20), text_scale=1, text_thickness=2,background_color=COLORS.colors[1], text_padding=40)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)  # change the frame to RGB for Streamlit
        LiveStream.image(annotated_frame, channels="RGB", use_column_width=True)  # display in the streamlit app
        Incident_log_table.dataframe(pd.DataFrame(data=Incident_log,columns=["Index","Timestamp","Video"]),column_config={"Video":st.column_config.LinkColumn(width="large")},hide_index=True,)#update the table in streamlit
        return annotated_frame, Incident_log

def start():
    global IDX
    timer = 0
    start = Main_Work_Flow()
    while True:
        frame = start.Create_Frames()
        obj_detections, class_results, Incident = start.Run_Models(frame)
        annotated_frame, incident_log = start.Create_Outputs(frame, obj_detections, class_results, Incident)
        #cv2.imshow('test', annotated_frame)
        IDX += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

#front end using streamlit - run in terminal with command: streamlit run /Users/tobieabel/PycharmProjects/Demo_4_Fight_and_Fall/Main.py

with st.sidebar:
    st.button(':green[Start]',on_click=start)
    st.button('Stop',type = 'primary')

tab1, tab2 = st.tabs(["Live Stream", "Incident Logs"])
with tab1:
    st.header("CCTV Footage - Camera 1")
    LiveStream = st.image("YorkATS logo.png", width = 600)#placeholder for video stream
    st.subheader('Incident Log', divider='rainbow')
    Incident_log_table = st.dataframe(pd.DataFrame(data=None,columns=["Index","Timestamp","Video"]),hide_index=True,)

#start()

#if there is the invalid numeric value in matching.py script again, insert this code into line 32 to get rid of the NAN issues
#cost_matrix = np.nan_to_num(cost_matrix)






