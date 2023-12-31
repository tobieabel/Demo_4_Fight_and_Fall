import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import streamlit as st
import time
import datetime
import pandas as pd



#global variables
if 'frame' not in st.session_state:
    st.session_state['frame'] = "YorkATS logo.png"
if 'IDX' not in st.session_state:
    st.session_state['IDX'] = 0  #declare this globally so all functions are using the same frame index
if 'pause' not in st.session_state:
    st.session_state['pause'] = False
COLORS = sv.ColorPalette.default()
Base_output_video_path = '/Users/tobieabel/PycharmProjects/Demo_4_Fight_and_Fall/'#for creating video alerts
if 'Incident_log' not in st.session_state:
    st.session_state['Incident_log'] = [] #list to hold the info for displaying in Stremlit incident log table
if 'Timer_Reset' not in st.session_state:
    st.session_state['Timer_Reset']= 0 #timer to control when an incident has been declared, to avoid duplicate alerts

class Rules_Engine:
    def __init__(self, maximum: int, threshold: int):#when creating the instance, pass in the how many frames are needed(threshold) out of how many frames(Maximum)
        self.maximum = maximum
        self.threshold = threshold
        self.score = {}#a dictionary to contain a key of the IDX number, and the value of the 0 or 1 for whether people/incidents detected
        for i in range(self.maximum):
                self.score[i-self.maximum] = 0 #prepopulate the score dict with 0's, with the keys of negative numbers so they don't clash with th real IDX frames

    def Count_frames_rule(self,det:int): #can use same count rule for object detection, classification rules and out_of_bounds rules
        self.det = det #whether the model recognised a person/incident (1) or not (0)
        #add detection to the list of detections
        self.score[st.session_state.IDX] = det
        #remove the earliest IDX
        self.score.pop(min(self.score))
        if sum(self.score.values()) >= self.threshold:#if the values are more than the threshold (5/10 for obj detection and Out_of_bounds, 4/10 for incident)
            return True
        else:
            return False



class Main_Work_Flow:
    def __init__(self, vid_or_cam: list[str] = ["/Users/tobieabel/Desktop/Fighting and Falling 5.mov"], models: list[str] = ['yolov8s_class_custom.pt','yolov8n_custom_peopleCounterV2.pt'],
                 zones: list[np.ndarray] = [np.array([[400,320],[150,320],[270, 120],[430, 120]])]):
        self.vid_or_cam = vid_or_cam
        self.cam_or_vid_dict = {}  # Create a dictionary to hold each camera or video source
        for count, i in enumerate(self.vid_or_cam):
            self.cam_or_vid_dict[i] = cv2.VideoCapture(i) # create VideoCapture object for each video or camera in the list passed into the Main_Work_Flow class
            self.cam_or_vid_dict[i].open(self.vid_or_cam[count])
            #amount_of_frames = self.cam_or_vid_dict[i].get(cv2.CAP_PROP_FRAME_COUNT) #get the number of frames for videos so you know how many IDX will go up to
        time.sleep(1)  # allow cameras to initialise

        self.models = models
        self.class_model = YOLO(self.models[0])
        self.object_det_model = YOLO(self.models[1])
        self.zones = zones
        self.frame_dict = {} #dictionary to hold buffer of frames which get turned into video when incident detected
        self.incident_flag = False #flag to capture if there is currently an incident in progress so we do not duplicate alerts
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(color=COLORS) #for showing straightforward bounding boxes
        self.box_corner_annotator = sv.BoxCornerAnnotator(color=COLORS) #for showing corners of bounding boxes
        self.box_ellipse_annotator = sv.EllipseAnnotator(color=COLORS) #for showing elipses instead of bounding boxes
        self.trace_annotator = sv.TraceAnnotator(color=COLORS, position=sv.Position.CENTER, trace_length=100,thickness=2)

        self.obj_det_rule = Rules_Engine(maximum = 10, threshold = 5)#set thresholds for how often you need to detect people before rule says there are people in the scene
        self.class_rule = Rules_Engine(maximum=10, threshold=4) #and same for classification that there is an incident.
        self.out_of_bounds_rule = Rules_Engine(maximum=10, threshold=5) #and same for out of bounds rule on Out_of_Boudns.py

    def Create_Frames(self) -> np.ndarray:#need to put this in separate threads and use video generator to speed things up
        #if there are multiple videos or cameras, are you capturing frames from each and sending each to the same model?
        global IDX
        for vid in self.cam_or_vid_dict.values():#loop through the videocapture objects
            if st.session_state["pause"] is True: # this is for when we have pause function, need session state, and move it out of loop as it adds a lot of processing time
                vid.set(1,st.session_state.IDX)
                st.session_state.pause = False #set the pause variable back to false so we do not continually execute this if statement, it is too time consuming
            ret, frame  = vid.read()#read the next frame

            try:
                frame = cv2.resize(frame, (640, 480)) # resize the images to speed up processing - remember to train images at same size
            except:
                frame = None #looks like we're at the end of the video so return none for frame
                return frame

            if st.session_state.IDX % 2 ==0:#only process every other frame to speed things up
                return frame
            else: #if missing out frames the rest of this block needs to be indented to fit in the else case
                self.frame_dict[st.session_state.IDX] = frame #add frame to the dict of frames to use for creating videos of incidents, with IDX as the key
                if len(self.frame_dict) > 200: #if frame_dict dict has more than 200 frames, remove the first one
                    self.frame_dict.pop(min(self.frame_dict))

                return frame


    def Run_Models(self,frame:np.ndarray, Out_of_Bounds:bool = False) ->sv.detection:
        obj_det_result = self.object_det_model.predict(frame, device = "mps", verbose = False, conf=0.4,iou=0.7)[0]#send every frame to YOLO model
        obj_detections = sv.Detections.from_ultralytics(obj_det_result)  # pass the results from the model to the sv libary as they are easier to manipulate
        obj_detections= obj_detections[obj_detections.class_id == 0]  # filter the list of detections so it only shows category '0' which is people
        obj_detections = self.tracker.update_with_detections(obj_detections)  # pass the detections through the tracker to add tracker ID as additional field to detections object

        if obj_detections:
            det = 1

        else:
            det = 0

        if Out_of_Bounds: #if this method has been invoked by the out of bounds page there is no need to run rules engine
            return obj_detections
        else:
        #send result to first rule to update obj detection score and decide if we need to run classification model
            Obj_Det_Score = self.Call_Rules_Engine(det=det, rule_type="obj_det")

            if Obj_Det_Score == True and st.session_state.Timer_Reset == 0: #if we have passed the obj det threshold and there is not a current Incident, then run same frame through classification model
                if st.session_state.IDX % 5 == 0:#run every 5th frame, not every frame, to speed things up.
                    class_results = self.class_model(frame,device='mps')
                    for i in class_results:
                        classification = (i.probs.top1)#get the highest confidence classification from the model
                        if classification == 1:#unfortunately the incident label is actually 0, but I need it to be one for my threshold rules engine to calculate properly
                            classification = 0
                        else:
                            classification = 1
                            print("Incident detected IDX " + str(st.session_state.IDX))
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
        elif rule_type == "out_of_bounds":
            score = self.out_of_bounds_rule.Count_frames_rule(det)
            return score
        #score in this function is a true or false flag for whether we have crossed the threshold of an incident

    def Reset_Timer(self,Incident: bool):#function to update Timer_Reset which controls when we record videos, display red border and send frames to class model
        global Timer_Reset
        if Incident == True and st.session_state.Timer_Reset == 0: #we pass in the incident flag (works same for classification and out of bounds)
            st.session_state.Timer_Reset = 300 #if there has been an incident detected and the rest is at 0 meaning we're not counting down from a previous incident

        elif st.session_state.Timer_Reset > 0: #otherwise reduce the reset time by 1
            st.session_state.Timer_Reset -= 1

    def Create_Video(self, annotated_frame):  # if this is a new incident then write frames stored in frame_dict to mp4 video and store on streamlit
            #height, width, _ = annotated_frame.shape  # get dimensions of the frames - in this case the shape was an odd 988 x 1740 which cv2.Videowriter doesn't seem to handle
            # so I resize the image below to 640, 480 which is an acceptable resolution for mp4v, and hard coded that into video_writer variable

        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Define the codec and create the video writer.  use *'avc1' rather than *'mp4' so it also plays in chrome browser
        fps = 20  # Adjust as needed
        output_video_path = (Base_output_video_path + str(st.session_state.IDX) + ".mp4")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 480))
        # Loop through the image files and write them to the video
        for k, v in self.frame_dict.items():
            video = video_writer.write(v)  # Write the image to the video writer
        st.session_state.Incident_log.append((st.session_state.IDX, datetime.datetime.now(), output_video_path))
        # Release the video writer
        video_writer.release()
        annotated_frame = sv.draw_text(scene=annotated_frame, text=("Creating Video"),
                                       text_anchor=sv.Point(x=70, y=110), text_scale=0.5, text_thickness=1,
                                       background_color=COLORS.colors[0], text_padding=20)

        return annotated_frame


    def Create_Outputs(self, frame: np.ndarray, result: sv.Detections, class_results: sv.Detections, Incident:bool)-> tuple:
            annotated_frame = frame.copy() #make annotations on a copy if you don't want them to appear in the video created for incidents

            if result:  # draw boxes and tracker on frame
                #labels = [f"#{tracker_id}" for tracker_id in result.tracker_id]
                #annotated_frame = self.box_annotator.annotate(annotated_frame, result, skip_label=True)
                annotated_frame = self.box_corner_annotator.annotate(annotated_frame, result)
                #annotated_frame = self.trace_annotator.annotate(annotated_frame, result)

            if st.session_state.Timer_Reset > 150:  # if there is an incident currently being reported
                annotated_frame = cv2.copyMakeBorder(annotated_frame, top=15, bottom=15, left=15, right=15,
                                                     borderType=cv2.BORDER_CONSTANT, value=[0, 0, 255])
                annotated_frame = sv.draw_text(scene=annotated_frame, text=("Incident Detected"),
                                               text_anchor=sv.Point(x=70, y=70), text_scale=0.5, text_thickness=1,
                                               background_color=COLORS.colors[0], text_padding=20)

            if st.session_state.Timer_Reset == 300:
                annotated_frame = self.Create_Video(annotated_frame)

            #annotated_frame = sv.draw_text(scene=annotated_frame, text=("Frame " + str(st.session_state.IDX)),text_anchor=sv.Point(x=50, y=20), text_scale=0.5, text_thickness=1,background_color=COLORS.colors[1], text_padding=20)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)  # change the frame to RGB for Streamlit
            LiveStream.image(annotated_frame, channels="RGB", width=640)  # display in the streamlit app
            if st.session_state.Timer_Reset == 300:
                time.sleep(3)  # leave 'creating video' message on screen for 3 seconds, just looks better for demo purposes
            Incident_log_table.dataframe(pd.DataFrame(data=st.session_state.Incident_log,columns=["Index","Timestamp","Video"]),column_config={"Video":st.column_config.LinkColumn(width="large")},hide_index=True,)#update the table in streamlit
            st.session_state.frame = annotated_frame
            return annotated_frame, st.session_state.Incident_log

#Front-end code
def start():
    global IDX
    start = Main_Work_Flow()
    if 'start' not in st.session_state:
        st.session_state['start'] = start
    while True:
        frame = start.Create_Frames()
        if frame is None:
            break
        if st.session_state.IDX % 2 == 0:#skip every other frame to speed things up
            st.session_state.IDX +=1
            continue
        else: #if missing out frames then the rest of this block needs to be indented to fit in the else case
            obj_detections, class_results, Incident = start.Run_Models(frame)
            annotated_frame, incident_log = start.Create_Outputs(frame, obj_detections, class_results, Incident)
            #cv2.imshow('test', annotated_frame)
            st.session_state.IDX += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

def pause():
    st.session_state["pause"] = True #this is used in Create_Frames() to go back to the relevant IDX frame when app is restarted, and its values is put back to False at that point

Container1 = st.container()
with Container1:
    with st.sidebar:
        st.button(':green[Start]',key='start_ff', on_click=start)
        st.button('Pause',type ='primary',key='pause_ff', on_click=pause)


    st.header("CCTV Footage - Camera 1")

    LiveStream = st.image(st.session_state.frame, width=640)#placeholder for video stream
    st.subheader('Incident Log', divider='rainbow')
    Incident_log_table = st.dataframe(pd.DataFrame(data=st.session_state.Incident_log,columns=["Index","Timestamp","Video"]),column_config={"Video":st.column_config.LinkColumn(width="large")},hide_index=True,)

#start()

#if there is the invalid numeric value in matching.py script again, insert this code into line 32 to get rid of the NAN issues
#cost_matrix = np.nan_to_num(cost_matrix)

#front end using streamlit - run in terminal with command: streamlit run /Users/tobieabel/PycharmProjects/Demo_4_Fight_and_Fall/Incident_Detection.py





