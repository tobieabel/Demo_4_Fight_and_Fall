import time
import streamlit as st
import Incident_Detection
import supervision as sv
import cv2
import numpy as np
import pandas as pd

if 'rerun' not in st.session_state: #adding this in as streamlit seems to duplicate UI elements from the main page when first coming to this page
    st.session_state['rerun'] = False #so I've created this flag, and then at end of script I use st.rerun() to effectively click the page a second time
if 'frame' not in st.session_state:
    st.session_state['frame'] = "YorkATS logo.png"
if 'IDX' not in st.session_state:
    st.session_state['IDX'] = 0  #declare this globally so all functions are using the same frame index
if 'pause' not in st.session_state:
    st.session_state['pause'] = False
if 'Incident_log' not in st.session_state:
    st.session_state['Incident_log'] = [] #list to hold the info for displaying in Stremlit incident log table
if 'Timer_Reset' not in st.session_state:
    st.session_state['Timer_Reset']= 0 #timer to control when an incident has been declared, to avoid duplicate alerts

if 'Intruder' not in st.session_state:
    st.session_state['Intruder'] = False
def initiate_poylgon_zones(polygons:list[np.ndarray],frame_resolution_wh:tuple[int,int] = [640,480],triggering_position:sv.Position=sv.Position.CENTER)->list[sv.PolygonZone]:
    return[sv.PolygonZone(polygon,frame_resolution_wh,triggering_position)for polygon in polygons]

st.header("Monitoring out of Bounds Zones")
LiveStream_OOB = st.image(st.session_state.frame, width=640)
st.subheader('Incident Log', divider='rainbow')
Incident_log_table_OOB = st.dataframe(pd.DataFrame(data=st.session_state.Incident_log,columns=["Index","Timestamp","Video"]),column_config={"Video":st.column_config.LinkColumn(width="large")},hide_index=True,)

def start_OOB():
    if 'start' not in st.session_state:
        start = Incident_Detection.Main_Work_Flow()
        st.session_state['start'] = start
    while True:
        #st.session_state.Timer_Reset = 0 #use this line to start timer again when starting out of bounds module
        frame = st.session_state.start.Create_Frames() #st.session_state.start is the name of the Main_Work_flow instance from Fighting and Falling.py
        if frame is None:
            break
        results = st.session_state.start.Run_Models(frame,Out_of_Bounds=True) #returns the supervision detections object once passed through the object detections model

        frame = sv.draw_polygon(frame, st.session_state.start.zones[0], Incident_Detection.COLORS.colors[0]) #draw the out of bounds zone
        frame = st.session_state.start.box_ellipse_annotator.annotate(frame, results) #draw elipses on anyone detected in frame
        annotated_frame = frame.copy() #make copy at this point so the video frames have the polygon on but not other annotations

        #create supervision polygon object to use for detecting people within a zone
        zone = initiate_poylgon_zones(st.session_state.start.zones)
        detections = zone[0].trigger(results)#just taking the 0 index because I know there is only 1 zone.  Returns a boolean array of true/false for each detection as to whether its in the zone or not

        if np.any(detections): #check if any of the detections were in the zone(i.e. any of the boolean values returned = True
            st.session_state.Intruder = True
            det = 1 #for rules engine we pass in the detection results, 1 means true, 0 means false
        else:
            det = 0

        #send results to object detection rule.  return value is true is threshold passed for out of bounds (5/10 frames)
        Out_of_bounds_Score = st.session_state.start.Call_Rules_Engine(det=det, rule_type="out_of_bounds")

        st.session_state.start.Reset_Timer(Out_of_bounds_Score) #call reset timer to either decrease timer or reset it if new incident found

        if Out_of_bounds_Score is True:
            annotated_frame = cv2.copyMakeBorder(annotated_frame, top=15, bottom=15, left=15, right=15,
                                                 borderType=cv2.BORDER_CONSTANT, value=[0, 0, 255])
            annotated_frame = sv.draw_text(scene=annotated_frame, text=("INTRUDER ALERT!"),
                                           text_anchor=sv.Point(x=70, y=70), text_scale=0.5, text_thickness=1,
                                           background_color=Incident_Detection.COLORS.colors[0], text_padding=20)

        if st.session_state.Timer_Reset == 300: #new incident found - create video
            annotated_frame = st.session_state.start.Create_Video(annotated_frame)

        #annotated_frame = sv.draw_text(scene=annotated_frame, text=("Frame " + str(st.session_state.IDX)),text_anchor=sv.Point(x=50, y=20), text_scale=0.5, text_thickness=1,background_color=Incident_Detection.COLORS.colors[1], text_padding=20)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)#switch colour scheme to display in streamlit

        st.session_state.frame = annotated_frame
        st.session_state.IDX += 1
        st.session_state.Intruder = False


        LiveStream_OOB.image(st.session_state.frame,channels="RGB",width=640)
        if st.session_state.Timer_Reset == 300:
            time.sleep(3)
        Incident_log_table_OOB.dataframe(
            pd.DataFrame(data=st.session_state.Incident_log, columns=["Index", "Timestamp", "Video"]),
            column_config={"Video": st.column_config.LinkColumn(width="large")},
            hide_index=True, )  # update the table in streamlit

with st.sidebar:
    st.button(':green[Start]',key='start_OOB', on_click=start_OOB)
    st.button('Pause', type = 'primary', key='pause_OOB', on_click=Incident_Detection.pause())

if st.session_state.rerun is False:
    st.session_state.rerun = True
    st.rerun() #rerun the script when you first come to this page (st.sessionstate.rerun = false) to get around issue of UI elements from main page duplicating

#front end using streamlit - run in terminal with command: streamlit run /Users/tobieabel/PycharmProjects/Demo_4_Fight_and_Fall/Incident_Detection.py
