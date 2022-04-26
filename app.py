import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
import keras
from streamlit_option_menu import option_menu
from tensorflow import keras
import av
from keras.preprocessing.image import img_to_array
from streamlit_lottie import st_lottie
import json
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode




#----------------------------------------------------------------------------------------------------------
# Loading animation in form of json:-
def load_lottiefile(filepath: str):
  with open(filepath, "r") as f:
    return json.load(f)


#-------------------------------------------------------------------------------------------------------------
#Loading the harcascade file to detect the face
try:
  face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Face Detection
except Exception:
  st.write("Error loading cascade classifiers")

#-------------------------------------------------------------------------------------------------------------
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


#---------------------------------------------------------------------------------------------------------------
emotion_dict = {0:'angry',5:'disgust',2:'fear',3:'happy',4:'neutral',6:'suprise',7:'sad'}

classifier=keras.models.load_model('model.h5')
#--------------------------------------------------------------------------------------------------------------
# Emotion class to process video to detect face and process to predict emotion:-



class Faceemotion(VideoTransformerBase):
  def transform(self, frame):
    img = frame.to_ndarray(format="bgr24")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
      cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
      roi_gray = img_gray[y:y + h, x:x + w]
      roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
      if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = classifier.predict(roi)[0]
        maxindex = int(np.argmax(prediction))
        finalout = emotion_dict[maxindex]
        output = str(finalout)
        label_position = (x, y)
        cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img
#-------------------------------------------------------------------------------------------------------------
# Main Page

def main():
  st.set_page_config(page_title='Moody', page_icon='😁',
                   layout='centered', initial_sidebar_state='expanded')
  st.title('Moody')
  html= """<div style="background-color:#ff9d9d;padding:0.3px">
                                    <h4 style="color:black;text-align:center;">
                                    Real time Facial Emotion detection application</h4>
                                    </div>
                                    </br>"""
  st.markdown(html, unsafe_allow_html=True)
  with st.sidebar:
    selected=option_menu(
            menu_title='Choose the Activity:-',  # required
            options=['Home','Webcam'],  # required
            icons=["clock", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "black", "font-size": "15px"},
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "3px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "red"},
            }, )
  if selected==('Home'):
    st.text('')
    st.text('')
    st.text('')
    lottie_coding1=load_lottiefile("66468-face-id-scan.json")
    st_lottie(lottie_coding1,speed=1,reverse=False,loop=True,quality="low", height=300,width=500,key=None)
    st.text('')
    st.text('')
    st.subheader('About the project')
    st.markdown("**Mooder is a web based application that can analyse the mood of a person on basis of its facial emotion.**")
    st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.st.write
                 2. Real time face emotion recognization.
                 """)
    st.subheader('About the dataset:-')
    st.markdown("""**FER 2013 contains approximately 30,000 facial RGB images of different expressions with size restricted to 48×48, and the main labels of it can be divided into 7 types:  The Disgust expression has the minimal number of images – 600, while other labels have nearly 5,000 samples each.**""")
    st.text('1. Happy')
    st.text('2. Sad')
    st.text('3. Angry')
    st.text('4. Disgust')
    st.text('5. Suprise')
    st.text('6. Neutral')
    
  elif(selected=='Webcam'):
    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
   
if __name__=='__main__':
  main()
