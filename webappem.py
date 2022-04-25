# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:32:57 2022

@author: ASUS
"""


##EMOTION

import streamlit as st 
from PIL import Image
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.tools
from playsound import playsound
import moviepy.editor as mp
import cv2
import pandas as pd
import datetime
import cv2
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
import speech_recognition as sr 
import moviepy.editor as mp
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import moviepy.editor as mymovie

import tempfile
import plotly.express as px



# importing libraries 
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

# create a speech recognition object


# a function that splits the audio file into chunks
# and applies speech recognition

# hide_streamlit_style = """
# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style>

# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 






def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    t=0
    r = sr.Recognizer()
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                pass
                #print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                st.write("Conversation", t , ":", text)
                t+=1
                
      
                whole_text += text
    # return the text for all chunks detected
    # return whole_text
        






# parameters for loading data and images
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'VGG16_EMOTION_CLASSIFIER_7_emo_TEST_FINAL.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "neutral", "sad", "surprised"]






st.markdown("<h1 style='text-align: center; color: #99A3A4;'>EMOTION CLASSIFIER</h1>", unsafe_allow_html=True)



my_placeholder = st.empty()
my_placeholder1 = st.empty()




f = st.file_uploader("Upload file")
if st.button("PLAY VIDEO"):
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
        st.video(tfile.name)

import tempfile


if st.button("VIDEO ANALYSIS"):    
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
    
        camera = cv2.VideoCapture(tfile.name)
        
       # ---------
       

        
        clip = mp.VideoFileClip(tfile.name)
        
        clip.audio.write_audiofile("converted.wav")
        

        # count the number of frames
        frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = int(camera.get(cv2.CAP_PROP_FPS))
          
        # calculate dusration of the video
        seconds = int(frames / fps)
        video_time = str(datetime.timedelta(seconds=seconds))
        print("duration in seconds:", seconds)
        print("video time:", video_time)
        print("FPS",fps)
        print("Frames",frames)
        

        
        
        frame_no = 0
        k=0
        dic={}
        df = []
        # always inserting new rows at the first position - last row will be always on top 
        data=pd.DataFrame({'Emotion': ['Neutral'], 'Timestamp': [0]})

        
        while(camera.isOpened()):
            
            cf = camera.get(cv2.CAP_PROP_POS_FRAMES)-.1
            camera.set(cv2.CAP_PROP_POS_FRAMES, cf+30)
        
            
            ret, frame = camera.read()
        
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
                
            else:
                
        #         print("for frame : " + str(frame_no) + "   timestamp is: ", str(camera.get(cv2.CAP_PROP_POS_MSEC)/(60*1000)),"cf:",cf+30)
                
                
            
        
        
                #reading the frame
                frame = imutils.resize(frame,width=300)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detection.detectMultiScale(gray,scaleFactor=1.4,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        
                canvas = np.zeros((250, 300, 3), dtype="uint8")
                frameClone = frame.copy()
                if len(faces) > 0:
                    faces = sorted(faces, reverse=True,
                    key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                    (fX, fY, fW, fH) = faces
        
                    roi = frame[fY:fY + fH, fX:fX + fW]
                    roi = cv2.resize(roi, (224, 224))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
        
        
                    preds = emotion_classifier.predict(roi)[0]
                    emotion_probability = np.max(preds)
                    label = EMOTIONS[preds.argmax()]
                    
                    dic[round(camera.get(cv2.CAP_PROP_POS_MSEC)/(60*1000),3)]= label
                
                    df2 = {'Emotion': label, 'Timestamp': round(camera.get(cv2.CAP_PROP_POS_MSEC)/(60*1000),3)}
                    data = data.append(df2, ignore_index = True) 
                    data["Time difference"]= data["Timestamp"].diff()
                    data["Time difference"]=data["Time difference"].replace(np.nan,0)
                    
        
        
        
                else:continue
                    
                frame_no+=1
        
        
        
        
                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                            # construct the label text
                            text = "{}: {:.2f}%".format(emotion, prob * 100)
        
                            # draw the label + probability bar on the canvas
                           # emoji_face = feelings_faces[np.argmax(preds)]
        
        
                            w = int(prob * 300)
                            cv2.rectangle(canvas, (7, (i * 35) + 5),
                            (w, (i * 35) + 35), (0, 0, 255), -1)
                            cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)
                            cv2.putText(frameClone, label, (fX, fY - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                          (0, 0, 255), 2)

                
                my_placeholder.image(frameClone, use_column_width=False,channels="BGR",width=600)

                my_placeholder1.image(canvas, use_column_width=False,channels="BGR",width=300)
                

        
        
        camera.release()
        cv2.destroyAllWindows()
        data = {'Emotion': list(dic.values()) , 'Timestamp': list(dic.keys())}
        data=pd.DataFrame.from_dict(data) 
                
        fig = px.line(data, x='Timestamp', y="Emotion")
        st.plotly_chart(fig, use_container_width=True)
        files = glob.glob('audio-chunks/*')
        for f in files:
            os.remove(f)
        
 
      


       
if st.button("SPEECH TO TEXT"):
    

    get_large_audio_transcription("converted.wav")
    
    

