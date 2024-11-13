
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
import numpy as np
from collections import defaultdict
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys
from tkinter import ttk
import os


main = tkinter.Tk()
main.title("Displaying Emoji Based Facial Expressions")
main.geometry("1200x1200")

global value
global filename
global faces
global frame
detection_model_path = 'models/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprise","neutral"]
global songslist

def upload():
    global filename
    global value
    filename = askopenfilename(initialdir = "images")
    pathlabel.config(text=filename)
        

def preprocess():
    global filename
    global frame
    global faces
    text.delete('1.0', END)
    orig_frame = cv2.imread(filename)
    orig_frame = cv2.resize(orig_frame, (48, 48))
    frame = cv2.imread(filename,0)
    faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    text.insert(END,"Total number of faces detected : "+str(len(faces)))
    
def detectExpression():
    global faces
    if len(faces) > 0:
       faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
       (fX, fY, fW, fH) = faces
       roi = frame[fY:fY + fH, fX:fX + fW]
       roi = cv2.resize(roi, (48, 48))
       roi = roi.astype("float") / 255.0
       roi = img_to_array(roi)
       roi = np.expand_dims(roi, axis=0)
       preds = emotion_classifier.predict(roi)[0]
       emotion_probability = np.max(preds)
       label = EMOTIONS[preds.argmax()]
       img = cv2.imread('Emoji/'+label+".png")
       img = cv2.resize(img, (600,400))
       cv2.putText(img, "Facial Expression Detected As : "+label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
       cv2.imshow("Facial Expression Detected As : "+label, img)
       cv2.waitKey(0)       
    else:
       messagebox.showinfo("Facial Expression Prediction Screen","No face detceted in uploaded image")

def detectfromvideo(image):
    result = 'none'
    temp = image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    output = "none"
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = temp[fY:fY + fH, fX:fX + fW]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        output = label     
    return output

def detectWebcamExpression():
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        height, width, channels = img.shape
        result = detectfromvideo(img)
        if result != 'none':
            print(result)
            img1 = cv2.imread('Emoji/'+result+".png")
            img1 = cv2.resize(img1, (width,height))
            cv2.putText(img1, "Facial Expression Detected As : "+result, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
            cv2.imshow("Emoji Output",img1)
        cv2.putText(img, "Facial Expression Detected As : "+result, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        cv2.imshow("Facial Expression Output", img)
        if cv2.waitKey(650) & 0xFF == ord('q'):
            break   
    cap.release()
    cv2.destroyAllWindows()

font = ('times', 20, 'bold')
title = Label(main, text='Displaying Emoji Based Facial Expressions')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Image With Face", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=300,y=100)

preprocessbutton = Button(main, text="Preprocess & Detect Face in Image", command=preprocess)
preprocessbutton.place(x=50,y=150)
preprocessbutton.config(font=font1) 

emotion = Button(main, text="Detect Facial Expression", command=detectExpression)
emotion.place(x=50,y=200)
emotion.config(font=font1)

emotion = Button(main, text="Detect Facial Expression from WebCam", command=detectWebcamExpression)
emotion.place(x=50,y=250)
emotion.config(font=font1) 

font1 = ('times', 12, 'bold')
text=Text(main,height=10,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
