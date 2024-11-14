from tkinter import messagebox, simpledialog, filedialog
from tkinter import *
import tkinter
from tkinter import ttk
import cv2
import os
import imutils
from imutils import paths
import numpy as np
from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Initialize main application window
main = tkinter.Tk()
main.title("Display Emoji-Based Facial Expression Detection")
main.geometry("1000x800")
main.resizable(True, True)

# Set default theme colors
default_bg = '#282c34'
default_fg = '#ffffff'
main.configure(bg=default_bg)

# Style configuration
style = ttk.Style()
style.theme_use('clam')
style.configure('TButton', background='#61afef', foreground='white', font=('Helvetica', 12, 'bold'))

# Load models and set emotion labels
detection_model_path = 'models/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprise", "neutral"]

# Global variables
filename = None
faces = None
frame = None

# Functions for the application
def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="images", title="Select an Image",
                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    if filename:
        pathlabel.config(text=filename)
        status_bar.config(text="Image uploaded successfully")
    else:
        status_bar.config(text="Image upload canceled")

def preprocess():
    global filename, frame, faces
    text.delete('1.0', END)
    if not filename:
        messagebox.showerror("Error", "Please upload an image first.")
        return
    frame = cv2.imread(filename, 0)
    faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    text.insert(END, f"Total number of faces detected: {len(faces)}\n")
    status_bar.config(text="Preprocessing completed")

def detectExpression():
    global faces, frame
    if frame is None:
        messagebox.showerror("Error", "Please upload and preprocess an image first.")
        return
    if faces is None or len(faces) == 0:
        messagebox.showerror("Error", "No face detected. Please upload an image with a clear face.")
        return

    (fX, fY, fW, fH) = faces[0]
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    try:
        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred while predicting emotion: {e}")
        return

    try:
        img_path = f'Emoji/{label}.png'
        if not os.path.exists(img_path):
            messagebox.showerror("File Error", f"Emoji image for '{label}' not found at path: {img_path}")
            return

        img = cv2.imread(img_path)
        img = cv2.resize(img, (600, 400))
        cv2.putText(img, f"Facial Expression Detected As: {label}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(f"Facial Expression Detected As: {label}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        status_bar.config(text=f"Expression detected: {label}")
        
    except Exception as e:
        messagebox.showerror("Display Error", f"Could not display emoji image: {e}")

def detectfromvideo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    output = "none"
    if len(faces) > 0:
        (fX, fY, fW, fH) = faces[0]
        roi = image[fY:fY + fH, fX:fX + fW]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        output = EMOTIONS[preds.argmax()]
    return output

def detectWebcamExpression():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        result = detectfromvideo(img)
        if result != 'none':
            emoji_img = cv2.imread(f'Emoji/{result}.png')
            emoji_img = cv2.resize(emoji_img, (img.shape[1], img.shape[0]))
            cv2.putText(emoji_img, f"Facial Expression Detected As: {result}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Emoji Output", emoji_img)
        cv2.putText(img, f"Facial Expression Detected As: {result}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Facial Expression Output", img)
        if cv2.waitKey(650) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    status_bar.config(text="Webcam expression detection completed")

def showAbout():
    messagebox.showinfo("About", "Display Emoji-Based Facial Expression Detection\nVersion 2.0\nDeveloped by Karthik")

def exitApp():
    main.quit()

def saveResults():
    results = text.get('1.0', END)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             initialfile=f"results_{timestamp}.txt",
                                             filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if save_path:
        with open(save_path, 'w') as file:
            file.write(results)
        status_bar.config(text="Results saved successfully")

def changeTheme():
    new_bg = simpledialog.askstring("Theme", "Enter background color (e.g., #ffffff for white):", initialvalue=default_bg)
    new_fg = simpledialog.askstring("Theme", "Enter foreground color (e.g., #000000 for black):", initialvalue=default_fg)
    if new_bg and new_fg:
        main.config(bg=new_bg)
        for widget in main.winfo_children():
            widget.config(bg=new_bg, fg=new_fg)
        status_bar.config(text="Theme changed successfully")

# Adding Menu Bar
menubar = Menu(main)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Upload Image", command=upload)
filemenu.add_command(label="Save Results", command=saveResults)
filemenu.add_command(label="Change Theme", command=changeTheme)
filemenu.add_command(label="Exit", command=exitApp)
menubar.add_cascade(label="File", menu=filemenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=showAbout)
menubar.add_cascade(label="Help", menu=helpmenu)
main.config(menu=menubar)

# UI Elements
font = ('Helvetica', 20, 'bold')
Label(main, text='Advanced Emoji-Based Facial Expression Detection', bg=default_bg, fg=default_fg, font=font, height=3).grid(row=0, column=0, padx=10, pady=10)

font1 = ('Helvetica', 14, 'bold')
ttk.Button(main, text="Upload Image With Face", command=upload).grid(row=1, column=0, sticky="w", padx=10, pady=10)

pathlabel = Label(main, bg='brown', fg='white', font=font1)
pathlabel.grid(row=1, column=0, sticky="w", padx=300)

ttk.Button(main, text="Preprocess & Detect Face in Image", command=preprocess).grid(row=2, column=0, sticky="w", padx=10, pady=10)
ttk.Button(main, text="Detect Facial Expression", command=detectExpression).grid(row=3, column=0, sticky="w", padx=10, pady=10)
ttk.Button(main, text="Detect Facial Expression from WebCam", command=detectWebcamExpression).grid(row=4, column=0, sticky="w", padx=10, pady=10)

# Text box for output results
text = Text(main, height=5, width=160, font=font1)
text.grid(row=5, column=0, sticky="nsew", padx=10, pady=10)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
scroll.pack(side=RIGHT, fill=Y)

# Status Bar
status_bar = Label(main, text="Welcome to Display Emoji-Based Facial Expression Detection", bd=1, relief=SUNKEN, anchor=W)
status_bar.grid(row=6, column=0, sticky="we", padx=10, pady=5)

# Start the GUI loop
main.mainloop()
