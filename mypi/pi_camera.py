import cv2
from datetime import datetime
import requests
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from door import open_door

LANDMARKER_FILE = './model/shape_predictor_68_face_landmarks.dat'
CASC_PATH = './haarcascade_frontalface_default.xml'
URL = 'http://localhost:5000/recognition'

face_cascade = cv2.CascadeClassifier(CASC_PATH)

cap = cv2.VideoCapture(0)
time.sleep(1)

NUM_FRAME = 30


i = 1
count = 0
while i < NUM_FRAME:
    i += 1
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        # if cv2, there should be a flag,cv3 is not need that
    )
    if len(faces) > 0:
        # find face, save and post
        print('find face')
        now = datetime.now()
        img_name = '{}-{}-{}-{}-{}_{}.jpg'.format(
            now.year, now.month, now.day, now.hour, now.minute, count)
        count +=1
        cv2.imwrite(img_name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 59])
        r = requests.post(URL, files={'image': open(img_name, 'rb')})
        name = r.json().get('prediction')
        if name is 'unk':
            print('sorry, who are you?')
            print(r.text)
        else:
            print('welcome', name)
            # open_door()
            print('open door')


