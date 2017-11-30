# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:58:38 2017

@author: Administrator
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('C:\\Users\\Administrator\\Downloads\\IMG_20170903_171833_213.jpg',1)

small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

plt.imshow(img)


face_cascade = cv2.CascadeClassifier('C:/Users/Administrator/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Administrator/Desktop/haarcascade_eye.xml')

gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(small,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = small[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


cv2.imshow('img',small)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite( "C:/Users/Administrator/Desktop/hincal_ada2.jpg", small )



faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)


print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(small, (x, y), (x+w, y+h), (0, 255, 0), 2)
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow("Faces found", small)
cv2.waitKey(0)
cv2.imwrite( "C:/Users/Administrator/Desktop/hincal_ada.jpg", small )