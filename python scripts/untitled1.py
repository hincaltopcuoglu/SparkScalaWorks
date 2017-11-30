# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:21:43 2017

@author: Administrator
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt





face_cascade = cv2.CascadeClassifier('C:/Users/Administrator/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Administrator/Desktop/haarcascade_eye.xml')


cascade_file_src = "C:/Users/Administrator/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_file_src)


img=cv2.imread('C:\\Users\\Administrator\\Downloads\\IMG_20170903_171833_213.jpg',1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces in the image :
faces = faceCascade.detectMultiScale(gray, 1.2, 5)

# draw rectangles around the faces :
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(img)


# crop image and scale to 224x224 :
crpim = im.crop(box).resize((224,224))
plt.imshow(np.asarray(crpim))


from keras.models import Model
# output the 2nd last layer :
featuremodel = Model( input = facemodel.layers[0].input,
                      output = facemodel.layers[-2].output )