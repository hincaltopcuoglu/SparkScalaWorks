# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:45:43 2017

@author: Administrator
"""

# import the necessary packages
import keras
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import numpy as np
import argparse
import cv2

import keras.applications
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help='C:\\Users\\Administrator\\Downloads\\IMG_20170903_171833_213.jpg')
args = vars(ap.parse_args())
 
# load the original image via OpenCV so we can draw on it and display
# it to our screen later
orig = cv2.imread(args["image"])



from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions


import keras
from keras.preprocessing import image as image_utils
from keras.applications import imagenet_utils
from keras.applications import vgg16
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help='C:\\Users\\Administrator\\Downloads\\')
args = vars(ap.parse_args())



img=cv2.imread('C:\\Users\\Administrator\\Downloads\\IMG_20170903_171833_213.jpg',1)


image = image_utils.load_img(img, target_size=(224, 224))



from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications import imagenet_utils,preprocess_input
from keras.applications.resnet50 import preprocess_input, decode_predictions
model = ResNet50(weights='imagenet')

img_path = 'C:\\Users\\Administrator\\Downloads\\IMG_20170903_171833_213.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]





from keras.layers import MaxPooling2D

def convblock(cdim, nb, bits=3):
    L = []
    
    for k in range(1,bits+1):
        convname = 'conv'+str(nb)+'_'+str(k)
        L.append( Conv2D(cdim, 3, 3,
                                border_mode='same',
                                activation='relu',
                                name=convname) )
    
    L.append( MaxPooling2D((2, 2), strides=(2, 2)) )
    
    return L



from keras.models import Sequential
mdl = Sequential()

from keras.layers import Conv2D

for l in convblock(64, 1, bits=2):
    mdl.add(l)

for l in convblock(128, 2, bits=2):
    mdl.add(l)
        
for l in convblock(256, 3, bits=3):
    mdl.add(l)
            
for l in convblock(512, 4, bits=3):
    mdl.add(l)
            
for l in convblock(512, 5, bits=3):
    mdl.add(l)
