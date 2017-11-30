# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:33:45 2017

@author: Administrator
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



df=pd.read_csv('C:\\Users\\Administrator\\Desktop\\database_amir4.csv',sep=";",usecols=[0,1,2],dtype={"Date":object})

gb = df.groupby('Firm')
#make all the groups data frame in a dict
d = {}
for k in gb.groups:
 d[k] = pd.DataFrame(gb.get_group(k))
 
 
 
for key in d:  
    if len(d[key])==1:
        del d[key]
        

#sort it according to Date
for key in d:
    d[key]=d[key].groupby(['Firm']).apply(lambda x: x.sort_values(["Date"], ascending = True))



#drop column     
for key in d:
    d[key]=d[key].drop(d[key].columns[[0,1]], axis=1)
    
    
#dataset=d['AAP_Share']

# train_size = int(len(dataset) * 0.90)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
# print(len(train), len(test ))


# split into train and test sets
train_size={}
test_size={}
train={}
test={}
for key in d:
 train_size[key] = int(len(d[key]) * 0.90)
 test_size[key] = len(d[key]) - train_size[key]
 train[key], test[key] = d[key][0:train_size[key]], d[key][train_size[key]:len(d[key])]
 print(len(train[key]), len(test[key] ))


training_set={}
for key in train:
 training_set[key] = train[key].iloc[:,0:1].values

#training_set = train.iloc[:,0:1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

for key in training_set:
 training_set[key] = sc.fit_transform(training_set[key])

#training_set = sc.fit_transform(training_set)


# Creating a data structure with 20 timesteps and t+1 output
from collections import defaultdict

X_train = defaultdict(list)
y_train = defaultdict(list)
for key in training_set:
  for i in range(20, len(training_set[key])-1):
    X_train[key].append(training_set[key][i-20:i, 0])
    y_train[key].append(training_set[key][i+1, 0])
    
for key in X_train:   
 X_train[key], y_train[key] = np.array(X_train[key]), np.array(y_train[key])

# Reshaping
for key in X_train:
  X_train[key] = np.reshape(X_train[key], (X_train[key].shape[0], X_train[key].shape[1], 1))
 
wanted_keys = ('AAP_Share','ZOB_Share')
dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])
X_train=dictfilt(X_train, wanted_keys)
y_train=dictfilt(y_train, wanted_keys)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 3, input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
for key in X_train:
 regressor.fit(X_train[key], y_train[key], epochs = 100, batch_size = 32)

##############################
#dataset_test = pd.read_csv('C:\\Users\\Administrator\\Desktop\\database_amir5_w.csv',sep=";",usecols=[0,1,2],dtype={"Date":object})

# Adding a second LSTM layer


#test_set = pd.read_csv('C:\\Users\\Administrator\\Desktop\\database_amir5_w.csv',sep=";",usecols=[0,1,2],dtype={"Date":object})

#fb = test.groupby('Firm')
##make all the groups data frame in a dict
#h = {}
#for y in fb.groups:
# h[y] = pd.DataFrame(fb.get_group(y))
# 
 
 
#for key in h:  
#    if len(h[key])==1:
#        del h[key]
        

#sort it according to Date
#for key in h:
#    h[key]=h[key].groupby(['Firm']).apply(lambda x: x.sort_values(["Date"], ascending = True))



#drop column     
#for key in h:
#    h[key]=h[key].drop(h[key].columns[[0,1]], axis=1)
    
test_set={}
for key in test:
 test_set[key] = test[key].iloc[:,0:1].values

for key in test_set:  
    if len(test_set[key])==1:
        del test_set[key]
        
test_set=dictfilt(test_set, wanted_keys)

real_stock_price={}
for key in test_set:
 real_stock_price[key] = np.concatenate((training_set[key], test_set[key]), axis = 0)

# Getting the predicted stock price of 2017

from sklearn.preprocessing import Imputer

scaled_real_stock_price={}
for key in real_stock_price:
  scaled_real_stock_price[key] = sc.fit_transform(real_stock_price[key])
  
inputs = defaultdict(list)
for key in test_set:
 for i in range(len(training_set[key]), ((len(training_set[key])) + len(test_set[key]))):
    inputs[key].append(scaled_real_stock_price[key][i-20:i, 0])

for key in inputs:
 inputs[key] = np.array(inputs[key])
 inputs[key] = np.reshape(inputs[key], (inputs[key].shape[0], inputs[key].shape[1], 1))
 
predicted_stock_price={}
for key in inputs: 
 predicted_stock_price[key] = regressor.predict(inputs[key])
 

 
for key in predicted_stock_price:
 predicted_stock_price[key] = sc.inverse_transform(predicted_stock_price[key])

# Visualising the results
for key in real_stock_price:
 plt.plot(real_stock_price[key][len(training_set[key])-1:], color = 'red', label = 'Real Google Stock Price')
 plt.plot(predicted_stock_price[key], color = 'blue', label = 'Predicted Google Stock Price')
 plt.title('Google Stock Price Prediction')
 plt.xlabel('Time')
 plt.ylabel('Google Stock Price')
 plt.legend()
 plt.show()

#####################

import math
from sklearn.metrics import mean_squared_error

trainScore={}
for key in real_stock_price:
 trainScore[key] = math.sqrt(mean_squared_error(test_set[key][:,0], predicted_stock_price[key][:,0]))
 
 