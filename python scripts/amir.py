# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:45:10 2017

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

#df.dropna(inplace=True)
#indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#df2=df['Close'][indices_to_keep].astype(np.float64)
#
#df_c = pd.concat([df, df2], axis=1)
#
#df_c.columns = ['Firm','Date','Close','Closex']
#
#df3=df_c.drop(df_c.columns[[2]], axis=1)
#
scaler = MinMaxScaler(feature_range=(0, 1))

#fit transform
df['CloseX']=df.groupby(['Firm']).Close.transform(scaler.fit_transform)

df.head(5)
max(df['CloseX'])
min(df['CloseX'])



df=df.drop(df.columns[[2]], axis=1)
#df2=df.groupby(['Firm'])
#np.isnan(df.iloc[:,2].all())
#
#np.where(df.iloc[:,2].values >= np.finfo(np.float64).max)



from sklearn.cross_validation import train_test_split

def stratify_train_test(df, stratifyby, *args, **kwargs):
    train, test = pd.DataFrame(), pd.DataFrame()
    gb = df.groupby(stratifyby)
    for k in gb.groups:
        traink, testk = train_test_split(gb.get_group(k), *args, **kwargs)
        train = pd.concat([train, traink])
        test = pd.concat([test, testk])
    return train, test

train, test = stratify_train_test(df, 'Firm', test_size=.5)

train2=train.groupby(['Firm']).apply(lambda x: x.sort_values(["Date"], ascending = True))

test2=test.groupby(['Firm']).apply(lambda x: x.sort_values(["Date"], ascending = True))



def create_dataset(dataset, xx, look_back=1):
      dataX, dataY = [], []
      gb = dataset.groupby(xx)
      for k in gb.groups:
          for i in range(len(gb.get_group(k))-1):
           a = gb.get_group(k).iloc[i:(i+1), 2:3]
           dataX.append(a)
           dataY.append(gb.get_group(k).iloc[i+1, 2:3])
      return pd.DataFrame(dataX), pd.DataFrame(dataY)



trainX, trainY = create_dataset(train2,'Firm' ,look_back)

testX, testY = create_dataset(test2,'Firm' ,look_back)



# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train2,  epochs=100, batch_size=1, verbose=2)



##########################################################################################################################################

#make all the groups data frame in a dict
d = {}
for k in gb.groups:
 d[k] = pd.DataFrame(gb.get_group(k))




d['AAP_Share']
d2['AAP_Share']
dataset['AAP_Share']

#sort it according to Date
for key in d:
    d[key]=d[key].groupby(['Firm']).apply(lambda x: x.sort_values(["Date"], ascending = True))

#prepare for scaling
for key in d:
    d[key].dropna(inplace=True)
    indices_to_keep = ~d[key].isin([np.nan, np.inf, -np.inf]).any(1)
    d[key]['Close'][indices_to_keep].astype(np.float64)


#scaler
scaler = MinMaxScaler(feature_range=(0, 1))

#fit transform
for key in d:
    d[key]['CloseX']=d[key].groupby(['Firm']).Close.transform(scaler.fit_transform)
    
    
    
#drop column     
for key in d:
    d[key]=d[key].drop(d[key].columns[[2]], axis=1)
    
#drop other columns , left only CloseX     
d2 = {}    
for key in d:    
    d2[key]=d[key].drop(d[key].columns[[0,1]], axis=1)
  

#make np array 
dataset = {}  
for key in d2:    
    dataset[key] = d2[key].values
    dataset[key] = dataset[key].astype('float32')
    
    


# split into train and test sets
train_size={}
test_size={}
train = {}  
test = {}  
for key in dataset:  
    train_size[key] = int(len(dataset[key]) * 0.67)
    test_size[key] = len(dataset[key]) - train_size[key]
    train[key], test[key] = dataset[key][0:train_size[key],:], dataset[key][train_size[key]:len(dataset[key]),:]
    print(len(train[key]), len(test[key]))
    
    
 
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


trainX={}
trainY={}
testX = {}  
testY = {}  
for key in train:
 #for key in test:  
    look_back = 1
    trainX[key], trainY[key] = create_dataset(train[key], 1)
    testX[key], testY[key] = create_dataset(test[key], 1)
    
#for problem in shape
for key in testX:  
    print(len(testX[key]))

for key in testX:  
    if len(testX[key])==0:
        del testX[key]
        continue

# reshape input to be [samples, time steps, features]
for key in trainX:
    trainX[key] = np.reshape(trainX[key], (trainX[key].shape[0], 1, trainX[key].shape[1]))

for key in testX:  
    testX[key] = np.reshape(testX[key], (testX[key].shape[0], 1, testX[key].shape[1]))

  


# create and fit the LSTM network

for key in trainX:
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX['AAP_Share'], trainY['AAP_Share'], epochs=100, batch_size=1, verbose=2)
    
    
    
    

# make predictions
trainPredict = model.predict(trainX['AAP_Share'])
testPredict = model.predict(testX['AAP_Share'])
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY['AAP_Share']])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY['AAP_Share']])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))



# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset['AAP_Share'])
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset['AAP_Share'])
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset['AAP_Share'])-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset['AAP_Share']))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


######################################################################################################

#other solution

df=pd.read_csv('C:\\Users\\Administrator\\Desktop\\database_amir4.csv',sep=";",usecols=[0,1,2],dtype={"Date":object})

gb = df.groupby('Firm')
#make all the groups data frame in a dict
d = {}
for k in gb.groups:
 d[k] = pd.DataFrame(gb.get_group(k))




d['AAP_Share']
d2['AAP_Share']
dataset['AAP_Share']


#for problem in shape
for key in d:  
    print(len(d[key]))

for key in d:  
    if len(d[key])==1:
        del d[key]
        

#sort it according to Date
for key in d:
    d[key]=d[key].groupby(['Firm']).apply(lambda x: x.sort_values(["Date"], ascending = True))
    
    
    
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        window.index = range(30)
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

for key in d:

d['AAP_Share'].shape[1]

sequence_length = 30
result={}
for index in range(len(d['AAP_Share']['Close']) - sequence_length):
  result['AAP_Share'].append(d['AAP_Share']['Close'][index: index + sequence_length])
 

result = normalise_windows(result)
result = np.array(result)

#########################################################################################################3

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
    d[key]=d[key].drop(d[key].columns[[0]], axis=1)
    
    

#prepare for scaling
for key in d:
    d[key].dropna(inplace=True)
    indices_to_keep = ~d[key].isin([np.nan, np.inf, -np.inf]).any(1)
    d[key]['Close'][indices_to_keep].astype(np.float64)


#scaler
scaler = MinMaxScaler(feature_range=(0, 1))

#fit transform
for key in d:
    d[key]['CloseX']=d[key].groupby(['Firm']).Close.transform(scaler.fit_transform)
    
    
    
#drop column     
for key in d:
    d[key]=d[key].drop(d[key].columns[[1]], axis=1)


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg



values={}
for key in d:
 values[key] = d[key].values
 
from sklearn import preprocessing
# integer encode direction
encoder = preprocessing.LabelEncoder()

for key in values:
 values[key][:,0] = encoder.fit_transform(values[key][:,0])
 
for key in values:
 values[key] = values[key].astype('float32')
 
reframed={}
for key in values: 
 reframed[key] = series_to_supervised(values[key], 1, 30)
 
 
for key in reframed:  
 reframed[key].drop(reframed[key].columns[[0]], axis=1, inplace=True)
 
 
  
for key in reframed:  
 reframed[key].drop(reframed[key].columns[[1,2,3,4,5,6,7,8,9]], axis=1, inplace=True)
 
 
for key in reframed: 
 values[key] = reframed[key].values
 
for key in values:  
    if len(values[key])==0:
        del values[key] 
      
train_size={}       
for key in values:
 train_size[key] = int(len(values[key]) * 0.67)

train={}
test={}
for key in values:
 train[key], test[key] = values[key][0:train_size[key] ,:], values[key][train_size[key] :len(values[key]),:]
 
 
# split into input and outputs
train_X={}
train_y={}
for key in train:
 train_X[key], train_y[key] = train[key][:, :-1], train[key][:, -1]


test_X={}
test_y={}
for key in test:
 test_X[key], test_y[key] = test[key][:, :-1], test[key][:, -1]



# reshape input to be 3D [samples, timesteps, features]
for key in train_X:
 train_X[key] = train_X[key].reshape((train_X[key].shape[0], 1, train_X[key].shape[1]))
 
for key in test_X:
 test_X[key] = test_X[key].reshape((test_X[key].shape[0], 1, test_X[key].shape[1]))

for key in train_X:
 print(train_X[key].shape, train_y[key].shape, test_X[key].shape, test_y[key].shape)
 
 


# design network
model = Sequential()

for key in train_X:
 model.add(LSTM(50, input_shape=(train_X[key].shape[1], train_X[key].shape[2])))


model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
 

 
 
 