# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:54:40 2017

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Flatten

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)



# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
#df=pd.read_csv('C:\\Users\\Administrator\\Desktop\\database_amir4.csv',sep=";",usecols=[0,1,2],dtype={"Date":object})


df2=pd.read_csv('C:\\Users\\Administrator\\Desktop\\newtest\\KEAMA_Share_D.txt',sep=",",usecols=[0,1,5],dtype={"Date":object})

gb = df2.groupby('Firma')
#make all the groups data frame in a dict
d = {}
for k in gb.groups:
 d[k] = pd.DataFrame(gb.get_group(k))
 
 
 
for key in d:  
    if len(d[key])==1:
        del d[key]
        
        
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#sort it according to Date
for key in d:
    d[key]=d[key].groupby(['Firma']).apply(lambda x: x.sort_values(["Date"], ascending = True))



#drop column     
for key in d:
    d[key]=d[key].drop(d[key].columns[[0,1]], axis=1)

dataset=d['KEAMA_Share']

dataset = dataset.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.90)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
print(len(train), len(test ))

# reshape into X=t and Y=t+1
look_back = 50
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
#trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

from keras.layers import TimeDistributed

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units =32, return_sequences = True, input_shape = (look_back,1)))
model.add(LSTM(units = 32, return_sequences = True))
model.add(LSTM(units = 32, return_sequences = True))
model.add(LSTM(units = 32, return_sequences = True))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=25, batch_size=32, verbose=2)

###################################################################################
dataset_test = pd.read_csv('C:\\Users\\Administrator\\Desktop\\feolad_test2.txt')

test_set = scaler.fit_transform(dataset_test)

testX_2, testY_2 = create_dataset(test_set, look_back)

testX_2 = np.reshape(testX_2, (testX_2.shape[0], testX_2.shape[1], 1))

testPredict2 = model.predict(testX_2)

testPredict2 = scaler.inverse_transform(testPredict2)

plt.plot(testPredict2)
###################################################################################
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()




from keras.layers.recurrent import LSTM

#def predict_sequences_multiple(model, data, window_size, prediction_len):
#    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
#    prediction_seqs = []
#    for i in range(int(len(data)/(prediction_len))):
#        curr_frame = data[i*prediction_len]
#        predicted = []
#        for j in range(prediction_len):
#            predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
#            curr_frame = curr_frame[1:]
#            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
#        prediction_seqs.append(predicted)
#    return prediction_seqs
#
#from keras.layers import LSTM
#
#   #print(model.predict(curr_frame[np.newaxis,:,:]))
#def predict_sequences_multiple(model, firstValue,length):
#    prediction_seqs = []
#    curr_frame = firstValue    
#    for i in range(length): 
#        predicted = []     
#        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])        
#        curr_frame = curr_frame[0:]
#        curr_frame = np.insert(curr_frame[0:], i-1, predicted[-1], axis=0)
#        prediction_seqs.append(predicted[-1])
#    return prediction_seqs

  for i in range(int(len(data)/prediction_len)):
i*prediction_len

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        prediction_seqs = []  
        curr_frame = data[92]
        predicted = []
        #for i in range(int(len(data)/prediction_len)):
        for j in range(15):
            predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs
      
  

predictions = predict_sequences_multiple(model,testX,15,15)


inversed2=scaler.inverse_transform(np.array(predictions[14]).reshape(-1, 1))

plt.plot(inversed2)

 x1=pd.DataFrame(inversed2)


 x1.to_csv("C:\\Users\\Administrator\\Desktop\\FEOLAD_Share_30_Days_Predictions.csv",sep="\t",float_format='%10.2f')


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

plot_results_multiple(predictions, testY, 50)

plot_results_multiple(predictions, testY, 1)

x1=pd.DataFrame(testY[0].reshape(-1,1))
x2=pd.DataFrame(inversed2)

testY[0].reshape(-1,1).append(testPredict)
frames = [x1,x2]
 result = pd.concat(frames, axis=1)
 
 
 x1=pd.DataFrame(testPredict)
 x2=pd.DataFrame(testY[0].reshape(-1,1))
 frames = [x1,x2]
   result = pd.concat(frames, axis=1)
 
 
 x2.to_csv("C:\\Users\\Administrator\\Desktop\\AAP_Share_Test_Predictions30.csv",sep="\t",float_format='%10.2f')