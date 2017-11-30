# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:07:43 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:49:31 2017

@author: Administrator
"""

# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
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




# Importing the training set
df2=pd.read_csv('C:\\Users\\Administrator\\Desktop\\output6.txt',sep=",",usecols=[0,1,5],dtype={"Date":object})

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

dataset=d['SHAPNA_Share']

training_set = dataset.iloc[:,0:1].values
training_set = training_set.astype('float32')

#split it train and test
#train_size = int(len(training_set) * 0.95)
#test_size = len(training_set) - train_size
#train, test = training_set[0:train_size], training_set[train_size:len(training_set)]
#print(len(train), len(test ))

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 timesteps and t+1 output
X_train = []
y_train = []
for i in range(15, 1012):
    X_train.append(training_set_scaled[i-15:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 64, return_sequences = True, input_shape = (None, 1)))

# Adding a second LSTM layer
regressor.add(LSTM(units = 64, return_sequences = True))

# Adding a third LSTM layer
regressor.add(LSTM(units = 64, return_sequences = True))

# Adding a fourth LSTM layer
regressor.add(LSTM(units = 64, return_sequences = True))

# Adding a fifth LSTM layer
regressor.add(LSTM(units = 64))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 70, batch_size = 2)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price for February 1st 2012 - January 31st 2017
#dataset_test = pd.read_csv('C:\\Users\\Administrator\\Desktop\\actual.csv')
# Part 3 - Making the predictions and visualising the results

# Getting the real stock price for February 1st 2012 - January 31st 2017
dataset_test = pd.read_csv('C:\\Users\\Administrator\\Desktop\\feolad_test2.txt')
test_set = dataset_test.iloc[:,0:1].values
real_stock_price = np.concatenate((training_set[0:1012], test_set), axis = 0)

# Getting the predicted stock price of 2017
scaled_real_stock_price = sc.fit_transform(real_stock_price)
inputs = []
for i in range(1012, 1026):
    inputs.append(scaled_real_stock_price[i-15:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price[1012:], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
plt.show()


regressor.save('C:\\Users\\Administrator\\Desktop\\shapna_model2_good.h5')

####

x1=pd.DataFrame(real_stock_price[1449:])
x2=pd.DataFrame(predicted_stock_price)
frames = [x1,x2]
result = pd.concat(frames, axis=1)

x2.to_csv("C:\\Users\\Administrator\\Desktop\\SHAPNA_Share_Predictions_16112017_2.csv",sep="\t",float_format='%10.2f')


#######################

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        prediction_seqs = []  
        curr_frame = testX[1]
        predicted = []
        #for i in range(int(len(data)/prediction_len)):
        for j in range(20):
            predicted.append(regressor.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [15-1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs
      
test_set=test

test_set_scaled = sc.fit_transform(test_set)

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

testX, testY = create_dataset(test_set_scaled, 15)

b = np.reshape(b, (b.shape[0], b.shape[1], 1))


predictions = predict_sequences_multiple(regressor, b, 20,20)


inversed2=sc.inverse_transform(np.array(predictions).reshape(-1, 1))

b=[]

a=test_set_scaled[487:507,0]

b.append(a)

b.append(test_set_scaled[488:508,0])

b=np.array(b)














#######################

x3=pd.DataFrame(inversed2)


x3.to_csv("C:\\Users\\Administrator\\Desktop\\FEOLAD_Share_30_Days_Predictions_v3.csv",sep="\t",float_format='%10.2f')




#############################################################












