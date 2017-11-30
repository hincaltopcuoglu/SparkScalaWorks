# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:34:55 2017

@author: Administrator
"""
import pandas as pd
import numpy as np

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array


np.random.seed(7)
 
# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
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
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	diff_series = difference(raw_values, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test




# load dataset

df2=read_csv('C:\\Users\\Administrator\\Desktop\\output6.txt',sep=",",usecols=[0,1,5],dtype={"Date":object})

gb = df2.groupby('Firma')
#make all the groups data frame in a dict
d = {}
for k in gb.groups:
 d[k] = DataFrame(gb.get_group(k))
 
 
 
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
    



series=d['ZOB_Share']


# configure
n_lag = 120
n_seq = 15
n_test = 15
n_epochs = 100
n_batch = 1
n_neurons = 15

scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)

X, y = train[:, 0:n_lag], train[:, n_lag:]
X = X.reshape(X.shape[0], X.shape[1],1)

y=y.reshape(y.shape[0],y.shape[1],1)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]),return_sequences=True,stateful=True))
model.add(Dense(y.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')

# fit network
for i in range(150):
    model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
	 model.reset_states()
        
        
model = Sequential()
model.add(LSTM(units = 15, return_sequences = True, input_shape = (120,1)))
#model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=100, batch_size=1, verbose=2)
        
