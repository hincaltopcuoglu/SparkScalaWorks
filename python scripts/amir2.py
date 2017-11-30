# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:33:00 2017

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



import csv
import glob
import os



read_files = glob.glob("C:\\Users\\Administrator\\Desktop\\database\\d\\*.txt")

with open("C:\\Users\\Administrator\\Desktop\\result.csv", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
            
header_saved = False
with open('C:\\Users\\Administrator\\Desktop\\output6.txt','w') as fout:
    for filename in read_files:
        with open(filename) as fin:
            header = next(fin)
            if not header_saved:
                fout.write(header)
                header_saved = True
            for line in fin:
                fout.write(line)





##################################################################################################
df=pd.read_csv('C:\\Users\\Administrator\\Desktop\\database_amir4.csv',sep=";",usecols=[0,1,2],dtype={"Date":object})

df=

gb = df.groupby('Firm')

gb2 = df_d.groupby('Firm')

#make all the groups data frame in a dict
d = {}
for k in gb.groups:
 d[k] = pd.DataFrame(gb.get_group(k))
 
 #make all the groups data frame in a dict
d2 = {}
for k in gb2.groups:
 d2[k] = pd.DataFrame(gb2.get_group(k))
 
for key in d:  
    if len(d[key])==1:
        del d[key]
        

#sort it according to Date
for key in d:
    d[key]=d[key].groupby(['Firm']).apply(lambda x: x.sort_values(["Date"], ascending = True))

for key in d2:
    d2[key]=d2[key].groupby(['Firm']).apply(lambda x: x.sort_values(["Date"], ascending = True))

#drop column     
for key in d:
    d[key]=d[key].drop(d[key].columns[[0]], axis=1)
    
    
    
    
    
# convert time series into supervised learning problem
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
    
supervised={}
for key in d:	
 supervised[key] = series_to_supervised(d[key], 1, 30)
 
 
 
 
 
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	raw_values = raw_values.reshape(len(raw_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(raw_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return train, test


x=d['AAP_Share']['Close'].values
x.reshape(len(x), 1)

# configure
n_lag = 1
n_seq = 30
n_test = 10

# prepare data
train={}
test={}
for key in d:
 train[key], test[key] = prepare_data(d[key]['Close'], n_test, n_lag, n_seq)
 
 
for key in test:
 print(test[key])
 
for key in test:
 print('Train: %s, Test: %s' % (train[key].shape, test[key].shape))
 
 
 
 
 
 # make a persistence forecast
def persistence(last_ob, n_seq):
	return [last_ob for i in range(n_seq)]
 
 
 


# evaluate the persistence model
def make_forecasts(train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = persistence(X[-1], n_seq)
		# store the forecast
		forecasts.append(forecast)
	return forecasts
 
 
 
forecasts={}
for key in train:
 forecasts[key] = make_forecasts(train[key], test[key], 1, 30)
 
 
 


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = test[:,(n_lag+i)]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
 
    
 
for key in forecasts:  
    if len(forecasts[key])==0:
        del forecasts[key]
        
 
for key in forecasts:
 evaluate_forecasts(test[key], forecasts[key], 1, 30)
 
 
 
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i
		off_e = off_s + len(forecasts[i])
		xaxis = [x for x in range(off_s, off_e)]
		pyplot.plot(xaxis, forecasts[i], color='red')
	# show the plot
	pyplot.show()
    
    
    
plot_forecasts(test['AAP_Share'], forecasts['AAP_Share'], 30)
 
 d['AAP_Share']
 
 
 
 
 