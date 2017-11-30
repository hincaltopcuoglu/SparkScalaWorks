# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:45:58 2017

@author: Administrator
"""

from pandas import concat
from pandas import read_csv
from pandas import datetime
 
# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pandas.DataFrame(data)
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
	agg = pandas.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# transform series into train and test sets for supervised learning
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
 
# load dataset
import pandas
df=pandas.read_csv('C:\\Users\\Administrator\\Desktop\\database_amir4.csv',sep=";",usecols=[0,1,2],dtype={"Date":object})

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
    
    
dataset=d['AAP_Share']

# configure
n_lag = 1
n_seq = 3
n_test = 10
# prepare data
train, test = prepare_data(dataset, n_test, n_lag, n_seq)
print(test)
print('Train: %s, Test: %s' % (train.shape, test.shape))


def persistence(last_ob, n_seq):
	return [last_ob for i in range(n_seq)]



def make_forecasts(train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = persistence(X[-1], n_seq)
		# store the forecast
		forecasts.append(forecast)
	return forecasts

forecasts = make_forecasts(train, test, 1,5)

def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = test[:,(n_lag+i)]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = math.sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

import math
from sklearn.metrics import mean_squared_error

evaluate_forecasts(test, forecasts, 1, 5)



def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	matplotlib.pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i
		off_e = off_s + len(forecasts[i])
		xaxis = [x for x in range(off_s, off_e)]
		matplotlib.pyplot.plot(xaxis, forecasts[i], color='red')
	# show the plot
	matplotlib.pyplot.show()
    
import matplotlib.pyplot
    
plot_forecasts(dataset, forecasts, 3)

def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	matplotlib.pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - 12 + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		matplotlib.pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	matplotlib.pyplot.show()