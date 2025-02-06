#!/usr/bin/env python
# coding: utf-8

# In[13]:


# import packages (numpy, pandas, fredapi, gurufocus, yfinance, isotonic regression, parabolic curve fitting, RF, validation)

import requests
import json
from fredapi import Fred
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from numpy import arange
from pandas import read_csv
from datetime import date
from statistics import mean
from scipy.optimize import curve_fit
from matplotlib import pyplot
import matplotlib.pyplot as plt
import urllib.request, json
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

today = date.today()

data = pd.read_excel('C:\\Users\\Jeremiah\\OneDrive\\Desktop\\J\\Investing\\Analysis v2\\Data.xlsx', sheet_name='Data2')


# In[14]:


symbol = 'SPX'

model_data = data[[
'date', symbol, 
'VIX','unrate', 'unrate_1yrchg', 'gdp_1yrchg', 'cpi_1yrchg',
'tenyr_minus_threemth', 'tenyr_minus_twoyr', 'tenyr', 'oil_1yrchg',
'real_estate_1yrchg', 'M1_1yrchg', 'M2_1yrchg', 'GLD_1yrchg',
'EY_minus_10y_yield', '10y_yield_minus_cpi', 'EY_minus_cpi', 'PE_ratio',
'ShillerPE_ratio', 'PB_ratio', 'PS_ratio', 'Buyback_Yield', 'Div_Yield'
]]

model_data['1day_hist'] = round(model_data[symbol]/model_data[symbol].shift(periods=1)-1,3)
model_data['1wk_hist'] = round(model_data[symbol]/model_data[symbol].shift(periods=5)-1,3)
model_data['1mth_hist'] = model_data[symbol]/model_data[symbol].shift(periods=22)-1
model_data['3mth_hist'] = round(model_data[symbol]/model_data[symbol].shift(periods=65)-1,3)
model_data['6mth_hist'] = round(model_data[symbol]/model_data[symbol].shift(periods=130)-1,2)
model_data['9mth_hist'] = round(model_data[symbol]/model_data[symbol].shift(periods=196)-1,2)
model_data['12mth_hist'] = round(model_data[symbol]/model_data[symbol].shift(periods=261)-1,2)
model_data['1day_fwd'] = model_data[symbol].shift(periods=-1)/model_data[symbol]-1
model_data['1wk_fwd'] = model_data[symbol].shift(periods=-5)/model_data[symbol]-1
model_data['1mth_fwd'] = model_data[symbol].shift(periods=-22)/model_data[symbol]-1
model_data['3mth_fwd'] = model_data[symbol].shift(periods=-65)/model_data[symbol]-1
model_data['6mth_fwd'] = model_data[symbol].shift(periods=-130)/model_data[symbol]-1
model_data['9mth_fwd'] = model_data[symbol].shift(periods=-196)/model_data[symbol]-1
model_data['12mth_fwd'] = model_data[symbol].shift(periods=-261)/model_data[symbol]-1
model_data['24mth_fwd'] = model_data[symbol].shift(periods=-520)/model_data[symbol]-1

model_data['24mth_mvg_avg'] = model_data[symbol].rolling(520).mean()
model_data['12mth_mvg_avg'] = model_data[symbol].rolling(261).mean()
model_data['9mth_mvg_avg'] = model_data[symbol].rolling(196).mean()
model_data['6mth_mvg_avg'] = model_data[symbol].rolling(130).mean()
model_data['3mth_mvg_avg'] = model_data[symbol].rolling(65).mean()
model_data['1mth_mvg_avg'] = model_data[symbol].rolling(22).mean()
model_data['1wk_mvg_avg'] = model_data[symbol].rolling(5).mean()
model_data['1day_mvg_avg'] = model_data[symbol].rolling(1).mean()
model_data['12mth/24mth_mvg_avg'] = model_data['12mth_mvg_avg']/model_data['24mth_mvg_avg']
model_data['6mth/12mth_mvg_avg'] = model_data['6mth_mvg_avg']/model_data['12mth_mvg_avg']
model_data['3mth/6mth_mvg_avg'] = model_data['3mth_mvg_avg']/model_data['6mth_mvg_avg']
model_data['1mth/3mth_mvg_avg'] = model_data['1mth_mvg_avg']/model_data['3mth_mvg_avg']
model_data['1wk/1mth_mvg_avg'] = model_data['1wk_mvg_avg']/model_data['1mth_mvg_avg']
model_data['1day/1wk_mvg_avg'] = model_data['1day_mvg_avg']/model_data['1wk_mvg_avg']
#model_data = model_data.dropna() #subset=[symbol])
model_data


# In[15]:


timeframe = '12mth_fwd'

model_data2 = model_data[[
'date', 'VIX', 'unrate', 'unrate_1yrchg', 'gdp_1yrchg',
'cpi_1yrchg', 'tenyr_minus_threemth', 'tenyr_minus_twoyr', 'tenyr',
'oil_1yrchg', 'real_estate_1yrchg', 'M1_1yrchg', 'M2_1yrchg',
'GLD_1yrchg', 'EY_minus_10y_yield', '10y_yield_minus_cpi',
'EY_minus_cpi', 'PE_ratio', 'ShillerPE_ratio', 'PS_ratio', 'Div_Yield', 
'1day_hist', '1wk_hist', '1mth_hist', '3mth_hist', '6mth_hist', '9mth_hist', '12mth_hist', 
'6mth/12mth_mvg_avg', '3mth/6mth_mvg_avg', '1mth/3mth_mvg_avg', '1wk/1mth_mvg_avg', '1day/1wk_mvg_avg',
timeframe
]].iloc[300:]
#261
model_data2_predictors = model_data2.drop(timeframe, axis=1)

predictor_list = model_data2_predictors.columns

for predictor in predictor_list:
    model_data2[predictor + '_perc'] = model_data2[predictor].rank(pct=True)
    
model_data2.set_index('date', inplace=True)
model_data3 = model_data2.dropna().reset_index()
rows, columns = model_data3.shape
print(rows)

# increasing isotonic regression

predictor_list = [
'VIX_perc', 'unrate_perc', 'unrate_1yrchg_perc', 'gdp_1yrchg_perc',
'cpi_1yrchg_perc', 'tenyr_minus_threemth_perc', 'tenyr_minus_twoyr_perc', 'tenyr_perc',
'oil_1yrchg_perc', 'real_estate_1yrchg_perc', 'M1_1yrchg_perc', 'M2_1yrchg_perc',
'GLD_1yrchg_perc', 'EY_minus_10y_yield_perc', '10y_yield_minus_cpi_perc',
'EY_minus_cpi_perc', 'PE_ratio_perc', 'ShillerPE_ratio_perc', 'PS_ratio_perc', 'Div_Yield_perc', 
'1day_hist_perc', '1wk_hist_perc', '1mth_hist_perc', '3mth_hist_perc', '6mth_hist_perc', '9mth_hist_perc', '12mth_hist_perc', 
'6mth/12mth_mvg_avg_perc', '3mth/6mth_mvg_avg_perc', '1mth/3mth_mvg_avg_perc', '1wk/1mth_mvg_avg_perc', '1day/1wk_mvg_avg_perc',
]

for predictor in predictor_list:
    x = model_data3[predictor]
    y = model_data3[timeframe]
    
    # *** train the model on the dataset with target variable
    
    model = IsotonicRegression(increasing=True)
    model.fit(x, y)
    
    # *** apply predictions to full dataset
    
    model_data2[predictor + '_isotonic_inc'] = model.predict(model_data2[predictor])
    
# decreasing isotonic regression

predictor_list = [
'VIX_perc', 'unrate_perc', 'unrate_1yrchg_perc', 'gdp_1yrchg_perc',
'cpi_1yrchg_perc', 'tenyr_minus_threemth_perc', 'tenyr_minus_twoyr_perc', 'tenyr_perc',
'oil_1yrchg_perc', 'real_estate_1yrchg_perc', 'M1_1yrchg_perc', 'M2_1yrchg_perc',
'GLD_1yrchg_perc', 'EY_minus_10y_yield_perc', '10y_yield_minus_cpi_perc',
'EY_minus_cpi_perc', 'PE_ratio_perc', 'ShillerPE_ratio_perc', 'PS_ratio_perc', 'Div_Yield_perc', 
'1day_hist_perc', '1wk_hist_perc', '1mth_hist_perc', '3mth_hist_perc', '6mth_hist_perc', '9mth_hist_perc', '12mth_hist_perc', 
'6mth/12mth_mvg_avg_perc', '3mth/6mth_mvg_avg_perc', '1mth/3mth_mvg_avg_perc', '1wk/1mth_mvg_avg_perc', '1day/1wk_mvg_avg_perc',
]

for predictor in predictor_list:
    x = model_data3[predictor]
    y = model_data3[timeframe]
    
    # *** train the model on the dataset with target variable
    
    model = IsotonicRegression(increasing=False)
    model.fit(x, y)
        
    # *** apply predictions to full dataset

    model_data2[predictor + '_isotonic_dec'] = model.predict(model_data2[predictor])
    
print('isotonic regression complete')

# define predictors

model_data4 = model_data2

X = model_data2[[
'EY_minus_cpi_perc_isotonic_inc','M2_1yrchg_perc_isotonic_dec','VIX_perc_isotonic_inc',
'tenyr_minus_threemth_perc_isotonic_dec','EY_minus_10y_yield_perc_isotonic_inc','3mth_hist_perc_isotonic_dec',
'unrate_perc_isotonic_inc','M1_1yrchg_perc_isotonic_inc'
]]
print(X)

y = model_data2[[timeframe]]

rows, columns = model_data3.shape

X_train = X.iloc[0:rows-1]
y_train = y.iloc[0:rows-1]

# Train the model
model = RandomForestRegressor(n_estimators=100) #, oob_score=True)
model.fit(X_train, y_train)

# Make a prediction on the current day
prediction = round(pd.DataFrame(model.predict(X)),3)
prediction

result = pd.concat([model_data2[timeframe].reset_index(), prediction], axis=1)
result.columns = ['date',timeframe,'prediction']
result['timeframe'] = timeframe
result = result[['date','timeframe','prediction']].tail(1)
result['pred_annualized'] = round(result['prediction'],3)
result_12mth_fwd = result
print(result_12mth_fwd)

print('prediction completed')


# In[16]:


X2 = pd.concat([X,model_data2['12mth_fwd']], axis=1)
#X2.to_excel('C:\\Users\\Jeremiah\\OneDrive\\Desktop\\J\\Investing\\Analysis v2\\SPX_prediction_12mth_predictor_percentiles.xlsx')

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame to visualize feature importances
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)
feature_importances['importance'] = round(feature_importances['importance'],3)

print(feature_importances)


# In[17]:


timeframe = '6mth_fwd'

model_data2 = model_data[[
'date', 'VIX', 'unrate', 'unrate_1yrchg', 'gdp_1yrchg',
'cpi_1yrchg', 'tenyr_minus_threemth', 'tenyr_minus_twoyr', 'tenyr',
'oil_1yrchg', 'real_estate_1yrchg', 'M1_1yrchg', 'M2_1yrchg',
'GLD_1yrchg', 'EY_minus_10y_yield', '10y_yield_minus_cpi',
'EY_minus_cpi', 'PE_ratio', 'ShillerPE_ratio', 'PS_ratio', 'Div_Yield', 
'1day_hist', '1wk_hist', '1mth_hist', '3mth_hist', '6mth_hist', '9mth_hist', '12mth_hist', 
'6mth/12mth_mvg_avg', '3mth/6mth_mvg_avg', '1mth/3mth_mvg_avg', '1wk/1mth_mvg_avg', '1day/1wk_mvg_avg',
timeframe
]].iloc[261:]

model_data2_predictors = model_data2.drop(timeframe, axis=1)

predictor_list = model_data2_predictors.columns

for predictor in predictor_list:
    model_data2[predictor + '_perc'] = model_data2[predictor].rank(pct=True)
    
model_data2.set_index('date', inplace=True)
model_data3 = model_data2.dropna().reset_index()
rows, columns = model_data3.shape
print(rows)

# increasing isotonic regression

predictor_list = [
'VIX_perc', 'unrate_perc', 'unrate_1yrchg_perc', 'gdp_1yrchg_perc',
'cpi_1yrchg_perc', 'tenyr_minus_threemth_perc', 'tenyr_minus_twoyr_perc', 'tenyr_perc',
'oil_1yrchg_perc', 'real_estate_1yrchg_perc', 'M1_1yrchg_perc', 'M2_1yrchg_perc',
'GLD_1yrchg_perc', 'EY_minus_10y_yield_perc', '10y_yield_minus_cpi_perc',
'EY_minus_cpi_perc', 'PE_ratio_perc', 'ShillerPE_ratio_perc', 'PS_ratio_perc', 'Div_Yield_perc', 
'1day_hist_perc', '1wk_hist_perc', '1mth_hist_perc', '3mth_hist_perc', '6mth_hist_perc', '9mth_hist_perc', '12mth_hist_perc', 
'6mth/12mth_mvg_avg_perc', '3mth/6mth_mvg_avg_perc', '1mth/3mth_mvg_avg_perc', '1wk/1mth_mvg_avg_perc', '1day/1wk_mvg_avg_perc',
]

for predictor in predictor_list:
    x = model_data3[predictor]
    y = model_data3[timeframe]
    
    # *** train the model on the dataset with target variable
    
    model = IsotonicRegression(increasing=True)
    model.fit(x, y)
    
    # *** apply predictions to full dataset
    
    model_data2[predictor + '_isotonic_inc'] = model.predict(model_data2[predictor])
    
# decreasing isotonic regression

predictor_list = [
'VIX_perc', 'unrate_perc', 'unrate_1yrchg_perc', 'gdp_1yrchg_perc',
'cpi_1yrchg_perc', 'tenyr_minus_threemth_perc', 'tenyr_minus_twoyr_perc', 'tenyr_perc',
'oil_1yrchg_perc', 'real_estate_1yrchg_perc', 'M1_1yrchg_perc', 'M2_1yrchg_perc',
'GLD_1yrchg_perc', 'EY_minus_10y_yield_perc', '10y_yield_minus_cpi_perc',
'EY_minus_cpi_perc', 'PE_ratio_perc', 'ShillerPE_ratio_perc', 'PS_ratio_perc', 'Div_Yield_perc', 
'1day_hist_perc', '1wk_hist_perc', '1mth_hist_perc', '3mth_hist_perc', '6mth_hist_perc', '9mth_hist_perc', '12mth_hist_perc', 
'6mth/12mth_mvg_avg_perc', '3mth/6mth_mvg_avg_perc', '1mth/3mth_mvg_avg_perc', '1wk/1mth_mvg_avg_perc', '1day/1wk_mvg_avg_perc',
]

for predictor in predictor_list:
    x = model_data3[predictor]
    y = model_data3[timeframe]
    
    # *** train the model on the dataset with target variable
    
    model = IsotonicRegression(increasing=False)
    model.fit(x, y)
        
    # *** apply predictions to full dataset

    model_data2[predictor + '_isotonic_dec'] = model.predict(model_data2[predictor])
    
print('isotonic regression complete')

# define predictors

model_data4 = model_data2

X = model_data2[[
'EY_minus_cpi_perc_isotonic_inc','real_estate_1yrchg_perc_isotonic_inc','3mth/6mth_mvg_avg_perc_isotonic_inc',
'Div_Yield_perc_isotonic_inc','M1_1yrchg_perc_isotonic_dec','M2_1yrchg_perc_isotonic_dec',
'unrate_1yrchg_perc_isotonic_inc','oil_1yrchg_perc_isotonic_dec','10y_yield_minus_cpi_perc_isotonic_dec'
]]

y = model_data2[[timeframe]]

rows, columns = model_data3.shape

X_train = X.iloc[0:rows-1]
y_train = y.iloc[0:rows-1]

# Train the model
model = RandomForestRegressor(n_estimators=100) #, oob_score=True)
model.fit(X_train, y_train)

# Make a prediction on the current day
prediction = round(pd.DataFrame(model.predict(X)),3)
prediction

result = pd.concat([model_data2[timeframe].reset_index(), prediction], axis=1)
result.columns = ['date',timeframe,'prediction']
result['timeframe'] = timeframe
result = result[['date','timeframe','prediction']].tail(1)
result['pred_annualized'] = round(result['prediction'],3)*2
result_6mth_fwd = result
print(result_6mth_fwd)

print('prediction completed')


# In[18]:


result = pd.concat([result_12mth_fwd.tail(1),result_6mth_fwd.tail(1)])
result['asset'] = symbol
result = result[['date','asset','timeframe','prediction','pred_annualized']]
result


# In[19]:


result.to_excel('C:\\Users\\Jeremiah\\OneDrive\\Desktop\\J\\Investing\\Analysis v2\\SPX_prediction.xlsx')


# In[ ]:




