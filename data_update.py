#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

today = date.today()

fred_api_key = 'db600fbba0846789327d005efaca0359'
fred=Fred(api_key=fred_api_key)

link = 'https://api.gurufocus.com/public/user/d4f7f3d7dd88c2d6a2160efbd82c197f:24d1077035cbd4505191093e86835bf1/stock/SPY/summary'
response = urllib.request.urlopen(link)
content = response.read()
data = json.loads(content.decode('utf8'))

start_date = datetime(1900,1,1)
end_date = datetime(2026,1,1)

dates = pd.read_excel('C:\\Users\\Jeremiah\\OneDrive\\Desktop\\J\\Investing\\Analysis v2\\Data.xlsx', sheet_name='All_Dates')
dates = pd.DataFrame(dates)

# for inference

data
PE_ratio = float(data['summary']['ratio']['P/E(ttm)']['value'])
print(PE_ratio)
ShillerPE_ratio = float(data['summary']['ratio']['Shiller P/E']['value'])
print(ShillerPE_ratio)
PB_ratio = float(data['summary']['ratio']['P/B']['value'])
print(PB_ratio)
PS_ratio = float(data['summary']['ratio']['P/S']['value'])
print(PS_ratio)
Div_Yield = float(data['summary']['ratio']['Dividend Yield']['value'])
print(Div_Yield)
Buyback_Yield = float(data['summary']['ratio']['Buyback Yield %']['value'])
print(Buyback_Yield)
EY = 1/PE_ratio
print(EY)

# import fredapi key
# import gurufocus api key
# import base excel file for all historical data

symbol_list = ['^SPX','VXX','SVXY','GLD','BTC-USD','SH','KBE','TLT','XLY','VNQ','XLE','XLV']

data = yf.download('^VIX', start = start_date, end = end_date)
summary_table = data.reset_index()[['Date','Adj Close']]
summary_table.columns = ['date','^VIX']
print(summary_table)

for symbol in symbol_list:
    data = yf.download(symbol, start=start_date, end=end_date).reset_index()
    data2=data[['Date','Adj Close']]
    data2.columns = ['date',symbol]
    summary_table = summary_table.merge(data2, on='date', how='left')
    
summary_table.columns = ['date','VIX','SPX','VXX','SVXY','GLD','BTC-USD','SH','KBE','TLT','XLY','VNQ','XLE','XLV']
summary_table = summary_table[['date','SPX','VXX','SVXY','GLD','BTC-USD','SH','KBE','TLT','XLY','VNQ','XLE','XLV','VIX']]
print(summary_table)

# summary_table.to_excel('C:\\Users\\Jeremiah\\OneDrive\\Desktop\\J\\Investing\\Analysis v2\\summary data.xlsx')

unrate = pd.DataFrame(fred.get_series('UNRATE')).reset_index()
unrate.columns = ['date','unrate']
unrate['unrate'] = round(unrate['unrate'],1)
unrate['unrate_1yrchg'] = round(unrate['unrate']/unrate['unrate'].shift(periods=12)-1,2)

gdp = pd.DataFrame(fred.get_series('gdpc1')).reset_index()
gdp.columns = ['date','gdp']
gdp.fillna(method='ffill', inplace=True)
gdp['gdp_1yrchg'] = round(gdp['gdp']/gdp['gdp'].shift(periods=4)-1,2)

cpi = pd.DataFrame(fred.get_series('CPIAUCSL')).reset_index()
cpi.columns = ['date','cpi']
cpi['cpi_1yrchg'] = round(cpi['cpi']/cpi['cpi'].shift(periods=12)-1,3)

tenyr_minus_threemth = pd.DataFrame(fred.get_series('T10Y3M')).reset_index()
tenyr_minus_threemth.columns = ['date','tenyr_minus_threemth']
tenyr_minus_threemth = dates.merge(tenyr_minus_threemth, on='date', how='left')
tenyr_minus_threemth.fillna(method='ffill', inplace=True)

tenyr_minus_twoyr = pd.DataFrame(fred.get_series('T10Y2Y')).reset_index()
tenyr_minus_twoyr.columns = ['date','tenyr_minus_twoyr']
tenyr_minus_twoyr = dates.merge(tenyr_minus_twoyr, on='date', how='left')
tenyr_minus_twoyr.fillna(method='ffill', inplace=True)

tenyr = pd.DataFrame(fred.get_series('DGS10')).reset_index()
tenyr.columns = ['date','tenyr']
tenyr = dates.merge(tenyr, on='date', how='left')
tenyr.fillna(method='ffill', inplace=True)

oil = pd.DataFrame(fred.get_series('DCOILWTICO')).reset_index()
oil.columns = ['date','oil']
oil['oil_1yrchg'] = round(oil['oil']/oil['oil'].shift(periods=260)-1,3)

real_estate = pd.DataFrame(fred.get_series('CSUSHPINSA')).reset_index()
real_estate.columns = ['date','real_estate']
real_estate['real_estate_1yrchg'] = round(real_estate['real_estate']/real_estate['real_estate'].shift(periods=12)-1,3)

M1 = pd.DataFrame(fred.get_series('M1SL')).reset_index()
M1.columns = ['date','M1']
M1['M1_1yrchg'] = round(M1['M1']/M1['M1'].shift(periods=12)-1,3)

M2 = pd.DataFrame(fred.get_series('M2SL')).reset_index()
M2.columns = ['date','M2']
M2['M2_1yrchg'] = round(M2['M2']/M2['M2'].shift(periods=12)-1,3)

macro_data = unrate.merge(gdp[['date','gdp_1yrchg']], on='date', how='left')
macro_data3 = macro_data.merge(cpi[['date','cpi_1yrchg']], on='date', how='left')
macro_data4 = macro_data3.merge(tenyr_minus_threemth, on='date', how='left')
macro_data5 = macro_data4.merge(tenyr_minus_twoyr, on='date', how='left')
macro_data6 = macro_data5.merge(tenyr, on='date', how='left')
macro_data7 = macro_data6.merge(oil[['date','oil_1yrchg']], on='date', how='left')
macro_data8 = macro_data7.merge(real_estate[['date','real_estate_1yrchg']], on='date', how='left')
macro_data9 = macro_data8.merge(M1[['date','M1_1yrchg']], on='date', how='left')
macro_data10 = macro_data9.merge(M2[['date','M2_1yrchg']], on='date', how='left')

model_data = summary_table.merge(macro_data10, on='date', how='left')
model_data['GLD_1yrchg'] = round(model_data['GLD']/model_data['GLD'].shift(periods=260)-1,3)
model_data.fillna(method='ffill', inplace=True)
model_data['EY_minus_10y_yield'] = EY*100-model_data['tenyr'].tail(1)
model_data['10y_yield_minus_cpi'] = model_data['tenyr'].tail(1)/100-model_data['cpi_1yrchg'].tail(1)
model_data['EY_minus_cpi'] = EY-model_data['cpi_1yrchg'].tail(1)
model_data['PE_ratio'] = PE_ratio
model_data['ShillerPE_ratio'] = ShillerPE_ratio
model_data['PB_ratio'] = PB_ratio
model_data['PS_ratio'] = PS_ratio
model_data['Buyback_Yield'] = Buyback_Yield/100
model_data['Div_Yield'] = Div_Yield
model_data['date'] = model_data['date'].dt.date

model_data.columns

model_data.to_excel('C:\\Users\\Jeremiah\\OneDrive\\Desktop\\J\\Investing\\Analysis v2\\summary data.xlsx', index=False)

# VIX,unrate,unrate_1yrchg,gdp_1yrchg,cpi_1yrchg,tenyr_minus_threemth
# tenyr_minus_twoyr,month,pe ratio,cape ratio,pb ratio,ps ratio,div yield
# 1day_hist,1wk_hist,1mth_hist,3mth_hist,6mth_hist,9mth_hist,12mth_hist
# 1 wk/1mth mvg avg,1mth/3mth mvg avg,3mth/6mth mvg avg,6mth/12mth mvg avg


# In[12]:


import mysql.connector
import openpyxl  # For reading Excel files

# Database credentials (replace with your actual credentials)
mydb = mysql.connector.connect(
    host="localhost",
    user="woods334",
    password="MySQL.Humility1",
    database="investment_data"
)

mycursor = mydb.cursor()

def append_excel_row_to_mysql(excel_file, sheet_name, table_name):
    """Appends the last row of an Excel sheet to a MySQL table.

    Args:
        excel_file: Path to the Excel file.
        sheet_name: Name of the sheet in the Excel file.
        table_name: Name of the MySQL table.
    """
    try:
        # 1. Open the Excel workbook and select the sheet
        workbook = openpyxl.load_workbook(excel_file)
        sheet = workbook[sheet_name]

        # 2. Get the last row of the Excel sheet
        last_row = sheet.max_row

        if last_row is None or last_row == 0 : #check if the file is empty
            print("Error: The excel file is empty")
            return


        # 3. Read data from the last row
        row_data = []
        for cell in sheet[last_row]:  # Iterate over cells in the last row
            row_data.append(cell.value)

        # 4. Construct the SQL INSERT statement dynamically
        # Important: Sanitize data to prevent SQL injection!
        placeholders = ", ".join(["%s"] * len(row_data))  # Create placeholders
        sql = f"INSERT INTO {table_name} VALUES ({placeholders})"

        # 5. Execute the query with parameterized values
        mycursor.execute(sql, tuple(row_data))  # Pass data as a tuple
        mydb.commit()

        print(f"Last row from '{sheet_name}' appended to '{table_name}'.")

    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        mydb.rollback()
    except openpyxl.Error as err:
        print(f"Excel Error: {err}")
        mydb.rollback()
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
        mydb.rollback()

    finally:
        mycursor.close()
        mydb.close()


# Example usage:
excel_file = "C:\\Users\\Jeremiah\\OneDrive\\Desktop\\J\\Investing\\Analysis v2\\summary data.xlsx"  # Replace with your Excel file path
sheet_name = "Sheet1"  # Replace with your sheet name
table_name = "historical_market_data"  # Replace with your MySQL table name

append_excel_row_to_mysql(excel_file, sheet_name, table_name)


# In[ ]:




