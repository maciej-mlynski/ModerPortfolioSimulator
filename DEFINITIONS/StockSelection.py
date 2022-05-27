#!/usr/bin/env python
# coding: utf-8

# In[7]:

import sys
import numpy as np
import pandas as pd
import seaborn as sb
sb.set()
import scipy.optimize as optimization
import datetime as dt
from matplotlib import rcParams
rcParams['figure.figsize'] = 10,6
import math
import scipy.stats
import matplotlib.pyplot as plt
import time
import pandas_datareader as pdr
from ipywidgets import Checkbox
from dateutil.relativedelta import relativedelta

sys.path.insert(0,'/ModernPortfolio_project/DEFINITIONS/')
# # Select Stocks and Periods functions

# In[2]:


# def function that take selected tickers
def choese_stock(SPX_, FXI_, GMF_, EXS1_, VGK_,DBC_, FILL_, BNO_, PICK_,SGOL_, PPLT_, SIVR_, PALL_,BTC_,IEF_):

    # All aveliable tickers
    tickers = [SPX_, FXI_, GMF_, EXS1_, VGK_,DBC_, FILL_, BNO_, PICK_,SGOL_, PPLT_, SIVR_, PALL_,BTC_,IEF_]
    
    # Tickers that user selectes
    choesen_ticker = []

    # Add tickers to choesen ticker by loop
    for ticker in tickers:

        if ticker.get_interact_value() == True:
            ticker_name = ticker.description
            choesen_ticker.append(ticker_name)

        else:
            pass

    # Condition: You must select at least 2 stocks
    if len(choesen_ticker) <2:
        print('You must select at least 2 stocks in analyse perpouse')
    
    # If condiction is fulfilled then print choesen tickers
    else:
        print(f'Your choice: {choesen_ticker}')
        
    return choesen_ticker

def add_stock_by_hand(tickers, new_tickers):
    
    for ticker in new_tickers:

        # Check if ticker has not been taken already in checkbox
        if ticker in tickers:
                print(f'{ticker} has been already choesen priviously')
        else:

            # If no check if ticker exist in Yahoo Finance
            try:
                check_ticker = pdr.get_data_yahoo(ticker).first_valid_index()
                tickers.append(ticker)
                # If exist add to tickers array
                print(f'{ticker} exist in Yahoo Finance and has been added to your analysis')

            # If does not exist print comment
            except:
                print(f'{ticker} does not exist in Yahoo Finance')
                
    return tickers


# function that check first day that stock is aveliable on yahoo
# We must have all stock in same period, so the most recent day will be begining of analysis
# The conclusion is that start date does not have to be true period of analysis becouse it also depend on aveliability of data

def find_first_date_from_choesen_stosks(start, end, tickers):
    
    # download data
    df = pd.DataFrame(index=tickers)
    for ticker in tickers:
        # closing prices
        df[ticker] = pdr.get_data_yahoo(ticker, start=start, end=end).first_valid_index()
        df.append(df[ticker])

    # Check oldest start date for all tickers (max means oldest in this case)
    first_date = df.iloc[0].max()
    first_date_ticker = tickers[df.iloc[0].argmax()]
    
    print(f'First aveliable date to comapre stocks is {first_date}. This stock is {first_date_ticker}.')
    print('---------------------------------------------------')
    print('If this is to short period, You can always unmark stock with the lowest start date aveliable')
    print('Remember that if you want to delete ticker from analysis u must shift+enter all cells again')
    
    return first_date


# Check if user choese start working day date. If anserw is YES, then move forward to closest work day
def check_start_date(new_start):

    # Check if start and end date is weekend or weekday
    if new_start.weekday() == 5:

        new_start = new_start + dt.timedelta(days=2)

    elif new_start.weekday() == 6:

        new_start = new_start + dt.timedelta(days=1)

    else:
        pass

    return new_start


# Check if user choese end working day date. If anserw is YES, then move back to closest work day       
def check_end_date(end):
    
    end = pd.Timestamp(end)
    
    # Check if start and end date is weekend or weekday
    if end.weekday() == 5:

        end = end + dt.timedelta(days=-1)

    elif end.weekday() == 6:

        end = end + dt.timedelta(days=-2)

    else:
        pass

    return end

