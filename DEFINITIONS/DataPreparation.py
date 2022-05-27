#!/usr/bin/env python
# coding: utf-8

# In[7]:


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



# Function that check if any selected ticker is cryptocurrency
def check_quote_type(tickers, start, end):

    qType = pd.DataFrame(index=tickers)
    
    # Loop by all selected tickers and get ticker quota type from Yahoo
    for ticker in tickers:
        qType[ticker] = pdr.get_quote_yahoo(ticker, start = start, end = end)['quoteType'].iloc[0]

    # Store all unique quota types in variable
    unique_types = qType.iloc[0].unique()
    for i in range(len(unique_types)):
        
        # If any of assets is crypto then set freq = 'choese freq', which means that user must decide if he/she want to work on
        # full year range = 365 or business days year range = 252
        if unique_types[i] == 'CRYPTOCURRENCY':

            freq = 'choese freq'
            break

        # If there is no crypto just set freq to busieness days only
        else:
           
            freq = 'business days'
            
    if freq == 'choese freq':
        print('It looks like some of your assets are cryptocurrencies')
        print('Crypto markets are listed every single day in addition to traditional stocks')

    else:    
        print('It look like that all of your assets come from the traditional market, which is quoted every business day')
                        
    return freq, qType

# I created this function, becouse I noticed that Yahoo Finance has some missing dates for traditional Markets likie equity or commodity
# I usually hapens, when selecting end date that is very close to current date
# To avoid that issue I could just use my function that insert missing rows, but it would consume to much time, so
# I decided that I can just set crypto asset (if any) at first place, which usually work well and
# Thanks that I could just handle missing dates using fillna(method='bfill'), which copy previous value instead of Naan

# This function will detect if any asset is crypto and set it to first place
def sort_tickers(qType, tickers):
    
    # Create df with index as ticker name and values as ticker type
    tic_type = pd.DataFrame(index=qType.columns, data=qType.values[0])
    
    # Loop by tickers type to find if any tickers are crypto
    for i in range(len(tic_type)):
        
        if tic_type.iloc[i].values == 'CRYPTOCURRENCY':
            
            # If that is True, then add crypto ticker again at FIRST place
            tickers.insert(0,tic_type.index[i])
            # Then remove duplicates
            tickers = list(dict.fromkeys(tickers))
            break
        else:
            pass
    
    # Return sorted tickers    
    return tickers, tic_type

# Function that display checkbox if freq = 'choese freq'
def select_freq(full_DR, busieness_DR, freq):
    
    if freq == 'choese freq':
        
        print('Now you will decide if you want to work on full date range or only business days:')
        display(full_DR, busieness_DR)

    # if freq =! 'choese freq' just pass variables
    else:
        pass
        
    return full_DR, busieness_DR


# Function that read checkbox results, gives comment and return final choice
def choese_freq(full_DR, busieness_DR, freq):
    
    if freq == 'choese freq':

        # All aveliable tickers
        freqs = [full_DR, busieness_DR]

        # Tickers that user selectes
        choesen_freq = []

        # Add tickers to choesen ticker by loop
        for freq in freqs:

            if freq.get_interact_value() == True:
                freq_name = freq.description
                choesen_freq.append(freq_name)

            else:
                pass

        # Condition: You must select at least 1 freq
        if len(choesen_freq) <1:
            print('You must select at least 1 frequency to continue')
            
        elif len(choesen_freq) > 1:
            print('You can NOT select more than 1 frequency at a time')

        # If condiction is fulfilled then print choesen freq
        else:
            print(f'Your choice: {choesen_freq}')
            
    else:
        choesen_freq = ['252']
        

    return choesen_freq


# This class is set of function for users who selected 365 days freq
class PasteMissingRows:
    
    def __init__(self, ticker, start_date, end_date):
        
        self.df = None
        self.dates_check = None
        self.df_result = None
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

          
    def Download_df(self):
        
        # Download data from yahoo
        stock_data = pdr.get_data_yahoo([self.ticker], start = self.start_date, end = self.end_date)['Adj Close']
        
        df = pd.DataFrame(stock_data)
        # Make copy and reset index
        df.reset_index(inplace=True)
        
        return df
            
        
    def Create_df_with_all_dates(self):
        
        df = self.Download_df()

        # Range of dates
        dates_check = pd.DataFrame({"dates":pd.date_range(start = self.start_date, end = self.end_date)})

        return dates_check   
    
    # Function to insert row in the dataframe
    def Insert_value(self,row_number, df, row_value):

        # Slice above rows
        df1 = df[0:row_number]

        # Store below rows
        df2 = df[row_number:]

        # Insert the row in the upper half dataframe
        df1.loc[row_number] = row_value

        # Concat the two dataframes
        df_result = pd.concat([df1, df2])

        # Reassign the index labels
        df_result.index = [*range(df_result.shape[0])]
        
        return df_result

    # Function that detect mising rows in df and add previous values where missing
    def Insert_row_to_df(self):
        
        dates_check = self.Create_df_with_all_dates()
        df = self.Download_df()
        
        if len(df) < len(dates_check):
        
            for i in range(1, len(dates_check)):

                # if diff between next days is > 1
                if (df['Date'].loc[i] - df['Date'].loc[i-1]).days > 1:

                    # first [0]+1 = 1
                    row_number = i
                    row_value = [dates_check['dates'].iloc[i-1], df[self.ticker].iloc[i-1]]

                    # Let's call the function and insert the row
                    df = self.Insert_value(row_number, df, row_value)
                    df = df.sort_values(by="Date")

                else:
                    df = df.sort_values(by="Date")
                    
            df.drop_duplicates(subset=None, keep='first', inplace=True)
                    
        else:
            pass
            df = df[:-1]
  
    
        return df
    
# Final function that create dataset with 365 freq
def createPrepDataFrame(tickers, start, end, PasteMissingRows):    
    
    data = pd.DataFrame()

    # loop by tickers
    for ticker in tickers:

        # Use function to insert missing rows for each ticker
        dt = PasteMissingRows(ticker=ticker, start_date=start, end_date=end).Insert_row_to_df()

        # Set index for proper concating
        dt.set_index('Date', inplace=True)

        # Add column named as ticker with values from function
        data[ticker] = dt
    
    return data


# This 2 functions below is for users who picked 252 days freq or selected only traditional stocks (no crypto in wallet)

# Cutting funcion in perpouse to create proper dataset
def cut_weekends(start,end, data):
    
    # Create date range with proper busieness dates only
    prop_dRange = pd.bdate_range(start, end)
    
    # For some reason yahoo finance download one extra day afrer end date for crypt
    # So I must cut it
    data = data[:-1]
    
    # loop by any row in data
    for i in range(len(data)):
        
        # If weekday of specific date is equal 5 it means that this is saturday
        if data.index[i].weekday() == 5:
            
            # Drop saturday
            data.drop(data.index[i], inplace=True)
            
            # Now Sunday is in posiotion of i, becouse we already cut Saturday, so I just need to make one more drop here
            data.drop(data.index[i], inplace=True)
            
        else:
            # If len of busines day index created by hand is equal new cuted data, then brak loop
            if len(prop_dRange) == len(data):
                break
        
    return data

# Final function that create df for 252 freq
def create_data_without_weekends(tickers, start, end, cut_weekends):
    
    data = pd.DataFrame()

    # loop by tickers
    for ticker in tickers:

        # First we must check if asset is crypt
        qType = pdr.get_quote_yahoo(ticker, start = start, end = end)['quoteType'].iloc[0]

        if qType == 'CRYPTOCURRENCY':

            # download data
            dt = pdr.get_data_yahoo([ticker], start = start, end = end)['Adj Close']
            # Cut weekends
            dt = cut_weekends(start, end, dt)


        else:
            # If no, just download data
            dt = pdr.get_data_yahoo([ticker], start = start, end = end)['Adj Close']


        # Add column named as ticker with values from function
        data[ticker] = dt
        
    # In cease, there is some missing dates in Yahoo finance
    data.fillna(method='bfill', inplace=True)
    
    return data

# Compare all function to create datasets with choesen freq (252 or 365)
def createDF_with_choesen_ferq(choesen_freq, tickers, start, end, createPrepDataFrame, PasteMissingRows, create_data_without_weekends, cut_weekends):
   
    if choesen_freq[0] == '365':

        # Create 365days per year index and prepare data by functio
        data = createPrepDataFrame(tickers, start, end, PasteMissingRows)
        n = 365

    elif choesen_freq[0] == '252':

        # Create 252 days per year index and prepare data
        # Even if user did not choese crypto for his portfolio it will fill missinig values
        data = create_data_without_weekends(tickers, start, end, cut_weekends)
        n = 252


    else:
        print('You didn\'t selected frequency!')
        
    # It returns dataset and n, which is period that will be helpful in calculating annual vol and rets for Markowitz model
    return data, n

