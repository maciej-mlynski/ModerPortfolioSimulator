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



# # Rebalancing and summary of all portfolio functions

    
def prepare_data(investmentAmt, df, weights):
    
    df.index = pd.to_datetime(df.index)
    
    weights = weights.values[0]
    weights = np.round(weights.astype(float),4)
    
    # How many stock
    assetsQuant = len(df.columns)
    
    # Equal weights base on quantity of stock
    equal_weights = 1/len(df.columns)
    
    # How much inwested in singular stock base on chesen weights
    buyAmt = investmentAmt * weights
    
    # How much inwested in singular stock base on equal weights
    buyAmtEqual = investmentAmt * equal_weights
    
    # Period of analysis in mounths
    periodMonth = (df.index[-1].year - df.index[0].year) * 12 + (df.index[-1].month - df.index[0].month)
    
    # calculate cumulated returns
    retail_returns = df.pct_change(1).dropna()
    cum_returns = (1 + retail_returns).cumprod()
    
    # Let's save col names
    cols = cum_returns.columns
    
    # We want to see highest profitable stock alone
    highest_profit_ticker = cum_returns.iloc[-1].sort_values(ascending=False).index[0]
    cum_returns['highest_profitable_stock'] = cum_returns[highest_profit_ticker]*investmentAmt
    print(f'Highest profitable stocks is {highest_profit_ticker}')
    
    # Create new column which count cumulated value of portfolio with equal weights
    cum_returns['equalWeightedPortfolio'] = (cum_returns[cols]*buyAmtEqual).sum(axis=1)
    
    # Create new column which count cumulated value of portfolio with weights choesen in previous notebook
    cum_returns['YourPortfolio'] = (cum_returns[cols]*buyAmt).sum(axis=1)
        
    return cum_returns, buyAmt


def comparePortfolios_Plot(investmentAmt, cum_returns, max_y = 1.2, freq=20):
    
    fig, ax = plt.subplots()

    # PLot equal weighted portfolio
    cum_returns.reset_index().plot(x = 'Date', y = 'equalWeightedPortfolio', ax = ax, alpha=0.7)

    # PLot your portfolio, 
    cum_returns.reset_index().plot(x = 'Date', y = 'YourPortfolio', ax = ax)
    
    # PLot your portfolio
    cum_returns.reset_index().plot(x = 'Date', y = 'highest_profitable_stock', ax = ax)
    
    # lets use log view on chart
    plt.yscale('log')

    # Set min, max and freq of y label
    max_ = cum_returns.iloc[-1].max()
    plt.yticks(np.arange(investmentAmt, max_*max_y, investmentAmt*freq))

    # Format Y labels to be more readable for user
    y_value=['{:,.0f}'.format(x).replace(",", " ") + ' USD' for x in ax.get_yticks()]
    ax.set_yticklabels(y_value);
    plt.title('Portfolio size comparison for 3 different weights')

    return plt.show()

def calculate_proportions(cum_returns, df,weights):
    
    weights = weights.values[0]
    weights = np.round(weights.astype(float),4)

    # copy cymulated returns. Only tickers. cum_rets contain also total value, etc.
    prop = cum_returns.iloc[:, 0:len(df.columns)]
    
    # create data frame with index equal cum_rets index. Index is dates 
    proportions = pd.DataFrame(index=prop.index, columns=prop.columns.values)

    # In range len of df loop and calculate current weight for each day
    for i in range(len(prop.columns)):
        # multiplay 
        proportions.iloc[:,i] = prop.iloc[:,i]*weights[i] / np.sum(prop*weights, axis=1)*100
    
    return proportions

def plot_proportions(proportions):
    
    proportions.plot()
    
    plt.title('Weights of assets in your porfolio over time')
    
    return plt.show()


# THERE IS 2 POSIBLE WAY TO PROVIDE REBALANCING METHOD:
# 1. WE CAN INVESTIGATE PROPORTION OF THE PORTFOLIO BY CHOESEN PERIOD FOR EG. EACH QUATER OR EACH YEAR PASS
# 2. WE CAN INVESTIGATE PROPORTION OF THE PORTFOLIO BY MARGIN, FOR EG. IF ANY ASSETS'S PROPORTION WILL EXCEED THE ASSUMED MARGIN


# This 6 finction below is about FIRST method (period)

# First we need to check frequency, which user has choesen to do his rebalancing
def choese_rebalancing_frquency(quaterly, yearly):
    freqs = [quaterly, yearly]
    choesen_frq = []

    for freq in freqs:

        if freq.get_interact_value() == True:
            freq_name = freq.description
            choesen_frq.append(freq_name)

        else:
            pass

    # Condition: User must select 1 of the wallet
    if len(choesen_frq) == 0:
        print('You must select at least 1 frequency')
    
    elif len(choesen_frq) > 1:
        print('You CAN NOT select more than 1 frequency at one time')
        
    else: 
        print(f'Your choesen frequency: {choesen_frq}')
        
    return choesen_frq

# Create dates for rebalancing base on frequency
def create_dates_for_rebalancing(df, n, frequency):
    
    start = df.index[0]
    
    if frequency =='quaterly':

        # How many quaters in period
        quaters = (len(df)/n)*4
        # Number of months i one quater multiplyed by quaters
        quantity = np.floor(quaters*3).astype(int)
        rebalancing_dates = []
        for i in range(3, quantity, 3):
            
            rebalancing_dates.append(start + relativedelta(months=i))
            
            
        if rebalancing_dates[-1] > df.index[-1]:
            rebalancing_dates = rebalancing_dates[:-1]
            
        reb_quant = len(rebalancing_dates)
            
        print(f'In this period you could make {reb_quant} rebalancing')

    elif frequency =='yearly':

        years = len(df)/n
        quantity = np.floor(years+1).astype(int) # I add one becouse we start after first year in loop
        rebalancing_dates = []

        for i in range(1,quantity):      

            # Add i year from first transaction
            rebalancing_dates.append(start + relativedelta(years=i))
            
        if rebalancing_dates[-1] > df.index[-1]:
            rebalancing_dates = rebalancing_dates[:-1]
            
        reb_quant = len(rebalancing_dates)
            
        print(f'In this period you could make {reb_quant} rebalancing')

    else:
        print('You must select period of frequency')

    # Contain restults in 1 D array
    rebalancing_dates = np.array(rebalancing_dates)
    
    return rebalancing_dates

# This function will find first aveliable date to make rebalancing, becouse if user select 252 days frequency, there is no dates for weekend
# That is why we must go back to friday, which is closed working day
def repair_rebalancing_dates(rebalancing_dates, n):
    
    if n == 252:
        reb_dates = pd.DataFrame(rebalancing_dates)
        prep_reb_dates = []
        for i in range(len(reb_dates)):

            if reb_dates.iloc[i,0].weekday() == 5:
                new_date = reb_dates.iloc[i,0] + dt.timedelta(days=-1)
                prep_reb_dates.append(new_date)

            elif reb_dates.iloc[i,0].weekday() == 6:
                new_date = reb_dates.iloc[i,0] + dt.timedelta(days=-2)
                prep_reb_dates.append(new_date)

            else:
                new_date = reb_dates.iloc[i,0]
                prep_reb_dates.append(new_date)
                
    else:
        prep_reb_dates = rebalancing_dates
        
    return prep_reb_dates


def calculate_new_proportions(portfolio_value):
    
    # We need to count proportion for each stock again
    new_proportions = pd.DataFrame(index=portfolio_value.index, columns=portfolio_value.columns.values)

    # Loop by new portfolio value and count new prop
    for i in range(len(portfolio_value.columns)):

        new_proportions.iloc[:,i] = portfolio_value.iloc[:,i]*100 / np.sum(portfolio_value, axis=1)
        
    return new_proportions


    
def calc_portfolio_value_by_date(portfolio_value, new_proportions, rebalancing_dates, weights, i):
    
    # keep weights vales in array 
    weights = weights.values[0]
    weights = np.round(weights.astype(float),4)
    
    # That is how portfolio balance look like in chacking period
    check_day_prop = new_proportions.loc[[rebalancing_dates[i]]][new_proportions.columns].values

    # That is how we sould multiply our proportion to get proper one
    multiplier = weights*100/check_day_prop

    # It contains data year befor rebalancing date, which do not need to change anth
    previos_period = portfolio_value.loc[:rebalancing_dates[i]]

    # It contains data after Rebalancing with new prop proportion
    next_period = portfolio_value.loc[rebalancing_dates[i] + relativedelta(days=1):]*multiplier

    # Portfolio value in time with new wages
    portfolio_value = pd.concat([previos_period, next_period])
    
    return portfolio_value



def rebalancing_model_over_period(rebalancing_dates, weights, buyAmt, cum_returns, proportions, calc_portfolio_value_by_date, calculate_new_proportions):
    
    # This is the value of the portfolio from cumulated return multiplyed by buy Amount
    portfolio_value = cum_returns.iloc[:,:len(buyAmt)]*buyAmt
    
    # copy old proportions. New will be overwrtited
    new_proportions = proportions
    
    
    # Loop throw rabalancing dates and make changes if necessery
    for i in range(len(rebalancing_dates)):
        
        # Function that calulates portfolio values over all period
        portfolio_value = calc_portfolio_value_by_date(portfolio_value, new_proportions, rebalancing_dates, weights, i)
        
        # Function that calculate new proportions base on new portfolio vales in period
        new_proportions = calculate_new_proportions(portfolio_value)
        
    return portfolio_value, new_proportions


#-------------------------------------------------------------------------------------------------------------------

# This is method number 2, where rebalancing is done if any asset's proportion excced margin


# Function that calculate portfolio value by row (In first method, we iterate by rebalancing dates, here we ust iterate by all rows to find the values that exceed margin
def calc_portfolio_value_by_row(new_proportions, portfolio_value, weights, i):
    
    # Take proportion of all asets in choesen period
    check_day_prop = new_proportions.iloc[i].values

    # That is how we sould multiply our proportion to get proper one
    multiplier = weights*100/check_day_prop

     # It contains data year befor rebalancing date, which do not need to change anth
    previos_period = portfolio_value.iloc[:i]

    # It contains data after Rebalancing with new prop proportion
    next_period = portfolio_value.iloc[i:]*multiplier

    # Portfolio value in time with new wages
    portfolio_value = pd.concat([previos_period, next_period])
    
    return portfolio_value

def rebalancing_model_over_margin(selection, buyAmt, cum_returns, proportions, weights, calc_portfolio_value_by_row, calculate_new_proportions):
    
    # Format margin type to float number
    margin = float(selection)
    
    # if margin is 20% it means that asset must increase to 120% of it's value or decrease to 80% of it's value 
    up_margin = (1 + margin) * 100
    low_margin = (1 - margin) * 100
    
    # keep weights vales in array 
    weights = weights.values[0]
    weights = np.round(weights.astype(float),4)

    # Margin paramiters
    max_weights = weights * up_margin
    min_weights = weights * low_margin

    # Set first iterator as 0
    i = 0

    # Portfolio without rebalancing. It'll be owerrited
    portfolio_value = cum_returns.iloc[:,:len(buyAmt)]*buyAmt

    # copy old proportions. New will be overwrtited
    new_proportions = proportions
    
    # create variable that count how many rebalancing has been done
    counter = 0

    while i < len(proportions)-1:

        # add 1 to iterator in each sequence of the loop
        i+=1

         # iterate by all columns (assets)
        for k in range(len(proportions.columns)):

            # If proportions are over the margin then: do rebalancing
            if new_proportions.iloc[i,k] < min_weights[k] or new_proportions.iloc[i,k] > max_weights[k]:
                
                # Functions that calculates value of portfolio by choesen row
                portfolio_value = calc_portfolio_value_by_row(new_proportions, portfolio_value, weights, i)

                # Function that calculates proportion base on portfolio value
                new_proportions = calculate_new_proportions(portfolio_value)
                
                # iterate when function has done the operations
                counter+=1

                # We break the k loop. We do not need to check other assets, becouse if any is over the margin we do reb anyway
                break

            else:
                pass
            
    print(f'The algorithm did {counter} rebalancing')

    return new_proportions, portfolio_value



#-----------------------------------------------------------------------------------
# This functions compare all 2 methods and let user to choose his favourite one

# Select method function
def choese_rebalancing_method(period, margin):
    methods = [period, margin]
    choesen_method = []

    for method in methods:

        if method.get_interact_value() == True:
            method_name = method.description
            choesen_method.append(method_name)

        else:
            pass

    # Condition: User must select 1 of the wallet
    if len(choesen_method) == 0:
        print('You must select at least 1 frequency')
    
    elif len(choesen_method) > 1:
        print('You CAN NOT select more than 1 frequency at one time')
        
    else: 
        print(f'Your choesen frequency: {choesen_method}')
        
    return choesen_method

# Check what user selected and display next checkboxes:
def display_next_options(choesen_method, quaterly, yearly):
    if choesen_method == ['Over choesen period']:
        
        # If user select reb per period, then he must select the frequency
        print('Select frequency of rebalancing:')
        selection = display(quaterly, yearly)
    else:
        # if he select reb. per margin, then he must select acceptable margin
        print('Select margin of rebalancing:')
        selection = input()
        
    return selection


# run rebalancing with selected method
def run_selected_rebalancing_method(choesen_method, weights, buyAmt, cum_returns, proportions, n,quaterly, yearly, df, rebalancing_model_over_period, rebalancing_model_over_margin, selection, calc_portfolio_value_by_date,calc_portfolio_value_by_row, calculate_new_proportions, repair_rebalancing_dates, choese_rebalancing_frquency, create_dates_for_rebalancing):
    
    if choesen_method == ['Over choesen period']:

        choesen_frq = choese_rebalancing_frquency(quaterly, yearly)
        rebalancing_dates = create_dates_for_rebalancing(df,n, frequency=choesen_frq[0])
        rebalancing_dates = repair_rebalancing_dates(rebalancing_dates, n)
        portfolio_value, new_proportions = rebalancing_model_over_period(rebalancing_dates, weights, buyAmt, cum_returns, proportions, calc_portfolio_value_by_date, calculate_new_proportions)

    else:
        new_proportions, portfolio_value = rebalancing_model_over_margin(selection,buyAmt, cum_returns, proportions, weights, calc_portfolio_value_by_row, calculate_new_proportions)
        
    return new_proportions, portfolio_value




# Plots


def compare_balance_portfolio_vs_previous(investmentAmt, portfolio_value, cum_returns, freq=5):
    
   
    portfolio = pd.DataFrame(index = portfolio_value.index)
    # Sum of singular portfolio values in each period of analysis
    portfolio['BalancedPortfolio'] = np.sum(portfolio_value, axis=1)

    fig, ax = plt.subplots()

    # PLot equal weighted portfolio
    cum_returns.reset_index().plot(x = 'Date', y = 'equalWeightedPortfolio', ax = ax, alpha=0.7)

    # PLot your portfolio
    cum_returns.reset_index().plot(x = 'Date', y = 'YourPortfolio', ax = ax)

    # PLot your portfolio
    portfolio.reset_index().plot(x = 'Date', y = 'BalancedPortfolio', ax = ax)

    # lets use log view on chart
    plt.yscale('log')

    max_y = 1.1
    # Set min, max and freq of y label
    max_ = cum_returns.drop('highest_profitable_stock', axis=1).iloc[-1].max()
    plt.yticks(np.arange(investmentAmt, max_*max_y, investmentAmt*freq))

    # Format Y labels to be more readable for user
    y_value=['{:,.0f}'.format(x).replace(",", " ") + ' USD' for x in ax.get_yticks()]
    ax.set_yticklabels(y_value);
    
    plt.title('Portfolio size comparison with rebalancing included')
    
    return plt.show()


def create_table_with_all_portfolios(cum_returns, portfolio_value):
    
    # Create dataframe with 3 types of wallets value
    all_portfolios = cum_returns[['equalWeightedPortfolio', 'YourPortfolio']]
    all_portfolios['YourRebalancedPortfolio'] = np.sum(portfolio_value, axis=1)
    
    return all_portfolios

