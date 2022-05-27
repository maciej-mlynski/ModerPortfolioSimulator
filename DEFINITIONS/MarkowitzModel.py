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



# # Markowitz Model functions


def calculate_log_rets(df):

    # calculate logaritmic daly change of all asets
    log_rets = np.log(1 + df.pct_change(1)).dropna()
    return log_rets

def calculate_N(df):

    # Quantity of stock in portfolio
    N = len(df.columns)
    return N

def gen_weights(df):

    # generate N random weights, where N is quantity of stocks that user has choesen previosly
    N = calculate_N(df)
    weights = np.random.random(N)
    return weights / np.sum(weights)

def calculate_returns(weights, log_rets, n):
    
    # Calculate logaritmic returns for given weights
    returns = np.sum(log_rets.mean()*weights) * n
    return returns

def calculate_volatility(weights, log_rets, n):

    # We uce covariance instead of log_rets
    log_rets_cov = log_rets.cov()

    # Calc annual covariance for given weight
    annualized_cov = np.dot(log_rets_cov*n, weights)
    # Use covariance to calculate volotality and std
    vol = np.dot(weights.transpose(),annualized_cov)
    volatility = np.sqrt(vol)

    return volatility

def symulatePortfoliosWeights(df, iterations,n, calculate_N, calculate_log_rets, gen_weights, calculate_returns, calculate_volatility):        

    # Chceck how many stock been choesen
    N = calculate_N(df)
    # calcl log rets using previos function
    log_rets = calculate_log_rets(df)

    # For markowitz portfolio we count this 3 features
    portfolio_returns = []
    portfolio_vol = []
    portfolio_weights = []

    # generate portfolios with different weights and calculate volotality and returns
    # iteration means how many portfolio we want to generate
    for sim in range(iterations):

        weights = gen_weights(df)
        portfolio_weights.append(weights)

        sim_returns = calculate_returns(weights, log_rets, n)
        portfolio_returns.append(sim_returns)

        sim_vol = calculate_volatility(weights, log_rets, n)
        portfolio_vol.append(sim_vol)

    # Create DF wieth all calculated features
    results = pd.DataFrame({'weights':portfolio_weights, 'volatility':portfolio_vol, 'returns':portfolio_returns})

    return results, np.array(portfolio_returns), np.array(portfolio_vol), np.array(portfolio_weights)

    
def createResultsPlot(results):

    # We want to plot volotality and return calculated base on diff weights
    portfolio_vol = results['volatility']
    portfolio_returns = results['returns']
    # Calculate sharp ratio by divideing returns by volotality
    sharpe_ratios = portfolio_returns / portfolio_vol

    plt.figure(dpi=100,figsize=(10,5))
    plt.scatter(portfolio_vol,portfolio_returns,c=sharpe_ratios)
    plt.ylabel('EXPECTDE RETURN')
    plt.xlabel('EXPECTED VOLATILITY')

    plt.colorbar(label="SHARPE RATIO");
    plt.title('Markowitz model symulation plot')

    plot = plt.show()

    return plot

    
def findHighestSharpRatio(results):

    # Calculate portfolio with highest sharp ratio
    portfolio_vol = results['volatility']
    portfolio_returns = results['returns']    
    sharpe_ratios = portfolio_returns / portfolio_vol

    maxSR = sharpe_ratios.max()
    maxSRPosition = sharpe_ratios.argmax()
    expected_vol = portfolio_vol[maxSRPosition]
    expected_ret = portfolio_returns[maxSRPosition]
    new_weights = results['weights'].iloc[maxSRPosition]

    print(f'The highest sharp ratio for selected assets is {round(maxSR,4)}')
    print(f'Expected logaritmic annual return: {round(expected_ret,4)}')
    print(f'Expected annual volatility: {round(expected_vol,4)}')
    print(f'Weights of wallet: {new_weights}')


    return maxSR, expected_vol, expected_ret, new_weights


def findBestWeightsByMaxVol(results, max_vol_yearly):

    # find the best portfolio base on max acceptable volotality for user
    # min is 0.98% of max in case that there will not be any portfolios equal to max acceptable vol
    volrange = results[(results['volatility']>max_vol_yearly*0.98) & (results['volatility']<=max_vol_yearly)]
    
    # In case the vallet does not generate that vol range or it is just not posible to get from this assets:
    if len(volrange) == 0:

        print(f'The volatility you selected {max_vol_yearly} is too low or too high. Try selecting again. Make sure the volatility is within the range of volatility shown in the chart above. ')

    else:
        
        # There might be few wallets in that range, so in that case choese the most profitable of them
        maxProfit = volrange['returns'].max()
        # Take the position of portfolio with highest annual return
        maxProfitPosition = volrange['returns'].argmax()
        # Take weights of them
        bestWeights = volrange['weights'].iloc[maxProfitPosition]
        # returns
        my_rets = volrange['returns'].iloc[maxProfitPosition]
        # volotality
        my_vol = volrange['volatility'].iloc[maxProfitPosition]
        # sh
        sharpRatio = my_rets / my_vol

        print(f'The best portfolio consistent with the assumed annual volatility can achieve an average log return of {round(maxProfit*100,2)}%')
        print(f'Proposed portfolio weights: {bestWeights}')
        print(f'Expected sharp ratio: {sharpRatio}')

    return my_rets, my_vol, sharpRatio, bestWeights


def createResultsPlot2(results, expected_vol, expected_ret):

    # Same plot as before but we add star sign where the highest Sharp ratio portfolio is
    portfolio_vol = results['volatility']
    portfolio_returns = results['returns']
    sharpe_ratios = portfolio_returns / portfolio_vol

    plt.figure(dpi=100,figsize=(10,5))
    plt.scatter(portfolio_vol,portfolio_returns,c=sharpe_ratios)
    plt.ylabel('EXPECTDE RETS')
    plt.xlabel('EXPECTED VOL')
    plt.title('Markowitz model symulation plot with highest Sharp Ratio')

    # best sharp ratio by max sharpratio
    plt.plot(expected_vol, expected_ret, 'g*', markersize=20.0)
    

    plt.colorbar(label="SHARPE RATIO");
    plot = plt.show()

    return plot


def createResultsPlot3(results, expected_vol, expected_ret, my_vol, my_rets):
    
    # Same as previos but we add also our choesen max vol condition portfolio 
    portfolio_vol = results['volatility']
    portfolio_returns = results['returns']
    sharpe_ratios = portfolio_returns / portfolio_vol

    plt.figure(dpi=100,figsize=(10,5))
    plt.scatter(portfolio_vol,portfolio_returns,c=sharpe_ratios)
    plt.ylabel('EXPECTDE RETS')
    plt.xlabel('EXPECTED VOL')
    plt.title('Markowitz model symulation plot with highest Sharp Ratio and max risk aversion')


    # best sharp ratio by max sharpratio
    plt.plot(expected_vol, expected_ret, 'g*', markersize=20.0)
    
    # max vol portfolio with blue star
    plt.plot(my_vol, my_rets, 'g*', markersize=20.0, color='blue')

    plt.colorbar(label="SHARPE RATIO");
    plot = plt.show()

    return plot


def create_summary(maxSR, sharpRatio, expected_vol, my_vol, expected_ret, my_rets, new_weights, bestWeights):
    
    # Create summary of 2 types of wallet:
    # 1. Max sharp-ratio calculated by markowitz model
    # 2. Best return portfolio by max volotality
    df = pd.DataFrame({'Wallet': ['Highest Sharp Ratio', 'My Choice'], 'Sharp Ratio': [maxSR, sharpRatio]
                  , 'Expected volatility':[expected_vol, my_vol]
                  , 'Expected log return': [expected_ret, my_rets]
                  , 'Wallet weights' : [new_weights, bestWeights]
                 })
    df.set_index('Wallet', inplace=True)
    
    return df


def choese_wallet(HighestSR, MyWallet):
    tickers = [HighestSR, MyWallet]
    choesen_wallet = []

    for ticker in tickers:

        if ticker.get_interact_value() == True:
            ticker_name = ticker.description
            choesen_wallet.append(ticker_name)

        else:
            pass

    # Condition: User must select 1 of the wallet
    if len(choesen_wallet) == 0:
        print('You must select at least 1 wallet in analyse perpouse')
    
    elif len(choesen_wallet) > 1:
        print('You CAN NOT select more than 1 wallet in analyse perpouse')
        
    else: 
        print(f'Your choice: {choesen_wallet}')
        
    return choesen_wallet

# save choesen weights for future analysis
def create_weightsDataFrame(summary, tickers, df):

    choice = summary['Wallet weights'].loc[tickers].values
    choice = choice[0].astype(float)
    wallet = pd.Series(choice).astype(str).str.split(expand=True).T
    wallet.columns = df.columns
    
    return wallet



