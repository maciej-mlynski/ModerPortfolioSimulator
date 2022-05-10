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


# # Select Stocks and Periods functions

# In[2]:


# def function that take selected tickers
def choese_stock(SPX_, BTC_, IEF_, FILL_, PALL_):

    # All aveliable tickers
    tickers = [SPX_, BTC_, IEF_, FILL_, PALL_]
    
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


# # Data preparation functions

# In[8]:


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


# # Markowitz Model functions

# In[4]:


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

def calculate_returns(weights, log_rets):
    
    # Calculate logaritmic returns for given weights
    returns = np.sum(log_rets.mean()*weights) * 365
    return returns

def calculate_volatility(weights, log_rets):

    # We uce covariance instead of log_rets
    log_rets_cov = log_rets.cov()

    # Calc annual covariance for given weight
    annualized_cov = np.dot(log_rets_cov*365, weights)
    # Use covariance to calculate volotality and std
    vol = np.dot(weights.transpose(),annualized_cov)
    volatility = np.sqrt(vol)

    return volatility

def symulatePortfoliosWeights(df, iterations, calculate_N, calculate_log_rets, gen_weights, calculate_returns, calculate_volatility):        

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

        sim_returns = calculate_returns(weights, log_rets)
        portfolio_returns.append(sim_returns)

        sim_vol = calculate_volatility(weights, log_rets)
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


# Create checkbox values
# User can choose highest sharp ratio wallet weights or weights selected by max vol
HighestSR= Checkbox(description = 'Highest Sharp Ratio')
MyWallet = Checkbox(description = 'My Choice')

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


# # Rebalancing and summary of all portfolio functions

# In[5]:


# def dowload_data(weights, df):
    
#     # Stocks prices
#     df = pd.read_csv('DATA/PrepData.csv', date_parser=True)
#     df['Date'] = pd.to_datetime(df['Date'])
#     df.set_index('Date', inplace=True)
    
#     # Choesen wallet
#     weights = weights.values[0]
#     weights = np.round(weights,4)
    
#     return weights, df
    
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

def create_dates_for_rebalancing(df, frequency):
    
    start = df.index[0]
    
    if frequency =='quaterly':

        # How many quaters in period
        quaters = (len(df)/365)*4
        # Number of months i one quater multiplyed by quaters
        quantity = np.floor(quaters*3).astype(int)
        rebalancing_dates = []
        for i in range(3, quantity, 3):
            rebalancing_dates.append(start + relativedelta(months=i))
            
        print(f'In this period you could make {np.floor(quantity/3).astype(int)} rebalancing')

    elif frequency =='yearly':

        years = len(df)/365
        quantity = np.floor(years+1).astype(int) # I add one becouse we start after first year in loop
        rebalancing_dates = []

        for i in range(1,quantity):      

            # Add i year from first transaction
            rebalancing_dates.append(start + relativedelta(years=i))
            
        print(f'In this period you could make {quantity-1} rebalancing')

    else:
        print('You must select period of frequency')

    # Contain restults in 1 D array
    rebalancing_dates = np.array(rebalancing_dates)
    
    return rebalancing_dates

def create_new_wallet_with_rebalanceing(df, weights, rebalancing_dates, buyAmt, cum_returns, proportions):
    
    # keep weights vales in array 
    weights = weights.values[0]
    weights = np.round(weights.astype(float),4)
    
    # This is the value of the portfolio from cumulated return multiplyed by buy Amount
    portfolio_value = cum_returns.iloc[:,:len(df.columns)]*buyAmt
    
    # copy old proportions. New will be overwrtited
    new_proportions = proportions
    
    # Loop throw rabalancing dates and make changes if necessery
    for i in range(len(rebalancing_dates)):

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

        # We need to count proportion for each stock again
        proportions = pd.DataFrame(index=portfolio_value.index, columns=portfolio_value.columns.values)

        # Loop by new portfolio value and count new prop
        for i in range(len(portfolio_value.columns)):

            new_proportions.iloc[:,i] = portfolio_value.iloc[:,i]*100 / np.sum(portfolio_value, axis=1)
            
    return portfolio_value, new_proportions


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


def dropdown_summary(all_portfolios):
    
    # Create another table with dropdowns
    dropdowns = pd.DataFrame(round(all_portfolios.pct_change(30).dropna().min()*100,2)) # Montlhy
    dropdowns = dropdowns.rename(columns={0:'Month'})
    dropdowns['Quater'] = pd.DataFrame(round(all_portfolios.pct_change(90).dropna().min()*100,2)) # Quater
    dropdowns['Year'] = pd.DataFrame(round(all_portfolios.pct_change(365).dropna().min()*100,2))
    dropdowns[['Month', 'Quater', 'Year']] = dropdowns[['Month', 'Quater', 'Year']].astype(str) + '%'
    
    print('MAx dropdowns summary:')
    
    return dropdowns

def returns_summary(all_portfolios, investmentAmt):
    
    # create DF with all portfolios
    returns = pd.DataFrame(index=all_portfolios.columns)

    # Calculate Mnthly returns for 3 portfolios
    equalM = ((all_portfolios.iloc[-1,0]/investmentAmt)**(1/len(all_portfolios)*30) - 1)*100
    YourM = ((all_portfolios.iloc[-1,1]/investmentAmt)**(1/len(all_portfolios)*30) - 1)*100
    BalM = ((all_portfolios.iloc[-1,2]/investmentAmt)**(1/len(all_portfolios)*30) - 1)*100
    # append to DF
    returns['Monthly'] = [np.round(equalM,2), np.round(YourM,2), np.round(BalM,2)]

    # Calculate Quaterly returns for 3 portfolios
    equalQ = ((all_portfolios.iloc[-1,0]/investmentAmt)**(1/len(all_portfolios)*90) - 1)*100
    YourQ = ((all_portfolios.iloc[-1,1]/investmentAmt)**(1/len(all_portfolios)*90) - 1)*100
    BalQ = ((all_portfolios.iloc[-1,2]/investmentAmt)**(1/len(all_portfolios)*90) - 1)*100
    # append to df
    returns['Quaterly'] = [np.round(equalQ,2), np.round(YourQ,2), np.round(BalQ,2)]

    # Calculate Yearly returns for 3 portfolios
    equalY = ((all_portfolios.iloc[-1,0]/investmentAmt)**(1/len(all_portfolios)*365) - 1)*100
    YourY = ((all_portfolios.iloc[-1,1]/investmentAmt)**(1/len(all_portfolios)*365) - 1)*100
    BalY = ((all_portfolios.iloc[-1,2]/investmentAmt)**(1/len(all_portfolios)*365) - 1)*100
    # append to df
    returns['Yearly'] = [np.round(equalY,2), np.round(YourY,2), np.round(BalY,2)]
    
    # add percent sign
    returns[['Monthly', 'Quaterly', 'Yearly']] = returns[['Monthly', 'Quaterly', 'Yearly']].astype(str) + '%'
    
    print('Returns Summary:')
    
    return returns


# In[ ]:




