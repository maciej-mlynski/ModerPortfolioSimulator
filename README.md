# Modern Portfolio Simulator

I created this tool for individual investors who:
- invest long-term (more than 3 years)
- base their strategy not only on profit, but also on risk
- want to adjust the portfolio to an acceptable risk
- want to know how often and if it is worth rebalancing their portfolio

## Step by Step
Here I would like to dircribe what I have done in each step and explain theory and my idea behind this.

### 1. Stocks selection
 
User should select at least 1 stock for each markets (equity, commodity, bonds, crypto, metals, etc.). I create API that download data from Yahoo straight away.

**Future improvements:** I prepare few tickers that useer can choese, but in the future I will add posibility to select any ticker aveliable on Yahoo Finance. 
 
### 2. Period selection

User can select any date range, but it doesn't say that it will be used in next steps of analysis. All stock must be in the same date range, so becouse of that I wrote a function that find common date range for all stocks. It also checks day of the week and if user selected weekend it directly move to closest aveliable week day date.

### 3. Data preparation

That was hardest part. Mainly becouse I would like to develop a tool that users cloud compare all differents stocks. As known crypto martket is aveliable 24/7, unlikely to traditional market, which open at business days only. 

First of all, algo checks if any asset is cryptocurrency by conecting to API and checking quote type. If the anserw is YES, user can choese frequency period in year:

- 365: Adding missing dates to traditional markets. It just repeats Fridays listing vales for Saturdays and Sundays.
- 252: Cut weekend listings for any cryptocuriency user has in wallet

If there is no crypto in user's portfolio the checkbox will not apear. It will just take 252 days frequency per year straight away and prepare data.

**Future improvements:** In the future I will gain posibility that user could fill missing dates by mean value, calculated by mean vol before missing value occurred, instead of repeating lising values from friday. (In 365 freq case)

### 4. Markowitz Portfolio Simulation

Here user can simulate thousends of different weights of his portfolio's assets. Obviously he can select number of iterations he is intrested in. In this step it might take a while to get results. It depends on period, number of iterations and quantity of different stock that user selected priviously. 

**Future improvements:** The tool is prepared mainly for index investing like ETF or BTC. However some of ETF has high correlation with each other. In the near future I would like to prepare a function that will check correlation between stock and exclude the markets with the lowest sharp ratio from the markets that are correlated above 80% with user permission.

### 5. Find the best model for user

At this step user has 2 posibilities to find the best weights for his portfolio with given assets:

1. He can select portfolio with the highest Sharp Ratio (calculated by dividind expected annual log return by expected annual volotality)
2. Select max expected annual volatality and find the one with highest return. 

![](https://github.com/maciej-mlynski/ModerPortfolioSimulator/blob/main/Img/MarkowitzModelSimulation.png?raw=true)

The figure above prezents 30 000 different portfolio's annual log return vs expected volatility created by Markowitz simulation. 

**Green star** is portfolio with higherst Sharp Ratio selected automaticly

**Blue star** is portfolio with highest Return base on max acceptable volatality

At this step user has to choese which portfolio wants to analyze. 

### 6. Check portfolio weights over choesen period and do the rebalancing if necessary.

In order for the Markowitz model portfolio must has specific weights over full period of invetment. Some of sotck might grow faster than another. In this step I created a function that do rebalancing. Unfortunetly, there is no point to rebalance portfolio whenever the weights sligtly change, mainly becouse of transaction cost and taxes. So we presumed that rebalancing should not be done more then one time in Quater (3 months) and not less than once a Year. User can choose which one is better for his specific case. 

**Weights over time before rebalancing**

![](https://github.com/maciej-mlynski/ModerPortfolioSimulator/blob/main/Img/WeightsUnbalanced.png?raw=true)

--------------------------------------------------------------------------------------------------------------

**Weights over time after quaterly rebalancing**

![](https://github.com/maciej-mlynski/ModerPortfolioSimulator/blob/main/Img/WeightsAfterRebalanceing.png?raw=true)


### 7. Summary and comparison

In this step we compare rebalanced portfolio with unbalanced one and with portfolio with equal weights. It is important becouse we want to know if Markowitz model proved to deliver expected results. 

**Plot**

![](https://github.com/maciej-mlynski/ModerPortfolioSimulator/blob/main/Img/walletsComparsion.png?raw=true)

**Max dropdown summary**

| Portfolio type          | Month	  | Quater	 | Year    |
| ------------------------|---------|---------|---------|
| equalWeightedPortfolio  |	-54.02%	| -56.1%	 | -74.77% |
| YourPortfolio	          | -47.44%	| -49.24%	| -66.91% |
| YourRebalancedPortfolio	| -14.97%	| -8.89%	 | -13.6%  |


**Returns summary**

| Portfolio type          | Month	  | Quater	 | Year    |
| ------------------------|---------|---------|---------|
| equalWeightedPortfolio  |	3.27%	  | 10.15%	 | 47.98%  |
| YourPortfolio	          | 2.6%	   | 8.0%	  	| 36.63%  |
| YourRebalancedPortfolio	| 1.52%   | 4.64% 	 | 20.18%  |



### FUTURE UPDATES

1. Give more aveliable tickers / User can write ticker by hand.
2. Posibility to cut trading days instead of repeating values for weekends.
3. Remove most correlated assets.
4. Improve rebalancing method: It could do rebalanceing in a specified percentage range for. eg if weight of singular asset will exceed 10% of approved weight.
5. Deliver more summary indicators.
6. Benchmark your investment against a benchmark such as the S&P 500 (calculate beta)
7. Check how the parameters would change if you invested a certain amount regularly
8. Add: Sharp ratio with risk free rate or benchmark, add Sertino Ratio and give user posibilities to choose
9. Calculate daily VARby var or Monte Carlo simulation and propose the size of the opposite position in order to protect the capital (Black Scholes Model)






