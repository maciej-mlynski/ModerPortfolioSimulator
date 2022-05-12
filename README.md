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

That was hardest part. Mainly becouse I would like to develop a tool that users cloud compare all differents stocks. As known crypto martket is aveliable 24/7, unlikely to traditional market. Instead of reducing the number of days for the crypto market, I cloned the Friday closing prices for Saturday and Sunday in the traditional market. I was awere that some of patterns has slightly change, so I took it into account. I know that this solution has defects. For sure We can conclude that the traditional martket would be less volatality then, but in my opinion is fair move. 

**Future improvements:** In the future I will gain posibility that user could make this decision for himself. 

### 4. Markowitz Portfolio Simulation

Here user can simulate thousends of different weights of his portfolio's assets. Obviously he can select number of iterations he is intrested in. In this step it might take a while to get results. It depends on period, number of iterations and quantity of different stock that user selected priviously. 

**Future improvements:** The tool is prepared mainly for index investing like ETF or BTC. However some of ETF has high correlation with each other. In the near future I would like to prepare a function that will check correlation between stock and exclude the markets with the lowest sharp ratio from the markets that are correlated above 80% with user permission.

### 5. Find the best model for user

At this step user has 2 posibilities to find the best weights for his portfolio with given assets:

1. He can select portfolio with the highest Sharp Ratio (calculated by dividind expected annual log return by expected annual volotality)
2. Select max expected annual volatality and find the one with highest return. 

![](https://github.com/maciej-mlynski/ModerPortfolioSimulator/blob/main/Img/MarkowitzModelSimulation.png?raw=true)

The figure above prezents 30 000 different portfolio's annual log return vs expected volatility created by Markowitz simulation. 

<div color = 'red' >Green star is portfolio with higherst Sharp Ratio</div>



















