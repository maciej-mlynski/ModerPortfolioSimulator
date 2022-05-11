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

