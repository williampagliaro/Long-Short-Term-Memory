1. Strategy Overview

The strategy combines a diversified portfolio allocation inspired by the Harry Browne Permanent Portfolio with machine learning through a Long Short-Term Memory (LSTM) model to predict price movements and dynamically rebalance positions. It operates on four key assets—SPY (equities), TLT (bonds), GLD (gold), and SHY (cash equivalents)—designed to balance risk across different economic conditions. The strategy aims to optimize the portfolio based on LSTM-based price predictions and minimize risk using Markowitz Portfolio Optimization through a covariance matrix of asset volatilities.

Key Elements:


•	Technical Indicators: Uses historical price data to calculate expected returns.

•	LSTM Model: A recurrent neural network that uses recent historical prices to predict future price changes, helping to identify the best expected returns for each asset.
•	Feature Extraction: The main feature used is the closing price of each asset for training and predictions.


Core Strategy Process:


•	The strategy uses the LSTM model to calculate the best expected returns for each asset in the portfolio. These expected returns are then fed into the Markowitz Portfolio Allocation model, which determines the optimal weightage for each asset based on the expected returns and associated risk metrics.
•	The portfolio weights are rebalanced monthly based on the optimized allocation from the Markowitz model, ensuring the strategy dynamically adjusts to market conditions and strives to maintain an optimal balance between risk and return.

2. Technical Indicators Used

•	Historical Price Data:

•	The strategy utilizes daily closing prices of SPY, TLT, GLD, and SHY.

•	These prices are used to compute expected returns, which inform the portfolio optimization process.

•	Lookback Period:

•	A 252-day lookback is used to gather historical data for the LSTM model.

•	The portfolio is rebalanced monthly, integrating the LSTM model’s predictions with historical price data.

The combination of price data and predictions from the LSTM model helps optimize portfolio weights and adjust positions based on predicted future returns.

3. Long Short-Term Memory (LSTM) Model

The LSTM model is central to the strategy’s ability to predict future price movements and find the best expected returns for each asset, which directly influences portfolio weight adjustments.
 
LSTM Overview:


•	Long Short-Term Memory (LSTM): An advanced type of Recurrent Neural Network (RNN) designed to capture both short-term and long-term dependencies in sequential data like stock prices.
•	Why LSTM: Stock prices form a time series where the order of data matters. LSTMs are effective because they capture dependencies over time, allowing for accurate predictions of future price movements.
•	Purpose: The LSTM predicts future prices, which are then used to calculate expected returns. These returns are crucial for determining how the portfolio weights of each asset should be adjusted, optimizing the balance between risk and reward.

4. Training and Weight Optimization

The goal of training the LSTM is to detect patterns in historical data that allow accurate price predictions. The process is divided into training and testing phases, focusing on identifying the best expected returns.

Training Phase:


•	The LSTM processes up to 500 days of historical price data for each asset.

•	Model Architecture:

1.	Two LSTM layers with 50 units each to retain patterns over time.

2.	A Dense layer outputs the predicted price.

•	Training Steps:

1.	Price Scaling: Prices are scaled between 0 and 1 using MinMaxScaler for efficient processing.

2.	Sequence Creation: 252-day sequences of prices are used as input; the price on the 253rd day is the target prediction.

3.	Model Training: Trained over 50 epochs with a batch size of 32, minimizing mean squared error.


Testing Phase:


•	Post-training, the model predicts future prices using unseen data. During testing (from 2010 to 2024), the model’s weights remain fixed.
•	The immediate prediction (253rd day) helps guide decisions over a broader 21-day outlook for rebalancing.

Key Hyperparameters:


•	Lookback Period: 252 days.

•	Prediction Length: Predicts prices 21 days into the future by rolling forward daily predictions.

•	LSTM Units: 50 per layer.

•	Epochs: 50 with a batch size of 32.


5. Feature Extraction for the LSTM
 

The model utilizes a single feature from historical data:
 
•	Closing Price: Daily closing prices of SPY, TLT, GLD, and SHY are scaled between 0 and 1, fed into the LSTM for future price movement predictions over the next 21 days.

6. Trading Logic and Rules

The strategy uses the LSTM predictions to dynamically rebalance the portfolio based on the expected returns and asset volatility.

•	Long Entry Conditions:

•	A positive expected return from the LSTM prediction suggests that an asset’s price is likely to increase.
•	The portfolio increases the weight of assets with higher predicted returns, aiming to maximize the overall expected return of the portfolio.
•	Short Entry Conditions:

•	No shorting occurs; however, positions may be reduced in assets with low or negative expected returns to minimize exposure to assets expected to underperform.
•	Portfolio Rebalancing:

•	The portfolio is rebalanced monthly based on LSTM model predictions.

•	Weights are determined through Markowitz Portfolio Optimization, balancing expected returns against risk metrics like volatility.

7. Markowitz Portfolio Optimization

The strategy uses Markowitz Portfolio Optimization to determine optimal portfolio weights based on the LSTM-predicted returns.

•	Expected Returns: Derived from LSTM model predictions, representing the anticipated performance of each asset.
•	Volatility Calculation:

•	The strategy calculates the volatility of each asset by measuring the standard deviation of its returns over the lookback period (252 days). This measures how much an asset’s price fluctuates, providing an indication of risk.
•	Covariance Matrix Calculation:

•	The covariance matrix is a key component of the Markowitz Optimization. It measures the pairwise covariances of asset returns, showing how the returns of each asset move relative to the others.

•	Calculation Process:

•	For each pair of assets, the covariance is calculated using historical return data over the lookback period.
•	The covariance measures the degree to which two assets move together. A positive covariance indicates that the assets tend to move in the same direction, while a negative covariance indicates that they move in opposite directions.
•	This matrix helps the optimization process by quantifying the diversification benefits, allowing the strategy to balance assets that don’t move together to reduce overall portfolio risk.
 
•	Optimization: Balances the trade-off between maximizing expected returns and minimizing portfolio risk, adjusting the weights of each asset accordingly.
•	Constraints: Includes a minimum weight constraint to ensure diversification.


8. Predictions and Decision-Making

The LSTM model’s predictions are critical in shaping the strategy:


•	Positive Prediction: Indicates an expected price increase, triggering a higher portfolio weight for the asset.
•	Negative Prediction: Indicates a likely price decrease, leading to a reduced weight for the asset in the portfolio.
