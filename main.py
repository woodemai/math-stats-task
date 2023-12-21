from alpha_vantage.timeseries import TimeSeries
import numpy as np

api_key = 'AI8C0J707HVU06X0'

ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_weekly(symbol='YNDX')
def calculate_annual_returns(close_prices, open_prices):
    daily_returns = (close_prices - open_prices) / open_prices
    annual_returns = (1 + daily_returns).prod() - 1
    return annual_returns

open_prices = data['1. open'].values
close_prices = data['4. close'].values


result = np.mean(calculate_annual_returns(close_prices, open_prices))
print("Ожидаемая доходность:", result)
print(data.head())
