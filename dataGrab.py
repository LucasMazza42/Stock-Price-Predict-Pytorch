import requests
import pandas as pd

def get_stock_prices(symbol, api_key, start_date, end_date):
    base_url = "https://api.tiingo.com/tiingo/daily/{}/prices".format(symbol)
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Token {}".format(api_key),
    }
    params = {
        "startDate": start_date,
        "endDate": end_date,
    }

    response = requests.get(base_url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        print("Error:", response.status_code)
        return None

# Replace 'YOUR_API_KEY' with your actual Tiingo API key
api_key = "6867462a39c1a35e7b402886aa354cb97219a5a8"
symbol = "AAPL"
start_date = "2018-08-14"  # 5 years ago from the current date
end_date = "2023-08-14"    # Current date

stock_prices_df = get_stock_prices(symbol, api_key, start_date, end_date)
if stock_prices_df is not None:
    print("Stock Prices DataFrame:")
    print(stock_prices_df.head())
else:
    print("Failed to retrieve stock prices.")
