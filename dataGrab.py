import requests
import pandas as pd
from datetime import datetime


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


def checkingCurrentDate(file_path):

    indicator = False
    data = pd.read_csv(file_path)
    most_recent_date = data['Date'].max()
    #current Data 
    current_date = datetime.today().strftime('%Y-%m-%d')
    print(current_date, most_recent_date)
    if current_date == most_recent_date:
        indicator = True
        print("Date is up to date")
    return indicator 

# Replace 'YOUR_API_KEY' with your actual Tiingo API key
api_key = "6867462a39c1a35e7b402886aa354cb97219a5a8"
symbols = ["AAPL",'MSFT','TSLA','META']

file_path = '/Users/lucasmazza/Desktop/Stock_Price/Stock-Price-Predict-Pytorch/stock_train.csv' #training data
trainingDataCurrent = pd.read_csv(file_path)

start_date = trainingDataCurrent['Date'].max()  # start where data ends
end_date = datetime.today().strftime('%Y-%m-%d')    # Current date
uptoDate = checkingCurrentDate(file_path)

if uptoDate is False: 

    for symbol in  symbols:
        stock_prices_df = get_stock_prices(symbol, api_key, start_date, end_date)

        if stock_prices_df is not None:
            #format the data: Date,Open,High,Low,Close,Volume,Stock

            print(stock_prices_df.columns)
            features = ['date','open','high','low','close','volume']
            stock_prices_df = stock_prices_df[features]
            stock_prices_df['date'] = stock_prices_df['date'].str.split('T').str[0]
            stock_prices_df.columns = [col.capitalize() for col in stock_prices_df.columns]
            stock_prices_df['Stock'] = symbol
            with open(file_path, 'a') as f:
                f.write('\n')
            stock_prices_df.to_csv(file_path, mode='a', header=False, index=False)

            print("Data Appended " + symbol + " to CSV File Successfully.")

        else:
            print("Failed to retrieve stock prices.")


