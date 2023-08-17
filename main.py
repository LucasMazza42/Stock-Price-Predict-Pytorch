import dataGrab
import stock_prediction

import pandas as pd
from datetime import datetime


api_key = "6867462a39c1a35e7b402886aa354cb97219a5a8"
file_path = '/Users/lucasmazza/Desktop/Stock_Price/Stock-Price-Predict-Pytorch/UserRequest.csv ' #training data

start_date = "2015-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')    # Current date
#uptoDate = dataGrab.checkingCurrentDate(file_path)


RequestStock =  input("Please enter the symbol you would like to research: ")

df = dataGrab.get_stock_prices(RequestStock, api_key, start_date, end_date, file_path)

