import dataGrab
import stock_prediction
from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Your API key and file path
api_key = "6867462a39c1a35e7b402886aa354cb97219a5a8"
file_path = '/Users/lucasmazza/Desktop/Stock_Price/Stock-Price-Predict-Pytorch/UserRequest.csv'  # training data

# Start and end dates
start_date = "1994-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')  # Current date

# Route to handle user input and display predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    if request.method == 'POST':
        symbol = request.form.get('symbol')
        predicted_price = stock_prediction.predict_stock_price(symbol)  # Implement this function or logic
        
        df = dataGrab.get_stock_prices(symbol, api_key, start_date, end_date, file_path)
        stock_dict = stock_prediction.pre_process(file_path)
        data = stock_prediction.get_train_valid(stock_dict)
        net, train_loss, val_loss = stock_prediction.train(data, max_epochs=1000)
        plot_path = stock_prediction.plot_predictions(net, stock_dict)
        
        return render_template('index.html', predicted_price=predicted_price, symbol=symbol, plot_path=plot_path)
    
    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
