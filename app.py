import dataGrab
import stock_prediction
from flask import Flask, render_template, request, send_from_directory
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use the "Agg" backend
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Your API key and file path
api_key = "6867462a39c1a35e7b402886aa354cb97219a5a8"
file_path = '/Users/lucasmazza/Desktop/Stock_Price/Stock-Price-Predict-Pytorch/UserRequest.csv'  # training data

# Start and end dates
start_date = "1994-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')  # Current date
app = Flask(__name__, static_folder='static')
plot_image_filename = "static/prediction.png"
@app.route('/plot/predictions.png')
def serve_plot_image():
    return send_from_directory('static',plot_image_filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form.get('symbol')

        df = dataGrab.get_stock_prices(symbol, api_key, start_date, end_date, file_path)
        stock_dict = stock_prediction.pre_process(file_path)
        data = stock_prediction.get_train_valid(stock_dict)
        net, train_loss, val_loss = stock_prediction.train(data, max_epochs=1000)
        
        # Generate and save the plot image
        stock_prediction.plot_predictions(net, stock_dict)
        
        # Rename the generated plot image to match the expected format in the template
        plot_image_filename = "predictions.png"
        plot_path = os.path.join('static', plot_image_filename)
        os.rename('predictions.png', plot_path)

        return render_template('index.html', plot_image_filename=plot_image_filename)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
