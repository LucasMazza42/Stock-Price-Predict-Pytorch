import os
from math import pi
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from model import MyNetwork
import tensorflow as tf 
from MyCustomLoss import MyCustomLoss

try:
    from tqdm import tqdm
    
except ImportError:
    tqdm = None


def get_device() -> torch.device:
    """
    DO NOT MODIFY.

    Set the device to GPU if available, else CPU
    
    Args:
        None

    Returns:
        torch.device
            'cuda' if NVIDIA GPU is available
            'mps' if Apple M1/M2 GPU is available
            'cpu' if no GPU is available
    """

    device = torch.device('mps')
    



device = get_device()

def pre_process(file_path: str) -> dict:
   
    df = pd.read_csv(file_path)
    stocks = sorted(list(set(df['Stock']))) # stocks in alphabetical order
   
    stock_dict = dict()
    
    for stock in stocks:
   
        
        currentdf = df.loc[df['Stock'] == stock]
        dates = pd.to_datetime(df['Date'])
        currentdf['Date'] = dates
        currentdf.sort_values(by='Date', inplace=True)
        currentdf = currentdf.reset_index()
        stock_dict[stock] = currentdf
    
     
    
    return stock_dict
def plot_data(stock_dict: dict) -> None:
   

    stocks = list(stock_dict.keys())

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=3.0)
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(stock_dict[stocks[i*2+j]]['Close'].values)
            axs[i, j].set_title(f'{stocks[i*2+j]}')

    for ax in axs.flat:
        ax.set(xlabel='days', ylabel='close price')

    plt.savefig(os.path.join(os.path.dirname(__file__), 'stocks_history.png'))


def split_stock(stock_info: pd.DataFrame) -> tuple:


    #we use the first 5 days to then predict the 6th day closing price
    
    x = []
    y = []
    
    count = 0 
    close = stock_info['Close'].tolist()
    N = stock_info.shape[0]
   
    N = N - 5
    for i in range(N):
        end = count + 5
        x.append(close[count: end]) 
        y.append(close[end])
        count += 1
   
    num_train = int(len(x) * 0.7)
    X_train = x[0: num_train]
    y_train = y[0: num_train]

    X_test = x[num_train:]
    y_test = y[num_train:]
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    return(X_train, y_train, X_test, y_test)
   



def get_train_valid(stock_dict: dict) -> tuple:
    
    x_train, y_train, x_val, y_val = [],[],[],[]
    for stock in stock_dict:
        currentStock = stock_dict[stock]
         
        dataInfo = pd.DataFrame(currentStock)
    
        dataInfo = dataInfo.drop(['index'], axis = 1)
        values = split_stock(currentStock)
        
        x_train.append(values[0])
       
        y_train.append(values[1])
        x_val.append(values[2])
        y_val.append(values[3])
       
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)
    
  
    return (x_train, y_train, x_val, y_val)
def my_NLLloss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
   
    mse_loss = nn.MSELoss()
    loss = mse_loss(pred[:, 0], y)  # Only consider the predicted prices for MSE
    return loss


def train(data: tuple, max_epochs: int = 200, seed=12345) -> tuple:
    torch.manual_seed(seed)

    if tqdm is not None:
        iterator = tqdm(range(max_epochs))
    else:
        iterator = range(max_epochs)
        
    net = MyNetwork(5, 100, 2)

    x_train, y_train, x_val, y_val = data
    
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    x_train = x_train.to(torch.float32)
    y_train = y_train.to(torch.float32)
    x_val = x_val.to(torch.float32)
    y_val = y_val.to(torch.float32)

    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    net.to(device)

    train_losses = []
    val_losses = []

    print('---------- Training has started: -------------')
   
    optimizer = torch.optim.Adam(net.parameters())
    
    criterion = MyCustomLoss()  # Use the custom loss function

    for epoch in iterator: 
        optimizer.zero_grad()

        train_outputs = net(x_train)
        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_outputs = net(x_val)
            val_loss = criterion(val_outputs, y_val)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if tqdm is not None:
            iterator.set_description(f' Epoch: {epoch+1}')
            iterator.set_postfix(train_loss=round(train_loss.item(), 1),
                                 val_loss=round(val_loss.item(), 1))
        else:
            print(f'epoch {epoch+1}: train_loss = {train_loss}, val_loss = {val_loss}')

    print('---------- Training ended. -------------\n')
    
    plt.figure()
    epochs = list(range(max_epochs))
    plt.plot(epochs[5:], train_losses[5:])
    plt.plot(epochs[5:], val_losses[5:])
    plt.legend(['train', 'val'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'training_curve.png'))
    plt.close()

    return net, train_losses[-1], val_losses[-1]



def plot_predictions(model: nn.Module, stock_dict: dict) -> None:
    

    fig, axs = plt.subplots(2, 2) # axs may be useful
    fig.tight_layout(pad=3.0) # give some space between subplots
    
    for k, stock in enumerate(list(stock_dict.keys())):
        (_, _, x_val, y_val) = split_stock(stock_dict[stock])


   
        pred = model(torch.Tensor(x_val).to(device)).detach().cpu().numpy()
        

        pred_prices, pred_risks = pred[:, 0], np.sqrt(np.exp(pred[:, 1]))
        #RMSE 
        rmse = np.sqrt(np.mean((pred_prices - y_val) ** 2))
        print(f'RMSE for {stock} is: {rmse}')

        #R2
        mean_y_true = np.mean(y_val)
    
        ssr = np.sum(np.square(y_val - pred_prices))
        sst = np.sum(np.square(y_val - mean_y_true))
        
        r2 = 1 - (ssr / sst)
        print(f'R2 for {stock} is: {r2}')

        i, j = k // 2, k % 2

        prices_range = [pred_prices - pred_risks, pred_prices + pred_risks]
        axs[i, j].plot(y_val[:50])
        axs[i, j].plot(pred_prices[:50])
        axs[i, j].legend(['real', 'pred'])
        axs[i, j].fill_between(list(range(50)), prices_range[0]
                               [:50], prices_range[1][:50], color=None, alpha=.15)
        axs[i, j].set_title(f'{stock}')
    
    plt.savefig(os.path.join(os.path.dirname(__file__), 'predictions.png'))
    print('Predictions plotted.')



if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear')

    file_path = os.path.join(os.path.dirname(
        __file__), '/Users/lucasmazza/Desktop/Stock_Price/Stock-Price-Predict-Pytorch/stock_train.csv')
    stock_dict = pre_process(file_path)
   
    plot_data(stock_dict)

    data = get_train_valid(stock_dict)
    
    net, train_loss, val_loss = train(data, max_epochs=1000)
    print(plot_predictions(net, stock_dict))

