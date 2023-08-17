import os
from math import pi
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import xgboost as xgb

def split_stock(stock_info: pd.DataFrame) -> tuple:
    x = []
    y = []
    N = stock_info.shape[0]
    direction_labels = stock_info['LABEL'].tolist()
    count = 0  # Initialize count here
    N = N - 5
    for i in range(N):
        end = count + 5
        x.append(direction_labels[count: end])
        y.append(0 if direction_labels[end] <= 0 else 1)  # Map labels to 0 and 1
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

    return X_train, y_train, X_test, y_test


def train_xgboost_model(X_train, y_train, num_round=100):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        'objective': 'multi:softmax',  # For multiclass classification
        'num_class': 2,  # Number of classes
        'eval_metric': 'merror'  # Metric for evaluation
    }
    
    model = xgb.train(params, dtrain, num_round)
    return model

def evaluate_xgboost_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


df = pd.read_csv("TSLA.csv")
X_train, y_train, X_test, y_test = split_stock(df)

xgb_model = train_xgboost_model(X_train, y_train)
accuracy = evaluate_xgboost_model(xgb_model, X_test, y_test)
print(f'XGBoost Validation Accuracy: {accuracy:.4f}')
