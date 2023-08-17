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
from model import MyNetwork
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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


def calculate_feature_importance(model, feature_names):
    importance = model.get_booster().get_score(importance_type='weight')
    total = sum(importance.values())
    normalized_importance = {feature: score/total for feature, score in importance.items()}
    
    sorted_importance = sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_features, sorted_scores = zip(*sorted_importance)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_scores)), sorted_scores, align='center')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importance')
    plt.show()

df = pd.read_csv("TSLA.csv")
X_train, y_train, X_test, y_test = split_stock(df)

xgb_model = train_xgboost_model(X_train, y_train)
accuracy = evaluate_xgboost_model(xgb_model, X_test, y_test)
print(f'XGBoost Validation Accuracy: {accuracy:.4f}')
calculate_feature_importance(xgb_model, feature_names=X_train.columns)