# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys
import time
import seaborn as sns
import os
import math
import pickle as pkl

#%%
# ML
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer

import torch
import torch.nn as nn
import torch.optim as optim

# # Anomaly detection models
# import pyod
# from pyod.models.ocsvm import OCSVM
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import (LocalOutlierFactor, NearestNeighbors, KNeighborsClassifier)


def setup():
    print("Beginning Data Loading...")
    # Data Loading: first setup
    with open('../data/setup_1.pkl', 'rb') as f:
        setup1 = pkl.load(f)

    X_train, _, X_test, y_test = setup1['X_train'], setup1['y_train'], setup1['X_test'], setup1['y_test']

    X_train_without_id = X_train[:, 1:]  # Excluding 'id'
    X_test_without_id = X_test[:, 1:]  # Excluding 'id'

    # Todo: Try Different Scalers
    scaler = MinMaxScaler().fit(X_train_without_id)  # Initialize the MinMaxScaler and fit to the training set
    X_train_without_id_scaled = scaler.transform(X_train_without_id)  # the scaler is applied to the training set
    X_test_without_id_scaled = scaler.transform(X_test_without_id)  # the scaler is applied to the test set

    X_train_without_id_scaled_mini = X_test_without_id_scaled[:10000]  # Prototyping with only 10000 instances.
    print("Data Loading finished.")

    return X_train_without_id_scaled_mini, X_test_without_id_scaled, y_test

def model_training(X_train_without_id_scaled):
    print("Beginning model training...")
    # Timing and Training the model
    start_time = time.time()
    model_name = 'One-Class SVM'
    model = OneClassSVM().fit(X_train_without_id_scaled)
    duration = time.time() - start_time
    print(f"Training time: {duration:.2f} seconds")
    print("Model training finished.")
    return model, model_name

def evaluation(model, X_test_scaled_df, y_test, model_name):
    print("Beginning evaluation...")
    # Predict on the test set
    y_pred_test = model.predict(X_test_scaled_df)
    # Convert predictions to match y_test labels (0 for anomalies, 1 for normal)
    y_pred_test = (y_pred_test == 1).astype(int)

    # Calculate ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_test)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall Curve and AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_test)
    pr_auc = auc(recall, precision)

    # Generate a classification report
    class_report = classification_report(y_test, y_pred_test)

    # Plotting the ROC and Precision-Recall Curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic ' + model_name)
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve ' + model_name)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    print(class_report)
    print("Evaluation finished.")

def main():
    X_train_without_id_scaled, X_test_without_id_scaled, y_test = setup()
    model, model_name = model_training(X_train_without_id_scaled)
    evaluation(model, X_test_without_id_scaled, y_test, model_name)

if __name__ == '__main__':
    main()
