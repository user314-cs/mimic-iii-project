# MIMIC-III Project
# @author: Daniel Solá

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from hyperopt import hp, Trials, fmin, tpe
from mlxtend.evaluate import confusion_matrix as cm_mxt
from scipy import interp
from itertools import cycle
from features.get_features import Features
from labels.get_labels import PatientOutcomes
from services.plotting_service import PlottingService

# Function for imputing missing values (if necessary)
def impute_missing_values(df):
    # Implement missing value imputation logic here, if present
    return df

# Function to get neural network data (mock implementation)
def get_nn_data():
    # Mock implementation for getting neural network data
    X_train = np.random.rand(100, 10)
    X_test = np.random.rand(20, 10)
    Y_train = np.random.randint(2, size=(100, 3))
    Y_test = np.random.randint(2, size=(20, 3))
    return {'mortality_data': {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test}}

# Function for training the neural network
def train_neural_network(X_train, Y_train, X_test, Y_test, params):
    model = Sequential()
    for _ in range(params['n_layers']):
        model.add(Dense(params['n_neurons'], activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    optimizer = params['optimizer']
    if optimizer == 'SGD':
        optimizer = 'sgd'  # Changing to lowercase for Keras optimizer
    elif optimizer == 'Adam':
        optimizer = Adam(lr=0.001)  # Adjust learning rate if needed
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    return model

# Function for hyperparameter tuning
def hyperparameter_tuning(X_train, Y_train, X_test, Y_test):
    def f(params):
        return -1.0 * train_neural_network(X_train, Y_train, X_test, Y_test, params).evaluate(X_test, Y_test, verbose=0)[1]

    params_space = {
        'n_layers': hp.choice('n_layers', range(2, 8)),
        'n_neurons': hp.choice('n_neurons', [8, 16, 32, 64]),
        'optimizer': hp.choice('optimizer', ['SGD', 'Adam']),
        'epochs': hp.choice('epochs', [10, 25, 35]),
        'batch_size': hp.choice('batch_size', [1, 25, 50])
    }

    trials_mse = Trials()
    best_params = fmin(fn=f, space=params_space, algo=tpe.suggest, max_evals=50, trials=trials_mse)
    return best_params

# Function for plotting ROC curves
def plot_roc_curves(Y_test, prediction):
    n_classes = Y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'ROC curve (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiclass Classification')
    plt.legend(loc="lower right")
    plt.show()

# Function for plotting confusion matrix
def plot_confusion_matrix(Y_test, prediction):
    cm = cm_mxt(Y_test.argmax(axis=1), prediction.argmax(axis=1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function for main execution
def main():
    # Query and preprocess data
    nn_data = get_nn_data()
    X_train, X_test, Y_train, Y_test = nn_data['mortality_data']['X_train'], nn_data['mortality_data']['X_test'], nn_data['mortality_data']['Y_train'], nn_data['mortality_data']['Y_test']
    X_train, X_test = impute_missing_values(X_train), impute_missing_values(X_test)
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)

    # Hyperparameter tuning
    best_params = hyperparameter_tuning(X_train_scaled, Y_train, X_test_scaled, Y_test)

    # Training neural network with optimal parameters
    keras_model = train_neural_network(X_train_scaled, Y_train, X_test_scaled, Y_test, best_params)

    # Prediction
    prediction = keras_model.predict(X_test_scaled)

    # Evaluation
    roc_auc = roc_auc_score(Y_test, prediction)
    print(f'ROC AUC Score: {roc_auc}')

    # Plotting ROC curves and confusion matrix
    plot_roc_curves(Y_test, prediction)
    plot_confusion_matrix(Y_test, prediction)

    # Classification Report
    print(classification_report(Y_test.argmax(axis=1), prediction.argmax(axis=1)))

if __name__ == "__main__":
    main()
