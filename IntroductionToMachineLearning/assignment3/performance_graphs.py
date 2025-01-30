import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utility import *
import os

def plot_confusion_matrix(model, X, y, save_path, classes=[0,1]):
    """
    Plots the Confusion Matrix of the output of classification datasets that the models were trained on.

    Parameters:
    model (any model type): The model that was trained on the dataset
    X (numpy.ndarray): All X Values of the dataset
    y (numpy.ndarray): All true y values of the dataset
    save_path (string): Where to save the graph images to
    classes (array): Should have been internal but it just says to have either a 1 or 0 for true and false
    """
    predictions = model.predict(X)
    y_pred = (predictions > 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, fontsize=15)
    plt.yticks(ticks, classes, fontsize=15)

    # Label each cell with the counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=15)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), bbox_inches = "tight")
    plt.close()

def plot_regression_metrics(model, X, y, save_path):
    """
    Plots the Predicted values of the model and dispalys the metrics, only really need one of these graphs but two are generated

    Parameters:
    model (any model type): The model that was trained on the dataset
    X (numpy.ndarray): All X Values of the dataset
    y (numpy.ndarray): All true y values of the dataset
    save_path (string): Where to save the graph images to
    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r_squared(y, predictions)

    # Plot MSE
    plt.figure(figsize=(10, 5))
    plt.plot(y, predictions, 'o', label='Predictions')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Perfect Prediction')
    plt.title(f'Mean Squared Error (MSE): {mse:.4f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'mse_plot.png'))
    plt.close()

    # Plot R^2
    plt.figure(figsize=(10, 5))
    plt.plot(y, predictions, 'o', label='Predictions')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Perfect Prediction')
    plt.title(f'R² Score: {r2:.4f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'r2_plot.png'))
    plt.close()