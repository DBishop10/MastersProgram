import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Manually calculates the confusion matrix for a classification problem.
    
    Parameters:
    - y_true: Actual labels as a list or numpy array.
    - y_pred: Predicted labels as a list or numpy array.
    - num_classes: The number of unique classes.
    
    Returns:
    - cm: A num_classes x num_classes matrix where cm[i][j] is the number of times
          class i was predicted as class j.
    """
    # Convert labels to integer indices if necessary
    label_to_index = {label: idx for idx, label in enumerate(np.unique(y_true))}
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for actual, predicted in zip(y_true, y_pred):
        actual_idx = label_to_index[actual]
        predicted_idx = label_to_index[predicted]
        cm[actual_idx][predicted_idx] += 1
    
    return cm
def plot_confusion_matrix(cm, classes, filename='confusion_matrix.png', datasetname="", top_n=30):
    """
    Plots a confusion matrix using matplotlib and saves it as a PNG file.
    
    Parameters:
    - cm: The confusion matrix array.
    - classes: List of class names for labeling the axes.
    - filename: Name of the file to save the plot to.
    """
    if top_n:
        # Find the indices of the top_n largest values in the confusion matrix
        cm_flat = cm.flatten()
        largest_indices = np.argsort(cm_flat)[-top_n:]
        rows, cols = np.unravel_index(largest_indices, cm.shape)
        top_cm = np.zeros_like(cm)
        top_cm[rows, cols] = cm[rows, cols]
        cm = top_cm

    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, fontsize=6)
    plt.yticks(ticks, classes, fontsize=6)

    # Label each cell with the counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=6)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the plot as a PNG file
    plt.savefig(os.path.join('graphs', datasetname ,filename))
    plt.close()  # Close the plot to free up memory

def classification_error(cm):
    """
    Calculates the classification error from the confusion matrix.
    
    Parameters:
    - cm: The confusion matrix array.
    
    Returns:
    - error_rate: The classification error rate.
    """
    correct_predictions = np.trace(cm)
    total_predictions = np.sum(cm)
    error_rate = (total_predictions - correct_predictions) / total_predictions
    return error_rate

def plot_regression_results(y_true, y_pred, filename='regression_plot.png', datasetname=""):
    """
    Plots regression results as a scatter plot of actual vs predicted values and saves it as a PNG file.
    
    Parameters:
    - y_true: Actual target values.
    - y_pred: Predicted target values.
    - filename: Name of the file to save the plot to.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')  # Line showing perfect predictions
    plt.grid(True)
    plt.savefig(os.path.join('graphs', datasetname, filename))
    plt.close()