import matplotlib.pyplot as plt
import numpy as np

def save_classification_graph(classification_metrics_before, classification_metrics_after, filename):
    """
    Saves a bar graph of classification metrics before and after pruning.

    Args:
        classification_metrics_before (list): Metrics before pruning (Accuracy, Precision, Recall, F1 Score).
        classification_metrics_after (list): Metrics after pruning (Accuracy, Precision, Recall, F1 Score).
        filename (str): The filename to save the graph as.

    """
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    before_scores = classification_metrics_before
    after_scores = classification_metrics_after

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, before_scores, width, label='Before Pruning')
    rects2 = ax.bar(x + width/2, after_scores, width, label='After Pruning')

    ax.set_ylabel('Scores')
    ax.set_title(f'Metrics Before and After Pruning on {filename} Dataset', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')

    fig.tight_layout()
    plt.savefig(f'graphs/{filename}.png')

def save_regression_graph(regression_metrics_before, regression_metrics_after, filename):
    """
    Saves a bar graph of regression metrics before and after pruning.

    Args:
        regression_metrics_before (list): Metrics before pruning (MSE, R-Squared).
        regression_metrics_after (list): Metrics after pruning (MSE, R-Squared).
        filename (str): The filename to save the graph as.

    """
    labels = ['MSE', 'R-Squared']
    before_scores = regression_metrics_before
    after_scores = regression_metrics_after

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, before_scores, width, label='Before Pruning')
    rects2 = ax.bar(x + width/2, after_scores, width, label='After Pruning')

    ax.set_ylabel('Scores')
    ax.set_title(f'Metrics Before and After Pruning On {filename} Dataset', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.bar_label(rects1, padding=3, fmt='%.4f')
    ax.bar_label(rects2, padding=3, fmt='%.4f')

    fig.tight_layout()
    plt.savefig(f'graphs/{filename}.png')