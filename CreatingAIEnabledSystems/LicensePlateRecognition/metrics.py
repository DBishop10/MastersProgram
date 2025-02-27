import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

class Metrics:
    def __init__(self, results_dir='results', iou_threshold=0.5):
        self.results_dir = results_dir
        self.iou_threshold = iou_threshold
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def generate_report(self, y_predictions, y_labels):
        """
        Calculates performance metrics based on a single set of prediction and label.
        Parameters:
        y_prediction: A single prediction.
        y_label: A single ground truth label.
        Returns:
        matches: all matched predictions
        unmatched_predictions: all unmatched predictions
        unmatched_labels: unmatched labels
        """
        matches = []
        unmatched_predictions = list(range(len(y_predictions)))
        unmatched_labels = list(range(len(y_labels)))

        iou_scores = np.zeros((len(y_predictions), len(y_labels)))
        for i, pred in enumerate(y_predictions):
            for j, label in enumerate(y_labels):
                iou_scores[i, j] = self.calculate_iou(pred, label)

        while True:
            max_iou = np.max(iou_scores)
            if max_iou < self.iou_threshold:
                break  # Stop if no more matches above the threshold
            i, j = np.unravel_index(np.argmax(iou_scores, axis=None), iou_scores.shape)
            matches.append((i, j))
            iou_scores[i, :] = -1  # Prevent further matching with these
            iou_scores[:, j] = -1
            if i in unmatched_predictions:
                unmatched_predictions.remove(i)
            if j in unmatched_labels:
                unmatched_labels.remove(j)

        return matches, unmatched_predictions, unmatched_labels

    def run(self, y_predictions, y_labels, report_name="average_report.txt"):
        """
        Generates a performance report for a list of predictions and labels,
        calculates their metrics, and saves the report to a file.
       
        Parameters:
        y_predictions: A list of the model's predictions.
        y_labels: A list of the ground truth labels.
        report_name: The name of the report file to be saved.
        """
        matches, unmatched_predictions, unmatched_labels = self.generate_report(y_predictions, y_labels)
        
        true_positives = len(matches)
        false_positives = len(unmatched_predictions)
        false_negatives = len(unmatched_labels)
        
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        # Compute average IoU for matched predictions
        avg_iou = np.mean([self.calculate_iou(y_predictions[i], y_labels[j]) for i, j in matches]) if matches else 0

        report_content = f"Model Results: \nAverage IoU: {avg_iou:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1_score:.2f}\n"
        report_path = os.path.join(self.results_dir, report_name)
        with open(report_path, 'w') as file:
            file.write(report_content)
        print(f"Report generated and saved to: {report_path}")
    
    def calculate_iou(self, boxA, boxB):
        """
        calculate iou
       
        Parameters:
        boxA: bounding box A
        boxB: bounding box B
        """
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection
        interArea = max(0, xB - xA) * max(0, yB - yA)
        # Compute the area of both bounding boxes
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the IoU
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou