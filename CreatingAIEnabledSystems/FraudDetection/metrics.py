from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
import os
import base64
import matplotlib.pyplot as plt

class Metrics:
    """
    A class that handles the calculation of various metrics to evaluate a fraud detection model.

    Methods
    -------
    generate_report(y_true, y_pred, y_pred_proba, output_dir):
        Generates a metrics report based on predictions and true labels and writes it to a text file.
        
    precision(y_true, y_pred):
        Calculate the precision of the predictions.
        
    recall(y_true, y_pred):
        Calculate the recall of the predictions.
        
    sensitivity_specificity(y_true, y_pred):
        Calculate the sensitivity and specificity of the predictions.
        
    precision_recall_curve(self, y_true, y_pred_proba):
        Generate a precision-recall curve.
        
    roc_auc(y_true, y_pred_proba):
        Generate a ROC-AUC curve.
    """

    def precision(self, y_true, y_pred):
        """Calculate the precision of the predictions.
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels by the model.
            
        Returns
        -------
        precision_score: float
            The calculated precision score for the model
        """
        return precision_score(y_true, y_pred)

    def recall(self, y_true, y_pred):
        """Calculate the recall of the predictions.
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels by the model.
        
        Returns
        -------
        recall_score: float
            The calculated recall score for the model
        """
        return recall_score(y_true, y_pred)

    def sensitivity_specificity(self, y_true, y_pred):
        """Calculate the sensitivity and specificity of the predictions.
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels by the model.
        
        Returns
        -------
        sensitivity: float
            The calculated sensitivity score for the model

        specificity: float
            The calculated specificity score for the model
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return sensitivity, specificity

    def roc_auc(self, y_true, y_pred_proba):
        """
        Generate a ROC-AUC curve.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_pred_proba : array-like
            Target scores, can either be probability estimates of the positive class.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        img_path = 'results/roc_curve.png'
        plt.savefig(img_path)
        plt.close()

    def precision_recall_curve(self, y_true, y_pred_proba):
        """
        Generate a precision-recall curve.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_pred_proba : array-like
            Target scores, can either be probability estimates of the positive class.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        img_path = 'results/precision_recall_curve.png'
        plt.savefig(img_path)
        plt.close()
    
    def generate_report(self, y_true, y_pred, y_pred_proba, output_dir='results'):
        """
        Generate a report with various metrics and write it to a text file.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels by the model.
        y_pred_proba : array-like
            Predicted probabilities by the model.
        output_dir : str, optional
            Directory to store the output report (default is 'results').
        """
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics/generate plots
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        sensitivity, specificity = self.sensitivity_specificity(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        self.roc_auc(y_true, y_pred_proba)
        self.precision_recall_curve(y_true, y_pred_proba)
        
        text_content = f"""
        Model Results:
        Precision: {precision:.2%}
        Recall: {recall:.2%}
        Sensitivity: {sensitivity:.2%}
        Specificity: {specificity:.2%}
        ROC-AUC: {roc_auc:.2%}
        """
        
        # Write the report to a text file
        report_path = os.path.join(output_dir, 'metrics_report.txt')
        with open(report_path, 'a') as file:
            file.write(text_content)
        
        print(f"Report successfully written to {report_path}")