from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Metrics:
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate performance metrics for classification.

        Parameters:
        - y_true: The true labels.
        - y_pred: The predicted labels by the model.

        Returns:
        A dictionary with the calculated metrics.
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': mean_squared_error(y_true, y_pred, squared=False),  # squared=False to get RMSE
            'R^2': r2_score(y_true, y_pred)
        }
        return metrics