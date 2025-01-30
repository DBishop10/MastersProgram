import numpy as np
import pandas as pd
from collections import Counter

class Node:
    """
    Represents a node in the decision tree.

    Attributes:
        feature (int): Index of the feature to split on.
        threshold (float): Threshold value for the split.
        left (Node): Left child node.
        right (Node): Right child node.
        info_gain (float): Information gain from the split.
        prediction (any): Prediction at the leaf node.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, prediction=None):
        """
        Initializes a Node.

        Args:
            feature (int): Index of the feature to split on.
            threshold (float): Threshold value for the split.
            left (Node): Left child node.
            right (Node): Right child node.
            info_gain (float): Information gain from the split.
            prediction (any): Prediction at the leaf node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.prediction = prediction
    
    def is_leaf_node(self):
        """
        Checks if the node is a leaf node.

        Returns:
            bool: True if the node is a leaf, False otherwise.
        """
        return self.prediction is not None


class DecisionTreeClassifier:
    """
    A decision tree classifier.

    Attributes:
        max_depth (int): The maximum depth of the tree.
        debug (bool): Flag for debugging output.
        tree (Node): The root of the decision tree.
        first_gain_ratio (bool): Flag to control debugging output for gain ratio.
    """
    def __init__(self, max_depth=None, debug=False):
        """
        Initializes the DecisionTreeClassifier.

        Args:
            max_depth (int): The maximum depth of the tree.
            debug (bool): Flag for debugging output.
        """
        self.max_depth = max_depth
        self.tree = None
        self.debug = debug
        self.first_gain_ratio = True

    def fit(self, X, y):
        """
        Fits the decision tree to the data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        self.tree = self._build_tree(X, y)
    
    def _entropy(self, y):
        """
        Computes the entropy of a set of labels.

        Args:
            y (np.ndarray): Target vector.

        Returns:
            float: Entropy of the labels.
        """
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy
    
    def _gain_ratio(self, X_column, y, threshold):
        """
        Computes the gain ratio for a split.

        Args:
            X_column (np.ndarray): Feature column.
            y (np.ndarray): Target vector.
            threshold (float): Threshold value for the split.

        Returns:
            float: Gain ratio for the split.
        """
        parent_entropy = self._entropy(y)
        
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        info_gain = parent_entropy - child_entropy
        if(self.debug and self.first_gain_ratio):
            print(f"Parent Entropy: {parent_entropy}, Child Entropy: {child_entropy}, info gain is parent_entropy - child entropy so: {info_gain}")

        split_info = -((n_left / n) * np.log2(n_left / n) + (n_right / n) * np.log2(n_right / n))
        if split_info == 0:
            return 0
        
        gain_ratio = info_gain / split_info
        if(self.debug and self.first_gain_ratio):
            print(f"Information Gain: {info_gain}, Split Info: {split_info}, gain ratio = info gain/split info: {gain_ratio}")
            self.first_gain_ratio = False
        return gain_ratio
    
    def _split(self, X_column, split_thresh):
        """
        Splits a feature column into left and right subsets.

        Args:
            X_column (np.ndarray): Feature column.
            split_thresh (float): Threshold value for the split.

        Returns:
            tuple: Indices of the left and right subsets.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _best_split(self, X, y):
        """
        Finds the best split for the data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            tuple: Index of the best feature and the best threshold.
        """
        best_gain_ratio = -1
        split_idx, split_thresh = None, None
        for idx in range(X.shape[1]):
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain_ratio = self._gain_ratio(X_column, y, threshold)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    split_idx = idx
                    split_thresh = threshold
        return split_idx, split_thresh
    
    def _build_tree(self, X, y, depth=0):
        """
        Builds the decision tree recursively.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            depth (int): Current depth of the tree.

        Returns:
            Node: Root node of the decision tree.
        """
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        if num_labels == 1:
            return Node(prediction=np.unique(y)[0])

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(prediction=Counter(y).most_common(1)[0][0])

        if n_samples <= 1:
            return Node(prediction=Counter(y).most_common(1)[0][0])

        best_split = self._best_split(X, y)
        if best_split[0] is None:
            return Node(prediction=Counter(y).most_common(1)[0][0])

        feature_idx, threshold = best_split
        left_idxs, right_idxs = self._split(X[:, feature_idx], threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(prediction=Counter(y).most_common(1)[0][0])
        
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature=feature_idx, threshold=threshold, left=left, right=right, info_gain=self._gain_ratio(X[:, feature_idx], y, threshold))

    def predict(self, X):
        """
        Predicts the class labels for the input data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return np.array([self._predict(inputs, self.tree) for inputs in X])
    
    def _predict(self, inputs, node, first_traversal=False):
        """
        Recursively predicts the class label for a single input.

        Args:
            inputs (np.ndarray): Input features.
            node (Node): Current node in the tree.
            first_traversal (bool): Flag for debugging the first traversal.

        Returns:
            any: Predicted class label.
        """
        if node.is_leaf_node():
            if self.debug and first_traversal:
                print(f"Traversal: Reached leaf node. Predicted class: {node.prediction}")
                first_traversal = False
            return node.prediction

        if inputs[node.feature] <= node.threshold:
            if self.debug and first_traversal:
                print(f"Traversal: At node with feature {node.feature} and threshold {node.threshold}. Going left.")
            return self._predict(inputs, node.left, first_traversal)
        else:
            if self.debug and first_traversal:
                print(f"Traversal: At node with feature {node.feature} and threshold {node.threshold}. Going right.")
            return self._predict(inputs, node.right, first_traversal)
    
    def prune(self, X_val, y_val):
        """
        Prunes the decision tree using validation data.

        Args:
            X_val (np.ndarray): Validation feature matrix.
            y_val (np.ndarray): Validation target vector.
        """
        self._prune_tree(self.tree, X_val, y_val, first_pruning=True)
    
    def _prune_tree(self, node, X_val, y_val, first_pruning):
        """
        Recursively prunes the decision tree.

        Args:
            node (Node): Current node in the tree.
            X_val (np.ndarray): Validation feature matrix.
            y_val (np.ndarray): Validation target vector.
            first_pruning (bool): Flag for debugging the first pruning.

        Returns:
            Node: Pruned node.
        """
        if node.is_leaf_node():
            return node

        feature_idx = node.feature
        threshold = node.threshold
        left_idxs = np.argwhere(X_val[:, feature_idx] <= threshold).flatten()
        right_idxs = np.argwhere(X_val[:, feature_idx] > threshold).flatten()

        left_val = X_val[left_idxs, :]
        right_val = X_val[right_idxs, :]
        left_y_val = y_val[left_idxs]
        right_y_val = y_val[right_idxs]

        node.left = self._prune_tree(node.left, left_val, left_y_val, first_pruning)
        node.right = self._prune_tree(node.right, right_val, right_y_val, first_pruning)

        if not isinstance(node.left, Node) or not isinstance(node.right, Node):
            return node

        error_no_prune = np.sum(y_val != self.predict(X_val))
        combined_labels = np.concatenate([left_y_val, right_y_val])
        majority_label = Counter(combined_labels).most_common(1)[0][0]
        error_prune = np.sum(y_val != majority_label)

        if error_prune <= error_no_prune:
            if(self.debug and first_pruning):
                print(f"Pruning node and replacing with leaf predicting {majority_label} as Error without pruning: {error_no_prune}, Error with pruning: {error_prune}")
                first_pruning=False
            return Node(prediction=majority_label)

        return node
    
    def traverse_and_predict(self, inputs):
        """
        Predicts the class label for a single input with traversal debugging. Used for Testing Video

        Args:
            inputs (np.ndarray): Input features.

        Returns:
            any: Predicted class label.
        """
        return self._predict(inputs, self.tree, first_traversal=True)

class DecisionTreeRegressor:
    """
    A decision tree regressor.

    Attributes:
        max_depth (int): The maximum depth of the tree.
        debug (bool): Flag for debugging output.
        tree (Node): The root of the decision tree.
        first_mse (bool): Flag to control debugging output for MSE.
    """
    def __init__(self, max_depth=None, debug=False):
        """
        Initializes the DecisionTreeRegressor.

        Args:
            max_depth (int): The maximum depth of the tree.
            debug (bool): Flag for debugging output.
        """
        self.max_depth = max_depth
        self.tree = None
        self.debug = debug
        self.first_mse = True

    def fit(self, X, y):
        """
        Fits the decision tree to the data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        self.tree = self._build_tree(X, y)

    def _mse(self, y):
        """
        Computes the mean squared error of a set of values.

        Args:
            y (np.ndarray): Target vector.

        Returns:
            float: Mean squared error of the values.
        """
        mean_y = np.mean(y)
        mse = np.mean((y - mean_y) ** 2)
        if(self.debug and self.first_mse):
            print(f"y mean: {mean_y}, y: {y}, mse = mean((y- mean of y)^2): {mse}")
            self.first_mse = False
        return mse

    def _split(self, X_column, split_thresh):
        """
        Splits a feature column into left and right subsets.

        Args:
            X_column (np.ndarray): Feature column.
            split_thresh (float): Threshold value for the split.

        Returns:
            tuple: Indices of the left and right subsets.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _best_split(self, X, y):
        """
        Finds the best split for the data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            tuple: Index of the best feature and the best threshold.
        """
        best_mse = float('inf')
        split_idx, split_thresh = None, None
        for idx in range(X.shape[1]):
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X_column, threshold)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                mse_left = self._mse(y[left_idxs])
                mse_right = self._mse(y[right_idxs])
                mse = (len(left_idxs) * mse_left + len(right_idxs) * mse_right) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    split_idx = idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        """
        Builds the decision tree recursively.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            depth (int): Current depth of the tree.

        Returns:
            Node: Root node of the decision tree.
        """
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return Node(prediction=np.mean(y))

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(prediction=np.mean(y))

        best_split = self._best_split(X, y)
        if best_split[0] is None:
            return Node(prediction=np.mean(y))

        feature_idx, threshold = best_split
        left_idxs, right_idxs = self._split(X[:, feature_idx], threshold)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature=feature_idx, threshold=threshold, left=left, right=right)

    def predict(self, X):
        """
        Predicts the target values for the input data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted target values.
        """
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _predict(self, inputs, node, first_traversal=False):
        """
        Recursively predicts the target value for a single input.

        Args:
            inputs (np.ndarray): Input features.
            node (Node): Current node in the tree.
            first_traversal (bool): Flag for debugging the first traversal.

        Returns:
            float: Predicted target value.
        """
        if node.is_leaf_node():
            if self.debug and first_traversal: #Debug print statement, utilized for video
                print(f"Traversal: Reached leaf node. Predicted value: {node.prediction}")
                first_traversal = False
            return node.prediction

        if inputs[node.feature] <= node.threshold: 
            if self.debug and first_traversal:  #Debug print statement, utilized for video
                print(f"Traversal: At node with feature {node.feature} and threshold {node.threshold}. Going left.")
            return self._predict(inputs, node.left, first_traversal)
        else:
            if self.debug and first_traversal:  #Debug print statement, utilized for video
                print(f"Traversal: At node with feature {node.feature} and threshold {node.threshold}. Going right.")
            return self._predict(inputs, node.right, first_traversal)
    
    def prune(self, X_val, y_val):
        """
        Prunes the decision tree using validation data.

        Args:
            X_val (np.ndarray): Validation feature matrix.
            y_val (np.ndarray): Validation target vector.
        """
        self._prune_tree(self.tree, X_val, y_val, first_pruning=True)
    
    def _prune_tree(self, node, X_val, y_val, first_pruning):
        """
        Recursively prunes the decision tree.

        Args:
            node (Node): Current node in the tree.
            X_val (np.ndarray): Validation feature matrix.
            y_val (np.ndarray): Validation target vector.
            first_pruning (bool): Flag for debugging the first pruning.

        Returns:
            Node: Pruned node.
        """
        if node.is_leaf_node():
            return node

        feature_idx = node.feature
        threshold = node.threshold
        left_idxs = np.argwhere(X_val[:, feature_idx] <= threshold).flatten()
        right_idxs = np.argwhere(X_val[:, feature_idx] > threshold).flatten()

        left_val = X_val[left_idxs, :]
        right_val = X_val[right_idxs, :]
        left_y_val = y_val[left_idxs]
        right_y_val = y_val[right_idxs]

        node.left = self._prune_tree(node.left, left_val, left_y_val, first_pruning)
        node.right = self._prune_tree(node.right, right_val, right_y_val, first_pruning)

        if not isinstance(node.left, Node) or not isinstance(node.right, Node):
            return node

        error_no_prune = np.mean((y_val - self.predict(X_val)) ** 2)
        combined_y_val = np.concatenate([left_y_val, right_y_val])
        mean_val = np.mean(combined_y_val)
        error_prune = np.mean((y_val - mean_val) ** 2)

        if error_prune <= error_no_prune:
            if(self.debug and first_pruning):
                print(f"Pruning node and replacing with leaf predicting {mean_val} as Error without pruning: {error_no_prune}, Error with pruning: {error_prune}")
                first_pruning=False
            return Node(prediction=mean_val)

        return node
    
    def traverse_and_predict(self, inputs):
        """
        Predicts the target value for a single input with traversal debugging. Used for Video testing.

        Args:
            inputs (np.ndarray): Input features.

        Returns:
            float: Predicted target value.
        """
        return self._predict(inputs, self.tree, first_traversal=True)