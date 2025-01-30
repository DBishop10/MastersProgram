import numpy as np
import pandas as pd
from DecisionTree import DecisionTreeClassifier, DecisionTreeRegressor
from Tests import *

def print_tree_to_file(node, file_path, depth=0, position="Root"):
    """
    Recursively writes the tree structure to a file in a more readable format.
    
    Parameters:
    node (Node): The current node in the tree.
    file_path (str): The path to the file where the tree structure will be written.
    depth (int): The current depth of the tree (used for indentation).
    position (str): The position of the node (Root, Left, Right).
    """
    indent = " " * depth * 4
    with open(file_path, 'a') as f:
        if node.is_leaf_node():
            f.write(f"{indent}[{position}] Leaf: Predicts {node.prediction}\n")
        else:
            f.write(f"{indent}[{position}] Node: Feature {node.feature} <= {node.threshold}, Info Gain: {node.info_gain}\n")
            print_tree_to_file(node.left, file_path, depth + 1, "Left")
            print_tree_to_file(node.right, file_path, depth + 1, "Right")

abalone_path = '../Data/abalone/abalone.data'
abalone_columns = [
    'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 
    'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'
]
abalone_df = pd.read_csv(abalone_path, header=None, names=abalone_columns)
X_abalone = abalone_df.drop(columns=['Rings']).values
y_abalone = abalone_df['Rings'].values

# Breast Cancer dataset (Classification)
breast_cancer_path = '../Data/breastcancer/breast-cancer-wisconsin.data'
breast_cancer_columns = [
    'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
    'Normal Nucleoli', 'Mitoses', 'Class'
]
breast_cancer_df = pd.read_csv(breast_cancer_path, header=None, names=breast_cancer_columns)
breast_cancer_df.replace('?', np.nan, inplace=True)
breast_cancer_df['Bare Nuclei'] = breast_cancer_df['Bare Nuclei'].astype(float)
breast_cancer_df.fillna(breast_cancer_df.mean().iloc[0], inplace=True)
X_breast_cancer = breast_cancer_df.drop(columns=['Sample code number', 'Class']).values
y_breast_cancer = breast_cancer_df['Class'].values

# Train decision tree for classification
clf = DecisionTreeClassifier(max_depth=5, debug=True)
clf.fit(X_breast_cancer, y_breast_cancer)
y_pred_class = clf.predict(X_breast_cancer)

# Train decision tree for regression
reg = DecisionTreeRegressor(max_depth=5, debug=True)
reg.fit(X_abalone, y_abalone)
y_pred_reg = reg.predict(X_abalone)

#Get first fold for both a classification tree and regression tree
cross_val_metrics(clf, X_breast_cancer, y_breast_cancer, cv=2, repeats=1)
cross_val_metrics(reg, X_abalone, y_abalone, cv=2, repeats=1, task="regression")

#Print Classification and Regression Tree
print_tree_to_file(clf.tree, "classificationTree.txt")
#Print Classification and Regression Tree
print_tree_to_file(reg.tree, "regressionTree.txt")

clf.prune(X_breast_cancer, y_breast_cancer)
reg.prune(X_abalone, y_abalone)

#Print Classification and Regression Tree
print_tree_to_file(clf.tree, "classificationTreePruned.txt")
#Print Classification and Regression Tree
print_tree_to_file(reg.tree, "regressionTreePruned.txt")

print(X_breast_cancer[0])
clf.traverse_and_predict(X_breast_cancer[0])
print(X_abalone[0])
reg.traverse_and_predict(X_abalone[0])