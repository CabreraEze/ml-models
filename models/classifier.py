import numpy as np
import argue


class TreeNode:

    def __init_(self, node_label=None, feature_index=None, threshold=None, left_branch=None, rigth_branch=None):
        self.node_label = node_label
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_branch = left_branch
        self.right_branch = rigth_branch


class DessicionTreeClassifier:
    
    @argue.options(criterion=('gini', 'entropy'))
    def __init__(self, criterion='gini', max_depth=2, min_split=2, min_samples=1) -> None:
        self.criterion_funct = getattr(self, criterion)
        self.max_depth = max_depth
        self.min_split = min_split
        self.min_samples = min_samples

    def fit(self, X, y):
        self.n_features = X.shape[1]

        self.tree = self.insert_tree(np.hstack(X, np.array([y,]).T))

    def gini(labels):
        p = 0

        for label in np.unique(labels):
            p_label =  len(labels[labels == label]) / len(labels)
            p += p_label**2

        return 1-p
    
    def entropy(labels):
        h = 0

        for label in np.unique(labels):
            p_label = len(labels[labels == label]) / len(labels)
            h += p_label*np.log2(p_label)
        
        return -h

    def get_information_gain(self, left_labels, right_labels, labels):
        left_impurity = self.criterion_funct(left_labels) * len(left_labels) / len(labels)
        right_impurity = self.criterion_funct(right_labels) * len(right_labels) / len(labels)
        total_impurity = self.criterion_funct(labels)

        return total_impurity - left_impurity - right_impurity

    def best_split(self, data):
        info_gain = -np.inf
        labels = data[:,-1]

        split_params = [[] for i in range(5)]
        split_params[4] = -np.inf
        for feature_index in range(self.n_features):
            for feature_value in np.unique(data[:,feature_index]):
                left_data = data[data[:,feature_index] <= feature_value]
                rigth_data = data[data[:,feature_index] > feature_index]

                if len(left_data) != 0 and len(rigth_data) != 0:
                    left_labels = left_data[:,-1]
                    right_labels = rigth_data[:,-1]

                    new_info = self.get_information_gain(left_labels, right_labels, labels)

                    if new_info > info_gain:
                        info_gain = new_info

                        split_params[0] = left_data
                        split_params[1] = rigth_data
                        split_params[2] = feature_index
                        split_params[3] = feature_value
                        split_params[4] = info_gain

        return split_params

    def insert_tree(self, data, current_depth=0):
        samples = data.shape[1] - 1

        if samples >= self.min_split and current_depth <= self.max_depth:
            left_data, right_data, feature_index, threshold_value, info_gain = self.best_split(data)
            
            if info_gain > 0:
                if left_data.shape[1] - 1 >= self.min_samples and right_data.shape[1] - 1 >= self.min_samples:
                    left_tree = self.insert_tree(left_data, current_depth=current_depth+1)
                    right_tree = self.insert_tree(right_data, current_depth=current_depth+1)

                    return TreeNode(feature_index=feature_index, threshold=threshold_value, left_brach=left_tree, right_branch=right_tree)

        node_label, _ = max({item:data[:,-1].count(item) for item in np.unique(data[:,-1])}.items(), key=lambda item: item[1])
        return TreeNode(node_label=node_label)        
        
    def predict(self, X):
        return [self.single_predict(X_sample, self.tree) for X_sample in X]

    def single_predict(self, X, tree):

        if tree.node_class != None: return tree.node_class

        if X[tree.feature_index] <= tree.threshold:
            return self.single_predict(X, tree.left_branch)
        else:
            return self.single_predict(X, tree.right_branch)
