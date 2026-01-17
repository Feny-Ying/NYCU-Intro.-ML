import numpy as np


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.feature_importance = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.tree = self._grow_tree(X, y)
        self.n_features = X.shape[1]

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(set(y)) == 1:
            leaf_value = np.bincount(y).argmax()
            return {'type': 'leaf', 'class': leaf_value}
        
        feature_index, threshold, gain = find_best_split(X, y)
        #print(f"Depth: {depth}, Feature: {feature_index}, Threshold: {threshold}, Gain: {gain}")
        if feature_index is None or gain <= 0:
            leaf_value = np.bincount(y).argmax()
            return {'type': 'leaf', 'class': int(leaf_value)}
        
        left_indices, right_indices = split_dataset(X, y, feature_index, threshold)
        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return {'type': 'node', 'feature_index': feature_index, 'threshold': threshold,
                'left': left_subtree, 'right': right_subtree}

    def predict(self, X):
        X = np.asarray(X)
        predictions = [self._predict_tree(x, self.tree) for x in X]
        return np.array(predictions)

    def _predict_tree(self, x, tree_node):
        if tree_node['type'] == 'leaf':
            return tree_node['class']
        if x[tree_node['feature_index']] <= tree_node['threshold']:
            return self._predict_tree(x, tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])
        
    def compute_feature_importance(self):
        self.feature_importance = np.zeros(self.n_features)
        self._accumulate_importance(self.tree, 1.0)
        self.feature_importance /= np.sum(self.feature_importance)
        return self.feature_importance.tolist()
    
    def _accumulate_importance(self,node, importance):
        if node['type'] == 'leaf':
            return
        feature_index = node['feature_index']
        self.feature_importance[feature_index] += importance
        self._accumulate_importance(node['left'], importance / 2)
        self._accumulate_importance(node['right'], importance / 2)


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    left_indices = np.where(X[:, feature_index] <= threshold)[0]
    right_indices = np.where(X[:, feature_index] > threshold)[0]
    return left_indices, right_indices


# Find the best split for the dataset
def find_best_split(X, y):
    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices, right_indices = split_dataset(X, y, feature_index, threshold)
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            # Calculate information gain
            y_left = y[left_indices]
            y_right = y[right_indices]
            gain = entropy(y) - (y_left.size / y.size) * entropy(y_left) - (y_right.size / y.size) * entropy(y_right)
            
            if gain > best_gain:    
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
    return best_feature, best_threshold, best_gain

def entropy(y):
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs + 1e-12))
