import numpy as np
import pandas as pd
from loguru import logger
import random

import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc, preprocess
import matplotlib.pyplot as plt


def main():
    """You can control the seed for reproducibility"""
    random.seed(777)
    torch.manual_seed(777)

    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    feature_names = list(train_df.drop(['target'], axis=1).columns)

    """
    TODO: Implement you preprocessing function.
    """
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    """
    TODO: Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    # AdaBoost
    clf_adaboost = AdaBoostClassifier(X_train.shape[1])
    _ = clf_adaboost.fit(X_train, y_train)

    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
    )
    feature_importance = clf_adaboost.compute_feature_importance()
    plt.figure()
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Feature Importance')
    plt.title('AdaBoost Feature Importance')
    plt.show()

    # Bagging
    clf_bagging = BaggingClassifier(X_train.shape[1])
    _ = clf_bagging.fit(X_train, y_train)

    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
    )
    feature_importance = clf_bagging.compute_feature_importance()
    plt.figure()
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Feature Importance')
    plt.title('Bagging Feature Importance')
    plt.show()
    # Decision Tree
    clf_tree = DecisionTree(max_depth=7)
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    ginis = [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
    vals, counts = np.unique(ginis, return_counts=True)
    probs = counts / counts.sum()
    logger.info(f'DecisionTree - Gini index: {1.0 - np.sum(probs ** 2):.4f}')

    vals, counts = np.unique(ginis, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    logger.info(f'DecisionTree - entropy: {-np.sum(probs * np.log2(probs + 1e-12)):.4f}')

    feature_importance = clf_tree.compute_feature_importance()
    plt.figure()
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Feature Importance')
    plt.title('Decision Tree Feature Importance')
    plt.show()


if __name__ == '__main__':
    main()
