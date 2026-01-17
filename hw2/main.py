import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        samples, features = inputs.shape
        self.weights = np.zeros(features + 1)     # +1 for constant
        inputs = np.insert(inputs, features, 1, axis=1)

        targets = targets.squeeze()  # 去除batch維度

        for _ in range(self.num_iterations):
            z = inputs @ self.weights
            prediction = self.sigmoid(z)
            grad_cross_entropy = inputs.T @ (prediction - targets) / samples
            self.weights -= self.learning_rate * grad_cross_entropy

        self.intercept = self.weights[-1]

        # raise NotImplementedError

    def predict(self, inputs) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        samples, features = inputs.shape
        inputs = np.insert(inputs, features, 1, axis=1)
        z = inputs @ self.weights
        prediction = self.sigmoid(z)

        y_pred_probs = prediction
        y_pred_classes = (prediction > 0.5).astype(int)

        return y_pred_probs, y_pred_classes
        raise NotImplementedError

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        z = 1 / (1 + np.exp(-x))
        return z
        raise NotImplementedError


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None
        # 新增的
        self.threshold = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:

        # 分class
        X0 = inputs[targets == 0]
        X1 = inputs[targets == 1]

        # class means
        m0 = X0.mean(axis=0)
        m1 = X1.mean(axis=0)

        # within-class
        S0 = ((X0 - m0).T @ (X0 - m0))
        S1 = ((X1 - m1).T @ (X1 - m1))
        Sw = S0 + S1

        # between-class
        Sb = (m1 - m0).reshape(-1, 1) @ (m1 - m0).reshape(1, -1)

        # compute w
        Sw_inv = np.linalg.pinv(Sw)
        w = (Sw_inv @ (m1 - m0))

        m0_proj = m0 @ w
        m1_proj = m1 @ w
        threshold = (m0_proj + m1_proj) / 2
        slope = -w[0] / w[1]
        intercept = threshold / w[1]

        self.w = w
        self.m0 = m0
        self.m1 = m1
        self.sw = Sw
        self.sb = Sb
        self.slope = slope
        self.threshold = threshold
        self.intercept = intercept

        # raise NotImplementedError

    def predict(self, inputs) -> t.Sequence[t.Union[int, bool]]:
        z = inputs @ self.w  # projection
        preds = (z >= self.threshold).astype(int)
        return preds
        raise NotImplementedError

    def plot_projection(self, inputs: npt.NDArray[float], y_true: t.Sequence[int]):
        y_pred = self.predict(inputs)
        plt.figure(figsize=(7, 7))

        # projection line (gray)
        m = 0.5 * (self.m0 + self.m1)
        x_vals = np.linspace(np.min(inputs[:, 0]), np.max(inputs[:, 0]), 100)
        y_vals = m[1] + (self.w[1] / self.w[0]) * (x_vals - m[0])
        plt.plot(x_vals, y_vals, color='gray')

        # decision boundary (blue)
        x_vals = np.linspace(np.min(inputs[:, 0]), np.max(inputs[:, 0]), 100)
        y_vals = self.slope * x_vals + self.intercept
        plt.plot(x_vals, y_vals, color='blue', linestyle='--')

        # 資料的點
        for i in range(len(inputs)):
            color = 'green' if y_true[i] == y_pred[i] else 'red'
            marker = 'o' if y_true[i] == 0 else '^'
            plt.scatter(inputs[i, 0], inputs[i, 1], c=color, marker=marker, edgecolors='k')

        # 點的投影
        w_unit = self.w / np.linalg.norm(self.w)
        for i in range(len(inputs)):
            x = inputs[i]
            # 投影點 = m + ((x - m)·w_unit) * w_unit
            proj = m + np.dot((x - m), w_unit) * w_unit

            # 投影點的虛線
            plt.plot([x[0], proj[0]], [x[1], proj[1]], 'k--', linewidth=0.3, alpha=0.6)

            # 投影點
            plt.scatter(proj[0], proj[1], c='gray', s=10, zorder=3)

        plt.title(f"Projection onto FLD axis (w=[{self.w[0]:.3f},{self.w[1]:.3f}])")
        plt.axis('equal')
        plt.xlim(np.min(inputs[:, 0]) - 1, np.max(inputs[:, 0]) + 1)
        plt.ylim(np.min(inputs[:, 1]) - 1, np.max(inputs[:, 1]) + 1)
        plt.show()
        # raise NotImplementedError


def compute_auc(y_trues, y_preds):
    auc = roc_auc_score(y_trues, y_preds)
    return auc
    raise NotImplementedError


def accuracy_score(y_trues, y_preds):
    correct = 0
    for y_true, y_pred in zip(y_trues, y_preds):
        if y_true == y_pred:
            correct += 1
    acu = correct / len(y_trues)
    return acu
    raise NotImplementedError


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-3,  # You can modify the parameters as you want
        num_iterations=2000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)

    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_pred_classes = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_test, y_test)


if __name__ == '__main__':
    main()
