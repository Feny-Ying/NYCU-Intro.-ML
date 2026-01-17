"""
1. Complete the implementation for the `...` part
2. Feel free to take strategies to make faster convergence
3. You can add additional params to the Class/Function as you need. But the key print out should be kept.
4. Traps in the code. Fix common semantic/stylistic problems to pass the linting
"""

from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        """Question1
        Complete this function
        """
        X = np.insert(X, X.shape[1], 1, axis=1)
        y = np.array(y)
        self.weights = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.intercept = self.weights[-1]

    def predict(self, X):
        """Question4
        Complete this function
        """
        X = np.array(X)
        X = np.insert(X, X.shape[1], 1, axis=1)
        return X @ self.weights


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        # 標準化加強訓練穩定度
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / std

        samples, features = X.shape
        self.weights = np.zeros(features)
        self.intercept = 0
        losses = []
        y = y.squeeze()  # 去除batch維度

        for epoch in range(epochs):
            prediction = X @ self.weights + self.intercept
            loss = np.mean((prediction - y) ** 2)
            losses.append(loss)
            weight_gradient = (2 / samples) * (X.T @ (prediction - y))
            intercept_gradient = (2 / samples) * np.sum(prediction - y)
            self.weights -= learning_rate * weight_gradient
            self.intercept -= learning_rate * intercept_gradient

            if epoch % 1000 == 0:
                logger.info(f'EPOCH {epoch}, {loss=:.4f}, {learning_rate=:.4f}')

        # 回復原本的值
        self.intercept = self.intercept - np.sum((self.weights * mean) / std)
        self.weights = self.weights / std

        return losses

    def predict(self, X):
        return X @ self.weights + self.intercept


def compute_mse(prediction, ground_truth):
    mse = np.mean((prediction - ground_truth) ** 2)
    return mse


def main():
    train_df = pd.read_csv('./train.csv')   # Load training data
    test_df = pd.read_csv('./test.csv')     # Load test data
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)

    """This is the print out of question1"""
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=0.02, epochs=2000)

    """
    This is the print out of question2
    Note: You need to screenshot your hyper-parameters as well.
    """
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    """
    Question3: Plot the learning curve.
    Implement here
    """
    plt.figure(figsize=(12, 5))
    plt.plot(losses)
    plt.title('Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    """Question4"""
    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
