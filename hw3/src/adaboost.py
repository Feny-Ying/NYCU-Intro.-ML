import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        """Free to add args as you need, like batch-size, learning rate, etc."""

        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.1):
        """
        TODO: Implement the training part
        """

        n_samples = X_train.shape[0]
        self.sample_weights = torch.ones(n_samples) / n_samples  

        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        for t, model in enumerate(self.learners):
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            # Train weak learner
            for epoch in range(num_epochs):
                outputs = model(X_tensor)
                outputs = outputs.squeeze()
                loss = entropy_loss(outputs, y_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(loss)

            with torch.no_grad():
                outputs = torch.sigmoid(model(X_tensor)).squeeze()
                predictions = torch.round(outputs)
                incorrect = (predictions != y_tensor.squeeze())
                epsilon = torch.sum(self.sample_weights * incorrect.float()) / torch.sum(self.sample_weights)
                
                # 避免 log(0) 或 log(∞)
                epsilon = torch.clamp(epsilon, min=1e-10, max=1-1e-10)
                
                alpha = 0.5 * torch.log((1 - epsilon) / epsilon)

                self.sample_weights = self.sample_weights * torch.exp(-alpha * y_tensor.squeeze() * (2 * predictions - 1))
                self.sample_weights = self.sample_weights / torch.sum(self.sample_weights)

            self.alphas.append(alpha.item())

        return
        raise NotImplementedError

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        pred_sum = torch.zeros(X_tensor.shape[0])
        pred_probs = []

        for t in range(len(self.alphas)):
            outputs = torch.sigmoid(self.learners[t](X_tensor)).squeeze()
            predictions = 2 * torch.round(outputs) - 1 
            pred_sum += self.alphas[t] * predictions
            pred_probs.append(outputs.detach().numpy())

        # 改為轉換回 0/1 而不是 -1/0/1
        y_pred_classes = (np.sign(pred_sum.detach().numpy()) + 1) / 2
        y_pred_classes = np.round(y_pred_classes).astype(int)
        
        return y_pred_classes, pred_probs
        raise NotImplementedError


    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        TODO: Implement the feature importance calculation
        """
        num_features = self.learners[0].fc1.in_features
        feature_importance = np.zeros(num_features)

        for t, model in enumerate(self.learners):
            weights = model.fc1.weight.detach().numpy().flatten()
            feature_importance += self.alphas[t] * np.abs(weights)

        feature_importance /= np.sum(feature_importance)

        return feature_importance.tolist()
        raise NotImplementedError