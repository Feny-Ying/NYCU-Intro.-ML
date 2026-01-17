import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        """Free to add args as you need, like batch-size, learning rate, etc."""

        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.1):
        """
        TODO: Implement the training part
        """
        n_samples = X_train.shape[0]
        self.sample_weights = torch.ones(n_samples) / n_samples  


        for t, model in enumerate(self.learners):
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X_train.iloc[idx].values if hasattr(X_train, "iloc") else X_train[idx]
            y_sample = y_train[idx]
            X_tensor = torch.tensor(X_sample, dtype=torch.float32)
            y_tensor = torch.tensor(y_sample, dtype=torch.float32).unsqueeze(1)
            # Train weak learner
            for epoch in range(num_epochs):
                outputs = model(X_tensor)
                outputs = outputs.squeeze()
                loss = entropy_loss(outputs, y_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(loss)
        return

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """
        TODO: Implement the training part
        """
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        pred_sum = torch.zeros(X_tensor.shape[0])
        pred_probs = []

        for t in range(len(self.learners)):
            outputs = torch.sigmoid(self.learners[t](X_tensor)).squeeze()
            predictions = 2 * torch.round(outputs) - 1 
            pred_sum += predictions
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
            feature_importance = np.abs(model.fc1.weight.detach().numpy().flatten())

        feature_importance /= np.sum(feature_importance)

        return feature_importance.tolist()
        raise NotImplementedError
