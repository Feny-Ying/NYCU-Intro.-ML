import typing as t
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation in the `intermediate layers` allowed.
    """
    def __init__(self, input_dim, hidden_dim: int = 10) -> None:
        super(WeakClassifier, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 1)
        #self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        return x

def entropy_loss(outputs, targets):
    outputs_flat = outputs.float().squeeze()  # 確保是 1D
    targets_flat = targets.float().squeeze()  # 確保是 1D
    
    eps = 1e-6
    outputs = torch.clamp(outputs_flat, eps, 1 - eps)
    loss = -torch.sum(targets_flat * torch.log(outputs) + (1 - targets_flat) * torch.log(1 - outputs)) / outputs.shape[0]
    
    return loss
    raise NotImplementedError


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    plt.figure()
    for i, y_pred in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc="lower right")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig(fpath)
    plt.close()

def preprocess(df: pd.DataFrame):
    df = df.copy()
    # 類別變數編碼
    categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    # 標準化數值特徵
    numeric_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    
    for col in numeric_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
    
    # 將所有欄位轉換為 float32
    df = df.astype(np.float32)
    
    return df