#!/usr/bin/env python3
"""
IPI Antibody Developability Prediction Platform
Final Production Version — DEC-2025
Supports: SEC & PSR | XGBoost & RF & CNN | ablang, antiberty, antiberta2, antiberta2-cssp
"""

# models/cnn.py
#  works with both DataFrame and numpy array inputs

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from config import MODEL_DIR


# ==================== DATASET — ROBUST VERSION ====================
class AntibodyDataset(Dataset):
    def __init__(self, embeddings, labels, barcodes):
        """
        embeddings: DataFrame or numpy array
        labels: np.array or list
        barcodes: list of BARCODEs (same length)
        """
        self.embeddings = embeddings
        self.labels = np.array(labels)
        self.barcodes = barcodes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Handle both DataFrame and numpy array
        if isinstance(self.embeddings, pd.DataFrame):
            emb = self.embeddings.iloc[idx].values
        else:
            emb = self.embeddings[idx]
        emb = torch.tensor(emb, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return emb, label, self.barcodes[idx]


# ==================== RESIDUAL BLOCK ====================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


# ==================== CNN MODEL ====================
class CNNModel(nn.Module):
    def __init__(self, embedding_dim=1024, dropout=0.4):
        super().__init__()
        self.res1 = ResidualBlock(1, 128)
        self.res2 = ResidualBlock(128, 256, dilation=2)
        self.res3 = ResidualBlock(256, 256, dilation=4)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


# ==================== MAIN CLASS ====================
class CNNClassifier:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, X, y, epochs=20, batch_size=16):
        print(f"Training CNN on {len(y)} samples → {X.shape[1]}D embedding")
        self.model = CNNModel(embedding_dim=X.shape[1]).to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # Convert to DataFrame if numpy array (for consistent indexing)
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
            barcodes = [f"seq_{i}" for i in range(len(X))]
        else:
            X_df = X
            barcodes = X.index.tolist()

        dataset = AntibodyDataset(X_df, y, barcodes)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for emb, label, _ in loader:
                emb, label = emb.to(self.device), label.to(self.device)
                optimizer.zero_grad()
                out = self.model(emb)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step(total_loss)
            if total_loss < best_loss:
                best_loss = total_loss
                self.best_state = self.model.state_dict()

        self.model.load_state_dict(self.best_state)
        print("CNN training completed.")
        return self

    def predict_proba(self, X):
        self.model.eval()
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
            barcodes = [f"pred_{i}" for i in range(len(X))]
        else:
            X_df = X
            barcodes = X.index.tolist()

        dataset = AntibodyDataset(X_df, [0]*len(X), barcodes)
        loader = DataLoader(dataset, batch_size=32)
        probs = []
        with torch.no_grad():
            for emb, _, _ in loader:
                out = self.model(emb.to(self.device))
                prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                probs.extend(prob)
        return np.array(probs)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"CNN model saved → {path}")

    @classmethod
    def load_old(cls, path, embedding_dim):
        instance = cls()
        instance.model = CNNModel(embedding_dim=embedding_dim)
        instance.model.load_state_dict(torch.load(path))
        instance.model.eval()
        return instance
    
    @classmethod
    @classmethod
    def load(cls, path, embedding_dim):
        """
        Load ANY old CNN model — works with:
        - BinderClassifier (your old name)
        - CNNClassifier (your new name)
        - Full model save or state_dict
        """
        instance = cls()
        instance.model = CNNModel(embedding_dim=embedding_dim)
        
        try:
            # Try loading as state_dict (new safe way)
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
            instance.model.load_state_dict(state_dict)
            print(f"Loaded model (state_dict): {path}")
        except:
            # Fall back to old full-model loading
            print(f"Loading legacy full model (old BinderClassifier): {path}")
            old_model = torch.load(path, map_location='cpu', weights_only=False)
            # Extract state_dict regardless of wrapper
            if hasattr(old_model, 'state_dict'):
                state_dict = old_model.state_dict()
            elif isinstance(old_model, dict) and 'state_dict' in old_model:
                state_dict = old_model['state_dict']
            else:
                state_dict = old_model  # direct state_dict save
            instance.model.load_state_dict(state_dict)
            print(f"Successfully loaded legacy model")

        instance.model.eval()
        instance.model.to(instance.device)
        return instance

    # ==================== K-FOLD CV — FINAL ROBUST VERSION ====================
    @classmethod
    def kfold_validation(cls, data, X, y, embedding_lm='antiberta2', title="CNN_SEC", kfold=10, target="psr_filter"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)

        accuracy, f1, precision, recall = [], [], [], []

        plt.figure(figsize=(7, 6))

        print(f"\nCNN {kfold}-Fold CV | {title} | {embedding_lm}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, data['HCDR3_CLUSTER_0.8']), 1):
            # ROBUST: handle both DataFrame and numpy array
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]

            y_train, y_test = y[train_idx], y[test_idx]

            print(f"  Fold {fold} | Train: {len(y_train)} | Test: {len(y_test)}")

            model = cls()
            model.train(X_train, y_train, epochs=11, batch_size=16)

            prob = model.predict_proba(X_test)
            pred = (prob >= 0.5).astype(int)

        
            f1.append(f1_score(y_test, pred))
            accuracy.append(accuracy_score(y_test, pred))
            precision.append(precision_score(y_test, pred, zero_division=0))
            recall.append(recall_score(y_test, pred, zero_division=0))

            fpr, tpr, _ = roc_curve(y_test, prob)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            plt.plot(fpr, tpr, alpha=0.3, lw=1, label=f'Fold {fold} (AUC={roc_auc:.3f})')

            model.save(f"{MODEL_DIR}/cnn_{target}_{embedding_lm}_fold{fold}_k{kfold}.pt")

        # === FOR TITLE ===
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_acc = np.mean(accuracy)
        mean_f1 = np.mean(f1)
        mean_prec = np.mean(precision)
        mean_rec = np.mean(recall)

        plt.plot(mean_fpr, mean_tpr, color='darkred', lw=4,
                 label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        plt.fill_between(mean_fpr, mean_tpr - np.std(tprs,axis=0), mean_tpr + np.std(tprs,axis=0),
                         color='pink', alpha=0.3)
        plt.plot([0,1],[0,1], '--', color='gray')
        plt.xlim([0,1]); plt.ylim([0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'CNN-{target}-{embedding_lm}\n{kfold}-Fold Cross-Validation ROC Curve\n'
                  f'Acc: {mean_acc:.3f}, F1: {mean_f1:.3f}, '
                  f'Prec: {mean_prec:.3f}, Rec: {mean_rec:.3f}',
                  fontsize=11, pad=20)
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(alpha=0.3)

        plot_path = f"{MODEL_DIR}/cnn_{target}_{embedding_lm}_k{kfold}_roc.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nFINAL RESULT ({kfold}-Fold CV)")
        print(f"AUC     : {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Accuracy: {mean_acc:.4f}")
        print(f"F1-score: {mean_f1:.4f}")
        print(f"Precision: {mean_prec:.4f}")
        print(f"Recall  : {mean_rec:.4f}")
        print(f"Plot saved → {plot_path}")