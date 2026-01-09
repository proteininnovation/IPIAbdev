# models/transformer_lm.py
# Transformer for LM Embeddings (ablang, antiberty, antiberta2, antiberta2-cssp)
# Supports train, predict, k-fold, etc.
# IPI Antibody Developability Prediction Platform
# Created by Hoan Nguyen | Final Production Version — DEC-2025


from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import sys
import yaml
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from config import MODEL_DIR

# AntibodyDataset (for embeddings)
class AntibodyDataset(Dataset):
    def __init__(self, embeddings, labels, barcodes):
        self.embeddings = embeddings
        self.labels = labels
        self.barcodes = barcodes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        barcode = self.barcodes[idx]
        return embedding, label, barcode

# Transformer Classifier— unchanged hyperparameters
class TransClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.2, num_classes=2):
        super().__init__()
        self.embedding_fc = nn.Linear(embedding_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, embedding):
        x = self.embedding_fc(embedding.unsqueeze(1))  # [batch, 1, hidden_dim]
        x = self.transformer(x)  # [batch, 1, hidden_dim]
        x = x.squeeze(1)  # [batch, hidden_dim]
        out = self.fc(x)
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, patience=10, device='cpu', model_save="psr/psr_transformer_best_model.pth"):
    model = model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    use_validation = val_loader is not None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        for embedding, label, barcode in train_loader:
            embedding, label = embedding.to(device).float(), label.to(device).long()
            optimizer.zero_grad()
            output = model(embedding)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            train_true.extend(label.cpu().numpy())
        train_acc = accuracy_score(train_true, train_preds)

        print_str = f"Epoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Train Acc {train_acc:.4f}"

        if use_validation:
         
            model.eval()
            val_loss = 0
            val_preds = []
            val_true = []
            with torch.no_grad():
                for embedding, label, barcode in val_loader:
                    embedding, label = embedding.to(device).float(), label.to(device).long()
                    output = model(embedding)
                    loss = criterion(output, label)
                    val_loss += loss.item()
                    val_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                    val_true.extend(label.cpu().numpy())
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_true, val_preds)
            print_str += f", Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}"
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_save)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break
        else:
            print("non validation")
            # Save every epoch if no validation
            torch.save(model.state_dict(), model_save)

        print(print_str)

    print("Training completed.")
    return model

# Framework-compatible wrapper class
class TransformerLMModel:
    def __init__(self, config_path="config/transformer_lm.yaml"):
        import yaml
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"Config loaded from {config_path}")
        else:
            print(f"Warning: Config file {config_path} not found — using defaults")
            self.config = {
                'model': {
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'num_heads': 4,
                    'dropout': 0.3
                },
                'training': {
                    'epochs': 20,
                    'batch_size': 128,
                    'lr': 1e-5,
                    'weight_decay': 0.001
                },
                'scheduler': {
                    'mode': 'min',
                    'factor': 0.5,
                    'patience': 3
                }
            }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def train(self, X, y, epochs=None, batch_size=None):
        cfg = self.config['training']
        epochs = epochs or cfg['epochs']
        batch_size = batch_size or cfg['batch_size']
        print(f"Training Transformer (LM Embeddings) on {len(y)} samples")

        #Ensure y is integer for bincount and loss
        y = np.asarray(y, dtype=int)

        # Compute class weights for imbalance
        class_counts = np.bincount(y)
        if len(class_counts) < 2:
            print("Warning: Only one class present — skipping class weighting")
            class_weights = torch.tensor([1.0, 1.0]).to(self.device)
        else:
            total_samples = len(y)
            class_weights = total_samples / (len(class_counts) * class_counts)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print(f"Class weights: {class_weights.tolist()}")

        barcodes = X.index if hasattr(X, 'index') else [f"ab_{i}" for i in range(len(y))]
        print(X.shape)
        print(y.shape)

        dataset = AntibodyDataset(X.values, y, barcodes)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        embedding_dim = X.shape[1]
        cfg_m = self.config['model']
        self.model = TransClassifier(
            embedding_dim=embedding_dim,
                    hidden_dim=cfg_m['hidden_dim'],
            num_layers=cfg_m['num_layers'],
            num_heads=cfg_m['num_heads'],
            dropout=cfg_m['dropout']
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        scheduler_cfg = self.config['scheduler']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg['mode'],
            factor=scheduler_cfg['factor'],
            patience=scheduler_cfg['patience']
        )   


        # original train_model
        model_save = f"{MODEL_DIR}/TransformerLM.pt"
        train_model(self.model, loader, None, criterion, optimizer, scheduler, epochs=epochs, patience=10, device=self.device, model_save=model_save)

        print("Training completed.")
        return self

    def predict_proba(self, X):
        self.model.eval()
        barcodes = X.index if hasattr(X, 'index') else [f"ab_{i}" for i in range(len(X))]
        dataset = AntibodyDataset(X.values, [0]*len(X), barcodes)
        loader = DataLoader(dataset, batch_size=128)

        probs = []
        with torch.no_grad():
            for embedding, _, _ in loader:
                embedding = embedding.to(self.device).float()
                out = self.model(embedding)
                prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                probs.extend(prob)
        return np.array(probs)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved → {path}")

    @classmethod

    def load(cls, path, embedding_dim, config_path="config/transformer_lm.yaml"):
        instance = cls(config_path)
        # Create model with provided embedding_dim and config params
        instance.model = TransClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=instance.config['model']['hidden_dim'],
            num_layers=instance.config['model']['num_layers'],
            num_heads=instance.config['model']['num_heads'],
            dropout=instance.config['model']['dropout']
        )
        # Load state_dict
        instance.model.load_state_dict(torch.load(path, map_location=instance.device))
        instance.model.eval()
        instance.model.to(instance.device)
        print(f"Model loaded from {path}")
        return instance

    @classmethod
    def kfold_validation_sgkf(cls, data, X, y, embedding_lm='ablang', title='PSR Prediction (Transformer)', kfold=10, target='psr_filter'):
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
        import matplotlib.pyplot as plt
        import numpy as np
        from torch.utils.data import DataLoader, Subset

        tprs = []
        aucs = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        mean_fpr = np.linspace(0, 1, 100)

        # Full dataset for labels and groups
        full_dataset = AntibodyDataset(X.values, y, data.index.tolist())
        labels = np.array([full_dataset[i][1] for i in range(len(full_dataset))])
        groups = data['HCDR3_CLUSTER_0.8'].values

        sgkf = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=42)
        plt.figure(figsize=(4, 4))

        print(f"\nTransformer (LM Embeddings) {kfold}-Fold CV | {title}")

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.arange(len(labels)), labels, groups)):
            print(f"\nTraining fold {fold+1}/{kfold}...")

            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            model = cls()
            
            train_loader = DataLoader(train_subset, batch_size=model.config['training']['batch_size'], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=model.config['training']['batch_size'], shuffle=False)

            # Class weights for this fold
            train_labels = np.array([full_dataset[i][1] for i in train_idx])
            class_counts = np.bincount(train_labels)
            total_samples = len(train_labels)
            class_weights = total_samples / (len(class_counts) * class_counts)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to('cpu')  # device not needed here
            print(f"Fold {fold+1} class weights: {class_weights.tolist()}, Class counts: {class_counts.tolist()}")

            # Model
       
            model.model = TransClassifier(embedding_dim=X.shape[1]).to('cpu')

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.AdamW(model.model.parameters(), lr=model.config['training']['lr'], weight_decay=model.config['training']['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=model.config['scheduler']['mode'], factor=model.config['scheduler']['factor'], patience=model.config['scheduler']['patience'])

            #original train_model call
            model_save = f"{MODEL_DIR}/TransformerLM_fold{fold+1}_best.pt"
            train_model(model.model, train_loader, val_loader, criterion, optimizer, scheduler,
                        epochs=model.config['training']['epochs'], patience=10, device='cpu', model_save=model_save)

            # Evaluation on val fold
            model.model.eval()
            probs = []
            preds = []
            true_labels = []
            with torch.no_grad():
                for embedding, label, barcode in val_loader:
                    output = model.model(embedding)
                    probs.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
                    preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                    true_labels.extend(label.cpu().numpy())

            if len(set(true_labels)) < 2 or len(true_labels) < 2:
                print(f"Skipping fold {fold+1} due to insufficient class diversity.")
                continue

            fpr, tpr, _ = roc_curve(true_labels, probs)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            accuracy = accuracy_score(true_labels, preds)
            f1 = f1_score(true_labels, preds, zero_division=0)
            precision = precision_score(true_labels, preds, zero_division=0)
            recall = recall_score(true_labels, preds, zero_division=0)
            accuracies.append(accuracy)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold+1} (AUC = {roc_auc:.3f})')

        if not aucs:
            print("No valid folds for metrics.")
            return None, None, None, None, None, None

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        mean_accuracy = np.mean(accuracies)
        mean_f1 = np.mean(f1_scores)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        std_tpr = np.std(tprs, axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2, label='±1 std. dev.')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.gca().set_xlabel('False Positive Rate', fontsize=9)
        plt.gca().set_ylabel('True Positive Rate', fontsize=9)
        plt.title(f'{title}\n10-Fold Cross-Validation ROC Curve\n'
                  f'Acc: {mean_accuracy:.3f}, F1: {mean_f1:.3f}, '
                  f'Prec: {mean_precision:.3f}, Rec: {mean_recall:.3f}', fontsize=8)
        plt.legend(loc='lower right', fontsize=5)
        plt.grid(True)
        plt.tight_layout()

        plot_path = f"{MODEL_DIR}/CV_ROC_TransformerLM_{target}_{embedding_lm}_k{kfold}.png"
        plt.savefig(plot_path)
        plt.close()

        print(f"\nMean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        print(f"Mean Accuracy: {mean_accuracy:.3f}")
        print(f"Mean F1-Score: {mean_f1:.3f}")
        print(f"Mean Precision: {mean_precision:.3f}")
        print(f"Mean Recall: {mean_recall:.3f}")

        return mean_auc, std_auc, mean_accuracy, mean_f1, mean_precision, mean_recall


    @classmethod
    def kfold_validation(cls, data, X, y, embedding_lm='ablang', title='PSR Prediction (Transformer)', kfold=10, target='psr_filter'):
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
        import matplotlib.pyplot as plt
        import numpy as np
        from torch.utils.data import DataLoader, Subset

        tprs = []
        aucs = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        mean_fpr = np.linspace(0, 1, 100)

        # Full dataset for labels and groups
        full_dataset = AntibodyDataset(X.values, y, data.index.tolist())
        labels = np.array([full_dataset[i][1] for i in range(len(full_dataset))])
        groups = data['HCDR3_CLUSTER_0.8'].values

        sgkf = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=42)
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)

        plt.figure(figsize=(4, 4))

        print(f"\nTransformer (LM Embeddings) {kfold}-Fold CV | {title}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, data['HCDR3_CLUSTER_0.8']), 1):

            model = cls()
            print(f"\nTraining fold {fold+1}/{kfold}...")

            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_test = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            y_train = y[train_idx]
            y_test = y[val_idx]
            print(X_train.index)
            # Create datasets
            train_dataset = AntibodyDataset(
                X_train.values, y_train, X_train.index.tolist()
            )
            test_dataset = AntibodyDataset(
                X_test.values, y_test, X_test.index.tolist()
            )

            train_loader = DataLoader(train_dataset, batch_size=model.config['training']['batch_size'], shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size=model.config['training']['batch_size'], shuffle=False)



    
            # Class weights for this fold
            train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
            class_counts = np.bincount(train_labels)
            print(class_counts)
            total_samples = len(train_labels)
            class_weights = total_samples / (len(class_counts) * class_counts)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to('cpu')  # device not needed here
            print(f"Fold {fold+1} class weights: {class_weights.tolist()}, Class counts: {class_counts.tolist()}")

            # Model
       
            model.model = TransClassifier(embedding_dim=X.shape[1]).to('cpu')

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.AdamW(model.model.parameters(), lr=model.config['training']['lr'], weight_decay=model.config['training']['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=model.config['scheduler']['mode'], factor=model.config['scheduler']['factor'], patience=model.config['scheduler']['patience'])

            # train_model call
            model_save = f"{MODEL_DIR}/TransformerLM_fold{fold+1}_best.pt"
            train_model(model.model, train_loader, val_loader, criterion, optimizer, scheduler,
                        epochs=model.config['training']['epochs'], patience=10, device='cpu', model_save=model_save)

            # Evaluation on val fold
            model.model.eval()
            probs = []
            preds = []
            true_labels = []
            with torch.no_grad():
                for embedding, label, barcode in val_loader:
                    output = model.model(embedding)
                    probs.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
                    preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                    true_labels.extend(label.cpu().numpy())

            if len(set(true_labels)) < 2 or len(true_labels) < 2:
                print(f"Skipping fold {fold+1} due to insufficient class diversity.")
                continue

            fpr, tpr, _ = roc_curve(true_labels, probs)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            accuracy = accuracy_score(true_labels, preds)
            f1 = f1_score(true_labels, preds, zero_division=0)
            precision = precision_score(true_labels, preds, zero_division=0)
            recall = recall_score(true_labels, preds, zero_division=0)
            accuracies.append(accuracy)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold+1} (AUC = {roc_auc:.3f})')

        if not aucs:
            print("No valid folds for metrics.")
            return None, None, None, None, None, None

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        mean_accuracy = np.mean(accuracies)
        mean_f1 = np.mean(f1_scores)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        std_tpr = np.std(tprs, axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2, label='±1 std. dev.')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.gca().set_xlabel('False Positive Rate', fontsize=9)
        plt.gca().set_ylabel('True Positive Rate', fontsize=9)
        plt.title(f'{title}\n10-Fold Cross-Validation ROC Curve\n'
                  f'Acc: {mean_accuracy:.3f}, F1: {mean_f1:.3f}, '
                  f'Prec: {mean_precision:.3f}, Rec: {mean_recall:.3f}', fontsize=8)
        plt.legend(loc='lower right', fontsize=5)
        plt.grid(True)
        plt.tight_layout()

        plot_path = f"{MODEL_DIR}/CV_ROC_TransformerLM_{target}_{embedding_lm}_k{kfold}.png"
        plt.savefig(plot_path)
        plt.close()

        print(f"\nMean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        print(f"Mean Accuracy: {mean_accuracy:.3f}")
        print(f"Mean F1-Score: {mean_f1:.3f}")
        print(f"Mean Precision: {mean_precision:.3f}")
        print(f"Mean Recall: {mean_recall:.3f}")

        return mean_auc, std_auc, mean_accuracy, mean_f1, mean_precision, mean_recall
