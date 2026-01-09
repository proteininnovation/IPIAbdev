# models/transformer_onehot.py
# Transformer with One-Hot Encoding + HCDR3 Attention (your original architecture)
# Includes single-antibody analysis, heatmap, mutagenesis, and IG
# IPI Antibody Developability Prediction Platform
# Created by Hoan Nguyen | Final Production Version — DEC-2025

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import Dataset, DataLoader
from captum.attr import IntegratedGradients
from config import MODEL_DIR
from matplotlib.patches import Patch

# Amino acid alphabet 
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

# One-hot encoding function 
def one_hot_encode_sequence(sequence, max_length):
    sequence = sequence.replace('-', '')
    encoding = np.zeros((max_length, len(AMINO_ACIDS)))
    for i, aa in enumerate(sequence[:max_length]):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1
    return encoding.flatten()

# Create AntibodyDataset
class AntibodyDataset(Dataset):
    def __init__(self, heavy_seqs, light_seqs, hcdr3_seqs, labels, barcode,
                 max_heavy_len=135, max_light_len=135, max_hcdr3_len=25):
        self.heavy_seqs = heavy_seqs
        self.light_seqs = light_seqs
        self.hcdr3_seqs = hcdr3_seqs
        self.labels = labels
        self.barcode = barcode
        self.max_heavy_len = max_heavy_len
        self.max_light_len = max_light_len
        self.max_hcdr3_len = max_hcdr3_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        h_seq = self.heavy_seqs[idx]
        l_seq = self.light_seqs[idx]
        hcdr3 = self.hcdr3_seqs[idx]
        label = self.labels[idx]
        barcode = self.barcode[idx]

        encoding_heavy = torch.tensor(one_hot_encode_sequence(h_seq, self.max_heavy_len)).float()
        encoding_light = torch.tensor(one_hot_encode_sequence(l_seq, self.max_light_len)).float()
        encoding_cdr3 = torch.tensor(one_hot_encode_sequence(hcdr3, self.max_hcdr3_len)).float()

        encoding = torch.cat([encoding_heavy, encoding_light], dim=0)
        return encoding, encoding_cdr3, label, barcode, h_seq, l_seq, hcdr3

# Transformer-onehot model with HCDR3 attention
class SEC_Classifier(nn.Module):
    def __init__(self, encoding_dim, cdr3_dim, config):
        super().__init__()
        cfg = config['model']
        hidden_dim = cfg['hidden_dim']
        num_layers = cfg['num_layers']
        nhead = cfg['num_heads']
        dim_feedforward = cfg['dim_feedforward']
        dropout = cfg['dropout']

        self.input_fc = nn.Linear(encoding_dim, hidden_dim)
        self.cdr3_fc = nn.Linear(cdr3_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cdr3_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention = nn.MultiheadAttention(hidden_dim, nhead)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, encoding, encoding_cdr3):
        x = self.input_fc(encoding.unsqueeze(1))
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        cdr3_x = self.cdr3_fc(encoding_cdr3.unsqueeze(1))
        cdr3_x = self.cdr3_transformer(cdr3_x)
        cdr3_x = cdr3_x.mean(dim=1)
        
        attn_output, _ = self.attention(cdr3_x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        attn_output = attn_output.squeeze(0)
        
        combined = torch.cat((x, attn_output), dim=1)
        out = self.fc(combined)
        return out

# Framework-compatible wrapper class
class TransformerOneHotModel:
    def __init__(self, config_path="config/transformer_onehot.yaml"):
        import yaml
        
        # Load config from YAML or use defaults
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            print(f"Config loaded from {config_path}")
        else:
            print(f"Warning: Config file {config_path} not found — using defaults")
            self.config = {
                'model': {
                    'hidden_dim': 128,
                    'num_heads': 8,
                    'num_layers': 6,
                    'dim_feedforward': 512,
                    'dropout': 0.2
                },
                'sequence_lengths': {
                    'max_vh_len': 135,
                    'max_vl_len': 135,
                    'max_hcdr3_len': 25
                },
                'training': {
                    'epochs': 20,
                    'batch_size': 16,
                    'lr': 0.0001,
                    'weight_decay': 1e-5,
                    'val_split': 0.2
                },
                'mutagenesis': {
                    'amino_acids': "ACDEFGHIKLMNPQRSTVWY"
                }
            }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # Compute dimensions from config (for compatibility)
        cfg = self.config['sequence_lengths']
        self.max_heavy_len = cfg['max_vh_len']
        self.max_light_len = cfg['max_vl_len']
        self.max_hcdr3_len = cfg['max_hcdr3_len']
        self.encoding_dim = (self.max_heavy_len + self.max_light_len) * len(AMINO_ACIDS)
        self.cdr3_dim = self.max_hcdr3_len * len(AMINO_ACIDS)



    def train(self, X, y, epochs=20, batch_size=16):
        print(f"Training Transformer (One-Hot + HCDR3 Attention) on {len(y)} samples")

        # Compute class weights for imbalance
        pos_weight = (len(y) - np.sum(y)) / np.sum(y)  # negative / positive
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(self.device)
        print(f"Class imbalance detected: Positive rate {np.mean(y):.1%}")
        print(f"Using class weighting (pos_weight = {pos_weight.item():.2f})")

        heavy = X['HSEQ'].values
        light = X['LSEQ'].values
        hcdr3 = X['CDR3'].values
        barcodes = X['BARCODE'].values if 'BARCODE' in X else [f"ab_{i}" for i in range(len(y))]

        dataset = AntibodyDataset(heavy, light, hcdr3, y, barcodes,
                                  self.max_heavy_len, self.max_light_len, self.max_hcdr3_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Pass config to SEC_Classifier
        self.model = SEC_Classifier(
            encoding_dim=self.encoding_dim,
            cdr3_dim=self.cdr3_dim,
            config=self.config
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))  # [negative, positive]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for encoding, encoding_cdr3, label, _, _, _, _ in loader:
                encoding, encoding_cdr3, label = encoding.to(self.device), encoding_cdr3.to(self.device), label.long().to(self.device)
                optimizer.zero_grad()
                out = self.model(encoding, encoding_cdr3)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

        print("Training completed with class weighting.")
        return self

    @classmethod
    def kfold_validation(cls, data, X, y, embedding_lm='onehot', title="TransformerOneHot", kfold=10, target_name="sec_filter"):
        
        #print(data)
        data['BARCODE'] = data.index.tolist()
        
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
        import matplotlib.pyplot as plt
        import numpy as np
        from torch.utils.data import Subset, DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
        tprs = []
        aucs = []
        fold_metrics = []

        # Load config
        #model = cls()  # Loads config in __init__


        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(figsize=(8, 7))

        print(f"\nTransformer (One-Hot) {kfold}-Fold CV | {title} | Target: {target_name.upper()}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, data['HCDR3_CLUSTER_0.8']), 1):
            print(f"\n--- Fold {fold}/{kfold} ---")

            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            print(X_train.index)
            # Create datasets
            train_dataset = AntibodyDataset(
                X_train['HSEQ'].values, X_train['LSEQ'].values, X_train['CDR3'].values,
                y_train, X_train.index.values
            )
            test_dataset = AntibodyDataset(
                X_test['HSEQ'].values, X_test['LSEQ'].values, X_test['CDR3'].values,
                y_test, X_test.index.values
            )

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size=16)

            # Train model
            model = cls()

            model.model = SEC_Classifier(
                encoding_dim=model.encoding_dim,
                cdr3_dim=model.cdr3_dim,
                config=model.config
            ).to(device)

            #model.model = SEC_Classifier(encoding_dim=model.encoding_dim, cdr3_dim=model.cdr3_dim).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.model.parameters(), lr=0.0001, weight_decay=1e-5)

            best_val_loss = float('inf')
            best_state = None

            for epoch in range(20):  # original 20 epochs
                model.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                for encoding, cdr3_enc, label, _, _, _, _ in train_loader:
                    encoding, cdr3_enc, label = encoding.to(device), cdr3_enc.to(device), label.long().to(device)
                    optimizer.zero_grad()
                    out = model.model(encoding, cdr3_enc)
                    loss = criterion(out, label)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, pred = torch.max(out, 1)
                    train_correct += (pred == label).sum().item()
                    train_total += label.size(0)

                train_acc = train_correct / train_total
                train_loss /= len(train_loader)

                # Validation
                model.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for encoding, cdr3_enc, label, _, _, _, _ in val_loader:
                        encoding, cdr3_enc, label = encoding.to(device), cdr3_enc.to(device), label.long().to(device)
                        out = model.model(encoding, cdr3_enc)
                        loss = criterion(out, label)
                        val_loss += loss.item()
                        _, pred = torch.max(out, 1)
                        val_correct += (pred == label).sum().item()
                        val_total += label.size(0)

                val_acc = val_correct / val_total
                val_loss /= len(val_loader)

                if epoch % 5 == 0 or epoch == 19:
                    print(f"  Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.model.state_dict()

            # Load best model
            model.model.load_state_dict(best_state)

            # Test evaluation
            model.model.eval()
            probs = []
            preds = []
            true = []
            with torch.no_grad():
                for encoding, cdr3_enc, label, _, _, _, _ in val_loader:
                    encoding, cdr3_enc = encoding.to(device), cdr3_enc.to(device)
                    out = model.model(encoding, cdr3_enc)
                    prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    probs.extend(prob)
                    preds.extend((prob >= 0.5).astype(int))
                    true.extend(label.numpy())

            fold_auc = roc_auc_score(true, probs)
            fold_acc = accuracy_score(true, preds)
            fold_f1 = f1_score(true, preds, zero_division=0)
            fold_prec = precision_score(true, preds, zero_division=0)
            fold_rec = recall_score(true, preds, zero_division=0)

            print(f"  Fold {fold} Results:")
            print(f"    AUC: {fold_auc:.4f} | Acc: {fold_acc:.4f} | F1: {fold_f1:.4f} | Prec: {fold_prec:.4f} | Rec: {fold_rec:.4f}")

            fold_metrics.append({
                'auc': fold_auc, 'acc': fold_acc, 'f1': fold_f1,
                'precision': fold_prec, 'recall': fold_rec
            })

            aucs.append(fold_auc)
            fpr, tpr, _ = roc_curve(true, probs)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            plt.plot(fpr, tpr, alpha=0.3, lw=1, label=f'Fold {fold} (AUC={fold_auc:.3f})')

            # Save fold model with target name
            model.save(f"{MODEL_DIR}/TransformerOneHot_{target_name}_{embedding_lm}_fold{fold}_k{kfold}.pt")

        # Final summary
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_acc = np.mean([m['acc'] for m in fold_metrics])
        mean_f1 = np.mean([m['f1'] for m in fold_metrics])
        mean_prec = np.mean([m['precision'] for m in fold_metrics])
        mean_rec = np.mean([m['recall'] for m in fold_metrics])

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        plt.plot(mean_fpr, mean_tpr, 'b', lw=3,
                 label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        plt.fill_between(mean_fpr, mean_tpr - np.std(tprs, axis=0), mean_tpr + np.std(tprs, axis=0),
                         color='lightblue', alpha=0.3)
        plt.plot([0,1],[0,1], '--', color='gray')
        plt.xlim([0,1]); plt.ylim([0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title} - {target_name.upper()}\n{kfold}-Fold Cross-Validation ROC Curve\n'
                  f'Acc: {mean_acc:.3f}, F1: {mean_f1:.3f}, Prec: {mean_prec:.3f}, Rec: {mean_rec:.3f}')
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(alpha=0.3)

        plot_path = f"{MODEL_DIR}/CV_ROC_TransformerOneHot_{target_name}_{embedding_lm}_k{kfold}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close
        #plt.show()

        print(f"\nFINAL RESULT ({kfold}-Fold CV) - Target: {target_name.upper()}")
        for fold in range(1, kfold + 1):
            m = fold_metrics[fold-1]
            print(f"  Fold {fold}: AUC {m['auc']:.4f} | Acc {m['acc']:.4f} | F1 {m['f1']:.4f} | Prec {m['precision']:.4f} | Rec {m['recall']:.4f}")
        print(f"\nMean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Mean Accuracy: {mean_acc:.4f}")
        print(f"Mean F1: {mean_f1:.4f}")
        print(f"Mean Precision: {mean_prec:.4f}")
        print(f"Mean Recall: {mean_rec:.4f}")
        print(f"Plot saved → {plot_path}")


    def predict_proba(self, X):
        self.model.eval()
        heavy = X['HSEQ'].values if 'HSEQ' in X else X
        light = X['LSEQ'].values if 'LSEQ' in X else [''] * len(heavy)
        hcdr3 = X['CDR3'].values if 'CDR3' in X else [''] * len(heavy)

        dataset = AntibodyDataset(heavy, light, hcdr3, [0]*len(heavy), ['temp']*len(heavy),
                                  self.max_heavy_len, self.max_light_len, self.max_hcdr3_len)
        loader = DataLoader(dataset, batch_size=64)

        probs = []
        with torch.no_grad():
            for encoding, cdr3_enc, _, _, _, _, _ in loader:
                out = self.model(encoding.to(self.device), cdr3_enc.to(self.device))
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
    def load_old(cls, path):
        instance = cls()
        instance.model = SEC_Classifier(encoding_dim=instance.encoding_dim, cdr3_dim=instance.cdr3_dim)
        instance.model.load_state_dict(torch.load(path, map_location=instance.device))
        instance.model.eval()
        return instance
    


    @classmethod
    def load_bk2(cls, path, config_path="config/transformer_onehot.yaml"):
        instance = cls(config_path)  # Loads config
        
        cfg = instance.config['sequence_lengths']
        encoding_dim = (cfg['max_vh_len'] + cfg['max_vl_len']) * len(AMINO_ACIDS)
        cdr3_dim = cfg['max_hcdr3_len'] * len(AMINO_ACIDS)
        
        # argument names
        model_config = instance.config['model']
        instance.model = SEC_Classifier(
            encoding_dim=encoding_dim,
            cdr3_dim=cdr3_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout']
        )
        
        instance.model.load_state_dict(torch.load(path, map_location=instance.device))
        instance.model.eval()
        instance.model.to(instance.device)
        print(f"Model loaded from {path}")
        return instance
    
    @classmethod
    def load(cls, path, config_path="config/transformer_onehot.yaml"):
        instance = cls(config_path)
        cfg = instance.config['sequence_lengths']
        encoding_dim = (cfg['max_vh_len'] + cfg['max_vl_len']) * len(AMINO_ACIDS)
        cdr3_dim = cfg['max_hcdr3_len'] * len(AMINO_ACIDS)

        instance.model = SEC_Classifier(encoding_dim, cdr3_dim, instance.config)

        # load full state dict including optimizer if needed
        #state_dict = torch.load(path, map_location=instance.device, weights_only=False)
        #instance.model.load_state_dict(state_dict)
        
        instance.model.load_state_dict(torch.load(path, map_location=instance.device))
        
        instance.model.eval()
        instance.model.to(instance.device)
        print(f"Model loaded from {path}")
        return instance

    # predict single antibody
    def predict_single(self, barcode, VH, VL, HCDR3):
        encoding_heavy = torch.tensor(one_hot_encode_sequence(VH, self.max_heavy_len)).float().unsqueeze(0).to(self.device)
        encoding_light = torch.tensor(one_hot_encode_sequence(VL, self.max_light_len)).float().unsqueeze(0).to(self.device)
        encoding_cdr3 = torch.tensor(one_hot_encode_sequence(HCDR3, self.max_hcdr3_len)).float().unsqueeze(0).to(self.device)
        encoding = torch.cat([encoding_heavy, encoding_light], dim=1)
        
        with torch.no_grad():
            output = self.model(encoding, encoding_cdr3)
            probability = torch.softmax(output, dim=1)[0][1].item()
        return probability

    # original plot_heatmap_for_sample (renamed for clarity)
    #def plot_cdr3_mutation_heatmap(self, barcode, dataset, output_path="SEC/image/sec_cdr3_mutation_heatmap.png"):
    def plot_cdr3_mutation_heatmap(self, vh_seq,vl_seq,hcdr3,barcode,label,target_analysis="PSR", output_path="SEC/image/sec_cdr3_mutation_heatmap.png"):
       
        #for i, (_, _, _, b, _, _, _) in enumerate(dataset):
        #    if b == barcode:
        #        sample_idx = i
        #        break
        #else:
        #    raise ValueError(f"Barcode {barcode} not found.")
        #_, _, label, _, VH, VL, HCDR3 = dataset[sample_idx]
        VH=vh_seq
        VL=vl_seq
        HCDR3=hcdr3
        barcode=barcode
        label=label

 
        input_data = [barcode, VH, VL, HCDR3]

        prob = self.predict_single(barcode, VH, VL, HCDR3)


        mutation_dict = self._generate_cdr3_mutations(HCDR3)
        for pos in mutation_dict:
            for aa in mutation_dict[pos]:
                mutant_cdr3 = HCDR3[:pos-1] + aa + HCDR3[pos:]
                VH_mutant = VH.replace(HCDR3, mutant_cdr3)
                mutation_dict[pos][aa] = self.predict_single(barcode, VH_mutant, VL, mutant_cdr3)

        heatmap_data, amino_acids, positions = self._create_heatmap_data(mutation_dict, HCDR3)

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", xticklabels=positions, yticklabels=list(amino_acids))
        plt.xlabel("Original CDR3 Amino Acid")
        plt.ylabel("Mutant Amino Acid")
        plt.title(f"{target_analysis} Prediction: Probability for CDR3 mutations \n ID :{barcode} - Original label:{label} Original predicted probability: {prob:.3f}")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.show()
        plt.close()

    def _generate_cdr3_mutations(self, original_cdr3):
        mutations = {}
        for i, aa in enumerate(original_cdr3):
            for new_aa in AMINO_ACIDS:
              #if new_aa != aa:
                pos = i + 1
                mutations.setdefault(pos, {})[new_aa] = None
        return mutations

    def _create_heatmap_data(self, mutation_dict,original_cdr3 ):
        cdr3_length=len(original_cdr3)
        heatmap_data = np.zeros((len(AMINO_ACIDS), cdr3_length))
        for pos in range(1, cdr3_length + 1):
            if pos in mutation_dict:
                for i, aa in enumerate(AMINO_ACIDS):
                    if aa in mutation_dict[pos]:
                        heatmap_data[i, pos-1] = mutation_dict[pos][aa]
        return heatmap_data, AMINO_ACIDS, list(original_cdr3) #list(range(1, cdr3_length + 1))

    # Integrated Gradients for single antibody (original logic)
    def integrated_gradients_single_bk(self, barcode, dataset, n_steps=50):
        for i, (_, _, _, b, _, _, _) in enumerate(dataset):
            if b == barcode:
                sample_idx = i
                break
        else:
            raise ValueError(f"Barcode {barcode} not found.")

        encoding, encoding_cdr3, _, _, _, _, _ = dataset[sample_idx]
        combined = torch.cat([encoding, encoding_cdr3], dim=0).unsqueeze(0).to(self.device).requires_grad_(True)
        baseline = torch.zeros_like(combined)

        ig = IntegratedGradients(self.model)
        attributions = ig.attribute(combined, baseline, target=1, n_steps=n_steps)
        return attributions.squeeze(0).cpu().numpy()

    def integrated_gradients_single(self, vh_seq, vl_seq, hcdr3_seq, n_steps=50):
        """Integrated Gradients for a single antibody using direct sequences (no dataset needed)"""
        self.model.eval()
        
        cfg = self.config['sequence_lengths']
        max_vh = cfg['max_vh_len']
        max_vl = cfg['max_vl_len']
        max_hcdr3 = cfg['max_hcdr3_len']

        # Helper to return flattened one-hot ( original style — 1D)
        def one_hot_flat(seq, max_len):
            seq = seq.replace('-', '').upper()
            enc = np.zeros(max_len * 20)
            for i, aa in enumerate(seq[:max_len]):
                if aa in AMINO_ACIDS:
                    enc[i * 20 + AMINO_ACIDS.index(aa)] = 1
            return enc

        # Encode as flattened (1D) — matches your original training input
        h_enc = one_hot_flat(vh_seq, max_vh)  # length: max_vh * 20
        l_enc = one_hot_flat(vl_seq, max_vl)  # length: max_vl * 20
        cdr3_enc = one_hot_flat(hcdr3_seq, max_hcdr3)  # length: max_hcdr3 * 20

        # Concatenate VH + VL
        encoding = np.concatenate([h_enc, l_enc])  # length: (max_vh + max_vl) * 20 = 5400
        encoding = torch.tensor(encoding, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 5400)

        cdr3_enc = torch.tensor(cdr3_enc, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 500)

        # Baseline
        baseline_encoding = torch.zeros_like(encoding)
        baseline_cdr3 = torch.zeros_like(cdr3_enc)

        # Integrated Gradients on tuple input
        ig = IntegratedGradients(self.model)
        attributions = ig.attribute(
            (encoding, cdr3_enc),
            baselines=(baseline_encoding, baseline_cdr3),
            target=1,
            n_steps=n_steps
        )

        # Sum over amino acids to get per-position importance
        attr_encoding = attributions[0].squeeze(0).cpu().numpy()  # (5400,)
        attr_cdr3 = attributions[1].squeeze(0).cpu().numpy()     # (500,)

        # Reshape to per-position
        attr_encoding = attr_encoding.reshape(max_vh + max_vl, 20).sum(axis=1)  # (270,)
        attr_cdr3 = attr_cdr3.reshape(max_hcdr3, 20).sum(axis=1)               # (25,)

        # Concatenate VH + VL + HCDR3 importance
        importance = np.concatenate([attr_encoding, attr_cdr3])

        return importance


    # Full-dataset IG analysis ( original global logic)
    def global_ig_analysis(self, dataset, output_prefix="global_ig"):
        print("\nRunning global Integrated Gradients analysis on full dataset...")
        self.model.eval()

        class ModelWrapperIG(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inputs):
                encoding = inputs[:, :self.model.encoding_dim]
                cdr3 = inputs[:, self.model.encoding_dim:]
                return self.model(encoding, cdr3)[:, 1]

        wrapper = ModelWrapperIG(self.model).to(self.device)
        ig = IntegratedGradients(wrapper)

        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        all_attributions = []

        for encoding, cdr3_enc, _, _, _, _, _ in loader:
            combined = torch.cat([encoding, cdr3_enc], dim=1).to(self.device).requires_grad_(True)
            baseline = torch.zeros_like(combined)

            attr = ig.attribute(combined, baseline, target=0, n_steps=50)
            all_attributions.append(attr.detach().cpu().numpy())

        attributions = np.concatenate(all_attributions, axis=0)

        heavy_seqs = [dataset[i][4] for i in range(len(dataset))]
        hcdr3_seqs = [dataset[i][6] for i in range(len(dataset))]

        top_features, mean_attr, _, top_residues = self._aggregate_feature_importance_ig(
            attributions, heavy_seqs, hcdr3_seqs
        )

        self._plot_global_importance(mean_attr, top_features, output_prefix)
        self._plot_hcdr3_residue_heatmap(attributions, heavy_seqs, hcdr3_seqs, output_prefix)

        print(f"Global IG analysis completed. Results saved with prefix: {output_prefix}")
        return top_residues



    def _aggregate_feature_importance_ig(self, attributions, heavy_seqs, hcdr3_seqs):
        # Compute mean attributions for plotting (with signs)
        mean_attr = np.mean(attributions, axis=0)
        # Compute mean absolute attributions for ranking top features
        mean_abs_attr = np.mean(np.abs(attributions), axis=0)
        encoding_dim = (self.max_heavy_len + self.max_light_len + self.max_hcdr3_len) * len(AMINO_ACIDS)
        
        # Select top features based on absolute importance
        max_feature = 60  # or make it a parameter
        top_features = np.argsort(mean_abs_attr)[::-1][:max_feature]
        top_attributions = mean_attr[top_features]  # Use signed values for plotting

        # Plot feature importance (waterfall)
        output_path = f"{main_path}/SEC/images/{prefix}_feature_importance_onehot_hl.png"
        plt.figure(figsize=(10, 8))
        colors = ['blue' if val >= 0 else 'red' for val in top_attributions]
        plt.barh(range(len(top_features)), top_attributions, color=colors)
        feature_names = ([f"H_{i}_{aa}" for i in range(self.max_heavy_len) for aa in AMINO_ACIDS] +
                         [f"L_{i}_{aa}" for i in range(self.max_light_len) for aa in AMINO_ACIDS] +
                         [f"HCDR3_{i}_{aa}" for i in range(self.max_hcdr3_len) for aa in AMINO_ACIDS])
        plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features], fontsize=8)
        plt.xlabel("Mean Integrated Gradients Attribution", fontsize=10)
        plt.ylabel("SEC-Transformers: VH+VL+HCDR3 OneHot Encodings", fontsize=10)
        plt.title(f"Top {max_feature} Feature Importance for Entire IPI SEC Dataset (Integrated Gradients)", fontsize=12)
        plt.legend(handles=[Patch(facecolor='blue'), Patch(facecolor='red')],
                   labels=['Positive Contribution', 'Negative Contribution'], loc='best', fontsize=8)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.show()
        plt.close()

        # Compute residue-level importance for HCDR3
        residue_importance_h = []
        total_encoding_dim = (self.max_heavy_len + self.max_light_len) * len(AMINO_ACIDS)

        for attr, h_seq, hcdr3 in zip(attributions, heavy_seqs, hcdr3_seqs):
            hcdr3_start = h_seq.find(hcdr3) if hcdr3 and h_seq else -1
            h_importance = np.zeros(self.max_heavy_len)
            if hcdr3_start >= 0:
                attr_mean = np.abs(attr)
                heavy_attr = attr_mean[:self.max_heavy_len * len(AMINO_ACIDS)].reshape(self.max_heavy_len, len(AMINO_ACIDS)).sum(axis=1)
                hcdr3_attr = attr_mean[total_encoding_dim:total_encoding_dim + self.max_hcdr3_len * len(AMINO_ACIDS)].reshape(self.max_hcdr3_len, len(AMINO_ACIDS)).sum(axis=1)
                h_importance[:len(h_seq)] = heavy_attr[:len(h_seq)]
                if len(hcdr3) <= len(hcdr3_attr):
                    h_importance[hcdr3_start:hcdr3_start + len(hcdr3)] += hcdr3_attr[:len(hcdr3)]
            residue_importance_h.append(h_importance)

        # Plot HCDR3 residue importance heatmap
        residue_heatmap_path = f"{main_path}/SEC/images/{prefix}_hcdr3_residue_importance_heatmap_onehot_hl.png"
        max_hcdr3_len = max(len(seq) for seq in hcdr3_seqs if seq) if any(hcdr3_seqs) else 1
        hcdr3_data = np.zeros((1, max_hcdr3_len))
        for i, seq in enumerate(hcdr3_seqs):
            if seq:
                start = heavy_seqs[i].find(seq) if heavy_seqs[i] else -1
                if start >= 0 and start < len(residue_importance_h[i]):
                    end = min(start + len(seq), len(residue_importance_h[i]))
                    hcdr3_data[0, :end-start] += residue_importance_h[i][start:end]
        hcdr3_data /= max(1, sum(1 for seq in hcdr3_seqs if seq))

        plt.figure(figsize=(10, 2))
        sns.heatmap(hcdr3_data, cmap='viridis', xticklabels=range(1, max_hcdr3_len + 1), yticklabels=['CDR3'])
        plt.title("HCDR3 Residue Importance (Integrated Gradients)")
        plt.xlabel("HCDR3 Position")
        plt.ylabel("Region")
        plt.tight_layout()
        plt.savefig(residue_heatmap_path)
        plt.show()
        plt.close()

        top_hcdr3_pos = np.argsort(np.sum(hcdr3_data[0], axis=0))[-5:][::-1]
        top_residues = [
            {'Chain': 'CDR3', 'Position': pos + 1, 'Importance': hcdr3_data[0, pos]} for pos in top_hcdr3_pos
        ]

        return top_features, top_attributions, [], top_residues

    def _plot_global_importance(self, mean_attr, top_features, prefix):
        #  original waterfall/top feature plot
        output_path = f"{main_path}/SEC/images/{prefix}_feature_importance_onehot_hl.png"
        max_feature = len(top_features)
        plt.figure(figsize=(10, 8))
        colors = ['blue' if val >= 0 else 'red' for val in mean_attr[top_features]]
        plt.barh(range(len(top_features)), mean_attr[top_features], color=colors)
        feature_names = ([f"H_{i}_{aa}" for i in range(self.max_heavy_len) for aa in AMINO_ACIDS] +
                         [f"L_{i}_{aa}" for i in range(self.max_light_len) for aa in AMINO_ACIDS] +
                         [f"HCDR3_{i}_{aa}" for i in range(self.max_hcdr3_len) for aa in AMINO_ACIDS])
        plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features], fontsize=8)
        plt.xlabel("Mean Integrated Gradients Attribution", fontsize=10)
        plt.ylabel("SEC-Transformers: VH+VL+HCDR3 OneHot Encodings", fontsize=10)
        plt.title(f"Top {max_feature} Feature Importance for Entire IPI SEC Dataset (Integrated Gradients)", fontsize=12)
        plt.legend(handles=[Patch(facecolor='blue'), Patch(facecolor='red')],
                   labels=['Positive Contribution', 'Negative Contribution'], loc='best', fontsize=8)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.show()
        plt.close()

    def _plot_hcdr3_residue_heatmap(self, attributions, heavy_seqs, hcdr3_seqs, prefix):
        #  original HCDR3 residue importance heatmap
        residue_heatmap_path = f"{main_path}/SEC/images/{prefix}_hcdr3_residue_importance_heatmap_onehot_hl.png"
        max_hcdr3_len = max(len(seq) for seq in hcdr3_seqs if seq) if any(hcdr3_seqs) else 1
        hcdr3_data = np.zeros((1, max_hcdr3_len))
        for i, seq in enumerate(hcdr3_seqs):
            if seq:
                start = heavy_seqs[i].find(seq) if heavy_seqs[i] else -1
                if start >= 0:
                    end = min(start + len(seq), max_hcdr3_len)
                    # Use absolute attribution for importance
                    attr = np.abs(attributions[i])
                    heavy_attr = attr[:self.max_heavy_len * len(AMINO_ACIDS)].reshape(self.max_heavy_len, len(AMINO_ACIDS)).sum(axis=1)
                    hcdr3_data[0, :end-start] += heavy_attr[start:end]
        hcdr3_data /= max(1, sum(1 for seq in hcdr3_seqs if seq))

        plt.figure(figsize=(10, 2))
        sns.heatmap(hcdr3_data, cmap='viridis', xticklabels=range(1, max_hcdr3_len + 1), yticklabels=['CDR3'])
        plt.title("HCDR3 Residue Importance (Integrated Gradients)")
        plt.xlabel("HCDR3 Position")
        plt.ylabel("Region")
        plt.tight_layout()
        plt.savefig(residue_heatmap_path)
        plt.show()
        plt.close()