# IPI-Institute For Protein Innovation
# SEC Prediction and Feature Contribution Interpretation
# AI-ML Architecture : 
#       Model: Transformer, epoch=20 
#       Feature importance discovery: Shap Gradient and Kernel Explainer:  
#
# IPI Trainning set: SPR :~5200 antibody 
#       Feature usages: Heavy, Light, HCDR3 one hot encoding 
#       Enchance sensitivity for HCDR3
#
# Training and Validation Set Splitting (80%-20%)
#       CDR3 clustering by greedy Levenshtein distance
#       Ensuring that there are no overlapping sequences between the two sets based on their cluster membership,
#       By placing all sequences from a given cluster into either the training or validation set,

# Mutational HCDR3 and heatmap predictive probability
# Shape Explainner 
# Module requirements: python, pytorch, shap, Biopython, seaborn, sklearn,transformers
# Designer: Hoan Nguyen, IPI Antibody Platform
# Latest update: July-2-2025 
from captum.attr import IntegratedGradients
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Subset
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
import sys
import itertools
import random
# Set the seed
random.seed(42)
np.random.seed(42)


sys.path.append("/Users/Hoan.Nguyen/ComBio/AbodyDiscoveryPipeline/")
main_path = "/Users/Hoan.Nguyen/ComBio/MachineLearning/"
os.chdir(main_path)




# Set the seed
random.seed(42)
np.random.seed(42)

# Define amino acid alphabet for one-hot encoding
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

# One-hot encoding function (unchanged)
def one_hot_encode_sequence(sequence, max_length):
    sequence = sequence.replace('-', '')
    encoding = np.zeros((max_length, len(AMINO_ACIDS)))
    for i, aa in enumerate(sequence[:max_length]):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1
    return encoding.flatten()

# Function to compute biophysical properties
def compute_biophysical_properties(sequence):
    analyzed_seq = ProteinAnalysis(sequence.replace('-', ''))  # Remove padding
    # Net charge (approximate, using pH 7.0)
    charge = analyzed_seq.charge_at_pH(7.0)
    # Hydrophobicity (Kyte-Doolittle scale, average over sequence)
    hydrophobicity = np.mean([ProteinAnalysis(aa).gravy() if aa != '-' else 0 for aa in sequence])
    return np.array([charge, hydrophobicity])

# Load dataset and generate one-hot encodings
def load_ipi_psr_dataset(max_heavy_len=150, max_light_len=150):
    module_dir = os.path.abspath('/Users/Hoan.Nguyen/ComBio/AbodyDiscoveryPipeline')
    sys.path.append(module_dir)
    import liabilities

    data = pd.read_excel("data/ipi_antibodydb_august2025.xlsx")
    data.loc[data['CDR3'].str.startswith('C'), 'CDR3'] = data['CDR3'].str[1:]
    data = liabilities.annotate_liabilities_2(data, cdr3_col='CDR3', label='HCDR3')
    data = data.dropna(subset=['HSEQ', 'LSEQ'])


    data = data[pd.notna(data['sec_filter'])]
    #data = data[data['sec_filter'] == data['psr_rf_elisa']]
    data = data[~data['antigen'].str.contains('test', na=False, case=False)]


    train_data, val_data = train_test_split_by_hcdr3_cluster(data, test_size=0.2)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_data['sec_filter'])
    y_test = encoder.transform(val_data['sec_filter'])

    train_heavy_encodings = np.array([one_hot_encode_sequence(seq, max_heavy_len) for seq in data.loc[train_data.index]['HSEQ']])
    train_light_encodings = np.array([one_hot_encode_sequence(seq, max_light_len) for seq in data.loc[train_data.index]['LSEQ']])

    test_heavy_encodings = np.array([one_hot_encode_sequence(seq, max_heavy_len) for seq in data.loc[val_data.index]['HSEQ']])
    test_light_encodings = np.array([one_hot_encode_sequence(seq, max_light_len) for seq in data.loc[val_data.index]['LSEQ']])
 
    train_encodings = np.concatenate([train_heavy_encodings, train_light_encodings], axis=1)
    test_encodings = np.concatenate([test_heavy_encodings, test_light_encodings], axis=1)

    return data, train_data, y_train, val_data, y_test, train_encodings, test_encodings


# Split by CDR3 clusters
def train_test_split_by_hcdr3_cluster(data, test_size=0.2):
    clusters = data['HCDR3_CLUSTER_0.8'].unique()
    train_clusters, val_clusters = train_test_split(clusters, test_size=test_size, random_state=42)
    train_data = data[data['HCDR3_CLUSTER_0.8'].isin(train_clusters)]
    val_data = data[data['HCDR3_CLUSTER_0.8'].isin(val_clusters)]
    return train_data, val_data

# Custom dataset with separate CDR3 encoding
class AntibodyDataset(Dataset):
    def __init__(self, heavy_seqs, light_seqs, hcdr3_seqs, labels, barcode, max_heavy_len=135, max_light_len=135, max_hcdr3_len=25):
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
        #antigen_seq = self.antigen_seqs[idx]
        hcdr3 = self.hcdr3_seqs[idx]
        label = self.labels[idx]
        barcode = self.barcode[idx]

        # One-hot encode all sequences
        encoding_heavy = torch.tensor(one_hot_encode_sequence(h_seq, self.max_heavy_len)).float()
        encoding_light = torch.tensor(one_hot_encode_sequence(l_seq, self.max_light_len)).float()
        encoding_cdr3 = torch.tensor(one_hot_encode_sequence(hcdr3, self.max_hcdr3_len)).float()

        # Concatenate all encodings except CDR3 for the main input
        encoding = torch.cat([encoding_heavy, encoding_light], dim=0)
        return encoding, encoding_cdr3, label, barcode, h_seq, l_seq, hcdr3


# Enhanced BinderClassifier with HCDR3-specific processing
class SEC_Classifier(nn.Module):
    def __init__(self, encoding_dim, cdr3_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.2, num_classes=2):
        super(SEC_Classifier, self).__init__()
        self.input_fc = nn.Linear(encoding_dim, hidden_dim)
        self.cdr3_fc = nn.Linear(cdr3_dim, hidden_dim)
        
        # Transformer for main sequence (VH + VL + Antigen)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Separate transformer for CDR3
        # Hope this will work!!
        self.cdr3_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attention mechanism to combine CDR3 with other features 
        # Next steps will incorporate top importance biophysical properties and motif : VERY IMPOPRTANT!!!!
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Combine CDR3 and other features

    def forward(self, encoding, encoding_cdr3):
        # Process main sequence (VH + VL + Antigen)
        x = self.input_fc(encoding.unsqueeze(1))  # Add sequence dimension
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        
        # Process CDR3 separately
        cdr3_x = self.cdr3_fc(encoding_cdr3.unsqueeze(1))
        cdr3_x = self.cdr3_transformer(cdr3_x)
        cdr3_x = cdr3_x.mean(dim=1)
        
        # Apply attention to combine CDR3 with main features
        attn_output, _ = self.attention(cdr3_x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        attn_output = attn_output.squeeze(0)
        
        # Concatenate tensors and final classification
        combined = torch.cat((x, attn_output), dim=1)
        out = self.fc(combined)
        return out

# Train model with class weights and CDR3 emphasis
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, device='cpu', cdr3_weight=1.0):
    model.train()
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for encoding, encoding_cdr3, label, _, _, _, _ in train_loader:
            encoding, encoding_cdr3, label = encoding.to(device), encoding_cdr3.to(device), label.long().to(device)
            optimizer.zero_grad()
            output = model(encoding, encoding_cdr3)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(output, 1)
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)
            train_loss += loss.item() * label.size(0)

        scheduler.step()
        train_loss /= train_total
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_correct = 0
        val_total = 0
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for encoding, encoding_cdr3, label, _, _, _, _ in val_loader:
                encoding, encoding_cdr3, label = encoding.to(device), encoding_cdr3.to(device), label.long().to(device)
                output = model(encoding, encoding_cdr3)
                loss = criterion(output, label)
                _, preds = torch.max(output, 1)
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)
                val_loss += loss.item() * label.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

# Predict binder with enhanced CDR3 sensitivity
def predict_binder(data_input, model, device, max_heavy_len=135, max_light_len=135, max_hcdr3_len=25):
    barcode, VH, VL, HCDR3 = data_input
    model.eval()
    encoding_heavy = torch.tensor(one_hot_encode_sequence(VH, max_heavy_len)).float().to(device).unsqueeze(0)
    encoding_light = torch.tensor(one_hot_encode_sequence(VL, max_light_len)).float().to(device).unsqueeze(0)
 
    encoding_cdr3 = torch.tensor(one_hot_encode_sequence(HCDR3, max_hcdr3_len)).float().to(device).unsqueeze(0)
    encoding = torch.cat([encoding_heavy, encoding_light], dim=1)
    
    with torch.no_grad():
        output = model(encoding, encoding_cdr3)
        probability = torch.softmax(output, dim=1)[0][1].item()
    return probability



# Plot training history 
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, epochs=20, output_path="SEC/images/SEC_transformers_vhvl_onehot_HDR3Enchance_training_history.png"):
    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs_range, train_losses, label='Training Loss', color='blue', lw=2)
    ax1.plot(epochs_range, val_losses, label='Validation Loss', color='red', lw=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax2.plot(epochs_range, train_accuracies, label='Training Accuracy', color='blue', lw=2)
    ax2.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='red', lw=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    plt.close()

# 10-fold cross-validation ROC curve 
def plot_roc_10fold(model_class, dataset, encoding_dim, cdr3_encoding_dim, n_folds=10, epochs=20, device='cpu', output_path="PSR/images/psr_transformers_vhvl_onehot_roc_10fold_curve.png", title='SEC Prediction (Transformers+OneHot Encoding+HCDR3 Attention)'):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    mean_fpr = np.linspace(0, 1, 100)

    # Extract labels and ensure they are 1D
    labels = np.array([dataset[i][2] for i in range(len(dataset))])  # Changed from dataset[i][1] to dataset[i][2] to get label
    if labels.ndim > 1:  # If labels are one-hot encoded or 2D
        labels = np.argmax(labels, axis=1)  # Convert to 1D by taking the class with max probability
    indices = np.arange(len(dataset))

    plt.figure(figsize=(4, 4))

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\nTraining fold {fold+1}/{n_folds}...")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)

        # Initialize model with correct parameters
        model = model_class(encoding_dim=encoding_dim, cdr3_dim=cdr3_encoding_dim, dropout=0.2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Train model
        model.train()
        for epoch in range(epochs):
            for encoding, encoding_cdr3, label, _, _, _, _ in train_loader:
                encoding, encoding_cdr3, label = encoding.to(device), encoding_cdr3.to(device), label.long().to(device)
                optimizer.zero_grad()
                output = model(encoding, encoding_cdr3)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate model
        model.eval()
        probs = []
        preds = []
        true_labels = []
        with torch.no_grad():
            for encoding, encoding_cdr3, label, _, _, _, _ in val_loader:
                encoding, encoding_cdr3 = encoding.to(device), encoding_cdr3.to(device)
                output = model(encoding, encoding_cdr3)
                probs.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
                preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                true_labels.extend(label.cpu().numpy())

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
    plt.title(f'{title}\n 10-Fold Cross-Validation ROC Curve\n'
              f'Acc: {mean_accuracy:.3f}, F1: {mean_f1:.3f}, '
              f'Prec: {mean_precision:.3f}, Rec: {mean_recall:.3f}', fontsize=8)
    plt.legend(loc='lower right', fontsize=5)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    plt.close()

    print(f"\nMean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"Mean Accuracy: {mean_accuracy:.3f}")
    print(f"Mean F1-Score: {mean_f1:.3f}")
    print(f"Mean Precision: {mean_precision:.3f}")
    print(f"Mean Recall: {mean_recall:.3f}")
    return mean_auc, std_auc, mean_accuracy, mean_f1, mean_precision, mean_recall




class ModelWrapperIG(nn.Module):
    def __init__(self, model, max_heavy_len=135, max_light_len=135, max_hcdr3_len=25):
        super(ModelWrapperIG, self).__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.encoding_dim = (max_heavy_len + max_light_len) * len(AMINO_ACIDS)
        self.cdr3_dim = max_hcdr3_len * len(AMINO_ACIDS)

    def forward(self, inputs):
        encoding = inputs[:, :self.encoding_dim]
        encoding_cdr3 = inputs[:, self.encoding_dim:]
        output = self.model(encoding, encoding_cdr3)
        return output[:, 1]  # Return logits for class 1
    
def explain_single_sample_ig(model, dataset, barcode, max_heavy_len=135, max_light_len=135, max_hcdr3_len=25, max_feature=15, n_steps=50, output_path="sec_ig_waterfall_plot_proba.png"):
    
    # Find the sample by barcode or index
    if isinstance(barcode, str):
        for i, (_, _, _, b, _, _, _) in enumerate(dataset):
            if b == barcode:
                sample_idx = i
                break
        else:
            raise ValueError(f"Barcode {barcode} not found in dataset.")
    else:
        sample_idx = barcode
        if sample_idx >= len(dataset):
            raise ValueError(f"Index {sample_idx} out of range. Dataset length: {len(dataset)}")
    
    model.eval()
    wrapper = ModelWrapperIG(model, max_heavy_len, max_light_len, max_hcdr3_len).to(device)
    ig = IntegratedGradients(wrapper)

    # Get sample data
    encoding, encoding_cdr3, label, barcode, h_seq, l_seq, hcdr3 = dataset[sample_idx]
    combined_input = torch.cat([encoding, encoding_cdr3], dim=0).unsqueeze(0).float().to(device).requires_grad_(True)
    baseline = torch.zeros_like(combined_input).to(device)

    # Compute Integrated Gradients
    try:
        attributions = ig.attribute(
            inputs=combined_input,
            baselines=baseline,
            target=None,
            n_steps=n_steps,
            return_convergence_delta=False
        ).detach().cpu().numpy()[0]
    except Exception as e:
        print(f"Error computing IG attributions for sample {sample_idx}: {str(e)}")
        return None

    # Define feature names
    feature_names = ([f"H_{i}_{aa}" for i in range(max_heavy_len) for aa in AMINO_ACIDS] +
                     [f"L_{i}_{aa}" for i in range(max_light_len) for aa in AMINO_ACIDS] +
                     [f"HCDR3_{i}_{aa}" for i in range(max_hcdr3_len) for aa in AMINO_ACIDS])

    # Compute expected value (average model output on baseline)
    with torch.no_grad():
        expected_value = wrapper(baseline).mean().item()
        fx = wrapper(combined_input).item()
    # Add expected value annotation
    prob_fx = 1 / (1 + np.exp(-fx))
    prob_efx = 1 / (1 + np.exp(-expected_value))
    # Select top features
    top_indices = np.argsort(np.abs(attributions))[::-1][:max_feature]
    top_attributions = attributions[top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]

    # Plot waterfall
    plt.figure(figsize=(8, 6))
    colors = ['blue' if val >= 0 else 'red' for val in top_attributions]
    bars = plt.barh(range(len(top_attributions)), top_attributions, color=colors)
    plt.yticks(range(len(top_attributions)), top_feature_names)
    plt.xlabel("Integrated Gradients Attribution")
    plt.title(f"IG ML Interpretability Plot: {barcode} - Label: {'SEC Pass' if label == 1 else ' SEC Fail'} - f(x): {fx:.3f}")


    #plt.text(1.0, 0.9, f'E[f(x)] = {expected_value:.3f} (P = {prob_efx:.3f})',
    #         verticalalignment='top', horizontalalignment='left',
    #         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.2))
    
    # Corrected legend
    plt.legend(handles=[Patch(facecolor='blue'), Patch(facecolor='red')],
               labels=['Positive Contribution', 'Negative Contribution'], loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(main_path, output_path))
    plt.show()
    plt.close()

    return attributions




# Integrated Gradients explanation for entire dataset
def explain_predictions_ig(model, dataset, device='cpu', batch_size=32, n_steps=50):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_attributions = []
    all_inputs = []
    heavy_seqs = []
    light_seqs = []
    antigen_seqs = []
    hcdr3_seqs = []
    indices = []

    wrapper = ModelWrapperIG(model).to(device)
    ig = IntegratedGradients(wrapper)

    for i, (encoding, encoding_cdr3, _, _, h_seqs, l_seqs, hcdr3s) in enumerate(data_loader):
        encoding = encoding.float().to(device)
        encoding_cdr3 = encoding_cdr3.float().to(device)
        combined_input = torch.cat([encoding, encoding_cdr3], dim=1).requires_grad_(True)

        # Define baseline (zero tensor)
        baseline = torch.zeros_like(combined_input).to(device)

        try:
            # Compute Integrated Gradients
            attributions = ig.attribute(
                inputs=combined_input,
                baselines=baseline,
                target=None,  # Target is class 1 (handled in wrapper)
                n_steps=n_steps,
                return_convergence_delta=False
            )
            all_attributions.append(attributions.detach().cpu().numpy())
            all_inputs.append(combined_input.detach().cpu().numpy())
            heavy_seqs.extend(h_seqs)
            light_seqs.extend(l_seqs)
            hcdr3_seqs.extend(hcdr3s)
            indices.extend(list(range(i * batch_size, min((i + 1) * batch_size, len(dataset)))))
            print(f"Processed batch {i+1}/{len(data_loader)}")
        except Exception as e:
            print(f"Error processing batch {i+1}: {str(e)}")
            continue

    if not all_attributions:
        raise ValueError("No attributions computed. Check model or data integrity.")

    attributions = np.concatenate(all_attributions, axis=0)
    inputs = np.concatenate(all_inputs, axis=0)

    return attributions, indices, inputs, heavy_seqs, light_seqs, hcdr3_seqs


# Compute residue-level attribution for IG
def compute_residue_attribution_ig(attributions, heavy_seqs, hcdr3_seqs, max_heavy_len=135, max_hcdr3_len=25):
    residue_importance_h = []
    encoding_dim = len(AMINO_ACIDS)
    total_encoding_dim = (max_heavy_len + 135) * encoding_dim  # VH + VL only

    for attr, h_seq, hcdr3 in zip(attributions, heavy_seqs, hcdr3_seqs):
        hcdr3_start = h_seq.find(hcdr3) if hcdr3 and h_seq else -1
        h_importance = np.zeros(max_heavy_len)
        if hcdr3_start >= 0:
            attr_mean = np.abs(attr)  # Use absolute values for importance
            heavy_attr = attr_mean[:max_heavy_len * encoding_dim].reshape(max_heavy_len, encoding_dim).sum(axis=1)
            hcdr3_attr = attr_mean[total_encoding_dim:total_encoding_dim + max_hcdr3_len * encoding_dim].reshape(max_hcdr3_len, encoding_dim).sum(axis=1)
            h_importance[:len(h_seq)] = heavy_attr[:len(h_seq)]
            if len(hcdr3) <= len(hcdr3_attr):
                h_importance[hcdr3_start:hcdr3_start + len(hcdr3)] += hcdr3_attr[:len(hcdr3)]
        residue_importance_h.append(h_importance)

    return np.array(residue_importance_h)

def aggregate_feature_importance_ig(attributions, input_subset, heavy_seqs, light_seqs, hcdr3_seqs, max_heavy_len=135, max_light_len=135, max_hcdr3_len=25, max_feature=60, output_path="ig_feature_importance_onehot_hl.png", heatmap_path="ig_correlation_heatmap_onehot_hl.png", residue_heatmap_path="ig_hcdr3_residue_importance_heatmap_onehot_hl.png"):
    # Compute mean attributions for plotting (with signs)
    mean_attr = np.mean(attributions, axis=0)
    # Compute mean absolute attributions for ranking top features
    mean_abs_attr = np.mean(np.abs(attributions), axis=0)
    encoding_dim = (max_heavy_len + max_light_len + max_hcdr3_len) * len(AMINO_ACIDS)
    
    # Select top features based on absolute importance
    top_features = np.argsort(mean_abs_attr)[::-1][:max_feature]
    top_attributions = mean_attr[top_features]  # Use signed values for plotting

    # Plot feature importance
    plt.figure(figsize=(10, 8))  # Increased figure size for 50 features
    colors = ['blue' if val >= 0 else 'red' for val in top_attributions]
    plt.barh(range(len(top_features)), top_attributions, color=colors)
    feature_names = ([f"H_{i}_{aa}" for i in range(max_heavy_len) for aa in AMINO_ACIDS] +
                     [f"L_{i}_{aa}" for i in range(max_light_len) for aa in AMINO_ACIDS] +
                     [f"HCDR3_{i}_{aa}" for i in range(max_hcdr3_len) for aa in AMINO_ACIDS])
    plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features], fontsize=8)
    plt.xlabel("Mean Integrated Gradients Attribution", fontsize=10)
    plt.ylabel("SEC-Transformers: VH+VL+HCDR3 OneHot Encodings", fontsize=10)
    plt.title(f"Top {max_feature} Feature Importance for Entire IPI SEC Dataset (Integrated Gradients)", fontsize=12)
    plt.legend(handles=[Patch(facecolor='blue'), Patch(facecolor='red')],
               labels=['Positive Contribution', 'Negative Contribution'], loc='best', fontsize=8)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(main_path, output_path))
    plt.show()
    plt.close()

    # Compute residue-level importance
    residue_importance_h = compute_residue_attribution_ig(attributions, heavy_seqs, hcdr3_seqs, max_heavy_len, max_hcdr3_len)
    max_hcdr3_len = max(len(seq) for seq in hcdr3_seqs if seq) if any(hcdr3_seqs) else 1
    hcdr3_data = np.zeros((1, max_hcdr3_len))
    for i, seq in enumerate(hcdr3_seqs):
        if seq:
            start = heavy_seqs[i].find(seq) if heavy_seqs[i] else -1
            if start >= 0 and start < residue_importance_h.shape[1]:
                end = min(start + len(seq), residue_importance_h.shape[1])
                hcdr3_data[0, :end-start] += residue_importance_h[i, start:end]
    hcdr3_data /= max(1, sum(1 for seq in hcdr3_seqs if seq))

    # Plot HCDR3 residue importance
    plt.figure(figsize=(10, 2))
    sns.heatmap(hcdr3_data, cmap='viridis', xticklabels=range(1, max_hcdr3_len + 1), yticklabels=['CDR3'])
    plt.title("HCDR3 Residue Importance (Integrated Gradients)")
    plt.xlabel("HCDR3 Position")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(os.path.join(main_path, residue_heatmap_path))
    plt.show()
    plt.close()

    top_hcdr3_pos = np.argsort(np.sum(hcdr3_data[0], axis=0))[-5:][::-1]
    top_residues = [
        {'Chain': 'CDR3', 'Position': pos + 1, 'Importance': hcdr3_data[0, pos]} for pos in top_hcdr3_pos
    ]

    return top_features, top_attributions, [], top_residues

# Function to generate all single mutations for CDR3 with positions
def generate_cdr3_mutations_with_positions(original_cdr3, amino_acids="ARNDCQEGHILKMFPSTWYV"):
    mutations = {}
    for i, aa in enumerate(original_cdr3):
        for new_aa in amino_acids:
            if new_aa != aa:
                pos = i + 1  # Position starting from 1
                if pos not in mutations:
                    mutations[pos] = {}
                mutations[pos][new_aa] = None  # Placeholder for probability
    return mutations

# Function to process mutations and get probabilities using the model
def process_mutations_with_prob(input_data, model, device):
    barcode, VH, VL, original_cdr3 = input_data
    
    # Generate mutation structure
    mutation_dict = generate_cdr3_mutations_with_positions(original_cdr3)
    
    # Fill probabilities
    for pos in mutation_dict:
        for aa in mutation_dict[pos]:
            mutant_cdr3 = original_cdr3[:pos-1] + aa + original_cdr3[pos:]
      
            VH_mutant=VH.replace(original_cdr3,mutant_cdr3)
   
            mutant_input = [barcode, VH_mutant, VL, mutant_cdr3]
            probability = predict_binder(mutant_input, model, device)
            mutation_dict[pos][aa] = probability
    
    return mutation_dict

# Create heatmap data with original CDR3 AAs as x-axis labels
def create_heatmap_data(input_data, model, device):
    mutation_dict = process_mutations_with_prob(input_data, model, device)
    cdr3_length = len(input_data[3])  
    original_cdr3 = input_data[3]
    amino_acids = "ARNDCQEGHILKMFPSTWYV"
    
    # Initialize heatmap matrix
    heatmap_data = np.zeros((len(amino_acids), cdr3_length))
    
    # Fill heatmap data
    for pos in range(1, cdr3_length + 1):
        if pos in mutation_dict:
            for i, aa in enumerate(amino_acids):
                if aa in mutation_dict[pos]:
                    heatmap_data[i, pos-1] = mutation_dict[pos][aa]
    
    return heatmap_data, amino_acids, list(original_cdr3)  # Use original CDR3 AAs as x-axis labels

# Plot heatmap for a specific barcode or index
def plot_heatmap_for_sample(model, dataset, barcode, device, output_path="SEC/images/sec_cdr3_mutation_heatmap.png"):
    # Find the sample by barcode or index
    if isinstance(barcode, str):
        for i, (_, _, _, b, _, _, _) in enumerate(dataset):
            if b == barcode:
                sample_idx = i
                break
        else:
            raise ValueError(f"Barcode {barcode} not found in dataset.")
    else:
        sample_idx = barcode
        if sample_idx >= len(dataset):
            raise ValueError(f"Index {sample_idx} out of range. Dataset length: {len(dataset)}")
    
    # Get sample data
    _, _,label, barcode, VH, VL, HCDR3 = dataset[sample_idx]
    input_data = [barcode, VH, VL, HCDR3]
    
  
    
    psr_probability = predict_binder(input_data, model, device)
    # Generate heatmap data
    heatmap_data, amino_acids, positions = create_heatmap_data(input_data, model, device)
 
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", xticklabels=positions, yticklabels=list(amino_acids))
    plt.xlabel("Original CDR3 Amino Acid")
    plt.ylabel("Mutant Amino Acid")
    plt.title(f"SEC Probability for CDR3 Mutations - {barcode} (Label: {'SEC PASS' if label == 1 else 'SEC Fail'})\n Transformers Model-VH-VL-CDR3 OneHot Prediction = {psr_probability}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    plt.close()



# Main Binder prediction workflow
if __name__ == "__main__":
  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_heavy_len = 135
    max_light_len = 135
    max_antigen_len = 1251
    max_hcdr3_len = 25
    encoding_dim = (max_heavy_len + max_light_len) * len(AMINO_ACIDS)
    cdr3_dim = max_hcdr3_len * len(AMINO_ACIDS)

    # loading data for trainning and testing
    data_elisa_ngs, train_data, y_train, val_data, y_test, train_encodings, test_encodings = load_ipi_psr_dataset(max_heavy_len, max_light_len)

    train_heavy = train_data['HSEQ'].values
    train_light = train_data['LSEQ'].values
    #train_antigen = train_data['antigen_aa'].values
    train_hcdr3 = train_data['CDR3'].values
    test_heavy = val_data['HSEQ'].values
    test_light = val_data['LSEQ'].values
    #test_antigen = val_data['antigen_aa'].values
    test_hcdr3 = val_data['CDR3'].values
    test_barcodes = val_data['BARCODE'].values


    train_dataset = AntibodyDataset(train_heavy, train_light,train_hcdr3, y_train, train_data['BARCODE'].values,
                                   max_heavy_len, max_light_len, max_hcdr3_len)
    test_dataset = AntibodyDataset(test_heavy, test_light, test_hcdr3, y_test, val_data['BARCODE'].values,
                                  max_heavy_len, max_light_len, max_hcdr3_len)
    full_dataset = ConcatDataset([train_dataset, test_dataset])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=16)    

    ## Tranformers model trainning
    model = SEC_Classifier(encoding_dim=encoding_dim, cdr3_dim=cdr3_dim, dropout=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=15, device=device
    )

    # plot trainning history
    plot_training_history(
        train_losses, val_losses, train_accuracies, val_accuracies,
        epochs=15, output_path="SEC/images/sec_transformer_vhvl_onehot_hcdr3enchance_training_history_july2025.png"
    )

    print("Standard threshold=0.5")
    model.eval()
    test_preds = []
    test_true = []
    test_probs = []
    with torch.no_grad():
        for encoding,encoding_cdr3,label, _, _, _, _ in val_loader:
            encoding = encoding.to(device)
            output = model(encoding,encoding_cdr3)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            test_preds.extend((probs >= 0.5).astype(int))
            test_true.extend(label.numpy())
            test_probs.extend(probs)
    
    test_accuracy = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds)
    test_auc = roc_auc_score(test_true, test_probs)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test roc_auc: {test_auc:.4f}")
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_true, test_preds)
    print("\nConfusion Matrix (Standard Threshold):")
    print(f"True Negatives (Non-Binders Correct): {cm[0,0]}")
    print(f"False Positives (Non-Binders as Binders): {cm[0,1]}")
    print(f"False Negatives (Binders as Non-Binders): {cm[1,0]}")
    print(f"True Positives (Binders Correct): {cm[1,1]}")

    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(test_true, test_probs, pos_label=1)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    print(f"Using Optimal threshold: {optimal_threshold:.4f}")
    test_preds = []
    test_true = []
    test_probs = []
    with torch.no_grad():
        for encoding,encoding_cdr3,label, _, _, _, _ in val_loader:
            encoding = encoding.to(device)
            output = model(encoding,encoding_cdr3)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            test_preds.extend((probs >= optimal_threshold).astype(int))
            test_true.extend(label.numpy())
            test_probs.extend(probs)
    
    test_accuracy = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds)
    test_auc = roc_auc_score(test_true, test_probs)
    print(f"\nTest Accuracy: {test_accuracy:.3f}")
    print(f"Test F1 Score: {test_f1:.3f}")
    print(f"Test roc_auc: {test_auc:.3f}")
    
    cm = confusion_matrix(test_true, test_preds)
    print("\nConfusion Matrix (Optimal Threshold):")
    print(f"True Negatives (Non-Binders Correct): {cm[0,0]}")
    print(f"False Positives (Non-Binders as Binders): {cm[0,1]}")
    print(f"False Negatives (Binders as Non-Binders): {cm[1,0]}")
    print(f"True Positives (Binders Correct): {cm[1,1]}")

    prob_df = pd.DataFrame({
        'BARCODE': test_barcodes,
        'True_Label': test_true,
        'Predicted_Label': test_preds,
        'Prob_Binder': test_probs
    })
    prob_df.to_csv(os.path.join(main_path, "SEC/images/sec_transformer_vhvl_onehot_hcdr3enchance_test_predictive_probabilities_withoutfilter.csv"), index=False)




    ### 10-Fold Cross validation
    mean_auc, std_auc, mean_accuracy, mean_f1, mean_precision, mean_recall = plot_roc_10fold(
        SEC_Classifier, full_dataset, encoding_dim=encoding_dim,cdr3_encoding_dim=cdr3_dim,
        n_folds=10, epochs=11, device=device, output_path="SEC/images/sec_transformer_vhvl_onehot_hcdr3enchance_roc_10fold_curve_withoutfiltering.png"
    )

    barcode="TAB0012652"

    barcode = "TAB0011772"
    barcode="TAB0011462"
    barcode="TAB0011599"
    barcode='TAB0011605'
    barcode='TAB0011612'
    barcode='TAB0009604'
    barcode='TAB0016438'
    barcode='TAB0015620'
    barcode='TAB0011971'
    barcode='TAB0010779'
    barcode='TAB0009257'
    barcode='TAB0014385'
    barcode='TAB0010163'
    barcode='TAB0010136' # premium
    barcode='TAB0017072' #SPR=premium
    plot_heatmap_for_sample(model,full_dataset, barcode, device)

    attributions_single = explain_single_sample_ig(
            model, full_dataset, barcode, max_heavy_len=max_heavy_len, max_light_len=max_light_len,
            max_hcdr3_len=max_hcdr3_len, max_feature=20, n_steps=1000,
            output_path="SEC/images/sec_onehotONLY_cdr3_enchance_shap_waterfall_plot_proba_sample1.png"
    )



    print("\nRunning IG analysis for entire dataset...")
    attributions, indices, input_subset, heavy_seqs, light_seqs, hcdr3_seqs = explain_predictions_ig(
        model, full_dataset, device, batch_size=32, n_steps=100
    )

    top_features, mean_attr, _, top_residues = aggregate_feature_importance_ig(
        attributions, input_subset, heavy_seqs, light_seqs, hcdr3_seqs,
        max_heavy_len, max_light_len, max_hcdr3_len,max_feature=50
    )

    print("\nGlobal Feature Importance for Entire Dataset (Integrated Gradients):")
    feature_names = ([f"H_{i}_{aa}" for i in range(max_heavy_len) for aa in AMINO_ACIDS] +
                     [f"L_{i}_{aa}" for i in range(max_light_len) for aa in AMINO_ACIDS] +
                     [f"antigen_{i}_{aa}" for i in range(max_antigen_len) for aa in AMINO_ACIDS] +
                     [f"HCDR3_{i}_{aa}" for i in range(max_hcdr3_len) for aa in AMINO_ACIDS])
    for idx, attr_val in zip(top_features, mean_attr):
        print(f"{feature_names[idx]}: Mean |IG Attribution| = {attr_val:.4f}")



    ## PREDICT ALL IPI DATA
    ipi_data = pd.read_excel("data/ipi_antibodydb_july2025.xlsx")
    ipi_data.loc[ipi_data['CDR3'].str.startswith('C'), 'CDR3'] = ipi_data['CDR3'].str[1:]

    ipi_data = ipi_data.dropna(subset=['HSEQ', 'LSEQ', 'CDR3'])

    ipi_heavy = ipi_data['HSEQ'].values
    ipi_light = ipi_data['LSEQ'].values
    ipi_hcdr3 = ipi_data['CDR3'].values
    ipi_dataset = AntibodyDataset(ipi_heavy, ipi_light,ipi_hcdr3, ipi_data['sec_filter'].values, ipi_data['BARCODE'].values,
                                   max_heavy_len, max_light_len, max_hcdr3_len)
    ipi_loader = DataLoader(ipi_dataset, batch_size=64, shuffle=True)

    print("Standard threshold=0.5")
    model.eval()
    test_preds = []
    test_true = []
    test_probs = []
    with torch.no_grad():
        for encoding,encoding_cdr3,label, _, _, _, _ in ipi_loader:
            encoding = encoding.to(device)
            output = model(encoding,encoding_cdr3)
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            test_preds.extend((probs >= 0.5).astype(int))
            test_true.extend(label.numpy())
            test_probs.extend(probs)
    
    ipi_data['sec_transformer']=test_preds
    ipi_data['sec_transformer_proba']=test_probs
    ipi_data.to_excel("data/ipi_antibodydb_july2025.xlsx",index=False)

    barcode='TAB0011471'
    barcode='TAB0011477'
    barcode='TAB0011481'
    barcode='TAB0011448'
    barcode='TAB0011449'
    barcode='TAB0011457'
    plot_heatmap_for_sample(model, ipi_dataset, barcode, device)

    attributions_single = explain_single_sample_ig(
            model, ipi_dataset, barcode, max_heavy_len=max_heavy_len, max_light_len=max_light_len,
            max_hcdr3_len=max_hcdr3_len, max_feature=20, n_steps=1000,
            output_path="SEC/images/sec_validation_"+barcode+".png"
    )
