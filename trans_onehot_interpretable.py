# transformer_onehot_interpretable.py
# Transformer with One-Hot Encoding and Integrated Gradients Interpretability
# Supports train, predict, k-fold, etc.
# IPI Antibody Developability Prediction Platform
# Created by Hoan Nguyen | Final Production Version — DEC-2025

import os
import matplotlib.pyplot as plt

import sys
import os

# Add the path to your project root (adjust if needed)
project_path = "/Users/Hoan.Nguyen/ComBio/MachineLearning/IPIPred"  # Change to your actual project path
sec_model_path = "/Users/Hoan.Nguyen/ComBio/MachineLearning/IPIPred/SEC/models_final/FINAL_sec_filter_onehot_transformer_onehot.pt"
psr_model_path = "/Users/Hoan.Nguyen/ComBio/MachineLearning/IPIPred/SEC/models_final/FINAL_psr_filter_onehot_transformer_onehot.pt"
    

sys.path.append(project_path)
from models.transformer_onehot import TransformerOneHotModel
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def one_hot_encode_sequence(sequence, max_length):
    sequence = sequence.replace('-', '').upper()
    encoding = np.zeros(max_length * 20)
    for i, aa in enumerate(sequence[:max_length]):
        if aa in AMINO_ACIDS:
            encoding[i * 20 + AMINO_ACIDS.index(aa)] = 1
    return encoding

def analyze_new_antibody(
    model_path: str,
    vh_seq: str,
    vl_seq: str,
    hcdr3_seq: str,
    barcode: str = "NEW_AB",
    original_label: str = "FAIL",
    target_analysis: str = "SEC",
    output_dir: str = "new_antibody_analysis",
    ig_n_steps: int = 1000,
    show_plots: bool = True
):
    """
    Complete analysis pipeline for a completely new antibody.
    """
    os.makedirs(output_dir, exist_ok=True)

    if (target_analysis not in ["SEC", "PSR"]):
        raise ValueError("target_analysis must be either 'SEC' or 'PSR'") 
    elif target_analysis == "SEC":
        model_path = sec_model_path
    else:  # PSR
        model_path = psr_model_path
    
    print(f"Loading model from {model_path}...")
    model = TransformerOneHotModel.load(model_path)


    print(f"\nAnalyzing new antibody: {barcode}")
    prob = model.predict_single("temp", vh_seq, vl_seq, hcdr3_seq)
    print(prob)
    pred_label = "PASS" if prob >= 0.5 else "FAIL"
    print(f"Prediction probability (PASS = 1): {prob:.4f} → Predicted: {pred_label}")

    print("Computing Integrated Gradients attribution...")
    importance = model.integrated_gradients_single(vh_seq, vl_seq, hcdr3_seq, n_steps=ig_n_steps)

    # Waterfall plot (your original style)
    plt.figure(figsize=(10, 8))
    top_n = 20
    abs_imp = np.abs(importance)
    top_idx = np.argsort(abs_imp)[-top_n:][::-1]
    top_values = importance[top_idx]

    # Generate labels: H.1.A, L.1.B, HCDR3.1.C
    labels = []
    pos = 1
    for aa in vh_seq:
        labels.append(f"H.{pos}.{aa}")
        pos += 1
    pos = 1
    for aa in vl_seq:
        labels.append(f"L.{pos}.{aa}")
        pos += 1
    pos = 1
    for aa in hcdr3_seq:
        labels.append(f"HCDR3.{pos}.{aa}")
        pos += 1

    top_labels = [labels[i] for i in top_idx]

    colors = ['blue' if v > 0 else 'red' for v in top_values]
    plt.barh(range(len(top_values)), top_values, color=colors)
    plt.yticks(range(len(top_values)), top_labels)
    plt.xlabel("Integrated Gradients Attribution")
    plt.title(f"IG ML Interpretability Plot: {barcode} - Label: PSR {pred_label} - f(x)={prob:.4f}")
    plt.legend(handles=[Patch(facecolor='blue'), Patch(facecolor='red')],
               labels=['Positive Contribution', 'Negative Contribution'], loc='lower right')
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()

    ig_path = os.path.join(output_dir, f"{barcode}_IG_waterfall.png")
    plt.savefig(ig_path, dpi=300)
    if show_plots:
        plt.show()
    plt.close()
    print(f"IG waterfall plot saved: {ig_path}")

    # CDR3 mutagenesis heatmap
    print("Generating CDR3 mutagenesis heatmap...")

    output_path=os.path.join(output_dir, f"{barcode}_cdr3_mutagenesis.png")
    print(output_path)
    #plot_cdr3_mutation_heatmap(self, barcode, dataset, output_path="SEC/image/sec_cdr3_mutation_heatmap.png"):
    model.plot_cdr3_mutation_heatmap(vh_seq, vl_seq, hcdr3_seq, barcode,original_label,target_analysis="SEC",output_path=output_path)

    print(f"\nAnalysis complete! Results saved in: {output_dir}")

    return {
        'barcode': barcode,
        'probability': prob,
        'predicted_label': pred_label,
        'importance': importance,
        'ig_plot': ig_path,
        'mutagenesis_plot': os.path.join(output_dir, f"{barcode}_cdr3_mutagenesis.png")
    }




# Example usage
if __name__ == "__main__":
    
    result = analyze_new_antibody(
        model_path="/Users/Hoan.Nguyen/ComBio/MachineLearning/IPIPred/SEC/models_final/FINAL_sec_filter_onehot_transformer_onehot.pt",
        vh_seq="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQEGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARSDEWWWGDIYVFDIWGQGTLVTVSS",
        vl_seq="DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPYTFGQGTKLEIKL",
        hcdr3_seq="ARSDEWWWGDIYVFDI",
        barcode="NEW AB001",
        original_label="UNKNOW",
        target_analysis="PSR",
        output_dir="/Users/Hoan.Nguyen/ComBio/MachineLearning/IPIPred/SEC/tmp")
    
    print(result)