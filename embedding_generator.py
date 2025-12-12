

"""
IPI Antibody Developability Prediction Platform
Final Production Version — DEC-2025 
Supports: SEC & PSR | XGBoost & RF & CNN , Transformers | ablang, antiberty, antiberta2, antiberta2-cssp
"""

# embedding_generator.py

# ablang → 480 columns
# antiberty → 512 columns (fixed — no zeros!)
# antiberta2 → 1024 columns

import os
import pandas as pd
import numpy as np
import torch
import math
import time
from tqdm import tqdm
from antiberty import AntiBERTyRunner
# ==================== ABLANG2 ====================
try:
    import ablang2
    ABLANG_OK = True
except ImportError:
    ABLANG_OK = False

# ==================== ANTIBERTY ====================
try:
    from antiberty import AntiBERTyRunner
    ANTIBERTY_OK = True
except ImportError:
    ANTIBERTY_OK = False

# ==================== ANTIBERTA2 ====================
try:
    from transformers import RoFormerTokenizer, RoFormerForMaskedLM
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False


def _insert_space_every_other_except_cls(input_string: str) -> str:
    """spacing function for AntiBERTy/AntiBERTa2"""
    parts = input_string.split('[CLS]')
    modified_parts = [''.join([char + ' ' for char in part]).strip() for part in parts]
    result = ' [CLS] '.join(modified_parts)
    return result


def batch_loader(data, batch_size):
    """batch loader"""
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        yield i, end_idx, data[i:end_idx]


def generate_embedding(input_excel: str, lm: str = "antiberta2", batch_size: int = 64, device: str = "cpu") -> str:
    """
    Generate embedding for antibody sequences in input_excel using specified language model (lm).
    """
    out_path = f"{input_excel}.{lm}.emb.csv"

    if os.path.exists(out_path):
        print(f"Embedding exists: {out_path}")
        return out_path

    print(f"\nGenerating {lm.upper()} embedding → {os.path.basename(out_path)}")

    df = pd.read_excel(input_excel)

    if 'HSEQ' not in df.columns or 'LSEQ' not in df.columns:
        raise ValueError("Excel must have 'HSEQ' and 'LSEQ' columns")

    df = df.dropna(subset=['HSEQ', 'LSEQ']).copy()
    if len(df) == 0:
        raise ValueError("No valid H+L pairs found!")

    if 'BARCODE' in df.columns:
        df = df.set_index('BARCODE')
    else:
        df.index = [f"seq_{i:05d}" for i in range(len(df))]

    embeddings = []

    # ==================== ABLANG2 — 480 columns ====================
    if lm == "ablang" and ABLANG_OK:
        print("Using ABLang2 → 480-dim vectors")
        ablang_model = ablang2.pretrained("ablang2-paired", device=device, random_init=False, ncpu=3)

        for idx in tqdm(df.index, desc="ABLang2"):
            try:
                VHVL = [str(df.loc[idx, 'HSEQ']), str(df.loc[idx, 'LSEQ'])]
                emb = ablang_model(VHVL, mode='seqcoding')[0]  # 480-dim
                embeddings.append(emb)
            except Exception as e:
                print(f"  Warning: ABLang2 failed on {idx}: {e} → zeros")
                embeddings.append(np.zeros(480))

        emb_df = pd.DataFrame(embeddings, index=df.index)
        emb_df.to_csv(out_path)
        print(f"SAVED ABLang2: {out_path} | Shape: {emb_df.shape}")
        return out_path

    # ==================== ANTIBERTY — 512 columns  ====================
    elif lm == "antiberty" and ANTIBERTY_OK:
        print("Using AntiBERTy → 512-dim vectors")
        runner = AntiBERTyRunner()

        # HL concatenation
        df_temp = df.copy()
        df_temp['HL'] = df_temp['HSEQ'].astype(str) + '[CLS][CLS]' + df_temp['LSEQ'].astype(str)
        sequences = df_temp['HL'].tolist()

        # spacing
        sequences = [s.replace('  ', ' ') for s in sequences]  # clean double spaces

        # exact batching
        n_seqs = len(sequences)
        dim = 512  # AntiBERTy dimension
        embeddings = torch.empty((n_seqs, dim))
        batch_size = 500  # 

        i = 1
        for start, end, batch in batch_loader(sequences, batch_size):
            print(f'Batch {i}/{math.ceil(n_seqs / batch_size)}')
            try:
                batch_embs = runner.embed(batch)  # list of tensors (L, 512)
                batch_mean = [e.mean(axis=0) for e in batch_embs]  # (512,)
                embeddings[start:end] = torch.stack(batch_mean)
            except Exception as e:
                print(f"  Batch {i} failed: {e} → zeros")
                embeddings[start:end] = torch.zeros(end - start, dim)
            i += 1

        emb_df = pd.DataFrame(embeddings.numpy(), index=df.index)
        emb_df.to_csv(out_path)
        print(f"SAVED AntiBERTy: {out_path} | Shape: {emb_df.shape} → 512 columns")
        return out_path

    # ==================== ANTIBERTA2 / CSSP — 1024 columns ====================
     # ==================== ANTIBERTA2 / CSSP — 1024 columns ====================
    elif lm in ["antiberta2", "antiberta2-cssp"] and TRANSFORMERS_OK:
        model_name = "alchemab/antiberta2-cssp" if lm == "antiberta2-cssp" else "alchemab/antiberta2"
        print(f"Using {model_name} → 1024-dim")
        tokenizer = RoFormerTokenizer.from_pretrained(model_name)
        model = RoFormerForMaskedLM.from_pretrained(model_name).to(device)
        model.eval()

        # Prepare sequences
        df_temp = df.copy()
        df_temp['HL'] = df_temp['HSEQ'].astype(str) + '[CLS][CLS]' + df_temp['LSEQ'].astype(str)
        sequences = df_temp['HL'].tolist()
        sequences = [_insert_space_every_other_except_cls(s) for s in sequences]
        sequences = [s.replace('  ', ' ') for s in sequences]

        n_seqs = len(sequences)
        dim = 1024
        embeddings = torch.empty((n_seqs, dim))

        batch_size = 128
        i = 1
        for start, end, batch in batch_loader(sequences, batch_size):
            print(f'Batch {i}/{math.ceil(n_seqs / batch_size)}')
            try:
                # CORRECT WAY — use return_tensors="pt"
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"  # ← THIS IS THE KEY!
                ).to(device)

                with torch.no_grad():
                    outputs = model(**encoded, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]  # (B, L, 1024)

                # Mean pooling over non-padding tokens
                attention_mask = encoded['attention_mask'].unsqueeze(-1)  # (B, L, 1)
                masked_hidden = hidden_states * attention_mask
                pooled = masked_hidden.sum(1) / attention_mask.sum(1)
                embeddings[start:end] = pooled.cpu()

            except Exception as e:
                print(f"  Batch {i} failed: {e} → using zeros")
                embeddings[start:end] = torch.zeros(end - start, dim)
            i += 1

        emb_df = pd.DataFrame(embeddings.numpy(), index=df.index)
        emb_df.to_csv(out_path)
        print(f"SAVED {lm}: {out_path} | Shape: {emb_df.shape}")
        return out_path
    else:
        raise ValueError(f"LM '{lm}' not supported or package missing")


def batch_loader(data, batch_size):
    """batch loader"""
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        yield i, end_idx, data[i:end_idx]