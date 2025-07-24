import numpy as np
import pandas as pd
import torch
import ablang2
import os

from anarci import anarci

# Extract CDR3 using ANARCI (IMGT)
def extract_cdr3(sequence, chain_type='H'):

    #sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARYDPTTDTSYFDYWGQGTLVTVSS"
   
    results = anarci([('seq',sequence)], scheme='imgt',output=False,allow=["H","L"])
    if not results[1] or not results[1][0]:
        return ''
    numbering = results[0][0][0][0]
    cdr3_start = 105  # IMGT CDR3 starts at 105
    cdr3_end = 117    # IMGT CDR3 ends at 117
    cdr3 = ''
    for pos, aa in numbering:
        pos_num = int(pos[0]) if isinstance(pos, tuple) else pos
        if cdr3_start <= pos_num <= cdr3_end and aa != '-':
            cdr3 += aa
    return cdr3

def precompute_embeddings(data, ablang2_paired, output_path="data/precomputed_embeddings.pt"):
    embeddings = []
    barcodes = []
    heavy_seqs = []
    light_seqs = []
    hcdr3_seqs = []
    lcdr3_seqs = []

    for idx, row in data.iterrows():
        h_seq = row['HSEQ']
        l_seq = row['LSEQ']
        barcode = row['BARCODE']
        
        # Compute embedding and convert to PyTorch tensor
        emb = ablang2_paired([h_seq, l_seq], mode='rescoding')  # Shape: (2, seq_len, embedding_dim)
        emb = torch.tensor(emb, dtype=torch.float32)  # Convert to PyTorch tensor
        
        # Extract CDR3 sequences
        hcdr3 = extract_cdr3(h_seq, chain_type='H')
        lcdr3 = extract_cdr3(l_seq, chain_type='L')
        
        embeddings.append(emb)
        barcodes.append(barcode)
        heavy_seqs.append(h_seq)
        light_seqs.append(l_seq)
        hcdr3_seqs.append(hcdr3)
        lcdr3_seqs.append(lcdr3)
    
    # Save to PyTorch tensor file
    torch.save({
        'barcodes': barcodes,
        'embeddings': embeddings,  # Now a list of PyTorch tensors
        'heavy_seqs': heavy_seqs,
        'light_seqs': light_seqs,
        'hcdr3_seqs': hcdr3_seqs,
        'lcdr3_seqs': lcdr3_seqs
    }, output_path)
    print(f"Saved precomputed embeddings to {output_path}")

# Example usage
if __name__ == "__main__":
    main_path = "/Users/Hoan.Nguyen/ComBio/MachineLearning/"
    os.chdir(main_path)
    ablang2_paired = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=10, device='cpu')
    #data = pd.read_excel("data/ipi_antibodydb.xlsx")
    data = pd.read_excel("data/psr_trainset_elisa_ngs.xlsx")
    #data = data.merge(data_elisa_ngs[['BARCODE', 'HSEQ', 'LSEQ']], on='BARCODE', how='inner')
    precompute_embeddings(data, ablang2_paired, output_path="data/psr_trainset_elisa_nbgs_ablang2_embeddings.pt")
