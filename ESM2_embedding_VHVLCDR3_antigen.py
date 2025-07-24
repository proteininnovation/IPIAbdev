import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer
import numpy as np
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


# Load the Excel file
df= pd.read_excel("data/ipi_antibodydb_july2025.xlsx")
# Load ESM-2 model and tokenizer
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)
model.eval()

# Device setup (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to compute mean-pooled embedding for a sequence
def get_embedding(sequence):
    if pd.isna(sequence) or not sequence:
        return None
    # Tokenize sequence
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling of the last hidden state
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

# Process each sequence column
embeddings_data = {
    "BARCODE": df["BARCODE"],
    "HSEQ_embedding": [],
    "LSEQ_embedding": [],
    "HCDR3_embedding":[],
    "antigen_aa_embedding": []
}

for idx, row in df.iterrows():
    # Generate embeddings for each sequence
    hseq_emb = get_embedding(row["HSEQ"])
    lseq_emb = get_embedding(row["LSEQ"])
    hcdr3_emb = get_embedding(row["CDR3"])
    antigen_emb = get_embedding(row["antigen_aa"])
    embeddings_data["HSEQ_embedding"].append(hseq_emb)
    embeddings_data["LSEQ_embedding"].append(lseq_emb)
    embeddings_data["HCDR3_embedding"].append(hcdr3_emb)
    embeddings_data["antigen_aa_embedding"].append(antigen_emb)

# Convert embeddings to DataFrame
output_df = pd.DataFrame(embeddings_data)

# Save embeddings to a new Excel file
output_file = "embedding/esm2_embeddings.csv"
output_df.to_csv(output_file, index=False)
print(f"Embeddings saved to {output_file}")



HCDR3_emb_df=pd.DataFrame(np.array(embeddings_data['HCDR3_embedding']))
HCDR3_emb_df.index=embeddings_data['BARCODE']
HCDR3_emb_df.to_csv("embedding/hcdr3_esm2_embeddings.csv")

HSEQ_emb_df=pd.DataFrame(np.array(embeddings_data['HSEQ_embedding']))
HSEQ_emb_df.index=embeddings_data['BARCODE']
HSEQ_emb_df.to_csv("embedding/hseq_esm2_embeddings.csv")

#antigen_emb_df=pd.DataFrame(np.array(embeddings_data['antigen_aa_embedding']))
#antigen_emb_df.index=embeddings_data['BARCODE']
#antigen_emb_df.to_csv("embedding/antigen_esm2_embeddings.csv")


# Parameters
fixed_length = 1280

# Inspect the first few embeddings
for i in range(5):  # Check first 5 elements
    emb = embeddings_data['LSEQ_embedding'][i]
    print(f"Element {i}: type={type(emb)}, len={len(emb) if isinstance(emb, (list, np.ndarray)) else 'N/A'}, shape={np.array(emb).shape if isinstance(emb, (list, np.ndarray)) else 'N/A'}")

# Filter valid embeddings
valid_embeddings = []
BARCODE=[]
for i, emb in enumerate(embeddings_data['LSEQ_embedding'][0:HCDR3_emb_df.shape[0]]):
    try:
        emb_array = np.array(emb)
        if emb_array.shape == (fixed_length,) or (emb_array.ndim == 1 and len(emb_array) == fixed_length):
            valid_embeddings.append(emb_array)
            BARCODE.append(HCDR3_emb_df.index[i])
        else:
            print(f"Skipping embedding {i}: invalid shape {emb_array.shape}")
    except Exception as e:
        print(f"Skipping embedding {i}: error {str(e)}")

# Convert to NumPy array and DataFrame
embeddings_array = np.array(valid_embeddings)
LSEQ_emb_df = pd.DataFrame(embeddings_array)
LSEQ_emb_df.index=BARCODE
LSEQ_emb_df.to_csv("embedding/LSEQ_esm2_embeddings.csv")



# Parameters
fixed_length = 1280
# Filter valid embeddings
valid_embeddings = []
BARCODE=[]
for i, emb in enumerate(embeddings_data['antigen_aa_embedding'][0:HCDR3_emb_df.shape[0]]):
    try:
        emb_array = np.array(emb)
        if emb_array.shape == (fixed_length,) or (emb_array.ndim == 1 and len(emb_array) == fixed_length):
            valid_embeddings.append(emb_array)
            BARCODE.append(HCDR3_emb_df.index[i])
        else:
            print(f"Skipping embedding {i}: invalid shape {emb_array.shape}")
    except Exception as e:
        print(f"Skipping embedding {i}: error {str(e)}")

# Convert to NumPy array and DataFrame
embeddings_array = np.array(valid_embeddings)
antigen_emb_df = pd.DataFrame(embeddings_array)
antigen_emb_df.index=BARCODE
antigen_emb_df.to_csv("embedding/antigen_esm2_embeddings.csv")



