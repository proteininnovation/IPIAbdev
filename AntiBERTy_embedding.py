#--------------------------------------------
# Generate antiBERTy embedding
# 
# AntiBERTy is available on PyPI:
# https://pypi.org/project/antiberty/
#
# Example usage: 
#
# python antiBERTy_embedding.py antibody_sequences_file 
#--------------------------------------------
from antiberty import AntiBERTyRunner
import argparse
import torch
import numpy as np
import pandas as pd
import time
from Bio import SeqIO
import pandas as pd
import math

#--------------------------------------------

parser = argparse.ArgumentParser(description="Generate antiBERTa2 embedding. Example usage: \n python antiBERTa2_embed.py")
parser.add_argument("sequence", type=str, help="Path to the antibody sequence  file.")

args = parser.parse_args()



#--------------------------------------------
def batch_loader(data, batch_size):
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        yield i, end_idx, data[i:end_idx]

def insert_space_every_other_except_cls(input_string):
    parts = input_string.split('[CLS]')
    modified_parts = [''.join([char + ' ' for char in part]).strip() for part in parts]
    result = ' [CLS] '.join(modified_parts)
    return result


###

input_seq=args.sequence
dat = pd.read_excel(input_seq)
#dat =pd.read_excel('/Users/Hoan.Nguyen/ComBio/AntigenDB/datasources/ipi_data/ipi_antibodydb.xlsx')

dat=dat[pd.notna(dat['LSEQ'])]
dat=dat[pd.notna(dat['HSEQ'])]
dat=dat.set_index('BARCODE')

dat['HL']=dat['HSEQ']+'[CLS][CLS]'+dat['LSEQ']
X=dat['HL']
X = X.apply(insert_space_every_other_except_cls)
sequences = X.str.replace('  ', ' ')
#sequences = X.values
max_length = 512-2



#--------------------------------------------
# Load the model
antiberty = AntiBERTyRunner()

##--------------------------------------------
## Embedding the sequences
start_time = time.time()
batch_size = 500
n_seqs = len(sequences)
dim = 512

n_batches = math.ceil(n_seqs / batch_size)
embeddings = torch.empty((n_seqs, dim))

i = 1
for start, end, batch in batch_loader(sequences, batch_size):
    print(f'Batch {i}/{n_batches}\n')
    x = antiberty.embed(batch)
    x = [a.mean(axis = 0) for a in x]
    embeddings[start:end] = torch.stack(x)
    i += 1

end_time = time.time()
print(end_time - start_time)

#torch.save(embeddings, args.output_file)

px = pd.DataFrame(embeddings).astype("float")
px.index=dat.index
#px.to_csv('C:/Users/Hoan.Nguyen/Bioinformatics/AntigenDB/datasources/ipi_data/ipi_antibodydb.antiberta2.emb.csv')
px.to_csv(input_seq+".antiberty.emb.csv")



