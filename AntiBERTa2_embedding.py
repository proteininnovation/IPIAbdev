#--------------------------------------------
# Generate antiBERTa2 embedding
# 
# AntiBERTa2 is available on huggingface:
# https://huggingface.co/alchemab/antiberta2
#
# Example usage: 
#
# python antiBERTa2_embedding.py sequences_file.xlsx 
#--------------------------------------------

import argparse
from datetime import date
import os
import warnings
warnings.simplefilter('ignore')
import torch
from transformers import (
    RoFormerModel,
    RoFormerForMaskedLM, 
    RoFormerTokenizer, 
    pipeline, 
    RoFormerForSequenceClassification
)
from Bio import SeqIO
import time
import math 
import pandas as pd
from decimal import Decimal, getcontext
getcontext().prec = 30
##--------------------------------------------
parser = argparse.ArgumentParser(description="Generate antiBERTa2 embedding. Example usage: \n python antiBERTa2_embed.py sequence_file output")
parser.add_argument("sequence", type=str, help="Path to the antibody sequence  file.")
parser.add_argument("output_file", type=str, help="Output file path")
args = parser.parse_args()

##--------------------------------------------
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

##--------------------------------------------
## Format the sequence


#dat =pd.read_excel('C:/Users/Hoan.Nguyen/Bioinformatics/MachineLearning/IPIdata_annotation/IPI_Miseq77_PSR.xlsx')
#dat =pd.read_excel('C:/Users/Hoan.Nguyen/Bioinformatics/AntigenDB/datasources/ipi_data/ipi_antibodydb.xlsx')

#dat =pd.read_excel('/Users/Hoan.Nguyen/ComBio/MachineLearning/Test/seq.xlsx')

#h='QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARWSYESDDFDYWGQGTLVTVSS'
#l='DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPYTFGQGTKLEIK'
#data = {'HSEQ': [h], 'LSEQ': [l]}
#dat = pd.DataFrame(data)


input_seq=args.sequence
dat = pd.read_excel(input_seq)
dat=dat[pd.notna(dat['LSEQ'])]
dat=dat[pd.notna(dat['HSEQ'])]
dat=dat.set_index('BARCODE')
dat['HL']=dat['HSEQ']+'[CLS][CLS]'+dat['LSEQ']
X=dat['HL']
X = X.apply(insert_space_every_other_except_cls)
X = X.str.replace('  ', ' ')
sequences = X.values
max_length=256

## ANTIBERTA2-CSSP
##--------------------------------------------
## Set up the model

tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2-cssp")
model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2-cssp")
#model = model.to('cuda')
model_size = sum(p.numel() for p in model.parameters())
print(f"Model loaded. Size: {model_size/1e6:.2f}M")

##--------------------------------------------
## Embedding the sequences
start_time = time.time()
batch_size = 128
n_seqs = len(sequences)
dim = 1024
n_batches = math.ceil(n_seqs / batch_size)
embeddings = torch.empty((n_seqs, dim))

i = 1
for start, end, batch in batch_loader(sequences, batch_size):
    print(f'Batch {i}/{n_batches}\n')
    x = torch.tensor([
    tokenizer.encode(seq, 
                     padding="max_length",
                     truncation=True,
                     max_length=max_length,
                     #return_special_tokens_mask=True) for seq in batch]).to('cuda')
                     return_special_tokens_mask=True) for seq in batch])
    
    attention_mask = (x != tokenizer.pad_token_id).float()
    #attention_mask = (x != tokenizer.pad_token_id).float().to('cuda')
    with torch.no_grad():
        outputs = model(x, attention_mask = attention_mask,
                       output_hidden_states = True)
        outputs = outputs.hidden_states[-1]
        outputs = list(outputs.detach())
    
    # aggregate across the residuals, ignore the padded bases
    for j, a in enumerate(attention_mask):
        outputs[j] = outputs[j][a == 1,:].mean(0)
        
    embeddings[start:end] = torch.stack(outputs)
    del x
    del attention_mask
    del outputs
    i += 1

#input_emb=pd.DataFrame(embeddings.tolist()[0]).transpose()
end_time = time.time()
print(end_time - start_time)

torch.save(embeddings, input_seq+".antiberta2-cssp.emb.tensor.pt")
px = pd.DataFrame(embeddings).astype("float")
px.index=dat.index
px.to_csv(input_seq+".antiberta2-cssp.emb.csv")


##----ANTIBERTA2
## Set up the model
tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2")
model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2")

#model = model.to('cuda')
model_size = sum(p.numel() for p in model.parameters())
print(f"Model loaded. Size: {model_size/1e6:.2f}M")

##--------------------------------------------
## Embedding the sequences
start_time = time.time()
batch_size = 128
n_seqs = len(sequences)
dim = 1024
n_batches = math.ceil(n_seqs / batch_size)
embeddings = torch.empty((n_seqs, dim))

i = 1
for start, end, batch in batch_loader(sequences, batch_size):
    print(f'Batch {i}/{n_batches}\n')
    x = torch.tensor([
    tokenizer.encode(seq, 
                     padding="max_length",
                     truncation=True,
                     max_length=max_length,
                     #return_special_tokens_mask=True) for seq in batch]).to('cuda')
                     return_special_tokens_mask=True) for seq in batch])
    
    attention_mask = (x != tokenizer.pad_token_id).float()
    #attention_mask = (x != tokenizer.pad_token_id).float().to('cuda')
    with torch.no_grad():
        outputs = model(x, attention_mask = attention_mask,
                       output_hidden_states = True)
        outputs = outputs.hidden_states[-1]
        outputs = list(outputs.detach())
    
    # aggregate across the residuals, ignore the padded bases
    for j, a in enumerate(attention_mask):
        outputs[j] = outputs[j][a == 1,:].mean(0)
        
    embeddings[start:end] = torch.stack(outputs)
    del x
    del attention_mask
    del outputs
    i += 1
    
end_time = time.time()
print(end_time - start_time)


torch.save(embeddings, input_seq+".antiberta2.emb.tensor.pt")
px = pd.DataFrame(embeddings).astype("float")
px.index=dat.index
px.to_csv(input_seq+".antiberta2.emb.csv")

