# This is prototype of PSR prediction based Antibody language model and Deep Learning Neural Network
#  Main Workflow:
#        1.  Generating embedding vector spaces for all IPI VH+VL sequences using  Antibody Language Model-ABLang2 (pretrained)
#        2.  Unsupervised clustering of sequence embedding vectors using PCA and K-mean.
#        3.  Build deeplearning model with  sequence emdeeding vectors
#        4.  Perform 10-Folds validation to estimate the prediction performance.
#        5.  Need more checking and cleaning on PSR trainninset and their PSR_FILTER label 
#        6.  Test with public data


#pip install scipy==1.11.4
import scipy 
from interpolation import interp
from sgt import SGT
import numpy as np
import pandas as pd
from itertools import chain
from itertools import product as iterproduct
import warnings
import pickle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics
import time
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import matplotlib.patches as patches
import numpy as np
import torch
import ablang2
np.random.seed(7) # fix random seed for reproducibility
from decimal import Decimal, getcontext
getcontext().prec = 30
import argparse
main_path="/Users/Hoan.Nguyen/ComBio/AntigenDB/datasources/ipi_data"

os.chdir(main_path)

def predict_prob(number):
  return [number[0],1-number[0]]

parser = argparse.ArgumentParser(description="Generate ABlang2 embedding. Example usage: \n python Ablange_embedding.py sequence_file")
parser.add_argument("sequence", type=str, help="Path to the antibody sequence  file.")
#parser.add_argument("output_file", type=str, help="Output file path")
args = parser.parse_args()
input_seq=args.sequence
#data =pd.read_excel('/Users/Hoan.Nguyen/ComBio/AntigenDB/datasources/ipi_data/ipi_antibodydb.xlsx')
#data =pd.read_excel('/Users/Hoan.Nguyen/ComBio/AntigenDB/datasources/ipi_data/IPI_Miseq77_PSR.xlsx')

data =pd.read_excel(input_seq)
data=data[pd.notna(data['LSEQ'])]
data=data[pd.notna(data['HSEQ'])]

ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=3, device='cpu')

data['VHVL']=''
data['ablang_emb']=''

for i in data.index:
   #data["VHVL"].loc[i]=[data['HSEQ'].loc[i]]
   data["VHVL"].loc[i]=[data['HSEQ'].loc[i],data['LSEQ'].loc[i]]
   seq= data["VHVL"].loc[i]
   print(i)
   try:
      print(i)
      data['ablang_emb'].loc[i]=ablang(seq, mode='seqcoding')[0]
   except:
      pass



ablang_emb=pd.DataFrame(data['ablang_emb_heavy'].to_list())
#ablang_emb=ablang_emb.rename(columns={"BARCODE": "id"})
ablang_emb.index=data['BARCODE']
ablang_emb.to_csv(input_seq+".ablang.emb.csv")

