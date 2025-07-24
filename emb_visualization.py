
# t-SNE
import plotly.express as px
import kaleido
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import torch
import ablang2
import pandas as pd
import kaleido
np.random.seed(42) # fix random seed for reproducibility
main_path="/Users/Hoan.Nguyen/ComBio/MachineLearning"
os.chdir(main_path)
from sklearn.preprocessing import LabelEncoder

###SPR
def load_full_ipp_dataset_sec(emb_matrix_file:str):

    #data=pd.read_excel("IPIdata_annotation/IPI_AB_MULTIPLE_LEVEL_ANNO_ABLANG_EMBEDDING_PSR_RF_ELISA_NOV.xlsx")
    #sprtest=pd.read_excel("IPIdata_annotation/Library3.1_All_Antibody_Characterization.xlsx")
    #data=data[~data['BARCODE'].isin(sprtest['TAB-ID'])].reset_index( drop=True)
    data=pd.read_excel("data/ipi_antibodydb.xlsx")
    name=os.path.basename(emb_matrix_file)
    data_embedding=pd.DataFrame()
    if (".xlsx" in name):
            data_embedding=pd.read_excel(emb_matrix_file)
    else:
        if (".csv" in name):
            data_embedding=pd.read_csv(emb_matrix_file)

    data_embedding=data_embedding.set_index("BARCODE")
    return data,data_embedding
# load NGS_PSR
def load_full_ipp_dataset_spr(emb_matrix_file:str):
    data=pd.read_excel("data/ipi_antibodydb.xlsx")
    #sprtest=pd.read_excel("IPIdata_annotation/Library3.1_All_Antibody_Characterization.xlsx")
    #data=data[~data['BARCODE'].isin(sprtest['TAB-ID'])].reset_index( drop=True)
    
    data=data[pd.notna(data['SPR Annot'])]
    data['SPR_FILTER']=''
    data.loc[data['SPR Annot'].str.contains('premium|strong|weak'), 'SPR_FILTER'] = 1
    data.loc[data['SPR Annot'].str.contains('fail'), 'SPR_FILTER'] = 0
    
    name=os.path.basename(emb_matrix_file)
    data_embedding=pd.DataFrame()
    if (".xlsx" in name):
            data_embedding=pd.read_excel(emb_matrix_file)
    else:
        if (".csv" in name):
            data_embedding=pd.read_csv(emb_matrix_file)

    data_embedding=data_embedding.set_index("BARCODE")
    return data,data_embedding


# all ipi with PSR_PASS_insulin>0.5, without_ELISA_RF_WRONG_PRED and ngs_psr
def get_trainset_spr(data:pd.DataFrame, data_embedding:pd.DataFrame):
    data_psr=data
    ablang_emb=data_embedding

    data_psr=data_psr[pd.notna(data_psr['SPR Annot'])]
    data_psr['SPR_FILTER']=''
    data_psr.loc[data_psr['SPR Annot'].str.contains('premium|strong|weak'), 'SPR_FILTER'] = 1
    data_psr.loc[data_psr['SPR Annot'].str.contains('fail'), 'SPR_FILTER'] = 0

    id=set(data_psr['BARCODE']).intersection(ablang_emb.index)
    ablang_emb=ablang_emb.iloc[pd.Index(ablang_emb.index).get_indexer(id)]
    data_psr=data_psr.iloc[pd.Index(data_psr['BARCODE']).get_indexer(id)]

  
    #ablang_emb=ablang_emb.iloc[pd.Index(ablang_emb.index).get_indexer(data_psr['BARCODE'])]


    X=ablang_emb
    y=data_psr['SPR_FILTER'].to_list()

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    return data_psr,ablang_emb,X,y


# all ipi with PSR_PASS_insulin>0.5, without_ELISA_RF_WRONG_PRED and ngs_psr
def get_trainset_sec(data:pd.DataFrame, data_embedding:pd.DataFrame):
    data_psr=data
    data_psr=data_psr[pd.notna(data_psr['sec_filter'])]
    ablang_emb=data_embedding
    ablang_emb=ablang_emb.iloc[pd.Index(ablang_emb.index).get_indexer(data_psr['BARCODE'])]

    X=ablang_emb
    y=data_psr['sec_filter'].to_list()

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    return data_psr,ablang_emb,X,y

##SPR



def load_full_ipp_dataset(emb_matrix_file:str):
    #data=pd.read_excel("data/IPI_AB_MULTIPLE_LEVEL_ANNO_ABLANG_EMBEDDING_PSR_RF_ELISA_NOV.xlsx")
    data=pd.read_excel("data/ipi_antibodydb.xlsx")
    data=data[pd.notna(data['psr_filter'])]
    data=data[pd.notna(data['psr_norm_insulin']) & pd.notna(data['psr_norm_dna']) & pd.notna(data['psr_norm_smp']) & pd.notna(data['psr_norm_avidin'])  ]
    data['psr_mean']=(data['psr_norm_insulin']+data['psr_norm_dna']+ data['psr_norm_smp']+ data['psr_norm_avidin'])/4
   
    
    name=os.path.basename(emb_matrix_file)
    data_embedding=pd.DataFrame()
    if (".xlsx" in name):
            data_embedding=pd.read_excel(emb_matrix_file)
    else:
        if (".csv" in name):
            data_embedding=pd.read_csv(emb_matrix_file)
    
    data_embedding=data_embedding.set_index("BARCODE")


    id=set(data['BARCODE']).intersection(data_embedding.index)
    data_embedding=data_embedding.iloc[pd.Index(data_embedding.index).get_indexer(id)]
    data=data.iloc[pd.Index(data['BARCODE']).get_indexer(id)]

    return data,data_embedding


# all ipi with PASS_norm_insulin >0.5 and only tabid>87000 , without_ELISA_RF_WRONG_PRED and ngs_psr
def get_trainset2_psr(data:pd.DataFrame, data_embedding:pd.DataFrame):
    data_psr=data

    pos_charge_filter=data_psr[(data_psr['CDR3_charge_value2']>2) & (data_psr['psr_filter']==1)]
    data_psr=data_psr[~data_psr['BARCODE'].isin(pos_charge_filter['BARCODE'])].reset_index( drop=True)
    pos_charge_filter2=data_psr[((data_psr['CDR3_charge_value2']<0) & (data_psr['psr_filter']==0) & (data_psr['psr_mean']<1))]
    data_psr=data_psr[~data_psr['BARCODE'].isin(pos_charge_filter2['BARCODE'])].reset_index( drop=True)
    pos_charge_filter3=data_psr[(data_psr['psr_filter']==1) & (data_psr['psr_norm_insulin']>0.3) & (data_psr['CDR3_charge_value2']>1)]
    data_psr=data_psr[~data_psr['BARCODE'].isin(pos_charge_filter3['BARCODE'])].reset_index( drop=True)
    data_psr=data_psr[data_psr['ID2']>8763]
    # remove ELISA_RF_WRONG_PRED ('diff_pred'==0)
    #data_psr=data_psr[data_psr['diff_pred']==1]

    ablang_emb=data_embedding
    ablang_emb=ablang_emb.iloc[pd.Index(ablang_emb.index).get_indexer(data_psr['BARCODE'])]

    X=ablang_emb
    y=data_psr['psr_filter'].to_list()
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    return data_psr,ablang_emb,X,y
  
def CREATE_TNSE_SPR(embedding_lm:str='ablang',label='heavy',mytitle='IPI SPR : '):

    data, emb=load_full_ipp_dataset_spr("data/ipi_antibodydb."+embedding_lm+".emb.csv")
    
    #data=data[data['UniProt_Name'].str.contains('SEM')]
    
    data_psr,emb,X,y=get_trainset_spr(data,emb)

    tsne = TSNE(n_components=2,perplexity=5,learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    tsne.kl_divergence_
    data_psr[label]=data_psr[label].astype(str)
    data_psr[label]= data_psr[label].replace("1.0","PASS")
    data_psr[label]= data_psr[label].replace("1","PASS")
    data_psr[label]= data_psr[label].replace("0.0","FAIL")
    data_psr[label]= data_psr[label].replace("0","FAIL")

    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=data_psr[label])
    fig.update_layout(
        title=mytitle+embedding_lm+" embedding",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
        legend = dict(font = dict(family = "Courier", size = 20) ,title=label),
        legend_title = dict(font = dict(family = "Courier", size = 20, color = "blue"))
    )
    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    #fig.show()
    fig.write_html("embedding/images/IPI_SPR_trainningset2_"+embedding_lm+"_"+label+".html")
    fig.write_image("embedding/images/IPI_SPR_trainningset2_"+embedding_lm+"_"+label+".png")


def CREATE_TNSE_SEC(embedding_lm:str='ablang',label='heavy',mytitle='IPI SEC : '):

    data, emb=load_full_ipp_dataset_sec("data/ipi_antibodydb."+embedding_lm+".emb.csv")
    
    
    data_psr,emb,X,y=get_trainset_sec(data,emb)

    tsne = TSNE(n_components=2,perplexity=5,learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    tsne.kl_divergence_
    data_psr[label]=data_psr[label].astype(str)
    data_psr[label]= data_psr[label].replace("1.0","PASS")
    data_psr[label]= data_psr[label].replace("1","PASS")
    data_psr[label]= data_psr[label].replace("0.0","FAIL")
    data_psr[label]= data_psr[label].replace("0","FAIL")

    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=data_psr[label])
    fig.update_layout(
        title=mytitle+embedding_lm+" embedding",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
        legend = dict(font = dict(family = "Courier", size = 20) ,title=label),
        legend_title = dict(font = dict(family = "Courier", size = 20, color = "blue"))
    )
    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    #fig.show()
    fig.write_html("/Users/Hoan.Nguyen/ComBio/MachineLearning/embedding/images/IPI_Sec_trainningset2_"+embedding_lm+"_"+label+".html")
    fig.write_image("/Users/Hoan.Nguyen/ComBio/MachineLearning/embedding/images/IPI_Sec_trainningset2_"+embedding_lm+"_"+label+".png")


def CREATE_TNSE_PSR(embedding_lm:str='ablang',label='heavy', mytilte='IPI PSR-trainning-set2: '):

    data, emb=load_full_ipp_dataset("data/ipi_antibodydb."+embedding_lm+".emb.csv")
    #data=data[data['UniProt_Name'].str.contains('SEM')]
    data_psr,emb,X,y=get_trainset2_psr(data,emb)

    tsne = TSNE(n_components=2,perplexity=7,learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    tsne.kl_divergence_
    data_psr[label]=data_psr[label].astype(str)
    data_psr[label]= data_psr[label].replace("1.0","PASS")
    data_psr[label]= data_psr[label].replace("1","PASS")
    data_psr[label]= data_psr[label].replace("0.0","FAIL")
    data_psr[label]= data_psr[label].replace("0","FAIL")

    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=data_psr[label])
    fig.update_layout(
        title=mytilte+embedding_lm+" embedding",
        xaxis_title="First t-SNE",
        yaxis_title="Second t-SNE",
        legend = dict(font = dict(family = "Courier", size = 20) ,title=label),
        legend_title = dict(font = dict(family = "Courier", size = 20, color = "blue"))
    )
    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    #fig.show()
    fig.write_html("/Users/Hoan.Nguyen/ComBio/NGS/Projects/SEMA/embedding/images/IPI_PSR_trainningset2_"+embedding_lm+"_"+label+".html")
    fig.write_image("/Users/Hoan.Nguyen/ComBio/NGS/Projects/SEMA/embedding/images/IPI_PSR_trainningset2_"+embedding_lm+"_"+label+".png")

def CREATE_UMAP_PSR(embedding_lm:str='ablang',label='heavy', mytitle='IPI PSR-trainning-set2: '):

    data, emb=load_full_ipp_dataset("data/ipi_antibodydb."+embedding_lm+".emb.csv")
    #data_psr,emb,X,y=get_trainset2_psr(data,emb)
    data_psr=data
    X=emb
    print(data.shape)
    print(X.shape)

    # Step 1: Standardize the embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(X)
    # Step 2: Apply UMAP for dimensionality reduction
    umap_reducer = umap.UMAP(n_components=2, random_state=42,n_neighbors=15, min_dist=0.1,n_epochs=100,learning_rate=2)
    umap_embeddings = umap_reducer.fit_transform(scaled_embeddings)

    data_psr[label]=data_psr[label].astype(str)
    data_psr[label]= data_psr[label].replace("1.0","PASS")
    data_psr[label]= data_psr[label].replace("1","PASS")
    data_psr[label]= data_psr[label].replace("0.0","FAIL")
    data_psr[label]= data_psr[label].replace("0","FAIL")

    fig = px.scatter(x=umap_embeddings [:, 0], y=umap_embeddings [:, 1], color=data_psr[label])
    fig.update_layout(
        title=mytitle+embedding_lm+" embedding",
        xaxis_title="Umap Dim 1",
        yaxis_title="Umap Dim 2",
        legend = dict(font = dict(family = "Courier", size = 20) ,title=label),
        legend_title = dict(font = dict(family = "Courier", size = 20, color = "blue"))
    )
    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    #fig.update_layout(width=400, height=400)
    #fig.show()
    fig.write_html("embedding/images/IPI_PSR_UMAP_"+embedding_lm+"_"+label+".html")
    #fig.write_image("embedding/images/IPI_PSR_UMAP_"+embedding_lm+"_"+label+".png")


def CREATE_UMAP_SEC(embedding_lm:str='ablang',label='heavy', mytitle='IPI PSR-trainning-set2: '):

    data, emb=load_full_ipp_dataset_sec("data/ipi_antibodydb."+embedding_lm+".emb.csv")
    #data=data[data['UniProt_Name'].str.contains('SEM')]
    data_psr,emb,X,y=get_trainset_sec(data,emb)

    # Step 1: Standardize the embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(X)
    # Step 2: Apply UMAP for dimensionality reduction
    umap_reducer = umap.UMAP(n_components=2, random_state=42,n_neighbors=15, min_dist=0.1,n_epochs=100,learning_rate=2)
    umap_embeddings = umap_reducer.fit_transform(scaled_embeddings)

    data_psr[label]=data_psr[label].astype(str)
    data_psr[label]= data_psr[label].replace("1.0","PASS")
    data_psr[label]= data_psr[label].replace("1","PASS")
    data_psr[label]= data_psr[label].replace("0.0","FAIL")
    data_psr[label]= data_psr[label].replace("0","FAIL")

    fig = px.scatter(x=umap_embeddings [:, 0], y=umap_embeddings [:, 1], color=data_psr[label])
    fig.update_layout(
        title=mytitle+embedding_lm+" embedding",
        xaxis_title="Umap Dim 1",
        yaxis_title="Umap Dim 2",
        legend = dict(font = dict(family = "Courier", size = 20) ,title=label),
        legend_title = dict(font = dict(family = "Courier", size = 20, color = "blue"))
    )
    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    #fig.update_layout(width=400, height=400)
    #fig.show()
    fig.write_html("embedding/images/IPI_SEC_UMAP_"+embedding_lm+"_"+label+".html")
    #fig.write_image("embedding/images/IPI_SEC_UMAP_"+embedding_lm+"_"+label+".png")


def CREATE_UMAP_SPR(embedding_lm:str='ablang',label='heavy', mytitle='IPI PSR-trainning-set2: '):

    data, emb=load_full_ipp_dataset_spr("data/ipi_antibodydb."+embedding_lm+".emb.csv")
    #data=data[data['UniProt_Name'].str.contains('SEM')]
    data_psr,emb,X,y=get_trainset_spr(data,emb)

    # Step 1: Standardize the embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(X)
    # Step 2: Apply UMAP for dimensionality reduction
    umap_reducer = umap.UMAP(n_components=2, random_state=42,n_neighbors=15, min_dist=0.1,n_epochs=100,learning_rate=2)
    umap_embeddings = umap_reducer.fit_transform(scaled_embeddings)

    data_psr[label]=data_psr[label].astype(str)
    data_psr[label]= data_psr[label].replace("1.0","PASS")
    data_psr[label]= data_psr[label].replace("1","PASS")
    data_psr[label]= data_psr[label].replace("0.0","FAIL")
    data_psr[label]= data_psr[label].replace("0","FAIL")

    fig = px.scatter(x=umap_embeddings [:, 0], y=umap_embeddings [:, 1], color=data_psr[label])
    fig.update_layout(
        title=mytitle+embedding_lm+" embedding",
        xaxis_title="Umap Dim 1",
        yaxis_title="Umap Dim 2",
        legend = dict(font = dict(family = "Courier", size = 20) ,title=label),
        legend_title = dict(font = dict(family = "Courier", size = 20, color = "blue"))
    )
    fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
    #fig.update_layout(width=400, height=400)
    #fig.show()
    fig.write_html("embedding/images/IPI_SPR_UMAP_"+embedding_lm+"_"+label+".html")
    #fig.write_image("embedding/images/IPI_SPR_UMAP_"+embedding_lm+"_"+label+".png")


if __name__ == "__main__":
     

     CREATE_TNSE_SPR(embedding_lm='ablang',label='heavy',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='ablang',label='light',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='ablang',label='SPR Annot',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='ablang',label='SPR_FILTER',mytitle='IPI Semaphorin : ')

     CREATE_TNSE_SPR(embedding_lm='antiberty',label='heavy',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='antiberty',label='light',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='antiberty',label='SPR Annot',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='antiberty',label='SPR_FILTER',mytitle='IPI Semaphorin : ')

     CREATE_TNSE_SPR(embedding_lm='antiberta2',label='heavy',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='antiberta2',label='light',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='antiberta2',label='SPR Annot',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='antiberta2',label='SPR_FILTER',mytitle='IPI Semaphorin : ')

     CREATE_TNSE_SPR(embedding_lm='antiberta2-cssp',label='heavy',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='antiberta2-cssp',label='light',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='antiberta2-cssp',label='SPR Annot',mytitle='IPI Semaphorin : ')
     CREATE_TNSE_SPR(embedding_lm='antiberta2-cssp',label='SPR_FILTER',mytitle='IPI Semaphorin : ')

    # PSR 

     CREATE_TNSE_PSR(embedding_lm='ablang',label='heavy',mytilte='IPI Semaphorin : ')
     CREATE_TNSE_PSR(embedding_lm='ablang',label='light',mytilte='IPI Semaphorin : ')
     CREATE_TNSE_PSR(embedding_lm='ablang',label='psr_filter',mytilte='IPI Semaphorin : ')



     CREATE_TNSE_PSR(embedding_lm='antiberty',label='heavy',mytilte='IPI Semaphorin : ')
     CREATE_TNSE_PSR(embedding_lm='antiberty',label='light',mytilte='IPI Semaphorin : ')
     CREATE_TNSE_PSR(embedding_lm='antiberty',label='psr_filter',mytilte='IPI Semaphorin : ')


     CREATE_TNSE_PSR(embedding_lm='antiberta2',label='heavy',mytilte='IPI Semaphorin : ')
     CREATE_TNSE_PSR(embedding_lm='antiberta2',label='light',mytilte='IPI Semaphorin : ')
     CREATE_TNSE_PSR(embedding_lm='antiberta2',label='psr_filter',mytilte='IPI Semaphorin : ')


     CREATE_TNSE_PSR(embedding_lm='antiberta2-cssp',label='heavy',mytilte='IPI Semaphorin : ')
     CREATE_TNSE_PSR(embedding_lm='antiberta2-cssp',label='light',mytilte='IPI Semaphorin : ')
     CREATE_TNSE_PSR(embedding_lm='antiberta2-cssp',label='psr_filter',mytilte='IPI Semaphorin : ')


    # SEC

     CREATE_TNSE_SEC(embedding_lm='ablang',label='heavy',mytitle='IPI SEC : ')
     CREATE_TNSE_SEC(embedding_lm='ablang',label='light',mytitle='IPI SEC : ')
     CREATE_TNSE_SEC(embedding_lm='ablang',label='sec_filter',mytilte='IPI SEC : ')


     CREATE_TNSE_SEC(embedding_lm='antiberty',label='heavy',mytitle='IPI SEC : ')
     CREATE_TNSE_SEC(embedding_lm='antiberty',label='light',mytitle='IPI SEC : ')
     CREATE_TNSE_SEC(embedding_lm='antiberty',label='sec_filter',mytitle='IPI SEC : ')


     CREATE_TNSE_SEC(embedding_lm='antiberta2',label='heavy',mytitle='IPI SEC : ')
     CREATE_TNSE_SEC(embedding_lm='antiberta2',label='light',mytitle='IPI SEC : ')
     CREATE_TNSE_SEC(embedding_lm='antiberta2',label='sec_filter',mytitle='IPI SEC : ')


     CREATE_TNSE_SEC(embedding_lm='antiberta2-cssp',label='heavy',mytitle='IPI SEC : ')
     CREATE_TNSE_SEC(embedding_lm='antiberta2-cssp',label='light',mytitle='IPI SEC : ')
     CREATE_TNSE_SEC(embedding_lm='antiberta2-cssp',label='sec_filter',mytitle='IPI SEC : ')



    #UMAP

     CREATE_UMAP_PSR(embedding_lm='ablang',label='heavy',mytitle='IPI PSR  : ')
     CREATE_UMAP_PSR(embedding_lm='antiberta2-cssp',label='heavy',mytitle='IPI PSR : ')
     CREATE_UMAP_PSR(embedding_lm='antiberta2',label='heavy',mytitle='IPI PSR : ')
     CREATE_UMAP_PSR(embedding_lm='antiberty',label='heavy',mytitle='IPI PSR : ')

     CREATE_UMAP_PSR(embedding_lm='ablang',label='light',mytitle='IPI PSR  : ')
     CREATE_UMAP_PSR(embedding_lm='antiberta2-cssp',label='light',mytitle='IPI PSR : ')
     CREATE_UMAP_PSR(embedding_lm='antiberta2',label='light',mytitle='IPI PSR : ')
     CREATE_UMAP_PSR(embedding_lm='antiberty',label='light',mytitle='IPI PSR : ')

     CREATE_UMAP_PSR(embedding_lm='ablang',label='psr_filter',mytitle='IPI PSR  : ')
     CREATE_UMAP_PSR(embedding_lm='antiberta2-cssp',label='psr_filter',mytitle='IPI PSR : ')
     CREATE_UMAP_PSR(embedding_lm='antiberta2',label='psr_filter',mytitle='IPI PSR : ')
     CREATE_UMAP_PSR(embedding_lm='antiberty',label='psr_filter',mytitle='IPI PSR : ')


     #CREATE_UMAP_PSR(embedding_lm='ablang',label='heavy',mytitle='IPI PSR-FILTERED  : ')
     #CREATE_UMAP_PSR(embedding_lm='antiberta2-cssp',label='heavy',mytitle='IPI PSR-FILTERED : ')
     #CREATE_UMAP_PSR(embedding_lm='antiberta2',label='heavy',mytitle='IPI PSR-FILTERED : ')
     #CREATE_UMAP_PSR(embedding_lm='antiberty',label='heavy',mytitle='IPI PSR-FILTERED : ')



     CREATE_UMAP_SEC(embedding_lm='ablang',label='heavy',mytitle='IPI SEC  : ')
     CREATE_UMAP_SEC(embedding_lm='antiberta2-cssp',label='heavy',mytitle='IPI SEC : ')
     CREATE_UMAP_SEC(embedding_lm='antiberta2',label='heavy',mytitle='IPI SEC : ')
     CREATE_UMAP_SEC(embedding_lm='antiberty',label='heavy',mytitle='IPI SEC : ')

     CREATE_UMAP_SPR(embedding_lm='ablang',label='heavy',mytitle='IPI SPR  : ')
     CREATE_UMAP_SPR(embedding_lm='antiberta2-cssp',label='heavy',mytitle='IPI SPR : ')
     CREATE_UMAP_SPR(embedding_lm='antiberta2',label='heavy',mytitle='IPI SPR : ')
     CREATE_UMAP_SPR(embedding_lm='antiberty',label='heavy',mytitle='IPI SPR: ')
