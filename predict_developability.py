#!/usr/bin/env python3
"""
IPI Antibody Developability Prediction Platform
Final Production Version — DEC-2025
Supports: SEC & PSR | XGBoost & RF & CNN | ablang, antiberty, antiberta2, antiberta2-cssp
"""

from config import MODEL_DIR, PREDICTION_DIR
import argparse
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Local imports
from embedding_generator import generate_embedding
from models.xgboost import XGBoostModel
from models.randomforest import RandomForestModel
from models.cnn import CNNClassifier  


# ========================= CONFIG =========================
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

def get_default_db_path():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return None
    files = [f for f in os.listdir(data_dir) if f.startswith("ipi_antibodydb") and f.endswith(".xlsx")]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
    return os.path.join(data_dir, files[0])

# ========================= LOAD DATA =========================
def load_data(db_path, lm="antiberta2", label_col="sec_filter"):
    print(f"\nLoading database: {os.path.basename(db_path)}")
    print(f"Target: {label_col} | Embedding: {lm}")

    df = pd.read_excel(db_path)
    required = ['BARCODE', 'HSEQ', 'LSEQ', label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if 'antigen' in df.columns:
        df = df[~df['antigen'].str.contains('test', na=False, case=False)]
    df = df.dropna(subset=required).set_index('BARCODE')

    if label_col == "sec_filter" and 'psr_filter' in df.columns:
        df = df[(df['psr_filter'] == 1) | (df['sec_filter'] == 0)]

    possible = [
        f"data/ipi_antibodydb.xlsx.{lm}.emb.csv",
        f"{Path(db_path).stem}.{lm}.emb.csv"
    ]
    emb_file = next((f for f in possible if os.path.exists(f)), None)
    if not emb_file:
        print(f"Embedding not found → generating {lm}...")
        emb_file = generate_embedding(db_path, lm=lm)

    embedding = pd.read_csv(emb_file, index_col=0)
    common = df.index.intersection(embedding.index)
    if len(common) == 0:
        raise ValueError("No overlapping BARCODEs!")

    X = embedding.loc[common].values
    y = df.loc[common, label_col].values
    data = df.loc[common].copy()

    print(f"Training set: {len(X)} samples × {X.shape[1]} features")
    print(f"Positive rate: {y.mean():.1%}")
    return embedding, data, X, y

# ========================= PREDICTION =========================
def auto_predict(input_excel, target="sec_filter", lm="antiberta2", model_type="xgboost"):
    print(f"\nPREDICTING: {os.path.basename(input_excel)}")
    print(f"Target: {target.upper()} | Model: {model_type.upper()} | LM: {lm}")

    emb_file = f"{input_excel}.{lm}.emb.csv"
    if not os.path.exists(emb_file):
        print("Generating embedding...")
        generate_embedding(input_excel, lm=lm)

    data = pd.read_excel(input_excel)
    if 'BARCODE' not in data.columns:
        data['BARCODE'] = data.index
    data = data.set_index('BARCODE')

    embedding = pd.read_csv(emb_file, index_col=0)
    common = data.index.intersection(embedding.index)
    X = embedding.loc[common].values
    data = data.loc[common]

    # Load correct model file
    if model_type == "cnn":
        model_path = f"{MODEL_DIR}/FINAL_{target}_{lm}_cnn.pt"
    else:
        model_path = f"{MODEL_DIR}/FINAL_{target}_{lm}_{model_type}.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if model_type == "xgboost":
        model = XGBoostModel.load(model_path)
    elif model_type == "rf":
        model = RandomForestModel.load(model_path)
    elif model_type == "cnn":
        model = CNNClassifier.load(model_path, embedding_dim=X.shape[1])

    scores = model.predict_proba(X)
    labels = (scores >= 0.5).astype(int)

    data[f"{model_type}_{lm}_score"] = scores
    data[f"{model_type}_{lm}_label"] = labels

    path = Path(input_excel)
    if path.suffix.lower() in ['.xlsx', '.xls']:
        data.reset_index().to_excel(input_excel, index=False)
    else:
        data.reset_index().to_csv(input_excel, index=False)

    print(f"Saved predictions to: {input_excel} (updated)")
    print(f"Positive rate: {labels.mean():.1%}\n")

# ========================= MAIN =========================
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", type=str)
    group.add_argument("--build-embedding", type=str)
    group.add_argument("--kfold", type=int)
    group.add_argument("--train", action="store_true")

    parser.add_argument("--target", choices=["sec_filter", "psr_filter"], default="sec_filter")
    parser.add_argument("--lm", default="antiberta2", choices=["ablang", "antiberty", "antiberta2", "antiberta2-cssp", "all"])
    parser.add_argument("--model", default="xgboost", choices=["xgboost", "rf", "cnn"])
    parser.add_argument("--db", type=str)

    args = parser.parse_args()

    db_path = args.db or get_default_db_path()
    if (args.kfold or args.train) and not db_path:
        parser.error("No training database found!")

    if args.build_embedding:
        lms = ["ablang", "antiberty", "antiberta2", "antiberta2-cssp"] if args.lm == "all" else [args.lm]
        for lm in lms:
            generate_embedding(args.build_embedding, lm=lm)
        return

    if args.kfold:
        _, data, X, y = load_data(db_path, lm=args.lm, label_col=args.target)
        if 'HCDR3_CLUSTER_0.8' not in data.columns and 'CDR3' in data.columns:
            from utils.clustering import greedy_clustering_by_levenshtein
            data['HCDR3_CLUSTER_0.8'] = greedy_clustering_by_levenshtein(data['CDR3'].tolist(), 0.8)

        title = f"{args.target.upper()}_{args.lm}"

        if args.model == "xgboost":
            XGBoostModel.kfold_validation(data, X, y, embedding_lm=args.lm, title=title, kfold=args.kfold)
        elif args.model == "rf":
            RandomForestModel.kfold_validation(data, X, y, embedding_lm=args.lm, title=title, kfold=args.kfold)
        elif args.model == "cnn":
            CNNClassifier.kfold_validation(data, X, y, embedding_lm=args.lm, title=title, kfold=args.kfold)
        return

    if args.train:
        _, _, X, y = load_data(db_path, lm=args.lm, label_col=args.target)
        if args.model == "xgboost":
            model = XGBoostModel()
        elif args.model == "rf":
            model = RandomForestModel()
        elif args.model == "cnn":
            model = CNNClassifier()

        model.train(X, y)
        
        ext = ".pt" if args.model == "cnn" else ".pkl"
        path = f"{MODEL_DIR}/FINAL_{args.target}_{args.lm}_{args.model}{ext}"
        model.save(path)
        print(f"FINAL MODEL SAVED: {path}")
        return

    if args.predict:
        auto_predict(args.predict, target=args.target, lm=args.lm, model_type=args.model)

if __name__ == "__main__":
    main()