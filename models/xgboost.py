"""
IPI Antibody Developability Prediction Platform
Final Production Version — DEC-2025 
Supports: SEC & PSR | XGBoost & RF & CNN , Transformers | ablang, antiberty, antiberta2, antiberta2-cssp
"""



# models/xgboost.py
from config import MODEL_DIR

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc, f1_score, accuracy_score,roc_auc_score,
    precision_score, recall_score, confusion_matrix
)

from scipy.stats import uniform, randint

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.best_params_ = None

    def train(self, X, y, n_iter=50, cv=5, scoring='roc_auc', random_state=42):
        print("Training XGBoost with RandomizedSearchCV...")

        class_counts = pd.Series(y).value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1

        xgb_base = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=random_state
        )

        param_dist = {
            'n_estimators': randint(3000, 6000),
            'max_depth': randint(4, 12),
            'learning_rate': uniform(0.005, 0.095),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'min_child_weight': randint(1, 10),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0.1, 2),
        }

        search = RandomizedSearchCV(
            xgb_base, param_dist, n_iter=n_iter, cv=cv,
            scoring=scoring, n_jobs=-1, random_state=random_state, verbose=0
        )
        search.fit(X, y)

        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        print(f"Best CV {scoring}: {search.best_score_:.4f}")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"XGBoost model saved → {path}")

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.model = joblib.load(path)
        return instance

    # ===================================================================
    # XGBOOST K-FOLD CV — CLASS METHOD
    # ===================================================================
    @classmethod
    def kfold_validation(cls, data, X, y, embedding_lm='antiberta2', title="XGB_SEC", kfold=10,target="sec_filter"):
        y = np.array(y)
        test_f1 = np.zeros(kfold)
        test_accuracy = np.zeros(kfold)
        test_precision = np.zeros(kfold)
        test_recall = np.zeros(kfold)
        test_auc = np.zeros(kfold)

        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
        fig = plt.figure(figsize=[6, 5])
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        print(f"\nXGBoost {kfold}-Fold CV | {title} | {embedding_lm}")
        print(f"{'Fold':<6} {'AUC':<8} {'Acc':<8} {'F1':<8} {'Rec':<8} {'Prec'}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, data['HCDR3_CLUSTER_0.8']), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # === SAVE BARCODES FOR THIS FOLD ===
            train_df = data.iloc[train_idx].copy()
            val_df   = data.iloc[test_idx].copy()
            train_df.to_csv(f"{MODEL_DIR}/xgboost_{target}_{embedding_lm}_fold{fold}_train.csv")
            val_df.to_csv(f"{MODEL_DIR}/xgboost_{target}_{embedding_lm}_fold{fold}_val.csv")
            print(f"  Fold {fold} → saved full TRAIN ({len(train_df)}) + VAL ({len(val_df)}) rows") 
            
            
            model = cls()
            model.train(X_train, y_train, n_iter=30, scoring='roc_auc')

            prob = model.predict_proba(X_test)
            pred = model.predict(X_test)

            test_auc[fold-1] = roc_auc_score(y_test, prob)
            test_f1[fold-1] = f1_score(y_test, pred)
            test_accuracy[fold-1] = accuracy_score(y_test, pred)
            test_precision[fold-1] = precision_score(y_test, pred, zero_division=0)
            test_recall[fold-1] = recall_score(y_test, pred, zero_division=0)

            fpr, tpr, _ = roc_curve(y_test, prob)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            fold_auc = auc(fpr, tpr)
            aucs.append(fold_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold} ({fold_auc:.3f})')

            print(f"{fold:<6} {test_auc[fold-1]:.4f}   {test_accuracy[fold-1]:.4f}   "
                  f"{test_f1[fold-1]:.4f}   {test_recall[fold-1]:.4f}   {test_precision[fold-1]:.4f}")

            save_path = f"SEC/models_final/xgboost_{target}_{embedding_lm}_fold{fold}.pkl"
            model.save(save_path)

        # Mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, 'b', lw=3, label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        plt.fill_between(mean_fpr, mean_tpr - np.std(tprs, axis=0), mean_tpr + np.std(tprs, axis=0),
                         color='lightblue', alpha=0.3)
        plt.plot([0,1],[0,1], '--', color='gray')
        plt.xlim([0,1]); plt.ylim([0,1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'{title} | {kfold}-Fold CV | {embedding_lm}', fontsize=10)
        plt.legend(loc='lower right', fontsize=8)
        plt.grid(alpha=0.3)

        plot_path = f"SEC/models_final/xgboost_{target}_{embedding_lm}_k{kfold}_roc.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\nMean AUC: {mean_auc:.4f} ± {std_auc:.3f}")
        print(f"Plot saved: {plot_path}")