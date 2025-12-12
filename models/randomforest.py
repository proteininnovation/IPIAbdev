"""
IPI Antibody Developability Prediction Platform
Final Production Version — DEC-2025 
Supports: SEC & PSR | XGBoost & RF & CNN , Transformers | ablang, antiberty, antiberta2, antiberta2-cssp
"""



# models/randomforest.py
from config import MODEL_DIR
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score
from scipy.stats import randint
  

class RandomForestModel:
    def __init__(self):
        self.model = None
        self.best_params_ = None

    def train(self, X, y, n_iter=30, cv=5, scoring='roc_auc', random_state=42):
        print("  Training RF with RandomizedSearchCV...", end=" ")

        try:
            base_rf = RandomForestClassifier(
                n_estimators=2000,
                max_depth=8,
                min_samples_leaf=5,
                min_samples_split=10,
                max_features=0.5,
                class_weight='balanced',    
                criterion='entropy',
                bootstrap=True,
                random_state=random_state,
                n_jobs=-1
            )

            param_dist = {
                'n_estimators': [1000, 1500, 2000, 3000],
                'max_depth': [6, 8, 10, 12, 14],
                'max_features': ['sqrt', 0.3, 0.4, 0.5],
                'min_samples_leaf': randint(3, 15),
                'min_samples_split': randint(5, 30),
            }

            search = RandomizedSearchCV(
                base_rf, param_dist, n_iter=n_iter, cv=cv,
                scoring=scoring, n_jobs=-1, random_state=random_state, verbose=0
            )
            #search.fit(X, y)
            base_rf.fit(X, y)  # TEMPORARY: disable hyperparam search
            self.model = base_rf

            #self.model = search.best_estimator_
            #self.best_params_ = search.best_params_
            #print(f"Success! CV AUC = {search.best_score_:.3f}")
        except Exception as e:
            print(f"\nWarning: Search failed → fallback model ({e})")
            self.model = RandomForestClassifier(
                n_estimators=2000, max_depth=10, class_weight='balanced',
                n_jobs=-1, random_state=random_state
            )
            self.model.fit(X, y)
            print("Fallback model trained.")

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained!")
        return self.model.predict_proba(X)[:, 1]   # probability of sec_filter=1 (GOOD)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        instance = cls()
        instance.model = joblib.load(path)
        return instance

    @classmethod
    def kfold_validation(cls, data, X, y, embedding_lm='antiberty', title="RF_SEC_PASS", kfold=10,target="sec_filter"):
        y = np.array(y)
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
        fig = plt.figure(figsize=[7, 5.5])
        f1, accuracy, precision, recall, tprs, aucs = [], [], [], [], [], []    
        mean_fpr = np.linspace(0, 1, 100)


        print(f"\nRandom Forest {kfold}-Fold CV")
        print(f"Target: {title} | Embedding: {embedding_lm}")
        print(f"Positive class = sec_filter=1 = PASS (good antibody)")
        print(f"{'Fold':<6} {'AUC':<8} {'Acc':<8} {'F1':<8} {'Recall(PASS)':<12} {'Prec(PASS)'}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, data.get('HCDR3_CLUSTER_0.8', y)), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # === SAVE BARCODES FOR THIS FOLD ===
            train_df = data.iloc[train_idx].copy()
            val_df   = data.iloc[test_idx].copy()
            train_df.to_csv(f"{MODEL_DIR}/rf_{target}_fold{fold}_TRAIN_full.csv")
            val_df.to_csv(f"{MODEL_DIR}/rf_{target}_fold{fold}_VAL_full.csv")
            print(f"  Fold {fold} → saved full TRAIN ({len(train_df)}) + VAL ({len(val_df)}) rows") 
            


            model = cls()
            model.train(X_train, y_train, n_iter=20, cv=3)

            prob = model.predict_proba(X_test)
            pred = model.predict(X_test)

            fold_auc = roc_auc_score(y_test, prob)
            aucs.append(fold_auc)
            fpr, tpr, _ = roc_curve(y_test, prob)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            plt.plot(fpr, tpr, lw=1, alpha=0.5, label=f'Fold {fold} ({fold_auc:.3f})')

            recall_good = recall_score(y_test, pred)        # Recall of good antibodies
            precision_good = precision_score(y_test, pred, zero_division=0)

            f1.append(f1_score(y_test, pred))
            accuracy.append(accuracy_score(y_test, pred))
            precision.append(precision_good)
            recall.append(recall_good)
        

            print(f"{fold:<6} {fold_auc:.4f}   {accuracy_score(y_test,pred):.4f}   "
                  f"{f1_score(y_test,pred):.4f}   {recall_good:.4f}        {precision_good:.4f}")

            model.save(f"{MODEL_DIR}/rf_{target}_{embedding_lm}_fold{fold}_k{kfold}.pkl")

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
    

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_acc = np.mean(accuracy)
        mean_f1 = np.mean(f1)
        mean_prec = np.mean(precision)
        mean_rec = np.mean(recall)


        plt.plot(mean_fpr, mean_tpr, color='b', lw=4,
                 label=f'Mean ROC (AUC = {mean_auc:.3f} ± {np.std(aucs):.3f})')
        plt.fill_between(mean_fpr, mean_tpr - np.std(tprs,axis=0), mean_tpr + np.std(tprs,axis=0),
                         color='grey', alpha=0.3)
        
   
        #plt.plot([0,1],[0,1],'--',color='gray')
        plt.plot([0,1],[0,1])
        plt.xlim([0,1]); plt.ylim([0,1.05])



        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Random Forest — SEC Prediction\n'
          f'{kfold}-Fold CV | {embedding_lm} \n'
          f'Acc: {mean_acc:.3f}, F1: {mean_f1:.3f}, '
          f'Prec: {mean_prec:.3f}, Rec: {mean_rec:.3f}',
          fontsize=11, pad=20)

        plt.legend(loc='lower right', fontsize=9)
        plt.grid(alpha=0.3)

        plot_path = f"{MODEL_DIR}/rf_{target}_{embedding_lm}_k{kfold}_roc.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        # Final summary
        print(f"\nMEAN PERFORMANCE ({kfold}-Fold CV)")
        print(f"AUC     : {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Accuracy: {mean_acc:.4f}")
        print(f"F1-score: {mean_f1:.3f}")
        print(f"Precision: {mean_prec:.3f}")
        print(f"Recall  : {mean_rec:.3f}")
        print(f"Plot saved: {plot_path}")