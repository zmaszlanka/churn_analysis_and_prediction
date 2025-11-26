import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import shap

# ============================================================
#   MODELING PIPELINE
# ============================================================

class ChurnModelPipeline:
    def __init__(self, df: pd.DataFrame, target_col="churn"):
        self.df = df
        self.target_col = target_col
        self.models = {}
        self.results = {}
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
    
    def _prepare_data(self):
        X = self.df.drop(columns=[self.target_col, "customer_id"], errors="ignore")
        y = self.df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
        )
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        return X_train_res, X_test, y_train_res, y_test
    
    def add_models(self):
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
            "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, random_state=42, class_weight="balanced")

        }
    
    def evaluate_model(self, model, name):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            "roc_auc": roc_auc_score(self.y_test, y_proba),
            "accuracy": accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test, y_pred, zero_division=0),
            "recall": recall_score(self.y_test, y_pred, zero_division=0),
            "f1": f1_score(self.y_test, y_pred, zero_division=0)
        }
        self.results[name] = metrics
        print(f"\n{name} Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['roc_auc']:.2f})")
    
    def plot_roc_curves(self):
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.show()
    
    def plot_precision_recall(self):
        plt.figure()
        for name, model in self.models.items():
            y_proba = model.predict_proba(self.X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
            plt.plot(recall, precision, label=name)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()
    
    def show_feature_importance(self, top_n=20):
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]
                plt.figure(figsize=(10,6))
                plt.title(f"{name} Feature Importances (Top {top_n})")
                plt.bar(range(top_n), importances[indices], align="center")
                plt.xticks(range(top_n), self.X_train.columns[indices], rotation=45, ha="right")
                plt.tight_layout()
                plt.show()

    
    def explain_shap(self):
        # Only for tree-based models
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(self.X_test)
                shap.summary_plot(shap_values, self.X_test, show=True, plot_type="bar")

    def run(self):
        self.add_models()
        plt.figure(figsize=(8,6))
        for name, model in self.models.items():
            self.evaluate_model(model, name)
        self.plot_roc_curves()
        self.plot_precision_recall()
        self.show_feature_importance()
        # self.explain_shap()
        return self.results

# ============================================================
#   USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    df_preprocessed = pd.read_csv("src/synthetic_dataset/generate/preprocessed.csv")
    pipeline = ChurnModelPipeline(df_preprocessed, target_col="churn")
    results = pipeline.run()
