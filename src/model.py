# src/model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import json

FEATURE_PATH = "data/features/loan_features.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "model_pipeline_noleak.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics_summary_noleak.csv")

def ks_statistic(y_true, y_scores):
    df = pd.DataFrame({"y": y_true, "s": y_scores})
    df = df.sort_values("s", ascending=False).reset_index(drop=True)
    df["cum_event"] = (df["y"] == 1).cumsum()
    df["cum_nonevent"] = (df["y"] == 0).cumsum()
    total_event = df["y"].sum()
    total_nonevent = (df["y"] == 0).sum()
    if total_event == 0 or total_nonevent == 0:
        return np.nan
    df["cum_event_pct"] = df["cum_event"] / total_event
    df["cum_nonevent_pct"] = df["cum_nonevent"] / total_nonevent
    return float((df["cum_event_pct"] - df["cum_nonevent_pct"]).abs().max())

def lift_at_k(y_true, y_scores, k=0.1):
    df = pd.DataFrame({"y": y_true, "s": y_scores})
    df = df.sort_values("s", ascending=False).reset_index(drop=True)
    top_n = int(len(df) * k)
    if top_n == 0 or df["y"].mean() == 0:
        return np.nan
    top = df.iloc[:top_n]
    return float(top["y"].mean() / df["y"].mean())

def train():
    df = pd.read_csv(FEATURE_PATH)
    # removendo features com leak se existir
    leak_cols = [c for c in ["meses_em_atraso","overdue_flag","serious_arrears"] if c in df.columns]
    df_noleak = df.drop(columns=leak_cols, errors="ignore")

    y = df_noleak["target"] if "target" in df_noleak.columns else (df_noleak["inadimplente"] if "inadimplente" in df_noleak.columns else None)
    if y is None:
        raise ValueError("Coluna target não encontrada em features.")

    # definir X num/cat
    numeric_features = [c for c in ["renda","idade","score","valor","loan_to_income","estimated_monthly_payment","pct_income_commitment"] if c in df_noleak.columns]
    categorical_features = [c for c in ["age_bucket","score_bucket"] if c in df_noleak.columns]

    X = df_noleak[numeric_features + categorical_features].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])

    clf = Pipeline(steps=[("pre", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])

    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:,1] if len(set(y_test))>1 else np.zeros(len(y_test))
    y_pred = clf.predict(X_test) if len(set(y_test))>1 else np.zeros(len(y_test)).astype(int)

    auc = roc_auc_score(y_test, y_proba) if len(set(y_test))>1 else np.nan
    gini = 2*auc - 1 if not np.isnan(auc) else np.nan
    ks = ks_statistic(y_test, y_proba)
    lift10 = lift_at_k(y_test, y_proba, k=0.1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    metrics = {
        "auc": auc, "gini": gini, "ks": ks, "lift10": lift10,
        "precision": precision, "recall": recall, "f1": f1,
        "n_train": len(X_train), "n_test": len(X_test)
    }

    # salvar
    joblib.dump(clf, MODEL_PATH)
    pd.Series(metrics).to_csv(METRICS_PATH)
    print("Modelo treinado e salvo:", MODEL_PATH)
    print("Métricas salvas:", METRICS_PATH)
    print(metrics)
    return metrics

if __name__ == "__main__":
    train()
