"""
app/loader.py
-------------
Responsabilidade: carregar e preparar todos os dados e modelos necessários pelo app Streamlit, com cache para evitar reprocessamento.
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.persistence     import load_xgboost, load_params
from src.training        import train_random_forest
from src.explainability  import compute_shap

BASE = Path(__file__).resolve().parent.parent   # raiz do projeto


@st.cache_resource(show_spinner="Carregando modelos e preparando dados…")
def load_everything() -> dict:
    """
    Executa uma única vez (cacheado pelo Streamlit) e retorna um dicionário
    com todos os objetos necessários pelas páginas do app.

    Estratégia de carregamento:
      - XGBoost       → models/xgboost.json (formato nativo, sem pkl)
      - Scaler/LE/RFE → models/params.json  (parâmetros JSON, reconstruídos aqui)
      - Random Forest → retreinado aqui com random_state=42 (determinístico)
      - Métricas      → calculadas aqui sobre o conjunto de teste real
      - SHAP          → calculado aqui com TreeExplainer
    """
    mdls = BASE / "models"

    # ── Artefatos JSON
    xgb                        = load_xgboost(mdls / "xgboost.json")
    scaler, le, support, ALL_FEAT = load_params(mdls / "params.json")
    SEL_FEAT = [f for f, s in zip(ALL_FEAT, support) if s]
    CLASSES  = list(le.classes_)   # ['Alto', 'Baixo', 'Médio']

    # ── Dataset 
    df_raw = pd.read_csv(BASE / "data" / "dataset_desmatamento.csv")
    df     = df_raw.drop(columns=["data", "risco_score"]).copy()
    df["target"] = le.transform(df["risco_desmatamento"])
    df     = df.drop(columns=["risco_desmatamento"])

    X    = df[ALL_FEAT].values
    y    = df["target"].values
    Xs   = scaler.transform(X)
    Xsel = Xs[:, support]

    Xtr, Xte, ytr, yte = train_test_split(
        Xsel, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Random Forest retreinado (determinístico, sem pkl) 
    rf = train_random_forest(Xtr, ytr)

    # ── Métricas calculadas em tempo real
    metrics = {}
    for name, mdl in [("Random Forest", rf), ("XGBoost", xgb)]:
        ypred = mdl.predict(Xte)
        acc   = accuracy_score(yte, ypred)
        skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv    = cross_val_score(mdl, Xtr, ytr, cv=skf, scoring="accuracy", n_jobs=-1)
        metrics[name] = {
            "acc":     round(float(acc), 4),
            "cv_mean": round(float(cv.mean()), 4),
            "cv_std":  round(float(cv.std()), 4),
            "cm":      confusion_matrix(yte, ypred),
            "report":  classification_report(yte, ypred, target_names=CLASSES, output_dict=True),
            "y_pred":  ypred,
            "y_true":  yte,
        }

    # ── SHAP calculado em tempo real 
    explainer_xgb, sv_xgb = compute_shap(xgb, Xte)   # sv: (n, feat, 3)
    _,              sv_rf  = compute_shap(rf,  Xte)

    # ── Correlação de Pearson 
    df_sel  = pd.DataFrame(Xsel, columns=SEL_FEAT)
    pearson = df_sel.corrwith(pd.Series(y)).sort_values(key=abs, ascending=False)

    return {
        # Modelos
        "rf":  rf,
        "xgb": xgb,
        # Pré-processamento
        "scaler":   scaler,
        "support":  support,
        "le":       le,
        # Features e classes
        "ALL_FEAT": ALL_FEAT,
        "SEL_FEAT": SEL_FEAT,
        "CLASSES":  CLASSES,
        # Dados
        "df_raw": df_raw,
        "Xtr":    Xtr,
        "Xte":    Xte,
        "ytr":    ytr,
        "yte":    yte,
        "Xsel":   Xsel,
        # Avaliação
        "metrics": metrics,
        # SHAP
        "sv_xgb":        sv_xgb,
        "sv_rf":         sv_rf,
        "explainer_xgb": explainer_xgb,
        # Análise
        "pearson": pearson,
    }
