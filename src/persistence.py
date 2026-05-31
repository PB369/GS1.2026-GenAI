"""
src/persistence.py
------------------
Responsabilidade: salvar e carregar artefatos do pipeline.
Zero arquivos binários (.pkl). Formatos usados:
  - xgboost.json       → XGBoost (formato nativo, texto legível)
  - params.json        → StandardScaler, LabelEncoder, máscara RFE
  - pipeline_meta.json → features selecionadas, resultados, melhor modelo
  - shap_importances.json → importâncias SHAP
"""

import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


# ── Salvar ────────────────────────────────────────────────────────────────────

def save_xgboost(model: XGBClassifier, path: str) -> None:
    """Salva o XGBoost no formato JSON nativo (legível, sem bytes binários)."""
    model.save_model(str(path))


def save_params(
    scaler: StandardScaler,
    le: LabelEncoder,
    support: np.ndarray,
    feature_names: list[str],
    path: str,
) -> None:
    """
    Persiste os parâmetros aprendidos do Scaler, LabelEncoder e máscara RFE
    em um único arquivo JSON.
    """
    params = {
        "scaler": {
            "mean_":          scaler.mean_.tolist(),
            "scale_":         scaler.scale_.tolist(),
            "var_":           scaler.var_.tolist(),
            "n_features_in_": int(scaler.n_features_in_),
        },
        "label_encoder": {
            "classes_": le.classes_.tolist(),
        },
        "rfe": {
            "support_":       support.tolist(),
            "n_features_in_": int(len(support)),
        },
        "feature_names": feature_names,
    }
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def save_pipeline_meta(
    sel_feat: list[str],
    all_feat: list[str],
    classes: list[str],
    best_model: str,
    results: list[dict],
    path: str,
) -> None:
    """Salva os metadados do pipeline (features, resultados, melhor modelo)."""
    meta = {
        "selected_features": sel_feat,
        "all_features":      all_feat,
        "target_classes":    classes,
        "best_model":        best_model,
        "results":           results,
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def save_shap_importances(importances: dict, path: str) -> None:
    """Salva o ranking de importâncias SHAP."""
    with open(path, "w") as f:
        json.dump(importances, f, indent=2)


# ── Carregar ──────────────────────────────────────────────────────────────────

def load_xgboost(path: str) -> XGBClassifier:
    """Carrega o XGBoost a partir do JSON nativo."""
    model = XGBClassifier()
    model.load_model(str(path))
    return model


def load_params(path: str) -> tuple[StandardScaler, LabelEncoder, np.ndarray, list[str]]:
    """
    Carrega e reconstrói StandardScaler, LabelEncoder e máscara RFE a partir do JSON.

    Retorna:
        scaler       – StandardScaler reconstruído
        le           – LabelEncoder reconstruído
        support      – máscara booleana (np.ndarray) das features selecionadas
        feature_names – lista de nomes de todas as features
    """
    with open(path) as f:
        p = json.load(f)

    scaler = StandardScaler()
    scaler.mean_          = np.array(p["scaler"]["mean_"])
    scaler.scale_         = np.array(p["scaler"]["scale_"])
    scaler.var_           = np.array(p["scaler"]["var_"])
    scaler.n_features_in_ = p["scaler"]["n_features_in_"]

    le = LabelEncoder()
    le.classes_ = np.array(p["label_encoder"]["classes_"])

    support       = np.array(p["rfe"]["support_"])
    feature_names = p["feature_names"]

    return scaler, le, support, feature_names
