"""
src/explainability.py
---------------------
Responsabilidade: calcular e interpretar valores SHAP com TreeExplainer.
Suporta modelos de árvore (XGBoost, Random Forest).
"""

import numpy as np
import shap


def compute_shap(model, X: np.ndarray) -> np.ndarray:
    """
    Calcula os valores SHAP para o conjunto X usando TreeExplainer.

    Retorna:
        ndarray de shape (n_samples, n_features, n_classes)
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = np.array(explainer.shap_values(X))
    return explainer, shap_values


def mean_abs_shap(shap_values: np.ndarray) -> np.ndarray:
    """
    Importância global: média do |SHAP| sobre amostras E classes.
    Shape entrada: (n_samples, n_features, n_classes)
    Shape saída  : (n_features,)
    """
    return np.abs(shap_values).mean(axis=(0, 2))


def shap_for_point(explainer, X_point: np.ndarray) -> np.ndarray:
    """
    Calcula SHAP para um único ponto de entrada.
    Shape saída: (1, n_features, n_classes)
    """
    return np.array(explainer.shap_values(X_point))


def shap_importances_dict(shap_values: np.ndarray, feature_names: list[str]) -> dict:
    """
    Retorna um dicionário {feature: importância_média_absoluta}
    ordenado de forma decrescente.
    """
    importance = mean_abs_shap(shap_values)
    ranked = sorted(
        zip(feature_names, importance.round(4).tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return dict(ranked)
