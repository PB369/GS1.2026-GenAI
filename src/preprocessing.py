"""
src/preprocessing.py
--------------------
Responsabilidade: carregar o dataset, codificar o target,
padronizar features e selecionar atributos via RFE.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split


def load_raw(csv_path: str) -> pd.DataFrame:
    """Carrega o CSV e descarta colunas auxiliares."""
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["data", "risco_score"])
    return df


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Codifica a coluna target com LabelEncoder e a remove do DataFrame."""
    le = LabelEncoder()
    df = df.copy()
    df["target"] = le.fit_transform(df["risco_desmatamento"])
    df = df.drop(columns=["risco_desmatamento"])
    return df, le


def build_scaler_and_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_features: int = 10,
) -> tuple[StandardScaler, np.ndarray, list[str]]:
    """
    Padroniza X com StandardScaler e seleciona as melhores features via RFE.

    Retorna:
        scaler    – StandardScaler ajustado
        support   – máscara booleana das features selecionadas
        sel_feat  – nomes das features selecionadas
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf_base, n_features_to_select=n_features, step=1)
    rfe.fit(X_scaled, y)

    support  = rfe.support_
    sel_feat = [f for f, s in zip(feature_names, support) if s]

    return scaler, support, sel_feat


def split_data(
    X_sel: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Divide em treino/teste com estratificação."""
    return train_test_split(
        X_sel, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def pearson_correlation(X_sel: np.ndarray, y: np.ndarray, sel_feat: list[str]) -> pd.Series:
    """Correlação de Pearson entre cada feature selecionada e o target."""
    df = pd.DataFrame(X_sel, columns=sel_feat)
    return df.corrwith(pd.Series(y)).sort_values(key=abs, ascending=False)
