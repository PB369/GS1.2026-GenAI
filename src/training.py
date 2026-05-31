"""
src/training.py
---------------
Responsabilidade: definir e treinar os modelos de ML.
Cada função recebe os dados de treino e devolve o modelo ajustado.
"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 300,
    max_depth: int = 12,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Treina o Random Forest com class_weight='balanced' para lidar
    com o desbalanceamento da classe 'Alto' (~14% do dataset).
    random_state=42 garante reprodutibilidade sem salvar o modelo em disco.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(
    X_train,
    y_train,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Treina o XGBoost (melhor modelo, acurácia 77.25%).
    Salvo em JSON nativo — sem pkl.
    """
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    return xgb
