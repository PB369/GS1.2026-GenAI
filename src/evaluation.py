"""
src/evaluation.py
-----------------
Responsabilidade: calcular e exibir métricas de avaliação dos modelos.
Inclui: acurácia, cross-validation, matriz de confusão e relatório de classificação.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def evaluate(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    cv_folds: int = 5,
) -> dict:
    """
    Avalia um modelo com acurácia, cross-validation e classification report.

    Retorna um dicionário com todas as métricas calculadas.
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv  = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)

    cm  = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    return {
        "acc":     round(float(acc), 4),
        "cv_mean": round(float(cv.mean()), 4),
        "cv_std":  round(float(cv.std()), 4),
        "cm":      cm,
        "report":  rep,
        "y_pred":  y_pred,
        "y_true":  y_test,
    }


def evaluate_all(
    models: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Avalia todos os modelos passados em `models` (dict nome → objeto).
    Retorna um dict nome → métricas e imprime uma comparação.
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate(model, X_train, X_test, y_train, y_test, class_names)
        m = results[name]
        print(f"\n{'─'*40}")
        print(f"Modelo: {name}")
        print(f"  Acurácia Teste  : {m['acc']:.4f}")
        print(f"  CV ({5}-fold)     : {m['cv_mean']:.4f} ± {m['cv_std']:.4f}")
        print(classification_report(y_test, m["y_pred"], target_names=class_names))

    best = max(results, key=lambda k: results[k]["acc"])
    print(f"\nMelhor modelo: {best}")
    return results


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], model_name: str, output_path: str) -> None:
    """Salva a matriz de confusão como imagem PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Matriz de Confusão — {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
