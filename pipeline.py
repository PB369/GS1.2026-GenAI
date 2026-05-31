"""
pipeline.py
-----------
Orquestrador do pipeline de ML. Importa os módulos de src/ e
executa as etapas em sequência:
  1. Pré-processamento
  2. Treinamento
  3. Avaliação
  4. Interpretabilidade (SHAP)
  5. Persistência dos artefatos (zero .pkl)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from src.preprocessing   import load_raw, encode_target, build_scaler_and_rfe, split_data, pearson_correlation
from src.training        import train_random_forest, train_xgboost
from src.evaluation      import evaluate_all, save_confusion_matrix
from src.explainability  import compute_shap, shap_importances_dict
from src.persistence     import (
    save_xgboost, save_params, save_pipeline_meta, save_shap_importances
)

BASE  = Path(__file__).resolve().parent
DATA  = BASE / "data"  / "dataset_desmatamento.csv"
MDLS  = BASE / "models"
FIGS  = BASE / "docs"  / "figures"
MDLS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ── 1. Pré-processamento 
print("=" * 55)
print("1. PRÉ-PROCESSAMENTO")
print("=" * 55)

df_raw              = load_raw(DATA)
df, le              = encode_target(df_raw)
ALL_FEAT            = [c for c in df.columns if c != "target"]
X                   = df[ALL_FEAT].values
y                   = df["target"].values
CLASSES             = list(le.classes_)

scaler, support, SEL_FEAT = build_scaler_and_rfe(X, y, ALL_FEAT, n_features=10)
X_sel               = scaler.transform(X)[:, support]

pearson = pearson_correlation(X_sel, y, SEL_FEAT)
print(f"Features selecionadas ({len(SEL_FEAT)}): {SEL_FEAT}")
print(f"\nCorrelação de Pearson (top 5):\n{pearson.head().to_string()}")

X_train, X_test, y_train, y_test = split_data(X_sel, y)
print(f"\nTreino: {X_train.shape}  |  Teste: {X_test.shape}")

# ── 2. Treinamento 
print("\n" + "=" * 55)
print("2. TREINAMENTO")
print("=" * 55)

rf  = train_random_forest(X_train, y_train)
xgb = train_xgboost(X_train, y_train)
print("Random Forest e XGBoost treinados.")

# ── 3. Avaliação 
print("\n" + "=" * 55)
print("3. AVALIAÇÃO")
print("=" * 55)

metrics = evaluate_all(
    models={"Random Forest": rf, "XGBoost": xgb},
    X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test,
    class_names=CLASSES,
)

for name, m in metrics.items():
    save_confusion_matrix(
        cm=m["cm"],
        class_names=CLASSES,
        model_name=name,
        output_path=str(FIGS / f"cm_{name.replace(' ', '_').lower()}.png"),
    )

best_name = max(metrics, key=lambda k: metrics[k]["acc"])
results   = [{"model": k, **{kk: vv for kk, vv in v.items()
              if kk in ("acc","cv_mean","cv_std")}}
             for k, v in metrics.items()]

# ── 4. Interpretabilidade SHAP 
print("\n" + "=" * 55)
print("4. INTERPRETABILIDADE (SHAP)")
print("=" * 55)

best_model          = xgb if best_name == "XGBoost" else rf
explainer, sv       = compute_shap(best_model, X_test)
shap_imp            = shap_importances_dict(sv, SEL_FEAT)
print(f"Top 3 features: {list(shap_imp.items())[:3]}")

# ── 5. Persistência
print("\n" + "=" * 55)
print("5. SALVANDO ARTEFATOS")
print("=" * 55)

save_xgboost(xgb,  MDLS / "xgboost.json")
save_params(scaler, le, support, ALL_FEAT, MDLS / "params.json")
save_pipeline_meta(SEL_FEAT, ALL_FEAT, CLASSES, best_name, results, MDLS / "pipeline_meta.json")
save_shap_importances(shap_imp, MDLS / "shap_importances.json")

print("Arquivos em /models/:")
for p in sorted(MDLS.iterdir()):
    print(f"  {p.name:<35} {p.stat().st_size:>10,} bytes")

print("\nPipeline concluído!")
