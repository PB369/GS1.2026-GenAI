# 🤖 Sentinel.AI — Modelo Preditivo de Desmatamento de Queimadas via Satélite

> Pipeline completo de Machine Learning para predição e classificação de risco de desmatamento e queimadas com dados sintéticos de sensoriamento remoto.

---

## 📌 Contexto do Problema

O desmatamento e as queimadas na Amazônia e no Cerrado representam uma das maiores crises ambientais da atualidade. Satélites como Landsat, Sentinel-2 e MODIS fornecem dados multiespectrais em tempo quase real, possibilitando monitoramento contínuo da cobertura vegetal.

Este projeto simula um **sistema de alerta precoce** de incêndios florestais, que classifica pontos geográficos em três níveis de risco — **Baixo, Médio e Alto** — com base em variáveis derivadas de imagens de satélite e dados climáticos gerados pelo monitoramento do desmatamento.

---

## 📦 Fonte dos Dados

Dataset **sintético** gerado com `data/generate_dataset.py` — **2.000 registros × 19 colunas**.

| Variável | Descrição |
|---|---|
| `temp_superficie` | Temperatura superficial em °C (LST) |
| `ndvi` | Normalized Difference Vegetation Index |
| `ndwi` | Normalized Difference Water Index |
| `umidade_relativa` | Umidade relativa do ar (%) |
| `precipitacao_30d` | Precipitação acumulada em 30 dias (mm) |
| `frp_fogo` | Fire Radiative Power — intensidade de fogo (MW) |
| `dist_estrada_km` | Distância à estrada mais próxima (km) |
| `dist_area_protegida_km` | Distância à unidade de conservação (km) |
| `dias_sem_chuva` | Dias consecutivos sem chuva |
| ... | + 10 variáveis adicionais de localização e clima |
| `risco_desmatamento` | **TARGET** — Baixo / Médio / Alto |

---

## 📁 Estrutura do Projeto

```
sentinel-ai/
│
├── data/
│   ├── generate_dataset.py       # Geração do dataset sintético (2.000 × 19)
│   └── dataset_desmatamento.csv  # Dataset gerado
│
├── src/                          # Módulos do pipeline (um por responsabilidade)
│   ├── preprocessing.py          # Carga, encoding, StandardScaler, RFE
│   ├── training.py               # Definição e treino de Random Forest e XGBoost
│   ├── evaluation.py             # Acurácia, CV, Matriz de Confusão, F1
│   ├── explainability.py         # SHAP TreeExplainer (global e por ponto)
│   └── persistence.py            # Salvar/carregar artefatos em JSON (zero .pkl)
│
├── app/                          # Aplicação Streamlit
│   ├── main.py                   # Ponto de entrada — sliders, roteamento
│   ├── loader.py                 # Carregamento cacheado de modelos e dados
│   ├── styles.py                 # CSS, paleta de cores, helpers Plotly
│   └── pages/
│       ├── prediction.py         # 🔮 Predição de Risco
│       ├── eda.py                # 📊 Análise Exploratória
│       ├── shap_page.py          # 🧠 Interpretabilidade SHAP
│       └── models_page.py        # 🏆 Comparação de Modelos
│
├── models/                       # Artefatos JSON dos modelos treinados
│   ├── xgboost.json              # XGBoost
│   ├── params.json               # Scaler, LabelEncoder, máscara RFE
│   ├── pipeline_meta.json        # Features selecionadas, resultados
│   └── shap_importances.json     # Ranking de importâncias SHAP
│
├── pipeline.py                   # Orquestrador — importa src/ e executa as etapas
├── requirements.txt
└── README.md
```

---

## ⚙️ Metodologia e Pipeline

```
data/generate_dataset.py       → Geração do dataset sintético
         ↓
src/preprocessing.py           → Carga, LabelEncoder, StandardScaler, RFE (top-10 features)
         ↓
src/training.py                → Treino do Random Forest + XGBoost
         ↓
src/evaluation.py              → Acurácia, CV 5-fold, Matriz de Confusão, F1
         ↓
src/explainability.py          → SHAP TreeExplainer
         ↓
src/persistence.py             → Salvar em JSON (zero .pkl)
         ↓
app/main.py                    → Deploy Streamlit (4 páginas)
```

> O `pipeline.py` da raiz é o **orquestrador**: importa os módulos de `src/` e executa as etapas em sequência. Cada módulo tem uma única responsabilidade.

---

## 🤖 Resultado dos Modelos Testados

| Modelo | Acurácia Teste | CV 5-fold |
|---|---|---|
| Random Forest | 76.00% | 79.81% ± 1.83% |
| **XGBoost** ✅ | **77.25%** | **77.63% ± 2.08%** |

---

## 🧠 Interpretação com SHAP

| Rank | Feature | SHAP (mean \|value\|) | Interpretação |
|---|---|---|---|
| 1 | `temp_superficie` | 1.628 | LST elevada → vegetação estressada → alto risco |
| 2 | `ndwi` | 0.481 | Baixo NDWI → baixa umidade da vegetação |
| 3 | `umidade_relativa` | 0.350 | Baixa umidade do ar potencializa queimadas |
| 4 | `ndvi` | 0.320 | Menor NDVI → menor cobertura vegetal |
| 5 | `dist_area_protegida_km` | 0.272 | Longe de UCs → maior pressão antrópica |

---

## 💾 Serialização de Modelos

| Artefato | Formato | Estratégia |
|---|---|---|
| XGBoost | `xgboost.json` | Formato nativo JSON do XGBoost — texto legível |
| StandardScaler | `params.json` | `mean_`, `scale_`, `var_` salvos como listas JSON |
| LabelEncoder | `params.json` | `classes_` salvo como lista JSON |
| Máscara RFE | `params.json` | `support_` (booleanos) salvo como lista JSON |
| Random Forest | *(sem arquivo)* | Retreinado deterministicamente com `random_state=42` |

---

## 🔗 Acesso ao Deploy

Clique neste link para visualizar o deploy da solução:


---

## 🚀 Como Executar Localmente

```bash
# 1. Clone e instale
git clone (link do repositório)
cd sentinel-ai
pip install -r requirements.txt

# 2. Gere o dataset
python data/generate_dataset.py

# 3. Execute o pipeline completo (treina, avalia, salva JSON)
python pipeline.py

# 4. Inicie o app
streamlit run app/main.py
```

Acesse em: `http://localhost:8501`
