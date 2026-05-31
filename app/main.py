"""
app/main.py
-----------
Responsabilidade: ponto de entrada do app Streamlit.
Executa três responsabilidades centrais e delega o resto:
  1. Configuração da página e injeção de CSS (app/styles.py)
  2. Definição dos sliders e cálculo do ponto atual (sidebar)
  3. Roteamento para a página selecionada (app/pages/)

Comando para iniciar (a partir da raiz do projeto):
    streamlit run app/main.py
"""

# Garante que a raiz do projeto esteja no sys.path,
# independente de onde o Streamlit é chamado.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st

from app.loader import load_everything
from app.styles import inject_css, section_hdr, CLS_COLOR
from app.pages  import prediction, eda, shap_page, models_page
from src.explainability import shap_for_point

# ── Configuração da página 
st.set_page_config(
    page_title="Sentinel.AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Carregar dados e modelos (cacheado) 
D = load_everything()

# ── Sidebar — navegação e sliders 
with st.sidebar:
    st.markdown("### 🤖 Sentinel.AI")
    st.markdown("---")

    page = st.radio("Página", [
        "🔮 Predição de Risco",
        "📊 Análise Exploratória",
        "🧠 Interpretabilidade SHAP",
        "🏆 Comparação de Modelos",
    ])

    st.markdown("---")
    st.markdown("**Parâmetros do ponto satélite**")
    st.caption("Altere os valores abaixo e veja os resultados.")

    lat     = st.slider("Latitude",                -15.0,   5.0,  -5.0,  0.5)
    temp    = st.slider("Temp. superficial (°C)",   20.0,  45.0,  30.0,  0.5)
    umidade = st.slider("Umidade relativa (%)",     20.0, 100.0,  65.0,  1.0)
    ndvi    = st.slider("NDVI",                     -0.2,   1.0,   0.5,  0.01)
    ndwi    = st.slider("NDWI",                     -0.5,   0.8,   0.2,  0.01)
    precip  = st.slider("Precipitação 30d (mm)",     0.0, 300.0,  80.0,  5.0)
    frp     = st.slider("FRP — Fogo (MW)",           0.0,  50.0,   3.0,  0.5)
    d_road  = st.slider("Dist. estrada (km)",        0.0, 150.0,  25.0,  1.0)
    d_prot  = st.slider("Dist. área protegida (km)", 0.0, 200.0,  40.0,  2.0)
    dias_sr = st.slider("Dias sem chuva",            0.0,  90.0,  15.0,  1.0)

# ── Construir vetor de entrada a partir dos sliders 
slider_vals = {
    "latitude":              lat,
    "temp_superficie":       temp,
    "umidade_relativa":      umidade,
    "ndvi":                  ndvi,
    "ndwi":                  ndwi,
    "precipitacao_30d":      precip,
    "frp_fogo":              frp,
    "dist_estrada_km":       d_road,
    "dist_area_protegida_km": d_prot,
    "dias_sem_chuva":        dias_sr,
}

# Features não expostas nos sliders usam valores típicos
defaults_full = {
    "latitude":              lat,
    "longitude":             -60.0,
    "mes":                   7,
    "temp_superficie":       temp,
    "umidade_relativa":      umidade,
    "ndvi":                  ndvi,
    "ndwi":                  ndwi,
    "precipitacao_30d":      precip,
    "velocidade_vento":      12.0,
    "frp_fogo":              frp,
    "dist_estrada_km":       d_road,
    "dist_area_protegida_km": d_prot,
    "densidade_pop":         8.0,
    "altitude_m":            300.0,
    "dias_sem_chuva":        dias_sr,
    "estacao_seca":          int(precip < 50),
}

X_input_full = np.array([[defaults_full[f] for f in D["ALL_FEAT"]]])
X_input_sc   = D["scaler"].transform(X_input_full)
X_input_sel  = X_input_sc[:, D["support"]]

# ── Predição em tempo real 
pred_xgb  = D["xgb"].predict(X_input_sel)[0]
prob_xgb  = D["xgb"].predict_proba(X_input_sel)[0]
pred_rf   = D["rf"].predict(X_input_sel)[0]
prob_rf   = D["rf"].predict_proba(X_input_sel)[0]
label_xgb = D["le"].inverse_transform([pred_xgb])[0]
label_rf  = D["le"].inverse_transform([pred_rf])[0]

# ── SHAP do ponto atual 
shap_point      = shap_for_point(D["explainer_xgb"], X_input_sel)  # (1, feat, 3)
shap_point_mean = np.abs(shap_point[0, :, :]).mean(axis=1)

# ── Header global 
st.markdown("""
<div class="hero">
  <h1>🤖 Sentinel.AI</h1>
  <p>Seu sistema de IA para predições de risco de queimadas e desmatamento através de monitoramento via satélite.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="kpi"><div class="val" style="color:{CLS_COLOR[label_xgb]}">'
        f'{label_xgb}</div><div class="lbl">Risco atual (XGBoost)</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="kpi"><div class="val">{prob_xgb.max()*100:.1f}%</div>'
        f'<div class="lbl">Confiança da predição</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="kpi"><div class="val">{temp:.1f}°C</div>'
        f'<div class="lbl">Temperatura superficial</div></div>',
        unsafe_allow_html=True,
    )
with c4:
    top_feat = D["SEL_FEAT"][int(np.argmax(shap_point_mean))]
    st.markdown(
        f'<div class="kpi"><div class="val" style="font-size:1.1rem">{top_feat}</div>'
        f'<div class="lbl">Feature SHAP mais influente</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Roteamento para as páginas 
if page == "🔮 Predição de Risco":
    prediction.render(
        D=D,
        X_input_sel=X_input_sel,
        shap_point=shap_point,
        shap_point_mean=shap_point_mean,
        prob_xgb=prob_xgb,
        prob_rf=prob_rf,
        label_xgb=label_xgb,
        label_rf=label_rf,
    )

elif page == "📊 Análise Exploratória":
    eda.render(
        D=D,
        X_input_sel=X_input_sel,
        slider_vals=slider_vals,
        label_xgb=label_xgb,
        temp=temp,
        ndvi=ndvi,
    )

elif page == "🧠 Interpretabilidade SHAP":
    shap_page.render(
        D=D,
        X_input_sel=X_input_sel,
        shap_point=shap_point,
        shap_point_mean=shap_point_mean,
        prob_xgb=prob_xgb,
    )

elif page == "🏆 Comparação de Modelos":
    models_page.render(
        D=D,
        prob_xgb=prob_xgb,
        prob_rf=prob_rf,
        label_xgb=label_xgb,
        label_rf=label_rf,
    )
